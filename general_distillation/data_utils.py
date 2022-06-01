
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import (DataLoader, RandomSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory
from pathlib import Path
from tqdm import tqdm, trange

from utils import set_madvise

import logging

import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class PregeneratedDataset(Dataset):
    
    def __init__(self, training_path, epoch, num_data_epochs, n_divide=None):
        
        self.epoch = epoch
        self.data_epoch = int(epoch % num_data_epochs)
        
        
        logger.info('training_path: {}'.format(training_path))
        data_file = training_path / "epoch_{}.json".format(self.data_epoch)
        metrics_file = training_path / "epoch_{}_metrics.json".format(self.data_epoch)

        logger.info('data_file: {}'.format(data_file))
        logger.info('metrics_file: {}'.format(metrics_file))

        assert metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']

#         self.temp_dir = TemporaryDirectory()
        self.working_dir = Path(training_path)                            # training_path is /tmp/corpus
    
        self.n_divide = n_divide
        if self.n_divide is not None:
            self.data_start = num_samples // self.n_divide[1] * self.n_divide[0]
            self.data_end = self.data_start + num_samples // self.n_divide[1]
            if self.n_divide[0] + 1 == self.n_divide[1]:
                self.data_end = num_samples
        try:

            logging.info("Loading training examples for epoch {}".format(epoch))
        
            input_ids = np.memmap(filename=self.working_dir/f'input_ids{epoch}.memmap',
                                  mode='r', dtype=np.int32, shape=(num_samples, seq_len))
            segment_ids = np.memmap(filename=self.working_dir/f'segment_ids{epoch}.memmap',
                                    shape=(num_samples, seq_len), mode='r', dtype=np.bool)
            
            set_madvise(input_ids, 1) # best for random accessing
            set_madvise(segment_ids, 1) # best for random accessing
            
            self.has_mmap = True
            
        except:
            
            logging.info("Failed to load mmap file, try load raw json.")
            logging.info("Note the data will be iterated in order.")
            
            self.has_mmap = False
            
            raise Exception('TODO: not implement yet.')

            assert data_file.is_file()

        # assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.segment_ids = segment_ids

    def __len__(self):
        if self.n_divide is None:
            return self.num_samples
        else:
            return self.data_end - self.data_start

    def __getitem__(self, item):
        
        if self.n_divide is not None:
            item += self.data_start
        
        if self.has_mmap:                                                       # 这里has_mmap好像没有意义
            input_ids = torch.tensor(self.input_ids[item].astype(np.int64))
            input_masks = (input_ids!=0).long()
            segment_ids = torch.tensor(self.segment_ids[item].astype(np.int64))
            lm_label_ids = torch.zeros_like(input_ids)
            is_nexts = torch.zeros(1, dtype=torch.long)
        else:
            input_ids = torch.tensor(self.input_ids[item].astype(np.int64))
            input_masks = (input_ids!=0).long()
            segment_ids = torch.tensor(self.segment_ids[item].astype(np.int64))
            lm_label_ids = torch.zeros_like(input_ids)
            is_nexts = torch.zeros(1, dtype=torch.long)
            
        return (input_ids,
                input_masks,
                segment_ids,
                lm_label_ids,
                is_nexts,)

    
def get_dataloader(args, epoch=0):
    
    try:
        dataset = PregeneratedDataset(
            epoch=epoch, training_path=args.pregenerated_data,                  # pregenerated_data is /tmp/corpus
            num_data_epochs=args.num_train_epochs, n_divide=None,
        )
    except:
        logger.info(f'failed to find memmap for epoch {epoch}, try 0 instead.')
        dataset = PregeneratedDataset(
            epoch=0, training_path=args.pregenerated_data,
            num_data_epochs=args.num_train_epochs, n_divide=None,
        )
    
    if args.local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset, shuffle=True)
        train_sampler.set_epoch(epoch)
        
    return DataLoader(dataset, batch_size=args.train_micro_batch_size_per_gpu, 
                      sampler=train_sampler, num_workers=1)

