
import argparse
import csv
import logging
import os
import random
import sys
import json

sys.path.insert(1, os.path.join(sys.path[0], '..'))

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

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"
from skipbert.modeling import BertForPreTraining
from skipbert.modeling import SkipBertForPreTraining
from transformers import BertModel, BertConfig
from transformers import BertTokenizerFast as BertTokenizer

import deepspeed

# custom dependencies
from utils import get_sample_writer
from data_utils import get_dataloader

csv.field_size_limit(sys.maxsize)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


def get_argument_parser():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
            
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--job_name",
                        type=str,
                        default='skipbert')

    # Required parameters
    parser.add_argument("--pregenerated_data",
                        type=Path,
                        required=True)
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True)
    
    # Optional
    parser.add_argument("--load_dir",
                        default=None,
                        type=str)
    
    parser.add_argument("--load_ckpt_id",
                        default=None,
                        type=str)

    # Other parameters
    parser.add_argument("--student_model_class",
                        default='SkipBert',
                        type=str)
    
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--continue_train',
                        action='store_true',
                        help='Whether to train from checkpoints')
    
    parser.add_argument('--not_use_att',
                        action='store_true',
                        help='')
    
    parser.add_argument('--not_use_hid',
                        action='store_true',
                        help='')
    
    parser.add_argument('--n_gram_left',
                        default=1,
                        type=int,
                        help='')
    
    parser.add_argument('--n_gram_right',
                        default=1,
                        type=int,
                        help='')
    
    parser.add_argument("--num_masked_layers_teacher", default=6, type=int,)
    parser.add_argument("--num_masked_last_layers_teacher", default=0, type=int,)
    
    parser.add_argument("--num_full_hidden_layers", default=-1, type=int)
    parser.add_argument("--num_hidden_layers", default=-1, type=int)
    
    parser.add_argument('--att_layer_maps',
                        default=None,
                        nargs='+', type=int)
    parser.add_argument('--hid_layer_maps',
                        default=None,
                        nargs='+', type=int)

    # Additional arguments
    parser.add_argument('--eval_step',
                        type=int,
                        default=100)
    
    parser.add_argument('--save_step',
                        type=int,
                        default=5000)
    
    parser.add_argument("--ngram_masking",
                        type=float,
                        default=0.0)

    return parser

def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    
    deepspeed_config = json.load(
        open(args.deepspeed_config, 'r', encoding='utf-8'))
    
    args.train_batch_size = deepspeed_config['train_batch_size']
    args.train_micro_batch_size_per_gpu = deepspeed_config['train_micro_batch_size_per_gpu']
    args.steps_per_print = deepspeed_config['steps_per_print']
    args.gradient_accumulation_steps = 1

    return args

def prepare_optimizer_parameters(args, model):
    deepspeed_config = json.load(
        open(args.deepspeed_config, 'r', encoding='utf-8'))

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    if "weight_decay" in deepspeed_config["optimizer"]:
        weight_decay = deepspeed_config["optimizer"]["weight_decay"]
    else:
        weight_decay = 1e-4

    if deepspeed_config["optimizer"]["type"] not in ["OneBitAdam"]:
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': weight_decay
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
    else:
        # Because 1-bit compression cannot represent exact zero, it is required to
        # provide a momentum mask for those params that have constant exact zeros in their
        # momentums, otherwise the compression error would keep accumulating.
        # For example, for bert pre-training seq 128, bert.embeddings.position_embeddings.weight
        # always have exact zeros in its momentum for row 129 to 512, because it only
        # learns up to seq length 128 while the model supports up to 512 seq length.
        need_mask = ['position_embeddings.weight']
        need_mask_p = []
        need_mask_decay = []
        masks = []
        for n, p in param_optimizer:
            if any(nd in n for nd in need_mask):
                mask = torch.zeros_like(p.data)
                for position in range(args.max_seq_length):
                    for col in range(p.size()[1]):
                        mask[position][col] += 1
                if deepspeed_config["optimizer"]["type"] == "OneBitAdam":
                    mask = torch.flatten(mask)
                masks.append(mask)
                need_mask_p.append(p)
                if any(nd in n for nd in no_decay):
                    need_mask_decay.append(0.0)
                else:
                    need_mask_decay.append(weight_decay)

        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay + need_mask)
            ],
            'weight_decay':
            weight_decay
        }, {
            'params': [
                p for n, p in param_optimizer
                if (any(nd in n
                        for nd in no_decay) and not any(nd in n
                                                        for nd in need_mask))
            ],
            'weight_decay':
            0.0
        }]

        for i_mask in range(len(need_mask_p)):
            optimizer_grouped_parameters.append({
                'params': [need_mask_p[i_mask]],
                'weight_decay':
                need_mask_decay[i_mask],
                'exp_avg_mask':
                masks[i_mask]
            })

    return optimizer_grouped_parameters


def prepare_model_optimizer(args, model):

    # Optimizer parameters
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters)

    # Set DeepSpeed info
#     args.local_rank = model.network.local_rank
#     args.device = model.network.device
#     model.set_device(args.device)
#     args.fp16 = model.network.fp16_enabled()
#     args.use_lamb = model.network.optimizer_name(
#     ) == deepspeed.runtime.config.LAMB_OPTIMIZER

    # Prepare Summary Writer and saved_models path
    if args.local_rank == 0:
        summary_writer = get_sample_writer(name=args.job_name,
                                           base=args.output_dir)
        args.summary_writer = summary_writer

    return model, optimizer

def load_engine_checkpoint(args, model_engine):
    if args.load_dir is not None and args.load_ckpt_id is not None:
        _, client_state = model_engine.load_checkpoint(args.load_dir, args.load_ckpt_id)
        return client_state
    else:
        return {'step': 0}

def save_engine_checkpoint(args, step, model_engine, client_state=None):
    if client_state is None:
        client_state = {
            'step': step,
        }
    ckpt_id = step
    # note all processes need to call .save_checkpoint()
    model_engine.save_checkpoint(args.saved_engine_path, ckpt_id, client_state=client_state)

def save_step_model(args, step, model_engine, tokenizer):
    # Save a trained model
    if args.local_rank in [0,-1]: # only master process
        model_name = "step_{}_{}".format(step, WEIGHTS_NAME)
        logging.info("** ** * Saving Model ** ** * ")
        # Only save the model it-self
        model_to_save = model_engine.module

        output_model_file = os.path.join(args.saved_model_path, model_name)
        output_config_file = os.path.join(args.saved_model_path, CONFIG_NAME)

        # to be compatible with pytorch 1.4
        torch.save(model_to_save.state_dict(), output_model_file, 
                   pickle_protocol=2, _use_new_zipfile_serialization=False)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.saved_model_path)

def report_step_metrics(args, step, metrics):
    if args.local_rank in [0,-1]: # only master process
        for k, v in metrics.items():
            args.summary_writer.add_scalar(k, v, step)
            
        metrics['global_step'] = step
        output_eval_file = os.path.join(args.output_dir, "log.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            for k, v in sorted(metrics.items()):
                logger.info(f"  {k} = {v}")
                writer.write(f"{k} = {v}\n")
        

def main():
    
    args = get_arguments()
    
    ##
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    ##
    os.makedirs(args.output_dir, exist_ok=True)
    
    args.saved_model_path = os.path.join(args.output_dir, "saved_models/")
    args.saved_engine_path = os.path.join(args.output_dir, "saved_engines/")
    
    os.makedirs(args.saved_model_path, exist_ok=True)
    
    ##
    args.local_rank = int(os.environ['LOCAL_RANK'])
    
    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed(dist_backend='nccl')
    args.device = device
    
    ##
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)
    
    student_config = BertConfig.from_pretrained(args.student_model, continue_train=args.continue_train)
    student_config.num_full_hidden_layers = args.num_full_hidden_layers # TODO
    student_config.n_gram_left = args.n_gram_left
    student_config.n_gram_right = args.n_gram_right
    student_config.num_hidden_layers = args.num_hidden_layers # TODO
    student_config.ngram_masking = args.ngram_masking
    
    if args.continue_train:
        student_model = eval(f"{args.student_model_class}ForPreTraining").from_pretrained(args.student_model, config=student_config)
    else:
        student_model = eval(f"{args.student_model_class}ForPreTraining")(config=student_config)
    
    assert student_model.config.num_hidden_layers == len(student_model.bert.encoder.layer)
    
    teacher_config = BertConfig.from_pretrained(args.teacher_model)
    teacher_model = BertModel.from_pretrained(args.teacher_model, config=teacher_config)
    teacher_model.half().eval()

    student_model.to(device)
    teacher_model.to(device)
    
    student_model_engine, optimizer = prepare_model_optimizer(args, student_model)
    
    # if no ckpt is assigned, skip loading and global_step == 0
    client_state = load_engine_checkpoint(args, student_model_engine)
    global_step = client_state['step']
    
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        
        train_dataloader = get_dataloader(args, epoch=epoch)
    
        tr_loss = 0.
        tr_att_loss = 0.
        tr_rep_loss = 0.
        nb_tr_examples, nb_tr_steps = 0, 0
        
        rt_loss = 0.
        rt_att_loss = 0.
        rt_rep_loss = 0.
        nb_rt_examples, nb_rt_steps = 0, 0
        
        student_model_engine.train()
        
        with tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch), disable=args.local_rank not in [-1, 0]) as pbar:
            for step, batch in enumerate(train_dataloader):
                
                batch = tuple(t.to(device) for t in batch)
                
                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                if input_ids.size(0) != args.train_micro_batch_size_per_gpu:
                    continue
                 
                with torch.no_grad():
                    _ret = teacher_model(
                        input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                        output_attentions=True, output_hidden_states=True,
                    )
                    teacher_reps = _ret.hidden_states
                    teacher_atts = _ret.attentions
                    b, e = args.num_masked_layers_teacher, -args.num_masked_last_layers_teacher
                    if e == 0:
                        e = None
                    teacher_atts, teacher_reps = teacher_atts[:], teacher_reps[b:e]
                    if args.num_full_hidden_layers == 2:
                        teacher_atts = [teacher_atts[4], teacher_atts[9]]
                
                student_atts, student_reps = student_model_engine(
                    input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                    output_attentions=True, output_hidden_states=True,
                )
                
                att_loss = 0.
                rep_loss = 0.

                if args.att_layer_maps is None:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[(i + 1) * layers_per_block - 1]
                                       for i in range(student_layer_num)]
                    assert len(student_atts) == len(new_teacher_atts)
                else:
                    new_teacher_atts = []
                    for t2s in args.att_layer_maps:
                        if t2s >= 0:
                            new_teacher_atts.append(teacher_atts[t2s])
                        else:
                            new_teacher_atts.append(None)
                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    if teacher_att is None:
                        continue
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                              teacher_att)
                    att_loss += F.mse_loss(student_att, teacher_att)

                if args.hid_layer_maps is None:
                    teacher_layer_num = len(teacher_reps) - 1
                    student_layer_num = len(student_reps) - 1
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps
                    assert len(new_student_reps) == len(new_teacher_reps)
                else:
                    new_student_reps = student_reps
                    new_teacher_reps = []
                    for t2s in args.hid_layer_maps:
                        if t2s >= 0:
                            new_teacher_reps.append(teacher_reps[t2s])
                        else:
                            new_teacher_reps.append(None)
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    if teacher_rep is None:
                        continue
                    rep_loss += F.mse_loss(student_rep, teacher_rep)
                    
                if (not args.not_use_att) and (not args.not_use_hid):
                    loss = att_loss + rep_loss
                elif args.not_use_att and (not args.not_use_hid):
                    loss = rep_loss
                elif (not args.not_use_att) and args.not_use_hid:
                    loss = att_loss
                else:
                    raise Exception('No training loss is defined.')

                student_model_engine.backward(loss)
                student_model_engine.step()
                
                global_step += 1
                
                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()
                rt_att_loss += att_loss.item()
                rt_rep_loss += rep_loss.item()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                rt_loss += loss.item()
                nb_rt_examples += input_ids.size(0)
                nb_rt_steps += 1
                
                pbar.update(1)
                
                if global_step % args.eval_step == 0:
                    
                    mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                    mean_att_loss = tr_att_loss * args.gradient_accumulation_steps / nb_tr_steps
                    mean_rep_loss = tr_rep_loss * args.gradient_accumulation_steps / nb_tr_steps
                    
                    mean_rt_loss = rt_loss * args.gradient_accumulation_steps / nb_rt_steps
                    mean_rt_att_loss = rt_att_loss * args.gradient_accumulation_steps / nb_rt_steps
                    mean_rt_rep_loss = rt_rep_loss * args.gradient_accumulation_steps / nb_rt_steps
                
                    report_step_metrics(args, global_step, {
                        'avg_loss': mean_loss,
                        'avg_att_loss': mean_att_loss,
                        'avg_rep_loss': mean_rep_loss,
                        'recent_loss': mean_rt_loss,
                        'recent_att_loss': mean_rt_att_loss,
                        'recent_rep_loss': mean_rt_rep_loss,
                        'lr': student_model_engine.lr_scheduler.get_last_lr()[0], #TODO
                    })
                    
                    rt_loss = 0.
                    rt_att_loss = 0.
                    rt_rep_loss = 0.
                    nb_rt_examples, nb_rt_steps = 0, 0
                    
                if global_step % args.save_step == 0:
                    
                    save_step_model(args, global_step, student_model_engine, tokenizer)
                    save_engine_checkpoint(args, global_step, student_model_engine)

                    
            mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
            mean_att_loss = tr_att_loss * args.gradient_accumulation_steps / nb_tr_steps
            mean_rep_loss = tr_rep_loss * args.gradient_accumulation_steps / nb_tr_steps
            
            report_step_metrics(args, global_step, {
                'avg_loss': mean_loss,
                'avg_att_loss': mean_att_loss,
                'avg_rep_loss': mean_rep_loss,
                'lr': student_model_engine.lr_scheduler.get_last_lr()[0], #TODO
            })
            
            save_step_model(args, global_step, student_model_engine, tokenizer)
            save_engine_checkpoint(args, global_step, student_model_engine)
    

if __name__ == "__main__":
    main()
