
from __future__ import absolute_import, division, print_function
import time

import argparse
import csv
import logging
import os
import random
import sys
import shutil

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from data_processor.glue import glue_compute_metrics as compute_metrics
from data_processor.glue import glue_output_modes as output_modes
from data_processor.glue import glue_processors as processors
from skipbert.modeling import BertForSequenceClassification, BertForPreTraining
from skipbert.modeling import SkipBertForSequenceClassification
from transformers import BertConfig
from transformers import BertTokenizerFast as BertTokenizer

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def result_to_file(result, file_name, name='Eval'):
    with open(file_name, "a") as writer:
        logger.info(f"***** {name} results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def do_eval(args, model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    # for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
    for batch_ in eval_dataloader:
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            #print(label_ids.item())

            logits, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss

    return result

def do_predict(args, model, device, output_mode, tokenizer):
    task_name = args.task_name.lower()
    pred_task_names = ("mnli", "mnli-mm") if task_name == "mnli" else (task_name,)
    pred_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if task_name == "mnli" else (args.output_dir,)
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        if not os.path.exists(pred_output_dir):
            os.mkdir(pred_output_dir)
        processor = processors[pred_task]()
        label_list = processor.get_labels()
        pred_examples = processor.get_test_examples(args.data_dir)
        pred_features = convert_examples_to_features(pred_examples, label_list, args.max_seq_length, tokenizer,
                                                     output_mode)
        pred_data, pred_labels = get_tensor_data(output_mode, pred_features)
        pred_sampler = SequentialSampler(pred_data)
        pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=args.eval_batch_size)
        logger.info("  Num examples = %d", len(pred_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        preds = []
        for batch_ in tqdm(pred_dataloader, desc="predicting"):
            batch_ = tuple(t.to(device) for t in batch_)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
                logits, _, _ = model(input_ids, segment_ids, input_mask)

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

        label_map = {i: label for i, label in enumerate(label_list)}
        output_pred_file = os.path.join(pred_output_dir, pred_task.upper() + ".tsv")
        
        with open(output_pred_file, "w") as writer:
            logger.info("***** predict results *****")
            writer.write("index\tprediction\n")
            for index, pred in enumerate(tqdm(preds)):
                if pred_task == 'sts-b':
                    pred = round(pred, 3)
                else:
                    pred = label_map[pred]
                writer.write("%s\t%s\n" % (index, str(pred)))
    return preds
            
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def eval_milimm(args, device, global_step, label_list, num_labels, output_mode, student_model, tokenizer):

    task_name = "mnli-mm"
    processor = processors[task_name]()
    if not os.path.exists(args.output_dir + '-MM'):
        os.makedirs(args.output_dir + '-MM')
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    logger.info("***** Running mm evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)
    result = do_eval(args, student_model, task_name, eval_dataloader,
                     device, output_mode, eval_labels, num_labels)
    result['global_step'] = global_step
    tmp_output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
    result_to_file(result, tmp_output_eval_file)


def save_model(args, student_model, tokenizer, model_name):
    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
    output_model_file = os.path.join(args.output_dir, model_name)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)


def get_eval_result(args, device, eval_dataloader, eval_labels, global_step, num_labels, output_mode, step, student_model,
                    task_name, tr_att_loss, tr_cls_loss, tr_loss, tr_rep_loss):
    loss = tr_loss / (step + 1)
    cls_loss = tr_cls_loss / (step + 1)
    att_loss = tr_att_loss / (step + 1)
    rep_loss = tr_rep_loss / (step + 1)
    result = do_eval(args, student_model, task_name, eval_dataloader,
                     device, output_mode, eval_labels, num_labels)
    result['global_step'] = global_step
    result['cls_loss'] = cls_loss
    result['att_loss'] = att_loss
    result['rep_loss'] = rep_loss
    result['loss'] = loss
    return result

def distillation_loss(y, labels, teacher_scores, output_mode, T, alpha, reduction_kd='mean', reduction_nll='mean', reduce_T=1, is_teacher=True):
    teacher_T = T if is_teacher else 1
    rt = T*T/reduce_T if is_teacher else 1
    if output_mode == "classification":
        if teacher_scores is not None:
            student_likelihood = torch.nn.functional.log_softmax(y / T, dim=-1)
            targets_prob = torch.nn.functional.softmax(teacher_scores / T, dim=-1)
            d_loss = (- targets_prob * student_likelihood).mean() * T * T / reduce_T
        else:
            assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
            d_loss = 0.0
        nll_loss = torch.nn.functional.cross_entropy(y, labels, reduction=reduction_nll)
    elif output_mode == "regression":
        loss_mse = MSELoss()
        d_loss = loss_mse(y.view(-1), teacher_scores.view(-1))
        nll_loss = loss_mse(y.view(-1), labels.view(-1))
    else:
        assert output_mode in ["classification", "regression"]
        d_loss = 0.0
        nll_loss = 0.0
    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss
    return tol_loss, d_loss, nll_loss


def do_train(args):
    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if not args.do_predict and not args.do_eval and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        shutil.rmtree(args.output_dir, True)
        logger.info("exist Output directory ({}) removed.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    fw_args = open(args.output_dir + '/args.txt', 'w')
    fw_args.write(str(args))
    fw_args.close()

    task_name = args.task_name.lower()


#     logger.info('\nuse_logits:{}\n\nseperate:{}\ntrain_epoch:{}\n'.format(
#         args.use_logits, args.seperate, args.num_train_epochs))

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(                                              # obtain a tokenizer
        args.student_model_tokenizer if args.student_model_tokenizer is not None else args.student_model, 
        do_lower_case=args.do_lower_case)

    if not args.do_eval:
        if not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir)
        else:
            train_examples = processor.get_aug_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)
        train_data, _ = get_tensor_data(output_mode, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    teacher_config = BertConfig.from_pretrained(args.teacher_model)
    teacher_config.num_labels = num_labels
    teacher_config.fit_size = teacher_config.hidden_size
    student_config = BertConfig.from_pretrained(args.student_model)
    student_config.num_labels = num_labels
    student_config.fit_size = teacher_config.hidden_size
    if args.num_layers_student > 0:
        student_config.num_hidden_layers = args.num_layers_student
    if args.num_full_hidden_layers_student > 0:
        student_config.num_full_hidden_layers = args.num_full_hidden_layers_student
    else:
        student_config.num_full_hidden_layers = student_config.num_hidden_layers
    student_config.task_type = output_mode
    student_config.n_gram_left = args.n_gram_left
    student_config.n_gram_right = args.n_gram_right
    student_config.plot_mode = 'plot_passive'
    student_config.ngram_masking = 0.
    if not hasattr(student_config, 'enter_hidden_size'):
        student_config.enter_hidden_size = student_config.hidden_size
        
        
    if not args.do_eval:
        teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model, config=teacher_config)
        teacher_model.to(device)
        teacher_model.eval() # TODO
        
        
    student_model = eval(f"{args.model_type_student}ForSequenceClassification").from_pretrained(
        args.student_model, config=student_config, do_fit=args.do_fit, share_param=args.share_param)
    student_model.to(device)

    
    if args.freeze_lower_layers:
        for p in student_model.bert.embeddings.parameters():
            p.requires_grad = False
        for layer in student_model.bert.encoder.layer[:student_config.num_hidden_layers - student_config.num_full_hidden_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        try:
            for p in student_model.bert.shallow_skipping.linear.parameters():
                p.requires_grad = False
        except Exception as e:
            pass
        try:
            for p in student_model.bert.attn.parameters():
                p.requires_grad = False
        except Exception as e:
            pass
                
        student_model.bert.embeddings.dropout.p = 0.
        for layer in student_model.bert.encoder.layer[:student_config.num_hidden_layers - student_config.num_full_hidden_layers]:
            for m in layer.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.
    
    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(args, student_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            
    elif args.do_predict:
        logger.info("***** Running prediction *****")
        student_model.eval()
        do_predict(args, student_model, device, output_mode, tokenizer)
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
                
        schedule = args.schedule

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
        )

        loss_mse = MSELoss()
        def soft_cross_entropy(predicts, targets, T):
            student_likelihood = torch.nn.functional.log_softmax(predicts/T, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets/T, dim=-1)
            return (- targets_prob * student_likelihood).mean() * T ** 2 / 2

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            opt_level = 'O1'
            student_model, optimizer = amp.initialize(student_model, optimizer, opt_level=opt_level)

            teacher_model = teacher_model.half()

        
        for epoch_ in range(int(args.num_train_epochs)):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask)
                student_reps = student_reps[-args.num_full_hidden_layers_student-1:]

                with torch.no_grad():
                    teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)
                    b, e = args.num_masked_layers_teacher, (-args.num_masked_last_layers_teacher if args.num_masked_last_layers_teacher!=0 else None)
                    teacher_atts, teacher_reps = teacher_atts[:], teacher_reps[b:e]
                    
                # loss  
                att_loss = 0.
                rep_loss = 0.
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                              teacher_att)

                    tmp_loss = loss_mse(student_att, teacher_att)
                    att_loss += tmp_loss
                    

                teacher_layer_num = len(teacher_reps) - 1
                student_layer_num = len(student_reps) - 1
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                #new_teacher_reps = teacher_reps
                new_student_reps = student_reps
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss

                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()
                    
                if args.use_embedding:
                    embedding_loss = loss_mse(student_reps[0], teacher_reps[0])
                    
                if args.use_logits and epoch_ >= args.epochs_no_cls:
                    if isinstance(student_logits, tuple) or isinstance(student_logits, list):
                        cls_loss = None
                        _scale = 0.
                        for il, logits in enumerate(student_logits):
                            _loss, _, _ = distillation_loss(
                                logits, label_ids, teacher_logits, output_mode, T=args.T, alpha=args.alpha, reduce_T=args.reduce_T, is_teacher=args.is_teacher)
                            if cls_loss is None:
                                cls_loss = _loss
                            else:
                                cls_loss = _loss * (il+1.) + cls_loss
                            _scale += il + 1.
                            
                        cls_loss = cls_loss * (1./_scale)
                    else:
                        cls_loss, kd_loss, ce_loss = distillation_loss(
                            student_logits, label_ids, teacher_logits, output_mode, T=args.T, alpha=args.alpha, reduce_T=args.reduce_T, is_teacher=args.is_teacher)
                    
                    tr_cls_loss += cls_loss.item()
                else:
                    cls_loss = 0.
                    
                    
                if epoch_ >= args.epochs_no_cls:
                    beta = args.beta
                else:
                    beta = 1
                if args.use_embedding and args.use_att and args.use_rep:
                    loss = beta * (rep_loss + att_loss + embedding_loss) + cls_loss
                elif args.use_att and args.use_rep:
                    loss = beta * (rep_loss + att_loss) + cls_loss
                elif args.use_embedding and args.use_att:
                    loss = beta * (att_loss + embedding_loss) + cls_loss
                elif args.use_embedding and args.use_rep:
                    loss = beta * (rep_loss + embedding_loss) + cls_loss
                elif not args.use_embedding and args.use_att and not args.use_rep:
                    loss = beta * att_loss + cls_loss
                elif not args.use_embedding and not args.use_att and args.use_rep:
                    loss = beta * rep_loss + cls_loss
                else:
                    loss = cls_loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                if (global_step + 1) % args.eval_step == 0 and (not args.use_logits or epoch_ < args.epochs_no_cls):
                    logger.info("***** Save model *****")
                    logger.info(f"Epoch {epoch_}, Step {step}: Since 'cls_loss' is not enabled, save every ckpt.")
                    logger.info(f"loss: {tr_loss/nb_tr_steps:.4f}")
                    logger.info(f"att_loss: {tr_att_loss/nb_tr_steps:.4f}")
                    logger.info(f"rep_loss: {tr_rep_loss/nb_tr_steps:.4f}")
                    
                    save_model(args, student_model, tokenizer, model_name=WEIGHTS_NAME)
                    
                elif (global_step + 1) % args.eval_step == 0 and epoch_ >= args.epochs_no_cls and epoch_ >= args.epochs_no_eval:
                        
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    result = get_eval_result(args, device, eval_dataloader, eval_labels, global_step, num_labels, output_mode,
                                             step, student_model, task_name, tr_att_loss, tr_cls_loss, tr_loss,
                                             tr_rep_loss)
                    save_model(args, student_model, tokenizer, model_name='final_' + WEIGHTS_NAME)
                    result_to_file(result, output_eval_file)
                    
#                     result = get_eval_result(args, device, eval0_dataloader, eval0_labels, global_step, num_labels, output_mode,
#                                              step, student_model, task_name, tr_att_loss, tr_cls_loss, tr_loss,
#                                              tr_rep_loss)
                    
                    is_best = False
                    
                    if task_name == 'sts-b':
                        if result['corr'] > best_dev_acc:
                            best_dev_acc = result['corr']
                            is_best = True

                    elif task_name == 'cola':
                        if result['mcc'] > best_dev_acc:
                            best_dev_acc = result['mcc']
                            is_best = True
                            
                    elif task_name == 'mrpc' or task_name == 'qqp':
                        if result['acc'] > best_dev_acc:
                            best_dev_acc = result['acc']
                            is_best = True

                    elif result['acc'] > best_dev_acc:
                        best_dev_acc = result['acc']
                        is_best = True

                    if is_best:
                        logger.info("***** Save model *****")
                        save_model(args, student_model, tokenizer, model_name=WEIGHTS_NAME)
                        result['best_acc'] = best_dev_acc
                        result_to_file(result, output_eval_file, name='Eval0')

                        # Test mnli-mm
                        if task_name == "mnli":
                            eval_milimm(args, device, global_step, label_list, num_labels, output_mode, student_model, tokenizer)

                    student_model.train()



if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    taskname = "SST-2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=f"../data/{taskname}", type=str)
    parser.add_argument("--teacher_model", default=f"teacher/teacher_{taskname.lower()}",type=str)
    parser.add_argument("--student_model", default='', type=str,)
    parser.add_argument("--num_layers_student", default=-1, type=int,)
    parser.add_argument("--num_full_hidden_layers_student", default=-1, type=int,)
    parser.add_argument("--num_masked_layers_teacher", default=0, type=int,)
    parser.add_argument("--num_masked_last_layers_teacher", default=0, type=int,)
    parser.add_argument("--model_type_student", default='SkipBert', type=str,)
    parser.add_argument("--task_name", default=taskname, type=str)
    parser.add_argument("--student_model_tokenizer", default='bert-base-uncased', type=str)

    parser.add_argument("--output_dir", default=f"../model/{taskname}/6T6", type=str)

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument('--eval_step', type=int, default=200)
    
    parser.add_argument("--freeze_lower_layers", default=False, type=str2bool)
    
    parser.add_argument("--beta", type=float, default=0.01)
    
    parser.add_argument("--schedule", type=str, default='none') # 'warmup_linear'

    parser.add_argument('--T', type=float, default=1.)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--do_eval", default=False, type=str2bool)
    parser.add_argument("--do_predict", default=False, type=str2bool)
    parser.add_argument('--use_logits', default=True, type=str2bool)
    parser.add_argument("--use_att", default=True, type=str2bool)
    parser.add_argument("--use_rep", default=True, type=str2bool)
    parser.add_argument("--use_embedding", default=True, type=str2bool)
    
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--emb_linear", default=False, type=str2bool)
    parser.add_argument("--no_pretrain", action='store_true')
    parser.add_argument("--use_init_weight", action='store_true')
    parser.add_argument("--do_fit", default=False, type=str2bool)
    parser.add_argument("--share_param", default=True, type=str2bool)
    parser.add_argument("--do_lower_case", default=True, type=str2bool)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--aug_train', action='store_true')
    parser.add_argument('--data_url', type=str, default="")
    parser.add_argument('--reduce_T', type=float, default=1.0)
    parser.add_argument('--is_teacher', type=str2bool, default=True)
    
    parser.add_argument('--epochs_no_cls', type=int, default=0)
    parser.add_argument('--epochs_no_eval', type=int, default=0)
    
    parser.add_argument('--n_gram_left', type=int, default=1)
    parser.add_argument('--n_gram_right', type=int, default=1)

    args=parser.parse_args()

    dir_index = 0

    if args.seed is None:
        seed = random.randint(0, 100000)
        logger.info(f'randomly seeding: {seed}')
        args.seed = seed

    new_out = args.output_dir + "Model_" + args.student_model.split('/')[-1] + "_" + str(dir_index)
    while os.path.exists(new_out):
        dir_index += 1
        new_out = args.output_dir + "Model_" + args.student_model.split('/')[-1] + "_" + str(dir_index)
    args.output_dir = new_out
    os.makedirs(args.output_dir)

    default_params = {
        "cola": {"max_seq_length": 64},
        "mnli": {"max_seq_length": 128,},
        "mrpc": {"max_seq_length": 128},
        "sst-2": {"max_seq_length": 64,},
        "sts-b": {"max_seq_length": 128},
        "qqp": {"max_seq_length": 128,},
        "qnli": {"max_seq_length": 128},
        "rte": {"max_seq_length": 128},
        "wnli": {"max_seq_length": 128},
    }
    args.max_seq_length = default_params[args.task_name.lower()]["max_seq_length"]
    
    logger.info('The args: {}'.format(args))
    do_train(args)