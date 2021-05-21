
import os, sys, json, codecs
import argparse
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import *
from coref_model import *
from config import *
from coref_eval import *


def train(args):
    # path
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.cnn_filters = list(zip(args.cnn_filters[:-1:2], args.cnn_filters[1::2]))
    path_prefix = log_dir + "/{}.{}".format('coref', args.suffix)
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(str(args)))
    log_file.flush()

    # bert
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)

    # load data
    train_data, dev_data, test_data, vocabs = make_data(args, tokenizer)
    train_part_ids = random.sample(range(0, len(train_data)), (len(train_data) * args.train_ratio // 100))
    train_data = [train_data[i] for i in train_part_ids]
    print("Num training examples = {}".format(len(train_data)))
    print("Num dev examples = {}".format(len(dev_data)))
    print("Num test examples = {}".format(len(test_data)))

    # model
    print('Compiling model')
    model = AMRCorefModel(args, vocabs)
    model.to(args.device)

    # get pretrained performance
    best_f1 = 0.0
    if os.path.exists(path_prefix + ".model"):
        best_f1 = args.best_f1 if args.best_f1 and abs(args.best_f1) > 1e-4 \
            else eval_model(model, path_prefix, dev_data, test_data, log_file, best_f1)
        args.best_f1 = best_f1
        print("F1 score for pretrained model is {}".format(best_f1))

    # parameter grouping
    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm']
    grouped_params = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay) and 'bert' not in n],
                'weight_decay': 1e-4, 'lr': args.learning_rate},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay) and 'bert' not in n],
                'weight_decay': 0.0, 'lr': args.learning_rate}]
    assert sum(len(x['params']) for x in grouped_params) == len(named_params)

    # optimizer
    train_updates = len(train_data) * args.num_epochs
    if args.grad_accum_steps > 1:
        train_updates = train_updates // args.grad_accum_steps

    optimizer = optim.AdamW(grouped_params)
    # lr_schedular = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[10, 100], gamma=0.2)
    lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=3, verbose=False,
                                                        min_lr=1e-5)

    print("Starting the training loop, total *updating* steps = {}".format(train_updates))
    finished_steps, finished_epochs = 0, 0
    train_data_ids = list(range(0, len(train_data)))
    model.train()
    while finished_epochs < args.num_epochs:
        epoch_start = time.time()
        epoch_loss, epoch_loss_coref, epoch_loss_arg, epoch_acc = [], [], [], []

        random.shuffle(train_data_ids)

        for i in train_data_ids:
            inst = data_to_device(args, train_data[i])
            if len(inst['concept']) > 1500:
                continue
            outputs = model(inst)
            loss = outputs['loss']
            if args.use_classifier:
                loss_coref = outputs['loss_coref']
                loss_arg = outputs['loss_arg']
                acc = outputs['acc_arg']
                # print('Training step: %s, loss: %.3f ' % (i, loss.item()))

                epoch_loss_coref.append(loss_coref.item())
                epoch_loss_arg.append(loss_arg.item())
                epoch_acc.append(acc)
            epoch_loss.append(loss.item())
            if args.grad_accum_steps > 1:
                loss = loss / args.grad_accum_steps
            loss.backward() # just calculate gradient

            finished_steps += 1
            if finished_steps % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        lr = [group['lr'] for group in optimizer.param_groups]
        duration = time.time()-epoch_start
        print('\nCurrent epoch: %d  Current_Best_F1: %.3f Time: %.3f sec  Learning rate: %f ' %
              (finished_epochs, args.best_f1, duration, lr[0]))
        print('----Training loss: %.3f  Coref loss: %.3f  ARG loss: %.3f  ARG acc: %.3f' %
              (mean(epoch_loss), mean(epoch_loss_coref), mean(epoch_loss_arg), mean(epoch_acc)))
        lr_schedular.step(mean(epoch_loss))
        log_file.write('\nTraining loss: %s, time: %.3f sec\n' % (str(np.mean(epoch_loss)), duration))
        finished_epochs += 1
        best_f1 = eval_model(model, path_prefix, dev_data, test_data, log_file, best_f1, args)
        # lr_schedular.step(best_f1)


def eval_model(model, path_prefix, dev_batches, test_batches, log_file, best_f1, args):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cur_f1 = evaluate(model, dev_batches, log_file, args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if cur_f1 > best_f1:
        print('Saving weights, F1 {} (prev_best) < {} (cur)'.format(best_f1, cur_f1))
        log_file.write('Saving weights, F1 {} (prev_best) < {} (cur)\n'.format(best_f1, cur_f1))
        best_f1 = cur_f1
        args.best_f1 = cur_f1
        save_model(model, path_prefix)

    model.train()
    return best_f1


def save_model(model, path_prefix):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_bin_path = path_prefix + ".model"
    torch.save(model_to_save.state_dict(), model_bin_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_config(parser)
    # add
    # parser.add_argument("--model_path", default='ckpt/models')
    args = parser.parse_args()
    #
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)
    print("GPU available: %s      CuDNN: %s"
          % (torch.cuda.is_available(), torch.backends.cudnn.enabled))
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available() and args.gpu >= 0:
        print("Using GPU To Train...    GPU ID: ", args.gpu)
        args.device = torch.device('cuda', args.gpu)
        torch.cuda.manual_seed(args.random_seed)
    else:
        args.device = torch.device('cpu')
        print("Using CPU To Train... ")

    train(args)

