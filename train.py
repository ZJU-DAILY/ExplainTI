import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from dataloader import TypeDataset, RelationDataset
from model import LMNet
from utils import *

lm_path = {
    'bert-base-uncased': './pre-trained-models/bert-base-uncased',
    'bert-base-multilingual-uncased': './pre-trained-models/bert-base-multilingual-uncased',
    'roberta-base': './pre-trained-models/roberta-base',
    'bert-large-uncased': './pre-trained-models/bert-large-uncased',
    'roberta-large': './pre-trained-models/roberta-large'
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Models.',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_name', type=str, default='WikiTable')
    parser.add_argument("--column_length",type=int,default=64)
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument('--n_epoch', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--model",type=str,default="bert-base-uncased")
    parser.add_argument('--use_large', action='store_true')
    parser.add_argument('--le', default=0.0, type=float)
    parser.add_argument('--ge', default=0.0, type=float)
    parser.add_argument('--se', action='store_true')
    parser.add_argument("--window_size",type=int,default=16)
    parser.add_argument("--top_k",type=int,default=3)
    parser.add_argument("--num_type",type=int,default=255)
    parser.add_argument("--num_relation",type=int,default=121)
    parser.add_argument('--attention_method', default='dot')
    parser.add_argument("--path",type=str,default="./data/WikiTable")
    parser.add_argument("--save_path",type=str,default="./checkpoint")
    parser.add_argument('--seed', default=888, type=int)
    parser.add_argument('--vis_device', type=str, default='0')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument("--update_epoch",type=int,default=5)

    args = parser.parse_args(args)
    return args


def reshape_windows_and_get_mask(windows, batch_size, token_cnt, window_size):
    windows = torch.tensor(windows).cuda()
    windows_mask = windows.sum(dim=2)
    windows_mask[windows_mask > 0] = 1
    windows_mask[windows_mask <= 0] = 0
    addition = (torch.arange(batch_size) * token_cnt) \
        .unsqueeze(1) \
        .unsqueeze(2) \
        .expand(-1, windows.size(1), window_size) \
        .cuda()
    windows_add = windows + addition
    windows_reshape = torch.reshape(windows_add, (windows_add.size(0) * windows_add.size(1), windows_add.size(2)))
    return windows_reshape, windows_mask


def train(model, mode, train_set, optimizer, scaler, scheduler=None,
          alpha=0,
          beta=0,
          batch_size=256,
          token_cnt=64,
          window_size=1):
    iterator = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=TypeDataset.pad if mode == 'type' else RelationDataset.pad)
    criterion = nn.BCEWithLogitsLoss()
    model.train()

    if mode == "relation":
        window_size *= 2

    for batch in tqdm(iterator):
        x, labels, original_sentence, original_label, windows, neighbors = batch
        cur_batch_size = len(x)
        x = torch.tensor(x)
        y = torch.zeros(x.shape[0], model.num_type if mode == 'type' else model.num_relation)
        for i, label in enumerate(labels):
            for l in label:
                y[i, l] = 1
        x = x.cuda()
        y = y.cuda()

        windows_reshape, windows_mask = reshape_windows_and_get_mask(windows, cur_batch_size, token_cnt, window_size)

        # forward
        optimizer.zero_grad()

        with autocast():
            logits, local_logits, _, global_logits = model(x, mode, windows_reshape, windows_mask,
                                                           local_layer=alpha > 0,
                                                           global_layer=beta > 0,
                                                           neighbors=neighbors, )
            bce_loss = criterion(logits, y)
            loss = bce_loss
            if alpha > 0:
                local_bce_loss = criterion(local_logits, y)
                loss += alpha * local_bce_loss
            if beta > 0:
                global_bce_loss = criterion(global_logits, y)
                loss += beta * global_bce_loss

        # back propagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()


def train_git(model, train_set, optimizer, scaler, scheduler=None,
              alpha=0,
              beta=0,
              batch_size=256,
              token_cnt=64,
              window_size=1):
    iterator = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=TypeDataset.pad)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for batch in tqdm(iterator):
        x, labels, original_sentence, original_label, windows, neighbors = batch
        cur_batch_size = len(x)
        x = torch.tensor(x).cuda()
        labels = torch.tensor(labels).cuda()
        windows_reshape, windows_mask = reshape_windows_and_get_mask(windows, cur_batch_size, token_cnt, window_size)

        # forward
        optimizer.zero_grad()

        with autocast():
            logits, local_logits, _, global_logits = model(x, 'type', windows_reshape, windows_mask,
                                                           local_layer=alpha > 0,
                                                           global_layer=beta > 0,
                                                           neighbors=neighbors, )
            bce_loss = criterion(logits, labels)
            loss = bce_loss
            if alpha > 0:
                local_bce_loss = criterion(local_logits, labels)
                loss += alpha * local_bce_loss
            if beta > 0:
                global_bce_loss = criterion(global_logits, labels)
                loss += beta * global_bce_loss

        # back propagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()


def eval_model(model, mode, dataset):
    iterator = DataLoader(dataset=dataset,
                          batch_size=256,
                          collate_fn=TypeDataset.pad if mode == 'type' else RelationDataset.pad)
    model.eval()
    y_truth = []
    y_pre = []
    for batch in tqdm(iterator):
        x, labels, _, _, _, neighbors = batch
        x = torch.tensor(x).cuda()
        truth = np.zeros((x.shape[0], model.num_type if mode == 'type' else model.num_relation))
        pre = np.zeros((x.shape[0], model.num_type if mode == 'type' else model.num_relation))
        for i, label in enumerate(labels):
            for l in label:
                truth[i][l] = 1
            y_truth.append(truth[i])
        with torch.no_grad():
            logits, _, _, _ = model(x, mode, neighbors=neighbors)
            logits = logits.cpu().numpy()
            for i in range(logits.shape[0]):
                for j in range(logits.shape[1]):
                    if logits[i][j] > 0.5:
                        pre[i][j] = 1
                y_pre.append(pre[i])

    accuracy, f1_micro, f1_macro, f1_weighted = evaluate(np.array(y_truth), np.array(y_pre))
    metric = {'accuracy': accuracy, 'f1-micro': f1_micro, 'f1-macro': f1_macro, 'f1-weighted': f1_weighted}
    return metric


def eval_model_git(model, dataset):
    iterator = DataLoader(dataset=dataset,
                          batch_size=256,
                          collate_fn=TypeDataset.pad)
    model.eval()
    y_truth = []
    y_pre = []
    for batch in tqdm(iterator):
        x, labels, _, _, _, neighbors = batch
        x = torch.tensor(x).cuda()
        y_truth.extend(labels)
        with torch.no_grad():
            logits, _, _, _ = model(x, 'type', neighbors=neighbors)
            logits = torch.argmax(logits, dim=1)
            logits = logits.cpu().numpy().tolist()
            y_pre.extend(logits)
    accuracy, f1_micro, f1_macro, f1_weighted = evaluate(np.array(y_truth), np.array(y_pre))
    metric = {'accuracy': accuracy, 'f1-micro': f1_micro, 'f1-macro': f1_macro, 'f1-weighted': f1_weighted}
    return metric


def eval_local_explanation(model, mode, dataset, save_path, token_cnt=64, window_size=3, top_k=3):
    iterator = DataLoader(dataset=dataset,
                          batch_size=256,
                          collate_fn=TypeDataset.pad if mode == 'type' else RelationDataset.pad)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(lm_path[args.model])
    data = []
    assert mode in ["type", "relation"]
    if mode == "relation":
        window_size *= 2
    for batch in tqdm(iterator):
        x, labels, original_sentence, original_label, windows, neighbors = batch
        x = torch.tensor(x).cuda()
        cur_batch_size = len(x)
        windows_reshape, windows_mask = reshape_windows_and_get_mask(windows, cur_batch_size, token_cnt, window_size)
        with torch.no_grad():
            _, _, local_score, _ = model(x, mode, windows_reshape, windows_mask,
                                         local_layer=alpha > 0,
                                         global_layer=beta > 0,
                                         neighbors=neighbors, )
            top_scores = torch.topk(local_score, k=top_k)
            top_indices = top_scores.indices.cuda()
            windows = torch.tensor(windows)
            for i in range(len(x)):
                phrases = []
                for j in range(top_k):
                    input_ids = x[i][windows[i][top_indices[i]][j]].cpu().numpy().tolist()
                    if mode == "type":
                        phrases.append(str(tokenizer.decode(input_ids).strip()))
                    else:
                        phrases.append([
                            str(tokenizer.decode(input_ids[:int(window_size / 2)]).strip()),
                            str(tokenizer.decode(input_ids[int(window_size / 2):]).strip()),
                        ])
                data.append({
                    "columns_sentence": original_sentence[i],
                    "label": original_label[i],
                    "relevant_phrases": phrases,
                })

    with open(save_path, "w") as wt:
        json.dump(data, wt)


def eval_explanation(model, eval_mode, mode, dataset, save_path, top_k=3):
    iterator = DataLoader(dataset=dataset,
                          batch_size=256,
                          collate_fn=TypeDataset.pad if mode == 'type' else RelationDataset.pad)
    model.eval()
    data = []
    for batch in tqdm(iterator):
        x, labels, original_sentence, original_label, _, neighbors = batch
        x = torch.tensor(x).cuda()
        with torch.no_grad():
            if eval_mode == "global":
                sentences, labels = model.get_global_explanation(x, mode, top_k)
            elif eval_mode == "graph":
                sentences, labels = model.get_graph_explanation(x, neighbors, mode, top_k)
            for i in range(len(x)):
                relevant_sentences = []
                for j in range(len(sentences[i])):
                    relevant_sentences.append({
                        "sentence": sentences[i][j],
                        "label": labels[i][j],
                    })
                data.append({
                    "columns_sentence": original_sentence[i],
                    "label": original_label[i],
                    "relevant_sentences": relevant_sentences,
                })
    with open(save_path, "w") as wt:
        json.dump(data, wt)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.vis_device
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    column_length = args.column_length
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    model_name=args.model
    if args.use_large:
        model_name=model_name.replace('base', 'large')
    multi_columns=args.se
    alpha=args.le
    beta=args.ge
    local_layer = alpha > 0
    global_layer = beta > 0
    data_name = args.data_name
    window_size=args.window_size
    top_k=args.top_k
    num_type=args.num_type
    num_relation=args.num_relation
    path=args.path
    save_path=args.save_path
    update_epoch=args.update_epoch

    cur_time = ' ' + datetime.now().strftime('%F %T')
    logger_name = data_name.replace('/', '')

    logger_name += cur_time
    set_logger(logger_name)
    logging.info(args)

    logging.info('Seed: {}'.format(torch.initial_seed()))

    logging.info("[Start] Loading dataset...")

    train_type_set = TypeDataset(path, 'train', model_name, column_length, multi_columns, window_size)
    valid_type_set = TypeDataset(path, 'dev', model_name, column_length, multi_columns, window_size)
    test_type_set = TypeDataset(path, 'test', model_name, column_length, multi_columns, window_size)

    if data_name == 'WikiTable':
        train_relation_set = RelationDataset(path, 'train', model_name, column_length, multi_columns, window_size)
        valid_relation_set = RelationDataset(path, 'dev', model_name, column_length, multi_columns, window_size)
        test_relation_set = RelationDataset(path, 'test', model_name, column_length, multi_columns, window_size)

    logging.info("[End] Loading dataset...")

    model = LMNet(
        lm=model_name,
        data_path=path,
        num_type=num_type,
        num_relation=num_relation,
        multi_columns=multi_columns,
        attention_method=args.attention_method,
        window_size=window_size,
    )
    model = model.cuda()

    optimizer_type = AdamW(model.parameters(), lr=args.lr)
    scaler_type = GradScaler()
    num_type_steps = (len(train_type_set) // batch_size) * n_epoch
    scheduler_type = get_linear_schedule_with_warmup(optimizer_type, num_warmup_steps=0,
                                                     num_training_steps=num_type_steps)

    if data_name == 'WikiTable':
        optimizer_relation = AdamW(model.parameters(), lr=args.lr)
        scaler_relation = GradScaler()
        num_relation_steps = (len(train_relation_set) // batch_size) * n_epoch

        scheduler_relation = get_linear_schedule_with_warmup(optimizer_relation, num_warmup_steps=0,
                                                             num_training_steps=num_relation_steps)

    logging.info(model)

    best_valid_type_f1_weight = 0.0
    best_test_type_metric = None
    best_valid_relation_f1_weight = 0.0
    best_test_relation_metric = None

    if global_layer or multi_columns:
        logging.info("build global embedding store...")
        model.build_global_store(train_type_set + valid_type_set + test_type_set, "type")
        if data_name == 'WikiTable':
            model.build_global_store(train_relation_set + valid_relation_set + test_relation_set, "relation")

    logging.info("[Start] training")
    for epoch in range(n_epoch):
        logging.info('epoch: {} training type prediction...'.format(epoch))
        if data_name == 'WikiTable':
            train(model,
                  'type',
                  train_type_set,
                  optimizer_type,
                  scaler_type,
                  scheduler_type,
                  alpha, beta, batch_size, column_length, window_size)
        else:
            train_git(model,
                      train_type_set,
                      optimizer_type,
                      scaler_type,
                      scheduler_type,
                      alpha, beta, batch_size, column_length, window_size)

        if data_name == 'WikiTable':
            valid_metric = eval_model(model, 'type', valid_type_set)
        else:
            valid_metric = eval_model_git(model, valid_type_set)
        logging.info(
            '[Valid Type]  Accuracy: {:.4f}  F1-micro: {:.4f}  F1-macro: {:.4f}  F1-weighted: {:.4f}'.format(
                valid_metric['accuracy'],
                valid_metric['f1-micro'],
                valid_metric['f1-macro'],
                valid_metric['f1-weighted']), )

        if data_name == 'WikiTable':
            test_metric = eval_model(model, 'type', test_type_set)
        else:
            test_metric = eval_model_git(model, test_type_set)
        logging.info(
            '[Test Type]  Accuracy: {:.4f}  F1-micro: {:.4f}  F1-macro: {:.4f}  F1-weighted: {:.4f}'.format(
                test_metric['accuracy'],
                test_metric['f1-micro'],
                test_metric['f1-macro'],
                test_metric['f1-weighted']))

        if valid_metric['f1-weighted'] > best_valid_type_f1_weight:
            best_valid_type_f1_weight = valid_metric['f1-weighted']
            best_test_type_metric = test_metric
            if args.save_model:
                logging.info('Save model and Generate explanations...')
                torch.save(model.state_dict(),
                           os.path.join(save_path, '{}-{}-best-type.pt'.format(data_name, model_name)))
                if local_layer:
                    eval_local_explanation(model, "type", test_type_set,
                                           "./explanations/{}-{}-type-local-explanation.json".format(data_name,
                                                                                                     model_name),
                                           column_length,
                                           window_size, top_k)
                if global_layer:
                    eval_explanation(model, "global", "type", test_type_set,
                                     "./explanations/{}-{}-type-global-explanation.json".format(data_name, model_name),
                                     top_k)
                if multi_columns:
                    eval_explanation(model, "graph", "type", test_type_set,
                                     "./explanations/{}-{}-type-structure-explanation.json".format(data_name,
                                                                                                   model_name), top_k)
        if data_name == 'WikiTable':
            logging.info('epoch: {} training relation prediction...'.format(epoch))
            train(model,
                'relation',
                train_relation_set,
                optimizer_relation,
                scaler_relation,
                scheduler_relation,
                alpha, beta, batch_size, column_length, window_size)

            valid_metric = eval_model(model, 'relation', valid_relation_set)
            logging.info(
                '[Valid Relation]  Accuracy: {:.4f}  F1-micro: {:.4f}  F1-macro: {:.4f}  F1-weighted: {:.4f}'.format(
                    valid_metric['accuracy'],
                    valid_metric['f1-micro'],
                    valid_metric['f1-macro'],
                    valid_metric['f1-weighted']))

            test_metric = eval_model(model, 'relation', test_relation_set)
            logging.info(
                '[Test Relation]  Accuracy: {:.4f}  F1-micro: {:.4f}  F1-macro: {:.4f}  F1-weighted: {:.4f}'.format(
                    test_metric['accuracy'],
                    test_metric['f1-micro'],
                    test_metric['f1-macro'],
                    test_metric['f1-weighted']))

            if valid_metric['f1-weighted'] > best_valid_relation_f1_weight:
                best_valid_relation_f1_weight = valid_metric['f1-weighted']
                best_test_relation_metric = test_metric
                if args.save_model:
                    logging.info('Save model and Generate explanations...')
                    torch.save(model.state_dict(),
                            os.path.join(save_path, '{}-{}-best-relation.pt'.format(data_name, model_name)))
                    if local_layer:
                        eval_local_explanation(model, "relation", test_relation_set,
                                            "./explanations/{}-{}-relation-local-explanation.json".format(data_name,
                                                                                                            model_name),
                                            column_length,
                                            window_size, top_k)
                    if global_layer:
                        eval_explanation(model, "global", "relation", test_relation_set,
                                        "./explanations/{}-{}-relation-global-explanation.json".format(data_name,
                                                                                                        model_name), top_k)
                    if multi_columns:
                        eval_explanation(model, "graph", "relation", test_relation_set,
                                        "./explanations/{}-{}-relation-structure-explanation.json".format(data_name,
                                                                                                        model_name),
                                        top_k)

        if (global_layer or multi_columns) and (epoch + 1) % int(update_epoch) == 0 and epoch != n_epoch - 1:
            logging.info('Update embedding store...')
            model.eval()
            model.update_global_store()
            model.train()
            
    logging.info('Finished!')
    logging.info('[Type Prediction]')
    logging.info(best_test_type_metric)
    logging.info('[Relation Prediction]')
    logging.info(best_test_relation_metric)
