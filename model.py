import os
import pickle
import faiss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, RobertaModel, XLNetModel

from dataloader import TypeDataset, RelationDataset

lm_path = {
    'bert-base-uncased': 'bert-base-uncased',
    'bert-base-multilingual-uncased': 'bert-base-multilingual-uncased',
    'roberta-base': 'roberta-base',
    'bert-large-uncased': 'bert-large-uncased',
    'roberta-large': 'roberta-large'
}


class LMNet(nn.Module):
    def __init__(self,
                 lm='bert-base-uncased',
                 data_path=None,
                 num_type=None,
                 num_relation=None,
                 bert_path=None,
                 multi_columns=False,
                 attention_method='learned',
                 window_size=1):
        super().__init__()

        self.path = data_path
        self.num_type = num_type
        self.num_relation = num_relation
        self.lm = lm
        self.multi_columns = multi_columns

        # load the model or model checkpoint
        if bert_path is None:
            if lm == 'bert-base-cased' or lm == 'bert-large-cased' or lm == 'bert-base-uncased' or lm == 'bert-large-uncased':
                self.bert = BertModel.from_pretrained(lm_path[lm])
            elif lm == 'xlnet-base-cased':
                self.bert = XLNetModel.from_pretrained(lm_path[lm])
            elif lm == 'roberta-base' or lm == 'roberta-large':
                self.bert = RobertaModel.from_pretrained(lm_path[lm])

        else:
            output_model_file = bert_path
            model_state_dict = torch.load(output_model_file,
                                          map_location=lambda storage, loc: storage)
            if lm == 'bert-base-cased' or lm == 'bert-large-cased' or lm == 'bert-base-uncased' or lm == 'bert-large-uncased':
                self.bert = BertModel.from_pretrained(lm,
                                                      state_dict=model_state_dict)
            elif lm == 'xlnet-base-cased':
                self.bert = XLNetModel.from_pretrained(lm,
                                                       state_dict=model_state_dict)
            elif lm == 'roberta-base' or lm == 'roberta-large':
                self.bert = RobertaModel.from_pretrained(lm,
                                                         state_dict=model_state_dict)
        hidden_size = 768
        if 'large' in lm:
            hidden_size = 1024
        hidden_dropout_prob = 0.1

        self.dropout = nn.Dropout(hidden_dropout_prob)
        if self.multi_columns:
            self.type_linear = nn.Linear(hidden_size * 2, self.num_type)
            if self.num_relation != 0:
                self.relation_linear = nn.Linear(hidden_size * 2, self.num_relation)
            self.aggregator = Aggregator(input_dim=hidden_size, output_dim=hidden_size, attention_method=attention_method)
        else:
            self.type_linear = nn.Linear(hidden_size, self.num_type)
            if self.num_relation != 0:
                self.relation_linear = nn.Linear(hidden_size, self.num_relation)
        self.local_type_linear = nn.Linear(hidden_size, self.num_type)
        self.global_type_linear = nn.Linear(hidden_size, self.num_type)
        if self.num_relation != 0:
            self.local_relation_linear = nn.Linear(hidden_size, self.num_relation)
            self.global_relation_linear = nn.Linear(hidden_size, self.num_relation)

        # local
        self.window_size = window_size

        # global
        self.type_dataset = None
        self.relation_dataset = None
        self.type_global_index = faiss.IndexHNSWFlat(hidden_size, 32, faiss.METRIC_INNER_PRODUCT)
        self.relation_global_index = faiss.IndexHNSWFlat(hidden_size, 32, faiss.METRIC_INNER_PRODUCT)
        self.type_sentences = []
        self.type_labels = []
        self.type_embeddings = []
        self.relation_sentences = []
        self.relation_labels = []
        self.relation_embeddings = []

        # graph
        self.type_embeddings_tensor = None
        self.relation_embeddings_tensor = None

    def forward(self, x, mode, windows=None, windows_mask=None, local_layer=False, global_layer=False, neighbors=None):
        """Forward function of the models for classification."""
        output = self.bert(x)
        cls = output[0][:, 0, :]

        if self.multi_columns:
            neighbors = torch.tensor(neighbors)
            neighbors_embedding = []
            if mode == 'type':
                neighbors_embedding = self.type_embeddings_tensor[neighbors].cuda()
            elif mode == 'relation':
                neighbors_embedding = self.relation_embeddings_tensor[neighbors].cuda()
            neighbors_logits, attention_scores = self.aggregator(cls.unsqueeze(1), neighbors_embedding)
            cls = torch.cat((cls, neighbors_logits), dim=-1)
        cls = self.dropout(cls)
        logits = None
        if mode == 'type':
            logits = self.type_linear(cls)
        elif mode == 'relation':
            logits = self.relation_linear(cls)
        local_logits = None
        local_score = None
        global_logits = None
        if local_layer:
            local_logits, local_score = self.local_interpretability(mode, output, logits, windows, windows_mask)
        if global_layer:
            global_logits = self.global_interpretability(mode, output)
        return logits, local_logits, local_score, global_logits

    def local_interpretability(self, mode, output, logits, windows, windows_mask):
        # predicted label
        output = output[0].detach()
        label_pre = torch.sigmoid(logits).unsqueeze(dim=1)
        all_embeddings = output
        batch_size = all_embeddings.size(0)
        token_cnt = all_embeddings.size(1)
        embedding_dim = all_embeddings.size(2)
        cls_embeddings = output[:, 0, :].unsqueeze(dim=1).expand(-1, int(windows.size(0) / batch_size), -1)
        all_embeddings = all_embeddings.reshape(batch_size * token_cnt, embedding_dim)
        local_embeddings = all_embeddings[windows]
        local_embeddings = local_embeddings.mean(dim=1)
        local_embeddings = local_embeddings.reshape(
            (batch_size, int(local_embeddings.size(0) / batch_size), embedding_dim))
        cls_embeddings = local_embeddings - cls_embeddings
        assert mode in ["type", "relation"]
        if mode == "type":
            s = torch.sigmoid(self.local_type_linear(cls_embeddings))
        else:
            s = torch.sigmoid(self.local_relation_linear(cls_embeddings))
        # KL
        label_pre = label_pre.expand(-1, s.size(1), -1)
        local_score = torch.sum(
            F.kl_div(torch.log(torch.softmax(label_pre, dim=-1)), torch.softmax(s, dim=-1), reduction='none'),
            dim=2)
        # mask windows which equals [0,0,...,0]
        local_score *= windows_mask
        # weighted sum
        r_weight = F.softmax(local_score, dim=1)
        local_logits = torch.sum(torch.unsqueeze(r_weight, dim=2) * s, dim=1)
        del local_embeddings
        return local_logits, local_score

    def global_interpretability(self, mode, output, k=3):
        assert mode in ["type", "relation"]
        output = output[0].detach()
        cls = output[:, 0, :]
        vecs = cls.detach().cpu().numpy()
        faiss.normalize_L2(vecs)
        if mode == "type":
            D, I = self.type_global_index.search(vecs, k)
            vecs = [[self.type_embeddings[int(i)] for i in idx] for idx in I]
        else:
            D, I = self.relation_global_index.search(vecs, k)
            vecs = [[self.relation_embeddings[int(i)] for i in idx] for idx in I]
        D = torch.tensor(D).cuda()
        vecs = torch.tensor(vecs, requires_grad=False).cuda()
        weight = torch.unsqueeze(torch.softmax(D, dim=1), dim=2)
        vec_sum = torch.sum(vecs * weight, dim=1)
        if mode == "type":
            global_logits = torch.sigmoid(self.global_type_linear(vec_sum))
        else:
            global_logits = torch.sigmoid(self.global_relation_linear(vec_sum))
        return global_logits

    def get_global_explanation(self, x, mode, k=3):
        assert mode in ["type", "relation"]
        output = self.bert(x)
        cls = output[0][:, 0, :]
        cls = self.dropout(cls)
        vecs = cls.detach().cpu().numpy()
        faiss.normalize_L2(vecs)
        if mode == "type":
            D, I = self.type_global_index.search(vecs, k)
            sentences = [[self.type_sentences[int(i)] for i in idx] for idx in I]
            labels = [[self.type_labels[int(i)] for i in idx] for idx in I]
        else:
            D, I = self.relation_global_index.search(vecs, k)
            sentences = [[self.relation_sentences[int(i)] for i in idx] for idx in I]
            labels = [[self.relation_labels[int(i)] for i in idx] for idx in I]
        return sentences, labels

    def get_graph_explanation(self, x, neighbors, mode, k=10):
        assert mode in ["type", "relation"]
        output = self.bert(x)
        cls = output[0][:, 0, :]
        cls = self.dropout(cls)
        neighbors = torch.tensor(neighbors).cuda()
        neighbors_embedding = []
        if mode == 'type':
            neighbors_embedding = self.type_embeddings_tensor[neighbors].cuda()
        elif mode == 'relation':
            neighbors_embedding = self.relation_embeddings_tensor[neighbors].cuda()
        neighbors_logits, attention_scores = self.aggregator(cls.unsqueeze(1), neighbors_embedding)
        top_scores = torch.topk(attention_scores, k=k)
        top_indices = top_scores.indices.cuda()
        top_neighbors = torch.gather(neighbors, dim=1, index=top_indices)
        if mode == "type":
            sentences = [[self.type_sentences[int(i)] for i in idx] for idx in top_neighbors]
            labels = [[self.type_labels[int(i)] for i in idx] for idx in top_neighbors]
        else:
            sentences = [[self.relation_sentences[int(i)] for i in idx] for idx in top_neighbors]
            labels = [[self.relation_labels[int(i)] for i in idx] for idx in top_neighbors]
        return sentences, labels

    def build_global_store(self, dataset, mode):
        assert mode in ["type", "relation"]
        if mode == "type":
            self.type_dataset = dataset
        else:
            self.relation_dataset = dataset
        self._update_global_store(dataset, mode, first=True)

    def update_global_store(self):
        self.type_global_index.reset()
        self.type_embeddings = []
        self._update_global_store(self.type_dataset, "type")
        if self.relation_dataset is not None:
            self.relation_global_index.reset()
            self.relation_embeddings = []
            self._update_global_store(self.relation_dataset, "relation")

    def _update_global_store(self, dataset, mode, first=False):
        iterator = DataLoader(dataset=dataset,
                              batch_size=128,
                              shuffle=False,
                              collate_fn=TypeDataset.pad if mode == "type" else RelationDataset.pad)
        for batch in tqdm(iterator):
            x, labels, original_sentence, original_label, _, neighbors = batch
            x = torch.tensor(x).cuda()
            output = self.bert(x)
            cls = output[0][:, 0, :]
            vecs = cls.detach().cpu().numpy()
            faiss.normalize_L2(vecs)
            # update embeddings
            if mode == "type":
                self.type_global_index.add(vecs)
                self.type_embeddings.extend(vecs)
            else:
                self.relation_global_index.add(vecs)
                self.relation_embeddings.extend(vecs)
            # build first
            if first:
                if mode == "type":
                    self.type_sentences.extend(original_sentence)
                    self.type_labels.extend(original_label)
                else:
                    self.relation_sentences.extend(original_sentence)
                    self.relation_labels.extend(original_label)
        if mode == 'type':
            self.type_embeddings_tensor = torch.tensor(self.type_embeddings)
        else:
            self.relation_embeddings_tensor = torch.tensor(self.relation_embeddings)

    def save_global_store(self, save_path):
        with open(os.path.join(save_path, "type_embeddings.dict"), "wb") as wt:
            pickle.dump(self.type_embeddings, wt)
        with open(os.path.join(save_path, "type_sentences.dict"), "wb") as wt:
            pickle.dump(self.type_sentences, wt)
        with open(os.path.join(save_path, "type_labels.dict"), "wb") as wt:
            pickle.dump(self.type_labels, wt)
        with open(os.path.join(save_path, "relation_embeddings.dict"), "wb") as wt:
            pickle.dump(self.relation_embeddings, wt)
        with open(os.path.join(save_path, "relation_sentences.dict"), "wb") as wt:
            pickle.dump(self.relation_sentences, wt)
        with open(os.path.join(save_path, "relation_labels.dict"), "wb") as wt:
            pickle.dump(self.relation_labels, wt)
        faiss.write_index(self.type_global_index, os.path.join(save_path, "type_faiss.index"))
        faiss.write_index(self.relation_global_index, os.path.join(save_path, "relation_faiss.index"))

    def load_global_store(self, load_path):
        with open(os.path.join(load_path, "type_embeddings.dict"), "rb") as rd:
            self.type_embeddings = pickle.load(rd)
            self.type_embeddings_tensor = torch.tensor(self.type_embeddings)
        with open(os.path.join(load_path, "type_sentences.dict"), "rb") as rd:
            self.type_sentences = pickle.load(rd)
        with open(os.path.join(load_path, "type_labels.dict"), "rb") as rd:
            self.type_labels = pickle.load(rd)
        with open(os.path.join(load_path, "relation_embeddings.dict"), "rb") as rd:
            self.relation_embeddings = pickle.load(rd)
            self.relation_embeddings_tensor = torch.tensor(self.relation_embeddings)
        with open(os.path.join(load_path, "relation_sentences.dict"), "rb") as rd:
            self.relation_sentences = pickle.load(rd)
        with open(os.path.join(load_path, "relation_labels.dict"), "rb") as rd:
            self.relation_labels = pickle.load(rd)
        self.type_global_index = faiss.read_index(os.path.join(load_path, "type_faiss.index"))
        self.relation_global_index = faiss.read_index(os.path.join(load_path, "relation_faiss.index"))


class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim, attention_method='mean'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attention_method = attention_method
        if self.attention_method == 'self':
            self.query_linear = nn.Linear(input_dim, output_dim)
            self.key_linear = nn.Linear(input_dim, output_dim)
            self.value_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, neighbors):
        if self.attention_method == 'mean':
            result = torch.mean(neighbors, dim=0)
        elif self.attention_method == 'dot':
            attention_score = x * neighbors
            attention_score = torch.sum(attention_score, dim=-1)
            attention_score = F.softmax(attention_score, dim=-1)
            result = torch.bmm(attention_score.unsqueeze(1), neighbors).squeeze()
        elif self.attention_method == 'self':
            key = self.key_linear(x)
            queries = self.query_linear(neighbors)
            values = self.value_linear(neighbors)
            attention_score = torch.sum(key * queries, dim=-1)
            attention_score /= math.sqrt(self.output_dim)
            attention_score = F.softmax(attention_score, dim=-1)
            result = torch.bmm(attention_score.unsqueeze(1), values).squeeze()

        return result, attention_score
