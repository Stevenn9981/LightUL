"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import dgl
import torch
from dgl.nn.pytorch import GraphConv
from torch import nn
import torch.nn.functional as F


class IMPGCN(nn.Module):
    def __init__(self, params, sys_params):
        super(IMPGCN, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.latent_dim = 64
        self.f = nn.Sigmoid()
        if sys_params.rAdj:
            self.conv = GraphConv(64, 64, weight=False, bias=False, norm='right', allow_zero_in_degree=True)
            if sys_params.gamma == 100:
                self.conv = GraphConv(64, 64, weight=False, bias=False, norm='left', allow_zero_in_degree=True)
        else:
            self.conv = GraphConv(64, 64, weight=False, bias=False, allow_zero_in_degree=True)
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.imp_n_layers = sys_params.imp_n_layers
        self.imp_n_class = sys_params.imp_n_classes
        self.imp_classify = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim, self.imp_n_class)
        )
        self.dropout = nn.Dropout(sys_params.dropout)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def computer(self, graph):
        users_emb0 = self.user_embedding.weight
        items_emb0 = self.item_embedding.weight
        emb0 = torch.cat([users_emb0, items_emb0])
        emb1 = self.conv(graph, self.dropout(emb0))
        emb1_user, emb1_item = torch.split(emb1, [self.num_users, self.num_items])

        # Item grouping
        item_class_vec = self.imp_classify(items_emb0 + emb1_item)
        _, topk_idx = torch.topk(item_class_vec, 1)
        item_out_subgraph_map = dict()
        item_in_subgraph_map = dict()
        for i in range(self.imp_n_class):
            node_indices = (topk_idx.squeeze() == i).nonzero().squeeze()
            item_out_subgraph_map[i] = dgl.out_subgraph(graph, node_indices + users_emb0.shape[0])
            item_in_subgraph_map[i] = dgl.in_subgraph(graph, node_indices + users_emb0.shape[0])

        embs = [emb0, emb1]

        # Training sub-graphs
        emb_layers = []
        for cls in range(self.imp_n_class):
            emb_input = emb1
            emb_input = self.dropout(emb_input)
            emb_layers.append([])
            for layer in range(2, self.imp_n_layers):
                tmp_emb = self.conv(item_out_subgraph_map[cls], emb_input)
                tmp_emb += self.conv(item_in_subgraph_map[cls], emb_input)
                emb_input = tmp_emb
                emb_layers[cls].append(tmp_emb)
        for layer in range(len(emb_layers[0])):
            tmp_emb = torch.zeros_like(emb1)
            for cls in range(self.imp_n_class):
                tmp_emb += emb_layers[cls][layer]
            embs.append(tmp_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items
        # users_emb = self.user_embedding.weight
        # items_emb = self.item_embedding.weight
        # layer_emb = torch.cat([users_emb, items_emb])
        # embs = [layer_emb]
        # for layer in range(self.n_layers):
        #     layer_emb = self.conv(graph, layer_emb)
        #     embs.append(layer_emb)
        #
        # embs = torch.stack(embs, dim=1)
        # light_out = torch.mean(embs, dim=1)
        # users, items = torch.split(light_out, [self.num_users, self.num_items])
        # return users, items

    def get_user_ratings(self, users, graph):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users.long()]
        rating = self.f(torch.matmul(users_emb, all_items.T))
        return rating

    def getEmbedding(self, graph, users, pos_items=None, neg_items=None):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users]

        if neg_items is None:
            return users_emb, all_items, self.user_embedding(users), self.item_embedding.weight

        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.user_embedding(users)
        pos_emb_ego = self.item_embedding(pos_items)
        neg_emb_ego = self.item_embedding(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, graph, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(graph, users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        return loss, reg_loss
