"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import pdb

import torch
from dgl.nn.pytorch import GraphConv
from torch import nn
import torch.nn.functional as F

from model.MF import assign_phantom
from utils_func import minibatch


class LightGCN(nn.Module):
    def __init__(self, params, sys_params):
        super(LightGCN, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.latent_dim = sys_params.latent_dim
        self.n_layers = sys_params.num_layers
        self.f = nn.Sigmoid()
        self.sys_params = sys_params
        self.conv = GraphConv(64, 64, weight=False, bias=False, allow_zero_in_degree=True)
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if 'recul' in sys_params.unlearn:
            self.inf_users = torch.zeros(self.num_users, dtype=torch.bool, device=params['device'])
            self.inf_items = torch.zeros(self.num_items, dtype=torch.bool, device=params['device'])
            self.unlearn_layer = nn.ModuleDict({
                "user": nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.latent_dim // 2, self.latent_dim)
                ),
                "item": nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.latent_dim // 2, self.latent_dim)
                )
            })
            self.alpha = sys_params.alpha
            if sys_params.unlearn == 'recul_phantom':
                self.ph_u_num = int(self.num_users * sys_params.phr)
                if self.ph_u_num > 0:
                    self.ph_user_embedding = torch.nn.Embedding(self.ph_u_num, self.latent_dim)
                else:
                    sys_params.unlearn = 'recul'

    def computer(self, graph):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        layer_emb = torch.cat([users_emb, items_emb])
        embs = [layer_emb]
        for layer in range(self.n_layers):
            layer_emb = self.conv(graph, layer_emb)
            embs.append(layer_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer_ph(self, graph):
        users_emb = self.user_embedding.weight
        ph_emb = self.ph_user_embedding.weight
        users_emb = torch.cat([users_emb, ph_emb])

        items_emb = self.item_embedding.weight
        layer_emb = torch.cat([users_emb, items_emb])
        embs = [layer_emb]
        for layer in range(self.n_layers):
            layer_emb = self.conv(graph, layer_emb)
            embs.append(layer_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users + self.ph_u_num, self.num_items])
        return users, items

    def get_user_ratings(self, users, graph):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users.long()]
        rating = self.f(torch.matmul(users_emb, all_items.T))
        return rating

    def getEmbedding(self, graph, users, pos_items, neg_items):
        all_users, all_items = self.computer(graph)
        users_emb = all_users[users]

        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.user_embedding(users)
        pos_emb_ego = self.item_embedding(pos_items)
        neg_emb_ego = self.item_embedding(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getEmbedding_ph(self, graph, users, pos_items, neg_items):
        all_users, all_items = self.computer_ph(graph)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        u_emb = self.user_embedding.weight
        ph_emb = self.ph_user_embedding.weight
        user_emb = torch.cat([u_emb, ph_emb])
        users_emb_ego = user_emb[users]

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

    def bpr_loss_once(self, users, posI, negI, graph):
        num_users, reg_loss, diff = 0, 0, torch.tensor([], device=self.device)
        for user, pos, neg in minibatch(users, posI, negI, batch_size=self.sys_params.bs):
            (users_emb, pos_emb, neg_emb,
             userEmb0, posEmb0, negEmb0) = self.getEmbedding(graph, user.long(), pos.long(), neg.long())
            reg_loss += (1 / 2) * (userEmb0.norm(2).pow(2) +
                                  posEmb0.norm(2).pow(2) +
                                  negEmb0.norm(2).pow(2))
            num_users += float(len(user))
            pos_scores = torch.mul(users_emb, pos_emb)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(users_emb, neg_emb)
            neg_scores = torch.sum(neg_scores, dim=1)
            diff = torch.cat([diff, neg_scores - pos_scores])
        loss = torch.mean(F.softplus(diff))
        reg_loss /= num_users
        return loss, reg_loss

    def get_auc_ratings(self, users, items, graph):
        all_users, all_items = self.computer(graph)
        if 'recul' in self.sys_params.unlearn:
            user_masks, item_masks = self.inf_users, self.inf_items
            all_users[user_masks] = self.unlearn_layer['user'](all_users[user_masks])
            all_items[item_masks] = self.unlearn_layer['item'](all_items[item_masks])
        users_emb, item_emb = all_users[users.long()], all_items[items.long()]
        rating = self.f(torch.mul(users_emb, item_emb).sum(1))
        return rating

    def get_embeddings(self, graph):
        return self.computer(graph)

    def start_recul(self, unlearn_records, u_emb=None):
        for user in unlearn_records:
            self.inf_users[user] = True
            for item in unlearn_records[user]:
                self.inf_items[item] = True
        if self.sys_params.unlearn == 'recul_phantom':
            labels = assign_phantom(u_emb, self.ph_u_num)
            self.phantom_mapping = torch.tensor(labels, device=self.device)

    def recul_loss(self, users, unlearn_items, tr_items, neg_items, graph):
        all_users, all_items = self.computer(graph)
        user_emb = all_users[users]
        ul_item_emb = all_items[unlearn_items]
        tr_item_emb = all_items[tr_items]
        neg_item_emb = all_items[neg_items]

        tr_masks, neg_masks = self.inf_items[tr_items], self.inf_items[neg_items]
        ul_item_ul_emb = self.unlearn_layer['item'](ul_item_emb)
        user_ul_emb = self.unlearn_layer['user'](user_emb)
        tr_item_emb[tr_masks] = self.unlearn_layer['item'](tr_item_emb[tr_masks])
        neg_item_emb[neg_masks] = self.unlearn_layer['item'](neg_item_emb[neg_masks])

        bef_ul_pos = torch.mul(user_emb, ul_item_emb).sum(1)
        aft_ul_pos = torch.mul(user_ul_emb, ul_item_ul_emb).sum(1)
        aft_tr_pos = torch.mul(user_ul_emb, tr_item_emb).sum(1)
        aft_tr_neg = torch.mul(user_ul_emb, neg_item_emb).sum(1)

        l1 = torch.mean(F.softplus(aft_ul_pos - aft_tr_pos)) + torch.mean(F.softplus(aft_ul_pos - bef_ul_pos))
        # l2 = F.mse_loss(user_emb, user_ul_emb) + F.mse_loss(ul_item_emb, ul_item_ul_emb)
        l2 = torch.mean(F.softplus(aft_tr_neg - aft_tr_pos))
        return l1 + self.alpha * l2

    def phantom_loss(self, users, unlearn_items, neg_items, graph):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding_ph(graph, users.long(),
                                                         unlearn_items.long(), neg_items.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        return loss + 1e-4 * reg_loss
