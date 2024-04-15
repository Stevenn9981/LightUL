import pdb

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import random

from utils_func import minibatch


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def assign_phantom(u_emb, n_ph):
    if n_ph == u_emb.shape[0]:
        return np.arange(n_ph)
    print('Start phantom user assignment k-means!')
    # labels = KMeans(n_clusters=n_ph, n_init=10).fit(u_emb.detach().cpu().numpy()).labels_
    labels = np.random.randint(low=0, high=n_ph, size=u_emb.shape[0])
    print('Finish phantom user assignment k-means!')
    return labels


class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor, i_num):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.i_num = i_num

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], torch.randint(0, self.i_num, ())

    def __len__(self):
        return self.user_tensor.size(0)


class BPRMF(nn.Module):
    def __init__(self, params, sys_params):
        super(BPRMF, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.latent_dim = 64

        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)
        self.f = nn.Sigmoid()
        self.sys_params = sys_params
        if 'ncf' in sys_params.base:
            self.mlp = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, 1)
            )
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

    def forward(self, user_indices, pos_item_indices, neg_item_indices):
        user_vec = self.user_embedding(user_indices)
        pos_item_vec = self.item_embedding(pos_item_indices)
        neg_item_vec = self.item_embedding(neg_item_indices)
        if 'mf' in self.sys_params.base:
            pos_scores = self.f(torch.mul(user_vec, pos_item_vec).sum(dim=1))
            neg_scores = self.f(torch.mul(user_vec, neg_item_vec).sum(dim=1))
        else:
            pos_scores = self.mlp(torch.cat([user_vec, pos_item_vec], dim=1))
            neg_scores = self.mlp(torch.cat([user_vec, neg_item_vec], dim=1))
        cf_loss = torch.mean((-1.0) * F.logsigmoid(pos_scores - neg_scores))
        reg_loss = _L2_loss_mean(user_vec) + _L2_loss_mean(pos_item_vec) + _L2_loss_mean(neg_item_vec)
        return cf_loss + 1e-4 * reg_loss

    def bpr_loss_once(self, users, pos, neg):
        num_users, reg_loss, diff = 0, 0, torch.tensor([], device=self.device)
        for user_indices, pos_item_indices, neg_item_indices in minibatch(users, pos, neg, batch_size=self.sys_params.bs):
            user_vec = self.user_embedding(user_indices)
            pos_item_vec = self.item_embedding(pos_item_indices)
            neg_item_vec = self.item_embedding(neg_item_indices)
            reg_loss += (1 / 2) * (user_vec.norm(2).pow(2) +
                                   pos_item_vec.norm(2).pow(2) +
                                   neg_item_vec.norm(2).pow(2))
            num_users += float(len(user_indices))
            if 'mf' in self.sys_params.base:
                pos_scores = self.f(torch.mul(user_vec, pos_item_vec).sum(dim=1))
                neg_scores = self.f(torch.mul(user_vec, neg_item_vec).sum(dim=1))
            else:
                pos_scores = self.mlp(torch.cat([user_vec, pos_item_vec], dim=1))
                neg_scores = self.mlp(torch.cat([user_vec, neg_item_vec], dim=1))
            diff = torch.cat([diff, pos_scores - neg_scores])
        loss = torch.mean((-1.0) * F.logsigmoid(diff))
        reg_loss /= num_users
        return loss, reg_loss

    def get_user_ratings(self, user_indices):
        if 'ncf' in self.sys_params.base:
            user_emb = self.user_embedding(user_indices).repeat_interleave(self.num_items, dim=0)
            item_emb = self.item_embedding.weight.repeat(user_indices.shape[0], 1)
            return self.mlp(torch.cat([user_emb, item_emb], dim=1)).reshape(-1, self.num_items)
        return torch.matmul(self.user_embedding(user_indices), self.item_embedding.weight.T)

    def get_auc_ratings(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        if 'recul' in self.sys_params.unlearn:
            user_masks, item_masks = self.inf_users[user_indices], self.inf_items[item_indices]
            # user_emb[user_masks] = torch.matmul(user_emb[user_masks], self.unlearn_layer['user'])
            # item_emb[item_masks] = torch.matmul(item_emb[item_masks], self.unlearn_layer['item'])
            user_emb[user_masks] = self.unlearn_layer['user'](user_emb[user_masks])
            item_emb[item_masks] = self.unlearn_layer['item'](item_emb[item_masks])
            # user_emb = torch.matmul(user_emb, self.unlearn_layer['user'])
            # item_emb = torch.matmul(item_emb, self.unlearn_layer['item'])
        if 'ncf' in self.sys_params.base:
            return self.mlp(torch.cat([user_emb, item_emb], dim=1))
        return self.f(torch.mul(user_emb, item_emb).sum(1))

    def start_recul(self, unlearn_records, u_emb=None):
        for user in unlearn_records:
            self.inf_users[user] = True
            for item in unlearn_records[user]:
                self.inf_items[item] = True
        if self.sys_params.unlearn == 'recul_phantom':
            labels = assign_phantom(u_emb, self.ph_u_num)
            self.phantom_mapping = torch.tensor(labels, device=self.device)

        # for name, param in self.named_parameters():
        #     if 'unlearn' not in name:
        #         param.requires_grad = False

    def get_embeddings(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def recul_loss(self, users, unlearn_items, tr_items, neg_items):
        user_emb = self.user_embedding(users)
        ul_item_emb = self.item_embedding(unlearn_items)
        tr_item_emb = self.item_embedding(tr_items)
        neg_item_emb = self.item_embedding(neg_items)

        tr_masks, neg_masks = self.inf_items[tr_items], self.inf_items[neg_items]
        ul_item_ul_emb = self.unlearn_layer['item'](ul_item_emb)
        user_ul_emb = self.unlearn_layer['user'](user_emb)
        tr_item_emb[tr_masks] = self.unlearn_layer['item'](tr_item_emb[tr_masks])
        neg_item_emb[neg_masks] = self.unlearn_layer['item'](neg_item_emb[neg_masks])

        bef_ul_pos = self.f(torch.mul(user_emb, ul_item_emb).sum(1))
        aft_ul_pos = self.f(torch.mul(user_ul_emb, ul_item_ul_emb).sum(1))
        aft_tr_pos = self.f(torch.mul(user_ul_emb, tr_item_emb).sum(1))
        aft_tr_neg = self.f(torch.mul(user_ul_emb, neg_item_emb).sum(1))

        l1 = torch.mean(F.softplus(aft_ul_pos - bef_ul_pos)) + torch.mean(F.softplus(aft_ul_pos - aft_tr_pos))
        # l2 = F.mse_loss(user_emb, user_ul_emb) + F.mse_loss(ul_item_emb, ul_item_ul_emb)
        l2 = torch.mean(F.softplus(aft_tr_neg - aft_tr_pos))
        # reg_loss = _L2_loss_mean(user_ul_emb) + _L2_loss_mean(ul_item_ul_emb) + _L2_loss_mean(
        #     tr_item_emb) + _L2_loss_mean(neg_item_emb)
        return l1 + self.alpha * l2

    def phantom_loss(self, users, unlearn_items, neg_items):
        u_emb = self.user_embedding.weight
        ph_emb = self.ph_user_embedding.weight
        user_vec = torch.cat([u_emb, ph_emb])[users]
        pos_item_vec = self.item_embedding(unlearn_items)
        neg_item_vec = self.item_embedding(neg_items)
        if 'mf' in self.sys_params.base:
            pos_scores = self.f(torch.mul(user_vec, pos_item_vec).sum(dim=1))
            neg_scores = self.f(torch.mul(user_vec, neg_item_vec).sum(dim=1))
        else:
            pos_scores = self.mlp(torch.cat([user_vec, pos_item_vec], dim=1))
            neg_scores = self.mlp(torch.cat([user_vec, neg_item_vec], dim=1))
        cf_loss = torch.mean((-1.0) * F.logsigmoid(pos_scores - neg_scores))
        reg_loss = _L2_loss_mean(user_vec) + _L2_loss_mean(pos_item_vec) + _L2_loss_mean(neg_item_vec)
        return cf_loss + 1e-4 * reg_loss

    def emb_loss(self):
        user_emb, item_emb = self.user_embedding.weight, self.item_embedding.weight
        loss = F.mse_loss(user_emb[self.inf_users], self.unlearn_layer['user'](user_emb[self.inf_users]))
        loss = loss + F.mse_loss(item_emb[self.inf_items], self.unlearn_layer['item'](item_emb[self.inf_items]))
        return loss
