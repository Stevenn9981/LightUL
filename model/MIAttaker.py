import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MF import BPRMF


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class MIAttacker(nn.Module):
    def __init__(self, params, sys_params):
        super(MIAttacker, self).__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.device = params['device']
        self.dataset = params['dataset']
        self.latent_dim = 64

        self.shallow_model = BPRMF(params, sys_params).to(params['device'])

        self.shallow_model.load_state_dict(torch.load(
            f'checkpoints/{sys_params.dataset}/retrain/intr/mf-0.01.pt',
            map_location=params['device']),
            strict=False)

        self.MI_model = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim * 4),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 4, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, user_pos, pos_item_pos, neg_item_pos):

        all_users, all_items = self.shallow_model.get_embeddings()
        user_vec = all_users[user_pos]
        pos_item_vec = all_items[pos_item_pos]
        neg_item_vec = all_items[neg_item_pos]

        pos_scores = self.MI_model(torch.cat([user_vec, pos_item_vec], dim=1))
        neg_scores = self.MI_model(torch.cat([user_vec, neg_item_vec], dim=1))

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        mse_loss = F.mse_loss(pos_scores, pos_labels) + F.mse_loss(neg_scores, neg_labels)
        return mse_loss

    def get_MI_scores(self, user_pos, item_pos, model):
        all_users, all_items = model.get_embeddings()
        user_emb = all_users[user_pos]
        item_emb = all_items[item_pos]
        return self.MI_model(torch.cat([user_emb, item_emb], dim=1))

    def get_auc_ratings(self, user_indices, item_indices):
        all_users, all_items = self.shallow_model.get_embeddings()
        user_emb = all_users[user_indices]
        item_emb = all_items[item_indices]
        return self.MI_model(torch.cat([user_emb, item_emb], dim=1))
