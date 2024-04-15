import collections
import copy
import os
import pdb
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances

from model.LightGCN import LightGCN
from model.MF import BPRMF

from sklearn.cluster import KMeans

from utils_func import demo_sample, shuffle, minibatch, test_model_auc, test_model_ndcg


class LASER(nn.Module):
    def __init__(self, params, sys_params, store_dict=None):
        super(LASER, self).__init__()
        self.num_group = sys_params.n_cluster
        self.device = params['device']
        if sys_params.base == 'lg':
            self.models = nn.ModuleList([LightGCN(params, sys_params).to(self.device) for _ in range(self.num_group)])
        else:
            self.models = nn.ModuleList([BPRMF(params, sys_params).to(self.device) for _ in range(self.num_group)])
        self.f = nn.Sigmoid()
        self.user_dict = store_dict

    def bpr_loss(self, user_indices, pos_item_indices, neg_item_indices, graph):
        user_emb, item_emb = self.get_aggr_emb(graph)
        user_vec = user_emb[user_indices]
        pos_item_vec = item_emb[pos_item_indices]
        neg_item_vec = item_emb[neg_item_indices]
        pos_scores = torch.mul(user_vec, pos_item_vec).sum(dim=1)
        neg_scores = torch.mul(user_vec, neg_item_vec).sum(dim=1)
        cf_loss = torch.mean((-1.0) * F.logsigmoid(pos_scores - neg_scores))
        return cf_loss

    def get_auc_ratings(self, users, items, graph=None):
        if graph:
            return self.models[-1].get_auc_ratings(users, items, graph[-1] if isinstance(graph, list) else graph)
        else:
            return self.models[-1].get_auc_ratings(users, items)

    def get_user_ratings(self, users, graph=None):
        if graph:
            return self.models[-1].get_user_ratings(users, graph[-1] if isinstance(graph, list) else graph)
        else:
            return self.models[-1].get_user_ratings(users)


def conduct_laser(model: LASER, graphs, train_records, unlearn_record, acc_val_records, ul_val_records, tr_rds,
                  sys_params, params, start_time):
    n_cluster = sys_params.n_cluster
    shards = detect_groups(model, unlearn_record)
    print(f"[LASER] INFLUENCED GROUPS: {shards}, current time: {(time.time() - start_time) / 60:.2f}min.")
    print(f"[LASER] START training {n_cluster} SEQ_MODELS, current time: {(time.time() - start_time) / 60:.2f}min.")
    for idx in range(n_cluster):
        print(f"[LASER] START training seq_model {idx}, current time: {(time.time() - start_time) / 60:.2f}min.")
        train_seq_model(model, idx, tr_rds[idx], acc_val_records, ul_val_records, sys_params, params, graphs)
    print(f"[LASER] FINISH training SEQ_MODELS, current time: {(time.time() - start_time) / 60:.2f}min.")
    print(f"TOTAL RUNNING TIME: {(time.time() - start_time) / 60:.2f}min")


def detect_groups(model, unlearning_record):
    shard_set = set()
    for u in unlearning_record:
        shard_set.add(model.user_dict[u])
    return shard_set


def divide_groups(train_records, user_embed, n_cluster):
    print('Start grouping!')
    u_embed = user_embed.detach().cpu().numpy()
    labels = KMeans(n_clusters=n_cluster, n_init=10).fit(u_embed).labels_
    embeds = [[] for _ in range(n_cluster)]
    for i in range(u_embed.shape[0]):
        embeds[labels[i]].append(u_embed[i])
    dis = [pairwise_distances(emb).mean() for emb in embeds]
    label_order_dict = dict()
    sort_index = numpy.argsort(dis)
    for i, label in enumerate(sort_index):
        label_order_dict[label] = i
    for i in range(len(labels)):
        labels[i] = label_order_dict[labels[i]]
    print('Finish grouping!')
    tr_rds = [collections.defaultdict(list) for _ in range(n_cluster)]
    store_dict = dict()
    for u in range(user_embed.shape[0]):
        label = labels[u]
        tr_rds[label][u] = train_records[u]
        store_dict[u] = label
    return tr_rds, store_dict


def train_seq_model(recModel, index, train_records, acc_val_records, ul_val_records, sys_params, params, graphs):
    start_time = time.time()
    if index > 0:
        recModel.models[index] = copy.deepcopy(recModel.models[index - 1])
    model = recModel.models[index]
    optimizer = torch.optim.Adam(model.parameters(), lr=sys_params.lr)
    total_epoch, best_epoch, best_res = 2000, 0, 0.005
    device = params['device']
    graph = graphs[index] if graphs else None
    for epoch in range(total_epoch):
        model.train()
        total_loss = 0
        runs = 0
        users, posItems, negItems = demo_sample(params['num_items'], train_records)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        for user, pos, neg in minibatch(users, posItems, negItems, batch_size=sys_params.bs):
            optimizer.zero_grad()
            u, i, n = user.to(device), pos.to(device), neg.to(device)
            # forward pass
            if sys_params.base == 'lg':
                loss, reg_loss = model.bpr_loss(graph, u, i, n)
                loss = loss + reg_loss * 1e-4
            else:
                loss = model(u, i, n)
            # backward and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            runs += 1
        model.eval()

        auc_acc = test_model_auc(model, acc_val_records, graph)
        auc_ul = test_model_auc(model, ul_val_records, graph)
        ndcg_acc = test_model_ndcg(model, train_records, acc_val_records, graph)
        if ndcg_acc > best_res:
            best_res = ndcg_acc
            best_epoch = epoch
            if index == len(recModel.models) - 1:
                if not os.path.exists(f'checkpoints/{sys_params.dataset}/{sys_params.unlearn}/{sys_params.tst_mth}'):
                    os.makedirs(f'checkpoints/{sys_params.dataset}/{sys_params.unlearn}/{sys_params.tst_mth}')
                    print('Made new dir!')
                torch.save(recModel.state_dict(),
                           f'checkpoints/{sys_params.dataset}/{sys_params.unlearn}/{sys_params.tst_mth}/{sys_params.base}-{sys_params.ul_perc}.pt')
        if epoch - best_epoch > 50:
            break

        print(
            'LASER Seq-model {}, Epoch [{}/{}], Runs: {}, Loss: {:.4f}, '
            'UL AUC: {:.4f}, ACC AUC: {:.4f}, ACC NDCG@20: {:.4f},'
            ' Seq_model training time: {:.2f}min'.format(
                index,
                epoch + 1,
                total_epoch,
                runs,
                total_loss / runs,
                auc_ul, auc_acc, ndcg_acc, (time.time() - start_time) / 60))
