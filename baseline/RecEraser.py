import collections
import os
import pdb
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.LightGCN import LightGCN
from model.MF import BPRMF

from sklearn.cluster import KMeans

from utils_func import demo_sample, shuffle, minibatch, test_model_auc, test_model_ndcg


class RecEraser(nn.Module):
    def __init__(self, params, sys_params, store_dict=None):
        super(RecEraser, self).__init__()
        self.num_shard = sys_params.n_cluster
        self.device = params['device']
        if sys_params.base == 'lg':
            self.models = nn.ModuleList([LightGCN(params, sys_params).to(self.device) for _ in range(self.num_shard)])
        else:
            self.models = nn.ModuleList([BPRMF(params, sys_params).to(self.device) for _ in range(self.num_shard)])
        self.attention = nn.ModuleDict({
            'user': nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1, bias=False),
                nn.Softmax(dim=0)
            ),
            'item': nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1, bias=False),
                nn.Softmax(dim=0)
            )
        })
        self.f = nn.Sigmoid()
        self.user_dict = store_dict

    def get_aggr_emb(self, graphs=None):
        user_embs, item_embs = [], []
        for idx, md in enumerate(self.models):
            if graphs:
                ue, ie = md.get_embeddings(graphs[idx])
            else:
                ue, ie = md.get_embeddings()
            user_embs.append(ue)
            item_embs.append(ie)
        user_embs, item_embs = torch.stack(user_embs), torch.stack(item_embs)
        user_embs = torch.sum(self.attention['user'](user_embs) * user_embs, dim=0)
        item_embs = torch.sum(self.attention['item'](item_embs) * item_embs, dim=0)
        return user_embs, item_embs

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
        user_emb, item_emb = self.get_aggr_emb(graph)
        user_vec = user_emb[users]
        item_vec = item_emb[items]
        return self.f(torch.mul(user_vec, item_vec).sum(1))

    def get_user_ratings(self, users, graph=None):
        user_emb, item_emb = self.get_aggr_emb(graph)
        user_vec = user_emb[users]
        return self.f(torch.matmul(user_vec, item_emb.T))


def conduct_receraser(model: RecEraser, graphs, train_records, unlearn_record, acc_val_records, ul_val_records, tr_rds,
                      sys_params, params, start_time):
    n_cluster = sys_params.n_cluster
    shards = detect_shards(model, unlearn_record)
    print(f"[RecEraser] INFLUENCED SHARDS: {shards}, current time: {(time.time() - start_time) / 60:.2f}min.")
    print(f"[RecEraser] START training {n_cluster} SUB_MODELS, current time: {(time.time() - start_time) / 60:.2f}min.")
    for idx in range(n_cluster):
        print(f"[RecEraser] START training sub-model {idx}, current time: {(time.time() - start_time) / 60:.2f}min.")
        train_submodel(model, idx, tr_rds[idx], acc_val_records, ul_val_records, sys_params, params, graphs)
    print(f"[RecEraser] FINISH training SUB_MODELS, current time: {(time.time() - start_time) / 60:.2f}min.")
    print(f"[RecEraser] START training ATTENTION layer, current time: {(time.time() - start_time) / 60:.2f}min.")
    train_att_layer(model, train_records, acc_val_records, ul_val_records, sys_params, params, graphs)
    print(f"[RecEraser] FINISH training ATTENTION layer, current time: {(time.time() - start_time) / 60:.2f}min.")
    print(f"TOTAL RUNNING TIME: {(time.time() - start_time) / 60:.2f}min")


def detect_shards(model, unlearning_record):
    shard_set = set()
    for u in unlearning_record:
        shard_set.add(model.user_dict[u])
    return shard_set


def divide_shards(train_records, user_embed, n_cluster):
    # Use user-based k-means (Interaction-based will cost enormous time and memory consumptions due to large quantity.)
    print('Start calculating k-means!')
    labels = KMeans(n_clusters=n_cluster, n_init=10).fit(user_embed.detach().cpu().numpy()).labels_
    print('Finish calculating k-means!')
    tr_rds = [collections.defaultdict(list) for _ in range(n_cluster)]
    store_dict = dict()
    for u in range(user_embed.shape[0]):
        label = labels[u]
        tr_rds[label][u] = train_records[u]
        store_dict[u] = label
    return tr_rds, store_dict


def train_submodel(recModel, index, train_records, acc_val_records, ul_val_records, sys_params, params, graphs):
    start_time = time.time()
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
        if epoch - best_epoch > 50:
            break

        print(
            'RecEraser Sub-model {}, Epoch [{}/{}], Runs: {}, Loss: {:.4f}, '
            'UL AUC: {:.4f}, ACC AUC: {:.4f}, ACC NDCG@20: {:.4f},'
            ' Sub_model training time: {:.2f}min'.format(
                index,
                epoch + 1,
                total_epoch,
                runs,
                total_loss / runs,
                auc_ul, auc_acc, ndcg_acc, (time.time() - start_time) / 60))


def train_att_layer(recModel, train_records, acc_val_records, ul_val_records, sys_params, params, graph):
    start_time = time.time()
    for model in recModel.models:
        for para in model.parameters():
            para.requires_grad = False
    optimizer = torch.optim.Adam(recModel.attention.parameters(), lr=sys_params.lr)
    total_epoch, best_epoch, best_res = 2000, 0, 0.005
    device = params['device']
    for epoch in range(total_epoch):
        recModel.train()
        total_loss, runs = 0, 0
        users, posItems, negItems = demo_sample(params['num_items'], train_records)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        for user, pos, neg in minibatch(users, posItems, negItems, batch_size=sys_params.bs * sys_params.n_cluster):
            optimizer.zero_grad()
            u, i, n = user.to(device), pos.to(device), neg.to(device)
            # forward pass
            loss = recModel.bpr_loss(u, i, n, graph)
            # backward and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            runs += 1
        recModel.eval()
        auc_acc = test_model_auc(recModel, acc_val_records, graph)
        auc_ul = test_model_auc(recModel, ul_val_records, graph)
        ndcg_acc = test_model_ndcg(recModel, train_records, acc_val_records, graph)

        if ndcg_acc > best_res:
            best_res = ndcg_acc
            best_epoch = epoch
            if not os.path.exists(f'checkpoints/{sys_params.dataset}/{sys_params.unlearn}/{sys_params.tst_mth}'):
                os.makedirs(f'checkpoints/{sys_params.dataset}/{sys_params.unlearn}/{sys_params.tst_mth}')
                print('Made new dir!')
            torch.save(recModel.state_dict(),
                       f'checkpoints/{sys_params.dataset}/{sys_params.unlearn}/{sys_params.tst_mth}/{sys_params.base}-{sys_params.ul_perc}.pt')
        if epoch - best_epoch > 25:
            break
        print(
            'RecEraser Attention Layer Training. Epoch [{}/{}], Runs: {} Loss: {:.4f}, '
            'UL AUC: {:.4f}, ACC AUC: {:.4f}, ACC NDCG@20: {:.4f}, '
            'Attention layer training time: {:.2f}min'.format(
                epoch + 1, total_epoch,
                runs,
                total_loss / runs,
                auc_ul, auc_acc, ndcg_acc, (time.time() - start_time) / 60))
