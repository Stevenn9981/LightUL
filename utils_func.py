import collections
import copy
import os
import pdb
import random
import time

import dgl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, jaccard_score

from baseline.IFRU import IFRU

TOP_Ks = [20, 50, 100]


def parse_args(parser):
    parser.add_argument('--dataset', default='ml-1M', type=str, help='dataset')
    parser.add_argument('--base', default='mf', type=str, help='base model')
    parser.add_argument('--unlearn', default='recul', type=str, help='unlearning model: none; retrain; recul, receraser')
    parser.add_argument('--tst_mth', default='intr', type=str, help='test model: ori, user, item, intr')
    parser.add_argument('--alpha', default=2, type=float, help='coefficient of Loss_2 (suggest 3 for MF and 1 for LG)')
    parser.add_argument('--beta', default=4, type=float, help='coefficient of phantom loss')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--latent_dim', default=64, type=int, help='Latent dim')
    parser.add_argument('--n_cluster', default=3, type=int, help='Number of clusters (only used in RecEraser)')
    parser.add_argument('--num_layers', default=3, type=int, help='Number of LightGCN layers')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--ul_perc', default=0.01, type=float, help='unlearning set ratio: 0.005, 0.01, 0.02')
    parser.add_argument('--phr', default=0.4, type=float, help='ratio of phantom users, belongs to [0, 1]')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--bs', default=512, type=int, help='batch size')
    parser.add_argument('--iteration', default=100, type=int, help='iterations in IFRU')
    parser.add_argument('--scale', default=1e-4, type=int, help='scale in IFRU')
    parser.add_argument('--ncf', action='store_true', help='whether use NCF for MF model')
    parser.add_argument('--train', action='store_true', help='whether train a new model or use pretrain')
    parser.add_argument('--use_poison', action='store_true', help='whether use the poisoned data')
    parser.add_argument('--test_sim', action='store_true',
                        help='Calculate the Jaccard similarity between recommendation '
                             'lists of the retrain and unlearned model')
    return parser.parse_args()


def create_train_graph(train_records, params):
    src = []
    dst = []
    u_i_pairs = set()
    for uid in train_records:
        iids = train_records[uid]
        for iid in iids:
            if (uid, iid) not in u_i_pairs:
                src.append(int(uid))
                dst.append(int(iid))
                u_i_pairs.add((uid, iid))
    u_num, i_num = params['num_users'], params['num_items']
    src_ids = torch.tensor(src)
    dst_ids = torch.tensor(dst) + u_num
    g = dgl.graph((src_ids, dst_ids), num_nodes=u_num + i_num)
    g = dgl.to_bidirected(g)
    return g


def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def RecallPrecision_ATk_cointer(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """

    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = right_pred / recall_n
    precis = right_pred / precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r_cointer(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return ndcg


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 2048)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk_cointer(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r_cointer(groundTrue, r, k))
    return {'recall': recall,
            'precision': pre,
            'ndcg': ndcg}


def test_model_ndcg(model, train_records, test_records, graph=None, verbose=False):
    model.eval()
    if 'ndcg' in test_records:
        test_records = test_records['ndcg']
    max_K = max(TOP_Ks)
    results = {'precision': dict(),
               'recall': dict(),
               'ndcg': dict()}
    for topk in TOP_Ks:
        results['precision'][topk] = []
        results['recall'][topk] = []
        results['ndcg'][topk] = []
    with torch.no_grad():
        users = list(test_records.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        u_batch_size = 200 if hasattr(model, 'sys_params') and 'ncf' in model.sys_params.base else 8192
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(users, batch_size=u_batch_size):
            users_list.append(batch_users)
            allPos = [train_records[u] for u in batch_users]
            groundTrue = [test_records[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(model.device)
            if graph:
                rating = model.get_user_ratings(batch_users_gpu, graph)
            else:
                rating = model.get_user_ratings(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -float('inf')
            _, rating_K = torch.topk(rating, k=max_K)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, TOP_Ks))
        for result in pre_results:
            for idx, topk in enumerate(TOP_Ks):
                results['recall'][topk].extend(result['recall'][idx])
                results['precision'][topk].extend(result['precision'][idx])
                results['ndcg'][topk].extend(result['ndcg'][idx])
        if verbose:
            for i, topk in enumerate(TOP_Ks):
                print(f'Precision@{topk}: {sum(results["precision"][topk]) / len(results["precision"][topk]):.4f}; '
                      f'Recall@{topk}: {sum(results["recall"][topk]) / len(results["recall"][topk]):.4f};  '
                      f'NDCG@{topk}: {sum(results["ndcg"][topk]) / len(results["ndcg"][topk]):.4f};  ')

        return sum(results["ndcg"][20]) / len(results["ndcg"][20])


def test_model_auc(model, test_records, graph=None):
    model.eval()
    u_batch_size = 512
    if 'auc' in test_records:
        test_records = test_records['auc']
    with torch.no_grad():
        label_list = []
        rating_list = []
        for batch_intrs in minibatch(test_records, batch_size=u_batch_size):
            test_users = torch.LongTensor([r[0][0] for r in batch_intrs]).to(model.device)
            test_items = torch.LongTensor([r[0][1] for r in batch_intrs]).to(model.device)
            labels = torch.LongTensor([r[1] for r in batch_intrs]).to(model.device)
            if graph:
                rating = model.get_auc_ratings(test_users, test_items, graph)
            else:
                rating = model.get_auc_ratings(test_users, test_items)
            label_list.append(labels)
            rating_list.append(rating)
        label_list = torch.cat(label_list)
        rating_list = torch.cat(rating_list)
        auc = roc_auc_score(label_list.cpu().numpy(), torch.sigmoid(rating_list).cpu().numpy())
        return auc


def read_dataset(dataset, test_method, percentage):
    u_num = 0
    i_num = 0
    r_num = 0
    train_records = collections.defaultdict(list)
    val_records = {'auc': [], 'ndcg': collections.defaultdict(list)}
    test_records = []
    unlearn_records = collections.defaultdict(list)
    file = open(f'dataset/{dataset}/train.txt', 'r')
    for line in file.readlines():
        ele = line.strip().split(' ')
        user, items = ele[0], ele[1:]
        u_num = max(u_num, int(user))
        for item in items:
            r_num += 1
            i_num = max(i_num, int(item))
            train_records[int(user)].append(int(item))
    file.close()

    # validation set for accuracy test
    val_fn = f'dataset/{dataset}/test_auc.txt'
    file = open(val_fn, 'r')
    for line in file.readlines():
        ele = line.strip().split('\t')
        user, item, label = int(ele[0]), int(ele[1]), int(ele[2])
        val_records['auc'].append(((user, item), label))
        if label == 1:
            val_records['ndcg'][user].append(item)
    file.close()

    # test set for unlearning test
    test_fn = f'dataset/{dataset}/unlearn_test/{test_method}_{percentage}.txt'
    if test_method == 'ori':
        test_fn = f'dataset/{dataset}/unlearn_test/intr_0.01.txt'
    file = open(test_fn, 'r')
    for line in file.readlines():
        ele = line.strip().split('\t')
        user, item, label = int(ele[0]), int(ele[1]), int(ele[2])
        test_records.append(((user, item), label))
    file.close()

    if test_method != 'ori':
        file = open(f'dataset/{dataset}/unlearn_set/{test_method}_{percentage}.txt', 'r')
        for line in file.readlines():
            ele = line.strip().split(' ')
            user, items = ele[0], ele[1:]
            for item in items:
                unlearn_records[int(user)].append(int(item))
        file.close()

    params = {'num_users': u_num + 1,
              'num_items': i_num + 1,
              'num_train_intrs': r_num,
              'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'dataset': dataset}
    print(f'Read {dataset} dataset done! u_num: {u_num + 1}, i_num: {i_num + 1}')
    return train_records, val_records, test_records, unlearn_records, params


def train_mf_model(model, train_records, acc_val_records, ul_val_records, sys_params, params):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=sys_params.lr)
    total_epoch, best_epoch, best_res = 2000, 0, 0.005
    unlearn = sys_params.unlearn
    device = params['device']
    for epoch in range(total_epoch):
        model.train()
        tim1 = time.time()
        total_loss = 0
        runs = 0
        users, posItems, negItems = demo_sample(params['num_items'], train_records)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        for user, pos, neg in minibatch(users, posItems, negItems, batch_size=sys_params.bs):
            optimizer.zero_grad()
            u, i, n = user.to(device), pos.to(device), neg.to(device)
            # forward pass
            loss = model(u, i, n)
            # backward and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            runs += 1
        model.eval()
        auc_acc = test_model_auc(model, acc_val_records)
        auc_ul = test_model_auc(model, ul_val_records)
        ndcg_acc = test_model_ndcg(model, train_records, acc_val_records)
        if ndcg_acc > best_res:
            best_res = ndcg_acc
            best_epoch = epoch
            if not os.path.exists(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}'):
                os.makedirs(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}')
                print('Made new dir!')
            torch.save(model.state_dict(),
                       f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}/{sys_params.base}-{sys_params.ul_perc}.pt')
        if epoch - best_epoch > 60:
            break

        print(
            'Epoch [{}/{}], Loss: {:.4f}, UL AUC: {:.4f}, ACC AUC: {:.4f}, '
            'ACC NDCG@20: {:.4f}, Time: {:.2f}s'.format(
                epoch + 1, total_epoch,
                total_loss / runs,
                auc_ul,
                auc_acc,
                ndcg_acc,
                time.time() - tim1))
    print(f'Best NDCG@20: {best_res:.4f}, Best epoch: {best_epoch}, total time: {time.time() - start_time:.3f}s')


def train_miattacker_model(model, train_records, test_records, ul_val_records, sys_params, params):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=sys_params.lr)
    total_epoch, best_epoch, best_res = 2000, 0, 0.005
    unlearn = sys_params.unlearn
    device = params['device']
    users = [u for u in train_records for _ in train_records[u]]
    pos_items = [pos_i for u in train_records for pos_i in train_records[u]]
    neg_items = []
    for u in users:
        if u in test_records:
            neg_items.append(random.choice(test_records[u]))
        else:
            neg_items.append(random.randint(0, params['num_items'] - 1))
    for epoch in range(total_epoch):
        model.train()
        total_loss = 0
        runs = 0
        for user, pos, neg in minibatch(users, pos_items, neg_items, batch_size=sys_params.bs):
            optimizer.zero_grad()
            user, pos, neg = torch.LongTensor(user), torch.LongTensor(pos), torch.LongTensor(neg)
            u, i, n = user.to(device), pos.to(device), neg.to(device)
            # forward pass
            loss = model(u, i, n)
            # backward and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            runs += 1
        model.eval()
        auc_ul = test_model_auc(model, ul_val_records)
        if auc_ul > best_res:
            best_res = auc_ul
            best_epoch = epoch
            if not os.path.exists(f'checkpoints/{sys_params.dataset}/MIAttacker/'):
                os.makedirs(f'checkpoints/{sys_params.dataset}/MIAttacker/')
                print('Made new dir!')
            torch.save(model.state_dict(),
                       f'checkpoints/{sys_params.dataset}/MIAttacker/{sys_params.base}-{sys_params.ul_perc}.pt')
        if epoch - best_epoch > 60:
            break
        print(f'Current best auc: {best_res:.4f}')
    print(f'Best AUC: {best_res:.4f}, Best epoch: {best_epoch}, total time: {time.time() - start_time:.3f}s')


def train_lg_model(model, graph, train_records, acc_val_records, ul_val_records, sys_params, params):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=sys_params.lr)
    total_epoch, best_epoch, best_res = 2000, 0, 0.005
    unlearn = sys_params.unlearn
    device = params['device']
    for epoch in range(total_epoch):
        model.train()
        tim1, total_loss, runs = time.time(), 0, 0
        users, posItems, negItems = demo_sample(params['num_items'], train_records)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        for user, pos, neg in minibatch(users, posItems, negItems, batch_size=sys_params.bs):
            optimizer.zero_grad()
            u, i, n = user.to(device), pos.to(device), neg.to(device)
            # forward pass
            loss, reg_loss = model.bpr_loss(graph, u, i, n)
            loss = loss + reg_loss * 1e-4
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
            if not os.path.exists(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}'):
                os.makedirs(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}')
                print('Made new dir!')
            torch.save(model.state_dict(),
                       f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}/{sys_params.base}-{sys_params.ul_perc}.pt')
        if epoch - best_epoch > 60:
            break

        print(
            'Epoch [{}/{}], Loss: {:.4f}, UL AUC: {:.4f}, ACC AUC: {:.4f}, '
            'ACC NDCG@20: {:.4f}, Time: {:.2f}s'.format(
                epoch + 1, total_epoch,
                total_loss / runs,
                auc_ul,
                auc_acc,
                ndcg_acc,
                time.time() - tim1))
    print(f'Best NDCG@20: {best_res:.4f}, Best epoch: {best_epoch}, total time: {time.time() - start_time:.3f}s')


def poison_training_data(train_records, sys_params, params):
    if not os.path.exists(f'dataset/{sys_params.dataset}/poison_data/intr_{sys_params.ul_perc}'):
        os.makedirs(f'dataset/{sys_params.dataset}/poison_data/')
        u_num = params['num_users']
        i_num = params['num_items']
        r_num = params['num_train_intrs']

        p_nums = int(sys_params.ul_perc * r_num)

        poison_users = np.random.randint(0, u_num, p_nums)
        poison_items = np.random.randint(0, i_num, p_nums)
        poison_records = collections.defaultdict(list)

        for i in range(p_nums):
            user, item = poison_users[i], poison_items[i]
            poison_records[user].append(item)

        with open(f'dataset/{sys_params.dataset}/poison_data/intr_{sys_params.ul_perc}', 'w') as f:
            for user in poison_records:
                f.write(str(user) + ' ')
                for item in poison_records[user]:
                    f.write(str(item) + ' ')
                f.write('\n')

    with open(f'dataset/{sys_params.dataset}/poison_data/intr_{sys_params.ul_perc}', 'r') as file:
        for line in file.readlines():
            ele = line.strip().split(' ')
            user, items = ele[0], ele[1:]
            for item in items:
                train_records[int(user)].append(int(item))


def conduct_recul_unlearn(model, graph, train_records, acc_val_records, ul_val_records, unlearn_records, sys_params,
                          params):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=sys_params.lr)
    total_epoch, best_epoch, best_res, best_ndcg = 2000, 0, 0.005, 0
    unlearn = sys_params.unlearn
    device = params['device']
    ul_num = 0
    if sys_params.unlearn == 'recul_phantom':
        phantom_params = params.copy()
        phantom_params['num_users'] = params['num_users'] + int(sys_params.phr * params['num_users'])
        phantom_records = copy.deepcopy(train_records)
        for user in unlearn_records:
            ul_num += len(unlearn_records[user])
            phantom_records[int(model.phantom_mapping[user]) + params['num_users']].extend(unlearn_records[user])
        if graph:
            phantom_graph = create_train_graph(phantom_records, phantom_params).to(params['device'])
    print(f"[RECUL] PREPROCESS FINISHED. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
    for epoch in range(total_epoch):
        model.train()
        total_loss = 0
        runs = 0
        users, ulItems, tr_items, negItems = demo_unlearn_sample(params['num_items'], unlearn_records, train_records)
        users, ulItems, tr_items, negItems = shuffle(users, ulItems, tr_items, negItems)
        for user, ul, tr, neg in minibatch(users, ulItems, tr_items, negItems, batch_size=sys_params.bs):
            optimizer.zero_grad()
            u, i, t, n = user.to(device), ul.to(device), tr.to(device), neg.to(device)
            # forward pass
            loss = model.recul_loss(u, i, t, n, graph) if graph else model.recul_loss(u, i, t, n)
            if sys_params.unlearn == 'recul_phantom':
                phan_loss = model.phantom_loss(u, i, n, phantom_graph) if graph else model.phantom_loss(u, i, n)
                loss = loss + sys_params.beta * phan_loss
            # backward and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            runs += 1
        # if sys_params.unlearn == 'recul_phantom':
        #     for _ in range(3):
        #         users, ulItems, negItems = demo_sample_ph(params['num_items'], ul_num, phantom_records)
        #         users, ulItems, negItems = shuffle(users, ulItems, negItems)
        #         for user, ul, neg in minibatch(users, ulItems, negItems, batch_size=sys_params.bs):
        #             optimizer.zero_grad()
        #             u, i, n = user.to(device), ul.to(device), neg.to(device)
        #             # forward pass
        #             loss = model.phantom_loss(u, i, n, phantom_graph) if graph else model.phantom_loss(u, i, n)
        #             # backward and optimize
        #             loss.backward()
        #             optimizer.step()
        #             total_loss += loss.item()
        #             runs += 1

        model.eval()

        auc_acc = test_model_auc(model, acc_val_records, graph)
        auc_ul = test_model_auc(model, ul_val_records, graph)
        ndcg_acc = test_model_ndcg(model, train_records, acc_val_records, graph)

        # if sys_params.unlearn == 'recul_phantom':
        #     if ndcg_acc > best_ndcg:
        #         best_ndcg = ndcg_acc
        #         best_epoch = epoch
        #         if not os.path.exists(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}'):
        #             os.makedirs(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}')
        #             print('Made new dir!')
        #         torch.save(model.state_dict(),
        #                    f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}/{sys_params.base}-{sys_params.ul_perc}.pt')
        # else:
        if auc_ul > best_res + 0.01:
            best_res = auc_ul
            best_epoch = epoch
            if not os.path.exists(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}'):
                os.makedirs(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}')
                print('Made new dir!')
            torch.save(model.state_dict(),
                       f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}/{sys_params.base}-{sys_params.ul_perc}.pt')
        if epoch - best_epoch > 10:
            break

        print(
            '[RecUL] Epoch [{}/{}], Loss: {:.4f}, UL AUC: {:.4f}, ACC AUC: {:.4f}, '
            'ACC NDCG@20: {:.4f}, Current time: {:.2f}min'.format(
                epoch + 1, total_epoch,
                total_loss / runs,
                auc_ul,
                auc_acc,
                ndcg_acc,
                (time.time() - start_time) / 60))
    print(
        f'Best UL AUC: {best_res:.4f}, Best epoch: {best_epoch}, total time: {(time.time() - start_time) / 60:.2f}min')


def conduct_ifru_unlearn(model, graph, train_records, acc_val_records, ul_val_records, unlearn_records, sys_params,
                         params):
    start_time = time.time()
    device = params['device']
    total_epoch, best_epoch, best_res, best_ndcg = 2000, 0, 0.005, 0
    unlearn = sys_params.unlearn
    ifru_model = IFRU(params, sys_params)
    print(f"[IFRU] PREPROCESS FINISHED. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")

    tr_users, tr_posItems, tr_negItems = demo_sample(params['num_items'], train_records)
    tr_users, tr_posItems, tr_negItems = tr_users.to(device), tr_posItems.to(device), tr_negItems.to(device)
    ul_users, ul_Items, _, ul_negItems = demo_unlearn_sample(params['num_items'], unlearn_records, train_records)
    ul_users, ul_Items, ul_negItems = ul_users.to(device), ul_Items.to(device), ul_negItems.to(device)
    _, grad_all = ifru_model.forward_once_grad(model, tr_users, tr_posItems, tr_negItems, graph)
    _, grad_unl = ifru_model.forward_once_grad(model, ul_users, ul_Items, ul_negItems, graph)

    for epoch in range(total_epoch):
        model.train()

        ifru_model.if_approxi(model, grad_all, grad_unl)

        model.eval()

        auc_acc = test_model_auc(model, acc_val_records, graph)
        auc_ul = test_model_auc(model, ul_val_records, graph)
        ndcg_acc = test_model_ndcg(model, train_records, acc_val_records, graph)

        if auc_ul > best_res + 0.005:
            best_res = auc_ul
            best_epoch = epoch
            if not os.path.exists(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}'):
                os.makedirs(f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}')
                print('Made new dir!')
            torch.save(model.state_dict(),
                       f'checkpoints/{sys_params.dataset}/{unlearn}/{sys_params.tst_mth}/{sys_params.base}-{sys_params.ul_perc}.pt')
        if epoch - best_epoch > 50:
            break

        print(
            '[IFRU] Epoch [{}/{}], UL AUC: {:.4f}, ACC AUC: {:.4f}, '
            'ACC NDCG@20: {:.4f}, Current time: {:.2f}min'.format(
                epoch + 1, total_epoch,
                auc_ul,
                auc_acc,
                ndcg_acc,
                (time.time() - start_time) / 60))
    print(
        f'Best UL AUC: {best_res:.4f}, Best epoch: {best_epoch}, total time: {(time.time() - start_time) / 60:.2f}min')


def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def demo_sample(i_num, train_records):
    users = [u for u in train_records for _ in train_records[u]]
    pos_items = [pos_i for u in train_records for pos_i in train_records[u]]
    play_num = sum(len(train_records[x]) for x in train_records)
    neg_items = np.random.randint(0, i_num, play_num)

    return torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)


def demo_sample_ph(i_num, ul_num, train_records):
    users = np.array([u for u in train_records for _ in train_records[u]])
    pos_items = np.array([pos_i for u in train_records for pos_i in train_records[u]])
    idx = np.random.choice(len(users) - ul_num, ul_num)
    ul_idx = np.arange(len(users) - ul_num, len(users))
    idx = np.concatenate([idx, ul_idx])
    neg_items = np.random.randint(0, i_num, len(idx))
    return torch.LongTensor(users[idx]), torch.LongTensor(pos_items[idx]), torch.LongTensor(neg_items)


def demo_unlearn_sample(i_num, unlearn_records, train_records):
    users = [u for u in unlearn_records for _ in unlearn_records[u]]
    unlearn_items = [ul_i for u in unlearn_records for ul_i in unlearn_records[u]]
    trained_items = [random.choice(train_records[u]) if train_records[u] else random.randint(0, i_num - 1) for u in
                     users]
    neg_items = np.random.randint(0, i_num, len(users))
    return torch.LongTensor(users), torch.LongTensor(unlearn_items), torch.LongTensor(trained_items), torch.LongTensor(
        neg_items)


def test_model_sim_vs_gt(gt_model, model, train_records, gt_graph=None, graph=None):
    model.eval()
    gt_model.eval()
    max_K = max(TOP_Ks)
    results = collections.defaultdict(list)
    with torch.no_grad():
        users = list(train_records.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        u_batch_size = 200 if hasattr(model, 'sys_params') and 'ncf' in model.sys_params.base else 8192
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(users, batch_size=u_batch_size):
            users_list.append(batch_users)
            allPos = [train_records[u] for u in batch_users]
            # groundTrue = [test_records[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(model.device)
            if graph:
                rating = model.get_user_ratings(batch_users_gpu, graph)
                gt_rating = gt_model.get_user_ratings(batch_users_gpu, gt_graph)
            else:
                rating = model.get_user_ratings(batch_users_gpu)
                gt_rating = gt_model.get_user_ratings(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -float('inf')
            gt_rating[exclude_index, exclude_items] = -float('inf')
            _, rating_K = torch.topk(rating, k=max_K)
            _, gt_rating_K = torch.topk(gt_rating, k=max_K)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(gt_rating_K.cpu())
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        for x in X:
            for i in range(x[0].shape[0]):
                x0, x1 = x[0][i].numpy(), x[1][i].numpy()
                for topk in TOP_Ks:
                    results[topk].append(
                        len(np.intersect1d(x0[:topk], x1[:topk])) / len(np.union1d(x0[:topk], x1[:topk])))
        for i, topk in enumerate(TOP_Ks):
            print(f'Jar_Sim@{topk}: {sum(results[topk]) / len(results[topk]):.4f}; ', end=" ")
        print()
