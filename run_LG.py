import argparse
import os.path

import dgl

from baseline.LASER import LASER, divide_groups, conduct_laser
from baseline.RecEraser import RecEraser, conduct_receraser, divide_shards
from model.LightGCN import LightGCN
from utils_func import *

import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def run_model(train_records, val_records, test_records, unlearn_records, sys_paras, params):
    model = LightGCN(params, sys_paras).to(params['device'])
    sys_paras.base = 'lg'
    if sys_paras.unlearn == 'none' and sys_paras.train:
        start_time = time.time()
        print(f'INFO: A fully-trained model is needed before conducting unlearning! START TRAINING!')
        ul, tst, ul_perc = sys_paras.unlearn, sys_paras.tst_mth, sys_paras.ul_perc
        sys_paras.unlearn, sys_paras.tst_mth, sys_paras.ul_perc = 'none', 'ori', 0
        graph = create_train_graph(train_records, params).to(params['device'])
        train_lg_model(model, graph, train_records, val_records, test_records, sys_paras, params)
        sys_paras.unlearn, sys_paras.tst_mth, sys_paras.ul_perc = ul, tst, ul_perc
        print(f'TRAINING FINISHED! CURRENT TIME: {(time.time() - start_time) / 60:.2f}min')
    if os.path.exists(f'checkpoints/{sys_paras.dataset}/none/ori/{sys_paras.base}-0.pt'):
        print("Load a pre-trained model")
        model.load_state_dict(torch.load(
            f'checkpoints/{sys_paras.dataset}/none/ori/{sys_paras.base}-0.pt',
            map_location=params['device']),
            strict=False)
    graph = create_train_graph(train_records, params).to(params['device'])
    before_ul_auc, before_acc_auc, before_ndcg = test_model_auc(model, test_records, graph), test_model_auc(
        model, val_records, graph), test_model_ndcg(model, train_records, val_records, graph)
    print(f'UL AUC before unlearning: {before_ul_auc:.4f}')
    print(f'ACC AUC before unlearning: {before_acc_auc:.4f}')
    print(f'NDCG@20 before unlearning: {before_ndcg:.4f}')
    if sys_paras.unlearn == 'retrain':
        use_train_records = copy.deepcopy(train_records)
        for user in unlearn_records:
            use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        start_time = time.time()
        graph = create_train_graph(use_train_records, params).to(params['device'])
        model = LightGCN(params, sys_paras).to(params['device'])
        if sys_paras.train:
            print(f"[RETRAIN] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            train_lg_model(model, graph, use_train_records, val_records, test_records, sys_paras, params)
            print(f"[RETRAIN] TRAINING FINISHED. TOTAL TIME: {(time.time() - start_time) / 60:.2f}min.")
    elif sys_paras.unlearn == 'receraser':
        use_train_records = copy.deepcopy(train_records)
        start_time = time.time()
        u_emb = model.get_embeddings(graph)[0]
        for user in unlearn_records:
            use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        model = RecEraser(params, sys_paras).to(params['device'])
        tr_rds, store_dict = divide_shards(use_train_records, u_emb, sys_paras.n_cluster)
        graph = []
        for tr_rd in tr_rds:
            graph.append(create_train_graph(tr_rd, params).to(params['device']))
        model.user_dict = store_dict
        if sys_paras.train:
            print(f"[RecEraser] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            conduct_receraser(model, graph, use_train_records, unlearn_records, val_records, test_records, tr_rds,
                              sys_paras, params, start_time)
            print(f"[RecEraser] TRAINING FINISHED. TOTAL TIME: {(time.time() - start_time) / 60:.2f}min.")
    elif sys_paras.unlearn == 'laser':
        use_train_records = copy.deepcopy(train_records)
        start_time = time.time()
        u_emb = model.get_embeddings(graph)[0]
        for user in unlearn_records:
            use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        model = LASER(params, sys_paras).to(params['device'])
        tr_rds, store_dict = divide_groups(use_train_records, u_emb, sys_paras.n_cluster)
        graph = []
        for tr_rd in tr_rds:
            graph.append(create_train_graph(tr_rd, params).to(params['device']))
        model.user_dict = store_dict

        if sys_paras.train:
            print(f"[LASER] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            conduct_laser(model, graph, use_train_records, unlearn_records, val_records, test_records, tr_rds,
                          sys_paras, params, start_time)
            print(f"[LASER] TRAINING FINISHED. TOTAL TIME: {(time.time() - start_time) / 60:.2f}min.")
    elif 'recul' in sys_paras.unlearn:
        use_train_records = copy.deepcopy(train_records)
        for user in unlearn_records:
            use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        start_time = time.time()
        model.start_recul(unlearn_records, model.user_embedding.weight)
        graph = create_train_graph(use_train_records, params).to(params['device'])

        if sys_paras.train:
            print(f'AUC before unlearning: {test_model_auc(model, val_records, graph):.4f}')
            print(f'NDCG@20 before unlearning: {test_model_ndcg(model, use_train_records, val_records, graph):.4f}')
            print(f"[RECUL] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            conduct_recul_unlearn(model, graph, use_train_records, val_records, test_records, unlearn_records,
                                  sys_paras,
                                  params)
            print(f"[RECUL] TRAINING FINISHED. TOTAL TIME: {(time.time() - start_time) / 60:.2f}min.")
    elif sys_paras.unlearn == 'ifru':
        use_train_records = copy.deepcopy(train_records)
        # for user in unlearn_records:
        #     use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        start_time = time.time()
        graph = create_train_graph(use_train_records, params).to(params['device'])

        if sys_paras.train:
            print(f"[IFRU] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            conduct_ifru_unlearn(model, graph, use_train_records, val_records, test_records, unlearn_records,
                                 sys_paras,
                                 params)
            print(f"[IFRU] TRAINING FINISHED. TOTAL TIME: {(time.time() - start_time) / 60:.2f}min.")

    if os.path.exists(
            f'checkpoints/{sys_paras.dataset}/{sys_paras.unlearn}/{sys_paras.tst_mth}/{sys_paras.base}-{sys_paras.ul_perc}.pt'):
        model.load_state_dict(
            torch.load(
                f'checkpoints/{sys_paras.dataset}/{sys_paras.unlearn}/{sys_paras.tst_mth}/{sys_paras.base}-{sys_paras.ul_perc}.pt'),
            strict=False)
    else:
        print(f'ERROR: No unlearned model!')

    ul_auc = test_model_auc(model, test_records, graph)
    acc_auc = test_model_auc(model, val_records, graph)
    acc_ndcg = test_model_ndcg(model, train_records, val_records, graph, verbose=True)
    print(
        f'Before unlearning: ACC AUC: {before_acc_auc:.4f}, UL AUC: {before_ul_auc:.4f}, ACC NDCG@20: {before_ndcg:.4f}')
    print(f'After unlearning: ACC AUC: {acc_auc:.4f}, UL AUC: {ul_auc:.4f}, ACC NDCG@20: {acc_ndcg:.4f}')

    if sys_paras.test_sim:
        gt_model = LightGCN(params, sys_paras).to(params['device'])
        gt_graph = create_train_graph(train_records, params).to(params['device'])
        if os.path.exists(
                f'checkpoints/{sys_paras.dataset}/retrain/{sys_paras.tst_mth}/{sys_paras.base}-{sys_paras.ul_perc}.pt'):
            print("Load a re-train model")
            gt_model.load_state_dict(torch.load(
                f'checkpoints/{sys_paras.dataset}/retrain/{sys_paras.tst_mth}/{sys_paras.base}-{sys_paras.ul_perc}.pt',
                map_location=params['device']),
                strict=False)
        else:
            raise Exception("Sorry, No retrain model, please train a retrain model first.")
        test_model_sim_vs_gt(gt_model, model, train_records, gt_graph, graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for RecUL')
    sys_paras = parse_args(parser)
    set_random_seed(sys_paras.seed)
    dataset = sys_paras.dataset
    # tst for unlearning, val for accuracy
    trn_rs, val_rs, tst_rs, ul_rs, params = read_dataset(dataset, sys_paras.tst_mth, sys_paras.ul_perc)
    run_model(trn_rs, val_rs, tst_rs, ul_rs, sys_paras, params)
