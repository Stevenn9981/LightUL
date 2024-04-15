import argparse
import os.path
import pdb
import sys
import time
from baseline.LASER import LASER, divide_groups, conduct_laser
from baseline.RecEraser import conduct_receraser, RecEraser, divide_shards
from model.MF import BPRMF
from utils_func import *


def run_model(train_records, val_records, test_records, unlearn_records, sys_paras, params):
    trained_model = BPRMF(params, sys_paras).to(params['device'])
    if sys_paras.unlearn == 'none' and sys_paras.train:
        start_time = time.time()
        print(f'INFO: A fully-trained model is needed before conducting unlearning! START TRAINING!')
        ul, tst, ul_perc = sys_paras.unlearn, sys_paras.tst_mth, sys_paras.ul_perc
        sys_paras.unlearn, sys_paras.tst_mth, sys_paras.ul_perc = 'none', 'ori', 0
        train_mf_model(trained_model, train_records, val_records, test_records, sys_paras, params)
        sys_paras.unlearn, sys_paras.tst_mth, sys_paras.ul_perc = ul, tst, ul_perc
        print(f'TRAINING FINISHED! CURRENT TIME: {(time.time() - start_time) / 60:.2f}min')
    if os.path.exists(f'checkpoints/{sys_paras.dataset}/none/ori/{sys_paras.base}-0.pt'):
        print("Load a pre-trained model")
        trained_model.load_state_dict(torch.load(
            f'checkpoints/{sys_paras.dataset}/none/ori/{sys_paras.base}-0.pt',
            map_location=params['device']),
            strict=False)
    else:
        print(f"WARNING: No trained model here! Use a randomly initialized model.")

    before_ul_auc, before_acc_auc, before_ndcg = test_model_auc(trained_model, test_records), test_model_auc(
        trained_model, val_records), test_model_ndcg(trained_model, train_records, val_records)
    print(f' UL AUC before unlearning: {before_ul_auc:.4f}')
    print(f'ACC AUC before unlearning: {before_acc_auc:.4f}')
    print(f'NDCG@20 before unlearning: {before_ndcg:.4f}')
    if sys_paras.unlearn == 'retrain':
        use_train_records = copy.deepcopy(train_records)
        for user in unlearn_records:
            use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        start_time = time.time()
        trained_model = BPRMF(params, sys_paras).to(params['device'])

        if sys_paras.train:
            print(f"[RETRAIN] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            train_mf_model(trained_model, use_train_records, val_records, test_records, sys_paras, params)
            print(f"[RETRAIN] TRAINING FINISHED. TOTAL TIME: {(time.time() - start_time) / 60:.2f}min.")
    elif sys_paras.unlearn == 'receraser':
        use_train_records = copy.deepcopy(train_records)
        for user in unlearn_records:
            use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        start_time = time.time()
        u_emb = trained_model.get_embeddings()[0]
        trained_model = RecEraser(params, sys_paras).to(params['device'])

        if sys_paras.train:
            print(f"[RecEraser] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            tr_rds, store_dict = divide_shards(use_train_records, u_emb, sys_paras.n_cluster)
            trained_model.user_dict = store_dict
            conduct_receraser(trained_model, None, use_train_records, unlearn_records, val_records, test_records,
                              tr_rds,
                              sys_paras, params, start_time)
    elif sys_paras.unlearn == 'laser':
        use_train_records = copy.deepcopy(train_records)
        for user in unlearn_records:
            use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        start_time = time.time()
        u_emb = trained_model.get_embeddings()[0]
        trained_model = LASER(params, sys_paras).to(params['device'])

        if sys_paras.train:
            print(f"[LASER] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            tr_rds, store_dict = divide_groups(use_train_records, u_emb, sys_paras.n_cluster)
            trained_model.user_dict = store_dict
            conduct_laser(trained_model, None, use_train_records, unlearn_records, val_records, test_records, tr_rds,
                          sys_paras, params, start_time)
    elif 'recul' in sys_paras.unlearn:
        use_train_records = copy.deepcopy(train_records)
        for user in unlearn_records:
            use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        start_time = time.time()
        trained_model.start_recul(unlearn_records, trained_model.user_embedding.weight)

        if sys_paras.train:
            print(f"[RECUL] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            conduct_recul_unlearn(trained_model, None, use_train_records, val_records, test_records, unlearn_records,
                                  sys_paras,
                                  params)
            print(f"[RECUL] TRAINING FINISHED. TOTAL TIME: {(time.time() - start_time) / 60:.2f}min.")
    elif sys_paras.unlearn == 'ifru':
        use_train_records = copy.deepcopy(train_records)
        # for user in unlearn_records:
        #     use_train_records[user] = list(set(train_records[user]) - set(unlearn_records[user]))
        start_time = time.time()

        if sys_paras.train:
            print(f"[IFRU] START TRAINING. CURRENT TIME: {(time.time() - start_time) / 60:.2f}min.")
            conduct_ifru_unlearn(trained_model, None, use_train_records, val_records, test_records, unlearn_records,
                                 sys_paras,
                                 params)
            print(f"[IFRU] TRAINING FINISHED. TOTAL TIME: {(time.time() - start_time) / 60:.2f}min.")

    if os.path.exists(
            f'checkpoints/{sys_paras.dataset}/{sys_paras.unlearn}/{sys_paras.tst_mth}/{sys_paras.base}-{sys_paras.ul_perc}.pt'):
        trained_model.load_state_dict(
            torch.load(
                f'checkpoints/{sys_paras.dataset}/{sys_paras.unlearn}/{sys_paras.tst_mth}/{sys_paras.base}-{sys_paras.ul_perc}.pt',
                map_location=params['device']),
            strict=False)
    else:
        print(f'ERROR: No unlearned model!')

    ul_auc = test_model_auc(trained_model, test_records)
    acc_auc = test_model_auc(trained_model, val_records)
    acc_ndcg = test_model_ndcg(trained_model, train_records, val_records, verbose=True)
    print(
        f'Before unlearning: ACC AUC: {before_acc_auc:.4f}, UL AUC: {before_ul_auc:.4f}, ACC NDCG@20: {before_ndcg:.4f}')
    print(f'After unlearning: ACC AUC: {acc_auc:.4f}, UL AUC: {ul_auc:.4f}, ACC NDCG@20: {acc_ndcg:.4f}')

    if sys_paras.test_sim:
        gt_model = BPRMF(params, sys_paras).to(params['device'])
        if os.path.exists(
                f'checkpoints/{sys_paras.dataset}/retrain/{sys_paras.tst_mth}/{sys_paras.base}-{sys_paras.ul_perc}.pt'):
            print("Load a re-train model")
            gt_model.load_state_dict(torch.load(
                f'checkpoints/{sys_paras.dataset}/retrain/{sys_paras.tst_mth}/{sys_paras.base}-{sys_paras.ul_perc}.pt',
                map_location=params['device']),
                strict=False)
        else:
            raise Exception("Sorry, No retrain model, please train a retrain model first.")
        test_model_sim_vs_gt(gt_model, trained_model, train_records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for RecUL')
    sys_paras = parse_args(parser)
    set_random_seed(sys_paras.seed)
    dataset = sys_paras.dataset
    # tst for unlearning, val for accuracy
    trn_rs, val_rs, tst_rs, ul_rs, params = read_dataset(dataset, sys_paras.tst_mth, sys_paras.ul_perc)
    run_model(trn_rs, val_rs, tst_rs, ul_rs, sys_paras, params)
