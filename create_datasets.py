import collections
import os.path
import random


def crt_ml_data():
    train_records = collections.defaultdict(list)
    test_records = collections.defaultdict(list)
    dataset = 'ml-1M'

    file_m = open(f'dataset/{dataset}/movies.dat', 'r', encoding='utf-8')
    mid = 0
    uid = 0
    item_dict = dict()
    train_item_set = set()
    for line in file_m.readlines():
        ele = line.strip().split('::')
        item_dict[ele[0]] = mid
        mid += 1
    file_m.close()
    file = open(f'dataset/{dataset}/ratings.train', 'r')
    for line in file.readlines():
        ele = line.strip().split('::')
        train_records[int(ele[0]) - 1].append(item_dict[ele[1]])
        train_item_set.add(item_dict[ele[1]])
        if int(ele[0]) > uid:
            uid = int(ele[0])
    file.close()
    file = open(f'dataset/{dataset}/ratings.test', 'r')
    for line in file.readlines():
        ele = line.strip().split('::')
        # make sure all the users and items are appeared in training dataset.
        if int(ele[0]) - 1 in train_records and item_dict[ele[1]] in train_item_set:
            test_records[int(ele[0]) - 1].append(item_dict[ele[1]])
    file.close()

    train_records = dict(sorted(train_records.items()))
    test_records = dict(sorted(test_records.items()))

    file = open(f'dataset/{dataset}/train.txt', 'w')
    for user in train_records:
        file.write(str(user) + ' ')
        for item in train_records[user]:
            file.write(str(item) + ' ')
        file.write('\n')
    file.close()
    file = open(f'dataset/{dataset}/test.txt', 'w')
    for user in test_records:
        file.write(str(user) + ' ')
        for item in test_records[user]:
            file.write(str(item) + ' ')
        file.write('\n')
    file.close()


def crt_ul_test(dataset, percentage=0.01):
    u_num = 0
    i_num = 0
    train_records_intr = []
    train_records_user = collections.defaultdict(list)
    train_records_item = collections.defaultdict(list)
    file = open(f'dataset/{dataset}/train.txt', 'r')
    for line in file.readlines():
        ele = line.strip().split(' ')
        user, items = ele[0], ele[1:]
        u_num = max(u_num, int(user))
        for item in items:
            i_num = max(i_num, int(item))
            train_records_intr.append((user, item))
            train_records_user[user].append(item)
            train_records_item[item].append(user)
    if not os.path.exists(f'dataset/{dataset}/unlearn_set'):
        os.mkdir(f'dataset/{dataset}/unlearn_set')
    if not os.path.exists(f'dataset/{dataset}/unlearn_test'):
        os.mkdir(f'dataset/{dataset}/unlearn_test')
    split_way = ['user', 'item', 'intr']
    smp_num = int(len(train_records_intr) * percentage)
    for sw in split_way:
        unlearn_records = collections.defaultdict(list)
        cnt = 0
        if sw == 'intr':
            random.shuffle(train_records_intr)
            for intr in train_records_intr:
                unlearn_records[intr[0]].append(intr[1])
                cnt += 1
                if cnt >= smp_num:
                    break
        elif sw == 'user':
            users = list(train_records_user.keys())
            random.shuffle(users)
            for user in users:
                cnt += len(train_records_user[user])
                unlearn_records[user].extend(train_records_user[user])
                if cnt >= smp_num:
                    break
        else:
            items = list(train_records_item.keys())
            random.shuffle(items)
            for item in items:
                cnt += len(train_records_item[item])
                for user in train_records_item[item]:
                    unlearn_records[user].append(item)
                if cnt >= smp_num:
                    break
        unlearn_records = dict(sorted(unlearn_records.items(), key=lambda x: int(x[0])))

        remain_records = collections.defaultdict(list)
        random.shuffle(train_records_intr)
        tst_cnt = 0
        for idx, intr in enumerate(train_records_intr):
            if intr[0] not in unlearn_records or intr[1] not in unlearn_records[intr[0]]:
                remain_records[intr[0]].append(intr[1])
                tst_cnt += 1
            if tst_cnt >= cnt:
                break

        file = open(f'dataset/{dataset}/unlearn_set/{sw}_{percentage}.txt', 'w')
        for user in unlearn_records:
            file.write(str(user) + ' ')
            for item in unlearn_records[user]:
                file.write(str(item) + ' ')
            file.write('\n')
        file.close()
        file = open(f'dataset/{dataset}/unlearn_test/{sw}_{percentage}.txt', 'w')
        for user in unlearn_records:
            for item in unlearn_records[user]:
                file.write(f'{user}\t{item}\t0\n')
        for user in remain_records:
            for item in remain_records[user]:
                file.write(f'{user}\t{item}\t1\n')
        file.close()

        print(f'{dataset}_{sw}_{percentage} done! #unlearn: {cnt}, #remain: {tst_cnt}')


def crt_ori_test(dataset):
    u_num = 0
    i_num = 0
    train_records = []
    test_records = []
    smp_records = []
    file = open(f'dataset/{dataset}/train.txt', 'r')
    for line in file.readlines():
        ele = line.strip().split(' ')
        user, items = ele[0], ele[1:]
        u_num = max(u_num, int(user))
        for item in items:
            i_num = max(i_num, int(item))
            train_records.append((user, item))
    file.close()

    file = open(f'dataset/{dataset}/test.txt', 'r')
    for line in file.readlines():
        ele = line.strip().split(' ')
        user, items = ele[0], ele[1:]
        u_num = max(u_num, int(user))
        for item in items:
            i_num = max(i_num, int(item))
            test_records.append((user, item))
    file.close()

    user_list = list(range(u_num))
    item_list = list(range(i_num))
    tr_set = set(train_records)
    te_set = set(test_records)
    while len(smp_records) < len(test_records):
        s_u, s_i = random.choice(user_list), random.choice(item_list)
        if (s_u, s_i) not in tr_set and (s_u, s_i) not in te_set:
            smp_records.append((s_u, s_i))

    file = open(f'dataset/{dataset}/test_auc.txt', 'w')
    for s_u, s_i in test_records:
        file.write(f'{s_u}\t{s_i}\t1\n')
    for s_u, s_i in smp_records:
        file.write(f'{s_u}\t{s_i}\t0\n')
    file.close()
    print(f'{dataset} done!')


if __name__ == '__main__':
    datasets = ['ml-1M', 'gowalla', 'yelp2018']
    pers = [0.005, 0.01, 0.02]

    for ds in datasets:
        crt_ori_test(ds)
        # for per in pers:
        #     crt_ul_test(ds, per)
