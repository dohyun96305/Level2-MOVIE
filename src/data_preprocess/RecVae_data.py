# based on https://github.com/dawenl/vae_cf

import os
import sys

import numpy as np
from scipy import sparse
import pandas as pd

def split_train_test_proportion(df, test_prop=0.2):
    df_grouped_by_user = df.groupby('user')
    tr_list, te_list = list(), list()

    for i, (_, group) in enumerate(df_grouped_by_user):
        n_items_u = len(group)

        idx = np.zeros(n_items_u, dtype='bool')
        idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

        tr_list.append(group[np.logical_not(idx)])
        te_list.append(group[idx])


        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    df_tr = pd.concat(tr_list)
    df_te = pd.concat(te_list)

    return df_tr, df_te
    
def numerize(df, dict1, dict2):
    uid = list(map(lambda x: dict1[x], df['user']))
    sid = list(map(lambda x: dict2[x], df['item']))

    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def RecVae_dataloader(args) : 

    train_data = pd.read_csv(os.path.join(args.dataset_dir, 'train_ratings.csv'), header=0)

    return train_data

def RecVae_train_data(args, df) : 

    outputs_dir = args.outputs_dir
    n_heldout_users = args.heldout_users

    min_sc = args.min_users_per_item

    unique_uid = df['user'].unique()

    n_users = unique_uid.size
    tr_users = unique_uid[:(n_users - n_heldout_users)]

    train_plays = df.loc[df['user'].isin(tr_users)]

    unique_sid = pd.unique(train_plays['item'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    with open(os.path.join(outputs_dir, 'unique_sid.csv'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    with open(os.path.join(outputs_dir, 'unique_uid.csv'), 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)

    train_data = numerize(train_plays, profile2id, show2id)

    train_data.to_csv(os.path.join(outputs_dir, 'train.csv'), index=False)

##########################################################################################################3
    
def load_train_data(csv_file, n_items, n_users, global_indexing=False):
    tp = pd.read_csv(csv_file)
    
    n_users = n_users if global_indexing else tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data

def load_tr_te_data(csv_file_tr, csv_file_te, n_items, n_users, global_indexing=False):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    if global_indexing:
        start_idx = 0
        end_idx = len(unique_uid) - 1
    else:
        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

def get_data(args, global_indexing=False):
    unique_sid = list()
    with open(os.path.join(args.outputs_dir, 'unique_sid.csv'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    
    unique_uid = list()
    with open(os.path.join(args.outputs_dir, 'unique_uid.csv'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
            
    n_items = len(unique_sid)
    n_users = len(unique_uid)
    
    train_data = load_train_data(os.path.join(args.outputs_dir, 'train.csv'), n_items, n_users, global_indexing=global_indexing)

    # test_data_tr, test_data_te = load_tr_te_data(os.path.join(dataset, 'test_tr.csv'),
    #                                              os.path.join(dataset, 'test_te.csv'),
    #                                              n_items, n_users, 
    #                                              global_indexing=global_indexing)
    
    data = train_data, # test_data_tr, test_data_te
    data = (x.astype('float32') for x in data)
    
    return data