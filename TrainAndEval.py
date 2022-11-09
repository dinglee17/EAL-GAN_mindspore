# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from preprocess import preprocess, parameter
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils import load_weights, join_strings
from mindspore import context, Tensor
from dataloader import LoadDocumentData, LoadImageData, LoadTabularData
from configs import parse_args
from src.CB_GAN import CB_GAN
from src.pyod_utils import precision_recall_n_scores, gmean_scores
from src.pyod_utils import standardizer, AUC_and_Gmean
from mindspore import dtype as mstype
import mindspore as ms
import mindspore
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings


import os
import sys
import argparse
from time import time
import os


# supress warnings for clean output
warnings.filterwarnings("ignore")


# Define data file and result file
folder_name = 'data'
save_dir = 'results'

# define the number of iterations
n_ite = 1
n_classifiers = 1

df_columns = ['Data', '# Samples', '# Dimensions', 'Outlier Perc',
              'CB-GAN']

# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
rec_df = pd.DataFrame(columns=df_columns)
gmean_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

args = parse_args()

#context.set_context(mode=context.PYNATIVE_MODE,pynative_synchronize=True)
context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target=args.device, device_id=args.device_id)
dataset_args = parameter()

if args.data_path[-1]!='/':
    args.data_path=args.data_path+'/'
dataset_args.data_name = args.data_name
dataset_args.data_path = args.data_path
dataset_args.data_format = "mat"

X_train, y_train, X_val, y_val, X_test, y_test = preprocess(dataset_args)



X_train.astype(np.float32)
X_test.astype(np.float32)

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=None)

y = y.astype(np.int32)

outliers_fraction = np.count_nonzero(y) / len(y)  # 异常点所占的比例
outliers_percentage = round(outliers_fraction * 100, ndigits=4)

# construct containers for saving results
roc_list = [dataset_args.data_name, X.shape[0],
            X.shape[1], outliers_percentage]
prn_list = [dataset_args.data_name, X.shape[0],
            X.shape[1], outliers_percentage]
rec_list = [dataset_args.data_name, X.shape[0],
            X.shape[1], outliers_percentage]
gmean_list = [dataset_args.data_name,
              X.shape[0], X.shape[1], outliers_percentage]
time_list = [dataset_args.data_name,
             X.shape[0], X.shape[1], outliers_percentage]

X_train_norm, X_test_norm = standardizer(X_train, X_test)
X_train_norm_not_used, X_val_norm = standardizer(X_train, X_val)

X_train_pandas = pd.DataFrame(X_train_norm)
X_test_pandas = pd.DataFrame(X_test_norm)
X_val_pandas = pd.DataFrame(X_val_norm)
X_train_pandas.fillna(X_train_pandas.mean(), inplace=True)
X_test_pandas.fillna(X_train_pandas.mean(), inplace=True)
X_val_pandas.fillna(X_val_pandas.mean(), inplace=True)
X_train_norm = X_train_pandas.values
X_test_norm = X_test_pandas.values
X_val_norm = X_val_pandas.values

roc_mat = np.zeros([n_ite, n_classifiers])
pr_mat = np.zeros([n_ite, n_classifiers])
prn_mat = np.zeros([n_ite, n_classifiers])
rec_mat = np.zeros([n_ite, n_classifiers])
gmean_mat = np.zeros([n_ite, n_classifiers])
time_mat = np.zeros([n_ite, n_classifiers])

result_train = pd.DataFrame([])
result_test = pd.DataFrame([])
result_val = pd.DataFrame([])

for i in range(n_ite):
    print("\n... Processing", dataset_args.data_name, '...', 'Iteration', i + 1)


    X_train_norm = X_train_norm.astype(np.float32)
    X_test_norm = X_test_norm.astype(np.float32)
    X_val_norm = X_val_norm.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    y_val = y_val.astype(np.int32)
    data_x = X_train_norm
    data_y = y_train
    X_test = X_test_norm
    X_val = X_val_norm


    t0 = time()
    cb_gan = CB_GAN(args, data_x, data_y, X_test, y_test, X_val, y_val)
    if args.resume == True:
        print("reload trained parameters")

        cb_gan.netG, cb_gan.NetD_Ensemble = load_weights(
            cb_gan.netG, cb_gan.NetD_Ensemble, weights_root="./weights_"+dataset_args.data_name, epoch=args.resume_batch)
    else:
        start_time = time()
        cb_gan.fit()
        print(
            f'Time cost in training is {(time() - start_time):.2f} seconds\n')

    test_scores = cb_gan.predict(Tensor(X_test.astype(np.float32)), cb_gan.NetD_Ensemble)
    train_scores = cb_gan.predict(Tensor(data_x.astype(np.float32)), cb_gan.NetD_Ensemble)
    val_scores = cb_gan.predict(Tensor(X_val.astype(np.float32)), cb_gan.NetD_Ensemble)
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    test_scores=test_scores.asnumpy()
    train_scores=train_scores.asnumpy()
    val_scores=val_scores.asnumpy()
    roc, prn, rec, gmean = AUC_and_Gmean(y_test, test_scores)
    pr = average_precision_score(y_test, test_scores)
    roc_t, prn_t, rec_t, gmean_t = AUC_and_Gmean(y_train, train_scores)
    roc_v, prn_v, rec_v, gmean_v = AUC_and_Gmean(y_val, val_scores)

    test_label = 'golden_label'
    test_name = 'EALGAN_score'

    result_train_path = "./result_train_"+dataset_args.data_name+".csv"
    result_test_path = "./result_test_"+dataset_args.data_name+".csv"
    result_val_path = "./result_val_"+dataset_args.data_name+".csv"

    if i == 0:
        result_train[test_name] = train_scores
        result_train[test_label] = y_train
    else:
        result_train[test_name] += train_scores
        result_train[test_label] += y_train
    if i == n_ite - 1:
        result_train[test_name] /= n_ite
        result_train[test_label] /= n_ite

    if i == 0:
        result_test[test_name] = test_scores
        result_test[test_label] = y_test
    else:
        result_test[test_name] += test_scores
        result_test[test_label] += y_test
    if i == n_ite - 1:
        result_test[test_name] /= n_ite
        result_test[test_label] /= n_ite

    if i == 0:
        result_val[test_name] = val_scores
        result_val[test_label] = y_val
    else:
        result_val[test_name] += val_scores
        result_val[test_label] += y_val
    if i == n_ite - 1:
        result_val[test_name] /= n_ite
        result_val[test_label] /= n_ite

    print('AUC:{roc}, precision @ rank n:{prn}, recall @ rank n:{rec}, Gmean:{gmean}, train_AUC:{train_auc}, train_precision:{train_prn}, train_recall:{train_rec}, train_gmean:{train_gmean}, val_AUC:{val_auc}, val_precision:{val_prn}, val_recall:{val_rec}, val_gmean:{val_gmean},  execution time: {duration}s'.format(
        roc=roc, prn=prn, rec=rec, gmean=gmean, train_auc=roc_t, train_prn=prn_t, train_rec=rec_t, train_gmean=gmean_t, val_auc=roc_v, val_prn=prn_v, val_rec=rec_v, val_gmean=gmean_v, duration=duration))

    time_mat[i, 0] = duration
    roc_mat[i, 0] = roc
    pr_mat[i, 0] = pr
    prn_mat[i, 0] = prn
    rec_mat[i, 0] = rec
    gmean_mat[i, 0] = gmean

print("Average: ")
print('auc: ', np.mean(roc_mat, axis=0))
print('gmean: ', np.mean(gmean_mat, axis=0))
print('pr: ', np.mean(pr_mat, axis=0))
print('prn: ', np.mean(prn_mat, axis=0))
print('rec: ', np.mean(rec_mat, axis=0))


time_list = time_list + np.mean(time_mat, axis=0).tolist()
temp_df = pd.DataFrame(time_list).transpose()
temp_df.columns = df_columns
time_df = pd.concat([time_df, temp_df], axis=0)

roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
temp_df = pd.DataFrame(roc_list).transpose()
temp_df.columns = df_columns
roc_df = pd.concat([roc_df, temp_df], axis=0)

prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
temp_df = pd.DataFrame(prn_list).transpose()
temp_df.columns = df_columns
prn_df = pd.concat([prn_df, temp_df], axis=0)

rec_list = rec_list + np.mean(rec_mat, axis=0).tolist()
temp_df = pd.DataFrame(rec_list).transpose()
temp_df.columns = df_columns
rec_df = pd.concat([rec_df, temp_df], axis=0)

gmean_list = gmean_list + np.mean(gmean_mat, axis=0).tolist()
temp_df = pd.DataFrame(gmean_list).transpose()
temp_df.columns = df_columns
gmean_df = pd.concat([gmean_df, temp_df], axis=0)

# Save the results for each run
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if args.resume == True:
    save_path1 = os.path.join(save_dir, 'AUC_CB_GAN_' + folder_name + ".csv")
    save_path2 = os.path.join(save_dir, 'Gmean_CB_GAN_' + folder_name + ".csv")
    
    if os.path.exists(save_path1):
        AUC_result=pd.read_csv(save_path1)
    else:
        AUC_result = pd.DataFrame(
            columns=['Data', '# Samples', '# Dimensions', 'Outlier Perc', 'CB-GAN'])
    AUC_result = AUC_result.append(roc_df, ignore_index=True).drop_duplicates(subset=['Data'],keep="last",ignore_index=True)
    
    if os.path.exists(save_path1):
        Gmean_result=pd.read_csv(save_path2)
    else:
        Gmean_result = pd.DataFrame(
            columns=['Data', '# Samples', '# Dimensions', 'Outlier Perc', 'CB-GAN'])
    Gmean_result = Gmean_result.append(gmean_df, ignore_index=True).drop_duplicates(subset=['Data'],keep="last",ignore_index=True)
    
    AUC_result.to_csv(save_path1, index=False, float_format='%.3f')
    Gmean_result.to_csv(save_path2, index=False, float_format='%.3f')

