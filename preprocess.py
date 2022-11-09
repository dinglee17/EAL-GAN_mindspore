import os
import time
import argparse
import pandas as pd

from dataloader import LoadDocumentData, LoadImageData, LoadTabularData, LoadMatData


class parameter(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # data parameter
        parser.add_argument('--data_format', type=str, default='csv',  
                            help='Dataset format')
        parser.add_argument('--data_name', type=str, default='market',  
                            help='Dataset name')
        parser.add_argument('--data_path', type=str, default='./data/', 
                            help='Data path')
        parser.add_argument('--inject_noise', type=bool, default=False,
                            help='Whether to inject noise to train data')
        parser.add_argument('--cont_rate', type=float, default=0,
                            help='Inject noise to contamination rate')
        parser.add_argument('--anomal_rate', type=str, default='default',
                            help='Adjust anomaly rate')
        parser.add_argument('--seed', type=int, default=42,
                            help="Random seed.")
        parser.add_argument('--verbose', action='store_false', default=True,
                            help='Whether to print training details')

        if __name__ == '__main__':
            args = parser.parse_args()
        else:
            args = parser.parse_args([])

        self.__dict__.update(args.__dict__)


def preprocess(args=None):

    if args == None:
        args = parameter()
    # tabular

    if args.data_format == "mat":
        x_train, y_train, x_val, y_val, x_test, y_test = LoadMatData(args)

    elif args.data_name in ['attack', 'bcsc', 'creditcard', 'diabetic', 'donor', 'intrusion', 'market', 'thyroid']:
        x_train, y_train, x_val, y_val, x_test, y_test = LoadTabularData(args)
    # document
    elif args.data_name in ['20news', 'reuters']:
        dataloader = LoadDocumentData(args)
        # 需要指定anomal_rate
        if args.anomal_rate == 'default':
            args.anomal_rate = 0.05

        # 每一个类作为异常遍历
        for normal_idx in range(dataloader.class_num):
            x_train, x_test, y_train, y_test = dataloader.preprocess(
                normal_idx)
            x_val = x_train.copy()
            y_val = y_train.copy()

    # image
    elif args.data_name in ['mnist']:
        x_train, x_test, y_train, y_test = LoadImageData(args)
        x_val = x_train.copy()
        y_val = y_train.copy()

    return x_train, y_train, x_val, y_val, x_test, y_test
