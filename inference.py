# -*- coding: utf-8 -*-
# !@time: 2021/4/1 下午7:16
# !@author: superMC @email: 18758266469@163.com
# !@fileName: inference.py
import argparse
import os
import numpy as np
from tensorflow import keras
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def inference():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    train_cfg = cfg.get('train_cfg')
    model_cfg = cfg.get('model_cfg')
    dataset_cfg = cfg.get('dataset_cfg')
    work_dir = os.path.join('work_dirs', train_cfg.get('work_dir'))
    model_dir = os.path.join(work_dir, 'models', 'model')
    model = keras.models.load_model(model_dir)
    ## 测试
    dense_features = np.random.random((1, 10))
    embedding_features = np.array([[10]])
    raw_features = np.array([[1]])
    predict = model((dense_features, embedding_features, raw_features))
    print(predict)


if __name__ == '__main__':
    inference()
