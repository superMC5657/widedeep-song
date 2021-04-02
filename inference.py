# -*- coding: utf-8 -*-
# !@time: 2021/4/1 下午7:16
# !@author: superMC @email: 18758266469@163.com
# !@fileName: inference.py
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.base import get_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def inference():
    args = parse_args()
    train_cfg, model_cfg, dataset_cfg, inf_cfg, work_dir, model_dir = get_cfg(args)
    embed_model = keras.models.load_model(os.path.join(model_dir, 'embed_model'))
    ## 测试
    dense_features = np.random.random((1, 10))
    embedding_features = np.array([[10]])
    raw_features = np.array([[1]])
    predict = embed_model((dense_features, embedding_features, raw_features))
    print(tf.reduce_sum(predict))


if __name__ == '__main__':
    inference()
