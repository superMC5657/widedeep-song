# -*- coding: utf-8 -*-
# !@time: 2021/4/1 下午7:29
# !@author: superMC @email: 18758266469@163.com
# !@fileName: base.py
import csv
import os
import numpy as np
import pandas as pd

from mmcv import Config


def generate_dir(work_dir):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)


def get_cfg(args):
    cfg = Config.fromfile(args.config)
    train_cfg = cfg.get('train_cfg')
    model_cfg = cfg.get('model_cfg')
    dataset_cfg = cfg.get('dataset_cfg')
    inf_cfg = cfg.get('inf_cfg')
    work_dir = os.path.join('work_dirs', train_cfg.get('work_dir'))
    model_dir = os.path.join(work_dir, 'models')
    return train_cfg, model_cfg, dataset_cfg, inf_cfg, work_dir, model_dir


def read_csv(csv_path, label_embed_nums):
    """
    从csv中拿出face_features 和labels
    """
    name_features_dataframe = pd.read_csv(csv_path, sep=',')
    name_dataframe = name_features_dataframe[['index']]
    features_name = ['Features%d' % i for i in range(label_embed_nums)]
    features_dataframe = name_features_dataframe[features_name]
    indexes = name_dataframe.values
    features = features_dataframe.values
    indexes = np.squeeze(indexes).astype(np.int)
    return indexes, features


def write_csv(csv_path, data, label_embed_nums, indexes):
    print('创建embedding数据库')
    file_csv = open(csv_path, 'w', encoding='UTF-8')
    writer = csv.writer(file_csv)
    header = ['Features%d' % x for x in range(label_embed_nums)]
    header.insert(0, 'index')
    writer.writerow(header)
    writer = csv.writer(file_csv)
    for i in range(data.shape[0]):
        content = np.append(indexes[i], data[i, :])
        writer.writerow(content)
    file_csv.close()
