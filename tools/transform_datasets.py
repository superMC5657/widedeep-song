# -*- coding: utf-8 -*-
# !@time: 2021/3/30 下午7:29
# !@author: superMC @email: 18758266469@163.com
# !@fileName: transform_datasets.py
import argparse

import pandas as pd

from utils.base import get_cfg

'''
由歌单到歌曲的映射->歌曲到歌单的映射
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def function_pid(x):
    strs = ''
    for i, _ in enumerate(x):
        strs += str(_)
        if i != len(x) - 1:
            strs += ','
    return strs


def function_first(x):
    return x[0]


def main():
    args = parse_args()
    train_cfg, model_cfg, dataset_cfg, inf_cfg, work_dir, model_dir = get_cfg(args)
    data_df = pd.read_csv(dataset_cfg.get('raw_path'))
    df_groupBy = data_df.groupby(by=dataset_cfg.get('index_item'))
    series_pid = df_groupBy[dataset_cfg.get('label_item')].unique().apply(function_pid)
    sparse_feature_items = dataset_cfg.get('embedding_feature_items') + dataset_cfg.get(
        'raw_feature_items') + dataset_cfg.get('contact_items')
    # 连续特征: 歌曲时长、原声程度(0-1)、律动感(0-1)、冲击感(0-1)、歌唱部分占比(0-1)、现场感(0-1)、响度、重复度(0-1)、朗诵比例(0-1)、分钟节拍数、心理感受(0-1)
    dense_feature_items = dataset_cfg.get('dense_feature_items')
    df_new = pd.DataFrame({dataset_cfg.get('label_item'): series_pid})
    # other
    for feature in sparse_feature_items + dense_feature_items:
        series_feature = df_groupBy[feature].unique().apply(function_first)
        df_new[feature] = series_feature
    df_new.to_csv(dataset_cfg.get('data_path'))


if __name__ == '__main__':
    main()
