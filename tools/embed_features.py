# -*- coding: utf-8 -*-
# !@time: 2021/4/2 下午3:24
# !@author: superMC @email: 18758266469@163.com
# !@fileName: embed_features.py
import argparse
import os

from tensorflow import keras

from utils.dataset import create_song_dataset
from utils.base import get_cfg, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_cfg, model_cfg, dataset_cfg, inf_cfg, work_dir, model_dir = get_cfg(args)
    data_path = dataset_cfg.get('data_path')
    embed_dim = model_cfg.get('embed_dim')
    indexes, data_x = create_song_dataset(
        data_path, dataset_cfg, read_part=False, sample_num=1000, test_size=0, embed_dim=embed_dim)
    embed_model_dir = os.path.join(model_dir, 'embed_model')
    embed_model = keras.models.load_model(embed_model_dir)
    data_predict = embed_model.predict(data_x)
    embed_csv = os.path.join(work_dir, inf_cfg.get('embed_csv'))
    write_csv(embed_csv, data_predict, model_cfg.get('label_embed_nums'), indexes=indexes)


if __name__ == '__main__':
    main()
