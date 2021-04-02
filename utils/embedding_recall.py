# -*- coding: utf-8 -*-
# !@time: 2021/4/1 下午9:57
# !@author: superMC @email: 18758266469@163.com
# !@fileName: embedding_recall.py

import numpy as np
from tqdm import tqdm

from utils.base import read_csv


def split_buckets(indexes, df_embedding, buckets_num=5, label_embed_nums=64, bucket_w=30):
    buckets = []
    xs = []
    bucket_ls = []
    for i in range(buckets_num):
        bucket = []
        x = np.random.rand(label_embed_nums, 1)
        xs.append(x)
        h = np.matmul(df_embedding, x).squeeze(axis=-1)
        bucket_l = (h.max() - h.min()) / bucket_w
        h_min = h.min()
        bucket_ls.append((h_min, bucket_l))

        for j in tqdm(range(bucket_w)):
            sub_bucket = []
            for k in range(h.shape[0]):
                if h_min + j * bucket_l < h[k] <= h_min + (j + 1) * bucket_l:
                    sub_bucket.append(indexes[k])
            bucket.append(sub_bucket)
        buckets.append(bucket)
    return buckets, xs, bucket_ls


def get_lsh_index(embedding, buckets, xs, bucket_ls):
    ret_index = set()
    bucket_w = len(buckets[0])
    for i, x in enumerate(xs):
        bucket = buckets[i]
        h_e = np.matmul(embedding, x).squeeze(axis=-1)
        h_min, bucket_l = bucket_ls[i]
        for j in range(bucket_w):
            sub_bucket = bucket[j]
            if h_min + j * bucket_l < h_e <= h_min + (j + 1) * bucket_l:
                ret_index += set(sub_bucket)
    return ret_index


if __name__ == '__main__':
    indexes, features = read_csv(csv_path='work_dirs/setting_v1/embedding.csv', label_embed_nums=64)
    embedding = np.random.rand(1, 64)
    buckets, xs, bucket_ls = split_buckets(indexes, features, label_embed_nums=64)
    index = get_lsh_index(embedding, buckets, xs, bucket_ls)
    print(index)
