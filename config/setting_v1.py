# -*- coding: utf-8 -*-
# !@time: 2021/4/1 上午1:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: setting_v1.py

train_cfg = dict(
    work_dir='setting_v1',
    read_part=False,
    sample_num=500000,
    test_size=0.1,
    lr=0.008,
    batch_size=64,
    epochs=100,
)

model_cfg = dict(
    embed_dim=4,
    dnn_dropout=0.1,
    deep_hidden_units=[16, 32],  # sparse_features
    wide_hidden_units=[16, 32],  # dense_features
    label_embed_nums=64
)

dataset_cfg = dict(
    raw_path='data/data_sample.csv',
    data_path='data/groupby.csv',
    classes=3000,
    index_item='track_uri',
    contact_items=["album_uri", "artist_uri"],
    # 离散特征: 专辑链接（字符串）、艺术家链接（字符串）、曲调(0-12)、音符时值（0-5）、所属歌单（整数连续） /曲目链接（字符串）"track_uri" 不使用 "pid"  不使用"time_signature"
    embedding_feature_items=['key'],
    raw_feature_items=['mode'],
    # 连续特征: 歌曲时长、原声程度(0-1)、律动感(0-1)、冲击感(0-1)、歌唱部分占比(0-1)、现场感(0-1)、响度、重复度(0-1)、朗诵比例(0-1)、分钟节拍数、心理感受(0-1)
    dense_feature_items=["duration_ms_x", "acousticness", "danceability", 'energy', 'instrumentalness', 'liveness',
                         'loudness', 'speechiness', 'tempo', 'valence'],  # continuous
    # 标签 multi-hot
    label_item='pid'
)

inf_cfg = dict(
    embed_csv='embedding.csv',  # work_dir
    buckets_num=5,
    bucket_w=30,
    topk=10,
)
