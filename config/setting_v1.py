# -*- coding: utf-8 -*-
# !@time: 2021/4/1 上午1:03
# !@author: superMC @email: 18758266469@163.com
# !@fileName: setting_v1.py

train_cfg = dict(
    data_path='data/groupby.csv',
    work_dir='setting_v1',
    read_part=False,
    sample_num=500000,
    test_size=0.1,
    lr=0.008,
    batch_size=64,
    epochs=5,
)

model_cfg = dict(
    embed_dim=4,
    dnn_dropout=0.1,
    deep_hidden_units=[16, 32],  # sparse_features
    wide_hidden_units=[16, 32],  # dense_features
    label_embed_nums=64
)

dataset_cfg = dict(
    classes=3000,
)
