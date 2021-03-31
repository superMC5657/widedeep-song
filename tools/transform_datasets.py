# -*- coding: utf-8 -*-
# !@time: 2021/3/30 下午7:29
# !@author: superMC @email: 18758266469@163.com
# !@fileName: transform_datasets.py


import pandas as pd

'''
由歌单到歌曲的映射->歌曲到歌单的映射
'''

data_df = pd.read_csv('../data/data_sample.csv')
# 保留下来做feature engineer的特征
selected_features = ['']

# index: album_uri
index_f = 'track_uri'
label_f = 'pid'

sparse_feature_items = ["album_uri", "artist_uri", "key", "time_signature", "mode"]
# 连续特征: 歌曲时长、原声程度(0-1)、律动感(0-1)、冲击感(0-1)、歌唱部分占比(0-1)、现场感(0-1)、响度、重复度(0-1)、朗诵比例(0-1)、分钟节拍数、心理感受(0-1)
dense_feature_items = ["duration_ms_x", "acousticness", "danceability", 'energy', 'instrumentalness', 'liveness',
                       'loudness', 'speechiness', 'tempo', 'valence']


# continuous


def function_pid(x):
    strs = ''
    for i, _ in enumerate(x):
        strs += str(_)
        if i != len(x) - 1:
            strs += ','
    return strs


def function_first(x):
    return x[0]


# pid
df_groupBy = data_df.groupby(by=index_f)
series_pid = df_groupBy[label_f].unique()
df_new = pd.DataFrame({'pid': series_pid})
# other
for feature in sparse_feature_items + dense_feature_items:
    series_feature = df_groupBy[feature].unique().apply(function_first)
    df_new[feature] = series_feature
df_new.to_csv('data/groupby.csv')
