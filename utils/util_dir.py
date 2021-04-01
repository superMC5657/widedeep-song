# -*- coding: utf-8 -*-
# !@time: 2021/4/1 下午7:29
# !@author: superMC @email: 18758266469@163.com
# !@fileName: util_dir.py
import os


def generate_dir(work_dir):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
