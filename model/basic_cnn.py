# -*- encoding: utf8 -*-
# author: zerg
import sys
import os
import time
import yaml
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf

class ConvNet:

    # network_path :神经网络结构的配置参数存储位置
    # n_channel：输入图像的通道数
    # n_classes：多分类器预分类的数目，默认分10类
    # image_size：图像的大小
    def __init__(self, network_path, n_channel=3, n_classes=10, image_size=24):
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, image_size, image_size, n_channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable(
            0, dtype=tf.int32, name='global_step')

        # os.path.join（）该函数用以链接字符串
        network_option_path = os.path.join(network_path)
        self.network_option = yaml.load(open(network_option_path, 'r'))# 读取网络结构配置文件


