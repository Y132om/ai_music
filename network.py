#!/usr/bin/env python
# -*- coding:UTF-8 -*-
"""
RNN-LSTM的循环神经网络
"""

import tensorflow as tf
import logging
#神经网络的模型

"""
num_pitch：所用不重复的音调的数目
"""

logging.basicConfig(level=logging.INFO)

def network_model(inputs,num_pitch,weights_file=None):
    logging.info("构建神经网络。。。。")
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(
        512,#LSTM的神经元的数目是512，也就是LSTM的输出维度
        input_shape=(inputs.shape[1],
        inputs.shape[2]),#输入的形状，对第一个LSTM层必须设置
        return_sequences=True #输出所有的输出序列
        #堆叠LSTM的时候必须设置，最后一层LSTM不用设置
    ))
    model.add(tf.keras.layers.Dropout(0.3))#丢弃30%的神经元，防止过拟合
    model.add(tf.keras.layers.LSTM(512,return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512))
    model.add(tf.keras.layers.Dense(256))#256个神经元的全连接层
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(num_pitch)) #表示输出的数目，等于所有不重复的音调数目
    model.add(tf.keras.layers.Activation('softmax')) #用softmax激活函数计算概率



    model.compile(loss="categorical_crossentropy",optimizer="rmsprop")#交叉熵计算误差.循环神经网络来说比较优秀RMSRrop优化器

    if weights_file is not None:#如果生成音乐时
        #从HDF5 文件中加载所有神经网络层的参数（Weight）
        model.load_weights(weights_file)

    return model;


