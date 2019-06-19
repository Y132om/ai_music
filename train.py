#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import numpy as np
from utils import *
from network import *

import logging

logging.basicConfig(level=logging.INFO)

"""
术语：

Batch size:批次（样本）数目。一次迭代（Forward运算以及BackPropagation运算）所有的样本数目。Batch size越大，内存越大。 

Iteration:迭代。每一次迭代更新一次权重（网络参数），每一次权重更新需要Batch Size个数据进行Forward运算，在进行BP运算。

Epoch:纪元/时代。所有的训练样本完成一次迭代。

示例：训练集有1000个样本，Batch size=10
那么。训练完整个样本需要 100次Iteration. 1个Epoch

一般训练不值一个Epoch

"""


def train():
    logging.info("训练开始。。。。")
    notes=get_notes()

    num_pitch=len(set(notes)) #得到所有不重复的音调数目

    network_input,network_output=prepare_sequences(notes,num_pitch)

    model=network_model(network_input,num_pitch)

    filepath="weights-{epoch:02d}-{loss:.4f}.hdf5"

    #用checkpoints文件在每一个epoch结束时，保存模型参数Weights
    checkpoint=tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',
        save_best_only=True,
        mode='min'
    )

    callbacks_list=[checkpoint]

    #训练
    model.fit(network_input,network_output,epochs=10,batch_size=64,callbacks=callbacks_list)

def prepare_sequences(notes,num_pitch):
    """
    :param notes: 所有音调
    :param numpitch: 不重复的音调数目
    :return:

    该方法为神经网络准备好序列
    """

    sequence_length=100 #序列长度

    pitch_names=sorted(set(item for item in notes)) #得到所有不重复的名字，并排序

    pitch_to_int=dict((pitch,num) for num,pitch in enumerate(pitch_names)) #创建一个字典 音频，整数

    #创建神经网络输入和输出序列
    network_input=[]
    network_output=[]

    #序列长度为100
    for i in range(0,len(notes)-sequence_length,1):
        sequence_in=notes[i:i+sequence_length]
        sequence_out=notes[i+sequence_length]

        network_input.append([pitch_to_int[char] for char in sequence_in])
        network_output.append(pitch_to_int[sequence_out])

    n_paetterns=len(network_input)

    #将输入的形状转换成神经网络模型可以接受的
    network_input=np.reshape(network_input,(n_paetterns,sequence_length,1))

    #归一化
    network_input=network_input/float(num_pitch)

    network_output=tf.keras.utils.to_categorical(network_output)
    return network_input,network_output

if __name__ == '__main__':
    train()
