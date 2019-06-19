#!/usr/bin/env python
# -*- coding:UTF-8 -*-

"""
用训练好的神经网络模型参数来作曲
"""

import pickle
import numpy as np
from utils import *
from network import *
import logging


logging.basicConfig(level=logging.INFO)

#以之前训练所得的最佳参数来生成音乐
def generate():
    logging.info("生成音乐进行中。。。。")
    #加载用于训练神经网络的音乐数据
    with open('data/notes','rb') as filepath:
        notes=pickle.load(filepath)

    pitch_names=sorted(set(item for item in notes))

    num_pitch=len(set(notes))

    network_input,normalized_input=prepare_sequence(notes,pitch_names,num_pitch)

    model=network_model(normalized_input,num_pitch,"weights-01-4.6931.hdf5")

    prediction=generate_notes(model,network_input,pitch_names,num_pitch)

    create_music(prediction)

def prepare_sequence(notes,pitch_names,num_pitch):
    """
    为神经网络准备号提供的序列
    :param notes:
    :param num_pitch:
    :return:
    """

    sequence_legth=100

    pitch_to_int=dict((pitch,num)for num,pitch in enumerate(pitch_names))

    network_input=[]
    network_output=[]

    for i in range(0,len(notes)-sequence_legth,1):
        sequence_in=notes[i:i+sequence_legth]
        sequence_output=notes[i+sequence_legth]

        network_input.append([pitch_to_int[char] for  char in sequence_in])
        network_output.append([pitch_to_int[sequence_output]])

    n_patterns=len(network_input)
    normalized_input=np.reshape(network_input,(n_patterns,sequence_legth,1))

    #归一化
    normalized_input=normalized_input/float(num_pitch)

    return network_input,normalized_input

def generate_notes(model,network_input,pitch_names,num_pitch):
    """
    基于一序列音符，用神经网络来生成新的音符
    :param model:
    :param network_input:
    :param pitch_names:
    :param num_pitch:
    :return:
    """
    logging.info("generate_notes。。。。")
    start=np.random.randint(0,len(network_input)-1)

    logging.info('start==== %s '% start)

    int_to_pitch=dict((num,pitch)for num,pitch in enumerate(pitch_names))

    pattern=network_input[start]

    prediction_output=[]

    for note_index in range(700):
        prediction_input=np.reshape(pattern,(1,len(pattern),1))

        #输入归一化
        prediction_input=prediction_input/float(num_pitch)

        prediction=model.predict(prediction_input,verbose=0)

        index=np.argmax(prediction)

        result=int_to_pitch[index]

        prediction_output.append(result)

        pattern.append(index)

        pattern=pattern[1:len(pattern)]

    return prediction_output


if __name__ == '__main__':
    generate()
