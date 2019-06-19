#!/usr/bin/env python
# -*- coding:UTF-8 -*-

"""
midi_utils.py
MIDI相关函数
"""
import os
import subprocess
import pickle
import glob
from music21 import converter, instrument, note, chord, stream
import logging

logging.basicConfig(level=logging.INFO)


def convertMidi2Mp3(input_file):
    logging.info("转换MP3")
    """
    将神经网络生成的midi文件转成MP3文件

    :return:
    """

    output_file = "output.mp3"
    assert os.path.exists(input_file)

    print("Converting %s to MP3" % input_file)

    # 用timidity生成MP3文件
    command = 'timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {}'.format(input_file, output_file)
    subprocess.call(command,shell=True)
    print("Converted. Generated file is %s" %output_file)



def get_notes():
    """
    从music_midi目录中的所有MIDI文件里读取音符（notes）和和弦（chord）
    :return:

    chord是多个note的集合，所以我们简单的把它成为Note
    """

    notes=[]

    #读取midi文件，输出Stream流类型
    for file in glob.glob("midi/*.mid"):
        stream=converter.parse(file)


        #获取所有的乐器部分
        parts=instrument.partitionByInstrument(stream)

        if parts: #如果有乐器部分，取第一个乐器
            notes_to_parse=parts.parts[0].recurse()
        else:
            notes_to_parse=stream.flat.notes #没有乐器部分，纯音符组成

        #打印出每一个元素
        for element in notes_to_parse:
            #如果是Note类型，那么取它的音调（pitch）
            if isinstance(element,note.Note):
                #格式例如：E6
                notes.append(str(element.pitch))
            elif isinstance(element,chord.Chord):
                #转换后的格式例如4.15.7
                notes.append('.'.join(str(n) for n in element.normalOrder))

    if not os.path.exists("data"):
        os.mkdir("data")

    #将数据写入data/notes文件
    with open('data/notes','wb') as filepath:
        pickle.dump(notes,filepath)

    return notes


def create_music(prediction):
    logging.info("prediction %s" % prediction)
    logging.info("开始创建音乐")
    """
    用神经网络预测的音乐数据来生成midi文件，再转成MP3文件
    :param prediction:
    :return:
    """
    offset=0 #偏移
    output_notes=[]

    #生成Note音符或chrod和弦对象
    for data in prediction:
        #chord格式，例如4.15.7
        if ("." in data) or data.isdigit():#isdigit() 方法检测字符串是否只由数字组成。
            notes_in_chord=data.split('.')
            notes=[]
            for current_note in notes_in_chord:
                new_note=note.Note(int(current_note))
                new_note.storedInstrument=instrument.Piano()
                notes.append(new_note)

            new_chord=chord.Chord(notes)
            new_chord.offset=offset
            output_notes.append(new_chord)
        else:
            new_note=note.Note(data)
            new_note.offset=offset
            new_note.storedInstrument=instrument.Piano()
            output_notes.append(new_note)

        #每次迭代都将偏移增加，这样才不会交叉覆盖
        offset+=0.5

    #创建音乐流
    midi_stream=stream.Stream(output_notes)

    #写入MIDI文件
    midi_stream.write('midi',fp="output.mid")

    #将生成的midi文件转换成MP3
    convertMidi2Mp3("output.mid")




#单元测试
if __name__=="main":
    input_file=""
    convertMidi2Mp3(input_file)