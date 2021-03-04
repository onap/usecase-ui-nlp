from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import math
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import os




### 全局变量

# GPU编号，默认为-1，表示不使用
CUDA_VISIBLE_DEVICES="2"

# 待训练/预测的问题
questions = ['通信服务名称','最大用户设备数','下行链路数据速率','时延','上行链路数据速率','资源共享类别','移动性','区域']

# 配置文件
FLAGS_bert_config_file = '/home/run/chinese_L-12_H-768_A-12/bert_config.json'
FLAGS_vocab_file = '/home/run/chinese_L-12_H-768_A-12/vocab.txt'
FLAGS_init_checkpoint_squad = '/home/run/chinese_L-12_H-768_A-12/bert_model.ckpt'

max_seq_length = 512


tokenizer_ch = tokenization.FullTokenizer(vocab_file=FLAGS_vocab_file, do_lower_case=True)


