from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# import collections
# import math
# import modeling
# import optimization
import tokenization
# import six
# import tensorflow as tf
# import os

# Global variables

# GPU number, default: -1, means not used
CUDA_VISIBLE_DEVICES = "2"

# Questions to be trained/predicted
questions = ['Communication Service Name', 'Max Number of UEs', 'Data Rate Downlink', 'Latency', 'Data Rate Uplink', 'Resource Sharing Level', 'Mobility', 'Area']

# Configuration file
FLAGS_bert_config_file = '/home/run/uncased_L-12_H-768_A-12/bert_config.json'
FLAGS_vocab_file = '/home/run/uncased_L-12_H-768_A-12/vocab.txt'
FLAGS_init_checkpoint_squad = '/home/run/uncased_L-12_H-768_A-12/bert_model.ckpt'

max_seq_length = 512


tokenizer_ch = tokenization.FullTokenizer(vocab_file=FLAGS_vocab_file, do_lower_case=True)
