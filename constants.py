# coding: utf-8
"""
Defining global constants
"""

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
CLIP = 5.0

DEFAULT_UNK_ID = lambda: 0

MAX_INPUT_SEQ_LEN = 9
MAX_OUTPUT_SEQ_LEN = 48  #30

MAX_TRAIN_STEPS = 100000