# -*- coding: utf-8 -*-

# dataloader

import os
from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

def load_data(path, name, in_ext, out_ext):

	"""
	First attempt at creating a working Dataloader that will
	load the data into objects which can be used by torchtext.
	"""

	tokenizer = lambda x: x.split()

	src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
	                       pad_token=PAD_TOKEN, tokenize=tokenizer,
	                       batch_first=True, lower=lowercase,
	                       unk_token=UNK_TOKEN,
	                       include_lengths=True)

	trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
		                   pad_token=PAD_TOKEN, tokenize=tokenizer,
		                   unk_token=UNK_TOKEN,
		                   batch_first=True, lower=lowercase,
		                   include_lengths=True)

	

