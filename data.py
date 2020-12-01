# -*- coding: utf-8 -*-

# dataloader

import os
from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field, BucketIterator
from constants import PAD_TOKEN,EOS_TOKEN,UNK_TOKEN,BOS_TOKEN

def load_data(path, in_ext, out_ext):

	"""
	First attempt at creating a working Dataloader that will
	load the data into objects which can be used by torchtext.
	"""

	tokenizer = lambda x: x.split()
	lowercase = True

	src = data.Field(init_token=None, eos_token=EOS_TOKEN,
	                       pad_token=PAD_TOKEN, tokenize=tokenizer,
	                       batch_first=True, lower=lowercase,
	                       unk_token=UNK_TOKEN,
	                       include_lengths=True)

	trg = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
		                   pad_token=PAD_TOKEN, tokenize=tokenizer,
		                   unk_token=UNK_TOKEN,
		                   batch_first=True, lower=lowercase,
		                   include_lengths=True)

	train_data = TranslationDataset(path=path,
                                        exts=("." + in_ext, "." + out_ext),
                                        fields=(src, trg))
	# build the vocabulary
	src.build_vocab(train_data)
	trg.build_vocab(train_data)

	# make iterator for splits
	train_iter = data.BucketIterator(
            repeat=False, sort=False, dataset = train_data,
            batch_size=15, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=True)
	return train_iter,src,trg

