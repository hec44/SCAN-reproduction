# -*- coding: utf-8 -*-

# dataloader

import os
from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field
from constants import PAD_TOKEN,EOS_TOKEN,UNK_TOKEN,BOS_TOKEN

def load_data(path, name, in_ext, out_ext):

	"""
	First attempt at creating a working Dataloader that will
	load the data into objects which can be used by torchtext.
	"""

	tokenizer = lambda x: x.split()
	lowercase = True

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

	train_data = TranslationDataset(path=path,
                                        exts=("." + in_ext, "." + out_ext),
                                        fields=(src_field, trg_field))
	# build the vocabulary
	src_field.build_vocab(train_data)
	trg_field.build_vocab(train_data)

	# make iterator for splits
	train_iter = data.BucketIterator.splits(
            repeat=False, sort=False,
            batch_size=15, 
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
	return train_iter,src_field,trg_field

