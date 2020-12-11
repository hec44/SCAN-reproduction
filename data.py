# -*- coding: utf-8 -*-

# dataloader

import os
from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field, BucketIterator
from constants import PAD_TOKEN,EOS_TOKEN,UNK_TOKEN,BOS_TOKEN
import dill
import pdb
import torch
cwd = os.getcwd()

def load_data(path_train, path_test, in_ext, out_ext):

	"""
	First attempt at creating a working Dataloader that will
	load the data into objects which can be used by torchtext.
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	tokenizer = lambda x: x.split()
	lowercase = True

	src = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
						   pad_token=PAD_TOKEN, tokenize=tokenizer,
						   batch_first=False, lower=lowercase,
						   unk_token=UNK_TOKEN,
						   include_lengths=False)

	trg = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
						   pad_token=PAD_TOKEN, tokenize=tokenizer,
						   unk_token=UNK_TOKEN,
						   batch_first=False, lower=lowercase,
						   include_lengths=False)

	train_data = TranslationDataset(path=path_train,
										exts=("." + in_ext, "." + out_ext),
										fields=(src, trg))

	test_data = TranslationDataset(path=path_test,
										exts=("." + in_ext, "." + out_ext),
										fields=(src, trg))
	# build the vocabulary
	src.build_vocab(train_data)
	trg.build_vocab(train_data)

	with open(os.path.join(cwd, "data_fields", "src.Field"), "wb") as f:
		dill.dump(src, f)

	with open(os.path.join(cwd, "data_fields", "trg.Field"), "wb") as f:
		dill.dump(trg, f)

	# make iterator for splits
	train_iter = data.BucketIterator(
			repeat=False, sort=False, dataset = train_data,
			batch_size=64, sort_within_batch=True,
			sort_key=lambda x: len(x.src), shuffle=True, device=device)

	# make iterator for splits
	test_iter = data.BucketIterator(
			repeat=False, sort=False, dataset = test_data,
			batch_size=64, sort_within_batch=True,
			sort_key=lambda x: len(x.src), shuffle=True, device=device)

	#pdb.set_trace()

	return train_iter, test_iter, src, trg