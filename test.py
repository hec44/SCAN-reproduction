#test.py

import os
import torch
from tqdm import tqdm
import pdb
from sklearn.metrics import accuracy_score

cwd = os.getcwd()


def test(model, test_iter, eos_index):

	model.eval()
	print(len(test_iter))
	print('EVALUATING !!!')
	correct_count = 0
	with torch.no_grad():
		for _, batch in tqdm(enumerate(test_iter)):

			src = batch.src
			trg = batch.trg

			output = model(src, trg, train=False, eos_index=eos_index)
			true=list(torch.flatten(trg[1:]))
			if output == true:
				correct_count += 1

	pdb.set_trace()
	print("test done")
	print(correct_count/len(test_iter))

def evaluate(model_path, model, test_iter, eos_index):
	model.load_state_dict(torch.load(model_path))
	model.eval()
	print(len(test_iter))
	print('EVALUATING !!!')
	correct_count = 0
	with torch.no_grad():
		for _, batch in tqdm(enumerate(test_iter)):

			src = batch.src
			trg = batch.trg

			output = model(src, trg, train=False, eos_index=eos_index)
			true=list(torch.flatten(trg[1:]))
			if output == true:
				correct_count += 1

	pdb.set_trace()
	print("test done")
	print(correct_count/len(test_iter))