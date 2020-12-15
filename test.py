#test.py

import os
import torch
from tqdm import tqdm
import pdb
from sklearn.metrics import accuracy_score
import numpy as np

cwd = os.getcwd()

def test(model, test_iter, eos_index=3):

	model.eval()
	num_test_samples = len(test_iter)
	test_scores = np.zeros(num_test_samples)
	sample_idx = 0
	matches = 0
	with torch.no_grad():
		for _, batch in tqdm(enumerate(test_iter)):
			src = batch.src # S+1 X B
			trg = batch.trg # S+1 X B
			output = model(src, trg, train=False) # S1+1 X B X V
			#pdb.set_trace()

			# Find top1 predictions from outputs, and remove the sequence 0
			batch_size = output.shape[1]
			output = output[1:].argmax(2) # S1 X B
			trg = trg[1:] # S X B

			for i in range(batch_size):
				print("###########")
				output_sample = output[:, i]
				trg_sample = trg[:, i]
				trg_seq_len = list(trg_sample).index(eos_index)+1
				trg_sample = trg_sample[:trg_seq_len]
				output_sample = output_sample[:trg_seq_len]
				if list(trg_sample) == list(output_sample):
					matches += 1
					print(trg_sample)
					print(output_sample)
					print(f"Matched: {matches} / {sample_idx}")
					test_scores[sample_idx] = 1
				sample_idx += 1
	accuracy_score = matches / num_test_samples * 100.0
	print(f"Sample_idx: {sample_idx}")
	print(f"Num of test samples: {num_test_samples}")
	print(f"Matches: {matches}")
	print(f"Test Accuracy: {accuracy_score}")