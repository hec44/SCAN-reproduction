#test.py

import os
import torch
from tqdm import tqdm
import pdb

cwd = os.getcwd()

def test(model, test_iter):

	#model.eval()
	with torch.no_grad():
		for _, batch in tqdm(enumerate(test_iter)):

			src = batch.src
			trg = batch.trg

			pdb.set_trace()

			output = model(src, trg[0])