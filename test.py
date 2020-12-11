#test.py

import os
import torch
from tqdm import tqdm
import pdb
from sklearn.metrics import accuracy_score

cwd = os.getcwd()

def test(model, test_iter):

	model.eval()
	y_true=[]
	y_pred=[]
	with torch.no_grad():
		for _, batch in tqdm(enumerate(test_iter)):

			src = batch.src
			trg = batch.trg

			output = model(src, trg, train=False)
			#pdb.set_trace()
			y_true=y_true+list(torch.flatten(trg[1:]))
			y_pred=y_pred+list(torch.flatten(output.argmax(2)[1:]))
			#y_true=y_true+list(torch.flatten(trg[0][:,1:]))
			#y_pred=y_pred+list(torch.flatten(output.argmax(2)[:,1:]))
			#pdb.set_trace()
	trues = [tens.item() for tens in y_true]
	preds = [tens.item() for tens in y_pred]
	print("test done")
	print(accuracy_score(trues, preds))