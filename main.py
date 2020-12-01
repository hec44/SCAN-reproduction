from data import load_data
from train import load_model, train
import os


def main(path, in_ext, out_ext):
    train_iter,src,trg = load_data(path, in_ext, out_ext)
    model,optimizer,criterion=load_model(src,trg)
    train(model, train_iter, optimizer, criterion)
    

if __name__ == "__main__":
	cwd = os.getcwd()
	main(os.path.join(cwd, "data\\experiment1\\test"), "in", "out")