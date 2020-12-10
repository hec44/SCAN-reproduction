from data import load_data
from train import load_model, train
import os
from test import test

def main(path_train, path_test, in_ext, out_ext):
    train_iter, test_iter, src, trg = load_data(path_train, path_test, in_ext, out_ext)
    model, optimizer, criterion = load_model(src, trg)
    model = train(model, train_iter, optimizer, criterion)
    test(model, test_iter)

if __name__ == "__main__":
	cwd = os.getcwd()
	main(os.path.join(cwd, "data/experiment1/train"), os.path.join(cwd, "data/experiment1/test"), "in", "out")