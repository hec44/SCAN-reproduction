from data import load_data
from train import load_model, train
import os
from test import test
import argparse
import torch

parser = argparse.ArgumentParser(description='SCAN reproduction')
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--train_path', default='', type=str, metavar='PATH',
                    help='path to train dataset (e.g. experiment1/train')
parser.add_argument('--test_path', default='', type=str, metavar='PATH',
                    help='path to test dataset')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--hidden_dim', default=200, type=int,
                    metavar='N', help='LSTM Hidden Dimension (default: 200)')
parser.add_argument('--dropout', default=0.5, type=float, metavar='D',
                    help='dropout')
parser.add_argument('--rnn_type', default='lstm', type=str, metavar='LSTM',
                    help='RNN Tpe is lstm/gru (default: LSTM)')
parser.add_argument('--model_dir', default='', type=str, metavar='DIR',
                    help='path to model directory (default: none)')

def main():
    # Define Arguments
    cwd = os.getcwd()
    args = parser.parse_args()
    # EXPERIMENT 1:
    #path_train = os.path.join(cwd, 'data', 'experiment1', 'tasks_train_simple_p1')
    #path_test = os.path.join(cwd, 'data', 'experiment1', 'tasks_test_simple_p1')
    # EXPERIMENT 2:
    path_train = os.path.join(cwd, 'data', 'experiment2', 'tasks_train_length')
    path_test = os.path.join(cwd, 'data', 'experiment2', 'tasks_test_length')
    in_ext = "in"
    out_ext = "out" 
    """
    state = {
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'rnn_type': args.rnn_type
    }
    state = {
        'batch_size': 1,
        'hidden_dim': 200,
        'dropout': 0.0,
        'rnn_type': 'lstm'
    }
    """
    state = {
        'batch_size': 1,
        'hidden_dim': 50,
        'dropout': 0.5,
        'rnn_type': 'gru'
    }

    
    print(f"Run Config State: {state}")

    # Train and Test
    train_iter, test_iter, src, trg = load_data(path_train, path_test, in_ext, out_ext, 'pretrained', batch_size=1)
    model, optimizer, criterion = load_model(src, trg, state)
    model = train(model, train_iter, optimizer, criterion, model_dir='pretrained')
    test(model, test_iter,3)

if __name__ == "__main__":
	main()