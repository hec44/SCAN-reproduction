from data import load_data
from train import load_model, train
import os
from test import test, eval
import argparse
import torch
from constants import EOS_TOKEN

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
parser.add_argument('--eval', default=0, type=int, metavar='BOOL',
                    help='Will Evaluate (default: False)')

def main():
    # Define Arguments
    args = parser.parse_args()
    path_train = os.path.join(args.data_dir, args.train_path)
    path_test = os.path.join(args.data_dir, args.test_path)
    model_dir = args.model_dir
    model_path = os.path.join(args.model_dir, 'model_100000.pt')
    eval = args.eval
    batch_size = args.batch_size

    # EXPERIMENTS:
    #path_train = os.path.join('data', 'experiment1', 'tasks_train_simple')
    #path_test = os.path.join('data', 'experiment1', 'tasks_test_simple')
    #model_dir = 'models/tasks_simple'
    #model_path = os.path.join(model_dir, 'model_100000.pt')
    #eval = False
    #batch_size = 1

    in_ext = "in"
    out_ext = "out"

    state = {
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'rnn_type': args.rnn_type
    }
    """
    state = {
        'hidden_dim': 200,
        'dropout': 0.0,
        'rnn_type': 'lstm'
    }
    """
    print(f"Run Config State, Eval: {state}, {eval}")

    # Train and Test
    train_iter, test_iter, src, trg = load_data(path_train, path_test, in_ext, out_ext, model_dir, batch_size=batch_size)
    model, optimizer, criterion = load_model(src, trg, state)
    if eval == 0:
        print('Training !')
        model = train(model, train_iter, optimizer, criterion, model_dir=model_dir)
    else:
        print('Evaluating !')
        model.load_state_dict(torch.load(model_path))
    test(model, test_iter, eos_index=trg.vocab.stoi[EOS_TOKEN])

if __name__ == "__main__":
	main()