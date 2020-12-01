from data import load_data
from train import load_model, train


def main(path, name, in_ext, out_ext):
    train_iter,src_field,trg_field=load_data(path, name, in_ext, out_ext)
    model,optimizer,criterion=load_model(src_field,trg_field)
    train(model,train_iter,optimizer,criterion)
    

if __name__ == "__main__":
    main()