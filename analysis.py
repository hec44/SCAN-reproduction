import os

data_dir = 'data'
ds_dir = 'experiment3'
in_train_file = 'tasks_train_addprim_turn_left.in'
out_train_file = 'tasks_train_addprim_turn_left.out'
in_test_file = 'tasks_test_addprim_turn_left.in'
out_test_file = 'tasks_test_addprim_turn_left.out'

def count_max_length_sentence(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        max_len = 0
        for line in lines:
            word_len = len(line.split())
            if word_len > max_len:
                max_len = word_len
    print(f"Max length in {file} is: {max_len}")


if __name__ == '__main__':
    count_max_length_sentence(os.path.join(data_dir, ds_dir, in_train_file))
    count_max_length_sentence(os.path.join(data_dir, ds_dir, out_test_file))
