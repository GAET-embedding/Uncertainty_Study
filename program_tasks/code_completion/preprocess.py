import os
import javalang
from tqdm import tqdm
import torch
from program_tasks.code_completion.util import create_tsv_file
from collections import defaultdict
import tokenize

def parse_java(src_folder, dest_dir, dest_file_name, token_dict):
    token_dict[dest_file_name] = defaultdict(int)

    with open(os.path.join(dest_dir, dest_file_name), 'w') as write_file:
        for f in os.listdir(src_folder):
            subfolder = os.path.join(src_folder, f)
            if os.path.isdir(subfolder): # train/project_name/java
                print('tokenizing java code in {} ...'.format(subfolder))
                for file_path in tqdm(os.listdir(subfolder)):
                    if file_path.endswith(".java"):
                        file = open(os.path.join(subfolder, file_path), 'r')
                        file_string = ' '.join(file.read().splitlines()) # read in oneline
                        tokens = list(javalang.tokenizer.tokenize(file_string)) # ast java parse
                        # print('token list: \n', [x.value for x in tokens])
                        for tok in tokens:
                            token_dict[dest_file_name][tok.value] += 1
                        
                        token_str = " ".join([x.value for x in tokens])
                        # write new line each time, unicode escape
                        write_file.write(
                            token_str.encode('unicode_escape').decode('utf-8') + '\n')
            else: # project_name/java
                if subfolder.endswith(".java"):
                    file = open(os.path.join(subfolder, file_path), 'r')
                    file_string = ' '.join(file.read().splitlines()) # read in oneline
                    tokens = list(javalang.tokenizer.tokenize(file_string)) # ast java parse
                    for tok in tokens:
                        token_dict[dest_file_name][tok.value] += 1

                    token_str = " ".join([x.value for x in tokens])
                    # write new line each time, unicode escape
                    write_file.write(
                        token_str.encode('unicode_escape').decode('utf-8') + '\n')

        write_file.close()



def parse_python(src_folder, dest_dir, dest_file_name, token_dict):
    token_dict[dest_file_name] = defaultdict(int)

    with open(os.path.join(dest_dir, dest_file_name), 'w') as write_file:
        for f in os.listdir(src_folder):
            subfolder = os.path.join(src_folder, f)
            if os.path.isdir(subfolder): 
                print('tokenizing java code in {} ...'.format(subfolder))
                for file_path in tqdm(os.listdir(subfolder)):
                    if file_path.endswith(".py"):
                        with open(os.path.join(subfolder, file_path), 'rb') as f:
                            tokens = tokenize.tokenize(f.readline)
                            # print('token list: \n', [x.value for x in tokens])
                            token_str = " ".join([tok.string for tok in tokens])
                            # write new line each time, unicode escape
                            write_file.write(
                                token_str.encode('unicode_escape').decode('utf-8') + '\n'
                            )
            else: # project_name/java
                if subfolder.endswith(".py"):
                    with open(subfolder, 'rb') as f:
                        tokens = tokenize.tokenize(f.readline)
                        # print('token list: \n', [x.value for x in tokens])
                        token_str = " ".join([x.string for x in tokens])
                        # write new line each time, unicode escape
                        write_file.write(
                            token_str.encode('unicode_escape').decode('utf-8') + '\n'
                        )
            
        write_file.close()



if __name__ == '__main__':
    ###############################################################################
    # # handle java files
    # data_dir = 'java_data/different_project/data'
    # data_type = ['train', 'val', 'test1', 'test2', 'test3']
    # java_dict = {
    #     k + '.txt': os.path.join(data_dir, k) # 'train': data_dir/train/
    #     for k in data_type
    # }

    # dest_dir = "program_tasks/code_completion/dataset/different_time/java_project"
    # token_dict = {} # save token hist in dict
    # if not os.path.exists(dest_dir):
    #     os.makedirs(dest_dir)

    # for name, src in java_dict.items():
    #     parse_java(src, dest_dir, name, token_dict)

    # for name in java_dict:
    #     origin_file = os.path.join(dest_dir, name)
    #     dest_file = origin_file.rstrip('.txt') + '.tsv'
    #     create_tsv_file(origin_file, dest_file)

    # # save token dict
    # torch.save(token_dict, os.path.join(dest_dir, 'token_hist.res'))
    ###############################################################################
    ###############################################################################
    # handle python files
    data_dir = 'python_data'
    data_type = ['train', 'val', 'test1', 'test2', 'test3']
    java_dict = {
        k + '.txt': os.path.join(data_dir, k) # 'train': data_dir/train/
        for k in data_type
    }

    dest_dir = "program_tasks/code_completion/dataset/python_project"
    token_dict = {} # save token hist in dict
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for name, src in java_dict.items():
        parse_python(src, dest_dir, name, token_dict)

    for name in java_dict:
        origin_file = os.path.join(dest_dir, name)
        dest_file = origin_file.rstrip('.txt') + '.tsv'
        create_tsv_file(origin_file, dest_file)

    # save token dict
    torch.save(token_dict, os.path.join(dest_dir, 'token_hist.res'))
    ###############################################################################