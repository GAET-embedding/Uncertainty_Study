import numpy as np
import torch
import random
import javalang


BASEDICT = {
    '____UNKNOW____': 0,
    '____PAD____': 1,
    '____ST____': 2,
    '____ED____': 3
}


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_statement(code):
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
        tokens = [tk.value for tk in tokens]
        code = " ".join(tokens)
    except:
        pass
    return code

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_source(source_code):
    for i, code in enumerate(source_code):
        code = code.strip().lower()
        code = parse_statement(code)
        source_code[i] = code
    return source_code


def trans_vocab(vocab, vectors):
    new_vocab = BASEDICT.copy()
    for tk in vocab:
        new_vocab[tk] = vocab[tk].index + len(BASEDICT)
    dim = vectors.shape[1]
    tmp_vec = np.random.rand(len(BASEDICT), dim)
    vec = np.concatenate([tmp_vec, vectors])
    return new_vocab, vec


def collect_common_tk(word2index_list):
    def appear_all(k):
        for tar_dict in word2index_list:
            if k not in tar_dict:
                return False
        return True

    common_tk = []
    word2index = word2index_list[0]
    for tk in word2index:
        if tk in BASEDICT:
            continue
        if appear_all(tk):
            common_tk.append(tk)
    return common_tk

#
# def configure_exp(embed_type, embed_path):
#     train_path = args.train_data
#     test_path = args.test_data
#     pre_embedding_path = args.embedding_path
#     if args.embedding_type == 0:
#         d_word_index, embed = torch.load(pre_embedding_path)
#         print('load existing embedding vectors, name is ', pre_embedding_path)
#     elif args.embedding_type == 1:
#         v_builder = VocabBuilder(path_file=train_path)
#         d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
#         print('create new embedding vectors, training from scratch')
#     elif args.embedding_type == 2:
#         v_builder = VocabBuilder(path_file=train_path)
#         d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
#         embed = torch.randn([len(d_word_index), args.embedding_dim]).cuda()
#         print('create new embedding vectors, training the random vectors')
#     else:
#         raise ValueError('unsupported type')
#     if embed is not None:
#         if type(embed) is np.ndarray:
#             embed = torch.tensor(embed, dtype=torch.float).cuda()
#         assert embed.size()[1] == args.embedding_dim