from __future__ import print_function
import torch
import pandas as pd
import numpy as np
import random
import program_tasks.code_completion.util as ut
from torch.utils.data import DataLoader


class TextClassDataLoader(object):
    def __init__(self, path_file, word_to_index, batch_size=32):
        """
        Args:
            path_file:
            word_to_index:
            batch_size:
        """

        self.batch_size = batch_size
        self.word_to_index = word_to_index

        # read file
        df = pd.read_csv(path_file, delimiter='\t')
        df['body'] = df['body'].apply(ut._tokenize)
        self.old_samples = df['body'].copy()
        df['body'] = df['body'].apply(self.generate_indexifyer())
        self.samples = df.values.tolist()

        # for batch
        self.n_samples = len(self.samples)
        self.n_batches = int(self.n_samples / self.batch_size)
        self.max_length = self._get_max_length()
        self._create_indices()

        # self.report()

    # def _shuffle_indices(self):
    #     self.indices = np.random.permutation(self.n_samples)
    #     self.index = 0
    #     self.batch_index = 0
    def _create_indices(self):
        self.indices = np.arange(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _get_max_length(self):
        length = 0
        for sample in self.samples:
            length = max(length, len(sample[1]))
        return length

    def generate_indexifyer(self):

        def indexify(lst_text):
            indices = []
            for word in lst_text:
                if word in self.word_to_index:
                    indices.append(self.word_to_index[word])
                else:
                    indices.append(self.word_to_index['____UNKNOW____'])
            return indices

        return indexify

    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] = batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor([len(s) for s in string])

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        # SORT YOUR TENSORS BY LENGTH!
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        label = torch.LongTensor(label)
        label = label[perm_idx]

        return seq_tensor, label, seq_lengths

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._create_batch()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.samples)))
        print('max len: {}'.format(self.max_length))
        print('# vocab: {}'.format(len(self.word_to_index)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))


class Word2vecLoader(object):

    def __init__(self, path_file, word_to_index, batch_size=64, max_size=None, idx=None):
        self.batch_size = batch_size
        self.word_to_index = word_to_index

        # read file
        df = pd.read_csv(path_file, delimiter='\t')
        # select some data to reproduce
        if max_size is not None:
            size = min(len(df), max_size)
            df = df[:size]

        df['body'] = df['body'].apply(ut._tokenize)
        self.old_samples = df['body'].copy()
        df['body'] = df['body'].apply(self.generate_indexifyer())
        samples = df.values.tolist()
        self.samples = self.transfer_samples(samples)

        if idx is not None:
            self.samples = list(np.array(self.samples)[idx]) # idx: list of indices

        # for batch
        self.n_samples = len(self.samples)
        self.n_batches = int(self.n_samples / self.batch_size)
        self.max_length = self._get_max_length()
        self._create_indices()

    def transfer_samples(self, samples):
        new_samples = []
        for sample in samples:
            sample = sample[1]
            for i in range(5, len(sample) - 6):
                x = sample[i - 5:i + 6]
                x.pop(5)
                new_samples.append((sample[i], x))
        return new_samples
    
    def _create_indices(self):
        self.indices = np.arange(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _get_max_length(self):
        length = 0
        for sample in self.samples:
            length = max(length, len(sample[1]))
        return length

    def generate_indexifyer(self):

        def indexify(lst_text):
            indices = []
            for word in lst_text:
                if word in self.word_to_index:
                    indices.append(self.word_to_index[word])
                else:
                    indices.append(self.word_to_index['____UNKNOW____'])
            return indices

        return indexify

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.index = 0
        self.batch_index = 0 # do not shuffle for each enumeration
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] = batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        label, string = tuple(zip(*batch))

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor([len(s) for s in string])

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        # SORT YOUR TENSORS BY LENGTH!
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        label = torch.LongTensor(label)
        label = label[perm_idx]

        return seq_tensor, label, seq_lengths