import torch.nn as nn
import torch.nn.functional as F


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, model_type, hidden_layer, kernel_num=5):
        super().__init__()
        embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.add_module('0', embedding)
        if model_type == 'cnn':
            rnn_model = nn.Conv2d(1, 1, kernel_num, (5, embed_dim))
        elif model_type == 'bi_lstm':
            if __name__ == '__main__':
                rnn_model = nn.LSTM(
                    input_size=embed_dim,
                    hidden_size=embed_dim,
                    num_layers=hidden_layer,
                    batch_first=True,
                    bidirectional=True
                )
                self.add_module('1', rnn_model)
        elif model_type == 'lstm':
                rnn_model = nn.LSTM(
                    input_size=embed_dim,
                    hidden_size=embed_dim,
                    num_layers=hidden_layer,
                    batch_first=True,
                    bidirectional=True
                )
                self.add_module('1', rnn_model)
        else:
            raise ValueError
        self.add_module('2', nn.Dropout(0.3))
        self.add_module('3', nn.Linear(embed_dim, embed_dim))
        self.add_module('4', nn.Dropout(0.3))
        self.add_module('5', nn.ReLU(True))
        self.add_module('6', nn.Linear(embed_dim, num_class))
        #self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)