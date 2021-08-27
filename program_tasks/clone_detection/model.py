import dgl
import torch
import torch.nn as nn
from dgl.graph import DGLGraph
from dgl.nn.pytorch import GatedGraphConv, GraphConv
import numpy as np


class CloneModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, embedding_tensor=None,
                 padding_index=1, hidden_size=100, n_steps=3):
        super(CloneModel, self).__init__()
        self.embed_dim = embed_dim
        if torch.is_tensor(embedding_tensor):
            self.encoder = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_index, _weight=embedding_tensor)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_index)

        # self.ggnn = \
        #     GatedGraphConv(self.embed_dim, self.embed_dim, n_steps, 1, )
        #
        self.gcn = GraphConv(self.embed_dim, self.embed_dim,)
        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(1, 1, (1, 3))

        self.pool_1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), )
        self.conv_2 = nn.Conv2d(1, 1, (1, 1))

        self.pool_2 = nn.MaxPool2d(kernel_size=(3, 2), stride=(1, 2), padding=(1, 0))
        self.linear_1 = nn.Linear(int(self.embed_dim / 2 - 1), hidden_size)
        self.linear_2 = nn.Linear(int(self.embed_dim / 4 - 1), hidden_size)

        self.mlp = nn.Linear(int(self.embed_dim * 2), 2)
        # self.atten = nn.Att
        self.softmax = nn.Softmax()

    def get_localvec(self, n_list, device):
        local = []
        for node in n_list:
            node_local = []
            for n in node:
                emb = self.encoder(n.to(device))
                node_local.append(emb)
            node_local = torch.stack(node_local).to(device)
            local.append(node_local)  # normalize
        return local

    def forward(self, node_1, graph_1, node_2, graph_2, device):
        node_1 = [torch.tensor(n, dtype=torch.long, device=device) for n in node_1]
        node_2 = [torch.tensor(n, dtype=torch.long, device=device) for n in node_2]

        local_1 = self.get_localvec(node_1, device)
        local_1 = [self.dropout(l) for l in local_1]
        batch_graph_1 = self.perpare_dgl(graph_1, local_1, device)
        vec_1 = self.calRes(batch_graph_1, device)

        local_2 = self.get_localvec(node_2, device)
        local_2 = [self.dropout(l) for l in local_2]
        batch_graph_2 = self.perpare_dgl(graph_2, local_2, device)
        vec_2 = self.calRes(batch_graph_2, device)

        vec = torch.cat([vec_1, vec_2],  dim=1)
        return self.mlp(vec)   #torch.cosine_similarity(vec_1, vec_2)

    def calRes(self, batch_graph, device):
        local = batch_graph.ndata['tk'].to(device)
        feature = self.gcn(batch_graph, local)
        feature = self.dropout(feature)
        num = len(local)

        global_feature = torch.cat([local, feature], dim=1).view([1, 1, num, -1])
        # global_feature = global_feature.permute(0,2,1)
        local = local.view([1, 1, num, -1])
        # local = local.permute(0,2,1)
        z_vec = self.conv_1(global_feature)
        z_vec = self.relu(z_vec)
        z_vec = self.pool_1(z_vec)
        z_vec = self.conv_2(z_vec)
        z_vec = self.relu(z_vec)
        z_vec = self.pool_2(z_vec).view([num, -1])

        # y_vec = self.conv_1(local)
        # y_vec = self.relu(y_vec)
        # y_vec = self.pool_1(y_vec)
        # y_vec = self.conv_2(y_vec)
        # y_vec = self.relu(y_vec)
        # y_vec = self.pool_2(y_vec).view([num, -1])

        z_vec = self.linear_1(z_vec)
        # y_vec = self.linear_2(y_vec)

        #res = (z_vec * y_vec)
        res = z_vec
        batch_graph.ndata['res'] = res
        res = dgl.mean_nodes(batch_graph, 'res')
        return res


    @staticmethod
    def perpare_dgl(graph, local, device):
        dgl_list = []
        for i in range(len(local)):
            dgl_graph = DGLGraph()
            dgl_graph.add_nodes(len(local[i]))
            st, ed = np.nonzero(graph[i])
            dgl_graph.add_edges(st, ed)
            dgl_graph.ndata['tk'] = local[i].to(device=device)
       