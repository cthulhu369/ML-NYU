from sklearn.model_selection import train_test_split
import dgl
import tensorflow as tf
import numpy as np
from dgl.nn import GraphConv
import torch
import dgl
import torch
import torch.nn as nn
from dgl.nn import GraphConv
import networkx as nx
import matplotlib.pyplot as plt

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, output_dim)

    def forward(self, g, features):
        x = torch.relu(self.conv1(g, features))
        x = torch.relu(self.conv2(g, x))
        with g.local_scope():
            g.ndata['h'] = x
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)

def plot_graph(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    nx.draw(G, with_labels=True)
    plt.show()

