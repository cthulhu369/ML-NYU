from c import graph_pairs, labels
from sklearn.model_selection import train_test_split
import dgl
import tensorflow as tf
import numpy as np
from dgl.nn import GraphConv
import torch

train_pairs, test_pairs, train_labels, test_labels = train_test_split(graph_pairs, labels, test_size=0.2, random_state=42)



import dgl
import torch
import torch.nn as nn
from dgl.nn import GraphConv

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

# Initialize model, loss function, and optimizer
model = GIN(input_dim=16, hidden_dim=32, output_dim=1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training step function
def train_step(graphs, labels):
    model.train()
    total_loss = 0
    for g, label in zip(graphs, labels):
        g_dgl = dgl.graph((np.nonzero(g[0])[0], np.nonzero(g[0])[1]))
        g_dgl = dgl.add_self_loop(g_dgl)
        h = torch.ones((g[0].shape[0], 16), dtype=torch.float32)
        
        optimizer.zero_grad()
        prediction = model(g_dgl, h)
        prediction = prediction.view(-1)  # Reshape to match label dimension
        label_tensor = torch.tensor([label], dtype=torch.float32)
        loss = loss_fn(prediction, label_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(graphs)


# Example training loop
for epoch in range(10):
    loss = train_step(train_pairs, train_labels)
    print(f"Epoch {epoch}, Loss: {loss}")





# Training loop
for epoch in range(10):  # Number of epochs
    loss = train_step(train_pairs, train_labels)
    print(f"Epoch {epoch}, Loss: {loss.numpy()}")

tf.convert_to_tensor(labels)
tf.convert_to_tensor