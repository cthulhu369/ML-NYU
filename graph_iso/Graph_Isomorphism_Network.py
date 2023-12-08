from c import graph_pairs, labels
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


# Test step function
def evaluate(graphs, labels):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for g, label in zip(graphs, labels):
            g_dgl = dgl.graph((np.nonzero(g[0])[0], np.nonzero(g[0])[1]))
            g_dgl = dgl.add_self_loop(g_dgl)
            h = torch.ones((g[0].shape[0], 16), dtype=torch.float32)
            
            prediction = model(g_dgl, h)
            predicted_label = torch.round(torch.sigmoid(prediction))
            correct += (predicted_label == torch.tensor([label], dtype=torch.float32)).sum().item()
            total += 1

    return correct / total

def plot_graph(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)
    nx.draw(G, with_labels=True)
    plt.show()

if __name__ == "__main__":
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(graph_pairs, labels, test_size=0.2, random_state=42)
    
    # Initialize model, loss function, and optimizer
    model = GIN(input_dim=16, hidden_dim=32, output_dim=1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Example training loop
    for epoch in range(10):
        loss = train_step(train_pairs, train_labels)
        print(f"Epoch {epoch}, Loss: {loss}")

    test_accuracy = evaluate(test_pairs, test_labels)
    print(f"Test Accuracy: {test_accuracy}")

    # Manual inspection
    # Example: Select the first 5 graph pairs from the test set
    sample_graph_pairs = test_pairs[:5]
    sample_labels = test_labels[:5]
    sample_predictions = []
    for g in sample_graph_pairs:
        g_dgl = dgl.graph((np.nonzero(g[0])[0], np.nonzero(g[0])[1]))
        g_dgl = dgl.add_self_loop(g_dgl)
        h = torch.ones((g[0].shape[0], 16), dtype=torch.float32)
        
        with torch.no_grad():  # Disable gradient computation
            prediction = model(g_dgl, h)
            predicted_label = torch.round(torch.sigmoid(prediction)).item()
            sample_predictions.append(predicted_label)

    for i, (pred, label) in enumerate(zip(sample_predictions, sample_labels)):
        print(f"Graph Pair {i}: Predicted Label - {pred}, Actual Label - {label}")

    # Plot the first graph pair as an example
    plot_graph(sample_graph_pairs[1][0])
    plot_graph(sample_graph_pairs[1][1])

    plot_graph(sample_graph_pairs[0][0])
    plot_graph(sample_graph_pairs[0][1])

    plot_graph(graph_pairs[18][1])


    graph_pairs[1][0].shape
