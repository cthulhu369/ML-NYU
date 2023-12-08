from c import graph_pairs, labels
from sklearn.model_selection import train_test_split
import dgl
import tensorflow as tf
import numpy as np
from dgl.nn import GraphConv
import torch

train_pairs, test_pairs, train_labels, test_labels = train_test_split(graph_pairs, labels, test_size=0.2, random_state=42)



class GIN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = tf.keras.layers.Dense(output_dim)

    def call(self, g, features):
        x = tf.nn.relu(self.conv1(g, features))
        x = tf.nn.relu(self.conv2(g, x))
        with g.local_scope():
            g.ndata['h'] = x
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)

model = GIN(input_dim=16, hidden_dim=32, output_dim=1)  # Adjust dimensions as needed
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def train_step(graphs, labels):
    with tf.GradientTape() as tape:
        predictions = []
        for g in graphs:
            g_dgl = dgl.graph((np.nonzero(g[0])[0], np.nonzero(g[0])[1]))
            g_dgl = dgl.add_self_loop(g_dgl)
            
            # Update the placeholder features to match the input_dim of the model
            # Assuming your model's first GraphConv layer expects input_dim=16
            h = torch.ones((g[0].shape[0], 16), dtype=torch.float32)  # Change 16 to match your input_dim

            predictions.append(model(g_dgl, h))
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        predictions = torch.cat(predictions, dim=0)

        loss = loss_fn(labels_tensor, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss




# Training loop
for epoch in range(10):  # Number of epochs
    loss = train_step(train_pairs, train_labels)
    print(f"Epoch {epoch}, Loss: {loss.numpy()}")

tf.convert_to_tensor(labels)
tf.convert_to_tensor