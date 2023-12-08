import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def read_word(file):
    # Read two bytes (16 bits) using little-endian format
    data = file.read(2)
    if len(data) != 2:
        raise ValueError("Unexpected end of file")
    return int.from_bytes(data, 'little')

def read_graph(file_path):
    with open(file_path, 'rb') as file:
        # Read the number of nodes
        nodes = read_word(file)

        # Initialize the adjacency matrix with zeros
        matrix = np.zeros((nodes, nodes), dtype=int)

        # For each node...
        for i in range(nodes):
            # Read the number of edges coming out of node i
            edges = read_word(file)

            # For each edge out of node i...
            for _ in range(edges):
                # Read the destination node of the edge
                target = read_word(file)

                # Insert the edge in the adjacency matrix
                matrix[i][target] = 1

        return matrix

# Usage example
file_path = "iso_m2D_s16.A00"  # Replace with your actual file path
adjacency_matrix = read_graph(file_path)
print(adjacency_matrix)
print(adjacency_matrix.shape)
