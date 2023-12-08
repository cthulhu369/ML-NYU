import pandas as pd
import sys
import numpy as np
import os
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

def save_adjacency_matrix(adjacency_matrix, output_filename):
    try:
        with open(output_filename, "wb") as file:
            num_nodes = adjacency_matrix.shape[0]
            file.write(num_nodes.to_bytes(4, byteorder='big'))  # Write the number of nodes
            adjacency_matrix.tofile(file)
        print(f"Adjacency matrix saved to {output_filename}")
    except Exception as e:
        print(f"Error saving adjacency matrix: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_directory output_directory")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)
        adjacency_matrix = read_graph(input_file_path)
        
        if adjacency_matrix is not None:
            print(adjacency_matrix.shape)
            save_adjacency_matrix(adjacency_matrix, output_file_path)
