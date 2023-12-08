import numpy as np

def load_adjacency_matrix(file_path):
    with open(file_path, 'rb') as file:
        # Read the number of nodes
        num_nodes = int.from_bytes(file.read(4), byteorder='big')
        
        # Read the adjacency matrix
        # Assuming each element is stored as an int (4 bytes), hence dtype=np.int32
        matrix_flat = np.fromfile(file, dtype=np.int64)
        
        # Reshape to get the adjacency matrix
        matrix = matrix_flat.reshape((num_nodes, num_nodes))
        return matrix

# Example usage
# file_path = "iso_m2D_s16.A26"  # Replace with your actual file path
# adj_matrix = load_adjacency_matrix(file_path)
# print(adj_matrix)
