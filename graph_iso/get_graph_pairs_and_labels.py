from load_adj_mat import load_adjacency_matrix


def load_graph_pairs_and_labels(gtr_file_path : str, adj_mat_dir_path : str) -> list:
    graph_pairs = []
    labels = []
    with open(gtr_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            graph_name = parts[0]
            label = 1  # since all are isomorphic
            # # Construct file paths for the pair of graphs
            graph_a_path = f'{adj_mat_dir_path}/{graph_name}'
            graph_b_path = graph_a_path.replace('.A', '.B')
            # Load the adjacency matrices
            graph_a = load_adjacency_matrix(graph_a_path)
            graph_b = load_adjacency_matrix(graph_b_path)
            # Add the pair and label to your lists
            graph_pairs.append((graph_a, graph_b))
            labels.append(label)
    return graph_pairs, labels

# len(graph_pairs)
# type(graph_pairs[0])
# graph_pairs[0][0].shape
# graph_pairs[0][1].shape
# graph_pairs[0][0]

# imod.draw_graph_from_adj_matrix(graph_pairs[0][0])
# imod.draw_graph_from_adj_matrix(graph_pairs[0][1])