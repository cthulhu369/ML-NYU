import tkinter as tk
import math
import random


class Graph:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.node_count = len(adj_matrix)

    def draw(self, canvas):
        angle_step = 2 * math.pi / self.node_count
        radius = 150
        center_x, center_y = 200, 200,  # Center of the circle
        nodes_positions = []

        # Draw nodes
        for i in range(self.node_count):
            x = center_x + radius * math.cos(i * angle_step)
            y = center_y + radius * math.sin(i * angle_step)
            node_id = canvas.create_oval(x-5, y-5, x+5, y+5, fill="black")
            nodes_positions.append((x, y))

        # Draw edges
        for i in range(self.node_count):
            for j in range(i + 1, self.node_count):
                if self.adj_matrix[i][j] == 1:
                    start_x, start_y = nodes_positions[i]
                    end_x, end_y = nodes_positions[j]
                    canvas.create_line(start_x, start_y, end_x, end_y, fill="black")

def draw_graph_from_adj_matrix(adj_matrix):
    app = tk.Tk()
    app.title("Graph from Adjacency Matrix")
    canvas = tk.Canvas(app, width=400, height=400, bg="white")
    canvas.pack(padx=10, pady=10)

    graph = Graph(adj_matrix)
    graph.draw(canvas)

    app.mainloop()



def random_permute_rows(matrix):
    # Make a copy of the input matrix to avoid modifying the original
    permuted_matrix = matrix[:]
    
    # Shuffle the rows of the permuted matrix
    random.shuffle(permuted_matrix)
    
    return permuted_matrix

# Example usage
adj_matrix = [
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0]
]

mat2 = random_permute_rows(adj_matrix)

#draw_graph_from_adj_matrix(adj_matrix)
#draw_graph_from_adj_matrix(mat2)

adj_mat2= [
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0]
]

#draw_graph_from_adj_matrix(adj_mat2)

adj_mat3 = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

#draw_graph_from_adj_matrix(adj_mat3)