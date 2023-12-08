import tkinter as tk

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_matrix = [[0 for _ in range(vertices)] for _ in range(vertices)]
        self.nodes = {}
        self.edges = []

    def add_edge(self, start, end):
        if start in self.nodes and end in self.nodes:
            self.adj_matrix[self.nodes[start]][self.nodes[end]] = 1
            self.adj_matrix[self.nodes[end]][self.nodes[start]] = 1
            self.edges.append((start, end))

    def print_adj_matrix(self):
        for row in self.adj_matrix:
            print(' '.join(map(str, row)))

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graph Drawer")
        self.canvas = tk.Canvas(self, width=400, height=400, bg="white")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.add_node)
        self.canvas.bind("<Button-3>", self.add_edge)
        self.graph = Graph(vertices=0)
        self.node_counter = 0
        self.selected_node = None

    def add_node(self, event):
        node_id = self.canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill="black")
        self.graph.nodes[node_id] = self.node_counter
        self.node_counter += 1
        self.graph.vertices = self.node_counter
        self.graph.adj_matrix = [[0 for _ in range(self.node_counter)] for _ in range(self.node_counter)]
        for start, end in self.graph.edges:
            self.graph.adj_matrix[self.graph.nodes[start]][self.graph.nodes[end]] = 1
            self.graph.adj_matrix[self.graph.nodes[end]][self.graph.nodes[start]] = 1

    def add_edge(self, event):
        x, y = event.x, event.y
        overlapping = self.canvas.find_overlapping(x-5, y-5, x+5, y+5)
        if overlapping:
            node = overlapping[0]
            if self.selected_node is None:
                self.selected_node = node
            else:
                self.graph.add_edge(self.selected_node, node)
                self.draw_edge(self.selected_node, node)
                self.selected_node = None

    def draw_edge(self, start, end):
        start_coords = self.canvas.coords(start)
        end_coords = self.canvas.coords(end)
        self.canvas.create_line(start_coords[2], start_coords[3], end_coords[0], end_coords[1], fill="black")

    def print_adj_matrix(self):
        print("\nAdjacency Matrix:")
        self.graph.print_adj_matrix()

if __name__ == "__main__":
    app = Application()
    app.mainloop()
