import tkinter as tk

class Graph:
    def __init__(self):
        self.adj_matrix = []
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id):
        self.nodes[node_id] = len(self.nodes)
        for row in self.adj_matrix:
            row.append(0)
        self.adj_matrix.append([0] * len(self.nodes))

    def add_edge(self, start, end):
        if start in self.nodes and end in self.nodes and start != end:
            self.adj_matrix[self.nodes[start]][self.nodes[end]] = 1
            self.adj_matrix[self.nodes[end]][self.nodes[start]] = 1
            self.edges.append((start, end))

    def print_adj_matrix(self):
        for row in self.adj_matrix:
            print(' '.join(map(str, row)))
        print()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graph Drawer")
        self.canvas = tk.Canvas(self, width=400, height=400, bg="white")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.handle_click)
        self.graph = Graph()
        self.selected_node = None
        self.print_button = tk.Button(self, text="Print Adjacency Matrix", command=self.print_adj_matrix)
        self.print_button.pack(pady=10)

    def handle_click(self, event):
        overlapping = self.canvas.find_overlapping(event.x-5, event.y-5, event.x+5, event.y+5)
        if overlapping:
            node = overlapping[0]
            if self.selected_node is None:
                self.selected_node = node
            else:
                self.graph.add_edge(self.selected_node, node)
                self.draw_edge(self.selected_node, node)
                self.selected_node = None
        else:
            self.add_node(event)

    def add_node(self, event):
        node_id = self.canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, fill="black")
        self.graph.add_node(node_id)

    def draw_edge(self, start, end):
        start_coords = self.canvas.coords(start)
        end_coords = self.canvas.coords(end)
        self.canvas.create_line((start_coords[0]+start_coords[2])/2, (start_coords[1]+start_coords[3])/2, 
                                (end_coords[0]+end_coords[2])/2, (end_coords[1]+end_coords[3])/2, fill="black")

    def print_adj_matrix(self):
        self.graph.print_adj_matrix()

if __name__ == "__main__":
    app = Application()
    app.mainloop()
