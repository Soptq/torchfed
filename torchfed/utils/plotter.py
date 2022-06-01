import random

import matplotlib.pyplot as plt
import networkx as nx


class NetworkConnectionsPlotter:
    def __init__(self):
        self.graph = nx.DiGraph(directed=True)

    def add_node(self, node_name):
        if node_name not in self.graph:
            self.graph.add_node(node_name)

    def add_edge(self, from_node, to_node):
        self.add_node(from_node)
        self.add_node(to_node)
        self.graph.add_edge(from_node, to_node)

    def get_figure(self):
        pos = nx.circular_layout(self.graph)
        options = {"edgecolors": "#000000", "node_size": 600, "alpha": 0.90}
        figure = plt.figure(figsize=(5, 5), dpi=300)

        nx.draw_networkx_nodes(self.graph, pos,
                               # node_color=node_colors,
                               **options)
        nx.draw_networkx_edges(self.graph, pos,
                               width=1,
                               alpha=0.5,
                               # edgelist=edges,
                               # edge_color=weights,
                               edge_cmap=plt.cm.Greys,
                               edge_vmin=0.,
                               edge_vmax=1.,
                               arrowsize=16,
                               node_size=1000,
                               arrowstyle="-|>",
                               connectionstyle="arc3,rad=0.2")
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_color="black")

        plt.tight_layout()
        plt.axis("off")

        return figure


if __name__ == '__main__':
    ploter = NetworkConnectionsPlotter()
    for i in range(20):
        for j in random.sample(range(20), 5):
            ploter.add_edge(f"node_{i}", f"node_{j}")
    f = ploter.get_figure()
    plt.show()
