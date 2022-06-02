import random

import plotly.graph_objects as go
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

        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x, node_y = [], []
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=20,
                line_width=2))

        node_adjacencies = []
        node_text = []
        for adjacencies in self.graph.adjacency():
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(adjacencies[0])

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        return go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )


if __name__ == '__main__':
    ploter = NetworkConnectionsPlotter()
    for i in range(20):
        for j in random.sample(range(20), 5):
            ploter.add_edge(f"node_{i}", f"node_{j}")
    f = ploter.get_figure()
    f.show()
