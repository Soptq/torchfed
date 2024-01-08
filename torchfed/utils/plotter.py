import networkx
import plotly.graph_objects as go

import networkx as nx
import torch.nn
from prettytable import PrettyTable


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

    def remove_edge(self, from_node, to_node):
        try:
            self.graph.remove_edge(from_node, to_node)
        except networkx.NetworkXError:
            pass

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


class DataTransmitted:
    def __init__(self):
        self._data_transmitted = {}

    def add(self, from_node, to_node, size):
        if from_node == to_node:
            return

        if from_node not in self._data_transmitted:
            self._data_transmitted[from_node] = {}

        if to_node not in self._data_transmitted[from_node]:
            self._data_transmitted[from_node][to_node] = 0

        if to_node not in self._data_transmitted:
            self._data_transmitted[to_node] = {}

        if from_node not in self._data_transmitted[to_node]:
            self._data_transmitted[to_node][from_node] = 0

        self._data_transmitted[from_node][to_node] += size

    def get_size(self, from_node, to_node):
        if from_node == to_node:
            return 0
        if from_node not in self._data_transmitted:
            return 0
        if to_node not in self._data_transmitted:
            return 0
        if to_node not in self._data_transmitted[from_node]:
            return 0
        return self._data_transmitted[from_node][to_node]

    def get_total_outbound(self, node):
        if node not in self._data_transmitted:
            return 0

        total_size = 0
        for to_node, size in self._data_transmitted[node].items():
            total_size += size
        return total_size

    def get_total_inbound(self, node):
        total_size = 0
        for _, outbounds in self._data_transmitted.items():
            if node in outbounds:
                total_size += outbounds[node]
        return total_size

    def get_transmission_matrix_str(self):
        matrix = PrettyTable()
        axis = sorted(self._data_transmitted.keys())
        matrix.field_names = ["(bytes)"] + axis
        for from_node in axis:
            row = [from_node]
            for to_node in axis:
                row.append(self.get_size(from_node, to_node))
            matrix.add_row(row)
        return matrix.get_string()

    def get_figure(self):
        data = []
        axis = sorted(self._data_transmitted.keys())
        for from_node in axis:
            row = []
            for to_node in axis:
                row.append(self.get_size(from_node, to_node))
            data.append(row)

        layout = go.Layout(xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Receiver',
            )),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text='Sender',
                )
        ))
        fig = go.Figure(
            data=go.Heatmap(
                hovertemplate='Sender: %{y}<br>Receiver: %{x}<br>Size (bytes): %{z}<extra></extra>',
                z=data,
                x=axis,
                y=axis,
                xgap=3,
                ygap=3,
                hoverongaps=False,
                showscale=False,
            ),
            layout=layout)
        return fig


if __name__ == '__main__':
    # import random
    # ploter = NetworkConnectionsPlotter()
    # for i in range(20):
    #     for j in random.sample(range(20), 5):
    #         ploter.add_edge(f"node_{i}", f"node_{j}")
    # f = ploter.get_figure()
    # f.show()

    a = DataTransmitted()
    a.add("node_1", "node_2", 32)
    a.add("node_1", "node_3", 78)
    a.get_total_outbound("node_1")
    a.get_total_outbound("node_2")
    a.get_total_outbound("na")
    a.get_total_inbound("node_2")
    print(a.get_transmission_matrix_str())
    f = a.get_figure()
    f.show()
