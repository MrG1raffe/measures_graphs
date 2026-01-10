import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union, Callable, List

class GeometricGraph:
    coordinates: np.ndarray
    graph: nx.Graph
    dist_mat: np.ndarray
    adj_mat: np.ndarray
    names: List[str]

    def __init__(
            self,
            coordinates,
            dist_to_adj=Union[str, float, Callable],
            distance: Union[str, Callable] = "euclidean",
            names: List[str] = None
    ):
        """

        :param coordinates: An array of size (n, dim).
        :param dist_to_adj:
        :param distance: "euclidian" for d-dimensional euclidean distance.
        """
        self.coordinates = coordinates
        if names is None:
            self.names = list(range(len(coordinates)))
        else:
            self.names = names

        if distance == "euclidean":
            self.dist_mat = np.sqrt(((coordinates.T[:, :, None] - coordinates.T[:, None, :])**2).sum(axis=0))
        else:
            raise NotImplementedError("")

        if isinstance(dist_to_adj, (int, float)):
            self.adj_mat = np.astype(self.dist_mat <= dist_to_adj, int)
        else:
            raise NotImplementedError("")
        np.fill_diagonal(self.adj_mat, 0)

        self.graph = nx.from_numpy_array(self.adj_mat)
        if names is not None:
            mapping = {i: name for i, name in enumerate(names)}
            self.graph = nx.relabel_nodes(self.graph, mapping)

    def plot(self, **kwargs):
        pos = {name: coordinate for name, coordinate in zip(self.names, self.coordinates)}
        nx.draw(self.graph, pos, with_labels=True, node_color='lightgreen', node_size=70, font_size=7, edge_color='black', **kwargs)
        plt.show()