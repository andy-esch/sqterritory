import logging
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

import pysal
from shapely.geometry import Point, Polygon
from math import fabs
from itertools import product, combinations

from utils import flatten_list

class BSTCut:
    def __init__(self, context, geo_table, distance_table, distance_col, index_col=None, o_index_col=None, d_index_col=None):

        self.context = context
        self.geo_table = geo_table
        self.index_col = index_col
        self.distance_table = distance_table
        self.o_index_col = o_index_col
        self.d_index_col = d_index_col
        self.distance_col = distance_col

        self.geo = self.context.read(self.geo_table, decode_geom=True)
        self.geo = self.geo.set_index(self.index_col, drop=False)
        self.distance = self.context.read(self.distance_table)[[self.o_index_col, self.d_index_col, self.distance_col]]
        self.dist_dict = self.distance.set_index([self.o_index_col, self.d_index_col]).to_dict()

        self.output_table = None

    def _network_from_df(self, df, distance, distance_col):
        """
        This constructs a minimum spanning tree
        and full connectivitiy network
        from the geodata frame and adjacency matrix
        """
        contiguity = pysal.weights.Queen.from_dataframe(df)
        centroids = []
        for p in self.geo.geometry:
            if isinstance(p, Point):
                centroids.append((p.x, p.y))
            else:
                centroids.append((p.centroid.x, p.centroid.y))
        points_dict = dict(zip(df.index, centroids))

        G=nx.Graph()
        G.add_nodes_from(list(df.index))

        for node, neighbors in contiguity.neighbors.items():
            for n in neighbors:
                G.add_edge(node,
                           n,
                           distance=distance[distance_col][(node, n)])
        mst = nx.minimum_spanning_tree(G, weight=distance_col)
        return mst, G, points_dict, contiguity

    def _longest_crossing_distance (self, graph, distance, distance_col, no_direction=False):
        leaf_nodes = [k for k,d in graph.degree() if d == 1]
        if no_direction:
            leaf_nodes_pairs = combinations(leaf_nodes, 2)
        else:
            leaf_nodes_pairs = [(x, y) for x, y in product(leaf_nodes, leaf_nodes) if x!=y]
        max_ = 0
        for l1, l2 in leaf_nodes_pairs:
            max_ = max(distance[distance_col][(l1, l2)], max_)
        return max_

    def _cut_graph(self, graph, edge):
        tempGraph = graph.copy()
        tempGraph.remove_edge(edge[0], edge[1])
        return list(nx.connected_component_subgraphs(tempGraph))

    def _find_optimal_cut(self, graph, distance, distance_col, property=False):
        smallest_diff = None
        best_edge = None

        for index, edge in enumerate(graph.edges):
            children = self._cut_graph(graph,edge)
            diff_corssing_dist  = fabs(self._longest_crossing_distance(children[0], distance, distance_col) - self._longest_crossing_distance(children[1], distance, distance_col))
            if smallest_diff==None or diff_corssing_dist < smallest_diff:
                best_edge = edge
                smallest_diff = diff_corssing_dist
        return best_edge

    def _partition(self, graph, distance, distance_col, lcd_thresh, n_thresh, stopping_limit = 40, level=0):
        lcd = self._longest_crossing_distance(graph, distance, distance_col)
        logging.info(f'considering level: {level}, no. of nodes in graph is {len(graph.nodes)}, lcd: {lcd}')
        if (lcd < lcd_thresh or len(graph.nodes)<=n_thresh) or (level>stopping_limit):
            return graph
        else:
            cut = self._find_optimal_cut(graph, distance, distance_col)
            split = self._cut_graph(graph, cut)
            logging.info('splitting')
        return [self._partition(split[0], distance, distance_col, lcd_thresh, n_thresh, stopping_limit=stopping_limit, level=level+1),
                self._partition(split[1], distance, distance_col, lcd_thresh, n_thresh, stopping_limit=stopping_limit, level=level+1)]

    def calc(self, min_lcd=60*60, n_stop_split=2):
        mst_network, full_network, points, contiguity = self._network_from_df(self.geo, self.dist_dict, self.distance_col)
        result = self._partition(mst_network, self.dist_dict, self.distance_col, min_lcd, n_stop_split)
        flat_result = list(flatten_list(result))
        geo_result  = self.geo.copy().assign(cluster_id=0 , cluster_lsd=0)
        for index, region in enumerate(flat_result):
            lsd = self._longest_crossing_distance(region, self.dist_dict, self.distance_col)
            node_ids = [nid for nid in region.nodes]
            geo_result.loc[node_ids, ['cluster_id']] = index
            geo_result.loc[node_ids, ['cluster_lsd']] = lsd
        self.output_table = geo_result
        logging.info('calc done!')

    def results_to_table(self):
        out_table = 'bstcut_{}'.format(str(time.time())[-5:])
        logging.info('Writing output to {}'.format(out_table))
        self.context.write(self.output_table.reset_index(drop=True), out_table)
        logging.info('Table {} written to CARTO'.format(out_table))
        return out_table
