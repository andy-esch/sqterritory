"""Balanced territory assignments using Min Cost Flow
Modified from code written by Wenfei Xu, 2018
"""
import logging
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import geopandas as gpd
import networkx as nx
from geopy import distance

logger = logging.basicConfig(level=logging.DEBUG)

# distance functions
def calc_distance(lng1, lat1, lng2, lat2, arctype='great_circle'):
    if arctype =='great_circle':
        return distance.great_circle((lat1, lng1), (lat2, lng2)).km
    else:
        return distance.distance((lat1, lng1), (lat2, lng2)).km

def calc_euclideandistance(lng1, lat1, lng2, lat2):
    return np.linalg.norm(np.array([lng1, lat1]) - np.array([lng2, lat2]))

# create ufunc version of calc_distance (works on arrays)
dist_vect = np.vectorize(calc_distance)

def compactness_frompts(geoseries):
    """
    Inverse of 4 * pi * Area / perimeter^2
    """
    measure = (
        4 * 3.1415 * (
            (geoseries.unary_union).convex_hull.area)
        ) / (
            (geoseries.unary_union).convex_hull.length
        )
    return measure

def standardize(x):
    return (x - x.mean()) / x.std()

# Work in progress
class MinCostFlow:
    """Min Cost Flow implementation"""
    def __init__(self, origin_table, target_table, context, demand_col='value_demanded'):
        q = '''
            SELECT *, ST_Y(the_geom) as lat, ST_X(the_geom) as lng
            FROM {t}
            where the_geom is not null and abs(ST_Y(the_geom)) < 90
        '''
        self.context = context

        self.origin_table = origin_table
        logging.info(q.format(t=origin_table))
        self.origins = self.context.query(q.format(t=origin_table))
        self.origins['origin_id']  = self.origins.index

        self.demand_col = demand_col
        self.target_table = target_table
        self.targets = self.context.query(q.format(t=target_table))
        self.targets['target_id'] = self.origins.index.max() + self.targets.index
        self.targets['value_demanded_norm'] = standardize(self.targets[demand_col].values)

        self.nearest_targets = self._get_target_nearest()

        self.results = {}

    def _verify_inputs():
        """"""
        if 'value_demanded' not in self.origins.columns:
            raise ValueError('Origins must have a column called value_demanded')

    def _get_target_nearest(self):
        """Get nearest target for each origin"""
        reps_query = """
             SELECT DISTINCT ON(g2.cartodb_id)
                 g1.cartodb_id As origin_id,
                 g2.the_geom,
                 g2.cartodb_id + {maxorigin} as cartodb_id,
                 g2.the_geom_webmercator
             FROM {origin_table} As g1, {target_table} As g2
             ORDER BY g2.cartodb_id, g1.the_geom <-> g2.the_geom
        """.format(
            maxorigin=self.origins.index.max(),
            origin_table=self.origin_table,
            target_table=self.target_table
        )

        nearest_reps = self.context.query(
            reps_query,
            decode_geom=True
        )

        nearest_reps = gpd.GeoDataFrame(nearest_reps, geometry='geometry')
        init_labels = nearest_reps['origin_id'].values

        # update with new information
        self.targets['labels'] = init_labels
        logging.info('nearest targets retrieved')

        return nearest_reps

    def _get_demand_graph(self):
        """create demand graph"""
        # The number of clusters
        K = self.origins.shape[0]

        # Set the number of accounts in each cluster to be the same
        # as for the nearest neighbor solution
        demand = self.nearest_targets.groupby('origin_id')['geometry'].count().to_dict()

        # Set up the graph so we can extract and initialize the node labels.
        # For each iteration, we're going to sort all our data by their origin
        # label assignments in order to properly index our nodes.
        self.targets = self.targets.sort_values('labels').reset_index(drop=True)

        # Add target nodes
        g = nx.DiGraph()
        g.add_nodes_from(self.targets['target_id'], demand=-1)

        # Add origin nodes
        for idx in demand:
            g.add_node(int(idx), demand=demand[idx])

        # Dictionary of labels (corresponding to the sales rep) for
        # each med center node.
        dict_M = {
            i: (
                self.targets[self.targets['target_id'] == i]['labels'].values
                if i in self.targets.target_id.values
                else np.array([demand[i]])
            )
            for i in g.nodes
        }
        logging.info('Graph and demand dictionary created')
        return dict_M, demand

    def calc(self, maxiter=100, fixedprec=1e9):
        """Min Cost Flow"""
        source_data_holder = []
        N = self.targets.shape[0]
        K = self.origins.shape[0]

        # dict of labels for each target node
        M, demand = self._get_demand_graph()

        max_dist_trip = 400  # kilometers
        cost_holder = []
        itercnt = 0

        while True:
            itercnt += 1
            logging.info(f'Iter count: {itercnt}')

            # Setup the graph
            g = nx.DiGraph()

            self.targets = self.targets.sort_values('labels').reset_index(drop=True)
            # Supply of 1 (i.e. demand = -1) means that it can only be connected to one node
            g.add_nodes_from(self.targets['target_id'], demand=-1) # points

            for idx in self.nearest_targets.origin_id:
                g.add_node(int(idx), demand=demand[idx])

            ### Create the cluster centers calculate a distance cost
            cost_dist = dist_vect(
                np.tile(self.targets['lng'].values, K),
                np.tile(self.targets['lat'].values, K),
                np.repeat(self.origins['lng'].values, N),
                np.repeat(self.origins['lat'].values, N)
            )[:, np.newaxis].T

            scaler_dist = MinMaxScaler()
            cost_dist_trans = scaler_dist.fit_transform(cost_dist.T).T

            # Penalty for distances too large
            cost_dist_trans[cost_dist > max_dist_trip] = 10

            # Create the in-cluster sales and calculate the total volume of sales generated
            # TODO: rename this to something more generic, like cluster_demanded
            cluster_sales = self.targets.groupby('labels').sum()[self.demand_col][:, np.newaxis]
            D = cluster_sales.shape[1]

            cost_sales = abs(
                np.array([
                    np.linalg.norm(
                            np.repeat(cluster_sales, N)[:, np.newaxis] \
                            - np.tile(cluster_sales.mean(), (K * N))[:,np.newaxis],
                            axis=1
                        )
                    ])
                )

            scaler_sales = MinMaxScaler()
            cost_sales = scaler_sales.fit_transform(cost_sales.T).T

            # Total cost TO CHANGE??
            cost_total = cost_dist_trans + cost_sales

            cost_holder.append(sum(cost_total[0]))


            # Create the edges of points to centers
            data_to_center_edges = np.concatenate(
                (
                    np.tile(self.targets['target_id'], K).T[:, np.newaxis],
                    np.array([np.tile(int(i+1), self.targets.shape[0]) for i in range(K)]).reshape(self.targets.shape[0] * K, 1),
                    cost_total.T * 1e5
                ),
                axis=1
            ).astype(np.uint64)

            # Add these edges to the graph
            g.add_weighted_edges_from(data_to_center_edges)

            # Add the extra balance node
            # To balance out the network, we create an extra node that is:
            # -(K*(-1)+sum(demand_per_node))
            a = 99999
            g.add_node(a, demand=self.targets.shape[0] - np.sum(list(demand.values())))

            C_to_a_edges = np.concatenate(
                (
                    np.array([int(i + 1) for i in range(K)]).T[:, np.newaxis],
                    np.tile([[a, ]], K).T
                ),
                axis=1
            )
            g.add_edges_from(C_to_a_edges)

            # Calculate the minimum flow cost
            f = nx.min_cost_flow(g)

            # Update the new labels
            M_new = {}
            p = {}

            for i in list(g.nodes)[:-1]:
                # Sorts all the items in the dictionary and picks the cluster
                # with label = 1
                p = sorted(f[i].items(), key=lambda x: x[1])[-1][0]
                M_new[i] = p

            # Update the new labels in the df
            self.targets['labels'] = self.targets.apply(lambda x: M_new[x['target_id']], axis=1)

            # Set the capacity for all edges
            # TO DO: Figure how/whether we need to properly set a capacity for the edges.
            C = 50
            nx.set_edge_attributes(g, C, 'capacity')

            # Test whether we can stop
            # stop condition
            if np.all(M_new == M):
                print("All same")
                self.results = {
                    'dict_graph': M,
                    'min_cost_flow': f,
                    'nxgraph': g,
                    'model_labels': self.targets,
                    'costs': cost_holder
                }
                return True

            M = M_new

            source_data_holder.append(self.targets['labels'].values)
            if maxiter is not None and itercnt >= maxiter:
                # Max iterations reached
                self.results = {
                    'dict_graph': M,
                    'min_cost_flow': f,
                    'nxgraph': g,
                    'model_labels': self.targets,
                    'costs': cost_holder
                }
                return True

    def results_to_table(self):
        """Process self.results and send to carto table"""
        # Get Labels
        baseline_labels = self.nearest_targets['origin_id'].values
        mcf_labels = self.results['model_labels']['labels'].values

        # Create the outcomes
        outcome = pd.DataFrame({
                'the_geom': [
                    'SRID=4326;Point({lng} {lat})'.format(lng=v[0], lat=v[1])
                    for v in zip(self.results['model_labels']['lng'].values, self.results['model_labels']['lat'].values)],
                'target_lng': self.results['model_labels']['lng'].values,
                'target_lng': self.results['model_labels']['lat'].values,
                'origin_lng': self.origins.reindex(baseline_labels)['lng'].values,
                'origin_lat': self.origins.reindex(baseline_labels)['lat'].values,
                'target_id': self.results['model_labels'].target_id,
                'sales': self.results['model_labels'][self.demand_col].values,
                'labels': baseline_labels
            }
        )

        outcomes2 = pd.DataFrame({
                'the_geom': [
                    'SRID=4326;Point({lng} {lat})'.format(lng=v[0], lat=v[1])
                    for v in zip(
                        self.results['model_labels']['lng'].values,
                        self.results['model_labels']['lat'].values
                    )
                ],
                'target_lng': self.results['model_labels']['lng'].values,
                'target_lat': self.results['model_labels']['lat'].values,
                'origin_lng': self.origins.reindex(mcf_labels)['lng'].values,
                'origin_lat': self.origins.reindex(mcf_labels)['lat'].values,
                'target_id': self.results['model_labels'].target_id,
                'sales': self.results['model_labels'][self.demand_col].values,
                'labels': mcf_labels
            },
            index=self.results['model_labels'].target_id
        )
        now = datetime.datetime.now()
        out_table = 'mincostflow_{}'.format(now.strftime("%Y_%m_%d_%H_%M_%S"))
        logging.info('Writing output to {}'.format(out_table))
        self.context.write(outcomes2.reset_index(drop=True), out_table)
        logging.info('Table {} written to CARTO'.format(out_table))
        return out_table

def kmeans_territories(dataframe, columns, n_clusters):
    """"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, Imputer

    # preprocess the data
    # fill missing values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data = imp.fit_transform(dataframe[columns].values)
    # scale
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # build model
    km = KMeans(n_clusters=n_clusters).fit(data)
    dataframe['labels'] = km.labels_
    dataframe['labels'] = dataframe['labels'].astype(str)

    return dataframe
