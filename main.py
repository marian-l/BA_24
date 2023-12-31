import string
import sys
from typing import Tuple

import numpy as np
import pandas
from varname import nameof

import pandas as pd
import pm4py
from networkx import connected_components

from pm4py import OCEL
from pm4py.algo.discovery.ocel.ocpn import algorithm as ocpn_discovery
import networkx as nx
import matplotlib.pyplot as plt

import customJsonSetEncoder

OM_PATH = 'data/order_management/order-management.json'

GRAPH_DATA_PATH = 'data/order_management/graphs/'
PETRI_NET = 'data/order_management/graphs/PetriNetModelLanguage/'

TEXT_DATA_PATH = 'data/order_management/text/'

DATA_PATH = 'data/order_management/'


def discover_objects_graph(log: OCEL):
    obj_interaction = pm4py.discover_objects_graph(log, graph_type="object_interaction")
    obj_descendants = pm4py.discover_objects_graph(log, graph_type="object_descendants")
    obj_inheritance = pm4py.discover_objects_graph(log, graph_type="object_inheritance")
    obj_cobirth = pm4py.discover_objects_graph(log, graph_type="object_cobirth")
    obj_codeath = pm4py.discover_objects_graph(log, graph_type="object_codeath")

    object_graphs = [
        obj_interaction,
        obj_descendants,
        obj_inheritance,
        obj_cobirth,
        obj_codeath
    ]

    return object_graphs


def save_object_interactions(object_graphs: list[set[tuple[str, str]]]):
    i = 0
    for object_graph in object_graphs:
        f = open(DATA_PATH + 'text/object_interactions/' + str(i) + '.txt', 'x')
        f.write(object_graph.__str__())
        i += 1


def save_object_graph(object_graphs: list[set[tuple[str, str]]]):
    # Problem: Für alle Graphen außer Object-Codeath ergibt sich ein einziger, großer Eintrag, der nicht in weitere
    # Subgraphen unterteilt wird.
    for graph_type in object_graphs:
        graph = nx.Graph(incoming_graph_data=graph_type)  # undirected Graph
        # graph has a list of nodes type NodeView

        connected_components_result = connected_components(graph)
        largest_cc = max(nx.connected_components(graph), key=len)
        list_of_cc = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]

        di_graph = nx.DiGraph(incoming_graph_data=graph_type)  # directed Graph
        multi_graph = nx.MultiGraph(incoming_graph_data=graph_type, multigraph_input=True)  # undirected Multigraph
        di_multi_graph = nx.MultiDiGraph(incoming_graph_data=graph_type, multigraph_input=True)  # Directed MultiGraph.

        # https://stackoverflow.com/questions/47892944/python-networkx-find-a-subgraph-in-a-directed-graph-from-a-node-as-root
        something = (graph.subgraph(c) for c in connected_components(graph))

        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
        subgraph_enumeration = [graph.subgraph(c).copy() for c in connected_components(graph)]

        for i, subgraph in enumerate(subgraph_enumeration):
            nx.write_gexf(graph, DATA_PATH + graph_type + '.gexf')
            nx.write_graphml(graph, DATA_PATH + graph_type + '.graphml')
            plt.figure()
            nx.draw(subgraph, with_labels=True)
            plt.savefig(DATA_PATH + nameof(
                graph_type) + '_' + i + '.png')  # https://stackoverflow.com/questions/1534504/convert-variable-name-to-string
            plt.close()


def read_order_management():
    inner_order_log = pm4py.read.read_ocel2_json(OM_PATH)
    return inner_order_log


def process_order_log(orderlog: OCEL):
    order_obj_graphs = discover_objects_graph(orderlog)
    object_types = pm4py.ocel.ocel_get_object_types(orderlog)
    attribute_names = pm4py.ocel.ocel_get_attribute_names(orderlog)
    object_type_activities = pm4py.ocel.ocel_object_type_activities(orderlog)

    order_petri_net = pm4py.ocel.discover_oc_petri_net(orderlog, 'im',
                                                       True)  # ("im" for traditional; "imd" for the faster inductive miner directly-follows)
    pm4py.visualization.petri_net.visualizer.apply(order_petri_net)

    # perform flattening for different algorithms only available for traditional event logs
    flat_log = pm4py.ocel.ocel_flattening(orderlog, object_types[0])


def create_pnml(filename: string, net: Tuple):
    file_ending = '.pnml'
    file_path = DATA_PATH + filename + file_ending
    pm4py.write.write_pnml(petri_net=net[0], file_path=file_path, initial_marking=net[1], final_marking=net[2])


def read_pnml(filepath: string):
    # Example of loading a Petri net (both methods do the same)

    # loaded_orders_petri_net = pnml_importer.apply('data/order_management/orders.pnml')
    # loaded_orders_petri_net2 = pm4py.read.read_pnml(file_path='data/order_management/orders.pnml')

    return pm4py.read.read_pnml(file_path=filepath)


def object_codeath_graphs(object_codeath: [set[tuple[str, str]]]):  # was executed
    graph = nx.Graph(incoming_graph_data=object_codeath)

    subgraph_enumeration = [graph.subgraph(c).copy() for c in connected_components(graph)]

    for i, subgraph in enumerate(subgraph_enumeration):
        plt.figure()
        nx.draw(subgraph, with_labels=True)
        plt.savefig(DATA_PATH + 'graphs/objects_graphs/object_codeath/' + str(i) + '.png')
        plt.close()


def _todo(log: OCEL):
    # ["orders", "items", "packages", "customers", "products", "employees"]
    orders_cluster = pm4py.ocel.cluster_equivalent_ocel(ocel=log, object_type='orders')

    ocpn = ocpn_discovery.apply(log)

    pm4py.algo.discovery.ocel.ocdfg.algorithm.apply()

    # investigate object types
    object_count_dictionary = pm4py.ocel.ocel_objects_ot_count(log)
    object_interaction_summary = pm4py.ocel.ocel_objects_interactions_summary(log)

    pm4py.filtering.filter_between(log)
    pm4py.filtering.filter_prefixes(log)  # returns the activities before the given activity
    pm4py.filtering.filter_suffixes(log)
    pm4py.filtering.filter_variants(log)

    # ("im" for traditional; "imd" for the faster inductive miner directly-follows)
    order_petri_net = pm4py.ocel.discover_oc_petri_net(log, 'im', True)
    order_petri_net2 = pm4py.ocel.discover_oc_petri_net(log, 'imd', True)
    order_petri_net3 = pm4py.ocel.discover_oc_petri_net(log, 'im', False)
    order_petri_net4 = pm4py.ocel.discover_oc_petri_net(log, 'im', False)

    pm4py.visualization.petri_net.visualizer.apply(order_petri_net)

    var = pm4py.objects.ocel.validation.jsonocel.apply()


def try_filtering(log: OCEL):
    pass


def create_process_tree(net: Tuple):
    petri_net = net[0]
    marking1 = net[1]
    marking2 = net[2]

    # ProcessTree = pm4py.convert_to_process_tree(petri_net, marking1, marking2)  # is empty. not supported?


def create_clusters(log: OCEL):
    orders_cluster = pm4py.ocel.cluster_equivalent_ocel(ocel=log, object_type='orders', max_objs=200)
    f = open(TEXT_DATA_PATH + 'orders_cluster.txt', 'x')
    f.write(orders_cluster.__str__())


def show_library_versions():
    pd.DataFrame(
        [
            ['pandas', pd.__version__],
            ['numpy ', np.__version__],
            ['matplotlib', sys.modules['matplotlib'].__version__],
            ['ipywidgets', sys.modules['ipywidgets'].__version__],
            ['pm4py', pm4py.__version__],
        ],
        columns=['package', 'version']
    ).set_index('package')


def visualize_ocdfg(log: OCEL):
    ocdfg = pm4py.ocel.discover_ocdfg(log)
    ocdfg_digraph = pm4py.visualization.ocel.ocdfg.visualizer.apply(ocdfg)
    # pm4py.visualization.ocel.ocdfg.visualizer.save(gviz=ocdfg_digraph, output_file_path=GRAPH_DATA_PATH + 'gviz/ocdfg_digraph.png')
    pm4py.visualization.ocel.ocdfg.visualizer.view(ocdfg_digraph)


def evaluate_edge_metrics(log):
    result = pm4py.statistics.ocel.edge_metrics.find_associations_per_edge(log)

    aggregate_total_objects = pm4py.statistics.ocel.edge_metrics.aggregate_total_objects(result)
    # A dictionary associating to each object type another dictionary where to each edge (activity couple)
    # all the triples (source event, target event, object) are associated.

    aggregate_unique_objects = pm4py.statistics.ocel.edge_metrics.aggregate_unique_objects(result)
    # A dictionary associating to each object type another dictionary where to each edge (activity couple) all the involved objects are associated.

    aggregate_ev_couples = pm4py.statistics.ocel.edge_metrics.aggregate_ev_couples(result)
    # A dictionary associating to each object type another dictionary where to each edge (activity couple) all the couples of related events are associated.

    performance_ev_couples = (
        pm4py.statistics.ocel.edge_metrics.performance_calculation_ocel_aggregation(log, aggregate_ev_couples))
    performance_total_objects = (
        pm4py.statistics.ocel.edge_metrics.performance_calculation_ocel_aggregation(log, aggregate_total_objects))
    # For each object type, associate a dictionary where to each activity couple all the times between the activities are recorded.


def save_to_file(data, filepath: string, name: string):
    f = open(filepath + name + '.txt', 'x')
    # if data is string:
    #     f.write(data)
    # elif data is pandas.DataFrame:
    #     data.to_csv(filepath + name + '.txt', sep='\t', index=False)
    if type(data) == string:
        f.write(data)
    elif type(data) == pandas.DataFrame:
        data.to_csv(filepath + name + '.txt', sep='\t', index=False)
    f.close()

def ocel_package_functions(log: OCEL):
    O2O_enriched_ocel = pm4py.ocel.ocel_o2o_enrichment(log, discover_objects_graph(log))
    E2O_enriched_ocel = pm4py.ocel.ocel_e2o_lifecycle_enrichment(log)

    # _TODO
    # pm4py.ocel.sample_ocel_connected_components(log)

    correctly_ordered_log = pm4py.ocel.ocel_add_index_based_timedelta(ocel=log)


def validate_log(path: string, schema: string):
    pass


if __name__ == '__main__':
    log = read_order_management()

# Validation of JSON-OCEL and XML-OCEL: using the command ocel.validate(log path, schema path), it is possible to validate an event log against
# the OCEL schema. The schemas for JSON-OCEL and XML-OCEL are available inside the folder schemas of the repository.


    validate_log(OM_PATH, )

    save_to_file(pm4py.ocel.ocel_temporal_summary(log), TEXT_DATA_PATH, 'temporal_summary')
    save_to_file(pm4py.ocel.ocel_objects_summary(log), TEXT_DATA_PATH, 'objects_summary')
    save_to_file(pm4py.ocel.ocel_objects_interactions_summary(log), TEXT_DATA_PATH, 'object_interactions_summary')

    # visualize_ocdfg(log)

    # evaluate_edge_metrics(log)

    # _todo()

    print(" ")
