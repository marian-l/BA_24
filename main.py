import string
import sys
from typing import Tuple

import numpy as np
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


def process_order_log(orderlog: OCEL):
    # order_obj_graphs = discover_objects_graph(orderlog)
    # object_types = pm4py.ocel.ocel_get_object_types(orderlog)
    # attribute_names = pm4py.ocel.ocel_get_attribute_names(orderlog)
    # object_type_activities = pm4py.ocel.ocel_object_type_activities(orderlog)

    # ("im" for traditional; "imd" for the faster inductive miner directly-follows)
    order_petri_net = pm4py.ocel.discover_oc_petri_net(orderlog, 'imd', True)
    save_to_file(order_petri_net, TEXT_DATA_PATH, "whole_log_im_miner")
    order_petri_net = [(k, v) for k, v in order_petri_net.items()]
    save_to_file(order_petri_net, TEXT_DATA_PATH, "whole_log_im_miner_2")

    create_pnml("whole_log_imd_miner", order_petri_net)
    # pm4py.visualization.petri_net.visualizer.apply(order_petri_net)

    order_petri_net = pm4py.ocel.discover_oc_petri_net(orderlog, 'im', True)

    # pm4py.visualization.petri_net.visualizer.apply(order_petri_net)

    # perform flattening for different algorithms only available for traditional event logs
    # flat_log = pm4py.ocel.ocel_flattening(orderlog, object_types[0])


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


def try_filtering(log: OCEL):
    pass


def create_process_tree(ocpn):
    petri_net_list = ocpn['petri_nets']

    for petri_net in petri_net_list:
        im = pm4py.objects.petri_net.utils.initial_marking.discover_initial_marking(ocpn['petri_nets'][petri_net][0])
        fm = pm4py.objects.petri_net.utils.final_marking.discover_final_marking(ocpn['petri_nets'][petri_net][0])

    process_tree = pm4py.convert_to_process_tree((petri_net, im, fm))  # is empty. not supported?
    print(process_tree)

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
    if type(data) == str:
        f.write(data)
    elif type(data) == pd.DataFrame:
        data.to_csv(filepath + name + '.txt', sep='\t', index=False)
    elif type(data) == list:
        for item in data:
            for value in item:
                f.write(f'{value}\n')
    elif type(data) == tuple:
        for key, value in data.items():
            f.write(f'{key}: {value}\n')
    f.close()


def validate_log():
    return pm4py.objects.ocel.validation.jsonocel.apply(OM_PATH, 'data/schema.json')


def llm_methods(log):
    print(pm4py.llm.abstract_ocel_ocdfg(log, include_header=True, include_timestamps=True, max_len=10000))

    print(pm4py.llm.abstract_ocel(log))

    print(pm4py.llm.abstract_ocel_features(log))

    # print(pm4py.llm.abstract_event_stream(log))


def get_bpmn(log):
    pt = pm4py.discovery.discover_process_tree_inductive(log)
    bpmn = pm4py.convert.convert_to_bpmn(pt)


def get_log_info(log):
    print("log.objects: " + str(log.objects))
    print("log.e2e: " + str(log.e2e))
    print("log.changed_field: " + str(log.changed_field))
    print("log.event_activity: " + str(log.event_activity))
    print("log.event_id_column: " + str(log.event_id_column))
    print("log.event_timestamp: " + str(log.event_timestamp))
    print("log.events: " + str(log.events))
    print("log.globals: " + str(log.globals))
    print("log.o2o: " + str(log.o2o))
    print("log.object_changes: " + str(log.object_changes))
    print("log.object_id_column: " + str(log.object_id_column))
    print("log.object_type_column: " + str(log.object_type_column))
    print("log.parameters: " + str(log.parameters))
    print("log.qualifier: " + str(log.qualifier))
    print("log.relations: " + str(log.relations))
    print("log.get_extended_table: " + str(log.get_extended_table()))
    print("log.get_summary: " + str(log.get_summary()))
    print("log.is_ocel20: " + str(log.is_ocel20()))


def save_temporal_summary(log):
    # done
    temporal_summary_dataframe = pm4py.ocel.ocel_temporal_summary(log)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        save_to_file(temporal_summary_dataframe, TEXT_DATA_PATH, 'temporal_summary')

def get_all_keys(d):
    for key, value in ocpn.items():
        yield key
        if isinstance(value, dict):
            yield from get_all_keys(value)



def get_soundness(ocpn):
    soundness = []

    petri_net_list = ocpn['petri_nets']

    for petri_net in petri_net_list:
        im = pm4py.objects.petri_net.utils.initial_marking.discover_initial_marking(ocpn['petri_nets'][petri_net][0])
        fm = pm4py.objects.petri_net.utils.final_marking.discover_final_marking(ocpn['petri_nets'][petri_net][0])

        sound = (pm4py.analysis.check_soundness(ocpn['petri_nets'][petri_net][0], im, fm))

        diagnostics_list = []
        diagnostics = []

        for val in sound[1].values():
            diagnostics.append(val)

        diagnostics_list.append([petri_net, diagnostics[9]])
        soundness.append(diagnostics_list)

    return soundness


if __name__ == '__main__':
    log = pm4py.read.read_ocel2_json(OM_PATH)

    place_confirm_order_log = pm4py.filter_ocel_event_attribute(log, "ocel:activity", ["place order"], positive=True)
    ocpn = pm4py.discover_oc_petri_net(place_confirm_order_log, diagnostics_with_tbr=True)

    soundness = get_soundness(ocpn)

    print(soundness)


    # ocpn = pm4py.discover_oc_petri_net(log, diagnostics_with_tbr=True)
    # pm4py.view_ocpn(ocpn)
    # pm4py.save_vis_petri_net(petri_net=ocpn, initial_marking=ocpn['initial marking'])

    # process_order_log(log) # llm_methods(log) # get_bpmn(log) # visualize_ocdfg(log) # evaluate_edge_metrics(log) # _todo()

    print(" ")
