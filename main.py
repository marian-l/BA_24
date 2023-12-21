import json
import string
from typing import Tuple

import pandas as pd
import pm4py

from pm4py import OCEL
from pm4py.algo.discovery.ocel.ocpn import algorithm as ocpn_discovery
import networkx as nx
import matplotlib.pyplot as plt

import customJsonSetEncoder

OM_PATH = 'data/order_management/order-management.json'
DATA_PATH = 'data/order_management/'


def discover_objects_graph(log: OCEL):
    obj_interaction = pm4py.discover_objects_graph(log, graph_type="object_interaction")
    obj_descendants = pm4py.discover_objects_graph(log, graph_type="object_descendants")
    obj_inheritance = pm4py.discover_objects_graph(log, graph_type="object_inheritance")
    obj_cobirth = pm4py.discover_objects_graph(log, graph_type="object_cobirth")
    obj_codeath = pm4py.discover_objects_graph(log, graph_type="object_codeath")

    object_graphs = {
        'object interaction': obj_interaction,
        'object descendants': obj_descendants,
        'object_inheritance': obj_inheritance,
        'object_cobirth': obj_cobirth,
        'object_codeath': obj_codeath
    }

    # save_object_graph(object_graphs)
    return object_graphs


def save_object_graph(object_graphs: dict):
    graph_types = ['object_interaction',
                   'object_descendants',
                   'object_inheritance',
                   'object_cobirth',
                   'object_codeath']

    for graph_type in graph_types:
        obj_interaction = pm4py.discover_objects_graph(order_log, graph_type=graph_type)
        graph = nx.Graph(incoming_graph_data=obj_interaction)

        nx.write_gexf(graph, DATA_PATH+graph_type+'.gexf')
        nx.write_graphml(graph, DATA_PATH+graph_type+'.graphml')
        nx.draw(graph, with_labels=True)
        plt.savefig(DATA_PATH+graph_type+'.png')


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

    # investigate object types
    object_count_dictionary = pm4py.ocel.ocel_objects_ot_count(orderlog)
    object_interaction_summary = pm4py.ocel.ocel_objects_interactions_summary(orderlog)


def create_pnml(filename: string, net: Tuple):
    file_ending = '.pnml'
    file_path = DATA_PATH + filename + file_ending
    pm4py.write.write_pnml(petri_net=net[0], file_path=file_path, initial_marking=net[1], final_marking=net[2])


def read_pnml(filepath: string):
    # Example of loading a Petri net (both methods do the same

    # loaded_orders_petri_net = pnml_importer.apply('data/order_management/orders.pnml')
    # loaded_orders_petri_net2 = pm4py.read.read_pnml(file_path='data/order_management/orders.pnml')

    return pm4py.read.read_pnml(file_path=filepath)

def todo(log: OCEL):
    # ["orders", "items", "packages", "customers", "products", "employees"]
    orders_cluster = pm4py.ocel.cluster_equivalent_ocel(ocel=order_log, object_type='orders')

    ocpn = ocpn_discovery.apply(log)

    pm4py.algo.discovery.ocel.ocdfg.algorithm.apply()


def try_filtering(log: OCEL):
    pm4py.filtering.filter_between(log)
    pm4py.filtering.filter_prefixes(log) # returns the activities before the given activity
    pm4py.filtering.filter_suffixes(log)
    pm4py.filtering.filter_variants(log)


if __name__ == '__main__':
    order_log = read_order_management()

    # print(order_log.get_summary())
    # print('\n')
    # print(order_log.get_extended_table())

    file = open(DATA_PATH+'LogAsDataframe.txt', 'a')
    file.write(order_log.get_extended_table().to_string())


    # try_filtering(order_log)

    print(" ")
