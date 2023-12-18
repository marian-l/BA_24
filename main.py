# This is a sample Python script.
import json
import string
from typing import Tuple

import pandas as pd
import pm4py
import numpy as np
from bokeh.plotting import figure, save, gridplot, output_file, output_notebook
import sklearn
import glob
import os

from pm4py import OCEL

from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.petri_net.importer import importer as pnml_importer

import customJsonSetEncoder

OM_PATH = 'data/order_management/order-management.json'


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

    return object_graphs


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
    folder = 'data/order_management/'
    file_ending = '.pnml'
    file_path = folder + filename + file_ending
    pm4py.write.write_pnml(petri_net=net[0], file_path=file_path, initial_marking=net[1], final_marking=net[2])

def read_pnml(filepath: string):
    # Example of loading a Petri net (both methods do the same

    # loaded_orders_petri_net = pnml_importer.apply('data/order_management/orders.pnml')
    # loaded_orders_petri_net2 = pm4py.read.read_pnml(file_path='data/order_management/orders.pnml')

    return pm4py.read.read_pnml(file_path=filepath)

if __name__ == '__main__':
    order_log = read_order_management()

    order_petri_net = pm4py.ocel.discover_oc_petri_net(order_log)  # ("im" for traditional; "imd" for the faster
    # inductive miner directly-follows)

    create_pnml('packages', order_petri_net['petri_nets']['packages'])
    create_pnml('items', order_petri_net['petri_nets']['items'])
    create_pnml('products', order_petri_net['petri_nets']['products'])
    create_pnml('customers', order_petri_net['petri_nets']['customers'])
    create_pnml('employees', order_petri_net['petri_nets']['employees'])

    # gviz_graph = pm4py.visualization.petri_net.visualizer.apply(orders_petri_net[0])
    # pm4py.visualization.ocel.ocpn.visualizer.save(gviz_graph, 'data/order_management/OM_Petri_Net.gviz')
    # pm4py.visualization.ocel.ocpn.visualizer.view(gviz_graph)

    # order_DFG = pm4py.ocel.discover_ocdfg(order_log)
    # process_order_log(order_log)
