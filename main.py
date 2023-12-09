# This is a sample Python script.
import json
import pandas as pd
import pm4py
import numpy as np
from bokeh.plotting import figure, save, gridplot, output_file, output_notebook
import sklearn
import glob
import os

from pm4py import OCEL

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

    with open('data/order_management/object_types.txt', 'w') as description_file:
        description_file.write(json.dumps(object_types))

    with open('data/order_management/attribute_names.txt', 'w') as description_file:
        description_file.write(json.dumps(attribute_names))

    with open('data/order_management/order_obj_graphs.txt', 'w') as description_file:
        description_file.write(json.dumps(order_obj_graphs, cls=customJsonSetEncoder.CustomJsonSetEncoder))

    with open('data/order_management/object_type_activities.txt', 'w') as description_file:
        description_file.write(json.dumps(object_type_activities, cls=customJsonSetEncoder.CustomJsonSetEncoder))

    # perform flattening for different algorithms only available for traditional event logs
    flat_log = pm4py.ocel.ocel_flattening(orderlog, object_types[0])

    # investigate object types
    object_count_dictionary = pm4py.ocel.ocel_objects_ot_count(orderlog)
    object_interaction_summary = pm4py.ocel.ocel_objects_interactions_summary(orderlog)


if __name__ == '__main__':
    order_log = read_order_management()
    process_order_log(order_log)
