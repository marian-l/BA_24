# This is a sample Python script.
import json
import string

import pandas
import pandas as pd
import pm4py
import numpy
from bokeh.plotting import figure, save, gridplot, output_file, output_notebook
import sklearn

import sqlite3
import glob
import os

from pm4py import OCEL

OM_PATH = 'data/order_management/order-management.json'

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def prepare_data():
    eventTypeTableFilenames = [fn for fn in glob.glob('event_*.csv')
                               if not fn == "event_map_type.csv" and not fn == "event_object.csv"]
    objectTypeTableFilenames = [fn for fn in glob.glob('object_*.csv')
                                if not fn == "object_map_type.csv" and not fn == "object_object.csv"]

    TABLES = dict()

    TABLES["event"] = pd.read_csv("event.csv", sep=";")
    TABLES["event_map_type"] = pd.read_csv("event_map_type.csv", sep=";")
    TABLES["event_object"] = pd.read_csv("event_object.csv", sep=";")
    TABLES["object"] = pd.read_csv("object.csv", sep=";")
    TABLES["object_object"] = pd.read_csv("object_object.csv", sep=";")
    TABLES["object_map_type"] = pd.read_csv("object_map_type.csv", sep=";")

    for fn in eventTypeTableFilenames:
        table_name = fn.split(".")[0]
        table = pd.read_csv(fn, sep=";")
        TABLES[table_name] = table

    for fn in objectTypeTableFilenames:
        table_name = fn.split(".")[0]
        table = pd.read_csv(fn, sep=";")
        TABLES[table_name] = table

    sql_path = "data/order_management/order-management.sqlite"
    if os.path.exists(sql_path):
        os.remove(sql_path)

    conn = sqlite3.connect(sql_path)
    for tn, df in TABLES.items():
        df.to_sql(tn, conn, index=False)
    conn.close()


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


def read_p2p(path: string):
    p2p_log = pm4py.read_ocel2_sqlite(path)
    return p2p_log


def read_angular(path: string):
    angular_log = pm4py.read_ocel(path)
    return angular_log


def read_container_logistics(path: string):
    container_log = pm4py.read_ocel_json(path)
    return container_log


def read_order_management():
    order_log = pm4py.read.read_ocel2(OM_PATH)
    return order_log


def process_p2p(p2p_log):
    p2p_obj_graphs = discover_objects_graph(p2p_log)
    object_types = pm4py.ocel.ocel_get_object_types(p2p_log)
    attribute_names = pm4py.ocel.ocel_get_attribute_names(p2p_log)
    object_type_activities = pm4py.ocel.ocel_object_type_activities(p2p_log)

    with open('data/order_management/description.txt', 'w') as description_file:
        description_file.write(json.dumps(object_type_activities))

    # perform flattening for different algorithms only available for traditional event logs
    flat_log = pm4py.ocel.ocel_flattening(p2p_log)

    # investigate object types
    object_count_dictionary = pm4py.ocel.ocel_objects_ot_count(p2p_log)
    object_interaction_summary = pm4py.ocel.ocel_objects_interactions_summary(p2p_log)

    #


def read_p2p_sqlite():
    p2p_log = read_p2p("data/p2p/ocel2-p2p.sqlite")
    return p2p_log


def process_order_log(orderlog: OCEL):
    order_obj_graphs = discover_objects_graph(orderlog)
    object_types = pm4py.ocel.ocel_get_object_types(orderlog)
    attribute_names = pm4py.ocel.ocel_get_attribute_names(orderlog)
    object_type_activities = pm4py.ocel.ocel_object_type_activities(orderlog)

    with open('data/order_management/description.txt', 'w') as description_file:
        description_file.write(json.dumps(object_types))

    with open('data/order_management/description.txt', 'w') as description_file:
        description_file.write(json.dumps(attribute_names))

    # perform flattening for different algorithms only available for traditional event logs
    flat_log = pm4py.ocel.ocel_flattening(orderlog, object_types[0])



    # investigate object types
    object_count_dictionary = pm4py.ocel.ocel_objects_ot_count(orderlog)
    object_interaction_summary = pm4py.ocel.ocel_objects_interactions_summary(orderlog)


if __name__ == '__main__':
    order_log = read_order_management()
    process_order_log(order_log)
