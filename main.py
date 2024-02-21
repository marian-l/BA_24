import _collections_abc
import collections
import os
import string
import sys
from array import array
from typing import Tuple, Collection

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


def validate_log():
    return pm4py.objects.ocel.validation.jsonocel.apply(OM_PATH, 'data/schema.json')


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


def get_variety_of_ocdfg(ocdfg):
    ocdfg = pm4py.discover_ocdfg(log, business_hours=True, business_hour_slots=[(6 * 60 * 60, 20 * 60 * 60)])

    thresholds = [0, 10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1500, 2000]
    edge_metrics = ['event_couples', 'unique_objects', 'total_objects']
    act_metrics = ['events', 'unique_objects', 'total_objects']
    annotations = ['frequency', 'performance']
    rankdirs = ['LR', 'TB']
    bgcolors = ['white', 'black']
    performance_aggregations = ['mean', 'median', 'min', 'max', 'sum']

    for threshold in thresholds:
        for performance_aggregation in performance_aggregations:
            for annotation in annotations:
                pm4py.save_vis_ocdfg(

                    ocdfg=ocdfg,
                    file_path=GRAPH_DATA_PATH + '/DFG/ganzes Log/' + str(
                        threshold) + ' ' + performance_aggregation + ' ' + annotation + '.jpeg',
                    act_threshold=threshold,
                    edge_threshold=threshold,
                    performance_aggregation=performance_aggregation,
                    annotation=annotation,
                    # rankdir=rankdir,
                    # act_metric=act_metric,
                    # edge_metric=edge_metric,
                    # bgcolor=bgcolor

                )

    def get_ocpn_and_subnets(ocel: OCEL, name: str):
        ocpn = pm4py.discover_oc_petri_net(ocel, diagnostics_with_tbr=True)

        pm4py.save_vis_ocpn(ocpn, file_path=GRAPH_DATA_PATH + '/OCPN/ocpn_diagnostics.jpeg')

        petri_net_list = ocpn['petri_nets']
        for petri_net in petri_net_list:
            im = pm4py.objects.petri_net.utils.initial_marking.discover_initial_marking(
                ocpn['petri_nets'][petri_net][0])
            fm = pm4py.objects.petri_net.utils.final_marking.discover_final_marking(ocpn['petri_nets'][petri_net][0])

            pm4py.save_vis_petri_net(petri_net=ocpn['petri_nets'][petri_net][0], initial_marking=im, final_marking=fm,
                                     file_path=GRAPH_DATA_PATH + '/OCPN/' + petri_net + ' ' + name + '.jpeg')


def exercise_323(log):
    df = log.objects
    filtered_df = df[df['role'].notnull()]
    filtered_df.drop(axis=1, inplace=True, columns=['price', 'weight'])

    sales_df = filtered_df[filtered_df['role'] == 'Sales']
    warehouse_df = filtered_df[filtered_df['role'] == 'Warehousing']
    shipment_df = filtered_df[filtered_df['role'] == 'Shipment']

    sales_persons: Collection[str] = sales_df['ocel:oid'].astype(str).tolist()
    warehouse_persons: Collection[str] = warehouse_df['ocel:oid'].astype(str).tolist()
    shipment_persons: Collection[str] = shipment_df['ocel:oid'].astype(str).tolist()

    sales_log = pm4py.filter_ocel_objects(ocel=log, object_identifiers=sales_persons, positive=True, level=1)
    # Object-Centric Event Log (number of events: 2000, number of objects: 5, number of activities: 1, number of object types: 1, events-objects relationships: 2000)
    # Activities occurrences: {'confirm order': 2000}
    # Object types occurrences (number of objects): {'employees': 5}

    # sales_log = pm4py.filter_ocel_objects(ocel=log, object_identifiers=sales_persons, positive=True, level=2)
    # Object-Centric Event Log (number of events: 21008, number of objects: 9699, number of activities: 11, number of object types: 5, events-objects relationships: 128198)
    # Activities occurrences: {'pick item': 7659, 'place order': 2000, 'confirm order': 2000, 'pay order': 2000, 'item out of stock': 1544, 'reorder item': 1544, 'create package': 1128, 'send package': 1128, 'package delivered': 1128, 'payment reminder': 566, 'failed delivery': 311}
    # Object types occurrences (number of objects): {'items': 7659, 'orders': 2000, 'products': 20, 'customers': 15, 'employees': 5}

    # sales_log = pm4py.filter_ocel_objects(ocel=log, object_identifiers=sales_persons, positive=True, level=3)
    # Object-Centric Event Log (number of events: 21008, number of objects: 10840, number of activities: 11, number of object types: 6, events-objects relationships: 147385)
    # Activities occurrences: {'pick item': 7659, 'place order': 2000, 'confirm order': 2000, 'pay order': 2000, 'item out of stock': 1544, 'reorder item': 1544, 'create package': 1128, 'send package': 1128, 'package delivered': 1128, 'payment reminder': 566, 'failed delivery': 311}
    # Object types occurrences (number of objects): {'items': 7659, 'orders': 2000, 'packages': 1128, 'products': 20, 'employees': 18, 'customers': 15}

    warehouse_log = pm4py.filter_ocel_objects(log, object_identifiers=warehouse_persons, positive=True)
    shipment_log = pm4py.filter_ocel_objects(log, object_identifiers=shipment_persons, positive=True)

    pm4py.view_ocpn(ocpn=pm4py.discover_oc_petri_net(sales_log))
    pm4py.view_ocpn(ocpn=pm4py.discover_oc_petri_net(warehouse_log))
    pm4py.view_ocpn(ocpn=pm4py.discover_oc_petri_net(shipment_log))
    # pm4py.view_ocdfg(ocdfg=pm4py.discover_ocdfg(sales_log), annotation="performance", performance_aggregation="mean")
    pm4py.view_ocdfg(ocdfg=pm4py.discover_ocdfg(sales_log))
    pm4py.view_ocdfg(pm4py.discover_ocdfg(warehouse_log))
    pm4py.view_ocdfg(pm4py.discover_ocdfg(shipment_log))


def get_deviations_from_aufgabenteilung(ocel):
    send_package_ocel = pm4py.filter_ocel_event_attribute(ocel, "ocel:activity", ["send package"], positive=True)
    send_package_ocel = pm4py.filter_ocel_object_types(send_package_ocel, ['employees'])

    df = send_package_ocel.relations
    shipper_df = df[df['ocel:qualifier'] == 'shipper']
    forwarder_df = df[df['ocel:qualifier'] == 'forwarder']

    shipper_df = shipper_df.drop(['ocel:timestamp', 'ocel:type', 'ocel:activity'], axis=1)
    forwarder_df = forwarder_df.drop(['ocel:timestamp', 'ocel:type', 'ocel:activity'], axis=1)

    merged_df = pd.merge(shipper_df, forwarder_df, on='ocel:eid', how='outer')

    only_shipper_list = []
    both_entries_list = []

    # categorize the entries
    for index, row in merged_df.iterrows():
        if (row['ocel:oid_x'] is pd.NA) or (row['ocel:oid_y'] is pd.NA):
            only_shipper_list.append(row)
        else:
            both_entries_list.append(row)

    shippers_helping_shippers = []
    warehousers_helping_shippers = []

    # extract the shippers and the warehousers
    shipper_persons = set(shipper_df['ocel:oid'].astype(str).tolist())

    # categorize the entries
    for row in both_entries_list:
        if row['ocel:oid_y'] in shipper_persons:
            shippers_helping_shippers.append(row)
        else:
            warehousers_helping_shippers.append(row)

    print(len(only_shipper_list) + len(both_entries_list))
    print('len(only_shipper_list):' + str(len(only_shipper_list)))
    print('len(both_entries_list):' + str(len(both_entries_list)))

    print(len(shippers_helping_shippers) + len(warehousers_helping_shippers))
    print('len(warehousers_helping_shippers):' + str(len(warehousers_helping_shippers)))
    print('len(shippers_helping_shippers):' + str(len(shippers_helping_shippers)))


def get_object_lifecycle_and_relations(log):
    objects = ['items', 'orders', 'packages', 'products', 'employees', 'customers']

    # print('Nach Objekt gefilterte OCELs: ')
    # for ocel_object in objects:
    #    print(ocel_object)
    #    print(pm4py.filter_ocel_object_types(ocel=log, obj_types=[ocel_object], positive=True, level=1))
    #    print('\n')
    #
    # print('Nach Startaktivität gefilterte OCELs: ')
    # for ocel_object in objects:
    #    print(ocel_object)
    #    print(pm4py.filtering.filter_ocel_start_events_per_object_type(log, ocel_object))
    #    print('\n')
    #
    # print('Nach Endaktivität gefilterte OCELs: ')
    # for ocel_object in objects:
    #    print(ocel_object)
    #    print(pm4py.filtering.filter_ocel_end_events_per_object_type(log, ocel_object))
    #    print('\n')

    # temp_ocel = pm4py.sample_ocel_objects(log, 500)
    # i=0

    # for ocel_object in objects:
    #     cluster = pm4py.ocel.cluster_equivalent_ocel(ocel=log, object_type=ocel_object, max_objs=30)
    #     for key in cluster:
    #         ocel_collection = cluster[key]
    #         i+=1
    #         for ocel in ocel_collection:
    #             pm4py.write_ocel2(ocel=ocel, file_path='data/order_management/clusters/order-management-'+str(i) + '-' + ocel_object +'.jsonocel')

    # for i in range(0, 5, 1):
    #     sample_cc_ocels.append(pm4py.sample_ocel_connected_components(log))

    print("")
    # pm4py.sample_ocel_objects()
    # pm4py.filter_ocel_cc_otype()
    # pm4py.filter_ocel_cc_object()
    # pm4py.filter_ocel_cc_length()
    # pm4py.filter_ocel_cc_activity()

    # pm4py.filter_ocel_object_types_allowed_activities()
    # pm4py.filter_ocel_object_types()
    # pm4py.filter_ocel_end_events_per_object_type()
    # pm4py.filter_ocel_object_per_type_count()
    # pm4py.filter_ocel_events_timestamp()
    # pm4py.filter_ocel_cc_otype()
    # pm4py.filter_ocel_cc_object()
    # pm4py.filter_ocel_cc_length()
    # pm4py.filter_ocel_cc_activity()
    # pm4py.filter_ocel_event_attribute()
    # pm4py.filter_ocel_object_attribute()
    # pm4py.filter_ocel_objects()


from pm4py.algo.transformation.ocel.split_ocel import algorithm as split_ocel

if __name__ == '__main__':
    log = pm4py.read.read_ocel2_json(OM_PATH)

    print(log.o2o['ocel:qualifier'].unique())

    var = ['comprises' 'is a' 'forwarded by' 'packed by' 'shipped by' 'contains'
           'primarySalesRep' 'secondarySalesRep' 'places']

    order_df = log.o2o[log.o2o['ocel:qualifier'] == 'comprises']
    customer_df = log.o2o[log.o2o['ocel:qualifier'] == 'places']
    package_df = log.o2o[log.o2o['ocel:qualifier'] == 'contains']
    item_df = log.o2o[log.o2o['ocel:qualifier'] == 'is a']

    current = ''
    result = []
    count = 0

    for row in order_df.iterrows():
        if current == '':
            current = row[1]['ocel:oid']
        if row[1]['ocel:oid'] == current:
            count += 1
        else:
            result.append([current, count])
            count = 1
            current = row[1]['ocel:oid']

    result.sort(key=lambda x: x[1], reverse=True)

    for i in range (0, 20, 1):
        print(result[i])

    for i in range(-1, -21, -1):
        print(result[i])

    # lst_ocels = split_ocel.apply(log, variant=split_ocel.Variants.ANCESTORS_DESCENDANTS,parameters={"object_type": "orders"})
    # get_object_lifecycle_and_relations(log)

    print("--------------------------------------------------------")

    # kein Vorher/Nachher-Unterschied?
    # log = pm4py.ocel.ocel_e2o_lifecycle_enrichment(log)

    # pm4py.view_ocdfg(pm4py.discover_ocdfg(send_package_ocel))
    # print(send_package_ocel)

    # pay_order_ocel = pm4py.filter_ocel_event_attribute(log, "ocel:activity", ["pay order"], positive=True)
    # payment_reminder_ocel = pm4py.filter_ocel_event_attribute(log, "ocel:activity", ["payment reminder"], positive=True)
    # pm4py.view_ocdfg(pm4py.discover_ocdfg(payment_reminder_ocel))
    # pm4py.view_ocdfg(pm4py.discover_ocdfg(pay_order_ocel))
    # print(payment_reminder_ocel)
    # print(pay_order_ocel)

    # create_process_tree(ocpn)

    # pm4py.view_ocpn(ocpn)

    # process_order_log(log) # llm_methods(log) # get_bpmn(log) # visualize_ocdfg(log) # evaluate_edge_metrics(log) # _todo()
