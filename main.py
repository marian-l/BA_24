import string
import numpy as np
import pandas as pd
import pm4py
import matplotlib.pyplot as plt

from typing import Collection
from pm4py import OCEL

OM_PATH = 'data/order_management/order-management.json'
GRAPH_DATA_PATH = 'data/order_management/graphs/'
PETRI_NET = 'data/order_management/graphs/PetriNetModelLanguage/'
TEXT_DATA_PATH = 'data/order_management/text/'
DATA_PATH = 'data/order_management/'


def show_library_versions():
    print(pd.DataFrame(
        [
            ['pandas', pd.__version__],
            ['numpy ', np.__version__],
            ['pm4py', pm4py.__version__],
        ],
        columns=['package', 'version']
    ).set_index('package'))


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


# UTILS
def save_to_file(data, filepath: string, name: string):
    f = open(filepath + name + '.txt', 'x')
    if type(data) == string:
        f.write(data)
    elif type(data) == pd.DataFrame:
        data.to_csv(filepath + name + '.txt', sep='\t', index=False)
    f.close()


def save_temporal_summary(log):
    # done
    temporal_summary_dataframe = pm4py.ocel.ocel_temporal_summary(log)
    temporal_summary_dataframe.to_csv()
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        save_to_file(temporal_summary_dataframe, TEXT_DATA_PATH, 'temporal_summary')


def filter_ocdfg(ocdfg):
    for key, dictionary in ocdfg['edges_performance']['event_couples'].items():
        for inner_key, object_list in dictionary.items():
            dictionary[inner_key] = list(filter(lambda ol: ol != 0.0, object_list))
            if not dictionary[inner_key]:
                # dictionary[inner_key] has to be iterable for "add_performance_edge" in classic.py
                dictionary[inner_key] = [0.0]

    return ocdfg


def exercise_3_1(ocel):
    # Pre-Processing

    # Ist das Log valides OCEL?
    validate_log()

    # liegt das Log im OCEL2.0 Format vor?
    print(ocel.is_ocel20())
    save_temporal_summary(ocel)

    # Conformance Checking
    ocpn = pm4py.discover_oc_petri_net(ocel, diagnostics_with_tbr=True)
    print(get_soundness(ocpn=ocpn))
    pm4py.view_ocpn(ocpn=ocpn, rankdir='TB')

    ocdfg = pm4py.discover_ocdfg(ocel)
    pm4py.view_ocdfg(ocdfg, annotation="frequency")

    # Performance Checking
    ocel = pm4py.filter_ocel_object_types(ocel, ["products"], positive=False)

    ocdfg = pm4py.discover_ocdfg(ocel, business_hours=True, business_hour_slots=[(6 * 60 * 60, 20 * 60 * 60)])
    pm4py.view_ocdfg(ocdfg, annotation="frequency", edge_threshold=200)

    ocdfg = filter_ocdfg(ocdfg)

    pm4py.view_ocdfg(ocdfg, performance_aggregation="min", annotation="performance")
    pm4py.view_ocdfg(ocdfg, performance_aggregation="max", annotation="performance")
    pm4py.view_ocdfg(ocdfg, performance_aggregation="mean", annotation="performance")
    pm4py.view_ocdfg(ocdfg, performance_aggregation="median", annotation="performance")


def exercise_3_4(ocel):
    import json
    import pm4py
    from tabulate import tabulate

    FILE_PATH = 'data/order_management/311-2-log.jsonocel'

    place_confirm_order_log = pm4py.filter_ocel_event_attribute(ocel, "ocel:activity", ["confirm order"], positive=True)
    place_confirm_order_log = pm4py.filter_ocel_object_attribute(place_confirm_order_log, "ocel:type",
                                                                 ["item", "product", "order"], positive=False)
    pm4py.write.write_ocel2(place_confirm_order_log, FILE_PATH)

    number_of_lines = 234279
    output_file_path = 'data/order_management/311-22-log.json'

    with open(FILE_PATH, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for i, line in enumerate(input_file):
            if i > number_of_lines:
                output_file.write(line)
            else:
                continue

    employees = ['Istvan Koren', 'Mara Nitschke', 'Jan Niklas Adams', 'Christine von Dobbert', 'Wil van der Aalst']
    customers = ['Danube Pharmaceuticals BV', 'Carpathian Financial Services plc', 'AlpenTech Innovations AG',
                 'Balkan Minerals d.o.o.', 'Riviera Robotics SAS', 'Iberian Sun Solaridades SA',
                 'Baltic Wave Energies Ltd.', 'Pantheon Art Gallery Kft.', 'Vesta Fashion House GmbH',
                 'Medi-terranea Luxury Cruises S.p.A.', 'Nordica Systems GmbH', 'Celtica Green Farms Oy',
                 'Golden Fleece Textiles S.A.R.L.', 'SwissPeak Timepieces AG', 'Eastern Oak Insurance Zrt.', ]

    with open(FILE_PATH, 'r') as file:
        event_log_data = json.load(file)

    table = [['CUSTOMER', 'SALES PERSON', 'APPEARANCES']]

    for customer in customers:
        for employee in employees:
            specific_customer = customer
            specific_sales_person = employee
            combination_count = 0
            for event in event_log_data["events"]:
                customer_present = any(rel['objectId'] == specific_customer for rel in event['relationships'])
                sales_person_present = any(
                    rel['objectId'] == specific_sales_person for rel in event['relationships'])
                if customer_present and sales_person_present:
                    combination_count += 1

            if combination_count > 0:
                table.append([specific_customer, specific_sales_person, combination_count])

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))


def exercise_3_6(ocel):
    # Conformance Checking
    df = log.objects
    filtered_df = df[df['role'].notnull()]
    filtered_df.drop(axis=1, inplace=True, columns=['price', 'weight'])

    sales_df = filtered_df[filtered_df['role'] == 'Sales']
    warehouse_df = filtered_df[filtered_df['role'] == 'Warehousing']
    shipment_df = filtered_df[filtered_df['role'] == 'Shipment']

    sales_persons: Collection[str] = sales_df['ocel:oid'].astype(str).tolist()
    warehouse_persons: Collection[str] = warehouse_df['ocel:oid'].astype(str).tolist()
    shipment_persons: Collection[str] = shipment_df['ocel:oid'].astype(str).tolist()

    sales_log = pm4py.filter_ocel_objects(ocel=log, object_identifiers=sales_persons, positive=True)
    warehouse_log = pm4py.filter_ocel_objects(ocel=log, object_identifiers=warehouse_persons, positive=True)
    shipment_log = pm4py.filter_ocel_objects(ocel=log, object_identifiers=shipment_persons, positive=True)

    ocpn = pm4py.discover_oc_petri_net(sales_log, diagnostics_with_tbr=True)
    print(get_soundness(ocpn=ocpn))
    pm4py.view_ocpn(ocpn=ocpn)

    ocpn = pm4py.discover_oc_petri_net(warehouse_log, diagnostics_with_tbr=True)
    print(get_soundness(ocpn=ocpn))
    pm4py.view_ocpn(ocpn=ocpn)

    ocpn = pm4py.discover_oc_petri_net(shipment_log, diagnostics_with_tbr=True)
    print(get_soundness(ocpn=ocpn))
    pm4py.view_ocpn(ocpn=ocpn)

    pm4py.view_ocdfg(ocdfg=pm4py.discover_ocdfg(sales_log))
    pm4py.view_ocdfg(ocdfg=pm4py.discover_ocdfg(warehouse_log))
    pm4py.view_ocdfg(ocdfg=pm4py.discover_ocdfg(shipment_log))


def exercise_3_2(ocel):
    # Nach "failed delivery" filtern
    failed_delivery_log = pm4py.filter_ocel_event_attribute(ocel, attribute_key='ocel:activity', attribute_values=['failed delivery'])

    # wert des Felds 'ocel:timestamp' auf den Wert der jeweiligen Stunde reduzieren
    failed_delivery_log.events['ocel:timestamp'] = failed_delivery_log.events['ocel:timestamp'].apply(lambda x: int(x.time().hour))

    # Zu Liste umformen, um Member-Methode "Sort" nutzen zu können
    failed_delivery_times = failed_delivery_log.events['ocel:timestamp'].tolist()

    # Sortieren (alternativ auch aus dem DataFrame heraus möglich
    failed_delivery_times.sort()

    # In Histogramm visualisieren
    plt.hist(x=failed_delivery_times, bins=20)
    plt.ylabel("Häufigkeit des Werts")
    plt.xlabel("Tageszeit in Stunden")
    plt.title('Werteverteilung der Zeitpunkte gescheiterter Zustellungen ')
    plt.show()


def exercise_3_8(ocel):
    send_package_ocel = pm4py.filter_ocel_event_attribute(ocel=ocel, attribute_key="ocel:activity",
                                                          attribute_values=["send package"], positive=True)
    send_package_ocel = pm4py.filter_ocel_object_types(ocel=send_package_ocel, obj_types=['employees'])

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

    shippers_helping_shippers_list = []
    warehousers_helping_shippers_list = []

    # extract the shippers and the warehousers
    shipper_persons = set(shipper_df['ocel:oid'].astype(str).tolist())

    # categorize the entries
    for row in both_entries_list:
        if row['ocel:oid_y'] in shipper_persons:
            shippers_helping_shippers_list.append(row)
        else:
            warehousers_helping_shippers_list.append(row)

    print(len(only_shipper_list) + len(both_entries_list))
    print('len(only_shipper_list):' + str(len(only_shipper_list)))
    print('len(both_entries_list):' + str(len(both_entries_list)))

    print(len(shippers_helping_shippers_list) + len(warehousers_helping_shippers_list))
    print('len(warehousers_helping_shippers):' + str(len(warehousers_helping_shippers_list)))
    print('len(shippers_helping_shippers_list):' + str(len(shippers_helping_shippers_list)))

    # PERFORMANCE CHECKING
    # Die EventIDs enthalten die ID des jeweiligen Pakets, zum Beispiel send_p-661128, sodass wir diese nur als Substring herausfiltern müssen
    send_package_ocel = pm4py.filter_ocel_event_attribute(ocel=ocel, attribute_key="ocel:activity",
                                                          attribute_values=["send package", "package delivered",
                                                                            "failed delivery"], positive=True)

    # Liste Packages mit den Paket-Objekten der jeweiligen Kategorie füllen und ein gefiltertes Log erstellen
    packages = []
    for series in only_shipper_list:
        packages.append(series['ocel:eid'].replace('send_', ''))
    only_shipper_log = pm4py.filter_ocel_objects(ocel=send_package_ocel, object_identifiers=packages)

    packages = []
    for series in shippers_helping_shippers_list:
        packages.append(series['ocel:eid'].replace('send_', ''))
    shippers_helping_shippers_log = pm4py.filter_ocel_objects(ocel=send_package_ocel, object_identifiers=packages)

    packages = []
    for series in warehousers_helping_shippers_list:
        packages.append(series['ocel:eid'].replace('send_', ''))
    warehousers_helping_shippers_log = pm4py.filter_ocel_objects(ocel=send_package_ocel, object_identifiers=packages)

    # Zur Absicherung OCPNs mit Diagnostiken herstellen, da gefiltert wurde
    ocpn = pm4py.discover_oc_petri_net(only_shipper_log, diagnostics_with_tbr=True)
    print(get_soundness(ocpn=ocpn))
    pm4py.view_ocpn(ocpn=ocpn)

    ocpn = pm4py.discover_oc_petri_net(shippers_helping_shippers_log, diagnostics_with_tbr=True)
    print(get_soundness(ocpn=ocpn))
    pm4py.view_ocpn(ocpn=ocpn)

    ocpn = pm4py.discover_oc_petri_net(warehousers_helping_shippers_log, diagnostics_with_tbr=True)
    print(get_soundness(ocpn=ocpn))
    pm4py.view_ocpn(ocpn=ocpn)

    ocdfg = filter_ocdfg(pm4py.discover_ocdfg(only_shipper_log))
    pm4py.view_ocdfg(ocdfg=ocdfg, annotation='performance', performance_aggregation='mean')
    pm4py.view_ocdfg(ocdfg=ocdfg, annotation='performance', performance_aggregation='median')

    ocdfg = filter_ocdfg(pm4py.discover_ocdfg(shippers_helping_shippers_log))
    pm4py.view_ocdfg(ocdfg=ocdfg, annotation='performance', performance_aggregation='mean')
    pm4py.view_ocdfg(ocdfg=ocdfg, annotation='performance', performance_aggregation='median')

    ocdfg = filter_ocdfg(pm4py.discover_ocdfg(warehousers_helping_shippers_log))
    pm4py.view_ocdfg(ocdfg=ocdfg, annotation='performance', performance_aggregation='mean')
    pm4py.view_ocdfg(ocdfg=ocdfg, annotation='performance', performance_aggregation='median')

    print("")


def exercise_3_3(ocel):
    objects = ['items', 'orders', 'packages', 'products', 'employees', 'customers']

    print('Nach Objekt gefilterte OCELs: ')
    for ocel_object in objects:
        print(ocel_object)
        print(pm4py.filter_ocel_object_types(ocel=ocel, obj_types=[ocel_object], positive=True, level=1))
        print('\n')
    #
    print('Nach Startaktivität gefilterte OCELs: ')
    for ocel_object in objects:
        print(ocel_object)
        print(pm4py.filtering.filter_ocel_start_events_per_object_type(ocel, ocel_object))
        print('\n')
    #
    print('Nach Endaktivität gefilterte OCELs: ')
    for ocel_object in objects:
        print(ocel_object)
        print(pm4py.filtering.filter_ocel_end_events_per_object_type(ocel, ocel_object))
        print('\n')

    print(ocel.o2o['ocel:qualifier'].unique())

    relations = ['comprises' 'is a' 'forwarded by' 'packed by' 'shipped by' 'contains'
                 'primarySalesRep' 'secondarySalesRep' 'places']

    order_df = ocel.o2o[ocel.o2o['ocel:qualifier'] == 'comprises']
    customer_df = ocel.o2o[ocel.o2o['ocel:qualifier'] == 'places']
    package_df = ocel.o2o[ocel.o2o['ocel:qualifier'] == 'contains']
    item_df = ocel.o2o[ocel.o2o['ocel:qualifier'] == 'is a']

    result = []

    all_dfs = [order_df, customer_df, package_df
               ]
    for df in all_dfs:
        current = ''
        tempresult = []
        count = 0

        for row in df.iterrows():
            if current == '':
                current = row[1]['ocel:oid']
            if row[1]['ocel:oid'] == current:
                count += 1
            else:
                tempresult.append([current, count])
                count = 1
                current = row[1]['ocel:oid']

        if current != '' and count > 0:
            tempresult.append([current, count])

        tempresult.sort(key=lambda x: x[1], reverse=True)
        result.append(tempresult)

    # order_df
    values = [item[1] for item in result[0]]
    mean = sum(values, 0) / len(values)
    max_articles_in_order = max(values)
    min_articles_in_order = min(values)
    print('mean: ' + str(mean))
    print('min: ' + str(min_articles_in_order))
    print('max: ' + str(max_articles_in_order))

    # customer_df
    values = [item[1] for item in result[1]]
    mean = sum(values, 0) / len(values)
    max_orders_in_customer = max(values)
    min_orders_in_customer = min(values)
    print('mean: ' + str(mean))
    print('max: ' + str(max_orders_in_customer))
    print('min: ' + str(min_orders_in_customer))

    # package_df
    values = [item[1] for item in result[2]]
    mean = sum(values, 0) / len(values)
    max_article_in_packages = max(values)
    min_article_in_packages = min(values)
    print('mean: ' + str(mean))
    print('max: ' + str(max_article_in_packages))
    print('min: ' + str(min_article_in_packages))

    # order_df
    values = [item[1] for item in result[0]]
    plt.hist(values, bins=20)
    plt.ylabel("Häufigkeit des Werts")
    plt.xlabel("Anzahl an Artikeln")
    plt.title('Werteverteilung der Artikel in Bestellungen')
    plt.show()
    #
    # customer_df
    values = [item[1] for item in result[1]]
    plt.hist(values, bins=20)
    plt.ylabel("Häufigkeit des Werts")
    plt.xlabel("Bestellungen")
    plt.title('Werteverteilung der Bestellungen pro Kunde')
    plt.show()
    #
    # package_df
    values = [item[1] for item in result[2]]
    plt.hist(values, bins=20)
    plt.ylabel("Häufigkeit des Werts")
    plt.xlabel("Anzahl an Artikeln")
    plt.title('Werteverteilung der Artikel in Paketen')
    plt.show()


def exercise_3_5(ocel):
    primarySalesReps = {}
    secondarySalesReps = {}

    # nach One-Face-Policy zugeordnete Mitarbeiter auslesen
    filtered_log = pm4py.filter_ocel_object_types(ocel, obj_types=['customers', 'employees'])

    for index, o2orel in filtered_log.o2o.iterrows():
        if o2orel['ocel:qualifier'] == 'primarySalesRep':
            primarySalesReps[o2orel['ocel:oid']] = o2orel['ocel:oid_2']
        elif o2orel['ocel:qualifier'] == 'secondarySalesRep':
            secondarySalesReps[o2orel['ocel:oid']] = o2orel['ocel:oid_2']

    # Bestellungen nach One-Face-Policy kategorisieren
    filtered_log = pm4py.filter_ocel_event_attribute(ocel, "ocel:activity", ["confirm order"], positive=True)
    filtered_log = pm4py.filter_ocel_object_types(filtered_log, obj_types=['items', 'products'], positive=False)

    df = filtered_log.relations
    df.drop(axis=1, inplace=True, columns=['ocel:timestamp', 'ocel:activity'])

    order_df = df[df['ocel:type'] == 'orders']
    customer_df = df[df['ocel:type'] == 'customers']
    employee_df = df[df['ocel:type'] == 'employees']

    merged_df = pd.merge(order_df, customer_df, on='ocel:eid', how='outer')
    merged_df = pd.merge(merged_df, employee_df, on='ocel:eid', how='outer')

    orders_with_primary = []
    orders_with_secondary = []
    orders_with_other = []

    for index, row in merged_df.iterrows():
        if primarySalesReps[row['ocel:oid_y']] == row['ocel:oid']:
            orders_with_primary.append(row['ocel:oid_x'])
        elif secondarySalesReps[row['ocel:oid_y']] == row['ocel:oid']:
            orders_with_secondary.append(row['ocel:oid_x'])
        else:
            orders_with_other.append(row['ocel:oid_x'])

    orders_with_primary_log = pm4py.filter_ocel_objects(ocel, orders_with_primary)
    orders_with_secondary_log = pm4py.filter_ocel_objects(ocel, orders_with_secondary)
    orders_with_other_log = pm4py.filter_ocel_objects(ocel, orders_with_other)

    # 1754 196 50

    # PrimarySalesRep
    ocdfg = pm4py.discover_ocdfg(orders_with_primary_log, business_hours=True,
                                 business_hour_slots=[(6 * 60 * 60, 20 * 60 * 60)])
    ocdfg = filter_ocdfg(ocdfg)
    pm4py.view_ocdfg(ocdfg, performance_aggregation="mean", annotation="performance")
    #
    # SecondarySalesRep
    ocdfg = pm4py.discover_ocdfg(orders_with_secondary_log, business_hours=True,
                                 business_hour_slots=[(6 * 60 * 60, 20 * 60 * 60)])
    ocdfg = filter_ocdfg(ocdfg)
    pm4py.view_ocdfg(ocdfg, performance_aggregation="mean", annotation="performance")
    #
    # OtherSalesRep
    ocdfg = pm4py.discover_ocdfg(orders_with_other_log, business_hours=True,
                                 business_hour_slots=[(6 * 60 * 60, 20 * 60 * 60)])
    ocdfg = filter_ocdfg(ocdfg)
    pm4py.view_ocdfg(ocdfg, performance_aggregation="mean", annotation="performance")


def exercise_3_5_2(ocel):
    primarySalesReps = {}
    secondarySalesReps = {}

    # nach One-Face-Policy zugeordnete Mitarbeiter auslesen
    filtered_log = pm4py.filter_ocel_object_types(ocel, obj_types=['customers', 'employees'])

    for index, o2orel in filtered_log.o2o.iterrows():
        if o2orel['ocel:qualifier'] == 'primarySalesRep':
            primarySalesReps[o2orel['ocel:oid']] = o2orel['ocel:oid_2']
        elif o2orel['ocel:qualifier'] == 'secondarySalesRep':
            secondarySalesReps[o2orel['ocel:oid']] = o2orel['ocel:oid_2']

    # Bestellungen nach One-Face-Policy kategorisieren
    filtered_log = pm4py.filter_ocel_event_attribute(ocel, "ocel:activity", ["confirm order"], positive=True)
    filtered_log = pm4py.filter_ocel_object_types(filtered_log, obj_types=['items', 'products'], positive=False)

    df = filtered_log.relations
    df.drop(axis=1, inplace=True, columns=['ocel:timestamp', 'ocel:activity'])

    order_df = df[df['ocel:type'] == 'orders']
    customer_df = df[df['ocel:type'] == 'customers']
    employee_df = df[df['ocel:type'] == 'employees']

    merged_df = pd.merge(order_df, customer_df, on='ocel:eid', how='outer')
    merged_df = pd.merge(merged_df, employee_df, on='ocel:eid', how='outer')

    orders_with_primary = []
    orders_with_secondary = []
    orders_with_other = []

    for index, row in merged_df.iterrows():
        if primarySalesReps[row['ocel:oid_y']] == row['ocel:oid']:
            orders_with_primary.append(row['ocel:oid_x'])
        elif secondarySalesReps[row['ocel:oid_y']] == row['ocel:oid']:
            orders_with_secondary.append(row['ocel:oid_x'])
        else:
            orders_with_other.append(row['ocel:oid_x'])

    filtered_log = pm4py.filter_ocel_object_types(ocel=ocel, obj_types=['items', 'orders'])

    lists_of_orders = [orders_with_primary, orders_with_secondary, orders_with_other]
    articles_per_order = []
    result = []

    # Um Zeilen zu sparen, loopen wir über alle drei Listen
    for orderlist in lists_of_orders:

        # Über jede Bestellung in der Liste
        for order in orderlist:

            # speichern die Artikel jeder Bestellung
            orders_articles = []

            # Vergleichen mit den Objekt-Objekt-Beziehungen um die Artikel zu jeder Bestellung zu finden
            for index, o2orel in filtered_log.o2o.iterrows():

                # Wenn die Bestellungsnummer größer oder kleiner ist als die aktuell im Scope, kann der nächste
                # Zyklus begonnen werden, da die Bestellungen sortiert vorliegen
                if (o2orel['ocel:oid'] > order) or (o2orel['ocel:oid'] < order):
                    continue

                # Die Bestellung, die aktuell analysiert wird
                elif o2orel['ocel:oid'] == order:

                    # Artikel aus der Bestellung wird zwischengespeichert
                    orders_articles.append([o2orel['ocel:oid_2']])

            # Im Zwischenergebnis sämtliche Artikel der Bestellungen der Liste speichern
            articles_per_order.append(orders_articles)

        article_count = 0
        for order in articles_per_order:
            article_count += len(order)

        average_articles_per_order = article_count / len(articles_per_order)
        result.append(' | durchschnittliche Artikel pro Bestellung : ' + str(average_articles_per_order))

    # [' | durchschnittliche Artikel pro Bestellung : 3.8363740022805017', ' | durchschnittliche Artikel pro Bestellung : 3.835384615384615', ' | durchschnittliche Artikel pro Bestellung : 3.8295'
    print(result)


if __name__ == '__main__':
    log = pm4py.read.read_ocel2_json(OM_PATH)

    # creates artificial O2O relationships based on the object interaction, descendants, inheritance, cobirth, codeath graph
    # log = pm4py.ocel_o2o_enrichment(log)

    # test = pm4py.discover_objects_graph()
#
    # # prints the O2O table
    # print(log.o2o)
#
    # object_descendants_graph = []
    # object_inheritance_graph = []
    # object_codeath_graph = []
    # object_interaction_graph = []
    # object_cobirth_graph = []
#
    # # object_descendants_graph object_inheritance_graph object_codeath_graph object_interaction_graph object_inheritance_graph
    # for index, row in log.o2o.iterrows():
    #     if row['ocel:qualifier'] == 'object_descendants_graph':
    #         object_descendants_graph.append(row)
    #     elif row['ocel:qualifier'] == 'object_inheritance_graph':
    #         object_inheritance_graph.append(row)
    #     elif row['ocel:qualifier'] == 'object_codeath_graph':
    #         object_codeath_graph.append(row)
    #     elif row['ocel:qualifier'] == 'object_interaction_graph':
    #         object_interaction_graph.append(row)
    #     elif row['ocel:qualifier'] == 'object_cobirth_graph':
    #         object_cobirth_graph.append(row)



    # sample_log = pm4py.sample_ocel_objects(log, 1000)
    # clusters = pm4py.cluster_equivalent_ocel(ocel=sample_log, object_type='packages')

    exercise_3_2(log)

    # exercise_3_1(log)
    # exercise_3_2(log)
    # exercise_3_3(log)
    # exercise_3_4(log)
    # exercise_3_5(log)
    # exercise_3_6(log)
    # exercise_3_7(log)
    # exercise_3_8(log)
    # exercise_3_8_2(log)

    # -----------------------------------------------------------------------------------------

    print("--------------------------------------------------------")

# 20s + 14m (create package) + ((0ns + 1m) || (25s)) + ((22m + 2m) || 1min) = 39min 20s || 15min 45s
# 20s + 1D + 24m

def eof():
    pass