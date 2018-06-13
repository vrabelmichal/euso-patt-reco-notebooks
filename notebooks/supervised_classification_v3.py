import sys
import os
import subprocess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import argparse
import getpass
import collections
import numpy as np
import psycopg2 as pg
import array
#import pandas as pd
#import pandas.io.sql as psql
# import matplotlib as mpl
# mpl.use("Agg")
# import matplotlib.pyplot as plt
import hashlib
import pickle
import json

import pandas as pd
import pandas.io.sql as psql

import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.neighbors
import sklearn.externals
import sklearn.preprocessing

from utility_funtions import str2bool_argparse

import event_processing_v3
import postgresql_v3_event_storage
import dataset_query_functions_v3


def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


def load_data(query_functions, columns_for_classification_dict,
              max_deviation_from_inter_class_balance=.05,
              max_deviation_from_intra_class_balance=.1, # of expected count
              allow_missing_dataset_parts=False,
              single_query_limit=200000, include_event_id=False,
              return_single_df=False,
              simu_event_relation_table_name='spb_processing_v3.simu_event_relation',
              simu_event_table_name = 'spb_processing_v3.simu_event',
              simu_event_additional_table_name = 'spb_processing_v3.simu_event_additional'):

    classes_entries_counts_ratios = [0.5, 0.5]

    simu_source_data_types_nums = [3, 30]
    simu_source_data_types_counts = [None, None]
    simu_source_data_types_ratios = [0.5, 0.5]

    noise_source_data_types_nums = [1, 30]
    noise_source_data_types_counts = [None, None]
    noise_source_data_types_ratios = [0.5, 0.5]

    flight_noise_joined_tables_list = []

    conn = query_functions.event_storage.connection
    cur = conn.cursor()

    # TODO simu_signal
    simu_where_clauses_str, simu_joined_tables_list = \
        query_functions.get_query_clauses__where_simu(
            gtu_in_packet_distacne=(42, 10), num_frames_signals_ge_bg__ge=3, num_frames_signals_ge_bg__le=999,
            etruth_theta__ge=None, etruth_theta__le=None,
            simu_event_relation_table_name=simu_event_relation_table_name,
            simu_event_table_name=simu_event_table_name,
            simu_event_additional_table_name=simu_event_additional_table_name
        )

    common_select_clause_str, common_joined_tables_list = query_functions.get_query_clauses__select(columns_for_classification_dict)

    simu_join_clauses_str = query_functions.get_query_clauses__join(common_joined_tables_list + simu_joined_tables_list)

    for i, source_data_type_num in enumerate(simu_source_data_types_nums):
        q = query_functions.get_events_selection_query_plain(
            source_data_type_num=source_data_type_num,
            select_additional='', join_additional=simu_join_clauses_str,
            where_additional=simu_where_clauses_str,
            order_by='event_id', limit=1, offset=0,
            base_select='COUNT(*)')

        cur.execute(q)
        res = cur.fetchone()
        if res is None or len(res) < 1:
            raise RuntimeError('Unable to query COUNT(*) for simu source_data_type_num={}'.format(source_data_type_num))
        simu_source_data_types_counts[i] = int(res[0])

    noise_join_clauses_str = query_functions.get_query_clauses__join(common_joined_tables_list + flight_noise_joined_tables_list)
    noise_where_clauses_str = ' AND abs(gtu_in_packet-42) >= 20 '

    for i, source_data_type_num in enumerate(noise_source_data_types_nums):
        q = query_functions.get_events_selection_query_plain(
            source_data_type_num=source_data_type_num,
            select_additional='', join_additional=noise_join_clauses_str,
            where_additional=noise_where_clauses_str,
            order_by='event_id', limit=1, offset=0,
            base_select='COUNT(*)')

        cur.execute(q)
        res = cur.fetchone()
        if res is None or len(res) < 1:
            raise RuntimeError('Unable to query COUNT(*) for simu source_data_type_num={}'.format(source_data_type_num))
        noise_source_data_types_counts[i] = int(res[0])

    classes_source_data_type_entries_counts = [simu_source_data_types_counts, noise_source_data_types_counts]
    classes_source_data_type_entries_expected_ratios = [simu_source_data_types_ratios, noise_source_data_types_ratios]
    for i, entries_counts in enumerate(classes_source_data_type_entries_counts):
        for j, entries_count in enumerate(entries_counts):
            if entries_count > single_query_limit:
                entries_counts[j] = single_query_limit

    classes_entries_tot_counts = [np.sum(simu_source_data_types_counts), np.sum(noise_source_data_types_counts)]

    argmax_expected_class_ratio =  np.argmax(classes_entries_counts_ratios).item()
    hundred_percent_count = classes_entries_tot_counts[argmax_expected_class_ratio]

    for i, (class_entries_tot_count, expected_ratio, entries_counts) in \
            enumerate(zip(classes_entries_tot_counts,
                          classes_entries_counts_ratios,
                          classes_source_data_type_entries_counts)):
        new_class_entries_tot_count = expected_ratio * hundred_percent_count
        class_entries_count_change_ratio = class_entries_tot_count / new_class_entries_tot_count
        if new_class_entries_tot_count - max_deviation_from_inter_class_balance * new_class_entries_tot_count > class_entries_tot_count:
            if allow_missing_dataset_parts:
                print('WARNING: Unable to acquire required percentage of a dataset part {} for class {}'.format(j, i),
                      file=sys.stderr)
            else:
                raise RuntimeError('Unable to acquire required percentage of a dataset part {} for class {}'.format(j, i))
        for j, entries_count in enumerate(entries_counts):
            entries_counts[j] = int(np.ceil(entries_count * class_entries_count_change_ratio))
        class_entries_tot_count[i] = np.sum(entries_counts) #new_class_entries_tot_count

    for i, (tot_count, entries_counts, expected_ratios) in \
            enumerate(zip(classes_entries_tot_counts,
                          classes_source_data_type_entries_counts,
                          classes_source_data_type_entries_expected_ratios)):

        argmax_expected_ratio = np.argmax(classes_source_data_type_entries_expected_ratios).item()
        hundred_percent_count = classes_source_data_type_entries_counts[argmax_expected_ratio] / \
                            classes_source_data_type_entries_expected_ratios[argmax_expected_ratio]

        for j, (entries_count, expected_ratio) in enumerate(zip(entries_counts, expected_ratios)):
            new_entries_count = int(np.ceil(expected_ratio * hundred_percent_count))
            if new_entries_count - max_deviation_from_intra_class_balance * new_entries_count > len(entries_counts):
                if allow_missing_dataset_parts:
                    print('WARNING: Unable to acquire required percentage of a dataset part {} for class {}'.format(j, i), file=sys.stderr)
                else:
                    raise RuntimeError('Unable to acquire required percentage of a dataset part {} for class {}'.format(j, i))
            entries_counts[j] = new_entries_count

    if not return_single_df:
        class_data = [None]*len(classes_entries_tot_counts)
    else:
        class_data = None

    query_base_select = query_functions.event_storage.data_table_name + '.event_id' if include_event_id else ''

    # this is probably not efficient

    for i, (source_data_type_nums, entries_counts, join_clauses_str, where_clauses_str) in enumerate([
        (simu_source_data_types_nums, simu_source_data_types_counts, simu_join_clauses_str, simu_where_clauses_str),
        (noise_source_data_types_nums, noise_source_data_types_counts, noise_join_clauses_str, noise_where_clauses_str),
    ]):
        for entries_count, source_data_type_num in zip(entries_counts, source_data_type_nums):
            q = query_functions.get_events_selection_query_plain(
                    source_data_type_num=source_data_type_num,
                    select_additional=common_select_clause_str, join_additional=join_clauses_str,
                    where_additional=where_clauses_str,
                    order_by='RANDOM()', limit=entries_count, offset=0,
                base_select=query_base_select)

            events_df = psql.read_sql(q, conn)
            if return_single_df:
                events_df['class'] = i

            if not return_single_df:
                curr_class_data_df = class_data[i]
            else:
                curr_class_data_df = class_data

            if curr_class_data_df is None:
                if return_single_df:
                    class_data = events_df
                else:
                    class_data[i] = events_df
            else:
                curr_class_data_df.append(events_df)

    return class_data

    # print('Loading data - visible showers...')
    # X = np.array(get_class_1_func(cur, columns), dtype=np.float32)
    # len_class_1 = len(X)
    #
    # print('Loading data - invisible showers...')
    # a = np.array(select_training_data__invisible_showers(cur, columns), dtype=np.float32)
    # if len(a) > 0:
    #     X = np.append(X, a, axis=0)
    # print('Loading data - low energy pmt...')
    # a = np.array(select_training_data__low_energy_in_pmt(cur, columns), dtype=np.float32)
    # if len(a) > 0:
    #     X = np.append(X, a, axis=0)
    # print('Loading data - LED...')
    # a = np.array(select_training_data__led(cur, columns), dtype=np.float32)
    # if len(a) > 0:
    #     X = np.append(X, a, axis=0)
    #
    # y = np.zeros(len(X), dtype=np.int8)
    # y[:len_class_1] = 1


def scale_data(X, scaler_picke_pathname, pickle_overwrite=False, scaler_class=sklearn.preprocessing.StandardScaler):
    scaler = None

    if scaler_picke_pathname:
        if isinstance(scaler_picke_pathname, str) and os.path.isdir(scaler_picke_pathname):
            print('Calculating hash of data ...')
            data_md5 = hashlib.md5(pickle.dumps(X, protocol=0))
            scaler_picke_pathname = os.path.join(scaler_picke_pathname, 'scaler_for_{}.joblib.pkl'.format(data_md5.hexdigest()))

        if os.path.exists(scaler_picke_pathname) and not pickle_overwrite:
            print("Loading existing scaler...")
            scaler = sklearn.externals.joblib.load(scaler_picke_pathname)

    if not scaler:
        print('StandardScaler - fitting and transforming data ...')
        scaler = sklearn.preprocessing.StandardScaler()

        if scaler_picke_pathname:
            print("Saving scaled data into file {}".format(scaler_picke_pathname))
            sklearn.externals.joblib.dump(scaler, scaler_picke_pathname)

        X = scaler_class().fit_transform(X)
    else:
        print('Scaler - transforming data ...')
        X = scaler.transform(X)

    return X


def main(argv):

    parser = argparse.ArgumentParser(description='Draw histograms of parameter values')
    parser.add_argument('-d','--dbname',default='eusospb_data')
    parser.add_argument('-U','--user',default='eusospb')
    parser.add_argument('-s','--host',default='localhost')
    parser.add_argument('--password')
    parser.add_argument('--out', default='.')
    # parser.add_argument('--get-class1-func', type=int, default=0)
    # parser.add_argument('--model-config', type=int, default=0)

    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('-c','--classifier', default='adaboost')
    parser.add_argument('--event-storage-provider-config', default='../config.ini')

    parser.add_argument('--classifier-params-json', default='{}')
    parser.add_argument('--meta-classifier-params-json', default='{}')
    parser.add_argument('--random-state', type=int, default=42)

    parser.add_argument('--do-test-train-split', type=str2bool_argparse, default=True, help='If true, sets are created')
    parser.add_argument('--apply-scaler', type=str2bool_argparse, default=True, help='If true, data are scaled')

    parser.add_argument('--overwrite', type=str2bool_argparse, default=False, help='Overwrite output model file')
    parser.add_argument('--classifier-file', default='', help='By default the filename is determined from data and out directory is used')
    parser.add_argument('--scaler-file', default='', help='By default the filename is determined from data and out directory is used')
    parser.add_argument('--read', default="", help='Only read exiting model')

    # parser.add_argument('--print-queries', type=str2bool_argparse, default='Only print queries')

    args = parser.parse_args(argv)

    # http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
    scaler_pickle_pathname = None
    if args.apply_scaler:
        if os.path.isdir(args.out):
            if not args.scaler_file:
                scaler_pickle_pathname = os.path.join(args.out, 'scaler_{class_name}_{data_md5}.pkl')
            else:
                scaler_pickle_pathname = os.path.join(args.out, args.scaler_file)
        elif args.scaler_file:
            scaler_pickle_pathname = args.scaler_file

    classifier_pickle_pathname = None
    if os.path.isdir(args.out):
        if not args.classifier_file:
            classifier_pickle_pathname = os.path.join(args.out, 'classifier_{class_name}_{data_md5}_{params_md5}.pkl')
        else:
            classifier_pickle_pathname = os.path.join(args.out, args.classifier_file)
    elif args.scaler_file:
        classifier_pickle_pathname = args.classifier_file

    metaclassifier_params = json.loads(args.meta_classifier_params_json)
    classifier_params = json.loads(args.classifier_params_json)

    if not isinstance(classifier_params,dict):
        raise RuntimeError('Invalid meta_classifier_params')
    if not isinstance(metaclassifier_params,dict):
        raise RuntimeError('Invalid classifier_params')

    classifier_params = {'random_state':args.random_state,**classifier_params}
    metaclassifier_params = {'random_state': args.random_state, **metaclassifier_params}

    params_str = str([args.seed, classifier_params, metaclassifier_params, args.random_state, args.do_test_train_split,
              args.apply_scaler])
    params_str_hexdigest = hashlib.md5(params_str.encode('utf8')).hexdigest()

    classifier = None
    if not args.password:
        args.password = getpass.getpass()

    np.random.seed(args.seed)

    event_v3_storage_provider = dataset_query_functions_v3.build_event_v3_storage_provider(
        event_storage_provider_config_file=args.event_storage_provider_config, table_names_version='ver3',
        event_storage_class=postgresql_v3_event_storage.PostgreSqlEventV3StorageProvider,
        event_processing_class=event_processing_v3.EventProcessingV3
    )

    query_functions = dataset_query_functions_v3.Ver3DatasetQueryFunctions(event_v3_storage_provider)

    # should be configurable
    columns_for_classification_dict = query_functions.get_columns_for_classification_dict__by_excluding(
        # excluded_columns_re_list=['gtu_in_packet', '.+_seed_coords_[xy].*', ],
        # included_columns_re_list=[]
        # default_excluded_columns_re_list=['source_file_.+'] + system_columns,
        included_columns_re_list=[('^$','^source_file_acquisition')]
    )

    all_classes_data_df = load_data(query_functions, columns_for_classification_dict)

    # class_hashes = [None] * len(all_classes_data_df)
    # for i, class_data in all_classes_data_df:
    #     print('Calculating hash of data for class {} ...'.format(i))
    #     class_hashes[i] = hashlib.md5(pickle.dumps(class_data[i].values, protocol=0))
    # merged_hexdigset = hashlib.md5("-".join([class_hash.hexdiget() for class_hash in class_hashes]))

    data_hexdigest = hashlib.md5(pickle.dumps(all_classes_data_df.values, protocol=0)).hexdigest()

    X = all_classes_data_df.loc[:, all_classes_data_df.columns != 'class']
    y = all_classes_data_df['class']

    if args.apply_scaler:
        X = scale_data(X,
                       scaler_picke_pathname=scaler_pickle_pathname.format(
                           class_name='StandardScaler', data_md5=data_hexdigest, params_md5=params_str_hexdigest),
                       pickle_overwrite=False,
                       scaler_class=sklearn.preprocessing.StandardScaler)

    classifier = None

    if args.read and os.path.exists(classifier_pickle_pathname):
        print("Loading existing classifier...")
        classifier = sklearn.externals.joblib.load(classifier_pickle_pathname)

    if args.do_test_train_split:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=args.random_state)
    else:
        X_train = X_test = X
        y_train, y_test = y

    if classifier is None:


        classifier = None
        if args.classifier == 'adaboost':
            # http://scikit-learn.org/stable/modules/ensemble.html
            classifier = sklearn.ensemble.AdaBoostClassifier(**classifier_params)
        elif args.classifier == 'randomforest':
            classifier = sklearn.ensemble.RandomForestClassifier(**classifier_params)
        elif args.classifier == 'extra_trees':
            classifier = sklearn.ensemble.ExtraTreesClassifier(**classifier_params)
        elif args.classifier == 'decision_tree':
            # http://scikit-learn.org/stable/modules/tree.html#tree
            classifier = sklearn.tree.DecisionTreeClassifier(**classifier_params)
        elif args.classifier == 'bagged_decision_tree':
            # http://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html#sphx-glr-auto-examples-ensemble-plot-bias-variance-py
            classifier = sklearn.ensemble.BaggingRegressor(sklearn.tree.DecisionTreeClassifier(**classifier_params), **metaclassifier_params)
        elif args.classifier == 'mlp':
            # if args.model_config == 1:
            #     classifier_params['hidden_layer_sizes'] = (300,150)
            # if args.model_config == 2:
            #     classifier_params['hidden_layer_sizes'] = (500, 400, 100)
            classifier = sklearn.neural_network.MLPClassifier(**classifier_params)
        elif args.classifier == 'naive_bayes':
            classifier_params = {}
            classifier = sklearn.naive_bayes.GaussianNB(**classifier_params)
        elif args.classifier == 'kneighbors':
            classifier = sklearn.neighbors.KNeighborsClassifier(**classifier_params)
        else:
            raise Exception('Unexpected classifier')


        if os.path.exists(classifier_pickle_pathname) and not args.overwrite and not args.read:
            print('WARNING: Model file "{}" already exists'.format(classifier_pickle_pathname))

        print("Fitting data using {}".format(classifier.__class__.__name__))

        # cross_val_score # k-fold ... http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
        classifier.fit(X_train, y_train)

        if classifier_pickle_pathname:
            print("Saving model {} into file {}".format(classifier.__class__.__name__, classifier_pickle_pathname))
            if os.path.exists(classifier_pickle_pathname):
                if args.overwrite:
                    print('WARNING: Overwriting file "{}".'.format(classifier_pickle_pathname))
                else:
                    print('File "{}" already exists - nothing saved.'.format(classifier_pickle_pathname))
            if not os.path.exists(classifier_pickle_pathname) or args.overwrite:
                sklearn.externals.joblib.dump(classifier, classifier_pickle_pathname)
    else:
        classifier = sklearn.externals.joblib.load(args.read)

    score = classifier.score(X_test, y_test)

    print("Score for {}: {}".format(classifier.__class__.__name__, score))


if __name__ == '__main__':
    main(sys.argv[1:])
