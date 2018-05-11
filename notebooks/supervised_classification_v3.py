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



def make_train_test(X, y, test_train_split_kwargs={'random_state':None},
                    scaler_pathname=None, scaler_pathname_overwrite=False,
                    balance_subsample=True, subsample_size=1.0):

    scaler = None
    X_data_md5 = None

    if scaler_pathname:
        if isinstance(scaler_pathname, str):
            if os.path.isdir(scaler_pathname):
                print('Calculating hash of data ...')
                X_data_md5 = hashlib.md5(X.tobytes())
                y_data_md5 = hashlib.md5(y.tobytes())
                scaler_pathname = os.path.join(scaler_pathname, 'scaler_for_{}_{}.joblib.pkl'.format(
                    X_data_md5.hexdigest(), y_data_md5.hexdigest()))

        if os.path.exists(scaler_pathname) and not scaler_pathname_overwrite:
            print("Loading existing scaler...")
            scaler = sklearn.externals.joblib.load(scaler_pathname)

        if not scaler:
            print('StandardScaler - fitting and transforming data ...')
            scaler = sklearn.preprocessing.StandardScaler()
            X = sklearn.preprocessing.StandardScaler().fit_transform(X)
        else:
            print('Scaler - transforming data ...')
            X = scaler.transform(X)

    if balance_subsample:
        X = balanced_subsample(X, y, subsample_size)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, **test_train_split_kwargs)
    return  X_train, X_test, y_train, y_test, scaler, scaler_pathname, X_data_md5


def load_data(query_functions, columns_for_classification_dict,
              max_inter_class_ratio_off_balance=.05, max_deviation_from_intra_class_balance=.1,
              allow_missing_dataset_parts=False):

    flight_noise_joined_tables_set = {}

    conn = query_functions.event_storage.connection
    cur = conn.cursor()

    simu_where_clauses_str, simu_joined_tables_set = \
        query_functions.get_query_clauses__where_simu(
            gtu_in_packet_distacne=(42, 10), num_frames_signals_ge_bg__ge=3, num_frames_signals_ge_bg__le=999,
            etruth_theta__ge=None, etruth_theta__le=None)

    common_select_clause_str = \
        query_functions.get_query_clauses__select(columns_for_classification_dict,
                                                  [simu_joined_tables_set, flight_noise_joined_tables_set])

    simu_join_clauses_str = query_functions.get_query_clauses__join(simu_joined_tables_set)

    simu_source_data_types_nums = [3, 30]
    simu_source_data_types_counts = [None, None]
    simu_source_data_types_ratios = [0.5, 0.5]

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

    noise_join_clauses_str = query_functions.get_query_clauses__join(flight_noise_joined_tables_set)
    noise_where_clauses_str = ' AND abs(gtu_in_packet-42) >= 20 '

    noise_source_data_types_nums = [1, 30]
    noise_source_data_types_counts = [None, None]
    noise_source_data_types_ratios = [0.5, 0.5]

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

    classification_class_entries_tot_counts = [np.sum(simu_source_data_types_counts), np.sum(noise_source_data_types_counts)]

    # TODO following is not correct especially for multiple dataset parts

    for i, (tot_count, entries_counts, expected_ratios) in enumerate(zip(classification_class_entries_tot_counts,
                                                        [simu_source_data_types_counts, noise_source_data_types_counts],
                                                        [simu_source_data_types_ratios, noise_source_data_types_ratios])):
        for j, (entries_count, expected_ratio) in enumerate(zip(entries_counts, expected_ratios)):
            actual_ratio = entries_count/tot_count
            if actual_ratio != 1:
                d = actual_ratio - expected_ratio
                if d > max_deviation_from_intra_class_balance:
                    entries_counts[j] = tot_count*(expected_ratio+max_deviation_from_intra_class_balance)
            elif not allow_missing_dataset_parts:
                raise RuntimeError('Missing dataset part {} of class {}'.format(j,i))
            else:
                print('WARNING: only dataset part {} available for class {}'.format(j,i))


    argmin_class_entries_count = np.argmin(classification_class_entries_tot_counts)
    min_class_entries_count = classification_class_entries_tot_counts[argmin_class_entries_count]

    for i, class_entries_count in enumerate(classification_class_entries_tot_counts):
        if i == argmin_class_entries_count:
            continue
        if abs(class_entries_count - min_class_entries_count)/min_class_entries_count > max_inter_class_ratio_off_balance:
            classification_class_entries_tot_counts[i] = min_class_entries_count*(1 + max_inter_class_ratio_off_balance)

    #

    pd

    # ....



    query_functions.get_events_selection_query_plain(
        source_data_type_num=source_data_type_num,
        select_additional=common_select_clause_str, join_additional=common_join_clauses+simu_join_clauses,
        where_additional=simu_where_clauses_str,
        order_by='event_id', limit=1000000, offset=0)



    positive_sample_queries = []

    for source_data_type_num in enumerate((3, 30)):

        positive_sample_queries.append(
            query_functions.get_event_selection_query__simu_by_num_frames__excluding_columns(
                source_data_type_num=source_data_type_num,
                num_frames_signals_ge_bg__ge=3, num_frames_signals_ge_bg__le=999,
                etruth_theta__ge=None, limit=single_query_limit, offset=offset,
                gtu_in_packet_distance=(42,10),
                excluded_columns_re_list=(
                  'gtu_in_packet', '.+_seed_coords_[xy].*',
                ),
                default_excluded_columns_re_list=(
                  # 'event_id','program_version', 'timestamp', 'global_gtu', 'packet_id', 'source_data_type_num', 'config_info_id',
                    'source_file_.+',
                ),
                default_included_columns_re_list=(
                  'event_id', 'program_version', 'timestamp', 'global_gtu', 'packet_id', 'source_data_type_num', 'config_info_id',
                )
            )
        )
        # subsample directly in the select ?

    negative_sample_queries = []

    for source_data_type_num in enumerate((1, 30)):
        negative_sample_queries.append(
            query_functions.get_event_selection_query_excluding_columns(
                source_data_type_num=source_data_type_num,
                where_additional=' AND abs(gtu_in_packet-42) >= 20 ',
                limit=single_query_limit, offset=offset,
                excluded_columns_re_list=(
                    'gtu_in_packet', '.+_seed_coords_[xy].*',
                ),
                default_excluded_columns_re_list=(
                    # 'event_id','program_version', 'timestamp', 'global_gtu', 'packet_id', 'source_data_type_num', 'config_info_id',
                    'source_file_.+',
                ),
                default_included_columns_re_list=(
                    'event_id', 'program_version', 'timestamp', 'global_gtu', 'packet_id', 'source_data_type_num',
                    'config_info_id',
                )
            )
        )


    # TODO on new data
    # TODO on kenji's events

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



def main(argv):

    parser = argparse.ArgumentParser(description='Draw histograms of parameter values')
    parser.add_argument('-d','--dbname',default='eusospb_data')
    parser.add_argument('-U','--user',default='eusospb')
    parser.add_argument('--password')
    parser.add_argument('-s','--host',default='localhost')
    parser.add_argument('--out', default='.')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--get-class1-func', type=int, default=0)
    parser.add_argument('--model-config', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('-c','--classifier', default='adaboost')
    parser.add_argument('--overwrite', type=str2bool_argparse, default=False, help='Overwrite output model file')
    parser.add_argument('--apply-scaler', type=str2bool_argparse, default=False, help='If true data are scaled')
    parser.add_argument('--scaler-file', default='', help='By default the filename is determined from data and out directory is used')
    parser.add_argument('--read', default="", help='Only read exiting model')

    # parser.add_argument('--print-queries', type=str2bool_argparse, default='Only print queries')

    args = parser.parse_args(argv)

    classifier = None
    if not args.password:
        args.password = getpass.getpass()

    np.random.seed(args.seed)

    event_v3_storage_provider = dataset_query_functions_v3.build_event_v3_storage_provider(
        event_storage_provider_config_file='config.ini', table_names_version='ver3',
        event_storage_class=postgresql_v3_event_storage.PostgreSqlEventV3StorageProvider,
        event_processing_class=event_processing_v3.EventProcessingV3
    )

    query_functions = dataset_query_functions_v3.Ver3DatasetQueryFunctions(
        event_v3_storage_provider,
        simu_table_name='spb_processing_v3.simu_event',
        simu_additional_table_name='spb_processing_v3.simu_event_additional'
    )

    # should be configurable
    columns_for_classification_dict = query_functions.get_columns_for_classification_dict__by_excluding(
        excluded_columns_re_list=['gtu_in_packet', '.+_seed_coords_[xy].*', ],
        default_included_columns_re_list=[]
        # default_excluded_columns_re_list=['source_file_.+'] + system_columns,
    )

    load_data(query_functions, columns_for_classification_dict)






    con = pg.connect(dbname=args.dbname, user=args.user, password=args.password, host=args.host)
    cur = con.cursor()


    columns = get_columns_for_classification()

    if args.get_class1_func == 0:
        get_class1_func = select_training_data__visible_showers
    elif args.get_class1_func == 1:
        get_class1_func = select_training_data__visible_showers_other_bgf
    else:
        raise Exception('Invalid class1 func')

    # http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
    scaler_pathname = None
    if args.apply_scaler:
        if os.path.isdir(args.out):
            if not args.scaler_file:
                scaler_pathname = args.out
            else:
                scaler_pathname = os.path.join(args.out, scaler_pathname)
        elif args.scaler_file:
            scaler_pathname = args.scaler_file
        else:
            scaler_pathname = True

    X_train, X_test, y_train, y_test, scaler, scaler_pathname, data_md5 = \
        load_train_test(cur, columns, args.random_state, get_class_1_func=get_class1_func, scaler_pathname=scaler_pathname, scaler_pathname_overwrite=args.overwrite)

    if not args.read:

        metaclassifier_params = {}
        classifier_params = {'random_state': args.random_state}

        classifier = None
        if args.classifier == 'adaboost':
            # http://scikit-learn.org/stable/modules/ensemble.html
            classifier = sklearn.ensemble.AdaBoostClassifier(**classifier_params)
        elif args.classifier == 'randomforest':
            classifier = sklearn.ensemble.RandomForestClassifier(**classifier_params)
        elif args.classifier == 'decision_tree':
            # http://scikit-learn.org/stable/modules/tree.html#tree
            classifier = sklearn.tree.DecisionTreeClassifier(**classifier_params)
        elif args.classifier == 'bagged_decision_tree':
            # http://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html#sphx-glr-auto-examples-ensemble-plot-bias-variance-py
            metaclassifier_params = {'random_state': args.random_state}
            classifier = sklearn.ensemble.BaggingRegressor(sklearn.tree.DecisionTreeClassifier(**classifier_params), **metaclassifier_params)
        elif args.classifier == 'mlp':
            if args.model_config == 1:
                classifier_params['hidden_layer_sizes'] = (300,150)
            if args.model_config == 2:
                classifier_params['hidden_layer_sizes'] = (500, 400, 100)
            classifier = sklearn.neural_network.MLPClassifier(**classifier_params)
        elif args.classifier == 'naive_bayes':
            classifier_params = {}
            classifier = sklearn.naive_bayes.GaussianNB(**classifier_params)
        elif args.classifier == 'kneighbors':
            classifier = sklearn.neighbors.KNeighborsClassifier(**classifier_params)
        else:
            raise Exception('Unexpected classifier')

        outfile_pathname = None

        if args.out:
            if os.path.isdir(args.out):
                outfile_pathname = os.path.join(args.out, "{}.{}.joblib.pkl".format(
                    args.classifier,
                    hashlib.md5((str(args) + str(metaclassifier_params) + str(classifier_params)).encode()).hexdigest()
                ))
            else:
                outfile_pathname = args.out

            if os.path.exists(outfile_pathname) and not args.overwrite and not args.read:
                raise Exception('Model file "{}" already exists'.format(outfile_pathname))

        print("Fitting data using {}".format(classifier.__class__.__name__))

        # cross_val_score # k-fold ... http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
        classifier.fit(X_train, y_train)


        if outfile_pathname:
            print("Saving model {} into file {}".format(classifier.__class__.__name__, outfile_pathname))
            sklearn.externals.joblib.dump(classifier, outfile_pathname)
    else:
        classifier = sklearn.externals.joblib.load(args.read)

    score = classifier.score(X_test, y_test)

    print("Score for {}: {}".format(classifier.__class__.__name__, score))


if __name__ == '__main__':
    main(sys.argv[1:])
