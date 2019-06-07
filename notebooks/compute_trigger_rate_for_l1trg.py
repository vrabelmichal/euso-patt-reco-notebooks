import sys
import os
import numpy as np
import re
import argparse
# import psycopg2 as pg
# import pandas as pd
# import pandas.io.sql as psql
# import getpass
import matplotlib as mpl
import hashlib
import math
import collections
import functools
import datetime
from sklearn.externals import joblib
import pickle
import physt

mpl.rcParams['figure.dpi'] = 80

# import matplotlib.pyplot as plt
# import seaborn as sns

app_base_dir = '/home/spbproc/euso-spb-patt-reco-v1'
if app_base_dir not in sys.path:
    sys.path.append(app_base_dir)

# import event_processing_v3
# import event_processing_v4
# import postgresql_v3_event_storage
# import dataset_query_functions_v3

# import tool.acqconv
# from data_analysis_utils import *
from data_analysis_utils_trigger_rate import *
# import supervised_classification as supc
# from utility_funtions import key_vals2val_keys

from utility_funtions import str2bool_argparse

# import event_reading

def main(argv):
    parser = argparse.ArgumentParser(description='Trigger rate for acquisition and l1trg.')
    parser.add_argument('--files-dir-abspath', default='/home/spbproc/SPBDATA_flight')
    parser.add_argument('-b', '--trg-files-dir-abspath-base-dir', default='/home/spbproc/SPBDATA_processed')
    parser.add_argument('-f', '--trg-files-dir-abspath-format', default='{base_dir}/{acq_dirname}/{acq_basename_no_ext}/l1_trigger_kenji/{l1trg_root_filename}')
    parser.add_argument('-m', '--flat-field-map-pathname', default='/home/spbproc/euso-spb-patt-reco-v1/resources/inverse_flat_average_directions_4m_flipud.txt')
    parser.add_argument('-s', '--data-snippets-dir', default='trigger_rate_for_l1trg')

    parser.add_argument('-p', '--file-name-prefix', default='')
    parser.add_argument('--trg-type', default='l1')

    parser.add_argument('--skipped-files-count', type=int, default=0)
    parser.add_argument('--bgf', type=float, default=0.5)

    parser.add_argument('--recreate-pickles', type=str2bool_argparse, default=False,)
    parser.add_argument('--one-trg-per-packet', type=str2bool_argparse, default=True,)
    parser.add_argument('--skip-missing-files', type=str2bool_argparse, default=True,)
    parser.add_argument('--skip-exceptions', type=str2bool_argparse, default=False,)

    args = parser.parse_args(argv)

    if args.file_name_prefix == '':
        file_name_prefix = hashlib.md5(';'.join([
            str(k)+str(v) for k,v in vars(args) if k not in ('recreate_pickles', 'file_name_prefix')
        ]).encode()).hexdigest()[0:8]
    else:
        file_name_prefix = args.file_name_prefix

    data_snippets_dir = args.data_snippets_dir if args.data_snippets_dir != '' else None

    if data_snippets_dir is not None:
        os.makedirs(data_snippets_dir, exist_ok=True)
        # os.makedirs(os.path.join(data_snippets_dir, 'figures'), exist_ok=True)

    def filter_func(f, d):
        r = os.path.splitext(f)[1] == ".root" and "ACQUISITION" in os.path.basename(f) and re.search(
            r'SPBDATA_flight/allpackets-SPBEUSO-ACQUISITION-20170(42[4-9]|430|5\d+)$', d) is not None
        return r

    processed_files = sorted(
        [os.path.join(dp, f) for dp, dn, fn in os.walk(args.files_dir_abspath) for f in fn if filter_func(f, dp)])

    if args.trg_type == 'l1':
        flat_field_map_pathname = None if args.flat_field_map_pathname == '' else args.flat_field_map_pathname

        l1trg_files = [create_l1_trigger_data_pathname(
            f, args.files_dir_abspath, args.trg_files_dir_abspath_base_dir, args.trg_files_dir_abspath_format, args.bgf, flat_field_map_pathname
        ) for f in processed_files]
    else:
        l1trg_files = None

    # for f in processed_files[0:3]:
    #     print(f, args.trg_files_dir_abspath_format, args.files_dir_abspath, args.bgf, args.flat_field_map_pathname)
    # print(processed_files[0])
    # print(l1trg_files[0]    )

    otgpp_file_trigger_datetime_list, otgpp_file_timedelta_list, otgpp_file_trigger_p_r_list, otgpp_trigger_num_per_file_list, otgpp_file_trigger_rate_list, otgpp_file_indices_list, \
    otgpp_trigger_num_per_file_list_pathname, otgpp_trigger_rate_per_file_list_pathname, otgpp_file_trigger_datetimes_list_pathname, \
    otgpp_file_trigger_p_r_list_pathname, otgpp_file_trigger_timedelta_list_pathname, otgpp_file_indices_list_pathname, \
    otgpp_info_pathname = \
        count_trigger_rate_per_file(
            processed_files[args.skipped_files_count:], l1trg_files[args.skipped_files_count:],
            data_snippets_dir=data_snippets_dir, file_name_prefix=file_name_prefix,
            trg_type='l1',
            return_filenames=True, recreate_pickles=args.recreate_pickles,
            one_trg_per_packet=args.one_trg_per_packet, packet_size=128,
            skip_missing_files=args.skip_missing_files,
            skip_exceptions=args.skip_exceptions,
        )

    print('-'*50)
    print('trigger_num_per_file_list_pathname   ', otgpp_trigger_num_per_file_list_pathname)
    print('trigger_rate_per_file_list_pathname  ', otgpp_trigger_rate_per_file_list_pathname)
    print('file_trigger_datetimes_list_pathname ', otgpp_file_trigger_datetimes_list_pathname)
    print('file_trigger_p_r_list_pathname       ', otgpp_file_trigger_p_r_list_pathname)
    print('file_trigger_timedelta_list_pathname ', otgpp_file_trigger_timedelta_list_pathname)
    print('file_indices_list_pathname           ', otgpp_file_indices_list_pathname)
    print('info_pathname                        ', otgpp_info_pathname)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
