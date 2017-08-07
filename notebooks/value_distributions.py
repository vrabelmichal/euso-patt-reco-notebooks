import sys
import os
import subprocess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import argparse
import getpass
import collections
import numpy as np
import psycopg2 as pg
#import pandas as pd
#import pandas.io.sql as psql
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 150
mpl.use("Agg")

import matplotlib.pyplot as plt
from tqdm import tqdm

# import tool.acqconv


def draw_distributions(query_format, con, num_columns_at_once=60, num_at_once=100000, max_rows=1000000,
                       save_img_format="/tmp/value_dist/event_prop_dist__cols_{first_col_index}_{last_col_index}.png",
                       drawn_columns=None, columns_offset=None):

    cur = con.cursor()

    cur.execute(query_format.format(columns='COUNT(*)', offset=0, limit=1, order=''))
    num_entries = cur.fetchone()[0]

    if drawn_columns is None:
        cur.execute(query_format.format(columns="*", offset=0, limit=1, order=''))
        all_columns = list(map(lambda x: x[0], cur.description))
        if columns_offset is None:
            columns_offset = 11
    else:
        all_columns = drawn_columns
        if columns_offset is None:
            columns_offset = 0

    first_col = columns_offset
    last_col = first_col + num_columns_at_once

    out = []

    while first_col < len(all_columns):
        columns = all_columns[first_col:last_col]

        num_figures = len(columns) #min(150, len(columns))
        #fig_ax_dict = collections.OrderedDict()


        vals = collections.OrderedDict()
        for col_i, col in enumerate(columns):
            vals[col] = collections.Counter()

        for i in range(0,1+num_entries//num_at_once):
            offset = num_at_once*i
            limit = num_at_once
            if max_rows is not None:
                if offset > max_rows:
                    break
                if offset + limit > max_rows:
                    limit = max_rows - offset
            q = query_format.format(columns=", ".join(columns), offset=offset, limit=limit, order='ORDER BY event_id')
            print(q)
            cur.execute(q)
            rows = cur.fetchall()

            for row in rows:
                for col_i, col in enumerate(columns):
                    if col_i >= num_figures: # unnecesarry
                        break
                    vals[col][row[col_i]] += 1
                    #break # debugging
            print('D')

        cols_in_fig = min(num_figures,5)
        rows_in_fig = (num_figures//cols_in_fig)+int((num_figures % cols_in_fig) > 0)
        fig, axs = plt.subplots(rows_in_fig, cols_in_fig)
        axs_flattened = axs.reshape(-1)
        fig.set_size_inches(cols_in_fig*34.28/5,rows_in_fig*70/20)

        col_i = 0
        for col_i, col in enumerate(columns):
            print("hist {}: {}".format(col_i, col))
            hist_vals = [float(k) for k,v in vals[col].items() if k is not None and v is not None]
            hist_weights = [float(v) for k,v in vals[col].items() if k is not None and v is not None]
            if hist_vals:
                axs_flattened[col_i].hist(#[float(v) for v in vals[col].elements() if v is not None],
                                          hist_vals,
                                          weights=hist_weights,
                                          normed=False, label=col,
                                          # bins=100
                                          bins=100
                )
            else:
                print("  No values to fill histogram")

            print("H")
            do_log_scale = True
            for k in vals[col].keys():
                if k < 0:
                    do_log_scale = False
                    break
            if do_log_scale:
                axs_flattened[col_i].set_yscale('log')
            axs_flattened[col_i].set_title(col)
            print("LT")

            col_i += 1

        if save_img_format:
            save_img_pathname = save_img_format.format(first_col_index=first_col, last_col_index=last_col)
            dir_pathname = os.path.dirname(save_img_pathname)
            if dir_pathname and dir_pathname != '.':
                os.makedirs(dir_pathname, exist_ok=True)
            print("Saving figure to {}".format(save_img_pathname))
            fig.savefig(save_img_pathname)
            plt.close('all')
            print('S')
        else:
            out.append((fig, axs))
            print('A')

        first_col = last_col
        last_col = first_col + num_columns_at_once

    return out


def main(argv):

    parser = argparse.ArgumentParser(description='Draw histograms of parameter values')
    parser.add_argument('-d','--dbname',default='eusospb_data')
    parser.add_argument('-U','--user',default='eusospb')
    parser.add_argument('--password')
    parser.add_argument('-s','--host',default='localhost')
    parser.add_argument('--odir', default='.')
    parser.add_argument('--max-rows', type=int, default=-1)
    parser.add_argument('which_query', nargs='+')

    args = parser.parse_args(argv)

    if not args.password:
        args.password = getpass.getpass()

    max_rows = args.max_rows if args.max_rows is not None and args.max_rows >= 0 else None

    con = pg.connect(dbname=args.dbname, user=args.user, password=args.password, host=args.host)
    cur = con.cursor()

    if 'flight' in args.which_query:
        print('Flight events:')
        flight_events_query = "SELECT {columns} FROM spb_processing_event_ver2 WHERE source_data_type_num=1 OR source_data_type_num IS NULL AND source_file_acquisition LIKE 'allpackets-SPBEUSO-ACQUISITION-2017%' "\
                    "{order} OFFSET {offset} LIMIT {limit}"
        draw_distributions(flight_events_query, con, max_rows=max_rows,
                           save_img_format=os.path.join(args.odir, "flight_event_prop_dist__cols_{first_col_index}_{last_col_index}.png"))

    if 'utah' in args.which_query:
        print('Utah events:')
        utah_events_query = "SELECT {columns} FROM spb_processing_event_ver2 WHERE source_data_type_num=2 OR source_data_type_num IS NULL AND source_file_acquisition LIKE 'allpackets-SPBEUSO-ACQUISITION-2016%' "\
                    "{order} OFFSET {offset} LIMIT {limit}"
        draw_distributions(utah_events_query, con, max_rows=max_rows,
                           save_img_format=os.path.join(args.odir, "utah_event_prop_dist__cols_{first_col_index}_{last_col_index}.png"))

    if 'simu' in args.which_query:
        print('Simu events:')
        simu_events_query = "SELECT {columns} FROM spb_processing_event_ver2 WHERE source_data_type_num=3 OR source_data_type_num IS NULL AND source_file_acquisition LIKE 'ev_%' "\
                    "{order} OFFSET {offset} LIMIT {limit}"
        draw_distributions(simu_events_query, con, max_rows=max_rows,
                           save_img_format=os.path.join(args.odir, "simu_event_prop_dist__cols_{first_col_index}_{last_col_index}.png"))


if __name__ == '__main__':
    main(sys.argv[1:])