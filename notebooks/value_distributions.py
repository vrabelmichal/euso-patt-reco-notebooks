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

import matplotlib.pyplot as plt
from tqdm import tqdm

# import tool.acqconv


def draw_distributions(query_format, con, num_columns_at_once=100, num_at_once=300000, max_rows=1000000,
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

        cols_in_fig = min(num_figures,5)
        rows_in_fig = (num_figures//cols_in_fig)+int((num_figures % cols_in_fig) > 0)
        fig, axs = plt.subplots(rows_in_fig, cols_in_fig)
        axs_flattened = axs.reshape(-1)
        fig.set_size_inches(cols_in_fig*34.28/5,rows_in_fig*70/20)

        col_i = 0
        for col_i, col in enumerate(columns):
            axs_flattened[col_i].hist([float(v) for v in vals[col].elements() if v is not None],
                                      # [float(k) for k,v in vals[col].items() if k is not None and v is not None],
                                      # weights=[float(v) for k,v in vals[col].items() if k is not None and v is not None],
                                      normed=False, label=col, bins='auto')
            axs_flattened[col_i].set_yscale('log')
            axs_flattened[col_i].set_title(col)
            print("hist", col_i, col)

            col_i += 1


        if save_img_format:
            save_img_pathname = save_img_format.format(first_col_index=first_col, last_col_index=last_col)
            dir_pathname = os.path.dirname(save_img_pathname)
            if dir_pathname and dir_pathname != '.':
                os.makedirs(dir_pathname, exist_ok=True)
            print("Saving figure to {}")
            fig.savefig(save_img_pathname)
            plt.close('all')
        else:
            out.append((fig, axs))


        first_col = last_col
        last_col = first_col + num_columns_at_once

    #return out


def main(argv):

    parser = argparse.ArgumentParser(description='Draw histograms of parameter values')
    parser.add_argument('--dbname',default='eusospb_data')
    parser.add_argument('--user',default='eusospb')
    parser.add_argument('--password')
    parser.add_argument('--host',default='localhost')
    parser.add_argument('--odir', default='.')

    args = parser.parse_args(argv)

    if not args.password:
        args.password = getpass.getpass()

    con = pg.connect(dbname=args.dbname, user=args.user, password=args.password, host=args.host)
    cur = con.cursor()

    flight_events_query = "SELECT {columns} FROM spb_processing_event_ver2 WHERE meta=1 OR meta IS NULL AND source_file_acquisition LIKE 'allpackets-SPBEUSO-ACQUISITION-2017%' "\
                "{order} OFFSET {offset} LIMIT {limit}"
    draw_distributions(flight_events_query, con, max_rows=1000,
                       save_img_format=os.path.join(args.odir, "flight_event_prop_dist__cols_{first_col_index}_{last_col_index}.png"))

    utah_events_query = "SELECT {columns} FROM spb_processing_event_ver2 WHERE meta=2 OR meta IS NULL AND source_file_acquisition LIKE 'allpackets-SPBEUSO-ACQUISITION-2016%' "\
                "{order} OFFSET {offset} LIMIT {limit}"
    draw_distributions(flight_events_query, con, max_rows=1000,
                       save_img_format=os.path.join(args.odir, "utah_event_prop_dist__cols_{first_col_index}_{last_col_index}.png"))

    flight_events_query = "SELECT {columns} FROM spb_processing_event_ver2 WHERE meta=3 OR meta IS NULL AND source_file_acquisition LIKE 'ev_%' "\
                "{order} OFFSET {offset} LIMIT {limit}"
    draw_distributions(flight_events_query, con,
                       save_img_format=os.path.join(args.odir, "simu_event_prop_dist__cols_{first_col_index}_{last_col_index}.png"))


if __name__ == '__main__':
    main(sys.argv[1:])