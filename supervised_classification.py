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

import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.neighbors
import sklearn.externals

from utility_funtions import str2bool_argparse


def get_columns_for_classification():
    columns_for_classification = \
        [
            # 'gtu_in_packet',
            'num_gtu',
            'num_triggered_pixels',
            'max_trg_box_per_gtu',
            'sum_trg_box_per_gtu',
            'avg_trg_box_per_gtu',
            'max_trg_pmt_per_gtu',
            'sum_trg_pmt_per_gtu',
            'avg_trg_pmt_per_gtu',
            'max_trg_ec_per_gtu',
            'sum_trg_ec_per_gtu',
            'avg_trg_ec_per_gtu',
            'max_n_persist',
            'sum_n_persist',
            'avg_n_persist',
            'max_sum_l1_pdm',
            'sum_sum_l1_pdm',
            'avg_sum_l1_pdm',
            'trigg_x_y_hough__dbscan_num_clusters_above_thr1',
            'trigg_x_y_hough__dbscan_num_clusters_above_thr2',
            'trigg_x_y_hough__max_peak_rho',
            'trigg_x_y_hough__max_peak_phi',
            'trigg_x_y_hough__peak_thr1_avg_rho',
            'trigg_x_y_hough__peak_thr1_avg_phi',
            'trigg_x_y_hough__peak_thr2_avg_rho',
            'trigg_x_y_hough__peak_thr2_avg_phi',
            'trigg_x_y_hough__peak_thr1_max_clu_rho',
            'trigg_x_y_hough__peak_thr1_max_clu_phi',
            'trigg_x_y_hough__peak_thr2_max_clu_rho',
            'trigg_x_y_hough__peak_thr2_max_clu_phi',
            'x_y_hough__dbscan_num_clusters_above_thr1',
            'x_y_hough__dbscan_num_clusters_above_thr2',
            'x_y_hough__dbscan_num_clusters_above_thr3',
            'x_y_hough__max_peak_rho',
            'x_y_hough__max_peak_phi',
            'x_y_hough__peak_thr1_avg_rho',
            'x_y_hough__peak_thr1_avg_phi',
            'x_y_hough__peak_thr2_avg_rho',
            'x_y_hough__peak_thr2_avg_phi',
            'x_y_hough__peak_thr3_avg_rho',
            'x_y_hough__peak_thr3_avg_phi',
            'x_y_hough__peak_thr1_max_clu_rho',
            'x_y_hough__peak_thr1_max_clu_phi',
            'x_y_hough__peak_thr2_max_clu_rho',
            'x_y_hough__peak_thr2_max_clu_phi',
            'x_y_hough__peak_thr3_max_clu_rho',
            'x_y_hough__peak_thr3_max_clu_phi',
            'x_y_active_pixels_num',
            'trigg_x_y_sum3x3_sum',
            'trigg_x_y_sum3x3_norm_sum',
            'trigg_x_y_sum3x3_avg',
            'trigg_x_y_groups_num',
            'trigg_x_y_groups_max_size',
            'trigg_x_y_groups_avg_size',
            'trigg_x_y_groups_sum_sum_sum3x3',
            'trigg_x_y_groups_max_sum_sum3x3',
            'trigg_x_y_groups_avg_sum_sum3x3',
            'trigg_x_y_hough__max_peak_line_rot',
            # 'trigg_x_y_hough__max_peak_line_coord_0_x',
            # 'trigg_x_y_hough__max_peak_line_coord_0_y',
            # 'trigg_x_y_hough__max_peak_line_coord_1_x',
            # 'trigg_x_y_hough__max_peak_line_coord_1_y',
            'x_y_neighbourhood_size',
            'x_y_neighbourhood_width',
            'x_y_neighbourhood_height',
            'x_y_neighbourhood_area',
            'x_y_neighbourhood_counts_sum',
            'x_y_neighbourhood_counts_avg',
            'x_y_neighbourhood_counts_norm_sum',
            'x_y_hough__max_peak_line_rot',
            # 'x_y_hough__max_peak_line_coord_0_x',
            # 'x_y_hough__max_peak_line_coord_0_y',
            # 'x_y_hough__max_peak_line_coord_1_x',
            # 'x_y_hough__max_peak_line_coord_1_y',
            'trigg_x_y_hough__peak_thr1__num_clusters',
            'trigg_x_y_hough__peak_thr1__max_cluster_width',
            'trigg_x_y_hough__peak_thr1__max_cluster_height',
            'trigg_x_y_hough__peak_thr1__avg_cluster_width',
            'trigg_x_y_hough__peak_thr1__avg_cluster_height',
            'trigg_x_y_hough__peak_thr1__max_cluster_area',
            'trigg_x_y_hough__peak_thr1__avg_cluster_area',
            # 'trigg_x_y_hough__peak_thr1__max_cluster_size',
            # 'trigg_x_y_hough__peak_thr1__avg_cluster_size',
            'trigg_x_y_hough__peak_thr1__max_cluster_counts_sum',
            'trigg_x_y_hough__peak_thr1__avg_cluster_counts_sum',
            'trigg_x_y_hough__peak_thr1__max_cluster_area_width',
            'trigg_x_y_hough__peak_thr1__max_cluster_area_height',
            # 'trigg_x_y_hough__peak_thr1__max_cluster_size_width',
            # 'trigg_x_y_hough__peak_thr1__max_cluster_size_height',
            'trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width',
            'trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_height',
            'trigg_x_y_hough__peak_thr1__max_peak_cluster_width',
            'trigg_x_y_hough__peak_thr1__max_peak_cluster_height',
            # 'trigg_x_y_hough__peak_thr1__avg_line_coord_0_x',
            # 'trigg_x_y_hough__peak_thr1__avg_line_coord_0_y',
            # 'trigg_x_y_hough__peak_thr1__avg_line_coord_1_x',
            # 'trigg_x_y_hough__peak_thr1__avg_line_coord_1_y',
            # 'trigg_x_y_hough__peak_thr1__max_clu_line_coord_0_x',
            # 'trigg_x_y_hough__peak_thr1__max_clu_line_coord_0_y',
            # 'trigg_x_y_hough__peak_thr1__max_clu_line_coord_1_x',
            # 'trigg_x_y_hough__peak_thr1__max_clu_line_coord_1_y',
            'trigg_x_y_hough__peak_thr2__num_clusters',
            'trigg_x_y_hough__peak_thr2__max_cluster_width',
            'trigg_x_y_hough__peak_thr2__max_cluster_height',
            'trigg_x_y_hough__peak_thr2__avg_cluster_width',
            'trigg_x_y_hough__peak_thr2__avg_cluster_height',
            'trigg_x_y_hough__peak_thr2__max_cluster_area',
            'trigg_x_y_hough__peak_thr2__avg_cluster_area',
            # 'trigg_x_y_hough__peak_thr2__max_cluster_size',
            # 'trigg_x_y_hough__peak_thr2__avg_cluster_size',
            'trigg_x_y_hough__peak_thr2__max_cluster_counts_sum',
            'trigg_x_y_hough__peak_thr2__avg_cluster_counts_sum',
            'trigg_x_y_hough__peak_thr2__max_cluster_area_width',
            'trigg_x_y_hough__peak_thr2__max_cluster_area_height',
            # 'trigg_x_y_hough__peak_thr2__max_cluster_size_width',
            # 'trigg_x_y_hough__peak_thr2__max_cluster_size_height',
            'trigg_x_y_hough__peak_thr2__max_cluster_counts_sum_width',
            'trigg_x_y_hough__peak_thr2__max_cluster_counts_sum_height',
            'trigg_x_y_hough__peak_thr2__max_peak_cluster_width',
            'trigg_x_y_hough__peak_thr2__max_peak_cluster_height',
            # 'trigg_x_y_hough__peak_thr2__avg_line_coord_0_x',
            # 'trigg_x_y_hough__peak_thr2__avg_line_coord_0_y',
            # 'trigg_x_y_hough__peak_thr2__avg_line_coord_1_x',
            # 'trigg_x_y_hough__peak_thr2__avg_line_coord_1_y',
            # 'trigg_x_y_hough__peak_thr2__max_clu_line_coord_0_x',
            # 'trigg_x_y_hough__peak_thr2__max_clu_line_coord_0_y',
            # 'trigg_x_y_hough__peak_thr2__max_clu_line_coord_1_x',
            # 'trigg_x_y_hough__peak_thr2__max_clu_line_coord_1_y',
            'x_y_hough__peak_thr1__num_clusters',
            'x_y_hough__peak_thr1__max_cluster_width',
            'x_y_hough__peak_thr1__max_cluster_height',
            'x_y_hough__peak_thr1__avg_cluster_width',
            'x_y_hough__peak_thr1__avg_cluster_height',
            'x_y_hough__peak_thr1__max_cluster_area',
            'x_y_hough__peak_thr1__avg_cluster_area',
            # 'x_y_hough__peak_thr1__max_cluster_size',
            # 'x_y_hough__peak_thr1__avg_cluster_size',
            'x_y_hough__peak_thr1__max_cluster_counts_sum',
            'x_y_hough__peak_thr1__avg_cluster_counts_sum',
            'x_y_hough__peak_thr1__max_cluster_area_width',
            'x_y_hough__peak_thr1__max_cluster_area_height',
            # 'x_y_hough__peak_thr1__max_cluster_size_width',
            # 'x_y_hough__peak_thr1__max_cluster_size_height',
            'x_y_hough__peak_thr1__max_cluster_counts_sum_width',
            'x_y_hough__peak_thr1__max_cluster_counts_sum_height',
            'x_y_hough__peak_thr1__max_peak_cluster_width',
            'x_y_hough__peak_thr1__max_peak_cluster_height',
            # 'x_y_hough__peak_thr1__avg_line_coord_0_x',
            # 'x_y_hough__peak_thr1__avg_line_coord_0_y',
            # 'x_y_hough__peak_thr1__avg_line_coord_1_x',
            # 'x_y_hough__peak_thr1__avg_line_coord_1_y',
            # 'x_y_hough__peak_thr1__max_clu_line_coord_0_x',
            # 'x_y_hough__peak_thr1__max_clu_line_coord_0_y',
            # 'x_y_hough__peak_thr1__max_clu_line_coord_1_x',
            # 'x_y_hough__peak_thr1__max_clu_line_coord_1_y',
            'x_y_hough__peak_thr2__num_clusters',
            'x_y_hough__peak_thr2__max_cluster_width',
            'x_y_hough__peak_thr2__max_cluster_height',
            'x_y_hough__peak_thr2__avg_cluster_width',
            'x_y_hough__peak_thr2__avg_cluster_height',
            'x_y_hough__peak_thr2__max_cluster_area',
            'x_y_hough__peak_thr2__avg_cluster_area',
            # 'x_y_hough__peak_thr2__max_cluster_size',
            # 'x_y_hough__peak_thr2__avg_cluster_size',
            'x_y_hough__peak_thr2__max_cluster_counts_sum',
            'x_y_hough__peak_thr2__avg_cluster_counts_sum',
            'x_y_hough__peak_thr2__max_cluster_area_width',
            'x_y_hough__peak_thr2__max_cluster_area_height',
            # 'x_y_hough__peak_thr2__max_cluster_size_width',
            # 'x_y_hough__peak_thr2__max_cluster_size_height',
            'x_y_hough__peak_thr2__max_cluster_counts_sum_width',
            'x_y_hough__peak_thr2__max_cluster_counts_sum_height',
            'x_y_hough__peak_thr2__max_peak_cluster_width',
            'x_y_hough__peak_thr2__max_peak_cluster_height',
            # 'x_y_hough__peak_thr2__avg_line_coord_0_x',
            # 'x_y_hough__peak_thr2__avg_line_coord_0_y',
            # 'x_y_hough__peak_thr2__avg_line_coord_1_x',
            # 'x_y_hough__peak_thr2__avg_line_coord_1_y',
            # 'x_y_hough__peak_thr2__max_clu_line_coord_0_x',
            # 'x_y_hough__peak_thr2__max_clu_line_coord_0_y',
            # 'x_y_hough__peak_thr2__max_clu_line_coord_1_x',
            # 'x_y_hough__peak_thr2__max_clu_line_coord_1_y',
            'x_y_hough__peak_thr3__num_clusters',
            'x_y_hough__peak_thr3__max_cluster_width',
            'x_y_hough__peak_thr3__max_cluster_height',
            'x_y_hough__peak_thr3__avg_cluster_width',
            'x_y_hough__peak_thr3__avg_cluster_height',
            'x_y_hough__peak_thr3__max_cluster_area',
            'x_y_hough__peak_thr3__avg_cluster_area',
            # 'x_y_hough__peak_thr3__max_cluster_size',
            # 'x_y_hough__peak_thr3__avg_cluster_size',
            'x_y_hough__peak_thr3__max_cluster_counts_sum',
            'x_y_hough__peak_thr3__avg_cluster_counts_sum',
            'x_y_hough__peak_thr3__max_cluster_area_width',
            'x_y_hough__peak_thr3__max_cluster_area_height',
            # 'x_y_hough__peak_thr3__max_cluster_size_width',
            # 'x_y_hough__peak_thr3__max_cluster_size_height',
            'x_y_hough__peak_thr3__max_cluster_counts_sum_width',
            'x_y_hough__peak_thr3__max_cluster_counts_sum_height',
            'x_y_hough__peak_thr3__max_peak_cluster_width',
            'x_y_hough__peak_thr3__max_peak_cluster_height',
            # 'x_y_hough__peak_thr3__avg_line_coord_0_x',
            # 'x_y_hough__peak_thr3__avg_line_coord_0_y',
            # 'x_y_hough__peak_thr3__avg_line_coord_1_x',
            # 'x_y_hough__peak_thr3__avg_line_coord_1_y',
            # 'x_y_hough__peak_thr3__max_clu_line_coord_0_x',
            # 'x_y_hough__peak_thr3__max_clu_line_coord_0_y',
            # 'x_y_hough__peak_thr3__max_clu_line_coord_1_x',
            # 'x_y_hough__peak_thr3__max_clu_line_coord_1_y',
            'trigg_gtu_x_hough__dbscan_num_clusters_above_thr1',
            'trigg_gtu_x_hough__dbscan_num_clusters_above_thr2',
            'trigg_gtu_x_hough__max_peak_rho',
            'trigg_gtu_x_hough__max_peak_phi',
            'trigg_gtu_x_hough__peak_thr1_avg_rho',
            'trigg_gtu_x_hough__peak_thr1_avg_phi',
            'trigg_gtu_x_hough__peak_thr2_avg_rho',
            'trigg_gtu_x_hough__peak_thr2_avg_phi',
            'trigg_gtu_x_hough__peak_thr1_max_clu_rho',
            'trigg_gtu_x_hough__peak_thr1_max_clu_phi',
            'trigg_gtu_x_hough__peak_thr2_max_clu_rho',
            'trigg_gtu_x_hough__peak_thr2_max_clu_phi',
            'gtu_x_hough__dbscan_num_clusters_above_thr1',
            'gtu_x_hough__dbscan_num_clusters_above_thr2',
            'gtu_x_hough__dbscan_num_clusters_above_thr3',
            'gtu_x_hough__max_peak_rho',
            'gtu_x_hough__max_peak_phi',
            'gtu_x_hough__peak_thr1_avg_rho',
            'gtu_x_hough__peak_thr1_avg_phi',
            'gtu_x_hough__peak_thr2_avg_rho',
            'gtu_x_hough__peak_thr2_avg_phi',
            'gtu_x_hough__peak_thr3_avg_rho',
            'gtu_x_hough__peak_thr3_avg_phi',
            'gtu_x_hough__peak_thr1_max_clu_rho',
            'gtu_x_hough__peak_thr1_max_clu_phi',
            'gtu_x_hough__peak_thr2_max_clu_rho',
            'gtu_x_hough__peak_thr2_max_clu_phi',
            'gtu_x_hough__peak_thr3_max_clu_rho',
            'gtu_x_hough__peak_thr3_max_clu_phi',
            'gtu_x_active_pixels_num',
            'trigg_gtu_x_sum3x3_sum',
            'trigg_gtu_x_sum3x3_norm_sum',
            'trigg_gtu_x_sum3x3_avg',
            'trigg_gtu_x_groups_num',
            'trigg_gtu_x_groups_max_size',
            'trigg_gtu_x_groups_avg_size',
            'trigg_gtu_x_groups_sum_sum_sum3x3',
            'trigg_gtu_x_groups_max_sum_sum3x3',
            'trigg_gtu_x_groups_avg_sum_sum3x3',
            'trigg_gtu_x_hough__max_peak_line_rot',
            # 'trigg_gtu_x_hough__max_peak_line_coord_0_x',
            # 'trigg_gtu_x_hough__max_peak_line_coord_0_y',
            # 'trigg_gtu_x_hough__max_peak_line_coord_1_x',
            # 'trigg_gtu_x_hough__max_peak_line_coord_1_y',
            'gtu_x_neighbourhood_size',
            'gtu_x_neighbourhood_width',
            'gtu_x_neighbourhood_height',
            'gtu_x_neighbourhood_area',
            'gtu_x_neighbourhood_counts_sum',
            'gtu_x_neighbourhood_counts_avg',
            'gtu_x_neighbourhood_counts_norm_sum',
            'gtu_x_hough__max_peak_line_rot',
            # 'gtu_x_hough__max_peak_line_coord_0_x',
            # 'gtu_x_hough__max_peak_line_coord_0_y',
            # 'gtu_x_hough__max_peak_line_coord_1_x',
            # 'gtu_x_hough__max_peak_line_coord_1_y',
            'trigg_gtu_x_hough__peak_thr1__num_clusters',
            'trigg_gtu_x_hough__peak_thr1__max_cluster_width',
            'trigg_gtu_x_hough__peak_thr1__max_cluster_height',
            'trigg_gtu_x_hough__peak_thr1__avg_cluster_width',
            'trigg_gtu_x_hough__peak_thr1__avg_cluster_height',
            'trigg_gtu_x_hough__peak_thr1__max_cluster_area',
            'trigg_gtu_x_hough__peak_thr1__avg_cluster_area',
            # 'trigg_gtu_x_hough__peak_thr1__max_cluster_size',
            # 'trigg_gtu_x_hough__peak_thr1__avg_cluster_size',
            'trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum',
            'trigg_gtu_x_hough__peak_thr1__avg_cluster_counts_sum',
            'trigg_gtu_x_hough__peak_thr1__max_cluster_area_width',
            'trigg_gtu_x_hough__peak_thr1__max_cluster_area_height',
            # 'trigg_gtu_x_hough__peak_thr1__max_cluster_size_width',
            # 'trigg_gtu_x_hough__peak_thr1__max_cluster_size_height',
            'trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width',
            'trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_height',
            'trigg_gtu_x_hough__peak_thr1__max_peak_cluster_width',
            'trigg_gtu_x_hough__peak_thr1__max_peak_cluster_height',
            # 'trigg_gtu_x_hough__peak_thr1__avg_line_coord_0_x',
            # 'trigg_gtu_x_hough__peak_thr1__avg_line_coord_0_y',
            # 'trigg_gtu_x_hough__peak_thr1__avg_line_coord_1_x',
            # 'trigg_gtu_x_hough__peak_thr1__avg_line_coord_1_y',
            # 'trigg_gtu_x_hough__peak_thr1__max_clu_line_coord_0_x',
            # 'trigg_gtu_x_hough__peak_thr1__max_clu_line_coord_0_y',
            # 'trigg_gtu_x_hough__peak_thr1__max_clu_line_coord_1_x',
            # 'trigg_gtu_x_hough__peak_thr1__max_clu_line_coord_1_y',
            'trigg_gtu_x_hough__peak_thr2__num_clusters',
            'trigg_gtu_x_hough__peak_thr2__max_cluster_width',
            'trigg_gtu_x_hough__peak_thr2__max_cluster_height',
            'trigg_gtu_x_hough__peak_thr2__avg_cluster_width',
            'trigg_gtu_x_hough__peak_thr2__avg_cluster_height',
            'trigg_gtu_x_hough__peak_thr2__max_cluster_area',
            'trigg_gtu_x_hough__peak_thr2__avg_cluster_area',
            # 'trigg_gtu_x_hough__peak_thr2__max_cluster_size',
            # 'trigg_gtu_x_hough__peak_thr2__avg_cluster_size',
            'trigg_gtu_x_hough__peak_thr2__max_cluster_counts_sum',
            'trigg_gtu_x_hough__peak_thr2__avg_cluster_counts_sum',
            'trigg_gtu_x_hough__peak_thr2__max_cluster_area_width',
            'trigg_gtu_x_hough__peak_thr2__max_cluster_area_height',
            # 'trigg_gtu_x_hough__peak_thr2__max_cluster_size_width',
            # 'trigg_gtu_x_hough__peak_thr2__max_cluster_size_height',
            'trigg_gtu_x_hough__peak_thr2__max_cluster_counts_sum_width',
            'trigg_gtu_x_hough__peak_thr2__max_cluster_counts_sum_height',
            'trigg_gtu_x_hough__peak_thr2__max_peak_cluster_width',
            'trigg_gtu_x_hough__peak_thr2__max_peak_cluster_height',
            # 'trigg_gtu_x_hough__peak_thr2__avg_line_coord_0_x',
            # 'trigg_gtu_x_hough__peak_thr2__avg_line_coord_0_y',
            # 'trigg_gtu_x_hough__peak_thr2__avg_line_coord_1_x',
            # 'trigg_gtu_x_hough__peak_thr2__avg_line_coord_1_y',
            # 'trigg_gtu_x_hough__peak_thr2__max_clu_line_coord_0_x',
            # 'trigg_gtu_x_hough__peak_thr2__max_clu_line_coord_0_y',
            # 'trigg_gtu_x_hough__peak_thr2__max_clu_line_coord_1_x',
            # 'trigg_gtu_x_hough__peak_thr2__max_clu_line_coord_1_y',
            'gtu_x_hough__peak_thr1__num_clusters',
            'gtu_x_hough__peak_thr1__max_cluster_width',
            'gtu_x_hough__peak_thr1__max_cluster_height',
            'gtu_x_hough__peak_thr1__avg_cluster_width',
            'gtu_x_hough__peak_thr1__avg_cluster_height',
            'gtu_x_hough__peak_thr1__max_cluster_area',
            'gtu_x_hough__peak_thr1__avg_cluster_area',
            # 'gtu_x_hough__peak_thr1__max_cluster_size',
            # 'gtu_x_hough__peak_thr1__avg_cluster_size',
            'gtu_x_hough__peak_thr1__max_cluster_counts_sum',
            'gtu_x_hough__peak_thr1__avg_cluster_counts_sum',
            'gtu_x_hough__peak_thr1__max_cluster_area_width',
            'gtu_x_hough__peak_thr1__max_cluster_area_height',
            # 'gtu_x_hough__peak_thr1__max_cluster_size_width',
            # 'gtu_x_hough__peak_thr1__max_cluster_size_height',
            'gtu_x_hough__peak_thr1__max_cluster_counts_sum_width',
            'gtu_x_hough__peak_thr1__max_cluster_counts_sum_height',
            'gtu_x_hough__peak_thr1__max_peak_cluster_width',
            'gtu_x_hough__peak_thr1__max_peak_cluster_height',
            # 'gtu_x_hough__peak_thr1__avg_line_coord_0_x',
            # 'gtu_x_hough__peak_thr1__avg_line_coord_0_y',
            # 'gtu_x_hough__peak_thr1__avg_line_coord_1_x',
            # 'gtu_x_hough__peak_thr1__avg_line_coord_1_y',
            # 'gtu_x_hough__peak_thr1__max_clu_line_coord_0_x',
            # 'gtu_x_hough__peak_thr1__max_clu_line_coord_0_y',
            # 'gtu_x_hough__peak_thr1__max_clu_line_coord_1_x',
            # 'gtu_x_hough__peak_thr1__max_clu_line_coord_1_y',
            'gtu_x_hough__peak_thr2__num_clusters',
            'gtu_x_hough__peak_thr2__max_cluster_width',
            'gtu_x_hough__peak_thr2__max_cluster_height',
            'gtu_x_hough__peak_thr2__avg_cluster_width',
            'gtu_x_hough__peak_thr2__avg_cluster_height',
            'gtu_x_hough__peak_thr2__max_cluster_area',
            'gtu_x_hough__peak_thr2__avg_cluster_area',
            # 'gtu_x_hough__peak_thr2__max_cluster_size',
            # 'gtu_x_hough__peak_thr2__avg_cluster_size',
            'gtu_x_hough__peak_thr2__max_cluster_counts_sum',
            'gtu_x_hough__peak_thr2__avg_cluster_counts_sum',
            'gtu_x_hough__peak_thr2__max_cluster_area_width',
            'gtu_x_hough__peak_thr2__max_cluster_area_height',
            # 'gtu_x_hough__peak_thr2__max_cluster_size_width',
            # 'gtu_x_hough__peak_thr2__max_cluster_size_height',
            'gtu_x_hough__peak_thr2__max_cluster_counts_sum_width',
            'gtu_x_hough__peak_thr2__max_cluster_counts_sum_height',
            'gtu_x_hough__peak_thr2__max_peak_cluster_width',
            'gtu_x_hough__peak_thr2__max_peak_cluster_height',
            # 'gtu_x_hough__peak_thr2__avg_line_coord_0_x',
            # 'gtu_x_hough__peak_thr2__avg_line_coord_0_y',
            # 'gtu_x_hough__peak_thr2__avg_line_coord_1_x',
            # 'gtu_x_hough__peak_thr2__avg_line_coord_1_y',
            # 'gtu_x_hough__peak_thr2__max_clu_line_coord_0_x',
            # 'gtu_x_hough__peak_thr2__max_clu_line_coord_0_y',
            # 'gtu_x_hough__peak_thr2__max_clu_line_coord_1_x',
            # 'gtu_x_hough__peak_thr2__max_clu_line_coord_1_y',
            'gtu_x_hough__peak_thr3__num_clusters',
            'gtu_x_hough__peak_thr3__max_cluster_width',
            'gtu_x_hough__peak_thr3__max_cluster_height',
            'gtu_x_hough__peak_thr3__avg_cluster_width',
            'gtu_x_hough__peak_thr3__avg_cluster_height',
            'gtu_x_hough__peak_thr3__max_cluster_area',
            'gtu_x_hough__peak_thr3__avg_cluster_area',
            # 'gtu_x_hough__peak_thr3__max_cluster_size',
            # 'gtu_x_hough__peak_thr3__avg_cluster_size',
            'gtu_x_hough__peak_thr3__max_cluster_counts_sum',
            'gtu_x_hough__peak_thr3__avg_cluster_counts_sum',
            'gtu_x_hough__peak_thr3__max_cluster_area_width',
            'gtu_x_hough__peak_thr3__max_cluster_area_height',
            # 'gtu_x_hough__peak_thr3__max_cluster_size_width',
            # 'gtu_x_hough__peak_thr3__max_cluster_size_height',
            'gtu_x_hough__peak_thr3__max_cluster_counts_sum_width',
            'gtu_x_hough__peak_thr3__max_cluster_counts_sum_height',
            'gtu_x_hough__peak_thr3__max_peak_cluster_width',
            'gtu_x_hough__peak_thr3__max_peak_cluster_height',
            # 'gtu_x_hough__peak_thr3__avg_line_coord_0_x',
            # 'gtu_x_hough__peak_thr3__avg_line_coord_0_y',
            # 'gtu_x_hough__peak_thr3__avg_line_coord_1_x',
            # 'gtu_x_hough__peak_thr3__avg_line_coord_1_y',
            # 'gtu_x_hough__peak_thr3__max_clu_line_coord_0_x',
            # 'gtu_x_hough__peak_thr3__max_clu_line_coord_0_y',
            # 'gtu_x_hough__peak_thr3__max_clu_line_coord_1_x',
            # 'gtu_x_hough__peak_thr3__max_clu_line_coord_1_y',
            'trigg_gtu_y_hough__dbscan_num_clusters_above_thr1',
            'trigg_gtu_y_hough__dbscan_num_clusters_above_thr2',
            'trigg_gtu_y_hough__max_peak_rho',
            'trigg_gtu_y_hough__max_peak_phi',
            'trigg_gtu_y_hough__peak_thr1_avg_rho',
            'trigg_gtu_y_hough__peak_thr1_avg_phi',
            'trigg_gtu_y_hough__peak_thr2_avg_rho',
            'trigg_gtu_y_hough__peak_thr2_avg_phi',
            'trigg_gtu_y_hough__peak_thr1_max_clu_rho',
            'trigg_gtu_y_hough__peak_thr1_max_clu_phi',
            'trigg_gtu_y_hough__peak_thr2_max_clu_rho',
            'trigg_gtu_y_hough__peak_thr2_max_clu_phi',
            'gtu_y_hough__dbscan_num_clusters_above_thr1',
            'gtu_y_hough__dbscan_num_clusters_above_thr2',
            'gtu_y_hough__dbscan_num_clusters_above_thr3',
            'gtu_y_hough__max_peak_rho',
            'gtu_y_hough__max_peak_phi',
            'gtu_y_hough__peak_thr1_avg_rho',
            'gtu_y_hough__peak_thr1_avg_phi',
            'gtu_y_hough__peak_thr2_avg_rho',
            'gtu_y_hough__peak_thr2_avg_phi',
            'gtu_y_hough__peak_thr3_avg_rho',
            'gtu_y_hough__peak_thr3_avg_phi',
            'gtu_y_hough__peak_thr1_max_clu_rho',
            'gtu_y_hough__peak_thr1_max_clu_phi',
            'gtu_y_hough__peak_thr2_max_clu_rho',
            'gtu_y_hough__peak_thr2_max_clu_phi',
            'gtu_y_hough__peak_thr3_max_clu_rho',
            'gtu_y_hough__peak_thr3_max_clu_phi',
            'gtu_y_active_pixels_num',
            'trigg_gtu_y_sum3x3_sum',
            'trigg_gtu_y_sum3x3_norm_sum',
            'trigg_gtu_y_sum3x3_avg',
            'trigg_gtu_y_groups_num',
            'trigg_gtu_y_groups_max_size',
            'trigg_gtu_y_groups_avg_size',
            'trigg_gtu_y_groups_sum_sum_sum3x3',
            'trigg_gtu_y_groups_max_sum_sum3x3',
            'trigg_gtu_y_groups_avg_sum_sum3x3',
            'trigg_gtu_y_hough__max_peak_line_rot',
            # 'trigg_gtu_y_hough__max_peak_line_coord_0_x',
            # 'trigg_gtu_y_hough__max_peak_line_coord_0_y',
            # 'trigg_gtu_y_hough__max_peak_line_coord_1_x',
            # 'trigg_gtu_y_hough__max_peak_line_coord_1_y',
            'gtu_y_neighbourhood_size',
            'gtu_y_neighbourhood_width',
            'gtu_y_neighbourhood_height',
            'gtu_y_neighbourhood_area',
            'gtu_y_neighbourhood_counts_sum',
            'gtu_y_neighbourhood_counts_avg',
            'gtu_y_neighbourhood_counts_norm_sum',
            'gtu_y_hough__max_peak_line_rot',
            # 'gtu_y_hough__max_peak_line_coord_0_x',
            # 'gtu_y_hough__max_peak_line_coord_0_y',
            # 'gtu_y_hough__max_peak_line_coord_1_x',
            # 'gtu_y_hough__max_peak_line_coord_1_y',
            'trigg_gtu_y_hough__peak_thr1__num_clusters',
            'trigg_gtu_y_hough__peak_thr1__max_cluster_width',
            'trigg_gtu_y_hough__peak_thr1__max_cluster_height',
            'trigg_gtu_y_hough__peak_thr1__avg_cluster_width',
            'trigg_gtu_y_hough__peak_thr1__avg_cluster_height',
            'trigg_gtu_y_hough__peak_thr1__max_cluster_area',
            'trigg_gtu_y_hough__peak_thr1__avg_cluster_area',
            # 'trigg_gtu_y_hough__peak_thr1__max_cluster_size',
            # 'trigg_gtu_y_hough__peak_thr1__avg_cluster_size',
            'trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum',
            'trigg_gtu_y_hough__peak_thr1__avg_cluster_counts_sum',
            'trigg_gtu_y_hough__peak_thr1__max_cluster_area_width',
            'trigg_gtu_y_hough__peak_thr1__max_cluster_area_height',
            # 'trigg_gtu_y_hough__peak_thr1__max_cluster_size_width',
            # 'trigg_gtu_y_hough__peak_thr1__max_cluster_size_height',
            'trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width',
            'trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_height',
            'trigg_gtu_y_hough__peak_thr1__max_peak_cluster_width',
            'trigg_gtu_y_hough__peak_thr1__max_peak_cluster_height',
            # 'trigg_gtu_y_hough__peak_thr1__avg_line_coord_0_x',
            # 'trigg_gtu_y_hough__peak_thr1__avg_line_coord_0_y',
            # 'trigg_gtu_y_hough__peak_thr1__avg_line_coord_1_x',
            # 'trigg_gtu_y_hough__peak_thr1__avg_line_coord_1_y',
            # 'trigg_gtu_y_hough__peak_thr1__max_clu_line_coord_0_x',
            # 'trigg_gtu_y_hough__peak_thr1__max_clu_line_coord_0_y',
            # 'trigg_gtu_y_hough__peak_thr1__max_clu_line_coord_1_x',
            # 'trigg_gtu_y_hough__peak_thr1__max_clu_line_coord_1_y',
            'trigg_gtu_y_hough__peak_thr2__num_clusters',
            'trigg_gtu_y_hough__peak_thr2__max_cluster_width',
            'trigg_gtu_y_hough__peak_thr2__max_cluster_height',
            'trigg_gtu_y_hough__peak_thr2__avg_cluster_width',
            'trigg_gtu_y_hough__peak_thr2__avg_cluster_height',
            'trigg_gtu_y_hough__peak_thr2__max_cluster_area',
            'trigg_gtu_y_hough__peak_thr2__avg_cluster_area',
            # 'trigg_gtu_y_hough__peak_thr2__max_cluster_size',
            # 'trigg_gtu_y_hough__peak_thr2__avg_cluster_size',
            'trigg_gtu_y_hough__peak_thr2__max_cluster_counts_sum',
            'trigg_gtu_y_hough__peak_thr2__avg_cluster_counts_sum',
            'trigg_gtu_y_hough__peak_thr2__max_cluster_area_width',
            'trigg_gtu_y_hough__peak_thr2__max_cluster_area_height',
            # 'trigg_gtu_y_hough__peak_thr2__max_cluster_size_width',
            # 'trigg_gtu_y_hough__peak_thr2__max_cluster_size_height',
            'trigg_gtu_y_hough__peak_thr2__max_cluster_counts_sum_width',
            'trigg_gtu_y_hough__peak_thr2__max_cluster_counts_sum_height',
            'trigg_gtu_y_hough__peak_thr2__max_peak_cluster_width',
            'trigg_gtu_y_hough__peak_thr2__max_peak_cluster_height',
            # 'trigg_gtu_y_hough__peak_thr2__avg_line_coord_0_x',
            # 'trigg_gtu_y_hough__peak_thr2__avg_line_coord_0_y',
            # 'trigg_gtu_y_hough__peak_thr2__avg_line_coord_1_x',
            # 'trigg_gtu_y_hough__peak_thr2__avg_line_coord_1_y',
            # 'trigg_gtu_y_hough__peak_thr2__max_clu_line_coord_0_x',
            # 'trigg_gtu_y_hough__peak_thr2__max_clu_line_coord_0_y',
            # 'trigg_gtu_y_hough__peak_thr2__max_clu_line_coord_1_x',
            # 'trigg_gtu_y_hough__peak_thr2__max_clu_line_coord_1_y',
            'gtu_y_hough__peak_thr1__num_clusters',
            'gtu_y_hough__peak_thr1__max_cluster_width',
            'gtu_y_hough__peak_thr1__max_cluster_height',
            'gtu_y_hough__peak_thr1__avg_cluster_width',
            'gtu_y_hough__peak_thr1__avg_cluster_height',
            'gtu_y_hough__peak_thr1__max_cluster_area',
            'gtu_y_hough__peak_thr1__avg_cluster_area',
            # 'gtu_y_hough__peak_thr1__max_cluster_size',
            # 'gtu_y_hough__peak_thr1__avg_cluster_size',
            'gtu_y_hough__peak_thr1__max_cluster_counts_sum',
            'gtu_y_hough__peak_thr1__avg_cluster_counts_sum',
            'gtu_y_hough__peak_thr1__max_cluster_area_width',
            'gtu_y_hough__peak_thr1__max_cluster_area_height',
            # 'gtu_y_hough__peak_thr1__max_cluster_size_width',
            # 'gtu_y_hough__peak_thr1__max_cluster_size_height',
            'gtu_y_hough__peak_thr1__max_cluster_counts_sum_width',
            'gtu_y_hough__peak_thr1__max_cluster_counts_sum_height',
            'gtu_y_hough__peak_thr1__max_peak_cluster_width',
            'gtu_y_hough__peak_thr1__max_peak_cluster_height',
            # 'gtu_y_hough__peak_thr1__avg_line_coord_0_x',
            # 'gtu_y_hough__peak_thr1__avg_line_coord_0_y',
            # 'gtu_y_hough__peak_thr1__avg_line_coord_1_x',
            # 'gtu_y_hough__peak_thr1__avg_line_coord_1_y',
            # 'gtu_y_hough__peak_thr1__max_clu_line_coord_0_x',
            # 'gtu_y_hough__peak_thr1__max_clu_line_coord_0_y',
            # 'gtu_y_hough__peak_thr1__max_clu_line_coord_1_x',
            # 'gtu_y_hough__peak_thr1__max_clu_line_coord_1_y',
            'gtu_y_hough__peak_thr2__num_clusters',
            'gtu_y_hough__peak_thr2__max_cluster_width',
            'gtu_y_hough__peak_thr2__max_cluster_height',
            'gtu_y_hough__peak_thr2__avg_cluster_width',
            'gtu_y_hough__peak_thr2__avg_cluster_height',
            'gtu_y_hough__peak_thr2__max_cluster_area',
            'gtu_y_hough__peak_thr2__avg_cluster_area',
            # 'gtu_y_hough__peak_thr2__max_cluster_size',
            # 'gtu_y_hough__peak_thr2__avg_cluster_size',
            'gtu_y_hough__peak_thr2__max_cluster_counts_sum',
            'gtu_y_hough__peak_thr2__avg_cluster_counts_sum',
            'gtu_y_hough__peak_thr2__max_cluster_area_width',
            'gtu_y_hough__peak_thr2__max_cluster_area_height',
            # 'gtu_y_hough__peak_thr2__max_cluster_size_width',
            # 'gtu_y_hough__peak_thr2__max_cluster_size_height',
            'gtu_y_hough__peak_thr2__max_cluster_counts_sum_width',
            'gtu_y_hough__peak_thr2__max_cluster_counts_sum_height',
            'gtu_y_hough__peak_thr2__max_peak_cluster_width',
            'gtu_y_hough__peak_thr2__max_peak_cluster_height',
            # 'gtu_y_hough__peak_thr2__avg_line_coord_0_x',
            # 'gtu_y_hough__peak_thr2__avg_line_coord_0_y',
            # 'gtu_y_hough__peak_thr2__avg_line_coord_1_x',
            # 'gtu_y_hough__peak_thr2__avg_line_coord_1_y',
            # 'gtu_y_hough__peak_thr2__max_clu_line_coord_0_x',
            # 'gtu_y_hough__peak_thr2__max_clu_line_coord_0_y',
            # 'gtu_y_hough__peak_thr2__max_clu_line_coord_1_x',
            # 'gtu_y_hough__peak_thr2__max_clu_line_coord_1_y',
            'gtu_y_hough__peak_thr3__num_clusters',
            'gtu_y_hough__peak_thr3__max_cluster_width',
            'gtu_y_hough__peak_thr3__max_cluster_height',
            'gtu_y_hough__peak_thr3__avg_cluster_width',
            'gtu_y_hough__peak_thr3__avg_cluster_height',
            'gtu_y_hough__peak_thr3__max_cluster_area',
            'gtu_y_hough__peak_thr3__avg_cluster_area',
            # 'gtu_y_hough__peak_thr3__max_cluster_size',
            # 'gtu_y_hough__peak_thr3__avg_cluster_size',
            'gtu_y_hough__peak_thr3__max_cluster_counts_sum',
            'gtu_y_hough__peak_thr3__avg_cluster_counts_sum',
            'gtu_y_hough__peak_thr3__max_cluster_area_width',
            'gtu_y_hough__peak_thr3__max_cluster_area_height',
            # 'gtu_y_hough__peak_thr3__max_cluster_size_width',
            # 'gtu_y_hough__peak_thr3__max_cluster_size_height',
            'gtu_y_hough__peak_thr3__max_cluster_counts_sum_width',
            'gtu_y_hough__peak_thr3__max_cluster_counts_sum_height',
            'gtu_y_hough__peak_thr3__max_peak_cluster_width',
            'gtu_y_hough__peak_thr3__max_peak_cluster_height',
            # 'gtu_y_hough__peak_thr3__avg_line_coord_0_x',
            # 'gtu_y_hough__peak_thr3__avg_line_coord_0_y',
            # 'gtu_y_hough__peak_thr3__avg_line_coord_1_x',
            # 'gtu_y_hough__peak_thr3__avg_line_coord_1_y',
            # 'gtu_y_hough__peak_thr3__max_clu_line_coord_0_x',
            # 'gtu_y_hough__peak_thr3__max_clu_line_coord_0_y',
            # 'gtu_y_hough__peak_thr3__max_clu_line_coord_1_x',
            # 'gtu_y_hough__peak_thr3__max_clu_line_coord_1_y',

        ]
    return columns_for_classification


def get_select_simu_events_query_format(num_frames_signals_ge_bg__ge=3, num_frames_signals_ge_bg__le=999, num_triggered_pixels__ge=0, num_triggered_pixels__le=99999, source_data_type_num=3, etruth_theta=0.261799):
    select_simu_events_query_format = '''SELECT 
          {columns}
        FROM spb_processing_event_ver2
        JOIN simu_event_spb_proc USING (event_id) 
        JOIN simu_event USING (simu_event_id) 
        JOIN simu_event_spb_proc_additional_info USING (relation_id) 
        WHERE 
        ''' + '''
         source_data_type_num = {source_data_type_num:d}
         AND etruth_truetheta > {etruth_theta:.4f}
         AND num_triggered_pixels BETWEEN {num_triggered_pixels__ge:d} AND {num_triggered_pixels__le:d}
         AND num_frames_signals_ge_bg BETWEEN {num_frames_signals_ge_bg__ge:d} AND {num_frames_signals_ge_bg__le:d} 
         '''.format(num_triggered_pixels__ge=num_triggered_pixels__ge, num_triggered_pixels__le=num_triggered_pixels__le,
                num_frames_signals_ge_bg__ge=num_frames_signals_ge_bg__ge, num_frames_signals_ge_bg__le=num_frames_signals_ge_bg__le,
                source_data_type_num=source_data_type_num, etruth_theta=etruth_theta) + '''
        ORDER BY num_triggered_pixels ASC, event_id ASC 
        OFFSET {offset:d} LIMIT {limit:d}
        ;'''
    return select_simu_events_query_format


def get_select_simu_events_other_bgf_query_format(t1_source_data_type=3, t2_source_data_type=5, gtu_in_packet_diff=5, num_frames_signals_ge_bg__ge=3, num_frames_signals_ge_bg__le=999, num_triggered_pixels__ge=8, num_triggered_pixels__le=800, etruth_theta=0.261799):
    select_simu_events_query_format = '''SELECT 
    FROM spb_processing_event_ver2 AS t1 
    JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    JOIN simu_event_spb_proc ON (t2.event_id = simu_event_spb_proc.event_id) 
    JOIN simu_event AS t2_simu_event USING (simu_event_id) 
    JOIN simu_event_spb_proc_additional_info AS t2_additional USING (relation_id) 
    WHERE 
         t1.source_data_type_num={t1_source_data_type_num:d} AND t2.source_data_type_num={t2_source_data_type_num:d}  AND abs(t1.gtu_in_packet - t2.gtu_in_packet) < {gtu_in_packet_diff:d} 
     AND t2_simu_event.etruth_truetheta > {etruth_theta:.4f} AND t2.num_triggered_pixels BETWEEN {num_triggered_pixels__ge:d} AND {num_triggered_pixels__le:d}
     AND t2_additional.num_frames_signals_ge_bg BETWEEN {num_frames_signals_ge_bg__ge:d} AND {num_frames_signals_ge_bg__le:d} 
    ORDER BY  t1.num_triggered_pixels ASC, t1.source_file_acquisition_full ASC, t1.event_id ASC 
    OFFSET {offset:d} LIMIT {limit:d}
    ;'''
    return select_simu_events_query_format


def select_events(cur, query_format, columns, offset=0, limit=100000, check_selected_columns=True, column_prefix=''):

    if not column_prefix:
        query_columns = columns
    else:
        query_columns = [column_prefix + column_name for column_name in columns]

    q = query_format.format(columns=','.join(query_columns), offset=offset, limit=limit)

    print("Executing query: ")
    print(q)

    cur.execute(q)
    all_rows = cur.fetchall()

    print('# Selected {} entries'.format(len(all_rows)))

    all_columns = list(map(lambda x: x[0], cur.description))

    if check_selected_columns and all_columns != columns:
        raise Exception('Selected columns are not equal to expected columns')

    return all_rows, all_columns


def select_training_data__visible_showers(cur, columns):
    all_rows, all_columns = select_events(cur, get_select_simu_events_query_format(3, 999, 3, 800, 3), columns, limit=100000)
    return all_rows


def select_training_data__invisible_showers(cur, columns):
    all_rows, all_columns = select_events(cur, get_select_simu_events_query_format(0, 2, 0, 1), columns, limit=100000)
    return all_rows


def select_training_data__visible_showers_other_bgf(cur, columns):
    all_rows, all_columns = select_events(cur, get_select_simu_events_other_bgf_query_format(), columns, limit=100000, column_prefix='t1.')
    return all_columns


def select_training_data__low_energy_in_pmt(cur, columns):
    select_low_energy_in_pmt_query_format = '''
    SELECT {columns} FROM spb_processing_event_ver2
    WHERE   
            source_data_type_num = 1
        AND num_gtu < 14
        AND gtu_in_packet BETWEEN 20 AND 50
        AND num_triggered_pixels BETWEEN 8 AND 500 
        AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 <= 2
        AND
         (
               (trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 50 AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
            OR (trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 70 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
            OR (trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 70)
         )
        AND
           ( abs(gtu_y_hough__peak_thr3_avg_phi)  BETWEEN -0.174 AND 0.1745  OR  abs(gtu_x_hough__peak_thr3_avg_phi) BETWEEN -0.174 AND 0.1745 )
    ORDER BY
        num_triggered_pixels DESC, event_id ASC 
    OFFSET {offset} LIMIT {limit}
    '''
    all_rows, all_columns = select_events(cur, select_low_energy_in_pmt_query_format, columns, limit=100000)
    return all_rows


def select_training_data__led(cur, columns):
    select_led_query_format = '''
    SELECT {columns} FROM spb_processing_event_ver2
    WHERE   
            source_data_type_num = 1
        AND num_triggered_pixels > 500 
    ORDER BY
        num_triggered_pixels DESC, event_id ASC 
    OFFSET {offset:d} LIMIT {limit:d}
    '''
    all_rows, all_columns = select_events(cur, select_led_query_format, columns, limit=100000)
    return all_rows


def load_train_test(cur, columns, random_state=None, get_class_1_func=select_training_data__visible_showers):
    X = np.array(get_class_1_func(cur, columns), dtype=np.float32)
    len_class_1 = len(X)

    a = np.array(select_training_data__invisible_showers(cur, columns), dtype=np.float32)
    if len(a) > 0:
        X = np.append(X, a, axis=0)
    a = np.array(select_training_data__low_energy_in_pmt(cur, columns), dtype=np.float32)
    if len(a) > 0:
        X = np.append(X, a, axis=0)
    a = np.array(select_training_data__led(cur, columns), dtype=np.float32)
    if len(a) > 0:
        X = np.append(X, a, axis=0)

    y = np.zeros(len(X), dtype=np.int8)
    y[:len_class_1] = 1

    return sklearn.model_selection.train_test_split(X, y, random_state=random_state)


# def fit_classifier(cur, classifier, X_train, y_train, random_state=None, outfile_pathname=None):
#     pass

# def score_classifier(classifier):
#     score = classifier.score(X_test, y_test)


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
    parser.add_argument('-c','--classifier', default='adaboost')
    parser.add_argument('--overwrite', type=str2bool_argparse, default=False, help='Overwrite output model file')
    parser.add_argument('--read', default="", help='Only read exiting model')

    # parser.add_argument('--print-queries', type=str2bool_argparse, default='Only print queries')

    args = parser.parse_args(argv)

    classifier = None
    if not args.password:
        args.password = getpass.getpass()

    con = pg.connect(dbname=args.dbname, user=args.user, password=args.password, host=args.host)
    cur = con.cursor()

    columns = get_columns_for_classification()

    if args.get_class1_func == 0:
        get_class1_func = get_select_simu_events_query_format
    elif args.get_class1_func == 1:
        get_class1_func = get_select_simu_events_other_bgf_query_format
    else:
        raise Exception('Invalid class1 func')

    # http://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
    X_train, X_test, y_train, y_test = load_train_test(cur, columns, args.random_state, get_class_1_func=get_class1_func)

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
                    hashlib.md5((str(metaclassifier_params) + str(classifier_params)).encode()).hexdigest()
                ))
            else:
                outfile_pathname = args.out

            if os.path.exists(outfile_pathname) and not args.overwrite and not args.read:
                raise Exception('Model file "{}" already exists'.format(outfile_pathname))

        # cross_val_score # k-fold ... http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
        classifier.fit(X_train, y_train)

        if outfile_pathname:
            sklearn.externals.joblib.dump(classifier, outfile_pathname)
    else:
        classifier = sklearn.externals.joblib.load(args.read)


    score = classifier.score(X_test, y_test)

    print("Score for {}: {}".format(classifier.__class__.__name__, score))




if __name__ == '__main__':
    main(sys.argv[1:])
