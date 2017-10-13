import sys
import os
import subprocess

app_base_dir = '/home/eusobg/EUSO-SPB/euso-spb-patt-reco-v1'
if app_base_dir not in sys.path:
    sys.path.append(app_base_dir)

import re
# import collections
import numpy as np
import psycopg2 as pg
import pandas as pd
import pandas.io.sql as psql
import getpass
import matplotlib as mpl
import argparse
import glob
# from tqdm import tqdm
import traceback
import hashlib

from utility_funtions import str2bool_argparse


def __check_agg():
    show_plots = False
    if '--show-plots' in sys.argv[1:]:
        args_parser = argparse.ArgumentParser(description='')
        args_parser.add_argument('--show-plots',type=str2bool_argparse,default=False,help='If true, plots are only showed in windows')
        args , _ = args_parser.parse_known_args(sys.argv[1:])
        show_plots = args.show_plots
    if not show_plots:
        mpl.use('Agg')


__check_agg()


mpl.rcParams['figure.dpi'] = 150

import matplotlib.pyplot as plt

#import ROOT

# import tool.npy_frames_visualization as npy_vis
import tool.acqconv
import data_analysis_utils


def get_selection_rules():
    cond_selection_rules = '''
        num_gtu BETWEEN 10 AND 40
    AND num_triggered_pixels BETWEEN 10 AND 150
    
    AND abs(gtu_in_packet - 40) <= 10
    
    AND (abs(gtu_y_hough__peak_thr2_avg_phi) > 0.084 OR abs(gtu_x_hough__peak_thr2_avg_phi) > 0.084)
    AND (abs(gtu_y_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806  OR  abs(gtu_x_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806 )

    AND
    (   
            (((trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 50 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 = 1) OR trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 45) AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 45) 
        OR (trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND ((trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 50 AND trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 = 1) OR trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 45) AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 45) 
        OR (trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND ((trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 90 AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 = 1) OR trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 50))
        OR (trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 15 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 15)
        OR (trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 15 AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 20)
        OR (trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 15 AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 20)
    )

    AND
    (
        (((gtu_y_hough__dbscan_num_clusters_above_thr1 = 1 AND gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 45) OR gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40) AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
        OR (gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND ((gtu_x_hough__dbscan_num_clusters_above_thr1 = 1 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 45) OR gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40) AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
        OR (gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND ((x_y_hough__dbscan_num_clusters_above_thr1 = 1 AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 60) OR x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 50))
        OR (gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 15 AND gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 15)
        OR (gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 15 AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 20)
        OR (gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 15 AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 20)
    ) 

    AND (
        (trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 4  AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 6)
        OR (trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 5  AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 4)
        OR (trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 5 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 5  AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 4)
        OR (trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 2 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 2)
        OR (trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 2 AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 3)
        OR (trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 2 AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 3)
    )

    AND gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND gtu_x_hough__dbscan_num_clusters_above_thr1 < 4 AND x_y_hough__dbscan_num_clusters_above_thr1 < 5

    AND (
        (trigg_gtu_y_hough__dbscan_num_clusters_above_thr2 < 3 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr2 < 3 AND trigg_x_y_hough__dbscan_num_clusters_above_thr2 < 5)
        OR (trigg_gtu_y_hough__dbscan_num_clusters_above_thr2 < 3 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr2 < 5 AND trigg_x_y_hough__dbscan_num_clusters_above_thr2 < 3)
        OR (trigg_gtu_y_hough__dbscan_num_clusters_above_thr2 < 5 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr2 < 3 AND trigg_x_y_hough__dbscan_num_clusters_above_thr2 < 3)
        OR (trigg_gtu_y_hough__dbscan_num_clusters_above_thr2 < 2 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr2 < 2)
        OR (trigg_gtu_y_hough__dbscan_num_clusters_above_thr2 < 2 AND trigg_x_y_hough__dbscan_num_clusters_above_thr2 < 2)
        OR (trigg_gtu_x_hough__dbscan_num_clusters_above_thr2 < 2 AND trigg_x_y_hough__dbscan_num_clusters_above_thr2 < 2)
    )

    AND (
        (gtu_y_hough__dbscan_num_clusters_above_thr2 < 3 AND gtu_x_hough__dbscan_num_clusters_above_thr2 < 3 AND x_y_hough__dbscan_num_clusters_above_thr2 < 5)
        OR (gtu_y_hough__dbscan_num_clusters_above_thr2 < 3 AND gtu_x_hough__dbscan_num_clusters_above_thr2 < 4 AND x_y_hough__dbscan_num_clusters_above_thr2 < 3)
        OR (gtu_y_hough__dbscan_num_clusters_above_thr2 < 4 AND gtu_x_hough__dbscan_num_clusters_above_thr2 < 3 AND x_y_hough__dbscan_num_clusters_above_thr2 < 3)
    )
    '''

    # '''
    #       num_gtu BETWEEN 13 AND 40
    #   AND num_triggered_pixels BETWEEN 10 AND 150
    #   AND (abs(gtu_y_hough__peak_thr2_avg_phi) > 0.261799 OR abs(gtu_x_hough__peak_thr2_avg_phi) > 0.261799)
    #   AND ( abs(gtu_y_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806  OR  abs(gtu_x_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806 )

    #   AND
    #   (
    #          (((trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 45 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 = 1) OR trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 35) AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 45)
    #       OR (trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND ((trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 45 AND trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 = 1) OR trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40) AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 45)
    #       OR (trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND ((trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 90 AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 = 1) OR trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 50))
    #   )

    #   AND
    #   (
    #        (((gtu_y_hough__dbscan_num_clusters_above_thr1 = 1 AND gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 45) OR gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40) AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
    #     OR (gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND ((gtu_x_hough__dbscan_num_clusters_above_thr1 = 1 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 45) OR gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40) AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
    #     OR (gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 35 AND ((x_y_hough__dbscan_num_clusters_above_thr1 = 1 AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 60) OR x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 50))
    #   )

    #   AND trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 3 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 3  AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 4

    #   AND gtu_y_hough__dbscan_num_clusters_above_thr1 < 3 AND gtu_x_hough__dbscan_num_clusters_above_thr1 < 3

    #   AND trigg_gtu_y_hough__dbscan_num_clusters_above_thr2 < 3 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr2 < 2 AND trigg_x_y_hough__dbscan_num_clusters_above_thr2 < 4

    #   AND gtu_y_hough__dbscan_num_clusters_above_thr2 < 3 AND gtu_x_hough__dbscan_num_clusters_above_thr2 < 3 /**/
    # '''

    # '''
    #   x_y_active_pixels_num > 1750 /* unnecesarry for simulated events */

    #   AND num_gtu BETWEEN 8 AND 40
    #   AND num_triggered_pixels BETWEEN 10 AND 150
    #   AND (abs(gtu_y_hough__peak_thr2_avg_phi) > 0.174533 OR abs(gtu_x_hough__peak_thr2_avg_phi) > 0.174533)
    #   AND ( abs(gtu_y_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806  OR  abs(gtu_x_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806 )

    #  AND
    #  (
    #        (trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 90 AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
    #     OR (trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 90 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
    #     OR (trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 90)
    #  )

    #   AND
    #   (
    #        (gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 90 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
    #     OR (gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 90 AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
    #     OR (gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 90)
    #   )

    #   AND trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND gtu_x_hough__dbscan_num_clusters_above_thr1 < 4

    # '''

    # '''
    #   x_y_active_pixels_num > 1750 /* unnecesarry for simulated events */

    #   AND num_gtu BETWEEN 8 AND 40
    #   AND num_triggered_pixels BETWEEN 10 AND 150
    #   AND (abs(gtu_y_hough__peak_thr2_avg_phi) > 0.174533 OR abs(gtu_x_hough__peak_thr2_avg_phi) > 0.174533)
    #   AND ( abs(gtu_y_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806  OR  abs(gtu_x_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806 )

    #  AND
    #  (
    #        (trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 90 AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
    #     OR (trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 90 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40)
    #     OR (trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width <= 40 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width <= 90)
    #  )

    #   AND gtu_y_hough__peak_thr1__max_cluster_counts_sum_width < 41 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width < 41

    #   AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width < 20 /* THIS SHOULD BE REMOVED */
    #   AND x_y_hough__peak_thr1__max_cluster_counts_sum_width < 25

    #   AND trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND gtu_x_hough__dbscan_num_clusters_above_thr1 < 4

    # '''

    # '''
    #   AND x_y_active_pixels_num > 1750 /* unnecesarry for simulated events */

    #   AND num_gtu BETWEEN 11 AND 40
    #   AND num_triggered_pixels BETWEEN 10 AND 150
    #   AND (abs(gtu_y_hough__peak_thr2_avg_phi) > 0.174533 OR abs(gtu_x_hough__peak_thr2_avg_phi) > 0.174533)
    #   AND ( abs(gtu_y_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806  OR  abs(gtu_x_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806 )

    #   AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width < 41 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width < 41
    #   AND trigg_gtu_y_hough__peak_thr2__max_cluster_counts_sum_width < 20 AND trigg_gtu_y_hough__peak_thr2__max_cluster_counts_sum_height < 20
    #   AND trigg_gtu_x_hough__peak_thr2__max_cluster_counts_sum_width < 20 AND trigg_gtu_x_hough__peak_thr2__max_cluster_counts_sum_height < 20

    #   AND gtu_y_hough__peak_thr1__max_cluster_counts_sum_width < 41 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width < 41

    #   AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width < 20
    #   AND x_y_hough__peak_thr1__max_cluster_counts_sum_width < 25

    #   AND trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND gtu_x_hough__dbscan_num_clusters_above_thr1 < 4

    # '''

    # '''
    #   AND x_y_active_pixels_num > 1750 /* unnecesarry for simulated events */

    #   AND num_gtu BETWEEN 11 AND 40
    #   AND num_triggered_pixels >= 10
    #   AND (abs(gtu_y_hough__peak_thr2_avg_phi) > 0.174533 OR abs(gtu_x_hough__peak_thr2_avg_phi) > 0.174533)
    #   AND num_triggered_pixels < 150
    #   AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width < 20
    #   AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width < 41 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width < 41
    #   AND x_y_hough__peak_thr1__max_cluster_counts_sum_width < 25
    #   AND trigg_gtu_y_hough__peak_thr2__max_cluster_counts_sum_width < 20 AND trigg_gtu_y_hough__peak_thr2__max_cluster_counts_sum_height < 20
    #   AND trigg_gtu_x_hough__peak_thr2__max_cluster_counts_sum_width < 20 AND trigg_gtu_x_hough__peak_thr2__max_cluster_counts_sum_height < 20
    #   AND gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND gtu_x_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND ( abs(gtu_y_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806  OR  abs(gtu_x_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806 )    /* TODO this rule might be very wrong !!!! */
    #   AND trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND gtu_y_hough__peak_thr1__max_cluster_counts_sum_width < 41 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width < 41
    # '''

    # '''
    #   AND num_gtu BETWEEN 12 AND 40
    #   AND num_triggered_pixels > 10 /* higher restriction than copy1 */
    #   /*AND max_n_persist > 4 */
    #   AND (abs(gtu_y_hough__peak_thr2_avg_phi) > 0.174533 OR abs(gtu_x_hough__peak_thr2_avg_phi) > 0.174533)
    #   AND num_triggered_pixels < 800
    #   AND trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width < 15
    #   AND trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width < 41 AND trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width < 41
    #   AND x_y_hough__peak_thr1__max_cluster_counts_sum_width < 25
    #   AND trigg_gtu_y_hough__peak_thr2__max_cluster_counts_sum_width < 20 AND trigg_gtu_y_hough__peak_thr2__max_cluster_counts_sum_height < 20
    #   AND trigg_gtu_x_hough__peak_thr2__max_cluster_counts_sum_width < 20 AND trigg_gtu_x_hough__peak_thr2__max_cluster_counts_sum_height < 20
    #   AND gtu_y_hough__dbscan_num_clusters_above_thr1 < 4 AND gtu_x_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 4
    #   AND ( abs(gtu_y_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806  OR  abs(gtu_x_hough__peak_thr3_avg_phi) NOT BETWEEN 1.48353 AND 1.65806 )
    #   AND trigg_gtu_y_hough__dbscan_num_clusters_above_thr1 < 3 AND trigg_gtu_x_hough__dbscan_num_clusters_above_thr1 < 3
    #   AND gtu_y_hough__peak_thr1__max_cluster_counts_sum_width < 41 AND gtu_x_hough__peak_thr1__max_cluster_counts_sum_width < 41
    # '''

    return cond_selection_rules


def get_conn(**kwargs):
    con = pg.connect(**kwargs) # "dbname=eusospb_data user=eusospb password= host=localhost"
    cur = con.cursor()
    return con, cur


def ensure_ext(base_file_name, ext='.png'):
    if not base_file_name.endswith(ext):
        return "{}{}".format(base_file_name, ext)
    return base_file_name


def save_csv(df,save_txt_dir,base_file_name, sep='\t'):
    if save_txt_dir:
        csv_path = os.path.join(save_txt_dir, ensure_ext(base_file_name,".{}sv".format('t' if sep=='\t' else 'c')))
        print('SAVING CSV {}'.format(csv_path))
        df.to_csv(csv_path, sep=sep)
        return csv_path
    return None


def get_spb_processing_event_ver2_columns(cur, queries_log=None):
    q = 'SELECT * FROM spb_processing_event_ver2 LIMIT 1'
    if queries_log:
        queries_log.write(q)
    cur.execute(q)
    spb_processing_event_ver2_columns = list(map(lambda x: x[0], cur.description))

    return spb_processing_event_ver2_columns


def get_all_bgf05_and_bgf1_simu_events__packet_count_by_energy(con, queries_log=None):
    all_bgf05_and_bgf1_simu_events__packet_count_by_energy_query = '''
    SELECT COUNT(sq.count_event_id) AS count_packets, sq.etruth_trueenergy AS etruth_trueenergy 
    FROM (
    SELECT COUNT(t1.event_id) AS count_event_id, t1.source_file_acquisition_full, etruth_trueenergy 
    FROM simu_event_spb_proc 
    JOIN simu_event USING(simu_event_id) 
    JOIN spb_processing_event_ver2 AS t1 USING(event_id) 
    JOIN spb_processing_event_ver2 AS t2 USING(source_file_acquisition_full) 
    WHERE t1.source_data_type_num=3 AND t2.source_data_type_num=5  AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4))
    GROUP BY etruth_trueenergy, t1.source_file_acquisition_full, t1.packet_id
    ) as sq GROUP BY etruth_trueenergy ORDER BY etruth_trueenergy;
    '''
    if queries_log:
        queries_log.write(all_bgf05_and_bgf1_simu_events__packet_count_by_energy_query)
    return psql.read_sql( all_bgf05_and_bgf1_simu_events__packet_count_by_energy_query, con)


def fig_saving_msg(path):
    print("SAVING FIGURE: {}".format(path))


def print_len(l, label, comment=''):
    print("len({}) = {}{}".format(label, len(l), '' if not comment else '   // {}'.format(comment)))


def save_figure(fig, *path_parts):
    path = ensure_ext(os.path.join(*path_parts), '.png')
    fig_saving_msg(path)
    fig.savefig(path)

def vis_df_etruth_trueenergy_count_packets(all_bgf05_and_bgf1_simu_events__packet_count_by_energy, save_fig_dir, fig_file_name='all_bgf05_and_bgf1_simu_events__count_packets_by_energy.png'):
    if len(all_bgf05_and_bgf1_simu_events__packet_count_by_energy) > 0:
        ax_all_bgf05_and_bgf1_simu_events__packet_count_by_energy = all_bgf05_and_bgf1_simu_events__packet_count_by_energy.plot(x='etruth_trueenergy', y='count_packets')
        if save_fig_dir is not None:
            save_figure(ax_all_bgf05_and_bgf1_simu_events__packet_count_by_energy.get_figure(), save_fig_dir, fig_file_name)
        else:
            plt.show()


def get_count_simu_entries_within_cond(con, cond_selection_rules, queries_log=None):
    q = '''
    SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 3 AND {conds} 
    '''.format(conds=cond_selection_rules)
    if queries_log:
        queries_log.write(q)
    count_simu_entries_within_cond = psql.read_sql(q, con)
    return count_simu_entries_within_cond['count'].iloc[0]

# to_csv(path, sep='\t')

def get_simu_entries_within_cond_bgf05_and_bgf1__only_1bgf_lt_05bgf(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log=None):
    cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)
    q = '''
    SELECT t1.event_id, t1.source_data_type_num, t1.gtu_in_packet, t1.num_gtu, t2.gtu_in_packet AS t2_gtu_in_packet, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full, t1.source_file_trigger FROM spb_processing_event_ver2 AS t1 
    JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    WHERE t1.source_data_type_num = 3 AND t2.source_data_type_num = 5  AND (/*(t1.gtu_in_packet < t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu) OR*/ (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu))   
    AND {conds} 
    ORDER BY t1.event_id ASC
    '''.format(conds=cond_selection_rules_t1_prefixed)
    # print(q)
    if queries_log:
        queries_log.write(cond_selection_rules_t1_prefixed)
    simu_entries_within_cond_bgf05_and_bgf1 = psql.read_sql(q, con)
    return simu_entries_within_cond_bgf05_and_bgf1


def get_simu_entries_within_cond_bgf05_and_bgf1(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log=None):
    cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)
    q = '''
    SELECT t1.event_id, t1.source_data_type_num, t1.gtu_in_packet, t1.num_gtu, t2.gtu_in_packet AS t2_gtu_in_packet, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full, t1.source_file_trigger FROM spb_processing_event_ver2 AS t1 
    JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    WHERE t1.source_data_type_num = 3 AND t2.source_data_type_num = 5  AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4)) 
    AND {conds} 
    ORDER BY t1.event_id ASC
    '''.format(conds=cond_selection_rules_t1_prefixed)
    #print(q)
    if queries_log:
        queries_log.write(q)
    simu_entries_within_cond_bgf05_and_bgf1 = psql.read_sql(q, con)
    #print(len(simu_entries_within_cond_bgf05_and_bgf1))
    return simu_entries_within_cond_bgf05_and_bgf1


def get_simu_entries_within_cond_bgf05_and_bgf1_v2(con, cond_selection_rules, queries_log=None):
    # t1.event_id AS t1_event_id, t2.event_id AS t2_event_id, t1.source_data_type_num AS t1_source_data_type_num, t2.source_data_type_num AS t2_source_data_type_num, t1.global_gtu AS t1_global_gtu, t2.global_gtu AS t2_global_gtu,
    # t1.gtu_in_packet AS t1_gtu_in_packet, t2.gtu_in_packet AS t2_gtu_in_packet, t1.num_gtu AS t1_num_gtu, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full AS t1_source_file_acquisition_full, t2.source_file_acquisition_full AS t2_source_file_acquisition_full,
    # t1.source_file_trigger AS t1_source_file_trigger, t2.source_file_trigger AS t2_source_file_trigger, t1.run_timestamp AS t1_run_timestamp, t2.run_timestamp AS t2_run_timestamp
    # , etruth_trueenergy /* */
    q = ''' SELECT  /* t1.event_id, t1.source_data_type_num, t1.gtu_in_packet, t1.num_gtu, t2.gtu_in_packet AS t2_gtu_in_packet, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full, t1.source_file_trigger */
    t1.event_id AS event_id, t2.event_id AS t2_event_id, t1.source_data_type_num AS source_data_type_num, t2.source_data_type_num AS t2_source_data_type_num, t1.global_gtu AS global_gtu, t2.global_gtu AS t2_global_gtu,
    t1.gtu_in_packet AS gtu_in_packet, t2.gtu_in_packet AS t2_gtu_in_packet, t1.num_gtu AS num_gtu, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full AS source_file_acquisition_full, t2.source_file_acquisition_full AS t2_source_file_acquisition_full, 
    t1.source_file_trigger AS source_file_trigger, t2.source_file_trigger AS t2_source_file_trigger, t1.run_timestamp AS run_timestamp, t2.run_timestamp AS t2_run_timestamp 
    , etruth_trueenergy /* */
    FROM /**/ simu_event_spb_proc 
    JOIN simu_event USING(simu_event_id) 
    JOIN (
    SELECT * /*event_id, source_data_type_num, source_file_acquisition_full, source_file_trigger, packet_id, gtu_in_packet, num_gtu*/
    FROM spb_processing_event_ver2
    WHERE
    source_data_type_num=3  
    AND {conds}
    ) AS t1 USING(event_id) /**/
    JOIN spb_processing_event_ver2 AS t2 USING(source_file_acquisition_full) 
    WHERE t2.source_data_type_num=5  AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4)) /* AND abs(t1.gtu_in_packet - t2.gtu_in_packet) < 5*/ 
    ORDER BY t1.event_id ASC'''.format(conds=cond_selection_rules)
    # print(q)
    if queries_log:
        queries_log.write(q)
    simu_entries_within_cond_bgf05_and_bgf1_v2 = psql.read_sql(q, con)
    return simu_entries_within_cond_bgf05_and_bgf1_v2


def find_multiple_event_id_rows(simu_entries_within_cond_bgf05_and_bgf1_v2):
    row_idxs = []
    row_event_ids = []
    for i, r in simu_entries_within_cond_bgf05_and_bgf1_v2.iterrows():
        srch = simu_entries_within_cond_bgf05_and_bgf1_v2[ simu_entries_within_cond_bgf05_and_bgf1_v2['event_id'] == r.event_id ]
        if len(srch) > 1:
            row_idxs.append(i)
            row_event_ids.append(r.event_id)
    return row_idxs, list(set(row_event_ids))


def get_count_utah_entries_within_cond(con, cond_selection_rules, queries_log=None):
    q = '''
    SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 2 AND {conds} 
    '''.format(conds=cond_selection_rules)
    if queries_log:
        queries_log.write(q)
    count_utah_entries_within_cond = psql.read_sql(q, con)
    return count_utah_entries_within_cond['count'].iloc[0]


def get_count_flight_entries_within_cond(con, cond_selection_rules, queries_log=None):
    q = '''
    SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 AND x_y_active_pixels_num > 1750 AND {conds} 
    '''.format(conds=cond_selection_rules)
    if queries_log:
        queries_log.write(q)
    count_flight_entries_within_cond = psql.read_sql(q, con)
    return count_flight_entries_within_cond['count'].iloc[0]


def get_count_flight_entries_within_cond_num_ec(con, cond_selection_rules, num_ec=3, queries_log=None):
    q = '''
    SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 AND {conds} 
    '''.format(conds=cond_selection_rules.replace('x_y_active_pixels_num > 1750','x_y_active_pixels_num > {}'.format(256*num_ec)))
    if queries_log:
        queries_log.write(q)
    get_count_flight_entries_within_cond_num_ec = psql.read_sql(q, con)
    return get_count_flight_entries_within_cond_num_ec['count'].iloc[0]


def get_cond_bgf05_and_bgf1_simu_events__packet_count_by_energy(con, cond_selection_rules, queries_log=None):
    cond_bgf05_and_bgf1_simu_events_by_energy_query = '''
    SELECT COUNT(sq.count_event_id) AS count_packets, sq.etruth_trueenergy AS etruth_trueenergy 
    FROM (
    SELECT COUNT(t1.event_id) AS count_event_id, t1.source_file_acquisition_full, etruth_trueenergy 
    FROM simu_event_spb_proc 
    JOIN simu_event USING(simu_event_id) 
    JOIN (
    SELECT event_id, source_file_acquisition_full, packet_id, gtu_in_packet, num_gtu
    FROM spb_processing_event_ver2
    WHERE
    source_data_type_num=3  
    AND {cond_selection_rules}
    ) AS t1 USING(event_id) 
    JOIN spb_processing_event_ver2 AS t2 USING(source_file_acquisition_full) 
    WHERE t2.source_data_type_num=5 AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4)) 
    GROUP BY etruth_trueenergy, t1.source_file_acquisition_full, t1.packet_id) AS sq 
    GROUP BY etruth_trueenergy ORDER BY etruth_trueenergy;
    '''.format(cond_selection_rules=cond_selection_rules)
    if queries_log:
        queries_log.write(cond_bgf05_and_bgf1_simu_events_by_energy_query)
    # print(cond_bgf05_and_bgf1_simu_events_by_energy_query)
    cond_bgf05_and_bgf1_simu_events_by_energy = psql.read_sql(cond_bgf05_and_bgf1_simu_events_by_energy_query, con)
    return cond_bgf05_and_bgf1_simu_events_by_energy


def vis_df_comparison(df1, df2, save_fig_dir=None, fig_file_name='{yaxis}_by_{xaxis}_comparison', xaxis='etruth_trueenergy', yaxis='count_packets', df1_label="All packets bgf=0.5 and bgf=1", df2_label="Selected packets bgf=0.5 and bgf=1", yscale='linear'):
    if len(df1) > 0:
        ax_df1 = df1.plot(x=xaxis, y=yaxis, marker='.', linestyle='-', color='blue', label=df1_label)
        #all_bgf10_simu_events_by_energy.plot(x='etruth_trueenergy',y='count_packets',marker='.',linestyle='-', color='red', ax=ax_all_simu_events_by_energy)
        df2.plot(x=xaxis,y=yaxis,marker='.',linestyle='-', color='green', ax=ax_df1, label=df2_label)

        ax_df1.set_yscale(yscale) #"log", nonposy='clip')

        #plt.show()
        if save_fig_dir is not None:
            save_figure(ax_df1.get_figure(), save_fig_dir, fig_file_name.format(xaxis=xaxis, yaxis=yaxis))
        else:
            plt.show()


def merge_cond_all_dataframes(cond_df, all_df, merge_on='etruth_trueenergy', fraction_column='count_fraction', count_column='count_packets'):
    if not isinstance(merge_on, (list,tuple)):
        merge_on = [merge_on]
    merged_df = pd.merge(cond_df, all_df,
                                        how='outer',
                                        suffixes=['_cond','_all'],
                                        on=merge_on)
    merged_nona_df = merged_df.dropna().copy()
    if count_column:
        merged_col_cond = count_column + '_cond'
        merged_col_all = count_column + '_all'
        merged_nona_df[fraction_column] = merged_nona_df[merged_col_cond] / merged_nona_df[merged_col_all]
    return merged_nona_df


def merge_all_cond_bgf05_and_bgf1_simu_events_by_energy(cond_bgf05_and_bgf1_simu_events_by_energy, all_bgf05_and_bgf1_simu_events__packet_count_by_energy, x_axis_column='etruth_trueenergy'):
    cond_all_merged_bgf05_simu_events_by_energy = merge_cond_all_dataframes(cond_bgf05_and_bgf1_simu_events_by_energy, all_bgf05_and_bgf1_simu_events__packet_count_by_energy, merge_on=x_axis_column)
    return cond_all_merged_bgf05_simu_events_by_energy


def calc_yerrs_for_merged_events_by_energy(cond_all_merged_bgf05_simu_events_by_energy, y_axis_cond_column='count_packets_cond', y_axis_all_column='count_packets_all'):
    n1 = cond_all_merged_bgf05_simu_events_by_energy[y_axis_cond_column]
    n2 = cond_all_merged_bgf05_simu_events_by_energy[y_axis_all_column]

    #frac = n1/n2
    #yerrs = list( np.sqrt( ((1-(frac))/n1) + (1/n2) ) * frac ) # cond_all_merged_bgf05_simu_events_by_energy['count_fraction']
    #yerrs = list( np.sqrt(n1 * (1 - n1/n2))/n2 )
    #return yerrs
    return calc_error_bars(n1, n2)


def vis_count_fraction(cond_all_merged_bgf05_simu_events_by_energy, xerrs=None, yerrs=None, save_fig_dir=None, fig_file_name='count_fraction_by_energy.png', x_axis_column='etruth_trueenergy', y_axis_column='count_fraction'):
    if len(cond_all_merged_bgf05_simu_events_by_energy):
        fig, axs = plt.subplots(2,1)
        fig.set_size_inches(18.5/1.8, 2*10.5/1.8)
        for ax_i, ax in enumerate(axs):
            ax = cond_all_merged_bgf05_simu_events_by_energy.plot(x=x_axis_column, y=y_axis_column, xerr=xerrs, yerr=yerrs, marker='.', linestyle='-', ecolor='green', linewidth=1, label='Fraction of all packets', ax=ax)
            ax.set_ylim([0,1.1])
            if ax_i == 1:
                ax.set_xscale('log')
            ax.grid(True)

        #plt.show()
        if save_fig_dir is not None:
            save_figure(fig,save_fig_dir, fig_file_name)
        else:
            plt.show()


def calc_error_bars(n1, n2):
    frac = n1/n2
    yerrs = list( np.sqrt( ((1-(frac))/n1) + (1/n2) ) * frac )
    return frac, yerrs


def fit_points_using_n1_n2(x,  n1, n2):
    frac, yerrs = calc_error_bars(n1, n2)
    return fit_points_using_yerrs(x, frac, yerrs)


def fit_points_using_yerrs(x, y, yerrs, norders=8):
    w = 1 - np.array(yerrs)

    fits_p =  []
    for i in range(1,norders+1):
        print(">>> POLYFIT OF {} DEGREE".format(i))
        z =  np.polyfit(x, y, i, w=w)
        fits_p.append(np.poly1d(z))

    return fits_p


def save_csv_of_fits(fits_p, *path_parts):
    sep = '\t'
    csv_path = ensure_ext(os.path.join(*path_parts),".{}sv".format('t' if sep=='\t' else 'c'))
    with open(csv_path,'w') as csv_file:
        for i, fit_p in enumerate(fits_p):
            print("{:d}".format(i+1) + sep + sep.join(["{:.4f}".format(n) for n in fit_p]), file=csv_file)


#def fit_points_cond_all_merged_bgf05_simu_events_by_energy(cond_all_merged_bgf05_simu_events_by_energy,
def fit_points_cond_all_merged(cond_all_merged,
                           x_axis_column='etruth_trueenergy', y_axis_cond_column='count_packets_cond', y_axis_all_column='count_packets_all'):

    x = cond_all_merged[x_axis_column]
    #y = # cond_all_merged['count_fraction']

    if len(x) == 0:
        raise RuntimeError('No x axis column values')

    n1 = cond_all_merged[y_axis_cond_column]
    n2 = cond_all_merged[y_axis_all_column]

    frac, yerrs = calc_error_bars(n1, n2)
    fits_p = fit_points_using_yerrs(x, frac, yerrs)

    return x, frac, yerrs, fits_p


def vis_count_fraction_fits(x, y, xerrs=None, yerrs=None, fits_p=[], save_fig_dir=None, fig_file_name='count_fraction_by_energy_fits', xlabel='Energy [MeV]'):

    if len(x) > 0 and len(y) > 0:

        xv = np.linspace(np.min(x),np.max(x),100)

        fig, eaxs = plt.subplots(2,1)
        fig.set_size_inches(18.5/1.8, 2*10.5/1.8)

        colors = ['pink', 'purple', 'red', 'black', 'yellow', 'royalblue', 'cyan', 'blue']
        line_styles = ['-',':']
        labels = ['{:d}st order poly', '{:d}nd order poly', '{:d}rd order poly', '{:d}nd order poly']

        for eax_i, eax in enumerate(eaxs):
            eax.errorbar(x,y, xerr=xerrs, yerr=yerrs,ecolor='g',fmt='.',label="Measurement")

            for j, fit_p in enumerate(fits_p):
                if not fit_p:
                    continue
                eax.plot(xv, fit_p(xv), linestyle=line_styles[(j//len(colors))%len(line_styles)], color=colors[j%len(colors)],
                         label=(labels[j] if j < len(labels) else labels[-1]).format(j+1))

            eax.set_ylim([0.3,1.1])
            eax.grid(True)
            eax.set_ylabel('Efficiency')
            eax.set_xlabel(xlabel)
            if eax_i == 1:
                eax.set_xscale('log')
            eax.legend()

        if save_fig_dir is not None:
            save_figure(fig, save_fig_dir, fig_file_name)
        else:
            plt.show()


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))


def thin_datapoints(x_vals, y_vals, num_steps=50):
    if len(x_vals) == 0:
        raise RuntimeError('Empty x_vals')
    max_x_val = x_vals[-1]
    x_val_step = max_x_val/num_steps
    r_i = 0
    cur_x_val_step = x_vals[0]
    cur_x_val = cur_x_val_step
    binned_x_vals = []
    binned_y_vals = []
    binned_x_errs = []
    binned_x_ranges_low = []
    binned_x_ranges_up = []

    while cur_x_val_step <= max_x_val:
        bin_x_vals = []
        bin_y_vals = []

        while cur_x_val_step <= cur_x_val < cur_x_val_step+x_val_step:
            bin_x_vals.append(cur_x_val)
            bin_y_vals.append(y_vals[r_i])

            if r_i+1 >= len(x_vals):
                break
            r_i += 1
            cur_x_val = x_vals[r_i]

        if bin_y_vals:
            x_bin_val, x_bin_std = weighted_avg_and_std(np.array(bin_x_vals), bin_y_vals)
            binned_y_vals.append(np.sum(bin_y_vals))
            binned_x_vals.append(x_bin_val)
            binned_x_errs.append(x_bin_std)
            binned_x_ranges_low.append(bin_x_vals[0])
            binned_x_ranges_up.append(bin_x_vals[-1])

        cur_x_val_step += x_val_step

    return np.array(binned_x_vals), np.array(binned_y_vals), np.array(binned_x_errs), np.array(binned_x_ranges_low), np.array(binned_x_ranges_up)


def calc_thin_datapoints_avg(e_vals_fc, e_fc_low, e_fc_up, e_vals_fa, e_fa_low, e_fa_up):
    e_avg_vals = (e_vals_fa + e_vals_fc)/2
    e_avg_low = (e_vals_fa - e_fa_low + e_vals_fc - e_fc_low)/2
    e_avg_up = (e_fa_up - e_vals_fa + e_fc_up - e_vals_fc)/2
    return e_avg_vals, (e_avg_low, e_avg_up)


def thin_datapoints_from_dataframe(df, x_axis_column='etruth_trueenergy', y_axis_cond_column='count_packets_cond', y_axis_all_column='count_packets_all', num_steps=20):
    e_vals_fc, y_cond_vals, e_fc_err, e_fc_low, e_fc_up = thin_datapoints(np.array(df[x_axis_column]), np.array(df[y_axis_cond_column]), num_steps)
    e_vals_fa, y_all_vals, e_fa_err, e_fa_low, e_fa_up = thin_datapoints(np.array(df[x_axis_column]), np.array(df[y_axis_all_column]), num_steps)
    # print(np.array(e_vals_fc))
    # print(len(e_vals_fc))
    # print(y_cond_vals)
    # print(len(y_cond_vals))
    # print(np.array(e_fc_err))
    # print("-"*30)
    # print(np.array(e_vals_fa))
    # print(len(e_vals_fa))
    # print(y_cond_all)
    # print(len(y_all_vals))
    # print(np.array(e_fa_err))
    # print("-"*30)
    # print(np.array(e_vals_fc) - np.array(e_vals_fa))
    # In[152]:
    # In[153]:
    # y = y_cond_vals/y_all_vals
    y, yerrs = calc_error_bars(y_cond_vals, y_all_vals)
    e_avg_vals, xerrs  = calc_thin_datapoints_avg(e_vals_fc, e_fc_low, e_fc_up, e_vals_fa, e_fa_low, e_fa_up)  # xerrs = (e_avg_low, e_avg_up)
    return e_avg_vals, y, xerrs, yerrs, (e_vals_fc, y_cond_vals, e_fc_err, e_fc_low, e_fc_up), (e_vals_fa, y_all_vals, e_fa_err, e_fa_low, e_fa_up)


def save_thinned_datapoints(x, y, xerrs, yerrs, cond_thinned, all_thinned, save_csv_dir, file_name, sep='\t'):
    columns = [x, y, xerrs[0], xerrs[1], yerrs, cond_thinned[0], cond_thinned[1], cond_thinned[2], cond_thinned[3], cond_thinned[4], all_thinned[0], all_thinned[1], all_thinned[2], all_thinned[3], all_thinned[4]]

    # TODO pass
    #np.column_stack(x,y,np.array(xerrs)[:0,], )

    csv_path = ensure_ext(os.path.join(save_csv_dir, file_name),".{}sv".format('t' if sep=='\t' else 'c'))
    with open(csv_path,'w') as csv_file:
        print(sep.join(("x", "y", "avg_x_low", "avg_x_up", "y_err", "cond_x", "cond_y", "cond_x_err", "cond_x_low", "cond_x_high", "all_x", "all_y", "all_x_err", "all_x_low", "all_x_high")), file=csv_file)
        for i in range(len(x)):
            print(sep.join([str(col[i]) for col in columns]), file=csv_file)


def vis_thinned_datapoints(cond_all_merged_bgf05_simu_events_by_energy,
                           e_vals_fc, y_cond_vals, e_fc_err, e_fc_low, e_fc_up,
                           e_vals_fa, y_all_vals, e_fa_err, e_fa_low, e_fa_up,
                           save_fig_dir, fig_file_name='comparison_thinned_datapoints.png',
                           x_axis_column='etruth_trueenergy', y_axis_cond_column='count_packets_cond', y_axis_all_column='count_packets_all'):

    if len(cond_all_merged_bgf05_simu_events_by_energy) == 0 or len(e_vals_fa) == 0 or len(e_vals_fc) == 0:
        return

    fig, ax = plt.subplots(1)
    fig.set_size_inches(18.5/1.5, 10.5/1.5)
    ax.plot(cond_all_merged_bgf05_simu_events_by_energy[x_axis_column], cond_all_merged_bgf05_simu_events_by_energy[y_axis_cond_column], color='yellow', #yerr=yerrs, , ecolor='green'
            marker='',linestyle='--', linewidth=0.5, label='Fraction of all packets')
    ax.plot(cond_all_merged_bgf05_simu_events_by_energy[x_axis_column], cond_all_merged_bgf05_simu_events_by_energy[y_axis_all_column], color='cyan', #yerr=yerrs, , ecolor='green'
            marker='',linestyle='--', linewidth=0.5, label='Fraction of all packets')
    ax.errorbar(e_vals_fa, y_all_vals, color='red', xerr=[e_vals_fa - e_fa_low, e_fa_up - e_vals_fa], # e_fc_err, #yerr=yerrs, , ecolor='green'
            marker='.',linestyle='-', linewidth=1, label='Fraction of all packets')
    ax.errorbar(e_vals_fc, y_cond_vals, color='purple', xerr=[e_vals_fc - e_fc_low, e_fc_up - e_vals_fc], # e_fa_err, #yerr=yerrs, , ecolor='green'
            marker='.',linestyle='-', linewidth=1, label='Fraction of all packets')
    ax.errorbar(e_vals_fa, y_all_vals, color='black', alpha=0.3, xerr=e_fa_err, # e_fa_err, #yerr=yerrs, , ecolor='green'
            marker='.',linestyle='', linewidth=1, label='Fraction of all packets')
    ax.errorbar(e_vals_fc, y_cond_vals, color='black', alpha=0.3, xerr=e_fc_err, # e_fa_err, #yerr=yerrs, , ecolor='green'
            marker='.',linestyle='', linewidth=1, label='Fraction of all packets')

    e_avg_vals, xerrs = calc_thin_datapoints_avg(e_vals_fc, e_fc_low, e_fc_up, e_vals_fa, e_fa_low, e_fa_up) # xerrs = (e_avg_low, e_avg_up)

    ax.errorbar(e_avg_vals, (y_all_vals+y_cond_vals)/2, color='green', alpha=.2, xerr=xerrs, # e_fa_err, #yerr=yerrs, , ecolor='green', xerr=[e_avg_low, e_avg_up]
            marker='.',linestyle=':', linewidth=1, label='Fraction of all packets')

    # ax.set_yscale('log')
    # ax.set_xscale('log')
    #ax.set_ylim([0,1.1])
    ax.grid(True)

    if save_fig_dir is not None:
        save_figure(fig, save_fig_dir, fig_file_name)
    else:
        plt.show()


def get_all_bgf05_and_bgf1_simu_events__packet_count_by_posz(con, queries_log=None):
    all_bgf05_and_bgf1_simu_events__packet_count_by_posz_query = '''
    SELECT COUNT(sq.count_event_id) AS count_packets, /*sq.etruth_trueenergy AS etruth_trueenergy,*/ sq.egeometry_pos_z AS egeometry_pos_z 
    FROM (
    SELECT COUNT(t1.event_id) AS count_event_id, t1.source_file_acquisition_full, /*etruth_trueenergy,*/ egeometry_pos_z 
    FROM simu_event_spb_proc 
    JOIN simu_event USING(simu_event_id) 
    JOIN spb_processing_event_ver2 AS t1 USING(event_id) 
    JOIN spb_processing_event_ver2 AS t2 USING(source_file_acquisition_full) 
    WHERE t1.source_data_type_num=3 AND t2.source_data_type_num=5  AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4))
    GROUP BY egeometry_pos_z, /*etruth_trueenergy,*/ t1.source_file_acquisition_full, t1.packet_id
    ) as sq GROUP BY egeometry_pos_z /*, etruth_trueenergy*/ ORDER BY egeometry_pos_z/*, etruth_trueenergy*/;
    '''
    if queries_log:
        queries_log.write(all_bgf05_and_bgf1_simu_events__packet_count_by_posz_query)
    all_bgf05_and_bgf1_simu_events__packet_count_by_posz = psql.read_sql( all_bgf05_and_bgf1_simu_events__packet_count_by_posz_query, con)
    #print("len(all_bgf05_and_bgf1_simu_events__packet_count_by_posz)",len(all_bgf05_and_bgf1_simu_events__packet_count_by_posz))
    return all_bgf05_and_bgf1_simu_events__packet_count_by_posz


def get_cond_bgf05_and_bgf1_simu_events__packet_count_by_posz(con, cond_selection_rules, queries_log=None):
    # IMPORTANT
    cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_query = '''
    SELECT COUNT(sq.count_event_id) AS count_packets, /*sq.etruth_trueenergy AS etruth_trueenergy,*/ sq.egeometry_pos_z AS egeometry_pos_z 
    FROM (
    SELECT COUNT(t1.event_id) AS count_event_id, t1.source_file_acquisition_full, /*etruth_trueenergy,*/ egeometry_pos_z 
    FROM simu_event_spb_proc 
    JOIN simu_event USING(simu_event_id) 
    JOIN (
    SELECT event_id, source_file_acquisition_full, packet_id, gtu_in_packet, num_gtu
    FROM spb_processing_event_ver2
    WHERE
    source_data_type_num=3  
    AND {cond_selection_rules}
    ) AS t1 USING(event_id) 
    JOIN spb_processing_event_ver2 AS t2 USING(source_file_acquisition_full) 
    WHERE t2.source_data_type_num=5 AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4)) 
    GROUP BY egeometry_pos_z, /*etruth_trueenergy,*/ t1.source_file_acquisition_full, t1.packet_id) AS sq 
    GROUP BY egeometry_pos_z/*, etruth_trueenergy*/ ORDER BY egeometry_pos_z /*, etruth_trueenergy*/;
    '''.format(cond_selection_rules=cond_selection_rules)
    #print(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_query)
    if queries_log:
        queries_log.write(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_query)
    cond_bgf05_and_bgf1_simu_events__packet_count_by_posz = psql.read_sql(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_query, con)
    #print("len(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz)", len(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz))
    return cond_bgf05_and_bgf1_simu_events__packet_count_by_posz


def get_cond_all_bgf05_and_bgf1_simu_events__packet_count_by_posz_merged(cond_bgf05_and_bgf1_simu_events_by_energy, all_bgf05_and_bgf1_simu_events__packet_count_by_energy):
    cond_all_bgf05_and_bgf1_simu_events__packet_count_by_posz_merged = merge_cond_all_dataframes(cond_bgf05_and_bgf1_simu_events_by_energy, all_bgf05_and_bgf1_simu_events__packet_count_by_energy, merge_on='egeometry_pos_z')
    return cond_all_bgf05_and_bgf1_simu_events__packet_count_by_posz_merged


def get_all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy(con, queries_log=None):
    # IMPORTANT
    all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy_query = '''
    SELECT COUNT(sq.count_event_id) AS count_packets, sq.etruth_trueenergy AS etruth_trueenergy, sq.egeometry_pos_z AS egeometry_pos_z 
    FROM (
    SELECT COUNT(t1.event_id) AS count_event_id, t1.source_file_acquisition_full, etruth_trueenergy, egeometry_pos_z 
    FROM simu_event_spb_proc 
    JOIN simu_event USING(simu_event_id) 
    JOIN spb_processing_event_ver2 AS t1 USING(event_id) 
    JOIN spb_processing_event_ver2 AS t2 USING(source_file_acquisition_full) 
    WHERE t1.source_data_type_num=3 AND t2.source_data_type_num=5  AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4))
    GROUP BY egeometry_pos_z, etruth_trueenergy, t1.source_file_acquisition_full, t1.packet_id
    ) as sq GROUP BY egeometry_pos_z, etruth_trueenergy ORDER BY egeometry_pos_z, etruth_trueenergy;
    '''
    if queries_log:
        queries_log.write(all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy_query)
    all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy = psql.read_sql(all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy_query, con)
    #print("len(all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy)", len(all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy))
    return all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy


def get_cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy(con, cond_selection_rules, queries_log=None):
    cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy_query = '''
    SELECT COUNT(sq.count_event_id) AS count_packets, sq.etruth_trueenergy AS etruth_trueenergy, sq.egeometry_pos_z AS egeometry_pos_z 
    FROM (
    SELECT COUNT(t1.event_id) AS count_event_id, t1.source_file_acquisition_full, etruth_trueenergy, egeometry_pos_z 
    FROM simu_event_spb_proc 
    JOIN simu_event USING(simu_event_id) 
    JOIN (
    SELECT event_id, source_file_acquisition_full, packet_id, gtu_in_packet, num_gtu
    FROM spb_processing_event_ver2
    WHERE
    source_data_type_num=3  
    AND {cond_selection_rules}
    ) AS t1 USING(event_id) 
    JOIN spb_processing_event_ver2 AS t2 USING(source_file_acquisition_full) 
    WHERE t2.source_data_type_num=5 AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4)) 
    GROUP BY egeometry_pos_z, etruth_trueenergy, t1.source_file_acquisition_full, t1.packet_id) AS sq 
    GROUP BY egeometry_pos_z, etruth_trueenergy ORDER BY egeometry_pos_z, etruth_trueenergy;
    '''.format(cond_selection_rules=cond_selection_rules)
    if queries_log:
        queries_log.write(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy_query)
    # print(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy_query)
    cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy = psql.read_sql(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy_query, con)
    #print("len(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy)", len(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy))
    return cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy


def get_cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit(cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona):
    uniq_posz = cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona['egeometry_pos_z'].unique()

    uniq_posz_plot_data = [None] * len(uniq_posz)
    #y_posz_vals = [None] * len(uniq_posz)
    #yerrs_posz_vals = [None] * len(uniq_posz)
    #e_avg_vals_posz_vals = [None] * len(uniq_posz)
    #e_avg_low_posz_vals = [None] * len(uniq_posz)
    #e_avg_up_posz_vals = [None] * len(uniq_posz)
    #fits_p_posz_vals = [None] * len(uniq_posz)

    for posz_i, posz_val in enumerate(uniq_posz):

        print(">> THINNING (EGeometry.Pos.Z={})".format(posz_val))

        single_posz_data = cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona[ cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona['egeometry_pos_z']==posz_val ]

        e_avg_vals, y, xerrs, yerrs, cond_thinned, all_thinned = thin_datapoints_from_dataframe(single_posz_data, x_axis_column='etruth_trueenergy', num_steps=100 if len(single_posz_data) < 10 else 10)  # e_avg_low, e_avg_up

        print(">> FITTING (EGeometry.Pos.Z={})".format(posz_val))

        fits_p = fit_points_using_yerrs(e_avg_vals, y, yerrs)

        uniq_posz_plot_data[posz_i] = ('EGeometry.Pos.Z={}'.format(posz_val), y, yerrs, e_avg_vals, xerrs, fits_p) # xerrs = e_avg_low, e_avg_up

        #y_posz_vals[posz_i] = y
        #yerrs_posz_vals[posz_i] = yerrs
        #e_avg_vals_posz_vals[posz_i] = e_avg_vals
        #e_avg_low_posz_vals[posz_i] = e_avg_low
        #e_avg_up_posz_vals[posz_i] = e_avg_up
        #fits_p_posz_vals[posz_i] = fits_p
        #vis_count_fraction_fits(x, frac, yerrs, fits_p, save_fig_dir, fig_file_name='cond_all_bgf05_and_bgf1_simu_events_count_fraction_by_posz_merged.png')

    return uniq_posz_plot_data #y_posz_vals, yerrs_posz_vals, e_avg_vals_posz_vals, e_avg_low_posz_vals, e_avg_up_posz_vals, fits_p_posz_vals


def vis_cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit(
        uniq_posz_plot_data,
        #y_posz_vals, yerrs_posz_vals, e_avg_vals_posz_vals, e_avg_low_posz_vals, e_avg_up_posz_vals, fits_p_posz_vals,
        save_fig_dir, fig_file_name='cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit.png', num_cols=3, col_width=18.5/1.8, row_height=10.5/1.6, xlabel='Energy [MeV]'):

    if len(uniq_posz_plot_data) == 0:
        return

    fig, axs = plt.subplots(int(np.ceil(len(uniq_posz_plot_data)/num_cols)), num_cols)
    axs_flattened = axs.flatten()

    fig.set_size_inches(num_cols * col_width, np.ceil(len(uniq_posz_plot_data) / num_cols) * row_height)

    colors = ['pink', 'purple', 'red', 'black', 'yellow', 'royalblue', 'cyan', 'blue']
    line_styles = ['-',':']
    labels = ['{:d}st order poly', '{:d}nd order poly', '{:d}rd order poly', '{:d}nd order poly']


    for ax_i, (label, y, yerrs, x, xerrs, fits_p) in enumerate(uniq_posz_plot_data): # xerrs = e_avg_low, e_avg_up
        eax = axs_flattened[ax_i]

        eax.errorbar(x, y, yerr=yerrs, xerr=xerrs, ecolor='g', fmt='.', label="Measurement") # xerrs = [e_avg_low, e_avg_up]
        eax.set_title("{}".format(label))

        xv = np.linspace(np.min(x),np.max(x),100)

        for j, fit_p in enumerate(fits_p):
            if not fit_p:
                continue
            eax.plot(xv, fit_p(xv), linestyle=line_styles[(j//len(colors))%len(line_styles)], color=colors[j%len(colors)],
                     label=(labels[j] if j < len(labels) else labels[-1]).format(j+1) )

        eax.set_ylim([0.3,1.1])
        eax.grid(True)
        eax.set_ylabel('Efficiency')
        eax.set_xlabel(xlabel)
        eax.legend()

    if save_fig_dir is not None:
        save_figure(fig, save_fig_dir, fig_file_name)
    else:
        plt.show()


def select_simu_events_within_cond(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log=None):
    cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)

    select_simu_event_within_cond_query_format = '''
    SELECT t1.event_id, t1.source_file_acquisition_full, t1.source_file_trigger_full, t1.packet_id, t1.num_gtu, t1.gtu_in_packet, t1.num_triggered_pixels, 
    t1.gtu_y_hough__peak_thr2_avg_phi, t1.gtu_x_hough__peak_thr2_avg_phi, t1.gtu_y_hough__peak_thr3_avg_phi, t1.gtu_x_hough__peak_thr3_avg_phi, 
    t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width,
    t1.gtu_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.gtu_x_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.x_y_hough__peak_thr1__max_cluster_counts_sum_width,
    t1.trigg_x_y_hough__dbscan_num_clusters_above_thr1, t1.trigg_gtu_y_hough__dbscan_num_clusters_above_thr1, t1.trigg_gtu_x_hough__dbscan_num_clusters_above_thr1, 
    t1.gtu_y_hough__dbscan_num_clusters_above_thr1, t1.gtu_x_hough__dbscan_num_clusters_above_thr1,
    etruth_trueenergy, egeometry_pos_z
    FROM spb_processing_event_ver2 AS t1 
    JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    JOIN simu_event_spb_proc ON (t1.event_id = simu_event_spb_proc.event_id)
    JOIN simu_event USING(simu_event_id) 
    WHERE t1.source_data_type_num = 3 AND t2.source_data_type_num = 5   
    AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4))
    AND ''' + cond_selection_rules_t1_prefixed + \
    '''
    ORDER BY
    t1.x_y_hough__peak_thr1__max_cluster_counts_sum_width DESC,
    t1.num_triggered_pixels ASC,
    (t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width + t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width + t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width) DESC
    
    OFFSET {offset} LIMIT {limit}
    '''

    # cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)
    # q = '''
    # SELECT t1.event_id, t1.source_data_type_num, t1.gtu_in_packet, t1.num_gtu, t2.gtu_in_packet AS t2_gtu_in_packet, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full, t1.source_file_trigger FROM spb_processing_event_ver2 AS t1 
    # JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    # WHERE t1.source_data_type_num = 3 AND t2.source_data_type_num = 5  AND (/*(t1.gtu_in_packet < t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu) OR*/ (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu))  /* abs() BETWEEN 3 AND 4 */ {conds} 
    # ORDER BY t1.event_id ASC
    # '''.format(conds=cond_selection_rules_t1_prefixed)


    # count_utah_entries_within_cond = psql.read_sql('''
    # SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 2 {conds} 
    # '''.format(conds=cond_selection_rules), con)
    # count_utah_entries_within_cond

    # count_flight_entries_within_cond = psql.read_sql('''
    # SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 {conds} 
    # '''.format(conds=cond_selection_rules), con)
    # count_flight_entries_within_cond

    # count_flight_entries_within_cond_3ec = psql.read_sql('''
    # SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 {conds} 
    # '''.format(conds=cond_selection_rules.replace('x_y_active_pixels_num > 1750','x_y_active_pixels_num > {}'.format(256*3))), con)
    # count_flight_entries_within_cond_3ec

    select_simu_event_within_cond_query_format = select_simu_event_within_cond_query_format.format(offset=0, limit=200000)

    if queries_log:
        queries_log.write(select_simu_event_within_cond_query_format)

    return psql.read_sql(select_simu_event_within_cond_query_format, con)
    #con.rollback()
    #all_rows_simu_event_within_cond, all_columns_simu_event_within_cond = data_analysis_utils.select_events(con, select_simu_event_within_cond_query_format, [], limit=200000, column_prefix='t1.')
    #print("Selected {} rows".format(len(all_rows)))
    #return all_rows_simu_event_within_cond, all_columns_simu_event_within_cond


def select_simu_event_not_within_cond(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log=None):
    cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns), r't1.\g<0>', cond_selection_rules)

    select_simu_event_not_within_cond_query_format = '''
    SELECT 
    t1.event_id, t1.source_file_acquisition_full, t1.source_file_trigger_full, t1.packet_id, t1.num_gtu, t1.gtu_in_packet, t1.num_triggered_pixels, 
    t1.gtu_y_hough__peak_thr2_avg_phi, t1.gtu_x_hough__peak_thr2_avg_phi, t1.gtu_y_hough__peak_thr3_avg_phi, t1.gtu_x_hough__peak_thr3_avg_phi, 
    t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width,
    t1.gtu_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.gtu_x_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.x_y_hough__peak_thr1__max_cluster_counts_sum_width,
    t1.trigg_x_y_hough__dbscan_num_clusters_above_thr1, t1.trigg_gtu_y_hough__dbscan_num_clusters_above_thr1, t1.trigg_gtu_x_hough__dbscan_num_clusters_above_thr1, 
    t1.gtu_y_hough__dbscan_num_clusters_above_thr1, t1.gtu_x_hough__dbscan_num_clusters_above_thr1,
    etruth_trueenergy, egeometry_pos_z

    FROM spb_processing_event_ver2 AS t1 
    JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    JOIN simu_event_spb_proc ON (t1.event_id = simu_event_spb_proc.event_id)
    JOIN simu_event USING(simu_event_id) 
    WHERE t1.source_data_type_num = 3 AND t2.source_data_type_num = 5   
    AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4))
    AND NOT (''' + cond_selection_rules_t1_prefixed + ')' + \
    '''
    ORDER BY
    t1.num_triggered_pixels DESC,
    (t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width + t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width + t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width) ASC
    
    OFFSET {offset} LIMIT {limit}
    '''

    # cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)
    # q = '''
    # SELECT t1.event_id, t1.source_data_type_num, t1.gtu_in_packet, t1.num_gtu, t2.gtu_in_packet AS t2_gtu_in_packet, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full, t1.source_file_trigger FROM spb_processing_event_ver2 AS t1 
    # JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    # WHERE t1.source_data_type_num = 3 AND t2.source_data_type_num = 5  AND (/*(t1.gtu_in_packet < t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu) OR*/ (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu))  /* abs() BETWEEN 3 AND 4 */ {conds} 
    # ORDER BY t1.event_id ASC
    # '''.format(conds=cond_selection_rules_t1_prefixed)


    # count_utah_entries_within_cond = psql.read_sql('''
    # SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 2 {conds} 
    # '''.format(conds=cond_selection_rules), con)
    # count_utah_entries_within_cond

    # count_flight_entries_within_cond = psql.read_sql('''
    # SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 {conds} 
    # '''.format(conds=cond_selection_rules), con)
    # count_flight_entries_within_cond

    # count_flight_entries_within_cond_3ec = psql.read_sql('''
    # SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 {conds} 
    # '''.format(conds=cond_selection_rules.replace('x_y_active_pixels_num > 1750','x_y_active_pixels_num > {}'.format(256*3))), con)
    # count_flight_entries_within_cond_3ec

    select_simu_event_not_within_cond_query_format = select_simu_event_not_within_cond_query_format.format(offset=0, limit=200000)

    if queries_log:
        queries_log.write(select_simu_event_not_within_cond_query_format)

    return psql.read_sql(select_simu_event_not_within_cond_query_format, con)
    #con.rollback()
    #all_rows_simu_event_not_within_cond, all_columns_simu_event_not_within_cond = data_analysis_utils.select_events(con, select_simu_event_not_within_cond_query_format, [], limit=2000, column_prefix='t1.')
    ##print("Selected {} rows".format(len(all_rows)))
    #return all_rows_simu_event_not_within_cond, all_columns_simu_event_not_within_cond


# all_rows_simu_event_within_cond,all_columns_simu_event_within_cond,
def group_rows_to_count_packets(df, groupby1_columns=['etruth_trueenergy','source_file_acquisition_full','packet_id'], groupby2_columns=['etruth_trueenergy']):
    #df = pd.DataFrame(all_rows_simu_event_within_cond, columns=all_columns_simu_event_within_cond)
    #  GROUP BY egeometry_pos_z, etruth_trueenergy, t1.source_file_acquisition_full, t1.packet_id) AS sq 
    #  GROUP BY egeometry_pos_z, etruth_trueenergy ORDER BY egeometry_pos_z, etruth_trueenergy;
    count_packets_df = df.groupby(groupby1_columns).count().groupby(groupby2_columns).count().loc[:,['event_id']].reindex(columns=['event_id']).reset_index().rename(columns={'event_id':'count_packets'}) #['event_id'] #.reindex(columns=['etruth_trueenergy','event_id'])
    return count_packets_df


def select_flight_events_within_cond(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log=None):
    cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)

    select_flight_events_within_cond_query_format = '''
    SELECT 
    t1.event_id, t1.source_file_acquisition_full, t1.source_file_trigger_full, t1.packet_id, t1.num_gtu, t1.gtu_in_packet, t1.num_triggered_pixels, 
    t1.gtu_y_hough__peak_thr2_avg_phi, t1.gtu_x_hough__peak_thr2_avg_phi, t1.gtu_y_hough__peak_thr3_avg_phi, t1.gtu_x_hough__peak_thr3_avg_phi, 
    t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width,
    t1.gtu_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.gtu_x_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.x_y_hough__peak_thr1__max_cluster_counts_sum_width,
    t1.trigg_x_y_hough__dbscan_num_clusters_above_thr1, t1.trigg_gtu_y_hough__dbscan_num_clusters_above_thr1, t1.trigg_gtu_x_hough__dbscan_num_clusters_above_thr1, 
    t1.gtu_y_hough__dbscan_num_clusters_above_thr1, t1.gtu_x_hough__dbscan_num_clusters_above_thr1, 
    t1.x_y_neighbourhood_counts_norm_sum, t1.x_y_neighbourhood_area, t1.x_y_neighbourhood_size, t1.x_y_neighbourhood_width, t1.x_y_neighbourhood_height, t1.x_y_neighbourhood_area,
    t1.gtu_y_neighbourhood_counts_norm_sum, t1.gtu_y_neighbourhood_area, t1.gtu_y_neighbourhood_size, t1.gtu_y_neighbourhood_width, t1.gtu_y_neighbourhood_height, t1.gtu_y_neighbourhood_area,
    t1.gtu_x_neighbourhood_counts_norm_sum, t1.gtu_x_neighbourhood_area, t1.gtu_x_neighbourhood_size, t1.gtu_x_neighbourhood_width, t1.gtu_x_neighbourhood_height, t1.gtu_x_neighbourhood_area

    FROM spb_processing_event_ver2 AS t1 
    WHERE t1.source_data_type_num = 1   
    AND x_y_active_pixels_num > 1750 AND ''' + cond_selection_rules_t1_prefixed + '' + \
    '''
    ORDER BY
    /*t1.x_y_hough__peak_thr1__max_cluster_counts_sum_width DESC,*/
    t1.num_triggered_pixels DESC,
    (t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width + t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width + t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width) ASC
    
    OFFSET {offset} LIMIT {limit}
    '''

    # cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)
    # q = '''
    # SELECT t1.event_id, t1.source_data_type_num, t1.gtu_in_packet, t1.num_gtu, t2.gtu_in_packet AS t2_gtu_in_packet, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full, t1.source_file_trigger FROM spb_processing_event_ver2 AS t1 
    # JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    # WHERE t1.source_data_type_num = 3 AND t2.source_data_type_num = 5  AND (/*(t1.gtu_in_packet < t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu) OR*/ (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu))  /* abs() BETWEEN 3 AND 4 */ {conds} 
    # ORDER BY t1.event_id ASC
    # '''.format(conds=cond_selection_rules_t1_prefixed)


    # count_utah_entries_within_cond = psql.read_sql('''
    # SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 2 {conds} 
    # '''.format(conds=cond_selection_rules), con)
    # count_utah_entries_within_cond

    # count_flight_entries_within_cond = psql.read_sql('''
    # SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 {conds} 
    # '''.format(conds=cond_selection_rules), con)
    # count_flight_entries_within_cond

    # count_flight_entries_within_cond_3ec = psql.read_sql('''
    # SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 {conds} 
    # '''.format(conds=cond_selection_rules.replace('x_y_active_pixels_num > 1750','x_y_active_pixels_num > {}'.format(256*3))), con)
    # count_flight_entries_within_cond_3ec

    select_flight_events_within_cond_query = select_flight_events_within_cond_query_format.format(offset=0, limit=100000)

    if queries_log:
        queries_log.write(select_flight_events_within_cond_query)

    flight_events_within_cond = psql.read_sql(select_flight_events_within_cond_query, con)
    # all_rows_flight_event_within_cond, all_columns_flight_event_within_cond = data_analysis_utils.select_events(con, select_flight_event_within_cond_query_format, [], limit=2000, column_prefix='t1.')

    # con.rollback()
    # all_rows_flight_event_within_cond, all_columns_flight_event_within_cond = data_analysis_utils.select_events(con, select_flight_event_within_cond_query_format, [], limit=2000, column_prefix='t1.')
    # #print("Selected {} rows".format(len(all_rows)))
    return flight_events_within_cond


def select_utah_events_within_cond(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log=None):
    cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)

    select_utah_events_within_cond_query_format = '''
    SELECT 
    t1.event_id, t1.source_file_acquisition_full, t1.source_file_trigger_full, t1.packet_id, t1.num_gtu, t1.gtu_in_packet, t1.num_triggered_pixels, 
    t1.gtu_y_hough__peak_thr2_avg_phi, t1.gtu_x_hough__peak_thr2_avg_phi, t1.gtu_y_hough__peak_thr3_avg_phi, t1.gtu_x_hough__peak_thr3_avg_phi, 
    t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width,
    t1.gtu_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.gtu_x_hough__peak_thr1__max_cluster_counts_sum_width, t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width, t1.x_y_hough__peak_thr1__max_cluster_counts_sum_width,
    t1.trigg_x_y_hough__dbscan_num_clusters_above_thr1, t1.trigg_gtu_y_hough__dbscan_num_clusters_above_thr1, t1.trigg_gtu_x_hough__dbscan_num_clusters_above_thr1, 
    t1.gtu_y_hough__dbscan_num_clusters_above_thr1, t1.gtu_x_hough__dbscan_num_clusters_above_thr1, 
    t1.x_y_neighbourhood_counts_norm_sum, t1.x_y_neighbourhood_area, t1.x_y_neighbourhood_size, t1.x_y_neighbourhood_width, t1.x_y_neighbourhood_height, t1.x_y_neighbourhood_area,
    t1.gtu_y_neighbourhood_counts_norm_sum, t1.gtu_y_neighbourhood_area, t1.gtu_y_neighbourhood_size, t1.gtu_y_neighbourhood_width, t1.gtu_y_neighbourhood_height, t1.gtu_y_neighbourhood_area,
    t1.gtu_x_neighbourhood_counts_norm_sum, t1.gtu_x_neighbourhood_area, t1.gtu_x_neighbourhood_size, t1.gtu_x_neighbourhood_width, t1.gtu_x_neighbourhood_height, t1.gtu_x_neighbourhood_area

    FROM spb_processing_event_ver2 AS t1 
    WHERE t1.source_data_type_num = 2   
    AND x_y_active_pixels_num > 1750 AND ''' + cond_selection_rules_t1_prefixed + '' + \
    '''
    ORDER BY
    /*t1.x_y_hough__peak_thr1__max_cluster_counts_sum_width DESC,*/
    t1.num_triggered_pixels DESC,
    (t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width + t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width + t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width) ASC
    
    OFFSET {offset} LIMIT {limit}
    '''

    select_utah_events_within_cond_query = select_utah_events_within_cond_query_format.format(offset=0, limit=100000)

    if queries_log:
        queries_log.write(select_utah_events_within_cond_query)

    utah_events_within_cond = psql.read_sql(select_utah_events_within_cond_query, con)

    return utah_events_within_cond



def rows_generator(iterrows):
    for t in iterrows:
        yield t[1]


def count_num_max_pix_on_pmt_and_ec(df, fractions=[0.6, 0.8, 0.9], save_npy_dir=None, npy_file_key=None, debug_messages=False):

    flight_events_num_max_pix_on_pmt = {}
    flight_events_num_max_pix_on_ec = {}

    pickled_flight_events_num_max_pix_on_pmt = {}
    pickled_flight_events_num_max_pix_on_ec = {}

    hashstr = '' #hashlib.md5(df.values.tobytes()).hexdigest()

    # TODO hash entries, then hash all

    # with open("/tmp/debug_{}_{}".format(npy_file_key,hashstr),'w') as f:
    #     for v in df.values:
    #         print(v, file=f)
    # with open("/tmp/debug_{}_{}_bytes".format(npy_file_key,hashstr),'w') as f:
    #     f.write(df.values.tobytes().decode())


    def get_npy_pathname(basename, frac, i, j):
        # global hashstr npy_file_key
        # .npy extension will be appended to the file name if it does not already have one
        pkl_pathname = os.path.join(save_npy_dir,
                                    '{npy_file_key}_{basename}_{hashstr}_{frac:.1f}_{i:d}_{j:d}.npy'.format(
                                        npy_file_key=npy_file_key, hashstr=hashstr, basename=basename, frac=frac,
                                        i=i, j=j))
        return pkl_pathname

    def try_load_npy_file(dest, basename, frac, i, j):
        # global hashstr save_npy_dir npy_file_key
        if save_npy_dir and npy_file_key:
            pkl_pathname = get_npy_pathname(basename, frac, i, j)
            if os.path.exists(pkl_pathname):
                if frac not in dest:
                    dest[frac] = {}
                dest[frac][(i, j)] = np.load(pkl_pathname)
                return True
            elif debug_messages:
                print("#### {} DOES NOT EXIST".format(pkl_pathname))
        return False

    if save_npy_dir:
        os.makedirs(save_npy_dir, exist_ok=True)

    for frac in fractions:
        flight_events_num_max_pix_on_pmt[frac] = {}
        for i in range(6):
            for j in range(6):
                load_successful = try_load_npy_file(pickled_flight_events_num_max_pix_on_pmt, 'num_max_pix_on_pmt', frac, i, j)
                if not load_successful:
                    flight_events_num_max_pix_on_pmt[frac][(i,j)] = np.zeros((len(df), 2))

    for frac in fractions:
        flight_events_num_max_pix_on_ec[frac] = {}
        for i in range(3):
            for j in range(3):
                load_successful = try_load_npy_file(pickled_flight_events_num_max_pix_on_ec, 'num_max_pix_on_ec', frac, i, j)
                if not load_successful:
                    flight_events_num_max_pix_on_ec[frac][(i,j)] = np.zeros((len(df), 2))

    load_files = False
    for frac in fractions:
        if frac not in pickled_flight_events_num_max_pix_on_ec:
            load_files = True
            print(">>> pickled_flight_events_num_max_pix_on_ec[{:.1f}] IS NOT LOADED - READING ALL EVENTS".format(frac))
            break
        if frac not in pickled_flight_events_num_max_pix_on_pmt:
            load_files = True
            print(">>> pickled_flight_events_num_max_pix_on_pmt[{:.1f}] IS NOT LOADED - READING ALL EVENTS".format(frac))
            break
        else:
            for i in range(6):
                for j in range(6):
                    if (i,j) not in pickled_flight_events_num_max_pix_on_pmt[frac]:
                        print(">>> pickled_flight_events_num_max_pix_on_pmt[{:.1f}][({:d},{:d})] IS NOT LOADED - READING ALL EVENTS".format(frac, i, j))
                        load_files = True
                        break
            for i in range(3):
                for j in range(3):
                    if (i,j) not in pickled_flight_events_num_max_pix_on_ec[frac]:
                        print(">>> pickled_flight_events_num_max_pix_on_ec[{:.1f}][({:d},{:d})] IS NOT LOADED - READING ALL EVENTS".format(frac, i, j))
                        load_files = True
                        break

    if load_files:
        for row_i, row in df.iterrows():
            if row_i % 1000 == 0:
                sys.stdout.write("{}\n".format(row_i))
                sys.stdout.flush()
            if row['source_file_acquisition_full'].endswith('.npy'):
                acquisition_arr = np.load(row['source_file_acquisition_full'])
                if acquisition_arr.shape[0] != 256:
                    raise RuntimeError('Unexpected number of frames in the acqusition file "{}" (#{}  ID {})'.format(
                        row['source_file_acquisition_full'], i, row['event_id']))
                frames_acquisition = acquisition_arr[
                                    row['packet_id'] * 128 + row['gtu_in_packet'] - 4:row['packet_id'] * 128 + row['gtu_in_packet'] - 4 + row['num_gtu']]
            elif row['source_file_acquisition_full'].endswith('.root'):
                frames_acquisition = tool.acqconv.get_frames(row['source_file_acquisition_full'],
                                                            row['packet_id'] * 128 + row['gtu_in_packet'] - 4,
                                                            row['packet_id'] * 128 + row['gtu_in_packet'] - 4 + row['num_gtu'], entry_is_gtu_optimization=True)
            else:
                raise RuntimeError('Unexpected source_file_acquisition_full "{}"'.format(row['source_file_acquisition_full']))

            ev_integrated = np.maximum.reduce(frames_acquisition)
            ev_integrated_max = np.max(ev_integrated)

            # print(np.transpose(np.where(ev_integrated > ev_integrated_max*0.9)))

        #     if row['event_id'] == 256024:
        #         print(row)
        #         print("-----------------------")
        #         for pos_y, pos_x in max_positions:
        #             pmt_y = pos_y // 8
        #             pmt_x = pos_x // 8
        #             ec_y = pos_y // 16
        #             ec_x = pos_x // 16
        #             print("y={} x={}    pmt_y={} pmt_x={}   ec_y={} ec_x={}".format(pos_y,pos_x, pmt_y,pmt_x, ec_y, ec_x))

            max_positions = {}

        #     for i in range(6):
        #         for j in range(6):
        #             max_positions[(i,j)] = {}
            for frac in fractions:
                max_positions = np.transpose(np.where(ev_integrated > ev_integrated_max*frac))
                # max_pos[i][index]
                # max_pos[index][i]
                for pos_y, pos_x in max_positions:
                    pmt_y = pos_y // 8
                    pmt_x = pos_x // 8

                    ec_y = pos_y // 16
                    ec_x = pos_x // 16

                    flight_events_num_max_pix_on_pmt[frac][(pmt_y,pmt_x)][row_i,0] += 1
                    flight_events_num_max_pix_on_ec[frac][(ec_y,ec_x)][row_i,0] += 1

        #             if row['event_id'] == 256024:
        #                 print("y={} x={}    pmt_y={} pmt_x={}   ec_y={} ec_x={}   flight_events_num_max_pix_on_ec[({ec_y},{ec_x})][{frac}][{row_i},0]={v}".format(
        #                     pos_y,pos_x, pmt_y,pmt_x, ec_y, ec_x, ec_y=ec_y, ec_x=ec_x, frac=frac, row_i=row_i, v=flight_events_num_max_pix_on_ec[(ec_y,ec_x)][frac][row_i,0]))

        #     if row_i > 50:
        #         break
            #visualized_projections = []
            #if vis_xy:
                #ev_integrated = np.maximum.reduce(frames_acquisition)
                #visualized_projections.append((ev_integrated, "x [pixel]", "y [pixel]"))
            #if vis_gtux:
                #max_integrated_gtu_y = []
                #for frame in frames_acquisition:
                    #max_integrated_gtu_y.append(np.max(frame, axis=1).reshape(-1, 1))  # max in the x axis
                #max_integrated_gtu_y = np.hstack(max_integrated_gtu_y)
                #visualized_projections.append((max_integrated_gtu_y, "GTU", "y [pixel]"))
            #if vis_gtuy:
                #max_integrated_gtu_x = []
                #for frame in frames_acquisition:
                    #max_integrated_gtu_x.append(np.max(frame, axis=0).reshape(-1, 1))  # max the y axis
                #max_integrated_gtu_x = np.hstack(max_integrated_gtu_x)
                #visualized_projections.append((max_integrated_gtu_x, "GTU", "x [pixel]"))

            #plt.show()


        # In[210]:


        for k in range(len(df)):
            for frac in fractions:
                for i in range(3):
                    for j in range(3):
                        for ii in range(3):
                            for jj in range(3):
                                if jj == j and ii == i:
                                    continue
                                flight_events_num_max_pix_on_ec[frac][(i,j)][k,1] += flight_events_num_max_pix_on_ec[frac][(ii,jj)][k,0]

        if save_npy_dir and npy_file_key:
            for frac in fractions:
                for i in range(3):
                    for j in range(3):
                        np.save(get_npy_pathname('num_max_pix_on_ec', frac, i, j), flight_events_num_max_pix_on_ec[frac][(i, j)])

        #                 if k == 4:
        #                     print("flight_events_num_max_pix_on_ec[({i},{j})][{frac}][{k},0] = {v}   flight_events_num_max_pix_on_ec[({i},{j})][{frac}][{k},1] = {v1}".format(
        #                         i=i,j=j,frac=frac,k=k,v=flight_events_num_max_pix_on_ec[(i,j)][frac][k,0], v1=flight_events_num_max_pix_on_ec[(i,j)][frac][k,1]))
        #     if k > 5:
        #         break


        for k in range(len(df)):
            for frac in fractions:
                for i in range(6):
                    for j in range(6):
                        for ii in range(6):
                            for jj in range(6):
                                if jj == j and ii == i:
                                    continue
                                flight_events_num_max_pix_on_pmt[frac][(i,j)][k,1] += flight_events_num_max_pix_on_pmt[frac][(ii,jj)][k,0]

        if save_npy_dir and npy_file_key:
            for frac in fractions:
                for i in range(6):
                    for j in range(6):
                        np.save(get_npy_pathname('num_max_pix_on_pmt', frac, i, j), flight_events_num_max_pix_on_pmt[frac][(i, j)])

        #                         if flight_events_num_pix[(ii,jj)][frac][k,0] > 0:
        #                             print('flight_events_num_pix[({i},{j})][{frac}][{k},1] += flight_events_num_pix[({ii},{jj})][{frac}][{k},1] # {v}'.format(
        #                                 i=i,j=j,frac=frac,k=k,ii=ii,jj=jj,v=flight_events_num_pix[(ii,jj)][frac][k,0]))
        #     if k > 5:
        #         break

        return flight_events_num_max_pix_on_pmt, flight_events_num_max_pix_on_ec
    else:

        return pickled_flight_events_num_max_pix_on_pmt, pickled_flight_events_num_max_pix_on_ec

    # In[211]:

def extend_df_with_num_max_pix(flight_events_within_cond, flight_events_num_max_pix_on_pmt, flight_events_num_max_pix_on_ec):

    # for c in flight_events_within_cond.columns:
    #     print(c)

    #np.where(flight_events_num_pix[(0,0)][.6] > 0)
    #flight_events_num_pix[(0,1)][0.6][:,1] > 0
    flight_events_within_cond_cp = flight_events_within_cond.copy() # this is NOT creating the copy
    k=0
    # for frac in fractions:
    #     for i in range(6):
    #         for j in range(6):
    for frac, frac_dict in flight_events_num_max_pix_on_pmt.items():
        for pmt_pos, pmt_counts in frac_dict.items():
            i, j = pmt_pos
            col = 'pmt_{:d}_{:d}'.format(i,j) + '_frac{:.1f}'.format(frac).replace('.','')
            #print(pd.Series(flight_events_num_pix[(i,j)][frac][:,0]))
            #print(col, flight_events_num_pix[(i,j)][frac][k,1] , flight_events_num_pix[(ii,jj)][frac][k,0])
            flight_events_within_cond_cp[col + '_in'] = pd.Series(pmt_counts[:,0])
            flight_events_within_cond_cp[col + '_out'] = pd.Series(pmt_counts[:,1])

    # for frac in fractions:
    #     for i in range(3):
    #         for j in range(3):
    for frac, frac_dict in flight_events_num_max_pix_on_ec.items():
        for ec_pos, ec_counts in frac_dict.items():
            i, j = ec_pos
            col = 'ec_{:d}_{:d}'.format(i,j) + '_frac{:.1f}'.format(frac).replace('.','')
            flight_events_within_cond_cp[col + '_in'] = pd.Series(ec_counts[:,0])
            flight_events_within_cond_cp[col + '_out'] = pd.Series(ec_counts[:,1])

    print(len(flight_events_within_cond_cp.columns))
    return flight_events_within_cond_cp


def vis_num_gtu_hist(flight_events_within_cond_cp, save_fig_dir, fig_file_name='num_gtu_hist.png'):
    # for k,v in flight_events_within_cond_cp[2000:].iloc[64].iteritems():
    #     if k in ["event_id"] or k.startswith('ec')
    #     print("{}\t{}".format(k,v))
    if len(flight_events_within_cond_cp) > 0:
        fig,ax = plt.subplots(1)
        flight_events_within_cond_cp['num_gtu'].hist(ax=ax, bins=30)
        fig.set_size_inches(25,5)
        if save_fig_dir is not None:
            save_figure(fig, save_fig_dir, fig_file_name)
        else:
            plt.show()


# def vis_col_num_gtu_hist(simu_entries_within_cond_bgf05_and_bgf1, save_fig_dir, fig_file_name='simu_entries_within_cond_bgf05_and_bgf1_num_gtu.png'):
#     if len(simu_entries_within_cond_bgf05_and_bgf1) > 0:
#         fig,ax = plt.subplots(1)
#         simu_entries_within_cond_bgf05_and_bgf1['num_gtu'].hist(ax=ax, bins=25)
#         fig.set_size_inches(25,5)
#         ax.set_yscale('log')
#
#         if save_fig_dir is not None:
#             save_figure(fig, save_fig_dir, fig_file_name)
#         else:
#             plt.show()


def df_difference(df1, df2, unique_column='event_id'):
    return df1[~df1[unique_column].isin(df2[unique_column])]
    # merged = df1.merge(df2, indicator=True, how='outer')
    # merged[merged['_merge'] == 'right_only']
    # return merged[merged['_merge'] == 'left_only']


def filter_out_by_fraction(flight_events_within_cond_cp, ec_0_0_frac_lt=0.5, ec_in_column='ec_0_0_frac06_in', ec_out_column='ec_0_0_frac06_out'):
    ec_0_0_frac = flight_events_within_cond_cp[ec_in_column] / flight_events_within_cond_cp[ec_out_column]
    # filtered_flight_events_within_cond = flight_events_within_cond_cp[ (flight_events_within_cond_cp['ec_0_0_frac06_out'] == 0) ]
    filtered_flight_events_within_cond = flight_events_within_cond_cp[ (flight_events_within_cond_cp[ec_out_column] != 0) & (ec_0_0_frac < ec_0_0_frac_lt) ]
    return filtered_flight_events_within_cond


def filter_out_col_thr(flight_events_within_cond_cp, num_gtu_gt=15, num_gtu_column='num_gtu'): # num_gtu BETWEEN 10 AND 40
    filtered_flight_events_within_cond = flight_events_within_cond_cp[ flight_events_within_cond_cp[num_gtu_column] > num_gtu_gt ]
    return filtered_flight_events_within_cond


def filter_out_by_fraction_and_col_thr(flight_events_within_cond_cp, ec_0_0_frac_lt=0.5, num_gtu_gt=15, ec_in_column='ec_0_0_frac06_in', ec_out_column='ec_0_0_frac06_out', num_gtu_column='num_gtu'): # num_gtu BETWEEN 10 AND 40
    ec_0_0_frac = flight_events_within_cond_cp[ec_in_column] / flight_events_within_cond_cp[ec_out_column]
    # filtered_flight_events_within_cond = flight_events_within_cond_cp[ (flight_events_within_cond_cp['ec_0_0_frac06_out'] == 0) ]
    filtered_flight_events_within_cond = flight_events_within_cond_cp[ (flight_events_within_cond_cp[ec_out_column] != 0) & (ec_0_0_frac < ec_0_0_frac_lt) & (flight_events_within_cond_cp[num_gtu_column] > num_gtu_gt) ]
    return filtered_flight_events_within_cond


def vis_events_df(events, save_fig_dir, base_file_name, events_per_figure=50, max_figures=10, vis_gtux=True, vis_gtuy=True, additional_printed_columns=[], close_after_vis=True):
    events_l = [ev for ev in rows_generator(events.iterrows())]
    vis_events_list(events_l, events.columns, save_fig_dir, base_file_name, events_per_figure, max_figures, additional_printed_columns=additional_printed_columns,
               vis_gtux=vis_gtux, vis_gtuy=vis_gtuy, close_after_vis=close_after_vis)


def vis_events_list(events, column_labels, save_fig_dir, base_file_name, events_per_figure=50, max_figures=10, vis_gtux=True, vis_gtuy=True, numeric_columns=False, additional_printed_columns=[], subplot_cols=9, close_after_vis=True):
    for i in range(0, min(len(events), events_per_figure*max_figures), events_per_figure):
       fig, axs = \
           data_analysis_utils.visualize_events(
               events[i:], column_labels, events_per_figure, additional_printed_columns=additional_printed_columns,
               vis_gtux=vis_gtux, vis_gtuy=vis_gtuy, subplot_cols=subplot_cols, numeric_columns=numeric_columns,
               plt_show=False, event_count_offset=i)

       if save_fig_dir is not None:
           # print("XXXXXXXXXXXXXX")
           # print(fig, save_fig_dir, "{}_{:d}_{:d}".format(base_file_name, i, min(i+events_per_figure, len(events))))
           save_figure(fig, save_fig_dir, "{}_{:d}_{:d}".format(base_file_name, i, min(i+events_per_figure, len(events))))
           # ,'x_y_neighbourhood_size','gtu_x_neighbourhood_size','gtu_y_neighbourhood_size'

       if close_after_vis:
            plt.close('all')


def main(argv):

    args_parser = argparse.ArgumentParser(description='Draw histograms of parameter values')
    args_parser.add_argument('-d', '--dbname', default='eusospb_data')
    args_parser.add_argument('-U', '--user', default='eusospb')
    args_parser.add_argument('--password')
    args_parser.add_argument('-s', '--host', default='localhost')
    args_parser.add_argument('-o', '--save-fig-dir', default='/tmp/event_classification_efficiency', help="Directory where figures are saved (default: /tmp/event_classification_efficiency)")
    args_parser.add_argument('-c', '--save-csv-dir', default='/tmp/event_classification_efficiency', help="Directory where csv are saved (default: /tmp/event_classification_efficiency)")
    args_parser.add_argument('-p', '--save-npy-dir', default='/tmp/event_classification_efficiency/npy', help="Directory where numpy arrays are stored (default: /tmp/event_classification_efficiency/npy)")
    args_parser.add_argument('--show-plots', type=str2bool_argparse, default=False, help='If true, plots are only showed in windows (default: no)')
    args_parser.add_argument('--exit-on-failure', type=str2bool_argparse, default=True, help='If true, exits on failure (default: yes)')
    args_parser.add_argument('--skip-vis-events', type=str2bool_argparse, default=False, help='If true, events are not visualized (default: no)')
    args_parser.add_argument('--do-flight', type=str2bool_argparse, default=True, help='If true, flight events are processed (default: yes)')
    args_parser.add_argument('--do-utah', type=str2bool_argparse, default=True, help='If true, utah events are processed (default: yes)')
    args_parser.add_argument('--do-simu', type=str2bool_argparse, default=True, help='If true, simu events are processed (default: yes)')
    args_parser.add_argument('--print-debug-messages', type=str2bool_argparse, default=False, help='If true, debug messages are printed (default: no)')
    args_parser.add_argument('--max-vis-pages-within-cond', type=int, default=10, help='Number of visualized pages/images of events within conditions (default: 10)')
    args_parser.add_argument('--max-vis-pages-not-within-cond', type=int, default=10, help='Number of visualized pages/images of events not within conditions (default: 10)')
    args_parser.add_argument('--max-vis-pages-within-cond-filtered', type=int, default=10, help='Number of visualized pages/images of events within conditions and passed through the filter (default: 10)')
    args_parser.add_argument('--max-vis-pages-within-cond-filtered-out', type=int, default=10, help='Number of visualized pages/images of events within conditions and not passed through the filter (default: 10)')

    args = args_parser.parse_args(argv)

    password = args.password
    if not password:
        password = getpass.getpass()

    save_npy_dir = os.path.realpath(args.save_npy_dir)
    save_csv_dir = os.path.realpath(args.save_csv_dir)

    if not args.show_plots:
        save_fig_dir = os.path.realpath(args.save_fig_dir)
    else:
        save_fig_dir = None

    if not os.path.exists(save_npy_dir):
        os.makedirs(save_npy_dir)
    elif not os.path.isdir(save_npy_dir):
        raise RuntimeError('{} is not directory'.format(save_npy_dir))

    if not os.path.exists(save_csv_dir):
        os.makedirs(save_csv_dir)
    elif not os.path.isdir(save_csv_dir):
        raise RuntimeError('{} is not directory'.format(save_csv_dir))

    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir)
    elif not os.path.isdir(save_fig_dir):
        raise RuntimeError('{} is not directory'.format(save_fig_dir))

    queries_log = open(os.path.join(save_csv_dir, 'queries.sql'), 'w')

    # -----------------------------------------------------
    # COND SELECTION RULES
    # -----------------------------------------------------
    cond_selection_rules = get_selection_rules()
    # -----------------------------------------------------

    con, cur = get_conn(dbname=args.dbname, user=args.user, host=args.host, password=password)

    # -----------------------------------------------------
    print("COLUMNS")
    # -----------------------------------------------------

    spb_processing_event_ver2_columns = get_spb_processing_event_ver2_columns(cur)
    if args.do_simu:
        # -----------------------------------------------------
        print("ALL SIMU EVENTS BY ENERGY")
        # -----------------------------------------------------

        all_bgf05_and_bgf1_simu_events__packet_count_by_energy = None
        try:
            all_bgf05_and_bgf1_simu_events__packet_count_by_energy = get_all_bgf05_and_bgf1_simu_events__packet_count_by_energy(con, queries_log)

            print_len(all_bgf05_and_bgf1_simu_events__packet_count_by_energy, 'all_bgf05_and_bgf1_simu_events__packet_count_by_energy')
            save_csv(all_bgf05_and_bgf1_simu_events__packet_count_by_energy, save_fig_dir, 'all_bgf05_and_bgf1_simu_events__packet_count_by_energy')

            vis_df_etruth_trueenergy_count_packets(all_bgf05_and_bgf1_simu_events__packet_count_by_energy, save_csv_dir, 'all_bgf05_and_bgf1_simu_events__count_packets_by_energy')

        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)

        # -----------------------------------------------------
        print("ALL SIMU EVENTS COUNT WITHIN CONDITIONS")
        # -----------------------------------------------------

        try:
            cond_simu_entries_count = get_count_simu_entries_within_cond(con, cond_selection_rules, queries_log)
            print("cond_simu_entries_count = {}".format(cond_simu_entries_count))
        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)

        # -----------------------------------------------------
        print("ALL SIMU EVENTS WITHIN CONDITIONS")
        # -----------------------------------------------------

        try:
            simu_entries_within_cond_bgf05_and_bgf1__only_1bgf_lt_05bgf = get_simu_entries_within_cond_bgf05_and_bgf1__only_1bgf_lt_05bgf(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log)
            print_len(simu_entries_within_cond_bgf05_and_bgf1__only_1bgf_lt_05bgf, 'simu_entries_within_cond_bgf05_and_bgf1__only_1bgf_lt_05bgf', 'expected to be empty')
            save_csv(simu_entries_within_cond_bgf05_and_bgf1__only_1bgf_lt_05bgf, save_csv_dir, 'simu_entries_within_cond_bgf05_and_bgf1__only_1bgf_lt_05bgf')

            simu_entries_within_cond_bgf05_and_bgf1 = get_simu_entries_within_cond_bgf05_and_bgf1(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log)
            print_len(simu_entries_within_cond_bgf05_and_bgf1, 'simu_entries_within_cond_bgf05_and_bgf1', 'as many as possible of {}'.format(cond_simu_entries_count))
            save_csv(simu_entries_within_cond_bgf05_and_bgf1, save_csv_dir, 'simu_entries_within_cond_bgf05_and_bgf1')

            simu_entries_within_cond_bgf05_and_bgf1_v2 = get_simu_entries_within_cond_bgf05_and_bgf1_v2(con, cond_selection_rules, queries_log)
            print_len(simu_entries_within_cond_bgf05_and_bgf1_v2, 'simu_entries_within_cond_bgf05_and_bgf1_v2', 'as many as possible of {} and same as {}'.format(cond_simu_entries_count, len(simu_entries_within_cond_bgf05_and_bgf1)))
            save_csv(simu_entries_within_cond_bgf05_and_bgf1_v2, save_csv_dir, 'simu_entries_within_cond_bgf05_and_bgf1')

            vis_num_gtu_hist(simu_entries_within_cond_bgf05_and_bgf1, save_fig_dir, 'simu_entries_within_cond_bgf05_and_bgf1__num_gtu')

            multiple_event_id_rows_row_idxs, multiple_event_id_rows_row_event_ids = find_multiple_event_id_rows(simu_entries_within_cond_bgf05_and_bgf1_v2)

            print_len(multiple_event_id_rows_row_idxs, 'multiple_event_id_rows_row_idxs')
        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)

        plt.close('all')


    # -----------------------------------------------------
    print("EVENTS COUNT WITHIN CONDITIONS")
    # -----------------------------------------------------

    try:
        if args.do_utah:
            count_utah_entries_within_cond = get_count_utah_entries_within_cond(con, cond_selection_rules, queries_log)
            print("count_utah_entries_within_cond = {}".format(count_utah_entries_within_cond))

        if args.do_flight:
            count_flight_entries_within_cond = get_count_flight_entries_within_cond(con, cond_selection_rules, queries_log)
            print("count_flight_entries_within_cond = {}   // 7 EC".format(count_flight_entries_within_cond))

            count_flight_entries_within_cond_3ec = get_count_flight_entries_within_cond_num_ec(con, cond_selection_rules, 3, queries_log)
            print("count_flight_entries_within_cond_3ec = {}   // 3 EC".format(count_flight_entries_within_cond_3ec))

    except Exception:
        traceback.print_exc()
        if args.exit_on_failure:
            sys.exit(2)

    if args.do_simu:
        # -----------------------------------------------------
        print("SIMU EVENTS BY ENERGY WITHIN CONDITIONS BY ENERGY")
        # -----------------------------------------------------

        try:
            cond_bgf05_and_bgf1_simu_events__packet_count_by_energy = get_cond_bgf05_and_bgf1_simu_events__packet_count_by_energy(con, cond_selection_rules, queries_log)

            print_len(cond_bgf05_and_bgf1_simu_events__packet_count_by_energy, 'cond_bgf05_and_bgf1_simu_events__packet_count_by_energy','should be similar to {}'.format(len(all_bgf05_and_bgf1_simu_events__packet_count_by_energy)))
            save_csv(cond_bgf05_and_bgf1_simu_events__packet_count_by_energy, save_csv_dir, 'cond_bgf05_and_bgf1_simu_events__packet_count_by_energy')

            if all_bgf05_and_bgf1_simu_events__packet_count_by_energy is None:
                raise RuntimeError('all_bgf05_and_bgf1_simu_events__packet_count_by_energy is not loaded')

            vis_df_comparison(all_bgf05_and_bgf1_simu_events__packet_count_by_energy, cond_bgf05_and_bgf1_simu_events__packet_count_by_energy, save_fig_dir, fig_file_name='cond_all_bgf05_and_bgf1_simu_events__packet_count_by_energy__comparison__linear', yscale='linear')
            vis_df_comparison(all_bgf05_and_bgf1_simu_events__packet_count_by_energy, cond_bgf05_and_bgf1_simu_events__packet_count_by_energy, save_fig_dir, fig_file_name='cond_all_bgf05_and_bgf1_simu_events__packet_count_by_energy__comparison__log', yscale='log')

            cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy = merge_cond_all_dataframes(cond_bgf05_and_bgf1_simu_events__packet_count_by_energy, all_bgf05_and_bgf1_simu_events__packet_count_by_energy, merge_on='etruth_trueenergy')

            print_len(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy')

            save_csv(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, save_csv_dir,  'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy')

            # -----------------------------------------------------
            print(">> FITTING")
            # -----------------------------------------------------

            #_, yerrs = calc_yerrs_for_merged_events_by_energy
            x, y, yerrs, fits_p  = fit_points_cond_all_merged(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, x_axis_column='etruth_trueenergy')

            save_csv_of_fits(fits_p, save_csv_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__fits')

            vis_count_fraction(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, None, yerrs, save_fig_dir, fig_file_name='cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy')

            vis_count_fraction_fits(x, y, None, yerrs, fits_p, save_fig_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__fits')
            vis_count_fraction_fits(x, y, None, yerrs, [fits_p[0]], save_fig_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__1poly_fit')

            # -----------------------------------------------------
            print(">> THINNING")
            # -----------------------------------------------------

            x, y, xerrs, yerrs, cond_thinned, all_thinned  = thin_datapoints_from_dataframe(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, x_axis_column='etruth_trueenergy') # xerrs = xerr_low, xerr_up

            save_thinned_datapoints(x, y, xerrs, yerrs, cond_thinned, all_thinned, save_csv_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy')

            # -----------------------------------------------------
            print(">> THINNING FITTED")
            # -----------------------------------------------------

            fits_p = fit_points_using_yerrs(x, y, yerrs)

            save_csv_of_fits(fits_p, save_csv_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__fits')

            vis_thinned_datapoints(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, *cond_thinned, *all_thinned, save_fig_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__thinned_comparison')

            vis_count_fraction_fits(x, y, xerrs, yerrs, fits_p, save_fig_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__thinned__fits')
            vis_count_fraction_fits(x, y, xerrs, yerrs, [fits_p[0]], save_fig_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__thinned__1poly_fit')

            plt.close('all')

        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)


        # -----------------------------------------------------
        print("ALL SIMU EVENTS BY POSZ")
        # -----------------------------------------------------

        all_bgf05_and_bgf1_simu_events__packet_count_by_posz = None
        try:
            all_bgf05_and_bgf1_simu_events__packet_count_by_posz = get_all_bgf05_and_bgf1_simu_events__packet_count_by_posz(con, queries_log)

            print_len(all_bgf05_and_bgf1_simu_events__packet_count_by_posz, 'all_bgf05_and_bgf1_simu_events__packet_count_by_posz')
            save_csv(all_bgf05_and_bgf1_simu_events__packet_count_by_posz, save_csv_dir, 'all_bgf05_and_bgf1_simu_events__packet_count_by_posz')

        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)


        # -----------------------------------------------------
        print("SIMU EVENTS WITHIN CONDITIONS BY POSZ")
        # -----------------------------------------------------

        try:
            cond_bgf05_and_bgf1_simu_events__packet_count_by_posz = get_cond_bgf05_and_bgf1_simu_events__packet_count_by_posz(con, cond_selection_rules, queries_log)

            print_len(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz, 'cond_bgf05_and_bgf1_simu_events__packet_count_by_posz')
            save_csv(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz, save_csv_dir, 'cond_bgf05_and_bgf1_simu_events__packet_count_by_posz')

            if all_bgf05_and_bgf1_simu_events__packet_count_by_posz is None:
                raise RuntimeError('all_bgf05_and_bgf1_simu_events__packet_count_by_posz is not loaded')

            vis_df_comparison(all_bgf05_and_bgf1_simu_events__packet_count_by_posz, cond_bgf05_and_bgf1_simu_events__packet_count_by_posz, save_fig_dir, 'cond_all_bgf05_and_bgf1_simu_events__packet_count_by_posz__comparison__linear', xaxis='egeometry_pos_z', yscale='linear')
            vis_df_comparison(all_bgf05_and_bgf1_simu_events__packet_count_by_posz, cond_bgf05_and_bgf1_simu_events__packet_count_by_posz, save_fig_dir, 'cond_all_bgf05_and_bgf1_simu_events__packet_count_by_posz__comparison__log', xaxis='egeometry_pos_z', yscale='log')

            cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz = merge_cond_all_dataframes(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz, all_bgf05_and_bgf1_simu_events__packet_count_by_posz, merge_on='egeometry_pos_z')

            print_len(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz')
            save_csv(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, save_csv_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz')

            # -----------------------------------------------------
            print(">> FITTING")
            # -----------------------------------------------------

            #_, yerrs = calc_yerrs_for_merged_events_by_energy
            x, y, yerrs, fits_p = fit_points_cond_all_merged(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, x_axis_column='egeometry_pos_z')

            save_csv_of_fits(fits_p, save_csv_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__fits')

            vis_count_fraction(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, None, yerrs, save_fig_dir, fig_file_name='cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz', x_axis_column='egeometry_pos_z')

            vis_count_fraction_fits(x, y, None, yerrs, fits_p, save_fig_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__fits', xlabel='Altitude (EGeimetry.Pos.Z')
            vis_count_fraction_fits(x, y, None, yerrs, [fits_p[0]], save_fig_dir, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__1poly_fit', xlabel='Altitude (EGeimetry.Pos.Z')

            plt.close('all')

        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)

        # -----------------------------------------------------
        print("ALL SIMU EVENTS BY ENERGY AND POSZ")
        # -----------------------------------------------------

        all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy = None
        try:
            all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy = get_all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy(con, queries_log)

            print_len(all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, 'all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy')
            save_csv(all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, save_csv_dir, 'all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy')

        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)

        # -----------------------------------------------------
        print("SIMU EVENTS WITHIN CONDITIONS BY ENERGY AND POSZ")
        # -----------------------------------------------------

        try:
            cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy = get_cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy(con, cond_selection_rules, queries_log)

            print_len(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, 'cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy')
            save_csv(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, save_csv_dir, 'cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy')

            #vis_df_comparison(all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, save_fig_dir, fig_file_name='cond_all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy__comparison__linear', yscale='linear')
            #vis_df_comparison(all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, save_fig_dir, fig_file_name='cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy__comparison__log', yscale='log')
            # cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona = merge_

            if all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy is None:
                raise RuntimeError('all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy is not loaded')

            cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy = merge_cond_all_dataframes(cond_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, merge_on=['etruth_trueenergy','egeometry_pos_z'],)

            print_len(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, 'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy')
            save_csv(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, save_csv_dir,  'cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy')

            cond_all_merged_bgf05_simu_events_by_energy_thin_fit_posz_groups = get_cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit(cond_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy)

            vis_cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit(cond_all_merged_bgf05_simu_events_by_energy_thin_fit_posz_groups, save_fig_dir, 'cond_all_merged_bgf05_simu_events_by_energy_thin_fit_posz_groups')

            plt.close('all')

        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)

    if args.do_flight:
        # -----------------------------------------------------
        print("FLIGHT EVENTS BY ENERGY WITHIN CONDITIONS")
        # -----------------------------------------------------

        try:
            flight_events_within_cond = select_flight_events_within_cond(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log)

            print_len(flight_events_within_cond, 'flight_events_within_cond')
            save_csv(flight_events_within_cond, save_csv_dir, 'flight_events_within_cond')

            print(">> COUNTING MAX PIXELS ON PMTS AND ECS")

            flight_events_num_max_pix_on_pmt, flight_events_num_max_pix_on_ec = count_num_max_pix_on_pmt_and_ec(flight_events_within_cond, [0.6, 0.8, 0.9], save_npy_dir, 'flight_events_within_cond', args.print_debug_messages)

            flight_events_within_cond_with_max_pix_count = extend_df_with_num_max_pix(flight_events_within_cond, flight_events_num_max_pix_on_pmt, flight_events_num_max_pix_on_ec)

            print_len(flight_events_within_cond_with_max_pix_count, 'flight_events_within_cond_with_max_pix_count')
            save_csv(flight_events_within_cond_with_max_pix_count, save_csv_dir, 'flight_events_within_cond_with_max_pix_count')

            print(">> FILTERING (EC_0_0/OTHER_EC < 0.5)")

            filtered_flight_events_within_cond_ec_0_0 = filter_out_by_fraction(flight_events_within_cond_with_max_pix_count, ec_0_0_frac_lt=0.5)

            print_len(filtered_flight_events_within_cond_ec_0_0, 'filtered_flight_events_within_cond_ec_0_0')
            save_csv(filtered_flight_events_within_cond_ec_0_0, save_csv_dir, 'filtered_flight_events_within_cond_ec_0_0')

            # -----------------------------------------------------

            print(">> FILTERING (EC_0_0/OTHER_EC < 0.6)")

            filtered_flight_events_within_cond_ec_0_0_lt06 = filter_out_by_fraction(flight_events_within_cond_with_max_pix_count, ec_0_0_frac_lt=0.6)

            print_len(filtered_flight_events_within_cond_ec_0_0_lt06, 'filtered_flight_events_within_cond_ec_0_0_lt06')
            save_csv(filtered_flight_events_within_cond_ec_0_0_lt06, save_csv_dir, 'filtered_flight_events_within_cond_ec_0_0_lt06')

            # -----------------------------------------------------

            print(">> FILTERING (EC_0_0/OTHER_EC < 0.6 AND NUM_GTU > 13)")

            filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu = filter_out_by_fraction_and_col_thr(flight_events_within_cond_with_max_pix_count, ec_0_0_frac_lt=0.6, num_gtu_gt=13)

            print_len(filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu, 'filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu')
            save_csv(filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu, save_csv_dir, 'filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu')

            # -----------------------------------------------------

            print(">> FILTERING (EC_0_0/OTHER_EC < 0.5 AND NUM_GTU > 15)")

            filtered_flight_events_within_cond = filter_out_by_fraction_and_col_thr(flight_events_within_cond_with_max_pix_count, ec_0_0_frac_lt=0.5, num_gtu_gt=15)

            print_len(filtered_flight_events_within_cond, 'filtered_flight_events_within_cond')
            save_csv(filtered_flight_events_within_cond, save_csv_dir, 'filtered_flight_events_within_cond')

            # -----------------------------------------------------

            print(">> SELECTING NOT PASSED THROUGH FILTER (EC_0_0/OTHER_EC < 0.5 AND NUM_GTU > 15)")

            flight_events_within_cond_not_filter = df_difference(flight_events_within_cond_with_max_pix_count, filtered_flight_events_within_cond)

            print_len(flight_events_within_cond_not_filter, 'flight_events_within_cond_not_filter')
            save_csv(flight_events_within_cond_not_filter, save_csv_dir, 'flight_events_within_cond_not_filter')

            # -----------------------------------------------------

            print(">> SELECTING NOT PASSED THROUGH FILTER (EC_0_0/OTHER_EC < 0.6 AND NUM_GTU > 13)")

            flight_events_within_cond_not_filter_ec_0_0_lt06_gt13gtu = df_difference(flight_events_within_cond_with_max_pix_count, filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu)

            print_len(flight_events_within_cond_not_filter_ec_0_0_lt06_gt13gtu, 'flight_events_within_cond_not_filter_ec_0_0_lt06_gt13gtu')
            save_csv(flight_events_within_cond_not_filter_ec_0_0_lt06_gt13gtu, save_csv_dir, 'flight_events_within_cond_not_filter_ec_0_0_lt06_gt13gtu')

            # -----------------------------------------------------

            print(">> VISUALIZING WITHIN CONDITIONS")
            vis_num_gtu_hist(flight_events_within_cond, save_fig_dir, fig_file_name='flight_events_within_cond__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(flight_events_within_cond, save_fig_dir, 'flight_events_within_cond', max_figures=args.max_vis_pages_within_cond)

            print(">> VISUALIZING FILTERED (EC_0_0/OTHER_EC < 0.5 AND NUM_GTU > 15) WITHIN CONDITIONS")
            vis_num_gtu_hist(filtered_flight_events_within_cond, save_fig_dir, fig_file_name='filtered_flight_events_within_cond__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(filtered_flight_events_within_cond, save_fig_dir, 'filtered_flight_events_within_cond', additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], max_figures=args.max_vis_pages_within_cond_filtered)

            print(">> VISUALIZING WITHIN CONDITIONS NOT PASSED THROUGH THE FILTER (EC_0_0/OTHER_EC < 0.5 AND NUM_GTU > 15)")
            vis_num_gtu_hist(flight_events_within_cond_not_filter, save_fig_dir, fig_file_name='flight_events_within_cond_not_filter__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(flight_events_within_cond_not_filter, save_fig_dir, 'flight_events_within_cond_not_filter', additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], max_figures=args.max_vis_pages_within_cond_filtered_out)

            print(">> VISUALIZING FILTERED (EC_0_0/OTHER_EC < 0.6 AND NUM_GTU > 13) WITHIN CONDITIONS")
            vis_num_gtu_hist(filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu, save_fig_dir, fig_file_name='filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu, save_fig_dir, 'filtered_flight_events_within_cond_ec_0_0_lt06_gt13gtu', additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], max_figures=args.max_vis_pages_within_cond_filtered)

            print(">> VISUALIZING WITHIN CONDITIONS NOT PASSED THROUGH THE FILTER (EC_0_0/OTHER_EC < 0.6 AND NUM_GTU > 13)")
            vis_num_gtu_hist(flight_events_within_cond_not_filter_ec_0_0_lt06_gt13gtu, save_fig_dir, fig_file_name='flight_events_within_cond_not_filter_ec_0_0_lt06_gt13gtu__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(flight_events_within_cond_not_filter_ec_0_0_lt06_gt13gtu, save_fig_dir, 'flight_events_within_cond_not_filter_ec_0_0_lt06_gt13gtu', additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], max_figures=args.max_vis_pages_within_cond_filtered_out)


        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)

    if args.do_utah:
        # -----------------------------------------------------
        print("UTAH EVENTS BY ENERGY WITHIN CONDITIONS")
        # -----------------------------------------------------

        try:
            utah_events_within_cond = select_utah_events_within_cond(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log)

            print_len(utah_events_within_cond, 'utah_events_within_cond')
            save_csv(utah_events_within_cond, save_csv_dir, 'utah_events_within_cond')

            print(">> COUNTING MAX PIXELS ON PMTS AND ECS")

            utah_events_num_max_pix_on_pmt, utah_events_num_max_pix_on_ec = count_num_max_pix_on_pmt_and_ec(utah_events_within_cond, [0.6, 0.8, 0.9], save_npy_dir, 'utah_events_within_cond', args.print_debug_messages)

            utah_events_within_cond_with_max_pix_count = extend_df_with_num_max_pix(utah_events_within_cond, utah_events_num_max_pix_on_pmt, utah_events_num_max_pix_on_ec)

            print_len(utah_events_within_cond_with_max_pix_count, 'utah_events_within_cond_with_max_pix_count')
            save_csv(utah_events_within_cond_with_max_pix_count, save_csv_dir, 'utah_events_within_cond_with_max_pix_count')

            print(">> FILTERING")
            filtered_utah_events_within_cond = filter_out_by_fraction_and_col_thr(utah_events_within_cond_with_max_pix_count, ec_0_0_frac_lt=0.5, num_gtu_gt=15)

            print_len(filtered_utah_events_within_cond, 'filtered_utah_events_within_cond')
            save_csv(filtered_utah_events_within_cond, save_csv_dir, 'filtered_utah_events_within_cond')

            print(">> SELECTING NOT PASSED THROUGH FILTER")

            utah_events_within_cond_not_filter = df_difference(utah_events_within_cond_with_max_pix_count, filtered_utah_events_within_cond)

            print_len(utah_events_within_cond_not_filter, 'utah_events_within_cond_not_filter')
            save_csv(utah_events_within_cond_not_filter, save_csv_dir, 'utah_events_within_cond_not_filter')

            print(">> VISUALIZING WITHIN CONDITIONS")
            vis_num_gtu_hist(utah_events_within_cond, save_fig_dir, fig_file_name='simu_events_within_cond__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(utah_events_within_cond, save_fig_dir, 'utah_events_within_cond', max_figures=args.max_vis_pages_within_cond)

            print(">> VISUALIZING FILTERED WITHIN CONDITIONS")
            vis_num_gtu_hist(filtered_utah_events_within_cond, save_fig_dir, fig_file_name='filtered_utah_events_within_cond__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(filtered_utah_events_within_cond, save_fig_dir, 'filtered_utah_events_within_cond', additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], max_figures=args.max_vis_pages_within_cond_filtered)

            print(">> VISUALIZING WITHIN CONDITIONS NOT PASSED THROUGH THE FILTER")
            vis_num_gtu_hist(utah_events_within_cond_not_filter, save_fig_dir, fig_file_name='utah_events_within_cond_not_filter__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(utah_events_within_cond_not_filter, save_fig_dir, 'utah_events_within_cond_not_filter', additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], max_figures=args.max_vis_pages_within_cond_filtered_out)

        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)

    if args.do_simu:
        # -----------------------------------------------------
        print("SIMU EVENTS WITHIN CONDITIONS")
        # -----------------------------------------------------

        try:
            simu_events_within_cond = select_simu_events_within_cond(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log)

            print_len(simu_events_within_cond, 'simu_events_within_cond')
            save_csv(simu_events_within_cond, save_csv_dir, 'simu_events_within_cond')
            vis_num_gtu_hist(simu_events_within_cond, save_fig_dir, fig_file_name='simu_events_within_cond__num_gtu')

            print(">> COUNTING MAX PIXELS ON PMTS AND ECS")

            simu_events_num_max_pix_on_pmt, simu_events_num_max_pix_on_ec = count_num_max_pix_on_pmt_and_ec(simu_events_within_cond, [0.6, 0.8, 0.9], save_npy_dir, 'simu_events_within_cond', args.print_debug_messages)

            simu_events_within_cond_with_max_pix_count = extend_df_with_num_max_pix(simu_events_within_cond, simu_events_num_max_pix_on_pmt, simu_events_num_max_pix_on_ec)

            print_len(simu_events_within_cond_with_max_pix_count, 'simu_events_within_cond_with_max_pix_count')
            save_csv(simu_events_within_cond_with_max_pix_count, save_csv_dir, 'simu_events_within_cond_with_max_pix_count')

            print(">> FILTERING")

            filtered_simu_events_within_cond = filter_out_by_fraction_and_col_thr(simu_events_within_cond_with_max_pix_count, ec_0_0_frac_lt=0.5, num_gtu_gt=15)
            print_len(filtered_simu_events_within_cond, 'filtered_simu_events_within_cond')
            save_csv(filtered_simu_events_within_cond, save_csv_dir, 'filtered_simu_events_within_cond')

            # -----------------------------------------------------
            print(">> GROUPING BY ENERGY ")
            # -----------------------------------------------------

            filtered_simu_events_within_cond__packet_count_by_energy = group_rows_to_count_packets(filtered_simu_events_within_cond)

            print_len(filtered_simu_events_within_cond__packet_count_by_energy, 'filtered_simu_events_within_cond__packet_count_by_energy')
            save_csv(filtered_simu_events_within_cond__packet_count_by_energy, save_csv_dir, 'filtered_simu_events_within_cond__packet_count_by_energy')

            if all_bgf05_and_bgf1_simu_events__packet_count_by_energy is None:
                raise RuntimeError('all_bgf05_and_bgf1_simu_events__packet_count_by_energy is not loaded')

            # -----------------------------------------------------
            print(">> MERGING (GROUPED BY ENERGY)")
            # -----------------------------------------------------

            filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy = merge_cond_all_dataframes(filtered_simu_events_within_cond__packet_count_by_energy, all_bgf05_and_bgf1_simu_events__packet_count_by_energy, merge_on='etruth_trueenergy')

            print_len(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy')
            save_csv(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, save_csv_dir,  'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy')

            # -----------------------------------------------------
            print(">> FITTING (GROUPED BY ENERGY)")
            # -----------------------------------------------------

            #_, yerrs = calc_yerrs_for_merged_events_by_energy
            x, y, yerrs, fits_p  = fit_points_cond_all_merged(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, x_axis_column='etruth_trueenergy')

            save_csv_of_fits(fits_p, save_csv_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__fits')

            vis_count_fraction(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, None, yerrs, save_fig_dir, fig_file_name='filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy')
            vis_count_fraction_fits(x, y, None, yerrs, fits_p, save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__fits')
            vis_count_fraction_fits(x, y, None, yerrs, [fits_p[0]], save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__1poly_fit')

            # -----------------------------------------------------
            print(">> THINNING (GROUPED BY ENERGY)")
            # -----------------------------------------------------

            x, y, xerrs, yerrs, cond_thinned, all_thinned  = thin_datapoints_from_dataframe(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, x_axis_column='etruth_trueenergy') # xerrs = xerr_low, xerr_up

            save_thinned_datapoints(x, y, xerrs, yerrs, cond_thinned, all_thinned, save_csv_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy')

            # -----------------------------------------------------
            print(">> THINNING FITTED (GROUPED BY ENERGY)")
            # -----------------------------------------------------

            fits_p = fit_points_using_yerrs(x, y, yerrs)

            save_csv_of_fits(fits_p, save_csv_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__fits')

            vis_thinned_datapoints(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy, *cond_thinned, *all_thinned, save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__thinned_comparison')

            vis_count_fraction_fits(x, y, xerrs, yerrs, fits_p, save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__thinned__fits')
            vis_count_fraction_fits(x, y, xerrs, yerrs, [fits_p[0]], save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_energy__thinned__1poly_fit')

            # -----------------------------------------------------

            # -----------------------------------------------------
            print(">> GROUPING BY POSZ ")
            # -----------------------------------------------------

            filtered_simu_events_within_cond__packet_count_by_posz = \
                group_rows_to_count_packets(filtered_simu_events_within_cond,
                                            groupby1_columns=['egeometry_pos_z','source_file_acquisition_full','packet_id'], groupby2_columns=['egeometry_pos_z'])

            print_len(filtered_simu_events_within_cond__packet_count_by_posz, 'filtered_simu_events_within_cond__packet_count_by_posz')
            save_csv(filtered_simu_events_within_cond__packet_count_by_posz, save_csv_dir, 'filtered_simu_events_within_cond__packet_count_by_posz')

            if all_bgf05_and_bgf1_simu_events__packet_count_by_posz is None:
                raise RuntimeError('all_bgf05_and_bgf1_simu_events__packet_count_by_posz is not loaded')

            # -----------------------------------------------------
            print(">> MERGING (GROUPED BY POSZ)")
            # -----------------------------------------------------

            filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz = merge_cond_all_dataframes(filtered_simu_events_within_cond__packet_count_by_posz, all_bgf05_and_bgf1_simu_events__packet_count_by_posz, merge_on='egeometry_pos_z')

            print_len(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz')
            save_csv(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, save_csv_dir,  'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz')

            # -----------------------------------------------------
            print(">> FITTING (GROUPED BY POSZ)")
            # -----------------------------------------------------

            #_, yerrs = calc_yerrs_for_merged_events_by_posz
            x, y, yerrs, fits_p  = fit_points_cond_all_merged(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, x_axis_column='egeometry_pos_z')

            save_csv_of_fits(fits_p, save_csv_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__fits')

            vis_count_fraction(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, None, yerrs, save_fig_dir, fig_file_name='filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz', x_axis_column='egeometry_pos_z')
            vis_count_fraction_fits(x, y, None, yerrs, fits_p, save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__fits', xlabel='Altitude (EGeometry.Pos.Z)')
            vis_count_fraction_fits(x, y, None, yerrs, [fits_p[0]], save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__1poly_fit', xlabel='Altitude (EGeometry.Pos.Z)')

            # -----------------------------------------------------
            print(">> THINNING (GROUPED BY POSZ)")
            # -----------------------------------------------------

            x, y, xerrs, yerrs, cond_thinned, all_thinned  = thin_datapoints_from_dataframe(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, x_axis_column='egeometry_pos_z') # xerrs = xerr_low, xerr_up

            save_thinned_datapoints(x, y, xerrs, yerrs, cond_thinned, all_thinned, save_csv_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz')

            # -----------------------------------------------------
            print(">> THINNING FITTED (GROUPED BY POSZ)")
            # -----------------------------------------------------

            fits_p = fit_points_using_yerrs(x, y, yerrs)

            save_csv_of_fits(fits_p, save_csv_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__fits')

            vis_thinned_datapoints(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz, *cond_thinned, *all_thinned, save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__thinned_comparison', x_axis_column='egeometry_pos_z')

            vis_count_fraction_fits(x, y, xerrs, yerrs, fits_p, save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__thinned__fits', xlabel='Altitude (EGeometry.Pos.Z)')
            vis_count_fraction_fits(x, y, xerrs, yerrs, [fits_p[0]], save_fig_dir, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz__thinned__1poly_fit', xlabel='Altitude (EGeometry.Pos.Z)')

            # =====================================================

            # -----------------------------------------------------
            print(">> GROUPING BY ENERGY AND POSZ")
            # -----------------------------------------------------

            filtered_simu_events_within_cond__packet_count_by_posz_and_energy = \
                group_rows_to_count_packets(filtered_simu_events_within_cond,
                                            groupby1_columns=['egeometry_pos_z','etruth_trueenergy','source_file_acquisition_full','packet_id'], groupby2_columns=['egeometry_pos_z','etruth_trueenergy'])

            print_len(filtered_simu_events_within_cond__packet_count_by_posz_and_energy, 'filtered_simu_events_within_cond__packet_count_by_posz_and_energy')
            save_csv(filtered_simu_events_within_cond__packet_count_by_posz_and_energy, save_csv_dir, 'filtered_simu_events_within_cond__packet_count_by_posz_and_energy')

            if all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy is None:
                raise RuntimeError('all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy is not loaded')

            # -----------------------------------------------------
            print(">> MERGING (GROUPED BY ENERGY AND POSZ)")
            # -----------------------------------------------------

            filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy = merge_cond_all_dataframes(filtered_simu_events_within_cond__packet_count_by_posz_and_energy, all_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, merge_on=['egeometry_pos_z','etruth_trueenergy'])

            print_len(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, 'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy')
            save_csv(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy, save_csv_dir,  'filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy')

            # -----------------------------------------------------
            print(">> THINNING AND FITTING (GROUPED BY ENERGY AND POSZ)")
            # -----------------------------------------------------

            filtered_all_merged_bgf05_simu_events_by_energy_thin_fit_posz_groups = get_cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit(filtered_all_merged_bgf05_and_bgf1_simu_events__packet_count_by_posz_and_energy)

            vis_cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit(filtered_all_merged_bgf05_simu_events_by_energy_thin_fit_posz_groups, save_fig_dir, 'filtered_all_merged_bgf05_simu_events_by_energy_thin_fit_posz_groups')

            # =====================================================

            print(">> SELECTING NOT PASSED THROUGH FILTER")

            simu_events_within_cond_not_filter = df_difference(simu_events_within_cond_with_max_pix_count, filtered_simu_events_within_cond)

            print_len(simu_events_within_cond_not_filter, 'simu_events_within_cond_not_filter')
            save_csv(simu_events_within_cond_not_filter, save_csv_dir,  'simu_events_within_cond_not_filter')

            plt.close('all')

            # -----------------------------------------------------

            print(">> VISUALIZING WITHIN CONDITIONS")
            vis_num_gtu_hist(simu_events_within_cond, save_fig_dir, fig_file_name='simu_events_within_cond__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(simu_events_within_cond, save_fig_dir, 'simu_events_within_cond', max_figures=args.max_vis_pages_within_cond)

            print(">> VISUALIZING FILTERED WITHIN CONDITIONS")
            vis_num_gtu_hist(filtered_simu_events_within_cond, save_fig_dir, fig_file_name='filtered_simu_events_within_cond__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(filtered_simu_events_within_cond, save_fig_dir, 'filtered_simu_events_within_cond', additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], max_figures=args.max_vis_pages_within_cond_filtered)

            print(">> VISUALIZING WITHIN CONDITIONS NOT PASSED THROUGH THE FILTER")
            vis_num_gtu_hist(simu_events_within_cond_not_filter, save_fig_dir, fig_file_name='simu_events_within_cond_not_filter__num_gtu')
            if not args.skip_vis_events:
                vis_events_df(simu_events_within_cond_not_filter, save_fig_dir, 'simu_events_within_cond_not_filter', additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], max_figures=args.max_vis_pages_within_cond_filtered_out)

        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)


        # -----------------------------------------------------
        print("SIMU EVENTS NOT WITHIN SQL CONDITIONS")
        # -----------------------------------------------------

        try:
            simu_events_not_within_cond = select_simu_event_not_within_cond(con, cond_selection_rules, spb_processing_event_ver2_columns, queries_log)

            print_len(simu_events_not_within_cond, 'simu_events_not_within_cond')
            save_csv(simu_events_not_within_cond, save_csv_dir, 'simu_events_not_within_cond')


            print(">> VISUALIZING")

            vis_num_gtu_hist(simu_events_not_within_cond, save_fig_dir, fig_file_name='simu_events_not_within_cond__num_gtu')
            plt.close('all')

            if not args.skip_vis_events:
                vis_events_df(simu_events_not_within_cond, save_fig_dir, 'simu_events_not_within_cond', max_figures=args.max_vis_pages_not_within_cond)

        except Exception:
            traceback.print_exc()
            if args.exit_on_failure:
                sys.exit(2)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
