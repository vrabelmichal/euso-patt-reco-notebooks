import sys
import os
import subprocess

app_base_dir = '/home/eusobg/EUSO-SPB/euso-spb-patt-reco-v1'
if app_base_dir not in sys.path:
    sys.path.append(app_base_dir)

import re
import collections
import numpy as np
import psycopg2 as pg
import pandas as pd
import pandas.io.sql as psql
import matplotlib as mpl

def __check_agg():
    args_parser = argparse.ArgumentParser(description='')
    args_parser.add_argument('--show-plots',type=str2bool_argparse,default=False,help='If true, plots are only showed in windows')
    args = args_parser.parse_args()

    if not args.show_plots:
        mpl.use('Agg')

__check_agg()        

mpl.rcParams['figure.dpi'] = 150

import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import getpass
#import ROOT

import tool.npy_frames_visualization as npy_vis
import tool.acqconv
import data_analysis_utils
from utility_functions import str2bool_argparse

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


def get_conn(*kwargs):
    con = pg.connect(**kwargs) # "dbname=eusospb_data user=eusospb password= host=localhost"
    cur = con.cursor()
    return con, cur


def save_csv(df,save_txt_dir,base_file_name, sep='\t'):
    if save_txt_dir:
        csv_path = os.path.join(save_txt_dir, "{}.{}sv".format(base_file_name, 't' if sep=='\t' else 'c'))
        print('SAVING CSV {}'.format(csv_path))
        df.to_csv(csv_path, sep=sep)
        return csv_path
    return None


def get_spb_processing_event_ver2_columns():
    cur.execute('SELECT * FROM spb_processing_event_ver2 LIMIT 1')
    spb_processing_event_ver2_columns = list(map(lambda x: x[0], cur.description))

    return spb_processing_event_ver2_columns    
    

def get_all_bgf05_simu_events__packet_count_by_energy(con):
    all_bgf05_simu_events__packet_count_by_energy_query = '''
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

    return psql.read_sql( all_bgf05_simu_events__packet_count_by_energy_query, con)


def ensure_ext(base_file_name, ext='.png'):
    if not base_file_name.endswith(ext):
        return "{}{}".format(base_file_name, ext)
    return base_file_name


def fig_saving_msg(path):
    print("SAVING FIGURE: {}".format(path))


def print_len(l, label, comment=''):
    print("len({}) = {}{}".format(label, len(l)), '' if not comment else '   // {}'.format(comment))
    

def save_figure(fig, *path_parts):
    path = ensure_ext(os.path.join(*path_args), '.png')
    fig_saving_msg(path)
    fig.savefig(path)

def vis_df_etruth_trueenergy_count_packets(all_bgf05_simu_events__packet_count_by_energy, save_fig_dir, fig_file_name='all_bgf05_simu_events__count_packets_by_energy.png'):
    ax_all_bgf05_simu_events__packet_count_by_energy = all_bgf05_simu_events__packet_count_by_energy.plot(x='etruth_trueenergy', y='count_packets')
    if save_fig_dir is not None:
        save_figure(ax_all_bgf05_simu_events__packet_count_by_energy.get_figure(), save_fig_dir, fig_file_name)
    else:
        plt.show()


def get_count_simu_entries_within_cond(con, cond_selection_rules):
    count_simu_entries_within_cond = psql.read_sql('''
    SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 3 AND {conds} 
    '''.format(conds=cond_selection_rules), con)
    return count_simu_entries_within_cond

# to_csv(path, sep='\t')

def get_simu_entries_within_cond_also_1bgf__only_1bgf_lt_05bgf(con, cond_selection_rules, spb_processing_event_ver2_columns):
    cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)
    q = '''
    SELECT t1.event_id, t1.source_data_type_num, t1.gtu_in_packet, t1.num_gtu, t2.gtu_in_packet AS t2_gtu_in_packet, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full, t1.source_file_trigger FROM spb_processing_event_ver2 AS t1 
    JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    WHERE t1.source_data_type_num = 3 AND t2.source_data_type_num = 5  AND (/*(t1.gtu_in_packet < t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu) OR*/ (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu))   
    AND {conds} 
    ORDER BY t1.event_id ASC
    '''.format(conds=cond_selection_rules_t1_prefixed)
    print(q)
    simu_entries_within_cond_also_1bgf = psql.read_sql(q, con)
    return simu_entries_within_cond_also_1bgf
    

def get_simu_entries_within_cond_also_1bgf(con, cond_selection_rules, spb_processing_event_ver2_columns):
    cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)
    q = '''
    SELECT t1.event_id, t1.source_data_type_num, t1.gtu_in_packet, t1.num_gtu, t2.gtu_in_packet AS t2_gtu_in_packet, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full, t1.source_file_trigger FROM spb_processing_event_ver2 AS t1 
    JOIN spb_processing_event_ver2 AS t2 ON (t1.source_file_acquisition_full = t2.source_file_acquisition_full) 
    WHERE t1.source_data_type_num = 3 AND t2.source_data_type_num = 5  AND ((t1.gtu_in_packet <= t2.gtu_in_packet AND t2.gtu_in_packet - t1.gtu_in_packet < t1.num_gtu - 4) OR (t2.gtu_in_packet < t1.gtu_in_packet AND t1.gtu_in_packet - t2.gtu_in_packet < t2.num_gtu - 4)) 
    AND {conds} 
    ORDER BY t1.event_id ASC
    '''.format(conds=cond_selection_rules_t1_prefixed)
    #print(q)
    simu_entries_within_cond_also_1bgf = psql.read_sql(q, con)
    #print(len(simu_entries_within_cond_also_1bgf))
    print simu_entries_within_cond_also_1bgf
    

def vis_col_num_gtu_hist(simu_entries_within_cond_also_1bgf, save_fig_dir, fig_file_name='simu_entries_within_cond_also_1bgf_num_gtu.png'): 
    fig,ax = plt.subplots(1)
    simu_entries_within_cond_also_1bgf['num_gtu'].hist(ax=ax, bins=25)
    fig.set_size_inches(25,5)
    ax.set_yscale('log')
    
    if save_fig_dir is not None:
        save_figure(fig, save_fig_dir, fig_file_name)
    else:
        plt.show()
    

def get_simu_entries_within_cond_also_1bgf_v2(con):
    q = ''' SELECT  /* t1.event_id, t1.source_data_type_num, t1.gtu_in_packet, t1.num_gtu, t2.gtu_in_packet AS t2_gtu_in_packet, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full, t1.source_file_trigger */
    t1.event_id AS t1_event_id, t2.event_id AS t2_event_id, t1.source_data_type_num AS t1_source_data_type_num, t2.source_data_type_num AS t2_source_data_type_num, t1.global_gtu AS t1_global_gtu, t2.global_gtu AS t2_global_gtu,
    t1.gtu_in_packet AS t1_gtu_in_packet, t2.gtu_in_packet AS t2_gtu_in_packet, t1.num_gtu AS t1_num_gtu, t2.num_gtu AS t2_num_gtu, t1.source_file_acquisition_full AS t1_source_file_acquisition_full, t2.source_file_acquisition_full AS t2_source_file_acquisition_full, 
    t1.source_file_trigger AS t1_source_file_trigger, t2.source_file_trigger AS t2_source_file_trigger, t1.run_timestamp AS t1_run_timestamp, t2.run_timestamp AS t2_run_timestamp 
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
    print(q)
    simu_entries_within_cond_also_1bgf_v2 = psql.read_sql(q, con)
    return simu_entries_within_cond_also_1bgf_v2


def find_multiple_event_id_rows(simu_entries_within_cond_also_1bgf_v2):
    row_idxs = []
    row_event_ids = []
    for i, r in simu_entries_within_cond_also_1bgf_v2.iterrows():
        srch = simu_entries_within_cond_also_1bgf_v2[ simu_entries_within_cond_also_1bgf_v2['t1_event_id'] == r.t1_event_id ]
        if len(srch) > 1:
            row_idxs.append(i)
            row_event_ids.append(r.t1_event_id)
    return row_idxs, list(set(row_event_ids))


def get_count_utah_entries_within_cond(con, cond_selection_rules):
    count_utah_entries_within_cond = psql.read_sql('''
    SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 2 AND {conds} 
    '''.format(conds=cond_selection_rules), con)
    return count_utah_entries_within_cond


def get_count_flight_entries_within_cond(con, cond_selection_rules):
    count_flight_entries_within_cond = psql.read_sql('''
    SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 AND x_y_active_pixels_num > 1750 AND {conds} 
    '''.format(conds=cond_selection_rules), con)
    return count_flight_entries_within_cond


def get_count_flight_entries_within_cond_num_ec(con, cond_selection_rules, num_ec=3):
    get_count_flight_entries_within_cond_num_ec = psql.read_sql('''
    SELECT COUNT(*) FROM spb_processing_event_ver2 WHERE source_data_type_num = 1 AND {conds} 
    '''.format(conds=cond_selection_rules.replace('x_y_active_pixels_num > 1750','x_y_active_pixels_num > {}'.format(256*num_ec))), con)
    return get_count_flight_entries_within_cond_num_ec


def get_cond_bgf05_also_1bgf_simu_events__packet_count_by_energy(con, cond_selection_rules):
    cond_bgf05_simu_events_by_energy_query = '''
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
    print(cond_bgf05_simu_events_by_energy_query)
    cond_bgf05_simu_events_by_energy = psql.read_sql(cond_bgf05_simu_events_by_energy_query, con)
    return cond_bgf05_simu_events_by_energy


def vis_all_cond__bgf05_simu_events_by_energy(all_bgf05_simu_events__packet_count_by_energy, cond_bgf05_simu_events_by_energy, yscale, save_fig_dir, fig_file_name='simu_entries_within_cond_also_1bgf_hist.png'):
    ax_all_bgf05_simu_events__packet_count_by_energy = all_bgf05_simu_events__packet_count_by_energy.plot(x='etruth_trueenergy',y='count_packets',marker='.',linestyle='-', color='blue', label="All packets bgf=0.5 and bgf=1")
    #all_bgf10_simu_events_by_energy.plot(x='etruth_trueenergy',y='count_packets',marker='.',linestyle='-', color='red', ax=ax_all_simu_events_by_energy)
    cond_bgf05_simu_events_by_energy.plot(x='etruth_trueenergy',y='count_packets',marker='.',linestyle='-', color='green', ax=ax_all_bgf05_simu_events__packet_count_by_energy, label="Selected packets bgf=0.5 and bgf=1")
    ax_all_bgf05_simu_events__packet_count_by_energy.set_yscale("log", nonposy='clip')
    #plt.show()
    if save_fig_dir is not None:
        ax_all_bgf05_simu_events__packet_count_by_energy.get_figure().save_fig(os.path.join(save_fig_dir, fig_file_name))
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
    

def merge_all_cond__bgf05_simu_events_by_energy(cond_bgf05_simu_events_by_energy, all_bgf05_simu_events__packet_count_by_energy, x_axis_column='etruth_trueenergy'):
    cond_all_merged_bgf05_simu_events_by_energy = merge_cond_all_dataframes(cond_bgf05_simu_events_by_energy, all_bgf05_simu_events__packet_count_by_energy, merge_on=x_axis_column)
    return cond_all_merged_bgf05_simu_events_by_energy


def calc_yerrs_for_all_cond_merged_bgf05_simu_events_by_energy(cond_all_merged_bgf05_simu_events_by_energy, y_axis_cond_column='count_packets_cond', y_axis_all_column='count_packets_all'):
    n1 = cond_all_merged_bgf05_simu_events_by_energy[y_axis_cond_column]
    n2 = cond_all_merged_bgf05_simu_events_by_energy[y_axis_all_column]

    #frac = n1/n2
    #yerrs = list( np.sqrt( ((1-(frac))/n1) + (1/n2) ) * frac ) # cond_all_merged_bgf05_simu_events_by_energy['count_fraction']
    #yerrs = list( np.sqrt(n1 * (1 - n1/n2))/n2 )
    #return yerrs
    return calc_error_bars(n1, n2)
    

def vis_cond_all_merged_bgf05_simu_events_count_fraction_by_energy(cond_all_merged_bgf05_simu_events_by_energy, save_fig_dir, fig_file_name='cond_all_merged_bgf05_simu_events_count_fraction_by_energy.png')
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(18.5/1.8, 2*10.5/1.8)
    for ax_i, ax in enumerate(axs):
        ax = cond_all_merged_bgf05_simu_events_by_energy.plot(x='etruth_trueenergy', y='count_fraction', yerr=yerrs, marker='.',linestyle='-', ecolor='green', linewidth=1, label='Fraction of all packets', ax=ax)
        ax.set_ylim([0,1.1])
        if ax_i == 1:
            ax.set_xscale('log')
        ax.grid(True)

    #plt.show()
    if save_fig_dir is not None:
        fig.save_fig(os.path.join(save_fig_dir, fig_file_name))
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
        z =  np.polyfit(x, y, i, w=w)
        fits_p.append(np.poly1d(z))
    
    return fits_p


#def fit_points_cond_all_merged_bgf05_simu_events_by_energy(cond_all_merged_bgf05_simu_events_by_energy, 
def fit_points_cond_all_merged(cond_all_merged, 
                           x_axis_column='etruth_trueenergy', y_axis_cond_column='count_packets_cond', y_axis_all_column='count_packets_all'):

    x = cond_all_merged[x_axis_column]
    #y = # cond_all_merged['count_fraction']

    n1 = cond_all_merged[y_axis_cond_column]
    n2 = cond_all_merged[y_axis_all_column]

    frac, yerrs = calc_error_bars(n1, n2)
    fits_p = fit_points_using_yerrs(x, frac, yerrs)
    
    return x, frac, yerrs, fits_p


def vis_fits_count_fraction(x, y, yerrs, fits_p, save_fig_dir, fig_file_name='cond_all_merged_bgf05_simu_events_count_fraction_by_energy_fit.png'):
    
    xv = np.linspace(np.min(x),np.max(x),100)

    fig, eaxs = plt.subplots(2,1)
    fig.set_size_inches(18.5/1.8, 2*10.5/1.8)

    colors = ['pink', 'purple', 'red', 'black', 'yellow', 'royalblue', 'cyan', 'blue']
    line_styles = ['-',':']
    labels = ['{:d}st order poly'. '{:d}nd order poly', '{:d}rd order poly', '{:d}nd order poly']

    for eax_i, eax in enumerate(eaxs):
        eax.errorbar(x,y,yerr=yerrs,ecolor='g',fmt='.',label="Measurement")

        for j, fit_p in enumerate(fits_p):
            if not fit_p:
                continue
            eax.plot(xv, fit_p(xv), (j//len(colors))%len(line_styles),color=j%len(colors), label=(labels[j] if j < len(labels) else labels[-1]).format(j+1) )

        eax.set_ylim([0.3,1.1])
        eax.grid(True)
        eax.set_ylabel('Efficiency')
        eax.set_xlabel('Energy [MeV]')
        if eax_i == 1:
            eax.set_xscale('log')
        eax.legend()
     
    if save_fig_dir is not None:
        fig.save_fig(os.path.join(save_fig_dir, fig_file_name))
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
    return e_avg_vals, e_avg_low, e_avg_up


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
    y = y_cond_vals/y_all_vals
    yerrs = calc_error_bars(y, n1, n2)
    e_avg_vals, e_avg_low, e_avg_up = calc_thin_datapoints_avg(e_vals_fc, e_fc_low, e_fc_up, e_vals_fa, e_fa_low, e_fa_up) 
    return y, yerrs, e_avg_vals, e_avg_low, e_avg_up


def vis_thinned_datapoints(cond_all_merged_bgf05_simu_events_by_energy, 
                           e_vals_fc, y_cond_vals, e_fc_err, e_fc_low, e_fc_up,
                           e_vals_fa, y_all_vals, e_fa_err, e_fa_low, e_fa_up, 
                           save_fig_dir, fig_file_name='comparison_thinned_datapoints.png', 
                           x_axis_column='etruth_trueenergy', y_axis_cond_column='count_packets_cond', y_axis_all_column='count_packets_all'):
    
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

    e_avg_vals, e_avg_low, e_avg_up = calc_thin_datapoints_avg(e_vals_fc, e_fc_low, e_fc_up, e_vals_fa, e_fa_low, e_fa_up) 

    ax.errorbar(e_avg_vals, (y_all_vals+y_cond_vals)/2, color='green', alpha=.2, xerr=[e_avg_low, e_avg_up], # e_fa_err, #yerr=yerrs, , ecolor='green'
            marker='.',linestyle=':', linewidth=1, label='Fraction of all packets')

    # ax.set_yscale('log')
    # ax.set_xscale('log')
    #ax.set_ylim([0,1.1])
    ax.grid(True)
    
    if save_fig_dir is not None:
        fig.save_fig(os.path.join(save_fig_dir, fig_file_name))
    else:
        plt.show()


def get_all_bgf05_simu_events_by_posz(con):
    all_bgf05_simu_events_by_posz_query = '''
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
    all_bgf05_simu_events_by_posz = psql.read_sql( all_bgf05_simu_events_by_posz_query, con)
    #print("len(all_bgf05_simu_events_by_posz)",len(all_bgf05_simu_events_by_posz))
    return all_bgf05_simu_events_by_posz

 
def get_cond_bgf05_simu_events_by_posz(con):
    # IMPORTANT
    cond_bgf05_simu_events_by_posz_query = '''
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
    print(cond_bgf05_simu_events_by_posz_query)

    cond_bgf05_simu_events_by_posz = psql.read_sql(cond_bgf05_simu_events_by_posz_query, con)
    #print("len(cond_bgf05_simu_events_by_posz)", len(cond_bgf05_simu_events_by_posz))
    return cond_bgf05_simu_events_by_posz

 
def get_cond_all_bgf05_simu_events_by_posz_merged(cond_bgf05_simu_events_by_energy, all_bgf05_simu_events__packet_count_by_energy):
    cond_all_bgf05_simu_events_by_posz_merged = merge_cond_all_dataframes(cond_bgf05_simu_events_by_energy, all_bgf05_simu_events__packet_count_by_energy, merge_on='egeometry_pos_z')
    return cond_all_bgf05_simu_events_by_posz_merged


def all_bgf05_simu_events_by_posz_and_energy(con):
    # IMPORTANT
    all_bgf05_simu_events_by_posz_and_energy_query = '''
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

    all_bgf05_simu_events_by_posz_and_energy = psql.read_sql( all_bgf05_simu_events_by_posz_and_energy_query, con)
    #print("len(all_bgf05_simu_events_by_posz_and_energy)", len(all_bgf05_simu_events_by_posz_and_energy))
    return all_bgf05_simu_events_by_posz_and_energy


def get_cond_bgf05_simu_events_by_posz_and_energy_query(con):
    cond_bgf05_simu_events_by_posz_and_energy_query = '''
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
    print(cond_bgf05_simu_events_by_posz_and_energy_query)

    cond_bgf05_simu_events_by_posz_and_energy = psql.read_sql(cond_bgf05_simu_events_by_posz_and_energy_query, con)
    #print("len(cond_bgf05_simu_events_by_posz_and_energy)", len(cond_bgf05_simu_events_by_posz_and_energy))
    return cond_bgf05_simu_events_by_posz_and_energy
  

def get_cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit(save_fig_dir, fig_file_name='comparison_thinned_datapoints.png'):
    uniq_posz = cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona['egeometry_pos_z'].unique()

    uniq_posz_plot_data = [None] * len(uniq_posz)
    #y_posz_vals = [None] * len(uniq_posz)
    #yerrs_posz_vals = [None] * len(uniq_posz) 
    #e_avg_vals_posz_vals = [None] * len(uniq_posz)
    #e_avg_low_posz_vals = [None] * len(uniq_posz)
    #e_avg_up_posz_vals = [None] * len(uniq_posz)
    #fits_p_posz_vals = [None] * len(uniq_posz)

    for posz_i, posz_val in enumerate(uniq_posz):
        
        single_posz_data = cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona[ cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona['egeometry_pos_z']==posz_val ]
        
        y, yerrs, e_avg_vals, e_avg_low, e_avg_up = thin_datapoints_from_dataframe(single_posz_data, x_axis_column='etruth_trueenergy', num_steps=100 if len(single_posz_data) < 10 else 10)
        fits_p = fit_points_using_yerrs(e_avg_vals, y, yerrs)
    
        uniq_posz_plot_data[posz_i] = ('EGeometry.Pos.Z={}'.format(posz_val), y, yerrs, e_avg_vals, e_avg_low, e_avg_up, fits_p)
        
        #y_posz_vals[posz_i] = y
        #yerrs_posz_vals[posz_i] = yerrs
        #e_avg_vals_posz_vals[posz_i] = e_avg_vals
        #e_avg_low_posz_vals[posz_i] = e_avg_low
        #e_avg_up_posz_vals[posz_i] = e_avg_up
        #fits_p_posz_vals[posz_i] = fits_p
        #vis_fits_count_fraction(x, frac, yerrs, fits_p, save_fig_dir, fig_file_name='cond_all_bgf05_simu_events_count_fraction_by_posz_merged.png')
        
    return uniq_posz_plot_data #y_posz_vals, yerrs_posz_vals, e_avg_vals_posz_vals, e_avg_low_posz_vals, e_avg_up_posz_vals, fits_p_posz_vals


def vis_cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit(
        uniq_posz_plot_data
        #y_posz_vals, yerrs_posz_vals, e_avg_vals_posz_vals, e_avg_low_posz_vals, e_avg_up_posz_vals, fits_p_posz_vals,
        save_fig_dir, fig_file_name='cond_all_merged_bgf05_simu_events_by_posz_and_energy_thin_fit.png', num_cols=5, col_width=18.5/1.8, col_height=10.5/1.8):
    
    fig, axs = plt.subplots(np.ceil(len(uniq_posz)/num_cols, num_cols))
    axs_flattened = axs.flatten()

    fig.set_size_inches(np.ceil(len(uniq_posz)/2)*col_width, num_rows)

    colors = ['pink', 'purple', 'red', 'black', 'yellow', 'royalblue', 'cyan', 'blue']
    line_styles = ['-',':']
    labels = ['{:d}st order poly'. '{:d}nd order poly', '{:d}rd order poly', '{:d}nd order poly']
    
    
    for ax_i, (label, y, yerrs, x, e_avg_low, e_avg_up, fits_p) in enumerate(uniq_posz_plot_data):
        eax = axs_flattened[ax_i]
        
        eax.errorbar(x, y, yerr=yerrs, xerr=[e_avg_low, e_avg_up], ecolor='g', fmt='.', label="Measurement")
        eax.set_title("{}".format(label))

        xv = np.linspace(np.min(x),np.max(x),100)
        
        for j, fit_p in enumerate(fits_p):
            if not fit_p:
                continue
            eax.plot(xv, fit_p(xv), (j//len(colors))%len(line_styles),color=j%len(colors), label=(labels[j] if j < len(labels) else labels[-1]).format(j+1) )

        eax.set_ylim([0.3,1.1])
        eax.grid(True)
        eax.set_ylabel('Efficiency')
        eax.set_xlabel('Energy [MeV]')
        eax.legend()
    
    if save_fig_dir is not None:
        fig.save_fig(os.path.join(save_fig_dir, fig_file_name))
    else:
        plt.show()


def select_simu_events_within_cond(con, cond_selection_rules):
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

    con.rollback()
    all_rows_simu_event_within_cond, all_columns_simu_event_within_cond = data_analysis_utils.select_events(con, select_simu_event_within_cond_query_format, [], limit=200000, column_prefix='t1.')
    #print("Selected {} rows".format(len(all_rows)))
    return all_rows_simu_event_within_cond, all_columns_simu_event_within_cond
  
  
def select_simu_event_not_within_cond(con, cond_selection_rules):
    cond_selection_rules_t1_prefixed = re.sub('|'.join(spb_processing_event_ver2_columns),r't1.\g<0>', cond_selection_rules)

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

    con.rollback()
    all_rows_simu_event_not_within_cond, all_columns_simu_event_not_within_cond = data_analysis_utils.select_events(con, select_simu_event_not_within_cond_query_format, [], limit=2000, column_prefix='t1.')
    #print("Selected {} rows".format(len(all_rows)))
    return all_rows_simu_event_not_within_cond, all_columns_simu_event_not_within_cond
  

def group_rows_to_count_packets(all_rows_simu_event_within_cond,all_columns_simu_event_within_cond, groupby1_columns=['etruth_trueenergy','source_file_acquisition_full','packet_id'], groupby2_columns=['etruth_trueenergy']):
    pd_all_rows_simu_event_within_cond = pd.DataFrame(all_rows_simu_event_within_cond, columns=all_columns_simu_event_within_cond)
    #  GROUP BY egeometry_pos_z, etruth_trueenergy, t1.source_file_acquisition_full, t1.packet_id) AS sq 
    #  GROUP BY egeometry_pos_z, etruth_trueenergy ORDER BY egeometry_pos_z, etruth_trueenergy;
    cond_bgf05_simu_events_by_energy_pd = pd_all_rows_simu_event_within_cond.groupby(groupby1_columns).count().groupby(groupby2_columns).count().loc[:,['event_id']].reindex(columns=['event_id']).reset_index().rename(columns={'event_id':'count_packets'}) #['event_id'] #.reindex(columns=['etruth_trueenergy','event_id'])
    return cond_bgf05_simu_events_by_energy_pd
  

def select_flight_events_within_cond(con, cond_selection_rules):
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

    flight_events_within_cond = psql.read_sql(select_flight_events_within_cond_query, con)
    # all_rows_flight_event_within_cond, all_columns_flight_event_within_cond = data_analysis_utils.select_events(con, select_flight_event_within_cond_query_format, [], limit=2000, column_prefix='t1.')

    # con.rollback()
    # all_rows_flight_event_within_cond, all_columns_flight_event_within_cond = data_analysis_utils.select_events(con, select_flight_event_within_cond_query_format, [], limit=2000, column_prefix='t1.')
    # #print("Selected {} rows".format(len(all_rows)))
    return flight_events_within_cond


def rows_generator(iterrows):
    for t in iterrows:
        yield t[1]


def count_num_max_pix_on_pmt_and_ec(flight_events_within_cond, fractions=[0.6, 0.8, 0.9]):
    
    flight_events_num_max_pix_on_pmt = {}
    flight_events_num_max_pix_on_ec = {}

    for i in range(6):
        for j in range(6):
            flight_events_num_max_pix_on_pmt[(i,j)] = {}
            for frac in fractions:
                flight_events_num_max_pix_on_pmt[(i,j)][frac] = np.zeros((len(flight_events_within_cond),2))

    for i in range(3):
        for j in range(3):
            flight_events_num_max_pix_on_ec[(i,j)] = {}
            for frac in fractions:
                flight_events_num_max_pix_on_ec[(i,j)][frac] = np.zeros((len(flight_events_within_cond),2))
                
    for row_i, row in flight_events_within_cond.iterrows():
        if row_i % 1000 == 0:
            sys.stdout.write("{}\n".format(row_i))
            sys.stdout.flush()
        if row['source_file_acquisition_full'].endswith('.npy'):
            acquisition_arr = np.load(row['source_file_acquisition_full'])
            if acquisition_arr.shape[0] != 256:
                raise Exception('Unexpected number of frames in the acqusition file "{}" (#{}  ID {})'.format(
                    row['source_file_acquisition_full'], i, row['event_id']))
            frames_acquisition = acquisition_arr[
                                row['packet_id'] * 128 + row['gtu_in_packet'] - 4:row['packet_id'] * 128 + row['gtu_in_packet'] - 4 + row['num_gtu']]
        elif row['source_file_acquisition_full'].endswith('.root'):
            frames_acquisition = tool.acqconv.get_frames(row['source_file_acquisition_full'],
                                                        row['packet_id'] * 128 + row['gtu_in_packet'] - 4,
                                                        row['packet_id'] * 128 + row['gtu_in_packet'] - 4 + row['num_gtu'], entry_is_gtu_optimization=True)
        else:
            raise Exception('Unexpected source_file_acquisition_full "{}"'.format(row['source_file_acquisition_full']))

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
                
                flight_events_num_max_pix_on_pmt[(pmt_y,pmt_x)][frac][row_i,0] += 1
                flight_events_num_max_pix_on_ec[(ec_y,ec_x)][frac][row_i,0] += 1
                
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


    for k in range(len(flight_events_within_cond)):
        for i in range(3):
            for j in range(3):
                for frac in fractions:
                    for ii in range(3):
                        for jj in range(3):
                            if jj == j and ii == i:
                                continue
                            flight_events_num_max_pix_on_ec[(i,j)][frac][k,1] += flight_events_num_max_pix_on_ec[(ii,jj)][frac][k,0]
    #                 if k == 4:
    #                     print("flight_events_num_max_pix_on_ec[({i},{j})][{frac}][{k},0] = {v}   flight_events_num_max_pix_on_ec[({i},{j})][{frac}][{k},1] = {v1}".format(
    #                         i=i,j=j,frac=frac,k=k,v=flight_events_num_max_pix_on_ec[(i,j)][frac][k,0], v1=flight_events_num_max_pix_on_ec[(i,j)][frac][k,1]))
    #     if k > 5:
    #         break

            
    for k in range(len(flight_events_within_cond)):
        for i in range(6):
            for j in range(6):
                for frac in fractions:
                    for ii in range(6):
                        for jj in range(6):
                            if jj == j and ii == i:
                                continue
                            flight_events_num_max_pix_on_pmt[(i,j)][frac][k,1] += flight_events_num_max_pix_on_pmt[(ii,jj)][frac][k,0]
    #                         if flight_events_num_pix[(ii,jj)][frac][k,0] > 0:
    #                             print('flight_events_num_pix[({i},{j})][{frac}][{k},1] += flight_events_num_pix[({ii},{jj})][{frac}][{k},1] # {v}'.format(
    #                                 i=i,j=j,frac=frac,k=k,ii=ii,jj=jj,v=flight_events_num_pix[(ii,jj)][frac][k,0]))
    #     if k > 5:
    #         break
    return flight_events_num_max_pix_on_pmt, flight_events_num_max_pix_on_ec


    # In[211]:

def extend_df_with_num_max_pix(flight_events_within_cond, flight_events_num_max_pix_on_pmt, flight_events_num_max_pix_on_ec):

    for c in flight_events_within_cond.columns:
        print(c)


    # In[212]:


    #np.where(flight_events_num_pix[(0,0)][.6] > 0)
    #flight_events_num_pix[(0,1)][0.6][:,1] > 0
    flight_events_within_cond_cp = flight_events_within_cond.copy() # this is NOT creating the copy
    k=0
    for frac in fractions:
        for i in range(6):
            for j in range(6):
                col = 'pmt_{:d}_{:d}'.format(i,j) + '_frac{:.1f}'.format(frac).replace('.','') 
                #print(pd.Series(flight_events_num_pix[(i,j)][frac][:,0]))
                #print(col, flight_events_num_pix[(i,j)][frac][k,1] , flight_events_num_pix[(ii,jj)][frac][k,0])
                flight_events_within_cond_cp[col + '_in'] = pd.Series(flight_events_num_max_pix_on_pmt[(i,j)][frac][:,0])
                flight_events_within_cond_cp[col + '_out'] = pd.Series(flight_events_num_max_pix_on_pmt[(i,j)][frac][:,1])
                

    for frac in fractions:
        for i in range(3):
            for j in range(3):            
                col = 'ec_{:d}_{:d}'.format(i,j) + '_frac{:.1f}'.format(frac).replace('.','') 
                flight_events_within_cond_cp[col + '_in'] = pd.Series(flight_events_num_max_pix_on_ec[(i,j)][frac][:,0])
                flight_events_within_cond_cp[col + '_out'] = pd.Series(flight_events_num_max_pix_on_ec[(i,j)][frac][:,1])
                
    print(len(flight_events_within_cond_cp.columns))
    return flight_events_within_cond_cp


def vis_num_gtu_hist(flight_events_within_cond_cp, save_fig_dir, fig_file_name='num_gtu_hist.png'):
    # for k,v in flight_events_within_cond_cp[2000:].iloc[64].iteritems():
    #     if k in ["event_id"] or k.startswith('ec')
    #     print("{}\t{}".format(k,v))
    fig,ax = plt.subplots(1)
    flight_events_within_cond_cp['num_gtu'].hist(ax=ax, bins=30)
    fig.set_size_inches(25,5)
    if save_fig_dir is not None:
        fig.save_fig(os.path.join(save_fig_dir, fig_file_name))
    else:
        plt.show()


def filter_out_top_left_ec(flight_events_within_cond_cp, ec_0_0_frac_lt=0.5, num_gtu_gt=15):
    ec_0_0_frac = flight_events_within_cond_cp['ec_0_0_frac06_in'] / flight_events_within_cond_cp['ec_0_0_frac06_out']
    # filtered_flight_events_within_cond = flight_events_within_cond_cp[ (flight_events_within_cond_cp['ec_0_0_frac06_out'] == 0) ]
    filtered_flight_events_within_cond = flight_events_within_cond_cp[ (flight_events_within_cond_cp['ec_0_0_frac06_out'] != 0) & (ec_0_0_frac < ec_0_0_frac_lt) & (flight_events_within_cond_cp['num_gtu'] > num_gtu_gt) ]
    return filtered_flight_events_within_cond


def main():
    
    args_parser = argparse.ArgumentParser(description='Draw histograms of parameter values')
    args_parser.add_argument('-d','--dbname',default='eusospb_data')
    args_parser.add_argument('-U','--user',default='eusospb')
    args_parser.add_argument('--password')
    args_parser.add_argument('-s','--host',default='localhost')
    args_parser.add_argument('-o','--save-fig-dir',default='/tmp/event_classification_efficiency', help="Directory where figures are saved")
    args_parser.add_argument('-o','--save-txt-dir',default='/tmp/event_classification_efficiency', help="Directory where dataframes are saved")
    args_parser.add_argument('--show-plots',type=str2bool_argparse,default=False,help='If true, plots are only showed in windows')

    args = args_parser.parse_args(argv)

    if not args.password:
        args.password = getpass.getpass()

    if not args.show_plots:
        save_fig_dir = os.path.realpath(args.save_fig_dir)
    else:
        save_fig_dir = None
    
    if not os.path.exist(save_fig_dir):
        os.makedirs(save_fig_dir)
    elif not os.path.isdir(save_fig_dir):
        raise Exception('{} is not directory'.format(save_fig_dir))
    
    con = get_conn(dbname=args['dbname'], user=args['user'], password=)

    # -----------------------------------------------------
    print("COLUMNS")
    # -----------------------------------------------------
    
    spb_processing_event_ver2_columns = get_spb_processing_event_ver2_columns()

    # -----------------------------------------------------
    print("ALL SIMU EVENTS BY ENERGY")
    # -----------------------------------------------------
    
    all_bgf05_simu_events__packet_count_by_energy = get_all_bgf05_simu_events__packet_count_by_energy(con)    
    
    print_len(all_bgf05_simu_events__packet_count_by_energy, 'all_bgf05_simu_events__packet_count_by_energy')
    save_csv(all_bgf05_simu_events__packet_count_by_energy, save_fig_dir, 'all_bgf05_simu_events__packet_count_by_energy')
    
    vis_df_etruth_trueenergy_count_packets(all_bgf05_simu_events__packet_count_by_energy, save_fig_dir, 'all_bgf05_simu_events__count_packets_by_energy')
    
    #vis_col_num_gtu_hist(all_bgf05_simu_events__packet_count_by_energy, 'all_bgf05_simu_events__packet_count_by_energy__num_gtu')
    
    # -----------------------------------------------------
    # COND SELECTION RULES
    # -----------------------------------------------------
    cond_selection_rules = get_selection_rules()
    # -----------------------------------------------------
    
    # -----------------------------------------------------
    print("ALL SIMU EVENTS COUNT WITHIN CONDITIONS")
    # -----------------------------------------------------    
       
    cond_simu_entries_count = get_count_simu_entries_within_cond(con, cond_selection_rules)
    print("cond_simu_entries_count = {}".format(cond_simu_entries_count))
    
    # -----------------------------------------------------
    print("ALL SIMU EVENTS WITHIN CONDITIONS")
    # -----------------------------------------------------   
    
    simu_entries_within_cond_also_1bgf__only_1bgf_lt_05bgf = get_simu_entries_within_cond_also_1bgf__only_1bgf_lt_05bgf(con, cond_selection_rules, spb_processing_event_ver2_columns)
    print_len(simu_entries_within_cond_also_1bgf__only_1bgf_lt_05bgf, 'simu_entries_within_cond_also_1bgf__only_1bgf_lt_05bgf', 'expected to be empty')
    save_csv(simu_entries_within_cond_also_1bgf__only_1bgf_lt_05bgf, save_fig_dir, 'simu_entries_within_cond_also_1bgf__only_1bgf_lt_05bgf')
    
    simu_entries_within_cond_also_1bgf = get_simu_entries_within_cond_also_1bgf(con, cond_selection_rules, spb_processing_event_ver2_columns) 
    print_len(simu_entries_within_cond_also_1bgf, 'simu_entries_within_cond_also_1bgf', 'as many as possible of {}'.format(cond_simu_entries_count))
    save_csv(simu_entries_within_cond_also_1bgf, save_fig_dir, 'simu_entries_within_cond_also_1bgf')
    
    simu_entries_within_cond_also_1bgf_v2 = get_simu_entries_within_cond_also_1bgf_v2(con, cond_selection_rules, spb_processing_event_ver2_columns) 
    print_len(simu_entries_within_cond_also_1bgf_v2, 'simu_entries_within_cond_also_1bgf_v2', 'as many as possible of {} and same as {}'.format(cond_simu_entries_count, len(simu_entries_within_cond_also_1bgf)))
    save_csv(simu_entries_within_cond_also_1bgf_v2, save_fig_dir, 'simu_entries_within_cond_also_1bgf')
    
    vis_col_num_gtu_hist(simu_entries_within_cond_also_1bgf, 'simu_entries_within_cond_also_1bgf__num_gtu')
    
    multiple_event_id_rows_row_idxs, multiple_event_id_rows_row_event_ids = find_multiple_event_id_rows(simu_entries_within_cond_also_1bgf_v2)
    
    print_len(multiple_event_id_rows_row_idxs, 'multiple_event_id_rows_row_idxs')
    
    
    # -----------------------------------------------------
    print("EVENTS COUNT WITHIN CONDITIONS")
    # -----------------------------------------------------   
    count_utah_entries_within_cond = get_count_utah_entries_within_cond(con, cond_selection_rules)
    print("count_utah_entries_within_cond = {}".format(count_utah_entries_within_cond))
    
    count_flight_entries_within_cond = get_count_flight_entries_within_cond(con, cond_selection_rules)
    print("count_flight_entries_within_cond = {}   // 7 EC".format(count_flight_entries_within_cond))
    
    count_flight_entries_within_cond_3ec = get_count_flight_entries_within_cond_num_ec(con, cond_selection_rules)
    print("count_flight_entries_within_cond_3ec = {}   // 3 EC".format(count_flight_entries_within_cond_3ec))
    
    # -----------------------------------------------------
    print("ALL SIMU EVENTS BY ENERGY WITHIN CONDITIONS BY ENERGY")
    # -----------------------------------------------------   
    
    cond_bgf05_also_1bgf_simu_events__packet_count_by_energy = get_cond_bgf05_also_1bgf_simu_events__packet_count_by_energy(con, cond_selection_rules)
    print_len(cond_bgf05_also_1bgf_simu_events__packet_count_by_energy, 'cond_bgf05_also_1bgf_simu_events__packet_count_by_energy','should be similar to {}'.format(len(all_bgf05_simu_events__packet_count_by_energy)))
    
    
    
    # TODO
    
    #vis_df_etruth_trueenergy_count_packets(simu_entries_within_cond_also_1bgf, save_fig_dir, 'simu_entries_within_cond_also_1bgf__count_packets_by_energy')
    #cond_all_merged_bgf05_simu_events_by_energy['count_packets_cond'].iloc[-1]
    # In[146]:
    # In[147]:
    # plt.errorbar(cond_all_merged_bgf05_simu_events_by_energy['etruth_trueenergy'], 
    #              cond_all_merged_bgf05_simu_events_by_energy['count_fraction'], yerr=yerrs, marker='.',linestyle='-', ecolor='green', linewidth=1)
    # plt.show()
    # In[148]:
    # In[149]:
    # # "Rebinned" data points
    # In[150]:
    # In[151]:
    
    y, yerrs, e_avg_vals, e_avg_low, e_avg_up = thin_datapoints_from_dataframe(cond_all_merged_bgf05_simu_events_by_energy, x_axis_column='etruth_trueenergy')
    fits_p = fit_points_using_yerrs(e_avg_vals, y, yerrs)
    vis_fits_count_fraction(e_avg_vals, y, yerrs, fits_p, save_fig_dir, fig_file_name='thinned_cond_all_merged_bgf05_simu_events_count_fraction_by_energy_fit.png')

    # In[154]:


    # simu2npy_pathanme_glob = "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_18000000/namefiles18000000.0A6.19999972557e+12E.txt/simu.2017-07-22-04h32m23s/simu2npy/ev_12_*signals.npy"
    # simu2npy_pathanme = glob.glob(simu2npy_pathanme_glob)
    # ev_frames = np.load(simu2npy_pathanme[0])
    # npy_vis.visualize_frame(np.maximum.reduce(ev_frames))


    # # Efficiency by Pos.Z

    # In[155]:

    # In[156]:
    cond_all_bgf05_simu_events_by_posz_merged = get_cond_all_bgf05_simu_events_by_posz_merged(cond_bgf05_simu_events_by_posz, all_bgf05_simu_events_by_posz)
    # In[157]:
    x, frac, yerrs, fits_p = fit_points_cond_all_merged(cond_all_bgf05_simu_events_by_posz_merged, x_axis_column='egeometry_pos_z')

    vis_fits_count_fraction(x, frac, yerrs, fits_p, save_fig_dir, fig_file_name='cond_all_bgf05_simu_events_count_fraction_by_posz_merged.png')

    # # Efficiency by Pos.Z and Energy

    # In[158]:


    # IMPORTANT
    #all_bgf05_simu_events_by_posz_and_energy_query 

    #cond_bgf05_simu_events_by_posz_and_energy_query


    # In[159]:
    all_bgf05_simu_events_by_posz_and_energy
    # In[160]:
    cond_bgf05_simu_events_by_posz_and_energy
    # In[161]:
    cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona = merge_cond_all_dataframes(cond_df, all_df, merge_on=['etruth_trueenergy','egeometry_pos_z'], fraction_column='count_fraction', count_column='count_packets')

    # In[162]:

    # In[163]:

    #uniq_posz = cond_all_merged_bgf05_simu_events_by_posz_and_energy_nona['egeometry_pos_z'].unique()

    # # Selecting examples of events

    # ## Simu events - starting from the worst

    # In[248]:


    #select_simu_event_within_cond_query_format


    # In[165]:
    data_analysis_utils.visualize_events(all_rows_simu_event_within_cond, all_columns_simu_event_within_cond, 100, additional_printed_columns=['gtu_y_hough__peak_thr2_avg_phi', 'gtu_x_hough__peak_thr2_avg_phi', 'gtu_y_hough__peak_thr3_avg_phi', 'gtu_x_hough__peak_thr3_avg_phi', 
    'trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width', 'trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width', 'trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width',
    'gtu_y_hough__peak_thr1__max_cluster_counts_sum_width', 'gtu_x_hough__peak_thr1__max_cluster_counts_sum_width', 'trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width', 'x_y_hough__peak_thr1__max_cluster_counts_sum_width',
    'trigg_x_y_hough__dbscan_num_clusters_above_thr1', 'trigg_gtu_y_hough__dbscan_num_clusters_above_thr1', 'trigg_gtu_x_hough__dbscan_num_clusters_above_thr1', 
    'gtu_y_hough__dbscan_num_clusters_above_thr1', 'gtu_x_hough__dbscan_num_clusters_above_thr1', 'etruth_trueenergy', 'egeometry_pos_z','etruth_trueenergy', 'egeometry_pos_z'], vis_gtux=False, vis_gtuy=False, subplot_cols=9)


    # In[166]:
    #select_simu_event_not_within_cond_query_format

    # In[167]:
    data_analysis_utils.visualize_events(all_rows_simu_event_not_within_cond, all_columns_simu_event_not_within_cond, 100, additional_printed_columns=['gtu_y_hough__peak_thr2_avg_phi', 'gtu_x_hough__peak_thr2_avg_phi', 'gtu_y_hough__peak_thr3_avg_phi', 'gtu_x_hough__peak_thr3_avg_phi', 
    'trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width', 'trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width', 'trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width',
    'gtu_y_hough__peak_thr1__max_cluster_counts_sum_width', 'gtu_x_hough__peak_thr1__max_cluster_counts_sum_width', 'trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width', 'x_y_hough__peak_thr1__max_cluster_counts_sum_width',
    'trigg_x_y_hough__dbscan_num_clusters_above_thr1', 'trigg_gtu_y_hough__dbscan_num_clusters_above_thr1', 'trigg_gtu_x_hough__dbscan_num_clusters_above_thr1', 
    'gtu_y_hough__dbscan_num_clusters_above_thr1', 'gtu_x_hough__dbscan_num_clusters_above_thr1', 'etruth_trueenergy', 'egeometry_pos_z'], vis_gtux=False, vis_gtuy=False, subplot_cols=9)


    # In[276]:
    cond_bgf05_simu_events_by_energy_pd = group_rows_to_count_packets(all_rows_simu_event_within_cond,all_columns_simu_event_within_cond)
    # In[277]:
    cond_bgf05_simu_events_by_energy_pd_merged_nona = merge_cond_all_dataframes(cond_bgf05_simu_events_by_energy_pd, all_bgf05_simu_events__packet_count_by_energy, merge_on='etruth_trueenergy')
    
    # In[205]:
    flight_events_within_cond = select_flight_events_within_cond(con, cond_selection_rules)


    # In[206]:
    #def rows_generator(iterrows):

    #for c in rows_generator(flight_events_within_cond.iterrows()):
        #print(c['event_id'])
        #break


    # In[207]:


    data_analysis_utils.visualize_events(rows_generator(flight_events_within_cond[0:].iterrows()), flight_events_within_cond.columns, 2, additional_printed_columns=[
    # 'gtu_y_hough__peak_thr2_avg_phi', 'gtu_x_hough__peak_thr2_avg_phi', 'gtu_y_hough__peak_thr3_avg_phi', 'gtu_x_hough__peak_thr3_avg_phi', 
    # 'trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width', 'trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width', 'trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width',
    # 'gtu_y_hough__peak_thr1__max_cluster_counts_sum_width', 'gtu_x_hough__peak_thr1__max_cluster_counts_sum_width', 'trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width', 'x_y_hough__peak_thr1__max_cluster_counts_sum_width',
    # 'trigg_x_y_hough__dbscan_num_clusters_above_thr1', 'trigg_gtu_y_hough__dbscan_num_clusters_above_thr1', 'trigg_gtu_x_hough__dbscan_num_clusters_above_thr1', 
    # 'gtu_y_hough__dbscan_num_clusters_above_thr1', 'gtu_x_hough__dbscan_num_clusters_above_thr1'
    ], vis_gtux=True, vis_gtuy=True, subplot_cols=9, numeric_columns=False)


    # In[208]:


    flight_events_within_cond.columns


    # In[209]:

    flight_events_num_max_pix_on_pmt, flight_events_num_max_pix_on_ec = count_num_max_pix_on_pmt_and_ec(flight_events_within_cond, fractions=[0.6, 0.8, 0.9]

    extend_df_with_num_max_pix(flight_events_within_cond, flight_events_num_max_pix_on_pmt, flight_events_num_max_pix_on_ec)

    # In[239]:
    vis_num_gtu_hist(flight_events_within_cond_cp, save_fig_dir, fig_file_name='num_gtu_hist.png')

    # In[214]:
    filtered_flight_events_within_cond = filter_out_top_left_ec(flight_events_within_cond_cp, ec_0_0_frac_lt=0.5, num_gtu_gt=15)
    # In[229]:
    print(len(filtered_flight_events_within_cond))


    data_analysis_utils.visualize_events(rows_generator(filtered_flight_events_within_cond[0:].iterrows()), flight_events_within_cond_cp.columns, 50, additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out','x_y_neighbourhood_size','gtu_x_neighbourhood_size','gtu_y_neighbourhood_size'], vis_gtux=True, vis_gtuy=True, subplot_cols=9, numeric_columns=False)
    data_analysis_utils.visualize_events(rows_generator(filtered_flight_events_within_cond[50:].iterrows()), flight_events_within_cond_cp.columns, 50, additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], vis_gtux=True, vis_gtuy=True, subplot_cols=9, numeric_columns=False)
    data_analysis_utils.visualize_events(rows_generator(filtered_flight_events_within_cond[100:].iterrows()), flight_events_within_cond_cp.columns, 50, additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], vis_gtux=True, vis_gtuy=True, subplot_cols=9, numeric_columns=False)
    data_analysis_utils.visualize_events(rows_generator(filtered_flight_events_within_cond[150:].iterrows()), flight_events_within_cond_cp.columns, 50, additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], vis_gtux=True, vis_gtuy=True, subplot_cols=9, numeric_columns=False)
    data_analysis_utils.visualize_events(rows_generator(filtered_flight_events_within_cond[200:].iterrows()), flight_events_within_cond_cp.columns, 50, additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], vis_gtux=True, vis_gtuy=True, subplot_cols=9, numeric_columns=False)
    data_analysis_utils.visualize_events(rows_generator(filtered_flight_events_within_cond[250:].iterrows()), flight_events_within_cond_cp.columns, 50, additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], vis_gtux=True, vis_gtuy=True, subplot_cols=9, numeric_columns=False)
    data_analysis_utils.visualize_events(rows_generator(filtered_flight_events_within_cond[300:].iterrows()), flight_events_within_cond_cp.columns, 50, additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], vis_gtux=True, vis_gtuy=True, subplot_cols=9, numeric_columns=False)
    data_analysis_utils.visualize_events(rows_generator(filtered_flight_events_within_cond[350:].iterrows()), flight_events_within_cond_cp.columns, 50, additional_printed_columns=['ec_0_0_frac06_in','ec_0_0_frac06_out'], vis_gtux=True, vis_gtuy=True, subplot_cols=9, numeric_columns=False)


    # In[226]:


    filtered_flight_events_within_cond_2 = flight_events_within_cond_cp[ (flight_events_within_cond_cp['ec_0_0_frac06_out'] != 0) & (ec_0_0_frac < 0.5) & (flight_events_within_cond_cp['num_gtu'] <= 15) & (flight_events_within_cond_cp['num_gtu'] >= 13) ]

    # In[227]:
    len(filtered_flight_events_within_cond_2)


    # In[ ]:


    pd_all_rows_simu_event_within_cond = pd.DataFrame(all_rows_simu_event_within_cond, columns=all_columns_simu_event_within_cond)
    #  GROUP BY egeometry_pos_z, etruth_trueenergy, t1.source_file_acquisition_full, t1.packet_id) AS sq 
    #  GROUP BY egeometry_pos_z, etruth_trueenergy ORDER BY egeometry_pos_z, etruth_trueenergy;
    cond_bgf05_simu_events_by_energy_pd = pd_all_rows_simu_event_within_cond.groupby(['etruth_trueenergy','source_file_acquisition_full','packet_id']).count().groupby(['etruth_trueenergy']).count().loc[:,['event_id']].reindex(columns=['event_id']).reset_index().rename(columns={'event_id':'count_packets'}) #['event_id'] #.reindex(columns=['etruth_trueenergy','event_id'])
    cond_bgf05_simu_events_by_energy_pd
    cond_bgf05_simu_events_by_energy_pd_merged = pd.merge(cond_bgf05_simu_events_by_energy_pd, all_bgf05_simu_events__packet_count_by_energy,
                                        how='outer',
                                        suffixes=['_cond','_all'],
                                        on=['etruth_trueenergy'])
    cond_bgf05_simu_events_by_energy_pd_merged_nona = cond_bgf05_simu_events_by_energy_pd_merged.dropna().copy()
    cond_bgf05_simu_events_by_energy_pd_merged_nona['count_fraction'] = cond_bgf05_simu_events_by_energy_pd_merged_nona['count_packets_cond'] / cond_all_merged_bgf05_simu_events_by_energy['count_packets_all']
    cond_bgf05_simu_events_by_energy_pd_merged_nona

