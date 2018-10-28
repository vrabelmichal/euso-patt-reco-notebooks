
# coding: utf-8

# In[1]:


import sys
import os
import subprocess

app_base_dir = '/home/spbproc/euso-spb-patt-reco-v1'
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

mpl.rcParams['figure.dpi'] = 80

import matplotlib.pyplot as plt

#import ROOT

# import tool.npy_frames_visualization as npy_vis
import tool.acqconv
from data_analysis_utils import *
from data_analysis_utils_dataframes import *
# import supervised_classification as supc



parser = argparse.ArgumentParser(description='Visualizes features')
parser.add_argument('-d','--dbname',default='eusospb_data')
parser.add_argument('-U','--user',default='eusospb')
parser.add_argument('--password')
parser.add_argument('-s','--host',default='localhost')
parser.add_argument('-o','--outdir',default='/home/spbproc/SPBDATA_uncategorized')
parser.add_argument('--source-data-type-num', type=int, default=10001, help="Value of source_data_type_num column")

parser.add_argument('-p', '--plot-type',default='scatter_matrix')

args = parser.parse_args()

if not args.password:
    args.password = getpass.getpass()

# In[2]:




con = con = pg.connect(dbname=args.dbname, user=args.user, host=args.host, password=args.password)
cur = con.cursor()


# In[3]:


simu_signals_all_query = '''
SELECT 
    spb_processing_v3_simu_signal.event.event_id,


    proc1_x_y_hough_peak_thr1.major_line_phi AS proc1_x_y_hough_peak_thr1_major_line_phi, 
    proc1_gtu_x_hough_peak_thr1.major_line_phi AS proc1_gtu_x_hough_peak_thr1_major_line_phi, 
    proc1_gtu_y_hough_peak_thr1.major_line_phi AS proc1_gtu_y_hough_peak_thr1_major_line_phi,

    proc1_x_y_hough_peak_thr1.major_line_rho AS proc1_x_y_hough_peak_thr1_major_line_rho, 
    proc1_gtu_x_hough_peak_thr1.major_line_rho AS proc1_gtu_x_hough_peak_thr1_major_line_rho, 
    proc1_gtu_y_hough_peak_thr1.major_line_rho AS proc1_gtu_y_hough_peak_thr1_major_line_rho,
    
    proc1_x_y_hough_peak_thr2.major_line_phi AS proc1_x_y_hough_peak_thr2_major_line_phi, 
    proc1_gtu_x_hough_peak_thr2.major_line_phi AS proc1_gtu_x_hough_peak_thr2_major_line_phi, 
    proc1_gtu_y_hough_peak_thr2.major_line_phi AS proc1_gtu_y_hough_peak_thr2_major_line_phi,

    proc1_x_y_hough_peak_thr2.major_line_rho AS proc1_x_y_hough_peak_thr2_major_line_rho, 
    proc1_gtu_x_hough_peak_thr2.major_line_rho AS proc1_gtu_x_hough_peak_thr2_major_line_rho, 
    proc1_gtu_y_hough_peak_thr2.major_line_rho AS proc1_gtu_y_hough_peak_thr2_major_line_rho,
    
    proc1_x_y_hough_peak_thr3.major_line_phi AS proc1_x_y_hough_peak_thr3_major_line_phi, 
    proc1_gtu_x_hough_peak_thr3.major_line_phi AS proc1_gtu_x_hough_peak_thr3_major_line_phi, 
    proc1_gtu_y_hough_peak_thr3.major_line_phi AS proc1_gtu_y_hough_peak_thr3_major_line_phi,
    
    proc1_x_y_hough_peak_thr3.major_line_rho AS proc1_x_y_hough_peak_thr3_major_line_rho, 
    proc1_gtu_x_hough_peak_thr3.major_line_rho AS proc1_gtu_x_hough_peak_thr3_major_line_rho, 
    proc1_gtu_y_hough_peak_thr3.major_line_rho AS proc1_gtu_y_hough_peak_thr3_major_line_rho,
    
    proc1_x_y_hough_peak_thr1.line_clusters_max_sum_clu_width AS proc1_x_y_hough_peak_thr1_line_clusters_max_sum_clu_width,
    proc1_x_y_hough_peak_thr1.line_clusters_max_peak_clu_width AS proc1_x_y_hough_peak_thr1_line_clusters_max_peak_clu_width,
    proc1_x_y_hough_peak_thr2.line_clusters_max_sum_clu_width AS proc1_x_y_hough_peak_thr2_line_clusters_max_sum_clu_width,
    proc1_x_y_hough_peak_thr2.line_clusters_max_peak_clu_width AS proc1_x_y_hough_peak_thr2_line_clusters_max_peak_clu_width,
    proc1_x_y_hough_peak_thr3.line_clusters_max_sum_clu_width AS proc1_x_y_hough_peak_thr3_line_clusters_max_sum_clu_width,
    proc1_x_y_hough_peak_thr3.line_clusters_max_peak_clu_width AS proc1_x_y_hough_peak_thr3_line_clusters_max_peak_clu_width,
    
    proc1_gtu_x_hough_peak_thr1.line_clusters_max_sum_clu_width AS proc1_gtu_x_hough_peak_thr1_line_clusters_max_sum_clu_width,
    proc1_gtu_x_hough_peak_thr1.line_clusters_max_peak_clu_width AS proc1_gtu_x_hough_peak_thr1_line_clusters_max_peak_clu_width,
    proc1_gtu_x_hough_peak_thr2.line_clusters_max_sum_clu_width AS proc1_gtu_x_hough_peak_thr2_line_clusters_max_sum_clu_width,
    proc1_gtu_x_hough_peak_thr2.line_clusters_max_peak_clu_width AS proc1_gtu_x_hough_peak_thr2_line_clusters_max_peak_clu_width,
    proc1_gtu_x_hough_peak_thr3.line_clusters_max_sum_clu_width AS proc1_gtu_x_hough_peak_thr3_line_clusters_max_sum_clu_width,
    proc1_gtu_x_hough_peak_thr3.line_clusters_max_peak_clu_width AS proc1_gtu_x_hough_peak_thr3_line_clusters_max_peak_clu_width,
    
    proc1_gtu_y_hough_peak_thr1.line_clusters_max_sum_clu_width AS proc1_gtu_y_hough_peak_thr1_line_clusters_max_sum_clu_width,
    proc1_gtu_y_hough_peak_thr1.line_clusters_max_peak_clu_width AS proc1_gtu_y_hough_peak_thr1_line_clusters_max_peak_clu_width,
    proc1_gtu_y_hough_peak_thr2.line_clusters_max_sum_clu_width AS proc1_gtu_y_hough_peak_thr2_line_clusters_max_sum_clu_width,
    proc1_gtu_y_hough_peak_thr2.line_clusters_max_peak_clu_width AS proc1_gtu_y_hough_peak_thr2_line_clusters_max_peak_clu_width,
    proc1_gtu_y_hough_peak_thr3.line_clusters_max_sum_clu_width AS proc1_gtu_y_hough_peak_thr3_line_clusters_max_sum_clu_width,
    proc1_gtu_y_hough_peak_thr3.line_clusters_max_peak_clu_width AS proc1_gtu_y_hough_peak_thr3_line_clusters_max_peak_clu_width,

    proc1_x_y_hough_peak_thr1.line_clusters_count AS proc1_x_y_hough_peak_thr1_line_clusters_count, 
    proc1_x_y_hough_peak_thr2.line_clusters_count AS proc1_x_y_hough_peak_thr2_line_clusters_count, 
    proc1_x_y_hough_peak_thr3.line_clusters_count AS proc1_x_y_hough_peak_thr3_line_clusters_count, 
    
    proc1_gtu_y_hough_peak_thr1.line_clusters_count AS proc1_gtu_y_hough_peak_thr1_line_clusters_count, 
    proc1_gtu_y_hough_peak_thr2.line_clusters_count AS proc1_gtu_y_hough_peak_thr2_line_clusters_count, 
    proc1_gtu_y_hough_peak_thr3.line_clusters_count AS proc1_gtu_y_hough_peak_thr3_line_clusters_count, 
    
    proc1_gtu_x_hough_peak_thr1.line_clusters_count AS proc1_gtu_x_hough_peak_thr1_line_clusters_count, 
    proc1_gtu_x_hough_peak_thr2.line_clusters_count AS proc1_gtu_x_hough_peak_thr2_line_clusters_count, 
    proc1_gtu_x_hough_peak_thr3.line_clusters_count AS proc1_gtu_x_hough_peak_thr3_line_clusters_count, 
    
    proc1_x_y_clusters.count              AS proc1_x_y_clusters_count, 
    proc1_x_y_clusters.max_sum_clu_width  AS proc1_x_y_clusters_max_sum_clu_width, 
    proc1_x_y_clusters.max_sum_clu_height AS proc1_x_y_clusters_max_sum_clu_height,
    
    proc1_gtu_y_clusters.count              AS proc1_gtu_y_clusters_count, 
    proc1_gtu_y_clusters.max_sum_clu_width  AS proc1_gtu_y_clusters_max_sum_clu_width, 
    proc1_gtu_y_clusters.max_sum_clu_height AS proc1_gtu_y_clusters_max_sum_clu_height ,
    
    proc1_gtu_x_clusters.count              AS proc1_gtu_x_clusters_count, 
    proc1_gtu_x_clusters.max_sum_clu_width  AS proc1_gtu_x_clusters_max_sum_clu_width, 
    proc1_gtu_x_clusters.max_sum_clu_height AS proc1_gtu_x_clusters_max_sum_clu_height
    
    
    
 FROM spb_processing_v3_simu_signal.event 

 JOIN spb_processing_v3_simu_signal.event_proc1_x_y_hough_peak_thr1 AS proc1_x_y_hough_peak_thr1 USING(event_id)
 JOIN spb_processing_v3_simu_signal.event_proc1_gtu_y_hough_peak_thr1 AS proc1_gtu_y_hough_peak_thr1 USING(event_id)
 JOIN spb_processing_v3_simu_signal.event_proc1_gtu_x_hough_peak_thr1 AS proc1_gtu_x_hough_peak_thr1 USING(event_id)

 JOIN spb_processing_v3_simu_signal.event_proc1_x_y_hough_peak_thr2 AS proc1_x_y_hough_peak_thr2 USING(event_id)
 JOIN spb_processing_v3_simu_signal.event_proc1_gtu_y_hough_peak_thr2 AS proc1_gtu_y_hough_peak_thr2 USING(event_id)
 JOIN spb_processing_v3_simu_signal.event_proc1_gtu_x_hough_peak_thr2 AS proc1_gtu_x_hough_peak_thr2 USING(event_id)

 JOIN spb_processing_v3_simu_signal.event_proc1_x_y_hough_peak_thr3 AS proc1_x_y_hough_peak_thr3 USING(event_id)
 JOIN spb_processing_v3_simu_signal.event_proc1_gtu_y_hough_peak_thr3 AS proc1_gtu_y_hough_peak_thr3 USING(event_id)
 JOIN spb_processing_v3_simu_signal.event_proc1_gtu_x_hough_peak_thr3 AS proc1_gtu_x_hough_peak_thr3 USING(event_id)

 JOIN spb_processing_v3_simu_signal.event_proc1_x_y_clusters   AS proc1_x_y_clusters    USING (event_id)
 JOIN spb_processing_v3_simu_signal.event_proc1_gtu_x_clusters AS proc1_gtu_x_clusters  USING (event_id)
 JOIN spb_processing_v3_simu_signal.event_proc1_gtu_y_clusters AS proc1_gtu_y_clusters  USING (event_id)
   
WHERE
    source_data_type_num IN ({})
    
'''.format(args.source_data_type_num)


# In[4]:


simu_signals_all_df = psql.read_sql(simu_signals_all_query, con)


# In[5]:


print("Selected entries count:", len(simu_signals_all_df))


# In[6]:

'''
col_format = 'proc{proci}_{proj_type}_hough_peak_thr{thri}_line_clusters_max_{clu_type}_clu_width'

plt.close('all')

for proj_type in ['x_y','gtu_x', 'gtu_y']:
            
    for clu_type in ['sum','peak']:
        for proci in [1]:
            cmp_l = []
            for thri in [1,2,3]:
                cmp_l.append(col_format.format(proci=proci, proj_type=proj_type, thri=thri, clu_type=clu_type))
            
            print(cmp_l)
            fig,ax = plt.subplots(1)
            ax.hist(simu_signals_all_df[cmp_l].values, 20, histtype='bar', label=cmp_l) 
            ax.legend(prop={'size': 10})
            fig.set_size_inches(8,8)
            plt.show()

    for proci in [1]:
        for thri in [1,2,3]:
            cmp_l = []
            for clu_type in ['sum','peak']:
                cmp_l.append(col_format.format(proci=proci, proj_type=proj_type, thri=thri, clu_type=clu_type))
            
            print(cmp_l)
            fig,ax = plt.subplots(1)
            ax.hist(simu_signals_all_df[cmp_l].values, 20, histtype='bar', label=cmp_l) 
            ax.legend(prop={'size': 10})
            fig.set_size_inches(8,8)
            plt.show()
            


# In[7]:


col_format = 'proc{proci}_{proj_type}_hough_peak_thr{thri}_line_clusters_count'

plt.close('all')

for proj_type in ['x_y','gtu_x', 'gtu_y']:
            
    for proci in [1]:
        cmp_l = []
        for thri in [1,2,3]:
            cmp_l.append(col_format.format(proci=proci, proj_type=proj_type, thri=thri, clu_type=clu_type))

        print(cmp_l)
        fig,ax = plt.subplots(1)
        ax.hist(simu_signals_all_df[cmp_l].values, 20, histtype='bar', label=cmp_l) 
        ax.legend(prop={'size': 10})
        fig.set_size_inches(8,8)
        plt.show()

    for proci in [1]:
        for thri in [1,2,3]:
            cmp_l = []
            cmp_l.append(col_format.format(proci=proci, proj_type=proj_type, thri=thri, clu_type=clu_type))
            
            print(cmp_l)
            fig,ax = plt.subplots(1)
            ax.hist(simu_signals_all_df[cmp_l].values, 20, histtype='bar', label=cmp_l) 
            ax.legend(prop={'size': 10})
            fig.set_size_inches(8,8)
            plt.show()
'''

# In[ ]:

columns = [
    'proc1_gtu_x_hough_peak_thr2_major_line_phi',
    'proc1_gtu_y_hough_peak_thr2_major_line_phi',
    'proc1_x_y_hough_peak_thr1_line_clusters_max_sum_clu_width',
    'proc1_x_y_hough_peak_thr1_line_clusters_max_peak_clu_width',
    'proc1_x_y_hough_peak_thr1_line_clusters_count',
    'proc1_gtu_x_hough_peak_thr1_line_clusters_max_sum_clu_width',
    'proc1_gtu_x_hough_peak_thr1_line_clusters_max_peak_clu_width',
    'proc1_gtu_x_hough_peak_thr1_line_clusters_count',
    'proc1_gtu_y_hough_peak_thr1_line_clusters_max_sum_clu_width',
    'proc1_gtu_y_hough_peak_thr1_line_clusters_max_peak_clu_width',
    'proc1_gtu_y_hough_peak_thr1_line_clusters_count',
    'proc1_x_y_clusters_count',
    'proc1_gtu_y_clusters_count',
    'proc1_gtu_x_clusters_count'
]

if args.plot_type == 'scatter_matrix':
    from pandas.plotting import scatter_matrix
    scatter_matrix(simu_signals_all_df[columns], alpha=0.2, figsize=(80, 80), diagonal='kde')
    #plt.show()
    plt.savefig(os.path.join(args.outdir,'simu_signal_{}_scatter_matrix.png'.format(args.source_data_type_num)))

elif args.plot_type == 'radviz':
    class_base_col_name = 'proc1_x_y_hough_peak_thr1_line_clusters_max_peak_clu_width'
    cut_col_name = class_base_col_name+'_cut'
    simu_signals_all_df[cut_col_name] = pd.cut(simu_signals_all_df[class_base_col_name], bins=9)
    columns.append(cut_col_name)
    
    from pandas.plotting import radviz
    plt.figure(figsize=(60, 60))
    radviz(simu_signals_all_df[columns],'proc1_x_y_hough_peak_thr1_line_clusters_max_peak_clu_width_cut');
    plt.savefig(os.path.join(args.outdir,'simu_signal_{}_radviz.png'.format(args.source_data_type_num)))

print('done')

# In[ ]:




