#!/usr/bin/env python
# coding: utf-8

# # Training machine learning algorithm to detect showers

# The goal of the method is to classify events into two categories - **shower**, **noise**
# 
# This procedure consists of the following parts:
# 1. [Data selection](#Data-selection)
#     - Visible simulated events (simu signal above the background, track recognized in simulated shower signal).
#     - Noisy simulated events (triggered well outside of track injection GTU). However, the background noise is from the flight data.
#     - Flight noise events (triggered well outside of GTU 40).
#     - Flight classified events - classified by the manual classification.
#     
#     
# 2. [Preparation of the testing and training datasets](#Preparation-of-the-testing-and-training-datasets)
#     - Limitation: only subset of all extracted features is used to decrease computational demands of feature selection. However, the subset should be large enough to contain most of the features that are expected to have some property allowing to distinguish between a shower track and noise. This procedure should be selecting around 1000 event features from the database.
#     - All visible simulated events should be included.
#     - Datasets should be balanced (same size of each class).
#     - Noise datased should be constructed by following priority: classified noise, unclassified flight, unclassified simu.
#     
#     
# 3. Conversion
#     
#     
# 4. Training
#     
#     
# 5. [Evaluation of the recognition efficiency](#Recognition-efficiency-RFECV-model)
#     - Accuracy of classification on the whole dataset.
#     - Accuracy (specificity) of classification on the labeled dataset of noise events.
#     - Dependence of sensitivity to true energy, azimuth, and zenith angles.
#     - Dependence of sensitivity to background intensity. *NOT DONE YET (TODO)*.

# ## Imports
# (section not in the report)

# In[1]:


import sys
import os
import subprocess
import re
import numpy as np
import psycopg2 as pg
import pandas as pd
import pandas.io.sql as psql
import getpass
import matplotlib as mpl
import argparse
import glob
import traceback
import hashlib
import math
import collections
import functools

mpl.rcParams['figure.dpi'] = 80
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sp_opt


# In[2]:


app_base_dir = '/home/spbproc/euso-spb-patt-reco-v1'
if app_base_dir not in sys.path:
    sys.path.append(app_base_dir)

import event_processing_v3
import event_processing_v4
import postgresql_v3_event_storage
import dataset_query_functions_v3

import tool.acqconv
from data_analysis_utils import *
from data_analysis_utils_dataframes import *
# import supervised_classification as supc    
from utility_funtions import key_vals2val_keys


# In[3]:


import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.ensemble 
# import sklearn.neural_network
import sklearn.discriminant_analysis
import sklearn.model_selection
import sklearn.metrics
import sklearn.pipeline
from sklearn.externals import joblib


# ## Directories

# In[84]:


data_snippets_dir = 'ver4_machine_learning_convnet_szakcs_w_labeled_flight_20190628_2__v2'
source_data_snippets_dir = 'ver4_machine_learning_w_labeled_flight_20190628_2'
os.makedirs(data_snippets_dir, exist_ok=True)
os.makedirs(os.path.join(data_snippets_dir, 'figures'), exist_ok=True)


# In[5]:


covnet_euso_base_dir = '/mnt/data_wdblue3d1/spbproc/convnet_euso'
covnet_euso_docker_dir = covnet_euso_base_dir
covnet_euso_dockerfile = os.path.join(covnet_euso_docker_dir, 'Dockerfile-gpu')
covnet_euso_src_dir = os.path.join(covnet_euso_base_dir, 'src')


# ## Data selection

# In[6]:


event_processing_cls = event_processing_v4.EventProcessingV4
event_v3_storage_provider_simu = dataset_query_functions_v3.build_event_v3_storage_provider(
    event_storage_provider_config_file=os.path.join(app_base_dir,'config_simu_w_flatmap.ini'), 
    table_names_version='ver4',
    event_storage_class=postgresql_v3_event_storage.PostgreSqlEventV3StorageProvider,
    event_processing_class=event_processing_cls
)

query_functions_simu = dataset_query_functions_v3.Ver3DatasetQueryFunctions(event_v3_storage_provider_simu)


# In[190]:


event_processing_cls = event_processing_v4.EventProcessingV4
event_v3_storage_provider_flight, config_w_flatmap_flight = dataset_query_functions_v3.build_event_v3_storage_provider(
    event_storage_provider_config_file=os.path.join(app_base_dir,'config_w_flatmap.ini'), 
    table_names_version='ver4',
    event_storage_class=postgresql_v3_event_storage.PostgreSqlEventV3StorageProvider,
    event_processing_class=event_processing_cls,
    return_global_config=True
)

query_functions_flight = dataset_query_functions_v3.Ver3DatasetQueryFunctions(event_v3_storage_provider_flight)


# In[194]:


def load_inverse_means_frame_arr(config, calibration_map_path=None, ret_calibration_map_path=False, ret_arr=True):
    import process_acquisition_file
    
    if not calibration_map_path:
        if 'ProcessAcquisitionsParams' in config and 'calibration_map_path' in config['ProcessAcquisitionsParams']:
            calibration_map_path = config['ProcessAcquisitionsParams']['calibration_map_path']
        if 'FeatureExtractionParams' in config and 'calibration_map_path' in config['FeatureExtractionParams']:
            calibration_map_path = config['FeatureExtractionParams']['calibration_map_path']

    if calibration_map_path:
        inverse_means_frame_pathname, inverse_means_frame_arr =             process_acquisition_file.prepare_inverse_means_file(
                calibration_map_path, os.path.dirname(calibration_map_path),
                exist_ok=True, load_if_exists=True)
        if ret_calibration_map_path:
            if ret_arr:
                return calibration_map_path, inverse_means_frame_arr
            return calibration_map_path
        elif ret_arr:
            return inverse_means_frame_arr
    if ret_calibration_map_path and ret_arr:
        return None, None
    elif ret_calibration_map_path or ret_arr:
        return None


# In[195]:


inverse_means_frame_arr = load_inverse_means_frame_arr(config_w_flatmap_flight)


# ### Selected columns
# 
# Unlinke machine learning approach, that would be trained directly on pixels and learn to identify important features as a part of a learning (for example convolutional neural network), this approach depends on a set of preselected features. Its possible advantage is that there is no need to discover identified features and after the feature extraction, the training is faster.
# 
# One of the sources of possible bias in the analysis might be initial selection of features that are analyzed by feature elimination methods.
# 
# For this experiment selected features include:
# - number of triggered pixels (`trg_count_nonzero`),
# - some properties describing the background frames and background frames projection,
# - similarly for all frames of an event
# - informations about line orientations in projections of a shower
# - informations about precision of estimation the orientation of a shower
# - ...

# In[8]:


common_included_columns_re_list = [
    ('^$','source_file_(acquisition|trigger)(_full)?|global_gtu|packet_id|gtu_in_packet|event_id|num_gtu'),
    'trg_((gtu|x)_[yx])_hough_peak_thr1_major_line_phi', 
    ('orig_x_y','count_nonzero')
]


# #### List of columns of simu data tables used for analysis

# In[9]:


common_columns_for_analysis_dict = query_functions_simu.get_columns_for_classification_dict__by_excluding(
    excluded_columns_re_list=('^.+$',),
    default_excluded_columns_re_list=[],
    included_columns_re_list=common_included_columns_re_list
)

print_columns_dict(common_columns_for_analysis_dict)


# In[10]:


common_df_columns = query_functions_simu.get_dataframe_columns_from_dict(common_columns_for_analysis_dict)


# #### List of columns of flight data tables used for analysis

# In[11]:


flight_columns_for_analysis_dict = query_functions_flight.get_columns_for_classification_dict__by_excluding(
    excluded_columns_re_list=('^.+$',),
    default_excluded_columns_re_list=[],
    included_columns_re_list=common_included_columns_re_list
)

# print_columns_dict(flight_columns_for_analysis_dict)


# ### Data selection queries

# #### Simu visible events (base)

# All positive samples for the training are simulated shower tracks with background from the flight data (see notebook ver4_flatmap_visible_events). Events considered as positive samples have to contain track signal (see ver4_test_selection_visualization__simu_signal notebook) and has to be considered as visible (see ver4_flatmap_simu_visible_events notebook). 
# 
# Visibility of the event is decided by a rule that **there should be at least two frames of the event which  contain a signal pixel that is greater or equal to maximum background intensity in the frame**.
# 
# Additionally there is rule that the first trigger of a visible event should be in GTU $42\pm10$.

# In[12]:


# not in the report

current_columns_for_analysis_dict = common_columns_for_analysis_dict

common_select_clause_str, common_tables_list =     query_functions_simu.get_query_clauses__select(current_columns_for_analysis_dict)

simu_where_clauses_str, simu_tables_list =     query_functions_simu.get_query_clauses__where_simu(
        gtu_in_packet_distacne=(40, 10), 
        num_frames_signals_ge_bg__ge=2, num_frames_signals_ge_bg__le=999
    )

joined_select_clause_str = common_select_clause_str + ', ' +     ', '.join(['{{database_schema_name}}.simu_event.{}'.format(attr) for attr in [
        'simu2npy_pathname', 'edetector_numphotons', 'edetector_numcellhits', 'edetector_numfee', 'eptttrigger_fnumtrigg', 
        'etruth_trueenergy', 'etruth_truetheta', 'etruth_truephi', 'egeometry_pos_z',
        'etruth_trueshowermaxpos_x', 'etruth_trueshowermaxpos_y', 'etruth_trueshowermaxpos_z'
    ]]) + ', ' + \
    ', '.join(['{{database_schema_name}}.simu_event_additional.{}'.format(attr) for attr in [
        'num_frames_counts_gt_bg', 'num_frames_signals_gt_bg', 'num_frames_signals_ge_bg'
    ]])

joined_tables_list = common_tables_list + simu_tables_list + [
    ('{database_schema_name}.simu_event_relation','{data_table_name}','event_id'),
    ('{database_schema_name}.simu_event_additional','{database_schema_name}.simu_event_relation','relation_id'),
    ('{database_schema_name}.simu_event','{database_schema_name}.simu_event_relation','simu_event_id'),
]

join_clauses_str =     query_functions_simu.get_query_clauses__join(joined_tables_list)

source_data_type_num = 3001

simu_events_selection_query = query_functions_simu.get_events_selection_query_plain(
    source_data_type_num=source_data_type_num,
    select_additional=joined_select_clause_str, 
    join_additional=join_clauses_str,
    where_additional=simu_where_clauses_str,
    order_by='{data_table_name}.event_id', 
    offset=0, 
    limit=350000,
    base_select='')

# print(simu_events_selection_query)


# In[13]:


simu_df = psql.read_sql(simu_events_selection_query, event_v3_storage_provider_simu.connection)


# In[14]:


simu_df.head()


# #### Simu noise events

# Simu noise events are events that are caused by a trigger well outside of GTU of shower injection into a packet. 
# 
# It is not ideal to use these these events as samples of the dataset because due the way the background of these events is added to the signal. Simply, if there is less packets providing the background than simualated signal tracks then same event might be repeated multiple times in the dataset. 
# Besides repetition of a background packet, background of the simualted event is created by repeating sequence of background frames, thus this might cause multiple events in a same packet. How often this situation happens has not been tested. It is not expected to be very typical.
# 
# Better method of constructing these events would help validity of this analysis.

# In[15]:


# not in the report

current_columns_for_analysis_dict = common_columns_for_analysis_dict

common_select_clause_str, common_tables_list =     query_functions_simu.get_query_clauses__select(current_columns_for_analysis_dict)

# simu_noise_where_clauses_str = ' AND abs(gtu_in_packet-42) >= 20 '

# OPTIMIZATION, ROWS WITH NULL SHOULD BE ALSO ANALYZED  - noise simu df is not used
simu_noise_where_clauses_str = '''
    AND abs(gtu_in_packet-42) >= 20 
    AND {database_schema_name}.event_trg_gtu_y_hough_peak_thr1.major_line_phi IS NOT NULL 
    AND {database_schema_name}.event_trg_gtu_x_hough_peak_thr1.major_line_phi IS NOT NULL 
    AND {database_schema_name}.event_trg_x_y_hough_peak_thr1.major_line_phi IS NOT NULL
'''

joined_select_clause_str = common_select_clause_str + ', ' +     ', '.join(['{{database_schema_name}}.simu_event.{}'.format(attr) for attr in [
        'simu2npy_pathname', 'edetector_numphotons', 'edetector_numcellhits', 'edetector_numfee', 'eptttrigger_fnumtrigg', 
        'etruth_trueenergy', 'etruth_truetheta', 'etruth_truephi', 'egeometry_pos_z'
    ]]) + ', ' + \
    ', '.join(['{{database_schema_name}}.simu_event_additional.{}'.format(attr) for attr in [
        'num_frames_counts_gt_bg', 'num_frames_signals_gt_bg', 'num_frames_signals_ge_bg'
    ]])

joined_tables_list = common_tables_list + simu_tables_list + [
    ('{database_schema_name}.simu_event_relation','{data_table_name}','event_id'),
    ('{database_schema_name}.simu_event_additional','{database_schema_name}.simu_event_relation','relation_id'),
    ('{database_schema_name}.simu_event','{database_schema_name}.simu_event_relation','simu_event_id'),
]

join_clauses_str =     query_functions_simu.get_query_clauses__join(joined_tables_list)

source_data_type_num = 3001

noise_simu_events_selection_query = query_functions_simu.get_events_selection_query_plain(
    source_data_type_num=source_data_type_num,
    select_additional=joined_select_clause_str, 
    join_additional=join_clauses_str,
    where_additional=simu_noise_where_clauses_str,
    order_by='{data_table_name}.event_id', 
    offset=0, 
    limit=350000,
    base_select='')

# print(noise_simu_events_selection_query)


# In[16]:


noise_simu_df = psql.read_sql(noise_simu_events_selection_query, event_v3_storage_provider_simu.connection)


# In[17]:


noise_simu_df.head()


# #### Flight improbable events

# More preferred set of background noise events consists of events that triggered outside of expected range of GTU. Note that these events were triggered in a configuration with lowered thresholds (number selected bin is halved). However, using such events on its own is not sufficient because the actual flight events are those that were triggered in default configuration.

# In[18]:


# not in the report
current_columns_for_analysis_dict = flight_columns_for_analysis_dict

unl_noise_flight_select_clause_str, unl_noise_flight_tables_list =     query_functions_flight.get_query_clauses__select(current_columns_for_analysis_dict)

unl_noise_flight_clauses_str =     query_functions_flight.get_query_clauses__join(unl_noise_flight_tables_list)

unl_noise_source_data_type_num = 1

unl_noise_flight_where_clauses_str = ''' 
    AND abs(gtu_in_packet-42) > 20
    AND {database_schema_name}.event_orig_x_y.count_nonzero > 256*6
''' 

# intentionally removed
#     AND {database_schema_name}.event_trg_gtu_y_hough_peak_thr1.major_line_phi IS NOT NULL 
#     AND {database_schema_name}.event_trg_gtu_x_hough_peak_thr2.major_line_phi IS NOT NULL 
#     AND {database_schema_name}.event_trg_x_y_hough_peak_thr1.major_line_phi IS NOT NULL

unl_noise_flight_events_selection_query =     query_functions_flight.get_events_selection_query_plain(
        source_data_type_num=unl_noise_source_data_type_num,
        select_additional=unl_noise_flight_select_clause_str, 
        join_additional=unl_noise_flight_clauses_str,
        where_additional=unl_noise_flight_where_clauses_str,
        order_by='{data_table_name}.event_id', 
        offset=0, 
        limit=80000,                            # intentionally selecting incomplete subset to save memory !!!!!!!!!!!!!
    #     limit=350000,
        base_select='')

# print(unl_noise_flight_events_selection_query)


# In[19]:


all_unl_noise_flight_df = psql.read_sql(unl_noise_flight_events_selection_query, event_v3_storage_provider_flight.connection)
# flight_df = psql.read_sql(flight_events_selection_query, event_v3_storage_provider_flight.connection)


# In[20]:


all_unl_noise_flight_df.head()


# In[21]:


len(all_unl_noise_flight_df)


# In[22]:


# intentional !!!
unl_noise_flight_df = all_unl_noise_flight_df


# #### Flight labeled events

# Important part of the dataset is set of events that were triggered by the hardware. These events are expected to be the hardest to recognize. Previous classification experiments without this set of events significantly limited usefulness of the method because it classified 60% of the flight events sample as a track (see ver4_test_selection_visualization__simu_20181018 notebook).
# Addition of a relatively small set of these events (around 1500) seems to help significantly (see ver4_machine_learning_flight_classification_tsne_cfg3 notebook).
# 
# The manually classified dataset has been created using web classification tool (script web_manual_classification.py). The tool is available at http://eusospb-data.michalvrabel.sk.

# In[23]:



EVENT_CLASSES = {
    'pixel': 2,
    'top_left_ec': 5,
    'blob': 12,
    'large_blob': 11,
    'short_single_gtu_track': 7,
    'single_gtu_track': 3,
    'noise': 1,
    'cartesian_line': 4,
    'strong_pmt_edge': 9,
    'few_pixels': 6,
    'bg_increased_suddenly': 10,
    'persistent_pixel': 14,
    'noise_unspecified': 0,
    'unspecified': 8,
    'shower': 13,
    '2pix_line': 15,
    'bright_blob': 16,
    'blob_and_pixels': 17,
    'pixel_w_blob_behind': 18,
    'storng_light': 19,
    'sparse_blobs': 20,
    'noise_with_weak_pixel': 21,
    #
    'unclassified': -1
}

INVERSE_EVENT_CLASSES = {v: k for k, v in EVENT_CLASSES.items()}

EVENT_CLASS_NUMBER_UNLABELED = -1
EVENT_CLASS_NUMBER_UNLABELED_NOISE = -2
EVENT_CLASS_LABLELED_NOISE_FLIGHT = -3  # in case of reduced classification

classification_table_name = event_v3_storage_provider_flight.database_schema_name + '.event_manual_classification'
classification_table_cls_column_name_simple = 'class_number'
classification_table_note_column_name_simple = 'note'
classification_table_last_modification_column_name_simple = 'last_modification'
classification_table_cls_column_name = classification_table_name + '.' + classification_table_cls_column_name_simple
classification_table_note_column_name = classification_table_name + '.' + classification_table_note_column_name_simple
classification_table_last_modification_column_name = classification_table_name + '.' + classification_table_last_modification_column_name_simple
classification_df_cls_column_name ='manual_classification_' + classification_table_cls_column_name_simple
classification_df_note_column_name ='manual_classification_' + classification_table_note_column_name_simple


# ##### Labeled filght noise In the database

# In[24]:


# not in the report
current_columns_for_analysis_dict = flight_columns_for_analysis_dict

lbl_noise_flight_select_clause_str, lbl_noise_flight_tables_list =     query_functions_flight.get_query_clauses__select({
        **current_columns_for_analysis_dict,
        classification_table_name: [classification_table_cls_column_name_simple]
    })

lbl_noise_flight_clauses_str = query_functions_flight.get_query_clauses__join(lbl_noise_flight_tables_list)

lbl_noise_source_data_type_num = 1

lbl_noise_flight_where_clauses_str = ''' 
    AND abs(gtu_in_packet-42) <= 20
    AND {{database_schema_name}}.event_orig_x_y.count_nonzero > 256*6
    AND {classification_table_cls_column_name} NOT IN ({event_class_shower}, {event_class_unspecified})
    
'''.format(
    classification_table_cls_column_name=classification_table_cls_column_name,
    classification_table_last_modification_column_name=classification_table_last_modification_column_name,
    event_class_shower=EVENT_CLASSES['shower'],
    event_class_unspecified=EVENT_CLASSES['unspecified']
)

# intentionally removed

# AND {{database_schema_name}}.event_trg_gtu_y_hough_peak_thr1.major_line_phi IS NOT NULL 
# AND {{database_schema_name}}.event_trg_gtu_x_hough_peak_thr2.major_line_phi IS NOT NULL 
# AND {{database_schema_name}}.event_trg_x_y_hough_peak_thr1.major_line_phi IS NOT NULL
# AND {classification_table_last_modification_column_name} < '2019-04-09'

lbl_noise_flight_events_selection_query =     query_functions_flight.get_events_selection_query_plain(
        source_data_type_num=lbl_noise_source_data_type_num,
        select_additional=lbl_noise_flight_select_clause_str, 
        join_additional=lbl_noise_flight_clauses_str,
        where_additional=lbl_noise_flight_where_clauses_str,
        order_by='{data_table_name}.event_id', 
        offset=0, 
        limit=10000,                            # intentionally selecting incomplete subset to save memory !!!!!!!!!!!!!
    #     limit=350000,
        base_select='')

# print(lbl_noise_flight_events_selection_query)


# In[25]:


lbl_noise_flight_db_df = psql.read_sql(lbl_noise_flight_events_selection_query, event_v3_storage_provider_flight.connection)
# lbl_noise_flight_df[classification_df_cls_column_name] 


# In[26]:


len(lbl_noise_flight_db_df)


# In[27]:


for k, v in lbl_noise_flight_db_df.groupby(classification_df_cls_column_name).count()['event_id'].items():
    print('{:<30}\t{:d}'.format(INVERSE_EVENT_CLASSES[k], v))


# In[28]:


lbl_noise_flight_db_df.head()


# ##### Labeled flight noise in the file

# In[29]:


lbl_noise_flight_df = pd.read_csv(os.path.join(source_data_snippets_dir, 'events/labeled_flight_noise.tsv.gz'), sep='\t')


# In[30]:


len(lbl_noise_flight_df)


# In[31]:


for k, v in lbl_noise_flight_df.groupby(classification_df_cls_column_name).count()['event_id'].items():
    print('{:<30}\t{:d}'.format(INVERSE_EVENT_CLASSES[k], v))


# In[32]:


lbl_noise_flight_df.head()


# #### Flight unclassified probable events

# Small subset of flight unclassified events, that were caused by trigger around GTU 42, are selected to be used for basic check of the data reduction capability.

# In[33]:


# not in the report
current_columns_for_analysis_dict = flight_columns_for_analysis_dict

unl_flight_select_clause_str, unl_flight_tables_list =     query_functions_flight.get_query_clauses__select(current_columns_for_analysis_dict)

unl_flight_clauses_str =     query_functions_flight.get_query_clauses__join(unl_flight_tables_list)

unl_flight_source_data_type_num = 1
# intentionally keeping trg conditions, for consistency
unl_flight_where_clauses_str = ''' 
    AND abs(gtu_in_packet-42) < 20
    AND {{database_schema_name}}.event_trg_gtu_y_hough_peak_thr1.major_line_phi IS NOT NULL 
    AND {{database_schema_name}}.event_trg_gtu_x_hough_peak_thr1.major_line_phi IS NOT NULL 
    AND {{database_schema_name}}.event_trg_x_y_hough_peak_thr1.major_line_phi IS NOT NULL
    AND {{database_schema_name}}.event_orig_x_y.count_nonzero > 256*6
    AND NOT EXISTS(
        SELECT {classification_table}.{{data_table_pk}} 
        FROM {classification_table} 
        WHERE {classification_table}.{{data_table_pk}} = {{data_table_name}}.{{data_table_pk}} LIMIT 1
    )
'''.format(
    classification_table=classification_table_name,
)

unl_flight_events_selection_query = query_functions_flight.get_events_selection_query_plain(
    source_data_type_num=unl_flight_source_data_type_num,
    select_additional=unl_flight_select_clause_str, 
    join_additional=unl_flight_clauses_str,
    where_additional=unl_flight_where_clauses_str,
    order_by='{data_table_name}.event_id',  # 'RANDOM()', # it might be skewed
    offset=0, 
    limit=10000,                            # intentionally selecting incomplete subset to save memory !!!!!!!!!!!!!
#     limit=350000,
    base_select='')

# print(unl_flight_events_selection_query)


# In[34]:


unl_flight_df = psql.read_sql(unl_flight_events_selection_query, event_v3_storage_provider_flight.connection)


# In[35]:


unl_flight_df.head()


# #### Classification column for unlabeled filght
# (not in the report)

# In[36]:


unl_flight_df[classification_df_cls_column_name] = EVENT_CLASS_NUMBER_UNLABELED
unl_noise_flight_df[classification_df_cls_column_name] = EVENT_CLASS_NUMBER_UNLABELED_NOISE


# ### Flight datasets in dict
# (not in the report)

# In[37]:


flight_df_dict = {
    'unl_noise_flight_df': unl_noise_flight_df, 
    'lbl_noise_flight_df': lbl_noise_flight_df, 
    'unl_flight_df': unl_flight_df
}


# ### Closing connections
# (not in the report)

# In[38]:


event_v3_storage_provider_simu.connection.close()
event_v3_storage_provider_flight.connection.close()


# ### Combined simulations dataset

# Simu dataframes are combined because they have same columns. 
# Then within this dataset events are classified into four groups based on 
# - **Query classification information** - Primary classification based on the original data selection query - original intention of the data selection.
# - **Simu signal classification information** - Secondary classification is addition of labeled simu signal events. The events are loaded from tables prepared in ver4_test_selection_visualization__simu_signal notebook.
# 
# The groups are the following:
# - **simu noise** - data selected by query intended to select visible events but simu signal is classified as noisy simu data
# - **simu track** - data selected by query intended to select visible events and simu signal is classified as a signal - <br> *these events will be used as positive samples for machine learning algorithms*
# - **noise track** - data selected by query intended to select noise events but simu signal is classified as a shower
# - **noise noise** - data selected by query intended to select noise events and contains simu signal classified as noisy simu data (could be used as a part of negative samples dataset, although it is not ideal)
# - **simu unclassified**, **noise unclassified** - data without any labelling for simu signal data, generaly should consist of short tracks or noisy tracks, in-between easily recognizable tracks and noise.
# - **simu noise underflow**, **simu noise overflow**, **simu track underflow**, **simu track overflow** - data selected by query intended to select visible events but no simu signal is present (ideally should be empty)
# - **noise noise underflow**, **noise noise overflow**, **noise track underflow**, **noise track overflow**   - data selected by query intended to select noise events and no simu signal is present - <br> *these events will be used as negative samples but with a low priority*

# In[39]:


combined_simu_df = pd.concat([simu_df, noise_simu_df])


# In[40]:


# flight_columns_list = list(lbl_noise_flight_df.columns.values)
# combined_flight_df = pd.concat([unl_noise_flight_df[flight_columns_list], lbl_noise_flight_df[flight_columns_list], unl_flight_df[flight_columns_list]])


# In[41]:


print('len(simu_df) =', len(simu_df))
print('len(noise_simu_df) =', len(noise_simu_df))
print('len(combined_simu_df) =', len(combined_simu_df))


# #### $R_{max}$ property of simulated showers

# In[42]:


# 'etruth_trueshowermaxpos_x', 'etruth_trueshowermaxpos_y', 'etruth_trueshowermaxpos_z'
combined_simu_df['calc_etruth_trueshower_rmax'] = np.hypot(combined_simu_df['etruth_trueshowermaxpos_x'], combined_simu_df['etruth_trueshowermaxpos_y'])


# #### Query classification information
# Primary classification based on the original data selection query - original intention of the data selection.

# In[43]:


combined_simu_df['cond_selection_query'] = 'undefined'
combined_simu_df.loc[combined_simu_df['event_id'].isin(simu_df['event_id']), 'cond_selection_query'] = 'simu'
combined_simu_df.loc[combined_simu_df['event_id'].isin(noise_simu_df['event_id']), 'cond_selection_query'] = 'noise'


# In[44]:


if('simu_df' in locals()): del simu_df
if('noise_simu_df' in locals()): del noise_simu_df
# if('unl_noise_flight_df' in locals()): del unl_noise_flight_df
# if('lbl_noise_flight_df' in locals()): del lbl_noise_flight_df
# if('unl_flight_df' in locals()): del unl_flight_df


# #### Simu signal classification information
# Secondary classification is addition of labeled simu signal events.
# The events are loaded from tables prepared in ver4_test_selection_visualization__simu_signal notebook.

# In[45]:


# [['event_id', 'source_file_acquisition', 'global_gtu', 'packet_id', 'gtu_in_packet', 'num_gtu', 'source_file_acquisition_full']]

simu_signal_data_snippets_dir = 'ver4_simu_signal_data_snippets'
simu_signal_visible_tracks_table_path = os.path.join(simu_signal_data_snippets_dir, 'visible_tracks_table.tsv')
simu_signal_noisy_events_table_path = os.path.join(simu_signal_data_snippets_dir, 'noisy_events_table.tsv')

combined_simu_df, unclassified_simu_df, track_simu_df, track_underflow_simu_df, track_overflow_simu_df, noise_simu_df, noise_underflow_simu_df, noise_overflow_simu_df, simu_signal_track_events_df, simu_signal_noisy_events_df =     add_classification_columns(
        combined_simu_df, 
        simu_signal_visible_tracks_table_path, simu_signal_noisy_events_table_path,
        ret_simu_signal=True, ret_under_over_track=True, ret_split_noise=True,
        simu_track_class='track', simu_noise_class='noise',
        simu_track_underflow_class='track_underflow', simu_track_overflow_class='track_overflow',
        simu_noise_underflow_class='noise_underflow', simu_noise_overflow_class='noise_overflow',
        simu_events_file_pathname_dir=data_snippets_dir)


# ##### Combined label - joining query and labeled simu class

# In[46]:


combined_simu_df['cond_selection_combined'] = 'undefined'

for selection_query in ['simu','noise']:
    for selection_simu_signal in ['noise','track']:
        for simu_signal_sync in ['', 'underflow', 'overflow']:
            t_selection_simu_signal = selection_simu_signal
            if len(simu_signal_sync) > 0:
                t_selection_simu_signal += '_' + simu_signal_sync
            combined_simu_df.loc[
                (combined_simu_df['cond_selection_query'] == selection_query ) & 
                (combined_simu_df['cond_selection_simple'] == t_selection_simu_signal), 
                'cond_selection_combined'] = selection_query + '_' + t_selection_simu_signal


# ##### Size of the  subsets

# ###### Simu signal labels

# In[47]:


print('len(simu_signal_track_events_df)', len(simu_signal_track_events_df))
print('len(simu_signal_noisy_events_df)', len(simu_signal_noisy_events_df))
print('-'*50)
print('len(combined_simu_df)           ', len(combined_simu_df))
print('-'*50)
print('len(unclassified_simu_df)       ', len(unclassified_simu_df))
print('len(track_simu_df)              ', len(track_simu_df))
print('len(track_underflow_simu_df)    ', len(track_underflow_simu_df))
print('len(track_overflow_simu_df)     ', len(track_overflow_simu_df))
print('len(noise_simu_df)              ', len(noise_simu_df))
print('len(noise_underflow_simu_df)    ', len(noise_underflow_simu_df))
print('len(noise_overflow_simu_df)     ', len(noise_overflow_simu_df))
print('-'*50)
print('                                   ', 
      len(unclassified_simu_df) + \
      len(track_simu_df) + len(track_underflow_simu_df) + len(track_overflow_simu_df) + \
      len(noise_simu_df) + len(noise_underflow_simu_df) + len(noise_overflow_simu_df)
     )
print('-'*50)
print('len(track_simu_df)/len(combined_simu_df)        = ', len(track_simu_df)/len(combined_simu_df))
print('len(unclassified_simu_df)/len(combined_simu_df) = ', len(unclassified_simu_df)/len(combined_simu_df))
print('len(noise_simu_df)/len(combined_simu_df)        = ', len(noise_simu_df)/len(combined_simu_df))


# ###### Selection query and simu signal labels

# In[48]:


for selection_query in ['simu','noise']:
    for selection_simu_signal in ['noise','track']:
        for simu_signal_sync in ['', 'underflow', 'overflow']:
            t_selection_simu_signal = selection_simu_signal
            if len(simu_signal_sync) > 0:
                t_selection_simu_signal += '_' + simu_signal_sync
            print('{:<30} {}'.format(
                '{} - {}'.format(selection_query, t_selection_simu_signal),
                np.count_nonzero(
                    (combined_simu_df['cond_selection_query'] == selection_query ) & \
                    (combined_simu_df['cond_selection_simple'] == t_selection_simu_signal))
            ))                


# ##### Example of track underflow subset

# In[49]:


track_underflow_simu_df.sort_values('gtu_in_packet', ascending=False).head()


# ##### Example of track overflow subset

# In[50]:


track_overflow_simu_df.sort_values('gtu_in_packet', ascending=True).head()


# ##### Visualization of a few events

# - Noise entries are sorted by number of simu signal pixles in x-y projection in descending order (`orig_x_y_count_nonzero`, sorted from the most potentially track-like),
# - Track entries are sorted by num frames where maximum signal is greater equal maximum background in acsending order (`num_frames_signals_ge_bg`, from the least visible track events). Non-track-like simu signal might not be necessarly incorrectly labeled entries, just a small portion of a track in signal.
# - Track underflow, track overflow should all contain empty simu signal data. Entries are sorted by GTU in packet in ascending or descending order, respectively.

# In[51]:


def vis_simu_signal_default(i, r, visualized_projections, fig, axs_flattened): 
    show_simu_event_row(i, r, 
        npy_pathname_column='simu2npy_signals_pathname', 
        single_proj_width=4, single_proj_height=4,
        print_info=False, warn_if_not_exact_simu=False)

def vis_simu_signal_with_original(i, r, visualized_projections, fig, axs_flattened):
    vis_simu_signal_default(i, r, visualized_projections, fig, axs_flattened)
    show_simu_event_row(i, r, 
        npy_pathname_column='simu2npy_signals_pathname', 
        single_proj_width=4, single_proj_height=4,
        print_info=False, warn_if_not_exact_simu=False,
        simu_gtu_override=(30,50))
'''
for label, events_to_vis_df in [
        ('noise', noise_simu_df.sort_values('orig_x_y_count_nonzero', ascending=False)), 
        ('track', track_simu_df.sort_values('num_frames_signals_ge_bg', ascending=True)), 
        ('track_underflow', track_underflow_simu_df.sort_values('gtu_in_packet', ascending=False)), 
        ('track_overflow', track_overflow_simu_df.sort_values('gtu_in_packet', ascending=True))
]:
    print('{} ({} entries)'.format(label, len(events_to_vis_df)))
    print('-' * 50)
    vis_events_df(
        events_to_vis_df, 
        events_per_figure=3, max_figures=1, vis_gtux=True, vis_gtuy=True, 
        close_after_vis=False, show=True, 
        additional_printed_columns=[
            'num_frames_signals_ge_bg', 'simu2npy_signals_pathname_short', 
            'cond_selection_query', 'cond_selection_simple'],
        by_one=True,
        extension_func=vis_simu_signal_with_original if label == 'track' else vis_simu_signal_default,
        single_proj_width=4, single_proj_height=4
    )
    print('=' * 50)
'''

# ### Number of NaN entries
# Events with NaN values in are currently rejected from showers dataset. 
# However, final decision about rejection is made considering only columns using in ML algorithm.
# Therefore, these numbers are not exactly indicative of the the final number of rejected events - only simu_track and noise_track should be indicative. (TODO requires check)

# #### Number of NaN entries by query and simu signal labels

# In[52]:


print('{:<30} {:<10} {}'.format(' ', 'NaN', 'Others'))
for selection_query in ['simu','noise']:
    for selection_simu_signal in ['noise','track']:
        for simu_signal_sync in ['', 'underflow', 'overflow']:
            t_selection_simu_signal = selection_simu_signal
            if len(simu_signal_sync) > 0:
                t_selection_simu_signal += '_' + simu_signal_sync
            subset_df = combined_simu_df[
                (combined_simu_df['cond_selection_query'] == selection_query ) & 
                (combined_simu_df['cond_selection_simple'] == t_selection_simu_signal)
            ]
            nan_row_count = np.count_nonzero(subset_df.isnull().any(axis=1))
            print('{:<30} {:<10} {}'.format(
                '{} - {}'.format(selection_query, t_selection_simu_signal),
                nan_row_count, len(subset_df) - nan_row_count
            ))


# Flight data were already selected excluding entries with NaN values (actually NULL in PostgreSQL table).

# In[53]:


for subset_label, subset_df in flight_df_dict.items():
    print('{:50}: {:d}'.format(subset_label, np.count_nonzero(subset_df.isnull().any(axis=1))))


# #### NaN columns
# Columns with a NaN value are either data from Hough transform on projections of triggered pixels - issue is a single pixel in a projection, thus it is impossible to determine orientation of a line. This impacts usable size of the dataset.
# Other source of NaN values are additional information calculated for simulated shower - it is number of frames where number of signal pixels satisfies certain condition. The NaN value is present when there are no signal present in an identified event.

# In[54]:


nan_columns = {}

for i, r in combined_simu_df[combined_simu_df.isnull().any(axis=1)].iterrows():
    for col, val in r.iteritems():
        if isinstance(val, numbers_Number) and math.isnan(val):
            if col not in nan_columns:
                nan_columns[col] = 0
            nan_columns[col] += 1

for col, val in nan_columns.items():
    print("{:<120} : {:<d}".format(col, val))

# del nan_columns


# ### Free memory
# (not in the report)

# In[55]:


if 'unclassified_simu_df' in locals(): del unclassified_simu_df
if 'track_simu_df' in locals(): del track_simu_df
if 'noisy_simu_df' in locals(): del noisy_simu_df
if 'simu_signal_track_events_df' in locals(): del simu_signal_track_events_df
if 'simu_signal_noisy_events_df' in locals(): del simu_signal_noisy_events_df


# In[56]:


# unclassified_simu_df, \
# track_simu_df, track_underflow_simu_df, track_overflow_simu_df, \
# noise_simu_df, noise_underflow_simu_df, noise_overflow_simu_df, \
# simu_signal_track_events_df, simu_signal_noisy_events_df


# ## Preparation of the testing and training datasets

# Selected datasets are used in training and testing of a machine learning algorithm. 
# Because of different inital number of noise and shower events, sizes of the datasets need to be balanced. This is done by decreasing a size of a smaller dataset.
# 
# Another potential solution would be to change class weights in the configuration of a machine learning algorithm.

# ### Column names
# (not in the report)

# In[57]:


dataset_condenser_df_columns = list(common_df_columns)
for col in [  
#         'event_id',
#         'source_file_acquisition_full',
        'source_file_trigger_full',
        'source_file_acquisition',
        'source_file_trigger',
        'global_gtu',
#         'packet_id',
#         'gtu_in_packet',
        'orig_x_y_count_nonzero',
        'bg_x_y_count_nonzero',
        'bg_count_nonzero',
        'orig_count_nonzero',
        'bg_size'

]:
    if col in dataset_condenser_df_columns:
        dataset_condenser_df_columns.remove(col)

# IMPORTANT - NaN columns excluded from the analysis
    
for col in nan_columns.keys():
    if col in dataset_condenser_df_columns:
        dataset_condenser_df_columns.remove(col)
    
simu_class_column = 'cond_selection_combined'
flight_class_column = classification_df_cls_column_name
    
dataset_condenser_df_columns_w_event_id = list(dataset_condenser_df_columns)
if 'event_id' not in dataset_condenser_df_columns:
    dataset_condenser_df_columns_w_event_id.append('event_id')
dataset_condenser_df_columns_w_event_id_simu_class = list(dataset_condenser_df_columns_w_event_id) + [simu_class_column]
dataset_condenser_df_columns_w_event_id_flight_class = list(dataset_condenser_df_columns_w_event_id) + [flight_class_column]


# In[58]:


dataset_condenser_df_columns


# ### Showers dataset
# Showers dataset consists of processed simulated showers that belong to the **"simu track"** class and potentially flight events classified as an air shower.
# 
# Another potential source in the future might consist set of laser shots from Utah tests.

# In[59]:


def query_simu_track(df):
    return df.query('cond_selection_combined == "simu_track"')

def query_event_class_shower(df):
    return df.query(
        '{classification_df_cls_column_name} == {event_class_shower}'.format(
            classification_df_cls_column_name=classification_df_cls_column_name,
            event_class_shower=EVENT_CLASSES['shower']
        )
    )


# this function is pointeless
# _flight_class
def get_labeled_shower(columns=dataset_condenser_df_columns_w_event_id):
    #  unsuitable name of the dict item
    #  expected to be empty
    return query_event_class_shower(flight_df_dict['lbl_noise_flight_df'])         [columns]         .dropna()

# _simu_class
def get_simu_shower_track(columns=dataset_condenser_df_columns_w_event_id):
    return query_simu_track(combined_simu_df)         [columns]         .dropna()


# In[60]:


EVENT_CLASS_LABELED_SHOWER_FLIGHT = 2
EVENT_CLASS_SIMU_TRACK = 1


# In[61]:


shower_subset_df_funcs_dict = {
    'lbl_shower_flight_df': get_labeled_shower,
    'combined_simu_df_shower_track': get_simu_shower_track
}

shower_subset_class_numbers_dict = {
    'lbl_shower_flight_df': EVENT_CLASS_LABELED_SHOWER_FLIGHT,
    'combined_simu_df_shower_track': EVENT_CLASS_SIMU_TRACK
}
shower_subset_priority_order = ['lbl_shower_flight_df', 'combined_simu_df_shower_track']

def get_shower_subsets_list(
        df_columns={
            'lbl_shower_flight_df': dataset_condenser_df_columns_w_event_id,
            'combined_simu_df_shower_track': dataset_condenser_df_columns_w_event_id
        }, 
        shower_subset_df_funcs_dict=shower_subset_df_funcs_dict,
        shower_subset_priority_order=shower_subset_priority_order
):
    shower_subsets_list = []
    
    for shower_subset_label in shower_subset_priority_order:
        
        this_df_columns = df_columns[shower_subset_label]             if isinstance(df_columns, dict) else df_columns
        
        shower_subsets_list.append(
            shower_subset_df_funcs_dict[shower_subset_label](this_df_columns)
        )
    
    return shower_subsets_list

shower_subsets_list = get_shower_subsets_list()


# In[62]:


showers_nonan_w_event_id_df = pd.concat(shower_subsets_list)


# Total size of the simualated showers dataset:

# In[63]:


print('len(showers_nonan_w_event_id_df)', len(showers_nonan_w_event_id_df))


# ### Non-showers dataset
# Noise dataset is presently constructed from three subsets, in the follwing priority
# 1. **Classified noise** - *Flight labeled events* excluding classes `shower` and `unspecified`.
# 2. **Unclassified flight** - Dataset of noise of that triggered using configuration with decreased thresholds (bgf=0.5) outside of window of expected cause of the hardware trigger in GTU 40 (Dataset *Flight improbable events* - 20 GTU before or after GTU 42). 
# 3. **Overflow simu** - In principle same as **unclassified flight** but on simu simulation - frames consist of a repeating sequence. The entries should be slightly more different form the **unclassified flight** than **underflow simu**. That's set events should be generally shorter than than the repeated sequence length, on the other hand, **overflow simu** contains some events of containing repetition of the frames sequence (should be verified).
# 3. **Unclassified simu** - In principle same as **unclassified flight** but on simu simulation - **overflow** and **noise noise"** classified events.

# In[64]:


EVENT_CLASS_NUMBER_SIMU_OVERFLOW = 0
EVENT_CLASS_NUMBER_SIMU_NOISE_NOISE = -4
EVENT_CLASS_NUMBER_SIMU_UNDERFLOW = -5


# In[65]:


def query_labeled_flight_noise(df):
    return df.query(
        '{classification_df_cls_column_name} >= {min_class_number:d} ' \
        'and {classification_df_cls_column_name} not in ({event_class_shower}, {event_class_unspecified})'.format(
            classification_df_cls_column_name=classification_df_cls_column_name,
            min_class_number=min(EVENT_CLASSES.values()),
            event_class_shower=EVENT_CLASSES['shower'],
            event_class_unspecified=EVENT_CLASSES['unspecified']
    ))

def query_unlabeled_flight_noise(df):
    return df.query('{classification_df_cls_column_name} == {EVENT_CLASS_NUMBER_UNLABELED_NOISE:d}'.format(
        classification_df_cls_column_name=classification_df_cls_column_name, 
        EVENT_CLASS_NUMBER_UNLABELED_NOISE=EVENT_CLASS_NUMBER_UNLABELED_NOISE,
    ))
    
def query_simu_noise_noise(df):
    return df.query('cond_selection_combined == "noise_noise"')

def query_simu_overflow(df):
    return df[df['cond_selection_simple'].isin(['noise_overflow', 'track_overflow'])]

def query_simu_underflow(df):
    return df[df['cond_selection_simple'].isin(['noise_underflow', 'track_underflow'])]

def concatenate_balanced(df_list):
    min_len = min([len(t_df) for t_df in df_list])
    df_shortened = [(t_df.iloc[np.random.randint(0, len(t_df), min_len)] if len(t_df) > min_len else t_df)                     for t_df in df_list]
    return pd.concat(df_shortened)

def get_labeled_flight_noise(columns=dataset_condenser_df_columns_w_event_id_flight_class):
    return query_labeled_flight_noise(flight_df_dict['lbl_noise_flight_df'])         [columns]         .dropna()

def get_unlabeled_flight_noise(columns=dataset_condenser_df_columns_w_event_id_flight_class):
    return query_unlabeled_flight_noise(flight_df_dict['unl_noise_flight_df'])         [columns]         .dropna()

def get_simu_noise_noise(columns=dataset_condenser_df_columns_w_event_id_simu_class):
    return query_simu_noise_noise(combined_simu_df)         [columns]         .dropna()

def get_simu_overflow(columns=dataset_condenser_df_columns_w_event_id_simu_class):
    return query_simu_overflow(combined_simu_df)         [columns]         .dropna()

def get_simu_underflow(columns=dataset_condenser_df_columns_w_event_id_simu_class):
    return query_simu_underflow(combined_simu_df)         [columns]         .dropna()


# Size of the dataset in progressively extended by non-shower data until it as large as shower data dataset. 
# If required number of events is lower than size of a subset, events are randomly sampled from the subset.

# In[66]:


noise_subset_df_funcs_dict = {
    'lbl_noise_flight_df': get_labeled_flight_noise, 
    'unl_noise_flight_df': get_unlabeled_flight_noise,
    'combined_simu_df_overflow': get_simu_overflow,
    'combined_simu_df_noise_noise': get_simu_noise_noise,
    'combined_simu_df_underflow': get_simu_underflow,
}

noise_subset_class_numbers_dict = {
    'lbl_noise_flight_df': EVENT_CLASS_LABLELED_NOISE_FLIGHT, 
    'unl_noise_flight_df': EVENT_CLASS_NUMBER_UNLABELED_NOISE, 
    'combined_simu_df_overflow': EVENT_CLASS_NUMBER_SIMU_OVERFLOW,
    'combined_simu_df_noise_noise': EVENT_CLASS_NUMBER_SIMU_NOISE_NOISE,
    'combined_simu_df_underflow': EVENT_CLASS_NUMBER_SIMU_UNDERFLOW
}

noise_subset_priority_order = [
    'lbl_noise_flight_df', 'unl_noise_flight_df', 
    #'combined_simu_df_overflow', 'combined_simu_df_noise_noise', 'combined_simu_df_underflow'
]

np.random.seed(123)

# -----------------------------------------------------------

def get_non_shower_subsets_list(
        df_columns={
            'lbl_noise_flight_df': dataset_condenser_df_columns_w_event_id,  #_flight_class,
            'unl_noise_flight_df': dataset_condenser_df_columns_w_event_id,  #_flight_class,
            'combined_simu_df_overflow': dataset_condenser_df_columns_w_event_id,  #_simu_class,
            'combined_simu_df_noise_noise': dataset_condenser_df_columns_w_event_id,  #_simu_class,
            'combined_simu_df_underflow': dataset_condenser_df_columns_w_event_id,  #_simu_class,
        }, 
        noise_subset_df_funcs_dict=noise_subset_df_funcs_dict,
        noise_subset_priority_order=noise_subset_priority_order,
        len_limit=len(showers_nonan_w_event_id_df)
):

    non_shower_subsets_list = []
    non_shower_subsets_tot_len = 0
    for noise_subset_label in noise_subset_priority_order:
        get_non_shower_events_func = noise_subset_df_funcs_dict[noise_subset_label]
        
        this_df_columns = df_columns[noise_subset_label]             if isinstance(df_columns, dict) else df_columns
        
        non_shower_subset_df = get_non_shower_events_func(this_df_columns)
        new_len = len(non_shower_subset_df) + non_shower_subsets_tot_len

        print('Current subset size: {:<7} ; Added {:<30} subset size: {:<7} ; '               'Potentional new dataset size: {:<7} ; Required size: {:<7}'.format(
            non_shower_subsets_tot_len, noise_subset_label, len(non_shower_subset_df),
            new_len, len_limit
        ))

        if new_len > len_limit:
            # !!!! THIS IS A BUG, sampling is incorrect
            non_shower_subset_df =                 non_shower_subset_df.iloc[
                    np.random.randint(0, len(non_shower_subset_df), 
                                      len_limit - non_shower_subsets_tot_len)
            ]
            # corrected solution
            # np.random.choice(len(non_shower_subset_df), len_limit - non_shower_subsets_tot_len, replace=False)

        non_shower_subsets_list.append(non_shower_subset_df)
        non_shower_subsets_tot_len += len(non_shower_subset_df)

        if new_len >= len_limit:
            break
            
    return non_shower_subsets_list

# -------------------------------------------------------------
            
non_shower_subsets_list = get_non_shower_subsets_list()


# In[67]:


non_showers_nonan_w_event_id_df = pd.concat(non_shower_subsets_list)


# Total number of noise subset required:

# In[68]:


len(non_shower_subsets_list)


# Concatenated noise subsets total size:

# In[69]:


print(len(non_showers_nonan_w_event_id_df))


# ### Concatenated arrays (np.ndarray)
# (not in the report)

# Transformation of multiple `pandas.DataFrame` objects into concatenated `numpy.ndarray`. 
# Following arrays are created:
# - `packets` - training data for an algorithm - packet data - source_file, packet_id, gtu_in_packet
# - `y` - training data for an algorithm - labels
# - `event_id` - event id of the data in the dataset - important after `test_train_split()`, used to associate predictions with the original events
# - `source_class` - source class of the data in the dataset - important after `test_train_split()`, used to associate predictions with the original events, especially to be able to expres accuracy of predictions for a specific source class of data - e.g. label flight noise events

# In[173]:


learning_data_dict = {
    'test': {},
    'train': {},
    'all': pd.concat([
        showers_nonan_w_event_id_df[dataset_condenser_df_columns], 
        non_showers_nonan_w_event_id_df[dataset_condenser_df_columns]
    ])
}
learning_data_dict['all']['target_class'] = np.concatenate([
    np.ones(len(showers_nonan_w_event_id_df)), 
    np.zeros(len(non_showers_nonan_w_event_id_df))
])      
learning_data_dict['all']['source_class'] = np.concatenate([
    *[np.ones(len(shower_subset_df)) * shower_subset_class_numbers_dict[shower_subset_label] \
      for shower_subset_df, shower_subset_label in zip(shower_subsets_list, shower_subset_priority_order)],
    *[np.ones(len(non_shower_subset_df)) * noise_subset_class_numbers_dict[noise_subset_label] \
      for non_shower_subset_df, noise_subset_label in zip(non_shower_subsets_list, noise_subset_priority_order)]
])


# In[174]:


def calc_learning_data_weights(learning_data__y, learning_data__source_class, print_info=True):
    
    learning_data__weights = np.ones_like(learning_data__y)
    
    uniq_noise_source_classes = np.unique(learning_data__source_class[learning_data__y != 1])

    num_shower_events = np.count_nonzero(learning_data__y == 1)
    num_noise_events = len(learning_data__y) - num_shower_events

    k = num_shower_events / (len(uniq_noise_source_classes) * num_noise_events)

    for noise_source_class in uniq_noise_source_classes:
        noise_source_class_mask = learning_data__source_class == noise_source_class
        
        num_noise_source_class = np.count_nonzero(noise_source_class_mask)

        w =  k * num_noise_events / num_noise_source_class 

        learning_data__weights[noise_source_class_mask] = w
        
        if print_info:
            print('{}: w={:.4f}, k={:.4f}, num_source_class={:<8.1f}, norm_num_source_class={:<8.1f}, 1/frac_noise={:.4f}, frac_noise={:.4f}, '.format(
                noise_source_class, w, k, 
                num_noise_source_class, 
                w*num_noise_source_class , 
                num_noise_events / num_noise_source_class, 
                num_noise_source_class / num_noise_events
            ))
    
    if print_info:
        s = np.sum(learning_data__weights[learning_data__y == 0])
        print('sum weights = {:.4f} , per class = {:.4f}'.format(s, s/len(uniq_noise_source_classes)))
    
    return learning_data__weights
        
learning_data_dict['all']['weights'] =     calc_learning_data_weights(learning_data_dict['all']['target_class'], learning_data_dict['all']['source_class'])


# ### Labeled concatenated dataset (pd.DataFrame)
# (not in the report)

# In[175]:


learning_data_dict['all']['source_class'].unique()


# ### Train-test split
# 
# The data are split into training and testing subsets in **60:40** ratio. The data are shuffled before splitting, thus there should not be a significat difference in ratios of source clases of the data within the testing and training sets.

# In[179]:


learning_data_dict['train'], learning_data_dict['test'] =     (pd.DataFrame(_df) for _df in sklearn.model_selection.train_test_split(
        learning_data_dict['all'],
        test_size=.4, 
        random_state=123, 
        shuffle=True
    ))


# In[180]:


for subset_name in ['train', 'test']:
    print('Subset:', subset_name)
    learning_data_dict[subset_name]['weights'] = calc_learning_data_weights(
        learning_data_dict[subset_name]['target_class'], learning_data_dict[subset_name]['source_class'], print_info=True)


# #### Number of entries in training and testing datasets

# In[185]:


for subset_name, _df in learning_data_dict.items():
    print('{:>5}: {}'.format(subset_name, len(_df)))


# #### Number of entries in training and testing datasets by a class (shower, non-shower)

# In[186]:


print('Num. non-shower entries in train', np.count_nonzero(learning_data_dict['train']['target_class'] == 0))
print('Num. shower entries in train    ', np.count_nonzero(learning_data_dict['train']['target_class'] == 1))
print('Num. non-shower entries in test ', np.count_nonzero(learning_data_dict['test']['target_class'] == 0))
print('Num. shower entries in test     ', np.count_nonzero(learning_data_dict['test']['target_class'] == 1))


# In[187]:

learning_data__simu_shower_track_mask_arr_all = \
    learning_data_dict['all']['source_class'] == shower_subset_class_numbers_dict['combined_simu_df_shower_track']
learning_data__simu_shower_track_mask_arr_test = \
    learning_data_dict['test']['source_class'] == shower_subset_class_numbers_dict['combined_simu_df_shower_track']

learning_data__lbl_noise_flight_mask_arr_all = \
    learning_data_dict['all']['source_class'] == noise_subset_class_numbers_dict['lbl_noise_flight_df']
learning_data__lbl_noise_flight_mask_arr_train = \
    learning_data_dict['train']['source_class'] == noise_subset_class_numbers_dict['lbl_noise_flight_df']
learning_data__lbl_noise_flight_mask_arr_test = \
    learning_data_dict['test']['source_class'] == noise_subset_class_numbers_dict['lbl_noise_flight_df']

# #### Number of entries in training and testing datasets considering only labeled noise

# In[188]:


print('lbl_noise_flight_df in train', np.count_nonzero(learning_data__lbl_noise_flight_mask_arr_train))
print('lbl_noise_flight_df in test ', np.count_nonzero(learning_data__lbl_noise_flight_mask_arr_test))


# ### Data dump

# In[306]:


def get_first_packet_frame_num__ranged(
        r, c=None, packet_id_column='packet_id', gtu_in_packet_column='gtu_in_packet', packet_size=128, gtu_in_packet_offset=-4,
        int_func=int, max_func=max
):
    _c_packet_id_column = c[packet_id_column] if c is not None else packet_id_column
    _c_gtu_in_packet_column = c[gtu_in_packet_column] if c is not None else gtu_in_packet_column
    gtu_packet_offset = int_func(r[_c_packet_id_column]) * packet_size
    gtu_offset = gtu_packet_offset + int_func(r[_c_gtu_in_packet_column]) + gtu_in_packet_offset
    return max_func((gtu_offset, gtu_packet_offset, np.zeros_like(gtu_offset)))

def get_last_packet_frame_num_high__ranged_w_overwrite(
        r, c=None, packet_id_column='packet_id', gtu_in_packet_column='gtu_in_packet', num_gtu_column='num_gtu', num_gtu_overwrite=None,
        packet_size=128, gtu_in_packet_offset=-4,
        int_func=int, min_func=max
):
    _c_packet_id_column = c[packet_id_column] if c is not None else packet_id_column
    _c_gtu_in_packet_column = c[gtu_in_packet_column] if c is not None else gtu_in_packet_column
    _c_num_gtu_column = c[num_gtu_column] if c is not None else num_gtu_column
    gtu_offset = int_func(r[_c_packet_id_column]) * packet_size
    num_gtu = int_func(r[_c_num_gtu_column]) if num_gtu_overwrite is None else num_gtu_overwrite
    last_frame_num_high = gtu_offset + int_func(r[_c_gtu_in_packet_column]) + gtu_in_packet_offset + num_gtu
    return min_func((last_frame_num_high, gtu_offset + packet_size))

def get_frame_array(
        all_rows, columns=None,
        ret_xy=True, ret_gtux=False, ret_gtuy=False, numeric_columns=False,
        event_id_column='event_id', packet_id_column='packet_id', gtu_in_packet_column='gtu_in_packet',
        num_gtu_column='num_gtu',
        source_file_acquisition_full_column='source_file_acquisition_full',
        inverse_means_arr=None,
        get_first_frame_num_func=get_first_packet_frame_num__ranged,
        get_last_frame_num_high_func=get_last_packet_frame_num_high__ranged_w_overwrite,
        print_info=True, allow_not_exists=False, status_update_every=100,
        x_y_num_gtu_overwrite=None, gtu_x_num_gtu_overwrite=None, gtu_y_num_gtu_overwrite=None
):
    import pandas as pd
    import datetime

    if numeric_columns and columns is None:
        raise RuntimeError('if numeric_columns is True, columns cannot be None')
    elif columns is None:
        columns = all_rows.keys()

    if numeric_columns:
        c = dict(zip(columns, [r[0] for r in enumerate(columns)]))
    else:
        c = dict(zip(columns, columns))  # pointless, could be rewritten into function

    visualized_projections_xy = []
    visualized_projections_gtux = []
    visualized_projections_gtuy = []

    if isinstance(all_rows, pd.DataFrame):
        _all_rows = all_rows.iterrows()
    else:
        _all_rows = enumerate(all_rows)

    num_npy = 0
    num_root = 0
    
    if print_info:
        t0 = datetime.datetime.now()
        print('Extracting frames start: {:%Y:%m:%d %H:%M:%S}'.format(t0))
    
    max_frames_overwrite =         [_n for _n in (x_y_num_gtu_overwrite, gtu_x_num_gtu_overwrite, gtu_y_num_gtu_overwrite) if _n is not None]
    if None in max_frames_overwrite and (x_y_num_gtu_overwrite or gtu_x_num_gtu_overwrite or gtu_y_num_gtu_overwrite):
        max_frames_overwrite = 128
        
    max_frames_overwrite = None if not max_frames_overwrite else max(max_frames_overwrite)
    
    for j, (i, r) in enumerate(_all_rows):
        
        if print_info:
            if j % status_update_every == 0 and j > 0:
                print('{:>5} / {:<5} ({:>5} npy, {:>5} root, {} since start)'.format(
                    j+1, len(all_rows), num_npy, num_root, datetime.datetime.now() - t0))
        
        sfa_str = r[c[source_file_acquisition_full_column]]
        if sfa_str.endswith('.npy'):
            if not os.path.exists(sfa_str):
                msg_str = "Npy file does not exist: {}".format(sfa_str)
                if allow_not_exists:
                    if print_info:
                        print(msg_str, file=sys.stderr)
                    continue
                else:
                    raise RuntimeError(msg_str)
            acquisition_arr = np.load(sfa_str)
            if acquisition_arr.shape[0] != 256:
                raise Exception('Unexpected number of frames in the acquisition file "{}" (#{}  ID {})'.format(
                    sfa_str, i, r[c[event_id_column]]))
            frames_acquisition = acquisition_arr[
                                 get_first_frame_num_func(r, c, packet_id_column, gtu_in_packet_column):
                                 get_last_frame_num_high_func(r, c, packet_id_column, gtu_in_packet_column,
                                                              num_gtu_column, num_gtu_overwrite=max_frames_overwrite)]
            num_npy += 1
        elif sfa_str.endswith('.root'):
            try:
                frames_acquisition = tool.acqconv.get_frames(
                    sfa_str,
                    get_first_frame_num_func(r, c, packet_id_column, gtu_in_packet_column),
                    get_last_frame_num_high_func(r, c, packet_id_column, gtu_in_packet_column, num_gtu_column, 
                                                 num_gtu_overwrite=max_frames_overwrite) - 1,
                    # CHANGED ON 2019/02/25
                    entry_is_gtu_optimization=True)
            except Exception as e:
                if print_info:
                    print('{:>5} / {:<5} ({:<5} npy, {:<5} root, {} since start)'.format(
                        j+1, len(all_rows), num_npy, num_root, datetime.datetime.now() - t0))
                    sys.stdout.flush()
                raise e
            num_root += 1
        else:
            raise Exception('Unexpected source_file_acquisition_full "{}"'.format(sfa_str))

        if inverse_means_arr is not None:
            if callable(inverse_means_arr):
                _inverse_means_arr = inverse_means_arr(
                    r, c,
                    event_id_column=event_id_column,
                    packet_id_column=packet_id_column,
                    gtu_in_packet_column=gtu_in_packet_column,
                    num_gtu_column=num_gtu_column,
                    source_file_acquisition_full_column=source_file_acquisition_full_column
                )
            else:
                _inverse_means_arr = inverse_means_arr

            if _inverse_means_arr is not None:
                frames_acquisition = frames_acquisition * _inverse_means_arr

        if ret_xy:
            _frames_acquisition = frames_acquisition[0:x_y_num_gtu_overwrite] if x_y_num_gtu_overwrite is not None else frames_acquisition
            ev_integrated = np.max(_frames_acquisition, axis=0)
            visualized_projections_xy.append(ev_integrated)
        if ret_gtuy:
            _frames_acquisition = frames_acquisition[0:gtu_y_num_gtu_overwrite] if gtu_y_num_gtu_overwrite is not None else frames_acquisition
            max_integrated_gtu_y = np.transpose(np.max(_frames_acquisition, axis=2))
            visualized_projections_gtuy.append(max_integrated_gtu_y)
        if ret_gtux:
            _frames_acquisition = frames_acquisition[0:gtu_x_num_gtu_overwrite] if gtu_x_num_gtu_overwrite is not None else frames_acquisition
            max_integrated_gtu_x = np.transpose(np.max(_frames_acquisition, axis=1))
            visualized_projections_gtux.append(max_integrated_gtu_x)

    ret = []
    if ret_xy:
        ret.append(visualized_projections_xy)
    if ret_gtuy:
        ret.append(visualized_projections_gtuy)
    if ret_gtux:
        ret.append(visualized_projections_gtux)

    return tuple(ret)


# In[349]:


a = np.array([1, 2, 40, 5])
b = np.array([1, 20, 4, 5])
c = np.array([10, 2, 4, 5])
np.max((a,b,c), axis=0)


# In[455]:


def get_first_packet_frame_num(
        r, packet_id_column='packet_id', gtu_in_packet_column='gtu_in_packet', packet_size=128, gtu_in_packet_offset=-4,
        int_func=int, max_func=max
):
    gtu_packet_offset = int_func(r[packet_id_column]) * packet_size
    total_gtu_offset = gtu_packet_offset + int_func(r[gtu_in_packet_column]) + gtu_in_packet_offset
    return max_func((total_gtu_offset, gtu_packet_offset))

def get_last_packet_frame_num_high(
        r, packet_id_column='packet_id', gtu_in_packet_column='gtu_in_packet', num_gtu_column='num_gtu', num_gtu_overwrite=None,
        packet_size=128, gtu_in_packet_offset=-4,
        int_func=int, max_func=max, min_func=min
):
    next_gtu_packet_offset = (int_func(r[packet_id_column]) + 1) * packet_size

    num_gtu = int_func(r[num_gtu_column]) if num_gtu_overwrite is None else num_gtu_overwrite
    last_frame_num_high = num_gtu +         get_first_packet_frame_num(r, packet_id_column=packet_id_column, gtu_in_packet_column=gtu_in_packet_column, 
            packet_size=packet_size, gtu_in_packet_offset=gtu_in_packet_offset,
            int_func=int_func, max_func=max_func)
        
    return min_func((last_frame_num_high, next_gtu_packet_offset))

def get_first_packet_frame_num_in_df(
        r, packet_id_column='packet_id', gtu_in_packet_column='gtu_in_packet', 
        packet_size=128, gtu_in_packet_offset=-4,
        int_func=lambda x:x, max_func=lambda x: np.max(x, axis=0)
):
    return get_first_packet_frame_num(
        r, packet_id_column=packet_id_column, gtu_in_packet_column=gtu_in_packet_column, 
        packet_size=packet_size, gtu_in_packet_offset=gtu_in_packet_offset,
        int_func=int_func, max_func=max_func
    )

def get_last_packet_frame_num_high_in_df(
        r, packet_id_column='packet_id', gtu_in_packet_column='gtu_in_packet', num_gtu_column='num_gtu', num_gtu_overwrite=None,
        packet_size=128, gtu_in_packet_offset=-4,
        int_func=lambda x:x, max_func=lambda x: np.max(x, axis=0), min_func=lambda x: np.min(x, axis=0)
):
    return get_last_packet_frame_num_high(
        r, packet_id_column=packet_id_column, gtu_in_packet_column=gtu_in_packet_column, num_gtu_column=num_gtu_column, 
        num_gtu_overwrite=num_gtu_overwrite,
        packet_size=packet_size, gtu_in_packet_offset=gtu_in_packet_offset,
        int_func=int_func, max_func=max_func, min_func=min_func
    )

def make_frame_sequence_projection(proj_label, frames_sequence):
    if proj_label == 'x_y':
        proj_image = np.max(frames_sequence, axis=0)
    elif proj_label == 'gtu_y':
        proj_image = np.transpose(np.max(frames_sequence, axis=2))
    elif proj_label == 'gtu_x':
        proj_image = np.transpose(np.max(frames_sequence, axis=1))
    return proj_image

def get_frame_array(
        all_rows, transform_func=make_frame_sequence_projection,
        return_x_y=True, return_gtu_x=False, return_gtu_y=False, return_offsets=False,
        event_id_column='event_id', packet_id_column='packet_id', gtu_in_packet_column='gtu_in_packet', num_gtu_column='num_gtu',
        source_file_acquisition_full_column='source_file_acquisition_full',
        inverse_means_arr=None,
        get_first_frame_num_func=get_first_packet_frame_num,
        get_last_frame_num_high_func=get_last_packet_frame_num_high,
        print_info=True, allow_not_exists=False, status_update_every=100,
        x_y_num_gtu_overwrite=None, gtu_x_num_gtu_overwrite=None, gtu_y_num_gtu_overwrite=None,
        packet_size=128, gtu_in_packet_offset=-4
):
    import pandas as pd
    import datetime

    visualized_projections_x_y = []
    visualized_projections_gtu_x = []
    visualized_projections_gtu_y = []

    num_npy = 0
    num_root = 0
    
    if print_info:
        t0 = datetime.datetime.now()
        print('Extracting frames start: {:%Y:%m:%d %H:%M:%S}'.format(t0))
    
    ####
    
    num_gtu_dict = {'x_y': x_y_num_gtu_overwrite, 'gtu_x': gtu_x_num_gtu_overwrite, 'gtu_y': gtu_y_num_gtu_overwrite}
    abs_start_frame_arr_dict = {'x_y': None, 'gtu_x': None, 'gtu_y': None}
    abs_end_frame_arr_dict = {'x_y': None, 'gtu_x': None, 'gtu_y': None}
    arr_slice_start_corr_arr_dict = {'x_y': None, 'gtu_x': None, 'gtu_y': None}
    # arr_slice_len  - same as num_gtu_dict
        
    abs_start_frame_arr = np.array(get_first_packet_frame_num_in_df(
        all_rows, packet_size=packet_size, gtu_in_packet_offset=gtu_in_packet_offset))

    abs_end_frame_arr = np.array(get_last_packet_frame_num_high_in_df(
        all_rows, packet_size=packet_size, gtu_in_packet_offset=gtu_in_packet_offset))
    
    for proj_label, num_gtu_overwrite in num_gtu_dict.items():
        
        if num_gtu_overwrite is None:
            abs_start_frame_arr_dict[proj_label] = abs_start_frame_arr
            abs_end_frame_arr_dict[proj_label] = abs_end_frame_arr
            num_gtu_dict[proj_label] = np.array(all_rows[num_gtu_column])  # memory inefficient
        else:
            abs_end_frame_at_owr_gtu_arr = np.array(get_last_packet_frame_num_high_in_df(
                all_rows, 
                num_gtu_overwrite=np.maximum(np.array(all_rows[num_gtu_column]), num_gtu_overwrite), 
                packet_size=packet_size, gtu_in_packet_offset=gtu_in_packet_offset))
            
            d = abs_end_frame_at_owr_gtu_arr - abs_start_frame_arr
            d_lt_num_gtu_overwrite_mask = d < num_gtu_overwrite

            abs_start_frame_at_owr_gtu_arr = np.array(abs_start_frame_arr)
                
            abs_start_frame_at_owr_gtu_arr[d_lt_num_gtu_overwrite_mask] =                 abs_start_frame_arr[d_lt_num_gtu_overwrite_mask] - (num_gtu_overwrite - d[d_lt_num_gtu_overwrite_mask])
            
            check_start_frame_at_owr_gtu_mask =                 abs_start_frame_at_owr_gtu_arr[d_lt_num_gtu_overwrite_mask] < all_rows[packet_id_column][d_lt_num_gtu_overwrite_mask]*packet_size

            if np.any(check_start_frame_at_owr_gtu_mask):
                print(np.count_nonzero(check_start_frame_at_owr_gtu_mask))
                # inefficient to increase compatibility
                first_event_id =  np.array(all_rows[event_id_column][d_lt_num_gtu_overwrite_mask][check_start_frame_at_owr_gtu_mask])[0]
                first_gtu_in_packet = np.array(all_rows[gtu_in_packet_column][d_lt_num_gtu_overwrite_mask][check_start_frame_at_owr_gtu_mask])[0]
                first_packet_id = np.array(all_rows[packet_id_column][d_lt_num_gtu_overwrite_mask][check_start_frame_at_owr_gtu_mask])[0]
                first_num_gtu = np.array(all_rows[num_gtu_column][d_lt_num_gtu_overwrite_mask][check_start_frame_at_owr_gtu_mask])[0]
                first_start_frame_at_owr_gtu = abs_start_frame_at_owr_gtu_arr[d_lt_num_gtu_overwrite_mask][check_start_frame_at_owr_gtu_mask][0]
                first_end_frame_at_owr_gtu = abs_end_frame_at_owr_gtu_arr[d_lt_num_gtu_overwrite_mask][check_start_frame_at_owr_gtu_mask][0]
                first_start_frame = abs_start_frame_arr[d_lt_num_gtu_overwrite_mask][check_start_frame_at_owr_gtu_mask][0]
                first_end_frame = abs_end_frame_arr[d_lt_num_gtu_overwrite_mask][check_start_frame_at_owr_gtu_mask][0]
                first_lower_packet_bound = (np.array(all_rows[packet_id_column][d_lt_num_gtu_overwrite_mask][check_start_frame_at_owr_gtu_mask])*packet_size)[0]
                
                raise RuntimeError(
                    'Cannot accommodate sufficient window of packet frames in projection {proj}, num_gtu_overwrite={num_gtu_overwrite}. '
                    '(First invalid entry: event_id={event_id}, gtu_in_packet={gtu_in_packet}, num_gtu={num_gtu}, packet_id={packet_id}, '
                    'computed frame range w/o overwrite=[{start_frame},{end_frame}], '
                    'computed frame range w/ overwrite=[{start_frame_at_owr_gtu},{end_frame_at_owr_gtu}], '
                    'packet_bounds=[{lower_packet_bound}, {higher_packet_bound}])'.format(
                        proj=proj_label, num_gtu_overwrite=num_gtu_overwrite,
                        event_id=first_event_id, 
                        gtu_in_packet=first_gtu_in_packet,
                        num_gtu=first_num_gtu, packet_id=first_packet_id,
                        start_frame_at_owr_gtu=first_start_frame_at_owr_gtu,
                        end_frame_at_owr_gtu=first_end_frame_at_owr_gtu,
                        start_frame=first_start_frame,
                        end_frame=first_end_frame,
                        lower_packet_bound=first_lower_packet_bound, 
                        higher_packet_bound=first_lower_packet_bound + packet_size
                ))

            abs_start_frame_arr_dict[proj_label] = abs_start_frame_at_owr_gtu_arr
            abs_end_frame_arr_dict[proj_label] = abs_end_frame_at_owr_gtu_arr

    
    # probably could be not necessary if max overwrite is determined sooner/better
    lowest_start_frame_arr = np.min(list(abs_start_frame_arr_dict.values()), axis=0)
    highest_end_frame_arr = np.max(list(abs_end_frame_arr_dict.values()), axis=0)
    
    for proj_label, abs_start_frame_arr in abs_start_frame_arr_dict.items():
        arr_slice_start_corr_arr_dict[proj_label] = abs_start_frame_arr - lowest_start_frame_arr 
    
    ####
    
    if isinstance(all_rows, pd.DataFrame):
        _all_rows = all_rows.iterrows()
    else:
        _all_rows = enumerate(all_rows)
        
    ####
        
    for j, (i, r) in enumerate(_all_rows):
        
        lowest_start_frame = lowest_start_frame_arr[j]
        highest_end_frame = highest_end_frame_arr[j]
        
        if print_info:
            if j % status_update_every == 0 and j > 0:
                print('{:>5} / {:<5} ({:>5} npy, {:>5} root, {} since start)'.format(
                    j+1, len(all_rows), num_npy, num_root, datetime.datetime.now() - t0))
        
        sfa_str = r[source_file_acquisition_full_column]
        if sfa_str.endswith('.npy'):
            if not os.path.exists(sfa_str):
                msg_str = "Npy file does not exist: {}".format(sfa_str)
                if allow_not_exists:
                    if print_info:
                        print(msg_str, file=sys.stderr)
                    continue
                else:
                    raise RuntimeError(msg_str)
            acquisition_arr = np.load(sfa_str)
            if acquisition_arr.shape[0] != 256:
                raise Exception('Unexpected number of frames in the acquisition file "{}" (#{}  ID {})'.format(
                    sfa_str, i, r[event_id_column]))
            frames_acquisition = acquisition_arr[lowest_start_frame:highest_end_frame]
            num_npy += 1
        elif sfa_str.endswith('.root'):
            try:
                frames_acquisition = tool.acqconv.get_frames(
                    sfa_str, lowest_start_frame, highest_end_frame-1,
                    entry_is_gtu_optimization=True)
            except Exception as e:
                if print_info:
                    print('{:>5} / {:<5} ({:<5} npy, {:<5} root, {} since start)'.format(
                        j+1, len(all_rows), num_npy, num_root, datetime.datetime.now() - t0))
                    sys.stdout.flush()
                raise e
            num_root += 1
        else:
            raise Exception('Unexpected source_file_acquisition_full "{}"'.format(sfa_str))

        if inverse_means_arr is not None:
            if callable(inverse_means_arr):
                _inverse_means_arr = inverse_means_arr(r)
            else:
                _inverse_means_arr = inverse_means_arr

            if _inverse_means_arr is not None:
                frames_acquisition = frames_acquisition * _inverse_means_arr
        
        for proj_label, ret, target_list in (
                ('x_y', return_x_y, visualized_projections_x_y), 
                ('gtu_y', return_gtu_y, visualized_projections_gtu_y), 
                ('gtu_x', return_gtu_x, visualized_projections_gtu_x)):
            if not return_x_y:
                continue
            
            arr_slice_start_corr = arr_slice_start_corr_arr_dict[proj_label][j]
            try:
                arr_slice_end_corr = int(num_gtu_dict[proj_label])
            except TypeError:
                arr_slice_end_corr = num_gtu_dict[proj_label][j]
                
            arr_slice_end_corr += arr_slice_start_corr
            
            _frames_acquisition = frames_acquisition[arr_slice_start_corr:arr_slice_end_corr]
            
            target_item = transform_func(proj_label, _frames_acquisition)                 if callable(transform_func) else _frames_acquisition
                        
            target_list.append(target_item)
        
    ret = []
    if return_x_y:
        ret.append(visualized_projections_x_y)
    if return_gtu_y:
        ret.append(visualized_projections_gtu_y)
    if return_gtu_x:
        ret.append(visualized_projections_gtu_x)

    if return_offsets:
        ret += [num_gtu_dict, abs_start_frame_arr_dict, abs_end_frame_arr_dict, 
                arr_slice_start_corr_arr_dict]
        
    return tuple(ret)


# In[398]:


def get_inverse_means_arr(r, inverse_means_frame_arr=inverse_means_frame_arr):
    if r['source_class'] in ((EVENT_CLASS_LABELED_SHOWER_FLIGHT, EVENT_CLASS_LABLELED_NOISE_FLIGHT, EVENT_CLASS_NUMBER_UNLABELED_NOISE)):
        return inverse_means_frame_arr
    return None


# In[ ]:





# In[237]:


learning_data_frames_dict = {}
for subset_name in ['train', 'test']:
    
    output_npy_pathname = os.path.join(data_snippets_dir, 'frames', '{}.npz'.format(subset_name))

    if not os.path.exists(output_npy_pathname):
        os.makedirs(os.path.dirname(output_npy_pathname), exist_ok=True)
        x_y_list, gtu_y_list, gtu_x_list = \
            get_frame_array(
                learning_data_dict[subset_name], inverse_means_arr=get_inverse_means_arr,
                x_y_num_gtu_overwrite=None, gtu_x_num_gtu_overwrite=30, gtu_y_num_gtu_overwrite=30,
                return_x_y=True, return_gtu_x=True, return_gtu_y=True)

        learning_data_frames_dict[subset_name] = _frames = {
            'x_y': np.array(x_y_list), 'gtu_y': np.array(gtu_y_list), 'gtu_x': np.array(gtu_x_list)
        }
        
        np.savez_compressed(output_npy_pathname, **_frames)
    
#    else:
#        learning_data_frames_dict[subset_name] = np.load(output_npy_pathname)
  
# In[246]:


#l = [a.shape[1] for a in gtu_y_list]
#print(min(l))
#print(max(l))


# In[ ]:





# In[243]:


# tmp_d = os.path.join(data_snippets_dir, 'tmp')
# os.makedirs(tmp_d, exist_ok=True)
# joblib.dump(x_y_list, tmp_d + '/x_y_list.pkl')
# joblib.dump(gtu_y_list, tmp_d + '/gtu_y_list.pkl')
# joblib.dump(gtu_x_list, tmp_d + '/gtu_x_list.pkl')


# In[245]:


# %%bash
# ls -lah ver4_machine_learning_convnet_szakcs_w_labeled_flight_20190628_2__v2/tmp


# In[ ]:





# In[113]:


# overwrite_existing_dump = True

# extra_metafields = ['weights', 'y', 'source_class']

# dataset_condenser_input_df_dict = {}
# dataset_condenser_input_df_path_dict = {}

# for subset_name in ['train', 'test']:
#     dataset_condenser_input_df_dict[subset_name] = \
#         pd.DataFrame(
#             np.hstack([learning_data[subset_name]['packets']] + \
#                       [learning_data[subset_name][m].reshape(-1,1) for m in extra_metafields]
#                      ), 
#             columns=list(dataset_condenser_df_columns) + extra_metafields
#         )
    
#     dump_pathname = os.path.join(data_snippets_dir, 'dataset_condenser_input_{}.tsv'.format(subset_name))
#     if not os.path.exists(dump_pathname) or overwrite_existing_dump:
#         print('Saving', dump_pathname)
#         dataset_condenser_input_df_dict[subset_name].to_csv(dump_pathname, sep='\t')
#     else:
#         print('Already exists', dump_pathname)
    
#     dataset_condenser_input_df_path_dict[subset_name] = dump_pathname


# In[116]:


# dataset_condenser_input_df_dict[subset_name].head()


# In[201]:


# import utility_functions

# max_depth = 4
# max_num_dirs = 4

# trie_prefix_len = 1
# sep='/'

# dataset_condenser_input_data_dirs_dict = {}

# def _add_dir_tree_node(prefix_groups_tree, t_i):
#     node_found = False
#     exchange_root = False
#     for start_prefix, v in prefix_groups_tree.items(): 
#         if len(start_prefix) < len(t_i[0]) and t_i[0].startswith(start_prefix):
#             _add_dir_tree_node(v, t_i)
#             node_found = True
#             break
#         elif len(start_prefix) > len(t_i[0]) and start_prefix.startswith(t_i[0]):
#             exchange_root = True
#             break
#     if exchange_root:
#         prefix_groups_tree[t_i[0]] = {start_prefix: prefix_groups_tree[start_prefix]}
#         del prefix_groups_tree[start_prefix]
#     elif not node_found:
#         prefix_groups_tree[t_i[0]] = {}

# def _get_dirs_by_depth(prefix_groups_tree, target_depth, cur_depth=1):
#     dirs_list = []
#     for k, v in prefix_groups_tree.items():
#         if cur_depth != target_depth:
#             if len(v) > 0:
#                 dirs_list += _get_dirs_by_depth(v, target_depth, cur_depth+1)
#             else:
#                 dirs_list.append(k)
#         else:
#             dirs_list.append(k)
#     return dirs_list

# for subset_name in ['train', 'test']:
    
#     dataset_condenser_input_data_dirs_dict[subset_name] = {}
    
#     print('Subset:', subset_name)
    
#     l = \
#         list(set(dataset_condenser_input_df_dict[subset_name]['source_file_acquisition_full'].apply(os.path.dirname)))
#     l_realpath = list(map(os.path.realpath, l))
    
#     for t_label, t_l in (('absolute', l), ('realpath',l_realpath)):

#         t_input_data_dirs = input_data_dirs = t_l
#         items_list = utility_functions.get_grouping_by_prefix(t_input_data_dirs, trie_prefix_len=1, sep='/')

#         row_keys_trie = utility_functions.StrPrefixTrie()
#         for k in items_list:
#             row_keys_trie.insert(k)

#         trie_prefixes = []
#         for prefix, multiplicity in row_keys_trie.getPrefixes(trie_prefix_len):
#             if prefix.endswith(sep):
#                 prefix_no_sep = prefix[:-len(sep)]
#                 if len(prefix_no_sep) == 0:
#                     continue
#                 trie_prefixes.append((prefix_no_sep, multiplicity))

#         trie_prefixes_filtered = [t for t in trie_prefixes if len(t[0].split('/'))-1 <= max_depth or t[1] == 1]

#         prefix_groups_tree = { trie_prefixes[0][0]: {} }

#         for t_i  in trie_prefixes[1:]:
#             _add_dir_tree_node(prefix_groups_tree, t_i)

#         for i in range(max_depth,0,-1):
#             dirs_list = _get_dirs_by_depth(prefix_groups_tree, i)
#             if len(dirs_list) <= max_num_dirs:
#                 break

#         dataset_condenser_input_data_dirs_dict[subset_name][t_label] = dirs_list
        
#         print(dirs_list)


# In[173]:





# ## Feature selection and classification

# (This is part that might be changed in the later machine learning procedures)
# 1. Variance thresholding to remove features without any variance.
# 2. Univariate feature selection to select smaller but still large enough subset of features (mainly to limit the computational demands). This particular procedure selects 400 features.
# 3. Recursive feature elimination with cross-validation - Training and validating Extremely Randomized Trees model (ExtraTreesClassifier) on multiple combinations of features aimed to select set of features that provide the best classification accuracy.

# #### Model training

# In[86]:


# subprocess.run(
#     'docker build --tag convnet_euso:0.2 -f {} {}'.format(
#         covnet_euso_dockerfile, covnet_euso_docker_dir
#     ), 
#     shell=True)


# In[107]:


# subprocess.run(
#     'docker run --rm  -v {}:/src -w /src  convnet_euso:0.2 python3 dataset_condenser.py --help'.format(
#         covnet_euso_src_dir
#     ),
#     shell=True
# )


# In[ ]:


# assert 'y' in extra_metafields

# for subset_name, dataset_condenser_input_df_path in dataset_condenser_input_df_path_dict.items():

#     output_name = subset_name
#     output_dir = os.path.join(data_snippets_dir, 'dataset_condenser', subset_name)  
#     # data_snippets_dir mus be base, otherwise addtional docker volume is required
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     data_volumes_str = ''
#     for t_label, t_l in dataset_condenser_input_data_dirs_dict[subset_name].items():
#         for d in t_l:
#             data_volumes_str += ' -v {d}:{d}:ro'.format(d=d)
    
#     dataset_condenser_run_cmd_str = (
#         'docker run --rm {data_volumes_str} -v {realpath_data_dir}:/{in_container_data_dir} -v {src}:/src -w /src convnet_euso:0.2 python3 dataset_condenser.py ' 
#             '--packet_dims 128 48 48 16 16 ' 
#             '-f /{input_df_path} ' 
#             '-d /{output_dir} '
#             '-n {output_name} '
#             '--store_yx --store_gtux --store_gtuy '
#             '--target _meta '
#             '--extra_metafields {extra_metafields_str} '
#             '--target_column {target_column} '
#             'gtupack --num_gtu_around 4 15' 
#     ).format(
#             src=covnet_euso_src_dir,
#             input_df_path=dataset_condenser_input_df_path,
#             target_column='y',
#             extra_metafields_str=' '.join(extra_metafields),
#             output_name=output_name,
#             output_dir=output_dir,
#             in_container_data_dir=data_snippets_dir,
#             realpath_data_dir=os.path.realpath(data_snippets_dir),
#             data_volumes_str=data_volumes_str
#         )
    
#     print('-'*100)
#     print(dataset_condenser_run_cmd_str)
#     print('-'*100)
    
#     subprocess.run(
#         dataset_condenser_run_cmd_str, shell=True
#     )


# In[ ]:





# In[89]:


# docker build --tag convnet_euso:0.2 -f Dockerfile-gpu /mnt/data_wdblue3d1/spbproc/convnet_euso
# docker run --rm  -v `realpath src`:/src -w /src  convnet_euso:0.2 dataset_condenser.py --help
# 


# ### Performance of the ExtraTreesClassifier model with RFECV features

# In[123]:


# y_test = learning_data__y_test
# y_test_pred = rfecv_selector_on_extra_trees_cls.predict(learning_data__var_th_X_test)

# # intentionally not T (for comparison with older)
# print(sklearn.metrics.confusion_matrix(
#     y_test, 
#     y_test_pred))

# print_confusion_matrix(
#     y_test, 
#     y_test_pred, output_format='ipython_html_styled')

# print_accuracy_cls_report(
#     y_test, 
#     y_test_pred)

# labeled_data_cls_stats = \
#     print_labeled_data_cls_stats(
#         mask_arr_test=learning_data__lbl_noise_flight_mask_arr_test,
#         y_test=y_test,
#         y_test_pred=y_test_pred)


# In[124]:


# # y_test, y_test_pred
# # sklearn.metrics.accuracy_score
# X_train = learning_data__var_th_X_train
# y_train = learning_data__y_train
# w_train = learning_data__weights_train

# X_test = learning_data__var_th_X_test
# y_test = learning_data__y_test
# y_test_pred = rfecv_selector_on_extra_trees_cls.predict(learning_data__var_th_X_test)
# w_test = learning_data__weights_test

# print('sklearn.metrics.accuracy_score:', sklearn.metrics.accuracy_score(y_test, y_test_pred, sample_weight=w_test))
# print('balanced_accuracy_score:       ', balanced_accuracy_score(y_test, y_test_pred, sample_weight=w_test))


# In[117]:


# fig, ax = plt.subplots(figsize=(4,3))
# ax.hist(rfecv_selector_on_extra_trees_cls.predict_proba(learning_data__var_th_X_test[learning_data__source_class_test == EVENT_CLASS_SIMU_TRACK])[:,1], 
#         bins=100, alpha=1, range=(0,1), label='Simu track')
# ax.set_ylabel('Number of events')
# ax.set_xlabel('Probability')
# ax.set_yscale('log')
# fig.savefig(os.path.join(data_snippets_dir, 'test_set_simu_track_proba_distribution_horizontal.svg'))
# plt.show()


# In[121]:


# fig, ax = plt.subplots(figsize=(4,3))
# ax.hist(rfecv_selector_on_extra_trees_cls.predict_proba(learning_data__var_th_X_test[learning_data__source_class_test == EVENT_CLASS_SIMU_TRACK])[:,1], 
#         bins=100, alpha=.5, range=(0,1), label='Simu track')
# ax.hist(rfecv_selector_on_extra_trees_cls.predict_proba(learning_data__var_th_X_test[learning_data__source_class_test != EVENT_CLASS_SIMU_TRACK])[:,1], 
#         bins=100, alpha=.5, range=(0,1), label='Noise')
# ax.set_ylabel('Number of events')
# ax.set_xlabel('Probability')
# ax.set_yscale('log')
# fig.savefig(os.path.join(data_snippets_dir, 'test_set_both_proba_distribution_horizontal.svg'))
# plt.show()


# #### Cross-validation

# In[125]:


# def cross_val_calc_weights(indices, learning_data__y=learning_data__y, learning_data__source_class=learning_data__source_class): 
#     return calc_learning_data_weights(learning_data__y[indices], learning_data__source_class[indices], print_info=False)

# extra_trees_cls_on_train_rfecv_for_crossvalidation = sklearn.ensemble.ExtraTreesClassifier(**rfe_extra_trees_params)
# # not entirely correct, feature selection should be also included in crossvalidation training

# learning_data__rfecv_var_th_X = \
#     rfecv_selector_on_extra_trees_cls.transform(
#         var_th_selector_on_scaled_train.transform(
#             learning_data__X
#         )
#     )

# extra_trees_cls_on_train_rfecv_for_crossvalidation_crv_results = \
#     cross_val_score_meta_scored(
#         extra_trees_cls_on_train_rfecv_for_crossvalidation, 
#         learning_data__rfecv_var_th_X, learning_data__y, 
#         meta_score_func=None,
#         score_func=balanced_accuracy_score,
#         cv=3, random_state=32, 
#         train_sample_weight_func=cross_val_calc_weights
# )

# print('Cross-validation accuracy:', extra_trees_cls_on_train_rfecv_for_crossvalidation_crv_results)
# print('Mean accuracy:            ', np.mean(extra_trees_cls_on_train_rfecv_for_crossvalidation_crv_results))
# print('Std accuracy:             ', np.std(extra_trees_cls_on_train_rfecv_for_crossvalidation_crv_results))

# extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results = \
# cross_val_score_meta_scored(
#     extra_trees_cls_on_train_rfecv_for_crossvalidation,
#     learning_data__rfecv_var_th_X, learning_data__y, 
#     cv=3, random_state=32, verbose=1,
#     meta_score_func=score_masked_using_indices_lbl_noise_flight_mask_arr_all,
#     train_sample_weight_func=cross_val_calc_weights
# )

# print('Cross-validation accuracy (lbl_noise):', extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results)
# print('Mean accuracy (lbl_noise):            ', np.mean(extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results))
# print('Std accuracy (lbl_noise):             ', np.std(extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results))
    


# #### Cross-validation of labeled noise data

# ##### random_state = 123

# In[208]:


# extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results_2 = \
# cross_val_score_meta_scored(
#     extra_trees_cls_on_train_rfecv_for_crossvalidation,
#     learning_data__rfecv_var_th_X, learning_data__y, 
#     cv=3, random_state=128, verbose=1,
#     meta_score_func=score_masked_using_indices_lbl_noise_flight_mask_arr_all,
#     train_sample_weight_func=cross_val_calc_weights
# )
# print('Cross-validation accuracy (lbl_noise, seed=123):', extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results_2)
# print('Mean accuracy (lbl_noise, seed=123):            ', np.mean(extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results_2))
# print('Std accuracy (lbl_noise, seed=123):             ', np.std(extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results_2))


# ## Recognition efficiency

# ### Test set sensitivity as function of the energy

# In[126]:


# extra_trees_cls_on_train_rfecv__test__numbers_by_energy = \
#     score_by_column(
#         rfecv_selector_on_extra_trees_cls, 
#         learning_data__var_th_X_test[learning_data__simu_shower_track_mask_arr_test], 
#         learning_data__y_test[learning_data__simu_shower_track_mask_arr_test], 
#         calc_cls_numbers, #sklearn.metrics.accuracy_score, 
#         learning_data__event_id_test[learning_data__simu_shower_track_mask_arr_test], 
#         combined_simu_df, 'etruth_trueenergy')


# In[179]:


# plt.close('all')
# for xscale in ('linear', 'log'):
#     fig, ax, errbr = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#             plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 20, 
#             xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#             xlabel = 'Energy [eV]', ylabel = 'Sensitivity', 
#             calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#             figsize=(6,3), ylim=(0,1.2), show=False)
#     ax.grid(linestyle='--')
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'test_set_sensitivity_function_of_energy_{}_ev.svg'.format(xscale)), dpi=150)
#     plt.show()


# In[180]:


# plt.close('all')
# for xscale in ('linear', 'log'):
#     fig, ax, errbr = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#             plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps=11, 
#             xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#             xlabel = 'Energy [MeV]', ylabel = 'Sensitivity', 
#             calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#             figsize=(6,3), ylim=(0,1.2), show=False)
#     ax.grid(linestyle='--')
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'test_set_sensitivity_function_of_energy_{}_11steps.svg'.format(xscale)), dpi=150)
#     plt.show()


# In[203]:


# plt.close('all')
# for xscale in ( 'linear', 'log', ):
#     fig, ax, errbr = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#             plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps=20, 
#             xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#             xlabel = 'Energy [MeV]', ylabel = 'Sensitivity', 
#             calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#             figsize=(5,3), ylim=(.2,1.2), show=False,
# #         show_fill_between=False, show_yerr=True,
# #         errorbar_attrs={**EFFICIENCY_STAT_ERRORBAR_DEFAULTS, 'ecolor': 'gray'}
#     )
#     ax.grid(linestyle='--')
#     if xscale == 'linear' :
# #         ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
#         ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
#         ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,n: '{:.3f}'.format(np.log10(v)) if v > 0 else '' ))
#         ax.set_xlabel('log(Energy [eV])')
        
# #     else:
# #         ax.set_xscale('symlog')
# #         ax.xaxis.set_minor_formatter(mpl.ticker.ExponentFormatter(minor_thresholds=(100, 100), labelOnlyBase=False))
# #         ax.xaxis.set_minor_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
# #         ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto', numdecs=4, numticks='auto'))
        
# #         ax.xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(_t_f))

# #         mil = ax.xaxis.get_minor_locator()

# #         print(dir(mil))
# #         print(mil._base)
# #         print(mil._subs)
# #         print(mil.numdecs)
# #         print(mil.numticks)
    
# #     plt.close('all')
    
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'test_set_sensitivity_function_of_energy_{}_20steps.svg'.format(xscale)), dpi=150)
#     plt.show()


# In[138]:


# plt.close('all')
# for xscale in ( 'linear', 'log', ):
#     fig, ax, errbr = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#             plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps=20, 
#             xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#             xlabel = 'Energy [MeV]', ylabel = 'Efficiency', 
#             calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#             figsize=(5,3), ylim=(.4,1.1), show=False,
# #         show_fill_between=False, show_yerr=True,
# #         errorbar_attrs={**EFFICIENCY_STAT_ERRORBAR_DEFAULTS, 'ecolor': 'gray'}
#     )
#     ax.grid(linestyle='--')
#     if xscale == 'linear' :
# #         ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
#         ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
#         ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,n: '{:.3f}'.format(np.log10(v)) if v > 0 else '' ))
#         ax.set_xlabel('log(Energy [eV])')
        
# #     else:
# #         ax.set_xscale('symlog')
# #         ax.xaxis.set_minor_formatter(mpl.ticker.ExponentFormatter(minor_thresholds=(100, 100), labelOnlyBase=False))
# #         ax.xaxis.set_minor_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
# #         ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto', numdecs=4, numticks='auto'))
        
# #         ax.xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(_t_f))

# #         mil = ax.xaxis.get_minor_locator()

# #         print(dir(mil))
# #         print(mil._base)
# #         print(mil._subs)
# #         print(mil.numdecs)
# #         print(mil.numticks)
    
# #     plt.close('all')
    
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'test_set_sensitivity_function_of_energy_{}_20steps_efficiency_range_04_11.svg'.format(xscale)), dpi=150)
#     plt.show()


# In[142]:


# plt.close('all')
# for xscale in ( 'linear', 'log', ):
#     fig, ax, errbr = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#             plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps=20, 
#             xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#             xlabel = 'Energy [MeV]', ylabel = 'Efficiency', 
#             calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#             figsize=(5,2.6), ylim=(.4,1.1), show=False,
# #         show_fill_between=False, show_yerr=True,
# #         errorbar_attrs={**EFFICIENCY_STAT_ERRORBAR_DEFAULTS, 'ecolor': 'gray'}
#     )
#     ax.grid(linestyle='--')
#     if xscale == 'linear' :
# #         ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
#         ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
#         ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,n: '{:.3f}'.format(np.log10(v)) if v > 0 else '' ))
#         ax.set_xlabel('log(Energy [eV])')
        
# #     else:
# #         ax.set_xscale('symlog')
# #         ax.xaxis.set_minor_formatter(mpl.ticker.ExponentFormatter(minor_thresholds=(100, 100), labelOnlyBase=False))
# #         ax.xaxis.set_minor_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
# #         ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto', numdecs=4, numticks='auto'))
        
# #         ax.xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(_t_f))

# #         mil = ax.xaxis.get_minor_locator()

# #         print(dir(mil))
# #         print(mil._base)
# #         print(mil._subs)
# #         print(mil.numdecs)
# #         print(mil.numticks)
    
# #     plt.close('all')
    
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'test_set_sensitivity_function_of_energy_{}_20steps_efficiency_range_04_11_h26.svg'.format(xscale)), dpi=150)
#     plt.show()


# In[135]:


# plt.close('all')
# for xscale in ( 'linear', 'log', ):
#     fig, ax, errbr = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#             plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps=20, 
#             xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#             xlabel = 'Energy [MeV]', ylabel = 'Sensitivity', 
#             calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#             figsize=(5,3), ylim=(.4,1.1), show=False,
#             show_fill_between=False, show_yerr=True,
#             errorbar_attrs={**EFFICIENCY_STAT_ERRORBAR_DEFAULTS, 'ecolor': '#000000', 'linestyle': '--', 'color': 'gray'}
#     )
#     ax.grid(linestyle='--')
#     if xscale == 'linear' :
# #         ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
#         ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
#         ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,n: '{:.3f}'.format(np.log10(v)) if v > 0 else '' ))
#         ax.set_xlabel('log(Energy [eV])')
        
# #     else:
# #         ax.set_xscale('symlog')
# #         ax.xaxis.set_minor_formatter(mpl.ticker.ExponentFormatter(minor_thresholds=(100, 100), labelOnlyBase=False))
# #         ax.xaxis.set_minor_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
# #         ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto', numdecs=4, numticks='auto'))
        
# #         ax.xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(_t_f))

# #         mil = ax.xaxis.get_minor_locator()

# #         print(dir(mil))
# #         print(mil._base)
# #         print(mil._subs)
# #         print(mil.numdecs)
# #         print(mil.numticks)
    
# #     plt.close('all')
    
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'test_set_sensitivity_function_of_energy_{}_20steps_range_04_11_errbars.svg'.format(xscale)), dpi=150)
#     plt.show()


# In[204]:


# plt.close('all')
# for xscale in ( 'linear',  ): #'log',
#     fig, ax, errbr = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#             plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps=20, 
#             xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#             xlabel = 'Energy [MeV]', ylabel = 'Sensitivity', 
#             calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#             figsize=(5,3), ylim=(.2,1.2), show=False,
# #         show_fill_between=False, show_yerr=True,
# #         errorbar_attrs={**EFFICIENCY_STAT_ERRORBAR_DEFAULTS, 'ecolor': 'gray'}
#     )
#     ax.grid(linestyle='--')
#     if xscale == 'linear' :
# #         ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
#         ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
#         ax.xaxis.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,n: '{:.3f}'.format(np.log10(v)) if v > 0 else '' ))
#         ax.set_xlabel('log(Energy [eV])')
        
#         print(ax.get_xlim())
#         print(ax.xaxis.get_major_locator())
        
# #     else:
# #         ax.set_xscale('symlog')
# #         ax.xaxis.set_minor_formatter(mpl.ticker.ExponentFormatter(minor_thresholds=(100, 100), labelOnlyBase=False))
# #         ax.xaxis.set_minor_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
# #         ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs='auto', numdecs=4, numticks='auto'))
        
# #         ax.xaxis.set_minor_formatter(mpl.ticker.FuncFormatter(_t_f))

# #         mil = ax.xaxis.get_minor_locator()

# #         print(dir(mil))
# #         print(mil._base)
# #         print(mil._subs)
# #         print(mil.numdecs)
# #         print(mil.numticks)
    
# #     plt.close('all')
    
# #     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
# #                              'test_set_sensitivity_function_of_energy_{}_20steps.svg'.format(xscale)), dpi=150)
#     plt.show()


# In[235]:


# plt.close('all')
# fig, ax, errbr = \
#     plot_efficiency_stat(
#         extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#         plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps=20, 
#         xscale='linear', xtranslate_func=lambda l: [v*1e6 for v in l],
#         xlabel = 'Energy [MeV]', ylabel = 'Sensitivity', 
#         calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#         figsize=(5,3), ylim=(.2,1.2), show=False,
# #         show_fill_between=False, show_yerr=True,
# #         errorbar_attrs={**EFFICIENCY_STAT_ERRORBAR_DEFAULTS, 'ecolor': 'gray'}
# )

# ax.grid(linestyle='--')
# # ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
# ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
# ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,n: '{:.3f}'.format(np.log10(v)) if v > 0 else '' ))
# ax.set_xlabel('log(Energy [eV])')

# plot_x, plot_y, plot_xerr, plot_yerr = \
#     get_efficiency_stat_plot_data(extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#                              plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 10, 
# #                              xtranslate_func=lambda l: [v*1e6 for v in l],
#                              calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint)

# def fit_func(x, A,B,C): 
#     return A*np.exp(-B*x) + C

# plot_x = np.array(plot_x) * 1e6

# popt, pcov = sp_opt.curve_fit(fit_func, plot_x, plot_y, method='lm', p0=(-0.56, 0.965/1e18, 0.965), 
#                               sigma=np.array(plot_yerr[1])-plot_yerr[0])
# print('popt:\n', popt.tolist())
# print('pcov:\n', pcov)

# orig_lim = ax.get_xlim()

# func_plt_x = np.linspace(*orig_lim,100)
# func_plt_y = fit_func(func_plt_x, *popt)

# func_plt_y_simple = fit_func(func_plt_x, -0.927, 1.123e-18, 0.96)


# # print(func_plt_x)
# # print(func_plt_y)

# # f,ax = plt.subplots()
# ax.plot(func_plt_x, func_plt_y, 'r', alpha=0.5, zorder=10)
# # ax.plot(func_plt_x, func_plt_y_simple, 'b', alpha=0.5, zorder=10)


# ax.set_xlim(*orig_lim)
# # ax.set_ylim(0.8,1)

# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_energy_linear_20steps_fitted.svg'), dpi=150)
# plt.show()


# ### Number of true positivie and positive samples as function of the energy

# #### Number of positive samples as function of the energy

# In[208]:


# for xscale in ('linear', 'log'):
#     fig, ax, errbr = \
#         plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#                              plotted_stat='num_positive', num_steps = 20, 
#                              xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#                              xlabel = 'Energy [eV]', ylabel = 'Num. positive', 
#                              figsize =(6,3), show=False)

#     ax.grid(linestyle='--')
#     if xscale == 'linear' :
# #         ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
#         ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
#         ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,n: '{:.3f}'.format(np.log10(v)) if v > 0 else '' ))
#         ax.set_xlabel('log(Energy [eV])')
        
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'Number of positive samples as function of the energy - {}.svg'.format(xscale)), dpi=150)
#     plt.show()
    


# ##### Number of true positive samples as function of the energy

# In[209]:


# for xscale in ('linear', 'log'):
#     fig, ax, errbr = \
#         plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#                              plotted_stat='num_true_positive', num_steps = 20, 
#                              xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#                              xlabel = 'Energy [eV]', ylabel = 'Num. true positive', 
#                              figsize = (6,3), show=False)
#     ax.grid(linestyle='--')
#     if xscale == 'linear' :
#         ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
#         ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,n: '{:.3f}'.format(np.log10(v)) if v > 0 else '' ))
# #         ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
#         ax.set_xlabel('log(Energy [eV])')
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'Number of true positive samples as function of the energy - {}.svg'.format(xscale)), dpi=150)
#     plt.show()


# ##### Number of true positive or positive samples as function of the energy - comparison

# In[210]:


# plt.close('all')
# for xscale in ('linear', 'log'):
#     fig, ax = plt.subplots()
#     fig, ax, errbr_num_positive = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#             plotted_stat='num_positive', num_steps = 20, xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#             xlabel = 'Energy [eV]', ylabel = 'Num. positive', label='Num. positive',
#             figsize = (6,3), errorbar_attrs=dict(linestyle='--', color='blue'), 
#             ax=ax, show=False)
#     fig, ax, errbr_num_true_positive = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
#             plotted_stat='num_true_positive', num_steps = 20, xscale=xscale, xtranslate_func=lambda l: [v*1e6 for v in l],
#             xlabel='Energy [eV]', ylabel = 'Num. true positive', label='Num. true positive',
#             figsize=(6,3), errorbar_attrs=dict(linestyle='-', color='green'),
#             ax=ax, show=False)
#     ax.set_ylabel('Num. samples')
#     ax.grid(linestyle='--')
#     if xscale == 'linear' :
#         ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=6))
#         ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,n: '{:.3f}'.format(np.log10(v)) if v > 0 else '' ))
# #         ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterExponent(minor_thresholds=(np.inf, np.inf), labelOnlyBase=False))
#         ax.set_xlabel('log(Energy [eV])')
#     ax.grid(linestyle='--')
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'Number of true positive or positive samples as function of the energy - comparison - {}.svg'.format(xscale)), dpi=150)
#     plt.show()


# #### Test set sensitivity as function of the theta (zenith angle)

# In[213]:


# extra_trees_cls_on_train_rfecv__test__numbers_by_theta = \
#     score_by_column(
#         rfecv_selector_on_extra_trees_cls, 
#         learning_data__var_th_X_test[learning_data__simu_shower_track_mask_arr_test], 
#         learning_data__y_test[learning_data__simu_shower_track_mask_arr_test], 
#         calc_cls_numbers,
#         learning_data__event_id_test[learning_data__simu_shower_track_mask_arr_test], 
#         combined_simu_df, 'etruth_truetheta')


# In[138]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_theta, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 90/2.5, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel ='True theta [deg]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (6,3), ylim=(0,1.5), show=False)
# ax.grid()
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_theta.svg'), dpi=150)
# plt.show()


# In[203]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_theta, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 90/5, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel ='Zenith angle [deg]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0.7,1.1), show=False,
#                         )
# ax.grid(linestyle='--')
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_theta_5deg.svg'), dpi=150)
# plt.show()


# In[215]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_theta, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 90/5, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel ='Zenith angle', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0.7,1.1), show=False,
#                         )
# ax.grid(linestyle='--')
# ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2.0f}°'))
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_theta_5deg_formatter.svg'), dpi=150)
# plt.show()


# In[216]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_theta, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 90/10, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel ='Zenith angle', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0.7,1.1), show=False,
#                         )
# ax.grid(linestyle='--')
# ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2.0f}°'))
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_theta_10deg_formatter.svg'), dpi=150)
# plt.show()


# #### Test set sensitivity as function of the phi (azimuth angle)

# In[202]:


# extra_trees_cls_on_train_rfecv__test__numbers_by_phi = \
#     score_by_column(
#         rfecv_selector_on_extra_trees_cls, 
#         learning_data__var_th_X_test[learning_data__simu_shower_track_mask_arr_test], 
#         learning_data__y_test[learning_data__simu_shower_track_mask_arr_test], 
#         calc_cls_numbers,
#         learning_data__event_id_test[learning_data__simu_shower_track_mask_arr_test], 
#         combined_simu_df, 'etruth_truephi')


# In[192]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 360/5, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel = 'Azmuth angle [deg]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (6,3), ylim=(0,1.5), show=False)
# ax.grid()
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_phi_10deg_5deg.svg'), dpi=150)
# plt.show()


# In[205]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 360/15, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel = 'Azimuth angle [deg]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0.7,1.1), show=False)
# ax.grid(linestyle='--')
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_phi_15deg.svg'), dpi=150)
# plt.show()


# In[211]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 360/15, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel = 'Azimuth angle', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0.7,1.1), show=False)
# ax.grid(linestyle='--')
# ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2.0f}°'))
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_phi_15deg_formatter.svg'), dpi=150)
# plt.show()


# In[213]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 360/10, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel = 'Azimuth angle [deg]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0.7,1.1), show=False)
# ax.grid(linestyle='--')
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_phi_10deg.svg'), dpi=150)
# plt.show()


# In[220]:


# plt.close('all')
# fig, ax = plt.subplots()
# fig, ax, errbr_num_positive = \
#     plot_efficiency_stat(
#         extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#         plotted_stat='num_positive', num_steps = 360/10, 
#         xtranslate_func=np.rad2deg,
#         xlabel = 'Azimuth angle [deg]', ylabel = 'Num. positive', label='Num. positive',
#         figsize = (5,3), errorbar_attrs=dict(linestyle='--', color='blue'), 
#         ax=ax, show=False)
# fig, ax, errbr_num_true_positive = \
#     plot_efficiency_stat(
#         extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#         plotted_stat='num_true_positive', num_steps = 360/10, 
#         xtranslate_func=np.rad2deg,
#         xlabel='Azimuth angle [deg]', ylabel = 'Num. true positive', label='Num. true positive',
#         figsize=(5,3), errorbar_attrs=dict(linestyle='-', color='green'),
#         ax=ax, show=False)
# ax.set_ylabel('Num. samples')
# ax.grid(linestyle='--')
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_num_positive_function_of_phi_comparison_10deg_{}.svg'.format(xscale)), dpi=150)
# plt.show()


# In[224]:


# plt.close('all')

# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 360/10, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel = 'Azimuth angle [deg]', ylabel = 'Sensitivity', label='Sensitivity',
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0.7,1.1), show=False)


# plot_x, plot_y, plot_xerr, plot_yerr = \
#     get_efficiency_stat_plot_data(
#         extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#         plotted_stat='num_positive', num_steps = 360/10, 
#         xtranslate_func=np.rad2deg)

# ax.plot(plot_x, plot_y/max(plot_y), color='red', label='Norm. num. positive')

# plot_x, plot_y, plot_xerr, plot_yerr = \
#     get_efficiency_stat_plot_data(
#         extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#         plotted_stat='num_true_positive', num_steps = 360/10, 
#         xtranslate_func=np.rad2deg)
# ax.plot(plot_x, plot_y/max(plot_y), color='magenta', label='Norm. num. true positive')
        
# ax.set_ylim(0.2, 1.05)        
# ax.set_ylabel('')
# ax.grid(linestyle='--')
# ax.legend()

# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_num_positive_function_of_phi_comparison_w_phi_10deg_{}.svg'.format(xscale)), dpi=150)
# plt.show()


# In[219]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 360/20, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel = 'Azimuth angle [deg]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0.7,1.1), show=False)
# ax.grid(linestyle='--')
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_phi_20deg.svg'), dpi=150)
# plt.show()


# In[207]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='sensitivity_err_mario', num_steps = 360/15, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel = 'Azimuth angle [deg]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0.7,1.1), show=False)
# ax.grid(linestyle='--')
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_phi_15deg_sensitivity_err_mario.svg'), dpi=150)
# plt.show()


# In[214]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 360/20, 
#                          xtranslate_func=np.rad2deg,
#                          xlabel = 'Azimuth angle [rad]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (6,3), ylim=(0,1.5), show=False)
# ax.grid()
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_phi_20deg_far.svg'), dpi=150)
# plt.show()


# In[171]:


# extra_trees_cls_on_train_rfecv__test__numbers_by_rmax = \
#     score_by_column(
#         rfecv_selector_on_extra_trees_cls, 
#         learning_data__var_th_X_test[learning_data__simu_shower_track_mask_arr_test], 
#         learning_data__y_test[learning_data__simu_shower_track_mask_arr_test], 
#         calc_cls_numbers,
#         learning_data__event_id_test[learning_data__simu_shower_track_mask_arr_test], 
#         combined_simu_df, 'calc_etruth_trueshower_rmax')


# In[172]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_rmax, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 10, 
#                          xtranslate_func=lambda l: [v/1e6 for v in l],
#                          xlabel = '$R_{max}$ [km]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0,1.2), show=False,
#                          filter_max_yerr=0.9
# #                          show_fill_between=False, show_yerr=True
#                         )
# ax.grid(linestyle='--')
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_rmax_10steps.svg'), dpi=150)
# plt.show()


# In[173]:


# fig, ax, errbr = \
#     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_rmax, 
#                          plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 7, 
#                          xtranslate_func=lambda l: [v/1e6 for v in l],
#                          xlabel = '$R_{max}$ [km]', ylabel = 'Sensitivity', 
#                          calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                          figsize = (5,3), ylim=(0,1.5), show=False)
# ax.grid(linestyle='--')
# fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                          'test_set_sensitivity_function_of_rmax_7steps.svg'), dpi=150)
# plt.show()


# In[ ]:


# -------------------------------------------------------------------------------


# ### Cross-validated recognition performance

# In[218]:


# TODO

# extra_trees_cls_on_train_rfecv_for_crossvalidation_per_trueenergy_results = \
#     cross_val_score_meta_scored(
    
#         extra_trees_cls_on_train_rfecv_for_crossvalidation, 
#         learning_data__rfecv_var_th_X, learning_data__y, 
#         meta_score_func=None,
#         score_func=calc_cls_numbers,
#         cv=3, random_state=32, 
#         train_sample_weight_func=cross_val_calc_weights
    
    
    
    
    
#         extra_trees_cls_on_train_rfecv_for_crossvalidation,
#         learning_data__rfecv_var_th_X, learning_data__y,
#         get_func_score_by_column_using_indices(None, learning_data__event_id, combined_simu_df, 'etruth_trueenergy'),
#         score_func=calc_cls_numbers,
#         cv=sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=10, random_state=123), verbose=1
#     )


# #### Considering all samples from cross-validations (not very correct)
# All results from the cross-validations are joined into a single set (list) and then this set is used to calculate error - this multiplies size of the dataset by number of cross-validation folds.
# 
# Functions are not using any results reduce function - parameter `dict_stats_yerr_reduce` is not set.

# In[219]:


# plt.close('all')
# for xscale in ['linear', 'log']:
#     fig, ax, errbr = \
#         plot_efficiency_stat(extra_trees_cls_on_train_rfecv_for_crossvalidation_per_trueenergy_results, 
#                              plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 20, xscale=xscale,
#                              calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                              xlabel = 'True energy [MeV]', ylabel = 'Sensitivity', 
#                              figsize = (10,6), ylim=(0,1.2), show=False)
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'Cross-validated - Considering all samples from cross-validations (not very correct) - sensitivity - {}.svg'.format(xscale)), dpi=150)
# for xscale in ['linear', 'log']: 
#     fig, ax = plt.subplots()
#     fig, ax, errbr_num_positive = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv_for_crossvalidation_per_trueenergy_results, 
#             plotted_stat='num_positive', num_steps = 20, xscale=xscale,
#             xlabel = 'True energy [MeV]', ylabel = 'Num. positive', label='Num. positive',
#             figsize = (10,6), errorbar_attrs=dict(linestyle='-', color='blue'),
#             ax=ax, show=False)
#     fig, ax, errbr_num_true_positive = \
#         plot_efficiency_stat(
#             extra_trees_cls_on_train_rfecv_for_crossvalidation_per_trueenergy_results, 
#             plotted_stat='num_true_positive', num_steps = 20, xscale=xscale,
#             xlabel='True energy [MeV]', ylabel = 'Num. true positive', label='Num. true positive',
#             figsize=(10,6), errorbar_attrs=dict(linestyle='-', color='green'),
#             ax=ax, show=False)
#     ax.set_ylabel('Num. samples')
#     ax.legend()
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'Cross-validated - Considering all samples from cross-validations (not very correct) - num samples - {}.svg'.format(xscale)), dpi=150)
#     plt.show()


# #### Averaging sensitivities, error is standard deviation
# Parameter `dict_stats_yerr_reduce` is set to compute standard deviation of different cross-validation results.

# In[220]:


# plt.close('all')
# for xscale in ['linear', 'log']:
#     fig, ax, errbr = \
#         plot_efficiency_stat(extra_trees_cls_on_train_rfecv_for_crossvalidation_per_trueenergy_results, 
#                              concat_dicts=False, dict_stats_yerr_reduce='std_y',
#                              plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', 
#                              calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                              num_steps = 20, xscale=xscale,
#                              xlabel = 'True energy [MeV]', ylabel = 'Sensitivity', 
#                              figsize = (10,6), ylim=(0,1.2), show=False)
#     fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                              'Cross-validated - Averaging sensitivities, error is standard deviation - sensitivity - {}.svg'.format(xscale)), dpi=150)
#     plt.show()


# #### Averaging sensitivities, error is min-max range

# In[221]:


# for confidence in [68, 95, 100]:
#     print('Confidence:', confidence)
#     for xscale in ['linear', 'log']:
#         fig, ax, errbr = \
#             plot_efficiency_stat(extra_trees_cls_on_train_rfecv_for_crossvalidation_per_trueenergy_results, 
#                                  concat_dicts=False, dict_stats_yerr_reduce='minmax_y',
#                                  plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_'+str(confidence), 
#                                  num_steps = 20, xscale=xscale,
#                                  calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                                  xlabel = 'True energy [MeV]', ylabel = 'Sensitivity', 
#                                  figsize = (10,6), ylim=(0,1.2), show=False)
#         fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                                  'Cross-validated - Averaging sensitivities, error is min-max range - sensitivity - {}.svg'.format(xscale)), dpi=150)
#         plt.show()


# #### Averaging sensitivities, error is avg_yerr_weighted

# Error should be an average of errors for cross-validated sets

# In[222]:


# for confidence in [68, 95, 100]:
#     print('Confidence:', confidence)
#     for xscale in ['linear', 'log']:
#         fig, ax, errbr = \
#             plot_efficiency_stat(extra_trees_cls_on_train_rfecv_for_crossvalidation_per_trueenergy_results, 
#                                  concat_dicts=False, dict_stats_yerr_reduce='avg_yerr_weighted',
#                                  plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_'+str(confidence), 
#                                  calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
#                                  num_steps = 20, xscale=xscale,
#                                  xlabel = 'True energy [MeV]', ylabel = 'Sensitivity', 
#                                  figsize = (10,6), ylim=(0,1.2))
#         fig.savefig(os.path.join(data_snippets_dir, 'figures', 
#                                  'Cross-validated - Averaging sensitivities, error is avg_yerr_weighted - sensitivity - {}.svg'.format(xscale)), dpi=150)
#         plt.show()


# In[ ]:





# ## TODO
# - investigate sensitivity(background), use bg_mean column
# - investigate sensitivity(shower_max_pos)

# # Flight data classification

# In[223]:


# pipeline_from_trained_models__extr_rfecv_vth__y_pred = \
#     pipeline_from_trained_models__extr_rfecv_vth.predict(
#         unl_flight_df[dataset_condenser_df_columns].dropna().values)


# In[224]:


# num_non_shower = np.count_nonzero(pipeline_from_trained_models__extr_rfecv_vth__y_pred == 0)
# num_shower = np.count_nonzero(pipeline_from_trained_models__extr_rfecv_vth__y_pred == 1)
# tot_entries = len(unl_flight_df[dataset_condenser_df_columns].dropna().values)

# print("Num. non-shower", num_non_shower)
# print("Num. shower", num_shower)
# print("All entries", tot_entries)
# print("-"*30)
# print("Fraction non-shower: {:.3f}".format(num_non_shower/tot_entries))
# print("Fraction shower: {:.3f}".format(num_shower/tot_entries))


# In[ ]:





# In[225]:


# tsne_on_learning_data_60_rfecv_column_names_hexdigest = hashlib.md5((','.join(rfecv_selector_on_extra_trees__column_names__sorted[0:60])).encode()).hexdigest()

# tsne_on_learning_data_60_rfecv_columns_alldata_pathname = \
#     os.path.join(data_snippets_dir, 'tsne_on_learning_data_60_rfecv_columns_alldata_{}.pkl'.format(
#         tsne_on_learning_data_60_rfecv_column_names_hexdigest))
# tsne_on_learning_data_60_rfecv_columns_scaler_alldata_pathname = \
#     os.path.join(data_snippets_dir, 'tsne_on_learning_data_60_rfecv_columns_{}_scale_alldatar.pkl'.format(
#         tsne_on_learning_data_60_rfecv_column_names_hexdigest))

# if refit_tsne_model or not os.path.exists(tsne_on_learning_data_60_rfecv_columns_alldata_pathname):
#     tsne_on_learning_data_60_rfecv_columns_alldata = sklearn.manifold.TSNE(learning_rate=100, verbose=10, n_iter=5000)
#     tsne_on_learning_data_60_rfecv_columns_scaler_alldata = sklearn.preprocessing.StandardScaler()
    
#     learning_data__X__tsne_learning_data_60_rfecv_columns_alldata = \
#         tsne_on_learning_data_60_rfecv_columns_alldata.fit_transform(
#             tsne_on_learning_data_60_rfecv_columns_scaler_alldata.fit_transform(
#                 rfecv_selector_on_extra_trees_cls.transform(learning_data__var_th_X_train).T[   # 232.T[
#                     rfecv_selector_on_extra_trees__column_indices__sorted[0:60]].T              #       232[0:60]].T
#             )
#         )
    
#     if dump_tsne_model: 
#         print(tsne_on_learning_data_60_rfecv_columns_alldata_pathname)
#         joblib.dump(tsne_on_learning_data_60_rfecv_columns_alldata, 
#                     tsne_on_learning_data_60_rfecv_columns_alldata_pathname, compress=1)
        
#         print(tsne_on_learning_data_60_rfecv_columns_scaler_alldata_pathname)
#         joblib.dump(tsne_on_learning_data_60_rfecv_columns_scaler_alldata, 
#                     tsne_on_learning_data_60_rfecv_columns_scaler_alldata_pathname, compress=1)
# else:
#     tsne_on_learning_data_60_rfecv_columns = joblib.load(tsne_on_learning_data_60_rfecv_columns_alldata_pathname)

#     learning_data__X__tsne_learning_data_60_rfecv_columns_alldata = \
#         tsne_on_learning_data_60_rfecv_columns.embedding_


# In[226]:


# joblib.dump(learning_data__X__tsne_learning_data_60_rfecv_columns_alldata, 
#                     tsne_on_learning_data_60_rfecv_columns_alldata_pathname, compress=1)


# In[227]:


# tsne_on_learning_data_60_rfecv_columns_alldata


# In[ ]:




