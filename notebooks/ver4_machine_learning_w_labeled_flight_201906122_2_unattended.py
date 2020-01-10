
# coding: utf-8

# # Training machine learning algorithm to detect showers

# - **NaN columns are removed from the list of analyzed features**
# - ExtraTreesClassifier: min_samples_leaf=10, min_samples_split=50
# - weighted sample
# - orig xy count nonzero, bg_x_y_count_nonzero, bg_count_nonzero, orig_count_nonzero, bg_size removed

# The goal of the method is to classify events into two categories - **shower**, **noise**
# 
# Main motivation behind the method was difficulty to formulate selection conditions by hand, although this was an initial plan behind applying feature extraction procedure on flight data. Defining simple manual rules and using them might still not be impossible. This is also motivation to use feature elimination methods to aid in formulating selection rules. Using decision tree-based methods for classification follows this line of thought. These methods include determining a feature importance and decision trees can be visualized to understand how is the decision being made.
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
# 3. [Feature selection and classification](#Feature-selection-and-classification)
#     - *(This is part that might be changed in the later machine learning procedures)*
#     - Variance thresholding to remove features without any variance.
#     - Univariate feature selection to select smaller but still large enough subset of features (mainly to limit the computational demands). This particular procedure selects 400 features.
#     - Recursive feature elimination with cross-validation - Training and validating *Extremely Randomized Trees model (ExtraTreesClassifier)* on multiple combinations of features aimed to select set of features that provide the best classification accuracy.
#     
#     
# 4. [T-SNE visualization of the dataset](#T-SNE-RFECV-features)
#     - *Currently not finished (TODO)*
#     - Using manifold learning approach to reduce dimensionality of the data (unsupervised) into two dimensions.
#     - There should be possibility to observe clusters in the reduced-dimensionality data and cluster should correlate with known classes of the data.
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

mpl.use('Agg')
mpl.rcParams['figure.dpi'] = 80

import matplotlib.pyplot as plt
import seaborn as sns


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
from utility_functions import key_vals2val_keys


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


# ## Data selection

# In[4]:


data_snippets_dir = 'ver4_machine_learning_w_labeled_flight_201906122_2'
os.makedirs(data_snippets_dir, exist_ok=True)
os.makedirs(os.path.join(data_snippets_dir, 'figures'), exist_ok=True)


# In[5]:


event_processing_cls = event_processing_v4.EventProcessingV4
event_v3_storage_provider_simu = dataset_query_functions_v3.build_event_v3_storage_provider(
    event_storage_provider_config_file=os.path.join(app_base_dir,'config_simu_w_flatmap.ini'), 
    table_names_version='ver4',
    event_storage_class=postgresql_v3_event_storage.PostgreSqlEventV3StorageProvider,
    event_processing_class=event_processing_cls
)

query_functions_simu = dataset_query_functions_v3.Ver3DatasetQueryFunctions(event_v3_storage_provider_simu)


# In[6]:


event_processing_cls = event_processing_v4.EventProcessingV4
event_v3_storage_provider_flight = dataset_query_functions_v3.build_event_v3_storage_provider(
    event_storage_provider_config_file=os.path.join(app_base_dir,'config_w_flatmap.ini'), 
    table_names_version='ver4',
    event_storage_class=postgresql_v3_event_storage.PostgreSqlEventV3StorageProvider,
    event_processing_class=event_processing_cls
)

query_functions_flight = dataset_query_functions_v3.Ver3DatasetQueryFunctions(event_v3_storage_provider_flight)


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

# In[7]:


common_included_columns_re_list = [
  ('^$','source_file_(acquisition|trigger)(_full)?|global_gtu|packet_id|gtu_in_packet|event_id|num_gtu'),
  ('^trg(_box_per_gtu|_pmt_per_gtu|_ec_per_gtu)?$', '^(count_nonzero|min|max|sum|mean)$'),
    
  ('^bg(_x_y)?$','^(mean_gz|mean|max|min|count_nonzero|sum|size)$'),
    
  ('^orig(_x_y)?$','^(count_nonzero|max|mean|mean_gz|sum|size)$'),

  '(proc\d|trg|alt\d)_(x|gtu)_(x|y)_hough_peak_thr[0-3]+_max_clu_major_line_(phi|rho)',
  '(proc\d|trg|alt\d)_(x|gtu)_(x|y)_hough_peak_thr[0-3]+_major_line_(phi|rho)',
  '(proc\d|trg|alt\d)_(x|gtu)_(x|y)_hough_peak_thr[0-3]+_line_clusters_((max_(peak|size|sum|area)_clu_(height|width|size))|count|sizes_max|clu_(widths|heights|areas)_max)',

  ('(proc\d|trg|alt\d)_(gtu|x)_[yx]_clusters',('^(count|sizes_max|sizes_min|clu_areas_max|max_(size|peak)_clu_(width|height|size))$')),  
  ('^proc\d_(x|gtu)_[yx]_hough_peak_thr3','major_line_coord_.*'),
    
]


# #### List of columns of simu data tables used for analysis

# In[8]:


common_columns_for_analysis_dict = query_functions_simu.get_columns_for_classification_dict__by_excluding(
    excluded_columns_re_list=('^.+$',),
    default_excluded_columns_re_list=[],
    included_columns_re_list=common_included_columns_re_list
)

print_columns_dict(common_columns_for_analysis_dict)


# In[9]:


common_df_columns = query_functions_simu.get_dataframe_columns_from_dict(common_columns_for_analysis_dict)


# #### List of columns of flight data tables used for analysis

# In[10]:


flight_columns_for_analysis_dict = query_functions_flight.get_columns_for_classification_dict__by_excluding(
    excluded_columns_re_list=('^.+$',),
    default_excluded_columns_re_list=[],
    included_columns_re_list=common_included_columns_re_list
)

print_columns_dict(flight_columns_for_analysis_dict)


# ### Data selection queries

# #### Simu visible events (base)

# All positive samples for the training are simulated shower tracks with background from the flight data (see notebook ver4_flatmap_visible_events). Events considered as positive samples have to contain track signal (see ver4_test_selection_visualization__simu_signal notebook) and has to be considered as visible (see ver4_flatmap_simu_visible_events notebook). 
# 
# Visibility of the event is decided by a rule that **there should be at least two frames of the event which  contain a signal pixel that is greater or equal to maximum background intensity in the frame**.
# 
# Additionally there is rule that the first trigger of a visible event should be in GTU $42\pm10$.

# In[11]:


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


# In[12]:


simu_df = psql.read_sql(simu_events_selection_query, event_v3_storage_provider_simu.connection)


# In[13]:


simu_df.head()


# #### Simu noise events

# Simu noise events are events that are caused by a trigger well outside of GTU of shower injection into a packet. 
# 
# It is not ideal to use these these events as samples of the dataset because due the way the background of these events is added to the signal. Simply, if there is less packets providing the background than simualated signal tracks then same event might be repeated multiple times in the dataset. 
# Besides repetition of a background packet, background of the simualted event is created by repeating sequence of background frames, thus this might cause multiple events in a same packet. How often this situation happens has not been tested. It is not expected to be very typical.
# 
# Better method of constructing these events would help validity of this analysis.

# In[14]:


# not in the report

current_columns_for_analysis_dict = common_columns_for_analysis_dict

common_select_clause_str, common_tables_list =     query_functions_simu.get_query_clauses__select(current_columns_for_analysis_dict)

# simu_noise_where_clauses_str = ' AND abs(gtu_in_packet-42) >= 20 '

# OPTIMIZATION, ROWS WITH NULL SHOULD BE ALSO ANALYZED 
simu_noise_where_clauses_str = '''
    AND abs(gtu_in_packet-42) >= 20 
    AND {database_schema_name}.event_trg_gtu_y_hough_peak_thr1.major_line_phi IS NOT NULL 
    AND {database_schema_name}.event_trg_gtu_x_hough_peak_thr2.major_line_phi IS NOT NULL 
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


# In[15]:


noise_simu_df = psql.read_sql(noise_simu_events_selection_query, event_v3_storage_provider_simu.connection)


# In[16]:


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
    AND {database_schema_name}.event_trg_gtu_y_hough_peak_thr1.major_line_phi IS NOT NULL 
    AND {database_schema_name}.event_trg_gtu_x_hough_peak_thr2.major_line_phi IS NOT NULL 
    AND {database_schema_name}.event_trg_x_y_hough_peak_thr1.major_line_phi IS NOT NULL
    AND {database_schema_name}.event_orig_x_y.count_nonzero > 256*6
''' 

unl_noise_flight_events_selection_query =     query_functions_flight.get_events_selection_query_plain(
        source_data_type_num=unl_noise_source_data_type_num,
        select_additional=unl_noise_flight_select_clause_str, 
        join_additional=unl_noise_flight_clauses_str,
        where_additional=unl_noise_flight_where_clauses_str,
        order_by='{data_table_name}.event_id', 
        offset=0, 
        limit=20000,                            # intentionally selecting incomplete subset to save memory !!!!!!!!!!!!!
    #     limit=350000,
        base_select='')

# print(unl_noise_flight_events_selection_query)


# In[19]:


unl_noise_flight_df = psql.read_sql(unl_noise_flight_events_selection_query, event_v3_storage_provider_flight.connection)
# flight_df = psql.read_sql(flight_events_selection_query, event_v3_storage_provider_flight.connection)


# In[20]:


unl_noise_flight_df.head()


# #### Flight labeled events

# Important part of the dataset is set of events that were triggered by the hardware. These events are expected to be the hardest to recognize. Previous classification experiments without this set of events significantly limited usefulness of the method because it classified 60% of the flight events sample as a track (see ver4_test_selection_visualization__simu_20181018 notebook).
# Addition of a relatively small set of these events (around 1500) seems to help significantly (see ver4_machine_learning_flight_classification_tsne_cfg3 notebook).
# 
# The manually classified dataset has been created using web classification tool (script web_manual_classification.py). The tool is available at http://eusospb-data.michalvrabel.sk.

# In[21]:


EVENT_CLASSES = {
    'dot': 2,
    'top_left_ec': 5,
    'blob': 12,
    'large_blob': 11,
    'short_single_gtu_track': 7,
    'single_gtu_track': 3,
    'noise': 1,
    'cartesian_line': 4,
    'strong_pmt_edge': 9,
    'few_dots': 6,
    'bg_increased_suddenly': 10,
    'persistent_dot': 14,
    'noise_unspecified': 0,
    'unspecified': 8,
    'shower': 13,
    '2pix_line': 15,
    'bright_blob': 16,
    'blob_and_dots': 17,
    'dot_w_blob_behind': 18,
    'storng_light': 19,
    'sparse_blobs': 20,
    'noise_with_week_dot': 21
}

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


# In[22]:


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
    AND {{database_schema_name}}.event_trg_gtu_y_hough_peak_thr1.major_line_phi IS NOT NULL 
    AND {{database_schema_name}}.event_trg_gtu_x_hough_peak_thr2.major_line_phi IS NOT NULL 
    AND {{database_schema_name}}.event_trg_x_y_hough_peak_thr1.major_line_phi IS NOT NULL
    AND {{database_schema_name}}.event_orig_x_y.count_nonzero > 256*6
    AND {classification_table_cls_column_name} NOT IN ({event_class_shower}, {event_class_unspecified})
    AND {classification_table_last_modification_column_name} < '2019-04-09'
'''.format(
    classification_table_cls_column_name=classification_table_cls_column_name,
    classification_table_last_modification_column_name=classification_table_last_modification_column_name,
    event_class_shower=EVENT_CLASSES['shower'],
    event_class_unspecified=EVENT_CLASSES['unspecified']
)
#TODO
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


# In[23]:


lbl_noise_flight_df = psql.read_sql(lbl_noise_flight_events_selection_query, event_v3_storage_provider_flight.connection)
# lbl_noise_flight_df[classification_df_cls_column_name] 


# In[24]:


len(lbl_noise_flight_df)


# In[25]:


lbl_noise_flight_df.head()


# #### Flight unclassified probable events

# Small subset of flight unclassified events, that were caused by trigger around GTU 42, are selected to be used for basic check of the data reduction capability.

# In[26]:


# not in the report
current_columns_for_analysis_dict = flight_columns_for_analysis_dict

unl_flight_select_clause_str, unl_flight_tables_list =     query_functions_flight.get_query_clauses__select(current_columns_for_analysis_dict)

unl_flight_clauses_str =     query_functions_flight.get_query_clauses__join(unl_flight_tables_list)

unl_flight_source_data_type_num = 1

unl_flight_where_clauses_str = ''' 
    AND abs(gtu_in_packet-42) < 20
    AND {{database_schema_name}}.event_trg_gtu_y_hough_peak_thr1.major_line_phi IS NOT NULL 
    AND {{database_schema_name}}.event_trg_gtu_x_hough_peak_thr2.major_line_phi IS NOT NULL 
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


# In[27]:


unl_flight_df = psql.read_sql(unl_flight_events_selection_query, event_v3_storage_provider_flight.connection)


# In[28]:


unl_flight_df.head()


# #### Classification column for unlabeled filght
# (not in the report)

# In[29]:


unl_flight_df[classification_df_cls_column_name] = EVENT_CLASS_NUMBER_UNLABELED
unl_noise_flight_df[classification_df_cls_column_name] = EVENT_CLASS_NUMBER_UNLABELED_NOISE


# ### Flight datasets in dict
# (not in the report)

# In[30]:


flight_df_dict = {
    'unl_noise_flight_df': unl_noise_flight_df, 
    'lbl_noise_flight_df': lbl_noise_flight_df, 
    'unl_flight_df': unl_flight_df
}


# ### Closing connections
# (not in the report)

# In[31]:


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

# In[32]:


combined_simu_df = pd.concat([simu_df, noise_simu_df])


# In[33]:


# flight_columns_list = list(lbl_noise_flight_df.columns.values)
# combined_flight_df = pd.concat([unl_noise_flight_df[flight_columns_list], lbl_noise_flight_df[flight_columns_list], unl_flight_df[flight_columns_list]])


# In[34]:


print('len(simu_df) =', len(simu_df))
print('len(noise_simu_df) =', len(noise_simu_df))
print('len(combined_simu_df) =', len(combined_simu_df))


# #### $R_{max}$ property of simulated showers

# In[35]:


# 'etruth_trueshowermaxpos_x', 'etruth_trueshowermaxpos_y', 'etruth_trueshowermaxpos_z'
combined_simu_df['calc_etruth_trueshower_rmax'] = np.hypot(combined_simu_df['etruth_trueshowermaxpos_x'], combined_simu_df['etruth_trueshowermaxpos_y'])


# #### Query classification information
# Primary classification based on the original data selection query - original intention of the data selection.

# In[36]:


combined_simu_df['cond_selection_query'] = 'undefined'
combined_simu_df.loc[combined_simu_df['event_id'].isin(simu_df['event_id']), 'cond_selection_query'] = 'simu'
combined_simu_df.loc[combined_simu_df['event_id'].isin(noise_simu_df['event_id']), 'cond_selection_query'] = 'noise'


# In[37]:


if('simu_df' in locals()): del simu_df
if('noise_simu_df' in locals()): del noise_simu_df
# if('unl_noise_flight_df' in locals()): del unl_noise_flight_df
# if('lbl_noise_flight_df' in locals()): del lbl_noise_flight_df
# if('unl_flight_df' in locals()): del unl_flight_df


# #### Simu signal classification information
# Secondary classification is addition of labeled simu signal events.
# The events are loaded from tables prepared in ver4_test_selection_visualization__simu_signal notebook.

# In[38]:


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

# In[39]:


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

# In[40]:


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

# In[41]:


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

# In[42]:


track_underflow_simu_df.sort_values('gtu_in_packet', ascending=False).head()


# ##### Example of track overflow subset

# In[43]:


track_overflow_simu_df.sort_values('gtu_in_packet', ascending=True).head()


# ##### Visualization of a few events

# - Noise entries are sorted by number of simu signal pixles in x-y projection in descending order (`orig_x_y_count_nonzero`, sorted from the most potentially track-like),
# - Track entries are sorted by num frames where maximum signal is greater equal maximum background in acsending order (`num_frames_signals_ge_bg`, from the least visible track events). Non-track-like simu signal might not be necessarly incorrectly labeled entries, just a small portion of a track in signal.
# - Track underflow, track overflow should all contain empty simu signal data. Entries are sorted by GTU in packet in ascending or descending order, respectively.

# In[43]:


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


# ### Adding new features

# #### Rank column
# The principle of a rank column is to combine values of features based on expected or calculated correlation of a column with likeliness of an event being a shower. The lowest value should correspond to the most probable shower track.
# 
# In this case, a short set of hand-picked list is utilized. More preferable way of doing this would be to use columns selected by a machine learning approach that calculates feature importance.
# 
# The provided features are normalized to 0-1 range, 
# optionally, the values of the features are inverted (1-val) and weighted. 
# Finally, the summed value is resulting rank of a record.

# In[44]:


rank_columns = ['proc1_x_y_hough_peak_thr2_line_clusters_count', 'proc1_x_y_hough_peak_thr2_line_clusters_max_peak_clu_width', 
                 'proc1_gtu_y_hough_peak_thr2_line_clusters_max_peak_clu_width', 'proc1_gtu_x_hough_peak_thr2_line_clusters_max_peak_clu_width',
                 'trg_count_nonzero', 'num_gtu']

def add_rank_column_default(data_df):
    add_rank_column(data_df, 
                    rank_columns, 
                    ascending=[True, True, True, True, False, False], 
                    column_weights=[2,2,1,1,1,1],
                    print_max_vals=True, add_norm_columns=True, do_copy=False)


# #### Difference columns
# Expected property of air shower event is that at least in one of the shower projections in time should contain a line with a slope different than 0 or 90 degrees. 
# In an ideal case an actual slope of a line is not important, only important information is slope difference to 0 or 90 degrees.

# In[45]:


# not in the report
area_columns_prefix_list = ('proc1', 'proc2', 'proc3', 'trg')
area_columns_proj_list = ('x_y', 'gtu_x', 'gtu_y')
area_columns_thr_i_list = list(range(1,4))

area_columns_line_types = ['peak', 'sum', 'size']
area_columns_col_name_prefixes = ['{{prefix}}_{{proj}}_hough_peak_thr{{thr_i}}_line_clusters_max_{}_clu_'.format(t) for t in area_columns_line_types]

diff_columns_proc_range = (1,4)
diff_columns_alt_range = (1,1)
diff_columns_thr_range = (1,4)

diff_columns_prefixes =     ['proc{}'.format(i) for i in range(*diff_columns_proc_range)] +     ['alt{}'.format(i) for i in range(*diff_columns_alt_range)]

diff_columns_projs = ['gtu_x', 'gtu_y']
diff_columns_diff_types = ['pi_over_2', '0']

diff_columns_gtu_y_gtu_x_diff_format = '{prefix}_gtu_y_gtu_x_hough_peak_thr{thr_i}_major_line_phi_diff'
diff_columns_proj_diff_format = '{prefix}_{proj}_hough_peak_thr{thr_i}_major_line_phi_diff_{diff_type}'

#

common_extension_columns = ['event_id', 'rank']

for col in rank_columns:
    common_extension_columns.append('norm_' + col)

for prefix in area_columns_prefix_list:
    for proj in area_columns_proj_list:
        for thr_i in area_columns_thr_i_list:
            for col_name_prefix in area_columns_col_name_prefixes:
                common_extension_columns.append(col_name_prefix.format(prefix=prefix, proj=proj, thr_i=thr_i) + 'area')
    
for prefix in diff_columns_prefixes:
    for thr_i in range(*diff_columns_thr_range):
        common_extension_columns.append(diff_columns_gtu_y_gtu_x_diff_format.format(prefix=prefix, thr_i=thr_i))
        for proj in diff_columns_projs:
            for diff_type in diff_columns_diff_types:
                common_extension_columns.append(diff_columns_proj_diff_format.format(prefix=prefix, thr_i=thr_i, proj=proj, diff_type=diff_type))


simu_extension_columns = common_extension_columns
flight_extension_columns = common_extension_columns

# print(common_extension_columns)


# #### Simu dataframe extension columns
# (not in the report)

# In[46]:


simu_event_ids_md5 = hashlib.md5(pickle.dumps(combined_simu_df['event_id'].values, protocol=0)).hexdigest()
simu_extension_columns_md5 = hashlib.md5(','.join(simu_extension_columns).encode()).hexdigest()
extension_columns_combined_simu_pathname = os.path.join(data_snippets_dir, 'extension_columns_simu_pathname_{}_{}.pkl.gz'.format(simu_event_ids_md5, simu_extension_columns_md5))
print(extension_columns_combined_simu_pathname)


# In[47]:


if not os.path.exists(extension_columns_combined_simu_pathname):
    print('Building calculating columns ...')
    print('num_frames_signals_ge_bg bin column ...')
    
    add_bin_column(combined_simu_df, 'num_frames_signals_ge_bg', 5)

    print('  area columns ...')
    
    for attr_prefix_format in area_columns_col_name_prefixes:    
        add_area_columns(combined_simu_df, prefix_list=area_columns_prefix_list, proj_list=area_columns_proj_list, thr_i_list=area_columns_thr_i_list,
                        attr_prefix_format=attr_prefix_format) 

    print('  diff columns ...')
    
    add_diff_columns(combined_simu_df, proc_range=diff_columns_proc_range, alt_range=diff_columns_alt_range, hough_peak_thr_range=diff_columns_thr_range)

    print('  rank column ...')
    
    add_rank_column_default(combined_simu_df)

    print('Saving pickle ...')
        
    combined_simu_df[simu_extension_columns].to_pickle(extension_columns_combined_simu_pathname, 'gzip')
    
else:
    print('Loading...')
    simu_extension_columns_df = pd.read_pickle(extension_columns_combined_simu_pathname, 'gzip')
    print('Merging ...')
    combined_simu_df = pd.merge(combined_simu_df, simu_extension_columns_df, on=['event_id'])
    del simu_extension_columns_df
    
combined_simu_df.head()


# #### Flight dataframe extension columns
# (not in the report)

# In[48]:


# if('unl_noise_flight_df' in locals()): del unl_noise_flight_df
# if('lbl_noise_flight_df' in locals()): del lbl_noise_flight_df
# if('unl_flight_df' in locals()): del unl_flight_df

extension_columns_flight_pathnames = {}

for subset_label, subset_df in         flight_df_dict.items():
    flight_event_ids_md5 = hashlib.md5(pickle.dumps(subset_df['event_id'].values, protocol=0)).hexdigest()
    flight_extension_columns_md5 = hashlib.md5(','.join(flight_extension_columns).encode()).hexdigest()
    extension_columns_flight_pathnames[subset_label] =         os.path.join(data_snippets_dir, 
                     'extension_columns_{}_{}_{}.pkl.gz'.format(
                         subset_label,
                         flight_event_ids_md5, flight_extension_columns_md5))
    print(extension_columns_flight_pathnames[subset_label])  
    
# flight_event_ids_md5 = hashlib.md5(pickle.dumps(combined_flight_df['event_id'].values, protocol=0)).hexdigest()
# flight_extension_columns_md5 = hashlib.md5(','.join(flight_extension_columns).encode()).hexdigest()
# extension_columns_flight_pathname = os.path.join(data_snippets_dir, 'extension_columns_flight_pathname_{}_{}.pkl.gz'.format(flight_event_ids_md5, flight_extension_columns_md5))
# print(extension_columns_flight_pathname)


# In[49]:


for subset_label, extension_columns_flight_pathname in extension_columns_flight_pathnames.items():
    if not os.path.exists(extension_columns_flight_pathname):
        subset_df = flight_df_dict[subset_label]
        
        print('  Building calculating columns ...')

        print('    area columns ...')

        for attr_prefix_format in area_columns_col_name_prefixes:    
            add_area_columns(subset_df, prefix_list=area_columns_prefix_list, proj_list=area_columns_proj_list, thr_i_list=area_columns_thr_i_list,
                            attr_prefix_format=attr_prefix_format) 

        print('    diff columns ...')

        add_diff_columns(subset_df, proc_range=diff_columns_proc_range, alt_range=diff_columns_alt_range, hough_peak_thr_range=diff_columns_thr_range)

        print('    rank column ...')

        add_rank_column_default(subset_df)

        print('  Saving pickle ...')

        subset_df[flight_extension_columns].to_pickle(extension_columns_flight_pathname, 'gzip')

    else:
        print('  Loading ...')
        flight_extension_columns_df = pd.read_pickle(extension_columns_flight_pathname, 'gzip')
        print('  Merging ...')
        flight_df_dict[subset_label] = pd.merge(flight_df_dict[subset_label], flight_extension_columns_df, on=['event_id'])
        del flight_extension_columns_df
    
#     flight_df_dict[subset_label].head()


# ### Number of NaN entries
# Events with NaN values in are currently rejected from showers dataset. 
# However, final decision about rejection is made considering only columns using in ML algorithm.
# Therefore, these numbers are not exactly indicative of the the final number of rejected events - only simu_track and noise_track should be indicative. (TODO requires check)

# #### Number of NaN entries by query and simu signal labels

# In[50]:


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

# In[51]:


for subset_label, subset_df in flight_df_dict.items():
    print('{:50}: {:d}'.format(subset_label, np.count_nonzero(subset_df.isnull().any(axis=1))))


# #### NaN columns
# Columns with a NaN value are either data from Hough transform on projections of triggered pixels - issue is a single pixel in a projection, thus it is impossible to determine orientation of a line. This impacts usable size of the dataset.
# Other source of NaN values are additional information calculated for simulated shower - it is number of frames where number of signal pixels satisfies certain condition. The NaN value is present when there are no signal present in an identified event.

# In[52]:


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


# ```
# trg_gtu_y_hough_peak_thr1_major_line_phi                                                                                 : 2207
# trg_gtu_y_hough_peak_thr1_major_line_rho                                                                                 : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_count                                                                            : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_sizes_max                                                                        : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_clu_widths_max                                                                   : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_clu_heights_max                                                                  : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_clu_areas_max                                                                    : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_max_area_clu_width                                                               : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_max_area_clu_height                                                              : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_max_size_clu_width                                                               : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_max_size_clu_height                                                              : 2207
# trg_gtu_y_hough_peak_thr1_line_clusters_max_sum_clu_width                                                                : 2207
# ...
# trg_gtu_x_hough_peak_thr1_major_line_phi                                                                                 : 2209
# trg_gtu_x_hough_peak_thr1_major_line_rho                                                                                 : 2209
# trg_gtu_x_hough_peak_thr1_line_clusters_count                                                                            : 2209
# trg_gtu_x_hough_peak_thr1_line_clusters_sizes_max                                                                        : 2209
# trg_gtu_x_hough_peak_thr1_line_clusters_clu_widths_max                                                                   : 2209
# trg_gtu_x_hough_peak_thr1_line_clusters_clu_heights_max                                                                  : 2209
# trg_gtu_x_hough_peak_thr1_line_clusters_clu_areas_max                                                                    : 2209
# trg_gtu_x_hough_peak_thr1_line_clusters_max_area_clu_width                                                               : 2209
# trg_gtu_x_hough_peak_thr1_line_clusters_max_area_clu_height                                                              : 2209
# trg_gtu_x_hough_peak_thr1_line_clusters_max_size_clu_width                                                               : 2209
# ...
# trg_x_y_hough_peak_thr1_major_line_phi                                                                                   : 1776
# trg_x_y_hough_peak_thr1_major_line_rho                                                                                   : 1776
# trg_x_y_hough_peak_thr1_line_clusters_count                                                                              : 1776
# trg_x_y_hough_peak_thr1_line_clusters_sizes_max                                                                          : 1776
# trg_x_y_hough_peak_thr1_line_clusters_clu_widths_max                                                                     : 1776
# trg_x_y_hough_peak_thr1_line_clusters_clu_heights_max                                                                    : 1776
# trg_x_y_hough_peak_thr1_line_clusters_clu_areas_max                                                                      : 1776
# trg_x_y_hough_peak_thr1_line_clusters_max_area_clu_width                                                                 : 1776
# trg_x_y_hough_peak_thr1_line_clusters_max_area_clu_height                                                                : 1776
# ...
# 
# num_frames_counts_gt_bg                                                                                                  : 123329
# num_frames_signals_gt_bg                                                                                                 : 123329
# num_frames_signals_ge_bg                                                                                                 : 123329
# ```

# ### Free memory
# (not in the report)

# In[53]:


if 'unclassified_simu_df' in locals(): del unclassified_simu_df
if 'track_simu_df' in locals(): del track_simu_df
if 'noisy_simu_df' in locals(): del noisy_simu_df
if 'simu_signal_track_events_df' in locals(): del simu_signal_track_events_df
if 'simu_signal_noisy_events_df' in locals(): del simu_signal_noisy_events_df


# In[54]:


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

# In[55]:


analyzed_common_df_columns = list(common_df_columns)
for col in [  
        'event_id',
        'source_file_acquisition_full',
        'source_file_trigger_full',
        'source_file_acquisition',
        'source_file_trigger',
        'global_gtu',
        'packet_id',
        'gtu_in_packet',
        'orig_x_y_count_nonzero',
        'bg_x_y_count_nonzero',
        'bg_count_nonzero', 
        'orig_count_nonzero',
        'bg_size'
]:
    analyzed_common_df_columns.remove(col)

# IMPORTANT - NaN columns excluded from the analysis
    
for col in nan_columns.keys():
    if col in analyzed_common_df_columns:
        analyzed_common_df_columns.remove(col)
    
simu_class_column = 'cond_selection_combined'
flight_class_column = classification_df_cls_column_name
    
analyzed_common_df_columns_w_event_id = list(analyzed_common_df_columns) + ['event_id']
analyzed_common_df_columns_w_event_id_simu_class = list(analyzed_common_df_columns_w_event_id) + [simu_class_column]
analyzed_common_df_columns_w_event_id_flight_class = list(analyzed_common_df_columns_w_event_id) + [flight_class_column]


# ### Showers dataset
# Showers dataset consists of processed simulated showers that belong to the **"simu track"** class and potentially flight events classified as an air shower.
# 
# Another potential source in the future might consist set of laser shots from Utah tests.

# In[56]:


def query_simu_track(df):
    return df.query('cond_selection_combined == "simu_track"')

def query_event_class_shower(df):
    return df.query(
        '{classification_df_cls_column_name} == {event_class_shower}'.format(
            classification_df_cls_column_name=classification_df_cls_column_name,
            event_class_shower=EVENT_CLASSES['shower']
        )
    )


# In[57]:


EVENT_CLASS_LABELED_SHOWER_FLIGHT = 2
EVENT_CLASS_SIMU_TRACK = 1


# In[58]:


shower_subset_class_numbers_dict = {
    'lbl_shower_flight_df': EVENT_CLASS_LABELED_SHOWER_FLIGHT,
    'combined_simu_df_shower_track': EVENT_CLASS_SIMU_TRACK
}
shower_subset_priority_order = ['lbl_shower_flight_df', 'combined_simu_df_shower_track']
shower_subsets_list = [
    
    # intentionally doing query first,
    #  unsuitable name of the dict item
    #  expected to be empty
    query_event_class_shower(flight_df_dict['lbl_noise_flight_df']) \
        [analyzed_common_df_columns_w_event_id_flight_class] \
        .dropna(),
    
    query_simu_track(combined_simu_df) \
        [analyzed_common_df_columns_w_event_id] \
        .dropna()
]


# In[59]:


showers_nonan_w_event_id_df = pd.concat(shower_subsets_list)


# Total size of the simualated showers dataset:

# In[60]:


print('len(showers_nonan_w_event_id_df)', len(showers_nonan_w_event_id_df))


# ### Non-showers dataset
# Noise dataset is presently constructed from three subsets, in the follwing priority
# 1. **Classified noise** - *Flight labeled events* excluding classes `shower` and `unspecified`.
# 2. **Unclassified flight** - Dataset of noise of that triggered using configuration with decreased thresholds (bgf=0.5) outside of window of expected cause of the hardware trigger in GTU 40 (Dataset *Flight improbable events* - 20 GTU before or after GTU 42). 
# 3. **Overflow simu** - In principle same as **unclassified flight** but on simu simulation - frames consist of a repeating sequence. The entries should be slightly more different form the **unclassified flight** than **underflow simu**. That's set events should be generally shorter than than the repeated sequence length, on the other hand, **overflow simu** contains some events of containing repetition of the frames sequence (should be verified).
# 3. **Unclassified simu** - In principle same as **unclassified flight** but on simu simulation - **overflow** and **noise noise"** classified events.

# In[61]:


EVENT_CLASS_NUMBER_SIMU_OVERFLOW = 0
EVENT_CLASS_NUMBER_SIMU_NOISE_NOISE = -4
EVENT_CLASS_NUMBER_SIMU_UNDERFLOW = -5


# In[62]:


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

def get_labeled_flight_noise():
    return query_labeled_flight_noise(flight_df_dict['lbl_noise_flight_df'])         [analyzed_common_df_columns_w_event_id_flight_class]         .dropna()

def get_unlabeled_flight_noise():
    return query_unlabeled_flight_noise(flight_df_dict['unl_noise_flight_df'])         [analyzed_common_df_columns_w_event_id_flight_class]         .dropna()

def get_simu_noise_noise():
    return query_simu_noise_noise(combined_simu_df)         [analyzed_common_df_columns_w_event_id_simu_class]         .dropna()

def get_simu_overflow():
    return query_simu_overflow(combined_simu_df)         [analyzed_common_df_columns_w_event_id_simu_class]         .dropna()

def get_simu_underflow():
    return query_simu_underflow(combined_simu_df)         [analyzed_common_df_columns_w_event_id_simu_class]         .dropna()


# Size of the dataset in progressively extended by non-shower data until it as large as shower data dataset. 
# If required number of events is lower than size of a subset, events are randomly sampled from the subset.

# In[63]:


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

non_shower_subsets_list = []
non_shower_subsets_tot_len = 0
for noise_subset_label in noise_subset_priority_order:
    get_non_shower_events_func = noise_subset_df_funcs_dict[noise_subset_label]
    non_shower_subset_df = get_non_shower_events_func()
    new_len = len(non_shower_subset_df) + non_shower_subsets_tot_len
    
    print('Current subset size: {:<7} ; Added {:<30} subset size: {:<7} ; '           'Potentional new dataset size: {:<7} ; Required size: {:<7}'.format(
        non_shower_subsets_tot_len, noise_subset_label, len(non_shower_subset_df),
        new_len, len(showers_nonan_w_event_id_df)
    ))
    
    if new_len > len(showers_nonan_w_event_id_df):
        non_shower_subset_df =             non_shower_subset_df.iloc[
                np.random.randint(0, len(non_shower_subset_df), 
                                  len(showers_nonan_w_event_id_df) - non_shower_subsets_tot_len)
        ]
        
    non_shower_subsets_list.append(non_shower_subset_df)
    non_shower_subsets_tot_len += len(non_shower_subset_df)
    
    if new_len >= len(showers_nonan_w_event_id_df):
        break


# In[64]:


non_showers_nonan_w_event_id_df = pd.concat(non_shower_subsets_list)


# Total number of noise subset required:

# In[65]:


len(non_shower_subsets_list)


# Concatenated noise subsets total size:

# In[66]:


print(len(non_showers_nonan_w_event_id_df))


# ### Export of the datasets into tsv

# Datasets are saved to be usable externally with different algorithms but reproducing these results.

# In[67]:


overwrite_tsv_dump_files = False


# In[68]:


tsv_dump_dir = os.path.join(data_snippets_dir, 'events')
os.makedirs(tsv_dump_dir, exist_ok=True)

# labeled_flight_shower_tsv = os.path.join(tsv_dump_dir, 'labeled_flight_shower.tsv.gz')
simu_track_tsv = os.path.join(tsv_dump_dir, 'simu_track.tsv.gz')
labeled_flight_noise_tsv = os.path.join(tsv_dump_dir, 'labeled_flight_noise.tsv.gz')
unlabeled_flight_noise_tsv = os.path.join(tsv_dump_dir, 'unlabeled_flight_noise.tsv.gz')

simu_overflow_tsv = os.path.join(tsv_dump_dir, 'simu_overflow.tsv.gz')
simu_noise_noise_tsv = os.path.join(tsv_dump_dir, 'simu_noise_noise.tsv.gz')
simu_underflow_tsv = os.path.join(tsv_dump_dir, 'simu_underflow.tsv.gz')

# print(labeled_flight_shower_tsv)
print(simu_track_tsv)
print(labeled_flight_noise_tsv)
print(unlabeled_flight_noise_tsv)

print(simu_overflow_tsv)
print(simu_noise_noise_tsv)
print(simu_underflow_tsv)

# query_event_class_shower(flight_df_dict['lbl_noise_flight_df']).to_csv(labeled_flight_shower_tsv, sep='\t', compression='gzip')
if overwrite_tsv_dump_files or not os.path.exists(simu_track_tsv):
    query_simu_track(combined_simu_df).to_csv(simu_track_tsv, sep='\t', compression='gzip')
if overwrite_tsv_dump_files or not os.path.exists(labeled_flight_noise_tsv):    
    query_labeled_flight_noise(flight_df_dict['lbl_noise_flight_df']).to_csv(labeled_flight_noise_tsv, sep='\t', compression='gzip')
if overwrite_tsv_dump_files or not os.path.exists(unlabeled_flight_noise_tsv):    
    query_unlabeled_flight_noise(flight_df_dict['unl_noise_flight_df']).to_csv(unlabeled_flight_noise_tsv, sep='\t', compression='gzip')

if overwrite_tsv_dump_files or not os.path.exists(simu_underflow_tsv):    
    query_simu_underflow(combined_simu_df).to_csv(simu_underflow_tsv, sep='\t', compression='gzip')
if overwrite_tsv_dump_files or not os.path.exists(simu_overflow_tsv):    
    query_simu_overflow(combined_simu_df).to_csv(simu_overflow_tsv, sep='\t', compression='gzip')
if overwrite_tsv_dump_files or not os.path.exists(simu_noise_noise_tsv):    
    query_simu_noise_noise(combined_simu_df).to_csv(simu_noise_noise_tsv, sep='\t', compression='gzip')


# #### Additional checks
# (not in the report)

# In[67]:


print(np.count_nonzero(combined_simu_df['num_frames_signals_ge_bg'] == combined_simu_df['num_frames_signals_gt_bg']))
print(np.count_nonzero(combined_simu_df['num_frames_signals_ge_bg'] != combined_simu_df['num_frames_signals_gt_bg']))


# In[68]:


combined_simu_df[combined_simu_df['num_frames_signals_ge_bg'].isnull()][['event_id', 'cond_selection_combined', 'num_frames_counts_gt_bg']].head()


# ### Concatenated arrays (np.ndarray)
# (not in the report)

# Transformation of multiple `pandas.DataFrame` objects into concatenated `numpy.ndarray`. 
# Following arrays are created:
# - `learning_data__X` - training data for an algorithm - data
# - `learning_data__y` - training data for an algorithm - labels
# - `learning_data__event_id` - event id of the data in the dataset - important after `test_train_split()`, used to associate predictions with the original events
# - `learning_data__source_class` - source class of the data in the dataset - important after `test_train_split()`, used to associate predictions with the original events, especially to be able to expres accuracy of predictions for a specific source class of data - e.g. label flight noise events

# In[69]:


learning_data__X = np.concatenate([
    showers_nonan_w_event_id_df[analyzed_common_df_columns].values, 
    non_showers_nonan_w_event_id_df[analyzed_common_df_columns].values
])
learning_data__y = np.concatenate([
    np.ones(len(showers_nonan_w_event_id_df)), 
    np.zeros(len(non_showers_nonan_w_event_id_df))
])
learning_data__event_id = np.concatenate([
    showers_nonan_w_event_id_df['event_id'].values, 
    non_showers_nonan_w_event_id_df['event_id'].values
])
learning_data__source_class = np.concatenate([
#     np.ones(len(showers_nonan_w_event_id_df)),
    *[np.ones(len(shower_subset_df)) * shower_subset_class_numbers_dict[shower_subset_label] \
      for shower_subset_df, shower_subset_label in zip(shower_subsets_list, shower_subset_priority_order)],
    *[np.ones(len(non_shower_subset_df)) * noise_subset_class_numbers_dict[noise_subset_label] \
      for non_shower_subset_df, noise_subset_label in zip(non_shower_subsets_list, noise_subset_priority_order)]
])


# In[70]:


print( 2- ( np.count_nonzero(learning_data__y == 0) / np.count_nonzero(learning_data__y != 0)) )


# In[71]:


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
        
learning_data__weights = calc_learning_data_weights(learning_data__y, learning_data__source_class)


# ### Labeled concatenated dataset (pd.DataFrame)
# (not in the report)

# In[72]:


labeled_data_nonan_w_event_id_w_cls_df = pd.concat([showers_nonan_w_event_id_df, non_showers_nonan_w_event_id_df])
labeled_data_nonan_w_event_id_w_cls_df['bin_class'] = learning_data__y
labeled_data_nonan_w_event_id_w_cls_df['class'] = learning_data__source_class

# noise_subset_class_numbers


# In[73]:


labeled_data_nonan_w_event_id_w_cls_df['class'].unique()


# In[74]:


labeled_data_nonan_w_event_id_w_cls_df.head()


# ### Train-test split
# The data are split into training and testing subsets in **60:40** ratio. The data are shuffled before splitting, thus there should not be a significat difference in ratios of source clases of the data within the testing and training sets.

# In[75]:


learning_data__X_train, learning_data__X_test, learning_data__y_train, learning_data__y_test, learning_data__event_id_train, learning_data__event_id_test, learning_data__source_class_train, learning_data__source_class_test =     sklearn.model_selection.train_test_split(
        learning_data__X, 
        learning_data__y, 
        learning_data__event_id,
        learning_data__source_class,
        test_size=.4, 
        random_state=123, 
        shuffle=True)


# In[76]:


learning_data__weights_train = calc_learning_data_weights(learning_data__y_train, learning_data__source_class_train, print_info=True)


# In[77]:


learning_data__weights_test = calc_learning_data_weights(learning_data__y_test, learning_data__source_class_test, print_info=True)


# #### Number of entries in training and testing datasets

# In[78]:


print('learning_data__X       ', len(learning_data__X_train), len(learning_data__X_test))
print('learning_data__y       ', len(learning_data__y_train), len(learning_data__y_test))
print('learning_data__event_id', len(learning_data__event_id_train), len(learning_data__event_id_test))


# #### Number of entries in training and testing datasets by a class (shower, non-shower)

# In[79]:


print('Num. non-shower entries in train', np.count_nonzero(learning_data__y_train == 0))
print('Num. shower entries in train    ', np.count_nonzero(learning_data__y_train == 1))
print('Num. non-shower entries in test ', np.count_nonzero(learning_data__y_test == 0))
print('Num. shower entries in test     ', np.count_nonzero(learning_data__y_test == 1))


# In[80]:


learning_data__simu_shower_track_mask_arr_all =     learning_data__source_class == shower_subset_class_numbers_dict['combined_simu_df_shower_track']
learning_data__simu_shower_track_mask_arr_test =     learning_data__source_class_test == shower_subset_class_numbers_dict['combined_simu_df_shower_track']

learning_data__lbl_noise_flight_mask_arr_all =     learning_data__source_class == noise_subset_class_numbers_dict['lbl_noise_flight_df']
learning_data__lbl_noise_flight_mask_arr_train =     learning_data__source_class_train == noise_subset_class_numbers_dict['lbl_noise_flight_df']
learning_data__lbl_noise_flight_mask_arr_test =     learning_data__source_class_test == noise_subset_class_numbers_dict['lbl_noise_flight_df']


# #### Number of entries in training and testing datasets considering only labeled noise

# In[81]:


print('lbl_noise_flight_df in train', np.count_nonzero(learning_data__lbl_noise_flight_mask_arr_train))
print('lbl_noise_flight_df in test ', np.count_nonzero(learning_data__lbl_noise_flight_mask_arr_test))


# ### Scaling
# Values of the features are scaled using [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) provided by scikit-learn library.
# The scaler in the applied configuration (default) centers values by subtracting a mean value and then scales data to unit standard deviation.
# 
# In this case a new value of a feature is calculated by the equation: $z=(x - u) / s$, where $x$ is an original value, $u$ is a mean value of particular feature values (column of a table), and $s$ is standard deviation of the feature values (column of a table). 
# 
# The training set is used to determine the mean and the standard variation.
# 
# Scaling might not be necessary for all machine learning approaches.

# In[82]:


standard_scaler_on_train = sklearn.preprocessing.StandardScaler()

learning_data__scaled_X_train = standard_scaler_on_train.fit_transform(learning_data__X_train)
learning_data__scaled_X_test = standard_scaler_on_train.transform(learning_data__X_test)

minmax_scaler_on_train = sklearn.preprocessing.MinMaxScaler()

learning_data__minmax_scaled_X_train = minmax_scaler_on_train.fit_transform(learning_data__X_train)
learning_data__minmax_scaled_X_test = minmax_scaler_on_train.transform(learning_data__X_test)


# ## Feature selection and classification

# (This is part that might be changed in the later machine learning procedures)
# 1. Variance thresholding to remove features without any variance.
# 2. Univariate feature selection to select smaller but still large enough subset of features (mainly to limit the computational demands). This particular procedure selects 400 features.
# 3. Recursive feature elimination with cross-validation - Training and validating Extremely Randomized Trees model (ExtraTreesClassifier) on multiple combinations of features aimed to select set of features that provide the best classification accuracy.

# ### Features at the start of the feature selection

# Number of features before the feature selection:

# In[83]:


len(analyzed_common_df_columns)


# In[84]:


analyzed_common_df_columns


# ### Variance thresholding
# Goal of the variance thresholding applied in this work to remove features without any variance, low variance features are preserved.

# #### Application of the variance thresholding

# In[85]:


var_th_selector_on_scaled_train = sklearn.feature_selection.VarianceThreshold(.0)

learning_data__var_th_scaled_X_train = var_th_selector_on_scaled_train.fit_transform(learning_data__scaled_X_train)
learning_data__var_th_scaled_X_test = var_th_selector_on_scaled_train.transform(learning_data__scaled_X_test)


# In[86]:


learning_data__var_th_X_train = var_th_selector_on_scaled_train.fit_transform(learning_data__X_train)
learning_data__var_th_X_test = var_th_selector_on_scaled_train.transform(learning_data__X_test)


# In[87]:


learning_data__var_th_scaled_columns = [n for n, b in                                         zip(analyzed_common_df_columns, var_th_selector_on_scaled_train.get_support())                                         if b]


# #### Result of the variance thresholding

# In[88]:


print('exclued features\t{}'.format(len(analyzed_common_df_columns)-np.count_nonzero(var_th_selector_on_scaled_train.get_support())))

for n, m in zip(analyzed_common_df_columns, var_th_selector_on_scaled_train.get_support()):
    if not m:
        col_values_scaled = learning_data__scaled_X_train[:, analyzed_common_df_columns.index(n)]
        col_values = learning_data__X_train[:, analyzed_common_df_columns.index(n)]

        if len(col_values) > 0:
            print("{:70} var={:5.3f} first value={}".format(
                  n, np.var(col_values_scaled),
                  col_values[0]))

print('-'*100)

print('included features\t{}'.format(np.count_nonzero(var_th_selector_on_scaled_train.get_support())))

for n, m in zip(analyzed_common_df_columns, var_th_selector_on_scaled_train.get_support()):
    if m:
        print("{:70} var={:5.3f}".format(n, np.var(learning_data__scaled_X_train[:, analyzed_common_df_columns.index(n)])))
        


# ##### Removed features

# - `orig_x_y_size`, `bg_x_y_size` - both size features are actually not useful and are just a “side-effect” of the implementation of the feature extraction procedure. Those matrices are always 48x48 cells thus size of the first dimension is always 48.
# - `bg_min` - minimal value of the background pixels is always 0

# ### Univariate (k-best) feature selection
# 
# Univariate feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator. ([Original description - Scikit-learn manual](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection))
# 
# 
# <font color="red">TODO: UPDATE DESCRIPTION</font>
# 
# Main motivation to use univariate feature selection in this analysis is to lower computational requirements of later applied algorithms. This could be potentially removed in the next analysis / training.
# 
# Method in this analysis uses **f_classif** scoring function, older tests have shown simiar results between **f_classif** and **chi2** method, was deemed better but more thorough analysis is needed. Scoring function  **mutual_info_classif** have not been tested. 
# (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif)
# 
# Notebook that visualizes radvis plot for different scoring functions is ver4_test_selection_visualization__simu_20181018.ipynb 

# In[89]:


kbest_f_classif_selector_on_var_th_sc_train = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, k='all')
kbest_f_classif_selector_on_var_th_sc_train.fit(learning_data__var_th_scaled_X_train, learning_data__y_train)

kbest_chi2_selector_on_var_th_sc_train = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k='all')
kbest_chi2_selector_on_var_th_sc_train.fit(learning_data__minmax_scaled_X_train, learning_data__y_train)

learning_data__kbest_var_th_scaled_X_train = kbest_f_classif_selector_on_var_th_sc_train.transform(learning_data__var_th_scaled_X_train)
learning_data__kbest_var_th_scaled_X_test = kbest_f_classif_selector_on_var_th_sc_train.transform(learning_data__var_th_scaled_X_test)


# In[90]:


refit_kbest_mutual_classif = False
kbest_mutual_classif_indices_sample_random_state = 23456
kbest_mutual_classif_indices_sample_size = 40000
mutual_info_classif_random_state = 1234


# In[91]:


kbest_mutual_classif_selector_on_var_th_sc_train_pathname =     os.path.join(data_snippets_dir, 'kbest_kall_mutual_info_classif_{}_{}_{}.pkl'.format(
        kbest_mutual_classif_indices_sample_random_state, kbest_mutual_classif_indices_sample_size,
        mutual_info_classif_random_state
    ))

def mutual_info_classif_w_random_state(X, y):
    return sklearn.feature_selection.mutual_info_classif(X, y, random_state=mutual_info_classif_random_state)

if refit_kbest_mutual_classif or not os.path.exists(kbest_mutual_classif_selector_on_var_th_sc_train_pathname):
        
    sample_indices = np.random.RandomState(kbest_mutual_classif_indices_sample_random_state).randint(
        0, len(learning_data__var_th_scaled_X_train), 
        size=min(kbest_mutual_classif_indices_sample_size, len(learning_data__var_th_scaled_X_train)))
    
    kbest_mutual_classif_selector_on_var_th_sc_train = sklearn.feature_selection.SelectKBest(
        mutual_info_classif_w_random_state, 
        k='all')
    
    print('Fitting ...')
    
    kbest_mutual_classif_selector_on_var_th_sc_train.fit(
        learning_data__var_th_scaled_X_train[sample_indices], learning_data__y_train[sample_indices])
    
    joblib.dump(kbest_mutual_classif_selector_on_var_th_sc_train, 
                kbest_mutual_classif_selector_on_var_th_sc_train_pathname, 
                compress=1)
    
    del sample_indices
    
else:
    print('Loading ...')
    kbest_mutual_classif_selector_on_var_th_sc_train = joblib.load(kbest_mutual_classif_selector_on_var_th_sc_train_pathname)


# In[92]:


learning_data__kbest_f_classif_var_th_sc_columns_sorted_table, learning_data__kbest_chi2_var_th_sc_columns_sorted_table, learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table = (
    pd.DataFrame(
        sorted(
            (t[0:2] for t in zip(
                learning_data__var_th_scaled_columns, 
                kbest_selector_on_var_th_sc_train.scores_,
                kbest_selector_on_var_th_sc_train.get_support()
            ) if t[2] and not np.isnan(t[1]) ), 
            key=lambda x: x[1], reverse=True
        ), 
        columns=['feature', 'score'])
    for kbest_selector_on_var_th_sc_train in
    (kbest_f_classif_selector_on_var_th_sc_train, 
     kbest_chi2_selector_on_var_th_sc_train, 
     kbest_mutual_classif_selector_on_var_th_sc_train,
    )
)


# In[93]:


pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", -1)


# #### f_classif scores

# In[94]:


learning_data__kbest_f_classif_var_th_sc_columns_sorted_table.iloc[:100]


# #### chi2 scores

# In[95]:


learning_data__kbest_chi2_var_th_sc_columns_sorted_table.iloc[:100]


# #### mutual_classif scores

# In[96]:


learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table.iloc[:100]


# In[97]:


pd.reset_option("display.max_rows")
pd.reset_option("display.max_colwidth")


# #### All scores

# In[98]:


learning_data__kbest_var_th_sc_columns_sorted_table =     pd.merge(
        pd.merge(
            learning_data__kbest_f_classif_var_th_sc_columns_sorted_table, 
            learning_data__kbest_chi2_var_th_sc_columns_sorted_table, 
            on=['feature'], suffixes=['_f_classif', '_chi2']).dropna(),
        learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table,
        on=['feature']).dropna()
learning_data__kbest_var_th_sc_columns_sorted_table.rename(columns={'score': 'score_mutual_classif'}, inplace=True)


# In[101]:


learning_data__kbest_var_th_sc_columns_sorted_table


# #### Correlations

# In[102]:


def corrcoef_score_columns(learning_data_sorted_table, score_column='score', score_std_column='score_std'):
    corrcoef_cols = [
        col for col in learning_data_sorted_table.columns.values \
        if score_column in col and score_std_column not in col]

    return learning_data_sorted_table[corrcoef_cols].corr()
    

def print_corrcoef_table(learning_data_sorted_table, score_column='score', score_std_column='score_std', width_extension=2):
        
    corrcoef_cols = [
        col for col in learning_data_sorted_table.columns.values \
        if score_column in col and score_std_column not in col]

    max_W = max([len(col) for col in corrcoef_cols]) + width_extension
    
    corrcoef =         np.corrcoef([learning_data_sorted_table[col] for col in corrcoef_cols])

    sys.stdout.write('{v:<{W}}'.format(v=' ', W=max_W))
    for col in corrcoef_cols:
        sys.stdout.write('{v:<{W}}'.format(v=col, W=len(col)+width_extension))
        
    sys.stdout.write('\n')
    for corrcoef_row, corrcoef_row_name in zip(corrcoef, corrcoef_cols):
        sys.stdout.write('{v:<{W}}'.format(v=corrcoef_row_name, W=max_W))
        for corrcoef_col_val, corrcoef_col_name in zip(corrcoef_row, corrcoef_cols):
            sys.stdout.write('{v:<{W}.3f}'.format(v=corrcoef_col_val, W=len(corrcoef_col_name)+width_extension))
        sys.stdout.write('\n')
    


# In[103]:


corrcoef_score_columns(learning_data__kbest_var_th_sc_columns_sorted_table)


# These are not continuous functions, similarty matrix is better

# euclidean  (or l2)
# 
# manhattan  (or l1)
# 
# cosine
# 
# braycurtis(u, v[, w])	Compute the Bray-Curtis distance between two 1-D arrays.
#                         The Bray curtis distance has a nice property that if all coordinates are postive, its value is between zero and one.
#                         Zero bray curtis represent exact similar coordinate.
#                         
# canberra(u, v[, w])	Compute the Canberra distance between two 1-D arrays.  # weighted Bray-Curtis
# 
# 
# 
# Probably not here:
# 
# scipy.spatial.distance.mahalanobis  The most common use for the Mahalanobis distance is to find multivariate outliers, 
# 

# ###### Feature score plot

# In[103]:


plt.close('all')

fig_common, ax_common = plt.subplots(1)
fig_common.set_size_inches(12,6)
ax_common.set_title('Comparison of function shapes')
ax_common.set_ylabel('Score (normalized)')
ax_common.set_xlabel('Feature number')

plot_lines = []

for (label, axes_data), c in zip((
    ('f_classif', (learning_data__kbest_f_classif_var_th_sc_columns_sorted_table.index,
    learning_data__kbest_f_classif_var_th_sc_columns_sorted_table['score'])),
    ('chi2', (learning_data__kbest_chi2_var_th_sc_columns_sorted_table.dropna().index,
    learning_data__kbest_chi2_var_th_sc_columns_sorted_table.dropna()['score'])),
    ('mutual_classif', (learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table.index,
    learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table['score'])),
), plt.rcParams['axes.prop_cycle']):
    print(label)
    plt.figure(figsize=(12,6))
    plt.title("{}: Feature scores".format(label))
    plt.plot(*axes_data, color=c['color'])
    plt.xlabel('Feature number')
    plt.ylabel('Score')
    plt.savefig(os.path.join(data_snippets_dir, 'figures', 'univariate_feature_score_plot_{}.svg'.format(label)), dpi=150)
    
    plot_lines.append(ax_common.plot(axes_data[0], axes_data[1]/np.max(axes_data[1]), label=label, color=c['color']))
    
ax_common.legend()
fig_common.savefig(os.path.join(data_snippets_dir, 'figures', 'univariate_feature_score_plot_all_methods.svg'), dpi=150)

plt.show()


# In[104]:


learning_data__kbest_var_th_sc_columns_list =     [col for col in learning_data__kbest_var_th_sc_columns_sorted_table.columns.values 
     if 'score' in col and 'score_std' not in col]

for sortby_column in learning_data__kbest_var_th_sc_columns_list:

    print('Features sorted by:', sortby_column)
    
    fig_common, axs = plt.subplots(len(learning_data__kbest_var_th_sc_columns_list), 1, sharex=True)
    fig_common.set_size_inches(14,14)
    # fig_common.suptitle('Comparison of function values')

    for i, (col, c) in enumerate(zip(learning_data__kbest_var_th_sc_columns_list, plt.rcParams['axes.prop_cycle'])):
        df_sorted_by_col = learning_data__kbest_var_th_sc_columns_sorted_table.sort_values(sortby_column, ascending=False)
        col_score_vals = df_sorted_by_col[col]
        axs[i].plot(range(len(df_sorted_by_col)), #df_sorted_by_col.index, 
                    col_score_vals/np.max(col_score_vals),
                    label=col.replace('score_',''), color=c['color'])
        axs[i].legend()
        axs[i].set_ylim(-0.1,1.2)
    axs[i].set_xlabel('Feature number')    
    fig_common.text(0.08, 0.5, 'Score (normalized)', va='center', rotation='vertical')
    fig_common.savefig(os.path.join(data_snippets_dir, 'figures', 
                                    'univariate_features_sorted_by_{}_plot.svg'.format(sortby_column)), dpi=150)
    
    
    # TODO correlation and distance metrics
    
    plt.show()


# ##### Commentary
# 
# The scores of the selected features range from 2,500 to 19,943. The top ten features have similar score. 
# Selected columns support general idea of the feature extraction method giving high score to features extracted by Hough transform. 
# 
# Explanation of a few selected features:
# 
# **alt1_x_y_hough_peak_thr1_line_clusters_clu_widths_max**
# - `alt1_x_y` - alternative thresholding method (yen thresholding method applied to x-y projection of an event)
# - `hough_peak_thr1` - hough transform is thresholded using the first level of thresholds (75% of maximum value in a hough space)
# - `line_clusters` - selecting clusters from the thresholded hough space
# - `clu_widths_max` - maximum cluster width considering clusters in the thresholded hough space
# 
# **proc2_x_y_hough_peak_thr1_line_clusters_count**
# - `proc2` - mean nonzero value + standard deviation * threshold level, in this case threshold level value is 1.0
# - `line_clusters_count` - number of clusters in a hough space
# 
# **alt1_x_y_hough_peak_thr1_line_clusters_max_area_clu_width**
# - `line_clusters_max_area_clu_width` - width of a cluster with the maximum area
# 
# **alt1_x_y_hough_peak_thr1_line_clusters_max_size_clu_width**
# - `line_clusters_max_size_clu_width` - width of a cluster with the <u>maximum count of nonzero pixels</u>
# 
# **alt1_x_y_hough_peak_thr1_line_clusters_max_sum_clu_width**
# - `line_clusters_max_sum_clu_width` - width of a cluster with the maximum sum of the counts
# 
# One of the more surprising selected options are $\rho$ related features - distance of a line from a origin (top left corner of the image), for instance 'alt1_gtu_y_hough_peak_thr3_major_line_rho'. This means that a line in some particular distance has impact on event classification. 
# 
# (TODO more examples)

# #### Free memory

# In[103]:


del minmax_scaler_on_train   # should not be necessary anymore


# ### Model evaluation functions

# #### Confusion matrix based metrics
# 
# Models are evaluated by same set of basic metrics. Postive entries are considered air shower events and negative entries are noise events.
# - confusion matrix - number of correctly and incorrectly classified entries for each class (function `print_confusion_matrix`)
# - accuracy - ratio of correctly classified entries count to all entries count (function `print_accuracy_cls_report`).
# \begin{equation}
# ACC = \frac{TP + TN}{P + N}
# \end{equation}
# - precision (positive predictive value) - ratio of correctly classified positive entries count to a sum of correctly and incorrectly entries counts (function `print_accuracy_cls_report`).
# \begin{equation}
# PPV = \frac{TP}{TP + FP}
# \end{equation}
# - sensitivity (recall, true positive rate) - ratio of correctly classified positive entries count to all positive entries count (function `print_accuracy_cls_report`).
# \begin{equation}
# TPR = \frac{TP}{P}
# \end{equation}
# - specificity (true negative rate) - ratio of correctly classified negative entries count to all negative entries count. This metric is expressed for classified noise events, because the ratio is related to the dataset of flight events (in an ideal case int would very be descriptive of air shower-classified flight events ratio); (function `print_labeled_data_cls_stats` using mask `test=learning_data__lbl_noise_flight_mask_arr_test`). 
# \begin{equation}
# TNR = \frac{TN}{N}
# \end{equation}
# 
# More generalized (*TODO check the word*) model evaluation is to test a model on different data of a same type. Because of limited dataset size, the cross-validation is a possible approach. The aim to check a model for overfitting but not impact size of the training datasets by introducing another disjoint subset - the validation set. 
# 
# Confusion matrix and related metrics are well described on [Wikipedia - Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).
# 
# #### Sensitivity as a function of a simulated attribute
# 
# Purpose of plotting sensitivity as a function of a simulated attribute is useful to understand an impact of that particular attribute of simulated event. This can be evaluated only for simulated air shower events, because naturally for instance there is no primary energy for a noise event.
# 
# Following attributes are *(or at least should be)* reviewed:
# - primary particle energy - the increase of the primary energy should increase the sensitivity
# - background intensity *(presently not implemented)* - the increase of the background should increase the sensitivity 
# - number of frames with signal pixel intensity greater than maximum background *(presently not implemented)* - the increase of the attribute value should increase the sensitivity
# - zenith angle - the increase of the zenith angle should increase the sensitivity because tracks should be longer, on the other hand tracks with a large zenith angle might be more incomplete and might be a more typical type of edge event (TODO verify)
# - azimuth angle - the increase of the azimuth angle should not have any impact on the sensitivity because the method should not prefer particular arrival direction
# 
# This functionality is implemented by the following functions:
# - `score_by_column(...)` - constructs dictionary mapping x->y, where x is a simulation attribute, y is confusion matrix related value
# - `cross_val_score_meta_scored(...)` - cross-validates a model, returns list of results, score function can be provided, mask of samples for scoring can be provided (typically labeled noise events)
# - `plot_efficiency_stat(...)` - creates plot from output of `score_by_column(...)` or `cross_val_score_meta_scored(...)`
# 
# #### Sensitivity  error
# 
# Descriptive power of model classification sensitivity is affected by a size of a used dataset. 
# Presently this number is not expressed for whole dataset estimations. 
# For sensitivity plots it is expressed using a function: 
# 
# \begin{equation}
# \sigma_R = \sqrt{ \frac{1 - \frac{TP}{P} }{TP} + \frac{1}{P} } * \frac{TP}{P}
# \end{equation}
# 
# *(This is the original function as provided by Mario Bertaina (and understood by me), actually it should be possible to transform the equation into a simpler form; the following equation should be checked)*
# 
# \begin{equation}
# \sigma_R = \frac{\sqrt{TP}}{P}
# \end{equation}
# 
# However, as soon as possible, it is planned to use a well documented [binomial proportion confidence interval (link to wikipedia)](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval) instead. The wikipedia page describes several possibilities. All of the approaches seem to be easy to implement.
# 
# Expressing the error gets a little bit more complicated when considering cross-validation, one options are either to sum up number samples in all folds and then calculate the error, or possibly more correct way, calculate the error for each validation and then reduce multiple results into one. The reduction in later presented plots happen either by averaging values or by taking maximal value. *(Or maybe each error bar should have its own error bars?)*
# 
# #### Feature importance
# 
# Feature importance is useful for understanding what is learned by a model - feature importance should support expectations behind the design of the feature extraction procedure. 
# Some of the expectations are:
# - visible tracks should usually have small number of clusters in all projections - *..._clusters_count* columns
# - visible tracks should have precisely defined single line, thus width and height of the most significant cluster in a hough space should be low - *..._clu_width*, *..._clu_height*, *..._clu_area*, *..._clu_size* attributes
# - air shower tracks should appear as a sloped line in GTU-X or GTU-Y projection - *_line_phi* attributes

# In[108]:


from data_analysis_utils_performance import *


# In[109]:


score_masked_using_indices_simu_shower_track_mask_arr_all = get_func_score_masked_using_indices_weigted(learning_data__simu_shower_track_mask_arr_all)
score_masked_using_indices_simu_shower_track_mask_arr_test = get_func_score_masked_using_indices_weigted(learning_data__simu_shower_track_mask_arr_test)

score_masked_using_indices_lbl_noise_flight_mask_arr_all = get_func_score_masked_using_indices(learning_data__lbl_noise_flight_mask_arr_all)

# learning_data__lbl_noise_flight_mask_arr_all


# ### Extra trees classifier on all features

# Extremely randomized trees create diverse set of classifiers by introducing randomness in the classifier construction. 
# 
# Extremely randomized trees is an ensemble classification method. 
# These methods are aimed at making classification by decision trees more stable (resilient to variations in data) and making more optimal decision than a single decision tree constructed using a heuristic algorithm. 
# The prediction of the ensemble is given as the averaged prediction of the individual classifiers.
# 
# In case of the extremely randomized trees, the splitting procedure at a node draws splits for K randomly selected features (attributes), where each split is based on a random cut point (a threshold). Then split with the best score is selected as the splitting rule of that node. The quality of a split is decided by the Gini impurity criterion. 
# The criterion describes probability of predicting two different outputs in two independent observations (based on description at [stackexchange](https://stats.stackexchange.com/questions/308885/a-simple-clear-explanation-of-the-gini-impurity).
# 
# In comparison to Random Forest ensemble method, in which the splitting procedure always selects a cut point  yielding the the best possible score, the Extremely randomized trees usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias ([More at scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html))
# 
# When number of features selected in the splitting procedure is 1, this amounts to building a totally random decision tree. ([More at scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html))
# 
# P. Geurts, D. Ernst., and L. Wehenkel, “Extremely randomized trees”, Machine Learning, 63(1), 3-42, 2006. Available at: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.65.7485&rep=rep1&type=pdf

# In this case the model is trained on 400 features selected by the univariate feature selection. 
# Except number of decision tree estimators that is 128 (parameters n_estimators), the configuration of the ExtraTreesClassifier is the default:
# - split quality criterion is Gini (another option is to use entropy based criterion; parameter criterion), 
# - depth of constructed trees is not limited (parameter max_depth), 
# - minimum number of samples required to split an internal node is by default 2 (parameter min_samples_split), and minimum number of samples in a leaf is 1 (parameter min_samples_leaf),
# - number of leaf nodes is not limited (parameter max_lead_nodes),
# - number of features selected (K in the procedure description and the paper, parameter max_features) in the splitting procedure is by default square root of total number of features (in this particular case 20),
# - node splitting is not conditioned on decrease of he impurity (parameter min_impurity_decrease=0.0),
# - seed of the random number generator is set to 0 (parameter random_state),
# - classes are weighted equally (parameter class_weight), in this case, weighting is not necessary because there is balanced number of shower and noise events.

# In[110]:


X_train = learning_data__var_th_scaled_X_train
y_train = learning_data__y_train
w_train = learning_data__weights_train

X_test = learning_data__var_th_scaled_X_test
y_test = learning_data__y_test

extra_trees_cls_on_train_all_feat = sklearn.ensemble.ExtraTreesClassifier(
    n_estimators=128, random_state=0, verbose=1, class_weight=None, n_jobs=-1
)
extra_trees_cls_on_train_all_feat.fit(X_train,y_train, w_train)

y_train_pred = extra_trees_cls_on_train_all_feat.predict(X_train)
y_test_pred = extra_trees_cls_on_train_all_feat.predict(X_test)

extra_trees_cls_on_train_all_feat__X_train = X_train
extra_trees_cls_on_train_all_feat__y_train = y_train
extra_trees_cls_on_train_all_feat__y_train_pred = y_train_pred
extra_trees_cls_on_train_all_feat__X_test = X_test
extra_trees_cls_on_train_all_feat__y_test = y_test
extra_trees_cls_on_train_all_feat__y_test_pred = y_test_pred


# #### Model performance

# ##### Confusion matrix

# In[111]:


print_confusion_matrix(
    extra_trees_cls_on_train_all_feat__y_test, 
    extra_trees_cls_on_train_all_feat__y_test_pred, )


# ##### Accuracy report (considering all samples of the test set)

# In[112]:


print_accuracy_cls_report(
    extra_trees_cls_on_train_all_feat__y_test, 
    extra_trees_cls_on_train_all_feat__y_test_pred)


# ##### Specificity (fraction of correctly classified labeled noise events)

# In[113]:


print_labeled_data_cls_stats(
    mask_arr_test=learning_data__lbl_noise_flight_mask_arr_test,
    y_test=extra_trees_cls_on_train_all_feat__y_test,
    y_test_pred=extra_trees_cls_on_train_all_feat__y_test_pred);


# ### Extra trees classifier on all features - unscaled feature values

# In[114]:


X_train = learning_data__var_th_X_train
y_train = learning_data__y_train
w_train = learning_data__weights_train

X_test = learning_data__var_th_X_test
y_test = learning_data__y_test

extra_trees_cls_on_train_all_feat_unscaled = sklearn.ensemble.ExtraTreesClassifier(
    n_estimators=128, random_state=0, verbose=1, class_weight=None, n_jobs=-1
)
extra_trees_cls_on_train_all_feat_unscaled.fit(X_train,y_train,w_train)

y_train_pred = extra_trees_cls_on_train_all_feat_unscaled.predict(X_train)
y_test_pred = extra_trees_cls_on_train_all_feat_unscaled.predict(X_test)

extra_trees_cls_on_train_all_feat_unscaled__X_train = X_train
extra_trees_cls_on_train_all_feat_unscaled__y_train = y_train
extra_trees_cls_on_train_all_feat_unscaled__y_train_pred = y_train_pred
extra_trees_cls_on_train_all_feat_unscaled__X_test = X_test
extra_trees_cls_on_train_all_feat_unscaled__y_test = y_test
extra_trees_cls_on_train_all_feat_unscaled__y_test_pred = y_test_pred


# #### Model performance

# ##### Confusion matrix

# In[115]:


print_confusion_matrix(
    extra_trees_cls_on_train_all_feat_unscaled__y_test, 
    extra_trees_cls_on_train_all_feat_unscaled__y_test_pred, )


# ##### Accuracy report (considering all samples of the test set)

# In[116]:


print_accuracy_cls_report(
    extra_trees_cls_on_train_all_feat_unscaled__y_test, 
    extra_trees_cls_on_train_all_feat_unscaled__y_test_pred)


# ##### Specificity (fraction of correctly classified labeled noise events)

# In[117]:


print_labeled_data_cls_stats(
    mask_arr_test=learning_data__lbl_noise_flight_mask_arr_test,
    y_test=extra_trees_cls_on_train_all_feat_unscaled__y_test,
    y_test_pred=extra_trees_cls_on_train_all_feat_unscaled__y_test_pred);


# ### Extra trees classifier on all features - limited min_samples

# In[118]:


X_train = learning_data__var_th_X_train
y_train = learning_data__y_train
w_train = learning_data__weights_train

X_test = learning_data__var_th_X_test
y_test = learning_data__y_test

extra_trees_cls_on_train_all_feat_unscaled = sklearn.ensemble.ExtraTreesClassifier(
    n_estimators=128, random_state=0, verbose=1, class_weight=None, n_jobs=-1,
    min_samples_leaf=10, min_samples_split=50
)
extra_trees_cls_on_train_all_feat_unscaled.fit(X_train,y_train,w_train)

y_train_pred = extra_trees_cls_on_train_all_feat_unscaled.predict(X_train)
y_test_pred = extra_trees_cls_on_train_all_feat_unscaled.predict(X_test)

extra_trees_cls_on_train_all_feat_unscaled__X_train = X_train
extra_trees_cls_on_train_all_feat_unscaled__y_train = y_train
extra_trees_cls_on_train_all_feat_unscaled__y_train_pred = y_train_pred
extra_trees_cls_on_train_all_feat_unscaled__X_test = X_test
extra_trees_cls_on_train_all_feat_unscaled__y_test = y_test
extra_trees_cls_on_train_all_feat_unscaled__y_test_pred = y_test_pred


# #### Model performance

# ##### Confusion matrix

# In[119]:


print_confusion_matrix(
    extra_trees_cls_on_train_all_feat_unscaled__y_test, 
    extra_trees_cls_on_train_all_feat_unscaled__y_test_pred, )


# ##### Accuracy report (considering all samples of the test set)

# In[120]:


print_accuracy_cls_report(
    extra_trees_cls_on_train_all_feat_unscaled__y_test, 
    extra_trees_cls_on_train_all_feat_unscaled__y_test_pred)


# ##### Specificity (fraction of correctly classified labeled noise events)

# In[121]:


print_labeled_data_cls_stats(
    mask_arr_test=learning_data__lbl_noise_flight_mask_arr_test,
    y_test=extra_trees_cls_on_train_all_feat_unscaled__y_test,
    y_test_pred=extra_trees_cls_on_train_all_feat_unscaled__y_test_pred);


# ### Random forest classifier on all features

# In[122]:


X_train = learning_data__var_th_scaled_X_train
y_train = learning_data__y_train
w_train = learning_data__weights_train

X_test = learning_data__var_th_scaled_X_test
y_test = learning_data__y_test

random_forest_cls_on_train_all_feat = sklearn.ensemble.RandomForestClassifier(
    n_estimators=128, random_state=0, verbose=1, class_weight=None, n_jobs=-1)
random_forest_cls_on_train_all_feat.fit(X_train,y_train,w_train)

y_train_pred = random_forest_cls_on_train_all_feat.predict(X_train)
y_test_pred = random_forest_cls_on_train_all_feat.predict(X_test)

# random_forest_cls_on_train_all_feat__X_train = X_train
# random_forest_cls_on_train_all_feat__y_train = y_train
# random_forest_cls_on_train_all_feat__y_train_pred = y_train_pred
# random_forest_cls_on_train_all_feat__X_test = X_test
random_forest_cls_on_train_all_feat__y_test = y_test
random_forest_cls_on_train_all_feat__y_test_pred = y_test_pred


# #### Model performance

# ##### Confusion matrix

# In[123]:


print_confusion_matrix(
    random_forest_cls_on_train_all_feat__y_test, 
    random_forest_cls_on_train_all_feat__y_test_pred, )


# ##### Accuracy report (considering all samples of the test set)

# In[124]:


print_accuracy_cls_report(
    random_forest_cls_on_train_all_feat__y_test, 
    random_forest_cls_on_train_all_feat__y_test_pred)


# ##### Specificity (fraction of correctly classified labeled noise events)

# In[125]:


print_labeled_data_cls_stats(
    mask_arr_test=learning_data__lbl_noise_flight_mask_arr_test,
    y_test=random_forest_cls_on_train_all_feat__y_test,
    y_test_pred=random_forest_cls_on_train_all_feat__y_test_pred);


# ### Adaboost classifier on all features

# In[126]:


refit_adaboost_cls_on_train_all_feat = False
adaboost_cls_on_train_all_feat_random_state = 0


# In[127]:


adaboost_cls_on_train_all_feat_pathname =     os.path.join(data_snippets_dir, 
                 'adaboost_cls_on_train_all_feat_{}.pkl'.format(adaboost_cls_on_train_all_feat_random_state))

X_train = learning_data__var_th_scaled_X_train
y_train = learning_data__y_train
w_train = learning_data__weights_train

X_test = learning_data__var_th_scaled_X_test
y_test = learning_data__y_test

if refit_adaboost_cls_on_train_all_feat or not os.path.exists(adaboost_cls_on_train_all_feat_pathname):
    
    adaboost_cls_on_train_all_feat = sklearn.ensemble.AdaBoostClassifier(n_estimators=128, random_state=adaboost_cls_on_train_all_feat_random_state)
    
    print('Fitting ...')
    
    adaboost_cls_on_train_all_feat.fit(X_train,y_train,w_train)
    
    joblib.dump(adaboost_cls_on_train_all_feat, 
                adaboost_cls_on_train_all_feat_pathname, 
                compress=1)
    
else:
    print('Loading ...')
    adaboost_cls_on_train_all_feat = joblib.load(adaboost_cls_on_train_all_feat_pathname)

y_train_pred = adaboost_cls_on_train_all_feat.predict(X_train)
y_test_pred = adaboost_cls_on_train_all_feat.predict(X_test)
    
# adaboost_cls_on_train_all_feat__X_train = X_train
# adaboost_cls_on_train_all_feat__y_train = y_train
# adaboost_cls_on_train_all_feat__y_train_pred = y_train_pred
# adaboost_cls_on_train_all_feat__X_test = X_test
adaboost_cls_on_train_all_feat__y_test = y_test
adaboost_cls_on_train_all_feat__y_test_pred = y_test_pred    


# #### Model performance

# ##### Confusion matrix

# In[128]:


print_confusion_matrix(
    adaboost_cls_on_train_all_feat__y_test, 
    adaboost_cls_on_train_all_feat__y_test_pred, )


# ##### Accuracy report (considering all samples of the test set)

# In[129]:


print_accuracy_cls_report(
    adaboost_cls_on_train_all_feat__y_test, 
    adaboost_cls_on_train_all_feat__y_test_pred)


# ##### Specificity (fraction of correctly classified labeled noise events)

# In[130]:


print_labeled_data_cls_stats(
    mask_arr_test=learning_data__lbl_noise_flight_mask_arr_test,
    y_test=adaboost_cls_on_train_all_feat__y_test,
    y_test_pred=adaboost_cls_on_train_all_feat__y_test_pred);


# ### Feature importance in classifiers
# The feature importance describes how much does a feature value affect the sample classification by a model.
# Features used by this model have received the following rank: 

# In[131]:


def make_feature_importance_table_df(classifier, learning_data_columns,
                                     feature_importances_attr='feature_importances_',
                                     estimators_attr='estimators_', estimator_feature_importances_attr='feature_importances_',
                                     feature_column='feature', score_column='score', score_std_column='score_std',
                                     do_std=True):
    importances = getattr(classifier, feature_importances_attr)

    column_data = [learning_data_columns, importances]
    column_names = [feature_column, score_column]
    
    if do_std:
        std = np.std([getattr(est, estimator_feature_importances_attr) for est in getattr(classifier, estimators_attr)], axis=0)
        column_data.append(std)
        column_names.append(score_std_column)
        
    learning_data_columns_sorted_table = pd.DataFrame(
        sorted((t for t in zip(*column_data) if not np.isnan(t[1])), key=lambda x: x[1], reverse=True), 
        columns=column_names)
    
    return learning_data_columns_sorted_table

def merge_feature_score_table_dfs(base_sorted_table_df, extension_table_df, score_label_suffix, 
                              feature_column='feature', score_column='score', score_std_column='score_std'):
    merged_sorted_table_df = pd.merge(base_sorted_table_df, extension_table_df, on=[feature_column]).dropna()
    columns_rename_dict = {score_column: score_column + '_' + score_label_suffix}
    if score_std_column in merged_sorted_table_df.columns.values:
        columns_rename_dict[score_std_column] = score_std_column + '_' + score_label_suffix
    merged_sorted_table_df.rename(columns=columns_rename_dict, inplace=True)
    return merged_sorted_table_df


# In[132]:


learning_data__extr_tr_var_th_sc_columns_sorted_table =     make_feature_importance_table_df(extra_trees_cls_on_train_all_feat, learning_data__var_th_scaled_columns)

learning_data__all_var_th_sc_columns_sorted_table = merge_feature_score_table_dfs(
    learning_data__kbest_var_th_sc_columns_sorted_table,
    learning_data__extr_tr_var_th_sc_columns_sorted_table,
    'extr_tr'
)

learning_data__rndfrst_var_th_sc_columns_sorted_table =     make_feature_importance_table_df(random_forest_cls_on_train_all_feat, learning_data__var_th_scaled_columns)

learning_data__all_var_th_sc_columns_sorted_table = merge_feature_score_table_dfs(
    learning_data__all_var_th_sc_columns_sorted_table,
    learning_data__rndfrst_var_th_sc_columns_sorted_table,
    'rndfrst'
)

learning_data__adabst_var_th_sc_columns_sorted_table =     make_feature_importance_table_df(adaboost_cls_on_train_all_feat, learning_data__var_th_scaled_columns)

learning_data__all_var_th_sc_columns_sorted_table = merge_feature_score_table_dfs(
    learning_data__all_var_th_sc_columns_sorted_table,
    learning_data__adabst_var_th_sc_columns_sorted_table,
    'adabst'
)


# In[134]:


learning_data__all_var_th_sc_columns_sorted_table


# #### Comparison of scores as function of feature number

# In[132]:


fig_common, ax_common = plt.subplots(1)
fig_common.set_size_inches(12,6)
ax_common.set_title('Comparison of function shapes')
ax_common.set_ylabel('Score (normalized)')
ax_common.set_xlabel('Feature number')

plot_lines = []

for (label, table_df), c in zip((
    ('Univariate - f_classif', learning_data__kbest_f_classif_var_th_sc_columns_sorted_table),
    ('Univariate - chi2', learning_data__kbest_chi2_var_th_sc_columns_sorted_table.dropna()),
    ('Univariate - mutual_info_classif', learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table),
    ('Extra trees classifier', learning_data__extr_tr_var_th_sc_columns_sorted_table),
    ('Random forest classifier', learning_data__rndfrst_var_th_sc_columns_sorted_table),
    ('Adaboost classifier', learning_data__adabst_var_th_sc_columns_sorted_table),
), plt.rcParams['axes.prop_cycle']):
    if not label.startswith('Univariate'):
        plt.figure(figsize=(12,6))
        plt.title("{}: Feature scores".format(label))
        plt.plot(table_df.index, table_df['score'], color=c['color'])
        plt.xlabel('Feature number')
        plt.ylabel('Score')
        plt.savefig(os.path.join(data_snippets_dir, 'figures', 'feature_importance_methods_feature_score_plot_{}.svg'.format(label)), dpi=150)
    
    plot_lines.append(ax_common.plot(table_df.index, table_df['score']/np.max(table_df['score']), label=label, color=c['color']))
        
ax_common.legend()
fig_common.savefig(os.path.join(data_snippets_dir, 'figures', 'feature_importance_methods_feature_score_plot_all_methods.svg'), dpi=150)
plt.show()


# #### Correlation between feature scoring approaches

# In[133]:


learning_data__all_var_th_sc_columns_list =     [col for col in learning_data__all_var_th_sc_columns_sorted_table.columns.values 
     if 'score' in col and 'score_std' not in col]


for sortby_column in learning_data__all_var_th_sc_columns_list:

    print('Features sorted by:', sortby_column)
    
    fig_common, axs = plt.subplots(len(learning_data__all_var_th_sc_columns_list), 1, sharex=True)
    fig_common.set_size_inches(14,19)
    # fig_common.suptitle('Comparison of function values')

    for i, (col, c) in enumerate(zip(learning_data__all_var_th_sc_columns_list, plt.rcParams['axes.prop_cycle'])):
        df_sorted_by_col = learning_data__all_var_th_sc_columns_sorted_table.sort_values(sortby_column, ascending=False)
        col_score_vals = df_sorted_by_col[col]
        axs[i].plot(range(len(df_sorted_by_col)), #df_sorted_by_col.index, 
                    col_score_vals/np.max(col_score_vals),
                    label=col.replace('score_',''), color=c['color'])
        axs[i].legend()
        axs[i].set_ylim(-0.1,1.2)
    axs[i].set_xlabel('Feature number')    
    fig_common.text(0.08, 0.5, 'Score (normalized)', va='center', rotation='vertical')
    fig_common.savefig(os.path.join(data_snippets_dir, 'figures', 
                                    'feature_importance_methods_sorted_by_{}_plot.svg'.format(sortby_column)), dpi=150)
    plt.show()


# In[134]:


corrcoef_score_columns(learning_data__all_var_th_sc_columns_sorted_table)


# TODO distances

# In[135]:


plt.close('all')
corr = corrcoef_score_columns(learning_data__all_var_th_sc_columns_sorted_table)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, square=True, cmap='inferno', annot=True, cbar_kws={'label': 'correlation coeff.'})
plt.gcf().set_size_inches(8,6)
plt.xticks(rotation=45)
plt.savefig(os.path.join(data_snippets_dir, 'figures', 'feature_importance_methods_correlation_coeffs.svg'), dpi=150)
plt.show()


# #### Importances of interesting features

# In[135]:


for feature_name_pattern in [
    'per_gtu', 'gtu_[xy]_hough_peak', 'x_y_clusters', 'gtu_[xy]_clusters', '^trg', '^alt', '^proc',
    'gtu_[xy].+_phi'
]:
    print(feature_name_pattern)
    for label, table_df in (
#         ('f_classif', learning_data__kbest_f_classif_var_th_sc_columns_sorted_table),
#         ('chi2', learning_data__kbest_chi2_var_th_sc_columns_sorted_table.dropna()),
#         ('mutual_classif', learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table),
        ('Univariate - f_classif', learning_data__kbest_f_classif_var_th_sc_columns_sorted_table),
        ('Univariate - chi2', learning_data__kbest_chi2_var_th_sc_columns_sorted_table.dropna()),
        ('Univariate - mutual_info_classif', learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table),
        ('Extra trees classifier', learning_data__extr_tr_var_th_sc_columns_sorted_table),
        ('Random forest classifier', learning_data__rndfrst_var_th_sc_columns_sorted_table),
        ('Adaboost classifier', learning_data__adabst_var_th_sc_columns_sorted_table),
    ):
        print(' ',label)
        iii = 0
        
        indices = []
        
        for ii, r in table_df.iterrows():
            if re.search(feature_name_pattern, r['feature']):
                if iii < 10:
                    print('    {:10} {}'.format(ii, r['feature']))
                    iii += 1
                indices.append(ii)
                
        plt.hist(indices, bins=len(table_df)//20)
        plt.show()
        
        print('-'*60)
    print('='*60)


# ### Recursive Feature Elimination

# The recursive feature elimination is done to see an impact of different number of features on classification accuracy. Ideally, a small number of features should be sufficient to do satisfactory estimation of an event class.
# 
# The procedure selects N best features for classification. It starts with training a model with all features and then iteratively (not recursively because [the scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) is actually iterative) removes the least important features and retrains the model until the desired number of features is reached. More details are available at [the scikit-learn documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html).

# In[136]:


overwrite_existing_rfe_model_files = False
refit_existing_rfe_model_files = False


# In[137]:


rfe_extra_trees_params = dict(
    n_estimators=128, random_state=0, verbose=0, class_weight=None, n_jobs=6,
    min_samples_leaf=10, min_samples_split=50)


# #### Fitting

# In[138]:


# previous inefficient code 

# X_train = learning_data__var_th_X_train
# y_train = learning_data__y_train

# X_test = learning_data__var_th_X_test
# y_test = learning_data__y_test

# # X_train = learning_data__k400best_var_th_X_train
# # y_train = learning_data__y_train

# # X_test = learning_data__k400best_var_th_X_test
# # y_test = learning_data__y_test

# rfe_selector_on_extra_trees_cls_dict = {}

# for n_features_to_select in range(10,200+1,10):
#     print('RFE: n_features_to_select={:d}'.format(n_features_to_select))
#     print('-' * 30)
#     rfe_selector_on_extra_trees_cls__model_plk_pathname = os.path.join(data_snippets_dir, 'rfe_selector_{:d}feat_on_extra_trees_cls.pkl'.format(n_features_to_select))
#     if not os.path.exists(rfe_selector_on_extra_trees_cls__model_plk_pathname) or refit_existing_rfe_model_files:
#         print('Fitting...')
#         extra_trees_cls_on_train_128_est = sklearn.ensemble.ExtraTreesClassifier(**rfe_extra_trees_params)
#         rfe_selector_on_extra_trees_cls = sklearn.feature_selection.RFE(extra_trees_cls_on_train_128_est, 
#                                                                         n_features_to_select=n_features_to_select, step=1, verbose=1)
#         rfe_selector_on_extra_trees_cls.fit(X_train, y_train)
        
#         if overwrite_existing_rfe_model_files or not os.path.exists(rfe_selector_on_extra_trees_cls__model_plk_pathname):
#             print('Saving...')
#             print(rfe_selector_on_extra_trees_cls__model_plk_pathname)
#             joblib.dump(rfe_selector_on_extra_trees_cls, 
#                         rfe_selector_on_extra_trees_cls__model_plk_pathname, 
#                         compress=1)
#     else:
#         print('Loading...')
#         rfe_selector_on_extra_trees_cls = sklearn.externals.joblib.load(rfe_selector_on_extra_trees_cls__model_plk_pathname)
    
#     rfe_selector_on_extra_trees_cls_dict[n_features_to_select] = rfe_selector_on_extra_trees_cls
    
#     print('=' * 30)


# In[139]:


# code mosty taken from the original scikt-learn implementation

from collections.abc import Iterable as collections_Iterable
from sklearn.base import clone as sklearn_base_clone
from sklearn.utils import safe_sqr as sklearn_utils_safe_sqr

def fit_multiple_rfe(X, y, estimator,
                     n_features_to_select=None, step=1,
                     verbose=0,
                     step_score=None, single_scores_list=True, 
                     fit_params={}):
    # Parameter step_score controls the calculation of self.scores_
    # step_score is not exposed to users
    # and is used when implementing RFECV
    # self.scores_ will not be calculated when calling _fit through fit
    
    X, y = sklearn.utils.validation.check_X_y(X, y, "csc")
    # Initialization
    n_features = X.shape[1]
    
    
    if isinstance(n_features_to_select, collections_Iterable):
        n_features_to_select_list = n_features_to_select
        min_n_features_to_select = min(n_features_to_select_list)
    else:
        if n_features_to_select is None:
            n_features_to_select = n_features // 2                                          #
        n_features_to_select_list = [n_features_to_select]
        min_n_features_to_select = n_features_to_select
        
        
    if 0.0 < step < 1.0:
        step = int(max(1, step * n_features))
    else:
        step = int(step)
    if step <= 0:
        raise ValueError("Step must be >0")

    support_ = np.ones(n_features, dtype=np.bool)
    ranking_ = np.ones(n_features, dtype=np.int)

    scores_ = []
    n_features_list = []

    output_rfe_instances = {}

    sum_support = np.sum(support_)
    
    orig_estimator = estimator

    # Elimination
    while sum_support > min_n_features_to_select:
        # Remaining features
        features = np.arange(n_features)[support_]

        # Rank the remaining features
        estimator = sklearn_base_clone(orig_estimator)                                                      
        if verbose > 0:                                                                   #
            print("Fitting estimator with %d features." % np.sum(support_))

        estimator.fit(X[:, features], y, **fit_params)

        # Get coefs
        if hasattr(estimator, 'coef_'):
            coefs = estimator.coef_
        else:
            coefs = getattr(estimator, 'feature_importances_', None)
        if coefs is None:
            raise RuntimeError('The classifier does not expose '
                                '"coef_" or "feature_importances_" '
                                'attributes')

        # Get ranks
        if coefs.ndim > 1:
            ranks = np.argsort(sklearn_utils_safe_sqr(coefs).sum(axis=0))
        else:
            ranks = np.argsort(sklearn_utils_safe_sqr(coefs))

        # for sparse case ranks is matrix
        ranks = np.ravel(ranks)

        # Eliminate the worse features
        threshold = min(step, np.sum(support_) - min_n_features_to_select)

        # Compute step score on the previous selection iteration
        # because 'estimator' must use features
        # that have not been eliminated yet
        if step_score:
            scores_.append(step_score(estimator, features))
            n_features_list.append(sum_support)
            
        support_[features[ranks][:threshold]] = False
        ranking_[np.logical_not(support_)] += 1

        # Main modification of the original scikit-learn algroitm
        sum_support = np.sum(support_)
        
        if sum_support in n_features_to_select_list or sum_support <= min_n_features_to_select:
        
            # Set final attributes
            features = np.arange(n_features)[support_]  
            estimator_ = sklearn_base_clone(orig_estimator)               #
            estimator_.fit(X[:, features], y, **fit_params)           #
            
            n_features_ = support_.sum()                                #

#             print('sum_support', sum_support)
#             print('n_features_', n_features_)
            
            rfe_instance = sklearn.feature_selection.RFE(estimator_, n_features_, step, verbose)
            rfe_instance.estimator_ = estimator_
            
            if step_score and not single_scores_list:
                rfe_instance.scores_ = list(scores_)
                
                rfe_instance.n_features_list = list(n_features_list)
                
            # Compute step score when only n_features_to_select features left
#             if step_score:
                rfe_instance.scores_.append(step_score(estimator_, features))        #
                n_features_list.append(n_features_)
                
            #support_ = support_                                         #
            #ranking_ = ranking_                                        #
            rfe_instance.n_features_ = n_features_
            rfe_instance.support_ = np.copy(support_)
            rfe_instance.ranking_ = np.copy(ranking_)
            
            output_rfe_instances[sum_support] = rfe_instance
        
    if not single_scores_list:
        return output_rfe_instances
    else:
        return output_rfe_instances, (scores_, n_features_list) 
            


# In[ ]:





# In[140]:


def score_estimator_by_datasets_dict(estimator, feature_indices, datasets_dict, score_func_dict):
    score_dict = {}
    for dataset_label, (X, y, w, masks_dict) in datasets_dict.items():
        score_dict[dataset_label] = {}
        for mask_label, mask in masks_dict.items():
            y_masked = y[mask]
            y_masked_pred = estimator.predict(X[mask][:, feature_indices])
            weights_masked = w[mask]
            score_dict[dataset_label][mask_label] = score_func_dict[mask_label](y_masked, y_masked_pred, weights_masked)
    return score_dict

def score_rfe_step_weighted(estimator, feature_indices):
    return score_estimator_by_datasets_dict(
        estimator, feature_indices, 
        {
            'test': (learning_data__var_th_X_test, learning_data__y_test, learning_data__weights_test, {
                'all': np.ones_like(learning_data__y_test, dtype=np.bool),
                'lbl_noise_flight': learning_data__lbl_noise_flight_mask_arr_test
            }),
            'train': (learning_data__var_th_X_train, learning_data__y_train, learning_data__weights_train, {
                'all': np.ones_like(learning_data__y_train, dtype=np.bool),
                'lbl_noise_flight': learning_data__lbl_noise_flight_mask_arr_train
            }),
        },
        {
            'all': calc_cls_numbers_stats_weighted_experimental,
            'lbl_noise_flight': lambda y_test, y_test_pred, sample_weight=None: calc_cls_numbers_stats(y_test, y_test_pred)
        }
    )

# Read by:
# lod_to_dol(lod_to_dol([score_rfe_step(est, ind), ... ]))


# In[141]:


n_features_to_select_list = [1]
n_features_to_select_list += list(range(5,30,5))
n_features_to_select_list += list(range(30,200+1,10))

hashlib.md5(str(n_features_to_select_list).encode()).hexdigest()


# In[142]:


X_train = learning_data__var_th_X_train
y_train = learning_data__y_train
w_train = learning_data__weights_train

X_test = learning_data__var_th_X_test
y_test = learning_data__y_test

# X_train = learning_data__k400best_var_th_X_train
# y_train = learning_data__y_train

# X_test = learning_data__k400best_var_th_X_test
# y_test = learning_data__y_test

rfe_selector_on_extra_trees_cls_dict = {}

missing_n_features_to_select_list = []

rfe_selectors_on_extra_tree_scores = None


n_features_to_select_list = list(range(10,200+1,10)) # in backup_rfe_selectors.20190225 directory

n_features_to_select_list = [1]
n_features_to_select_list += list(range(5,30,5))
n_features_to_select_list += list(range(30,250+1,10))

n_features_to_select_list_hexdigest_str = hashlib.md5(str(n_features_to_select_list).encode()).hexdigest()

rfe_selector_on_extra_trees_cls__scores_list_pathname =     os.path.join(data_snippets_dir, 'rfe_selectors_on_extra_trees_cls__{}_feat__scores_list.pkl'.format(
        n_features_to_select_list_hexdigest_str))

for n_features_to_select in n_features_to_select_list:
    
    rfe_selector_on_extra_trees_cls__model_pkl_pathname =         os.path.join(data_snippets_dir, 'rfe_selector_{:d}feat_on_extra_trees_cls.pkl'.format(n_features_to_select))
    if not os.path.exists(rfe_selector_on_extra_trees_cls__scores_list_pathname) or             not os.path.exists(rfe_selector_on_extra_trees_cls__model_pkl_pathname) or             refit_existing_rfe_model_files:
        missing_n_features_to_select_list.append(n_features_to_select)
    else:
        print('Loading ', rfe_selector_on_extra_trees_cls__model_pkl_pathname)
        rfe_selector_on_extra_trees_cls = sklearn.externals.joblib.load(rfe_selector_on_extra_trees_cls__model_pkl_pathname)
        rfe_selector_on_extra_trees_cls_dict[n_features_to_select] = rfe_selector_on_extra_trees_cls
        print('Loading ', rfe_selector_on_extra_trees_cls__scores_list_pathname)
        rfe_selectors_on_extra_tree_scores = joblib.load(rfe_selector_on_extra_trees_cls__scores_list_pathname)

print('Missing models of numbers of features from RFE:')
print(missing_n_features_to_select_list)
        
if len(missing_n_features_to_select_list) > 0:
    print('Fitting...')
    rfe_selectors_dict, rfe_selectors_on_extra_tree_scores =         fit_multiple_rfe(
            X_train, y_train, 
            estimator=sklearn.ensemble.ExtraTreesClassifier(**rfe_extra_trees_params),
            n_features_to_select=missing_n_features_to_select_list, step=1,
            verbose=1,
            step_score=score_rfe_step_weighted, single_scores_list=True, fit_params={'sample_weight': w_train})  
    
    if rfe_selectors_dict is None or rfe_selectors_on_extra_tree_scores is None:
        print('Unexpected result of rfe function', file=sys.stderr)
    else:
        if (not os.path.exists(rfe_selector_on_extra_trees_cls__scores_list_pathname) or overwrite_existing_rfe_model_files):
            if not np.array_equal(sorted(missing_n_features_to_select_list), sorted(n_features_to_select_list)):
                print('missing_n_features_to_select_list, n_features_to_select_list arrays do not have same items - not saving scores list', file=sys.stderr)
            else:
                joblib.dump(rfe_selectors_on_extra_tree_scores, rfe_selector_on_extra_trees_cls__scores_list_pathname)

        for n_features, rfe_selector in rfe_selectors_dict.items():
            rfe_selector_on_extra_trees_cls__model_pkl_pathname =                 os.path.join(data_snippets_dir, 'rfe_selector_{:d}feat_on_extra_trees_cls.pkl'.format(n_features))
            if overwrite_existing_rfe_model_files or not os.path.exists(rfe_selector_on_extra_trees_cls__model_pkl_pathname):
                print('Saving ', rfe_selector_on_extra_trees_cls__model_pkl_pathname)
                joblib.dump(rfe_selector, 
                            rfe_selector_on_extra_trees_cls__model_pkl_pathname, 
                            compress=1)
            rfe_selector_on_extra_trees_cls_dict[n_features] = rfe_selector


# In[129]:


plt.figure(figsize=(9,5))
for i, (dataset_mask, prop_name, label, alpha, color) in enumerate([
        ('all', 'accuracy', 'Accuracy (all test)', 1, 'C0'),
        ('all', 'specificity', 'Specificity (all test)', 1, 'C1'),
        ('all', 'sensitivity', 'Sensitivity (all test)', 1, 'C2'),
        ('all', 'negative_predictive_value', 'Negative predictive value (all test)', 1, 'C3'),
        ('all', 'precision', 'Precision (all test)', 1, 'C4'),
        ('lbl_noise_flight', 'specificity', 'Specificity (labeled flight noise in test)', 0.5, 'C5'),
]):
    plt.plot(
        rfe_selectors_on_extra_tree_scores[1], 
        [ds['test'][dataset_mask][prop_name] for ds in rfe_selectors_on_extra_tree_scores[0]],
        marker='', linestyle='-', zorder=90-i, alpha=alpha, label=label, 
    )
plt.legend()
plt.xlim(20+max(rfe_selectors_on_extra_tree_scores[1]), 0-20)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 'rfe_selectors_on_extra_tree_scores_plot_auto_color.svg'), dpi=150)
plt.show()


# In[164]:


plt.figure(figsize=(9,5))
for i, (dataset_mask, prop_name, label, alpha, color) in enumerate([
        ('lbl_noise_flight', 'specificity', 'Specificity (labeled flight noise in test)', 0.7, 'red'),
        ('all', 'accuracy', 'Accuracy (all test)', 1, 'black'),
        ('all', 'specificity', 'Specificity (all test)', 1, 'green'),
        ('all', 'sensitivity', 'Sensitivity (all test)', 1, 'blue'),
        ('all', 'negative_predictive_value', 'Negative predictive value (all test)', 1, 'cyan'),
        ('all', 'precision', 'Precision (all test)', 1, 'magenta'),
]):
    plt.plot(
        rfe_selectors_on_extra_tree_scores[1], 
        [ds['test'][dataset_mask][prop_name] for ds in rfe_selectors_on_extra_tree_scores[0]],
        marker='', linestyle='-', zorder=90-i, alpha=alpha, label=label, color=color
    )
plt.legend()
plt.xlim(20+max(rfe_selectors_on_extra_tree_scores[1]), 0-20)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 'rfe_selectors_on_extra_tree_scores_plot.svg'), dpi=150)
plt.show()


# In[165]:


plt.figure(figsize=(9,5))
for i, (dataset_mask, prop_name, label, alpha, color) in enumerate([
        ('lbl_noise_flight', 'specificity', 'Specificity (labeled flight noise in test)', 0.7, 'red'),
        ('all', 'accuracy', 'Accuracy (all test)', 1, 'black'),
        ('all', 'specificity', 'Specificity (all test)', 1, 'green'),
        ('all', 'precision', 'Precision (all test)', 1, 'magenta'),
]):
    plt.plot(
        rfe_selectors_on_extra_tree_scores[1], 
        [ds['test'][dataset_mask][prop_name] for ds in rfe_selectors_on_extra_tree_scores[0]],
        marker='', linestyle='-', zorder=90-i, alpha=alpha, label=label, color=color
    )
plt.legend()
plt.xlim(20+max(rfe_selectors_on_extra_tree_scores[1]), 0-20)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'rfe_selectors_on_extra_tree_scores_plot__specificity_accuracy_precision.svg'), dpi=150)
plt.show()


# In[172]:


plt.figure(figsize=(9,5))
for i, (dataset_mask, prop_name, label, alpha, color) in enumerate([
        ('lbl_noise_flight', 'specificity', 'Specificity (labeled flight noise in test)', 0.7, 'gray'),
        ('all', 'accuracy', 'Accuracy (all test)', 1, 'green'),
#         ('all', 'w_accuracy', 'Weighted accuracy (all test)', 1, 'blue'), # weighted in simple terms, confusion matrix is weighted
        ('all', 'balanced_accuracy', 'Balanced accuracy (all test)', 1, 'red'), 
]):
    plt.plot(
        rfe_selectors_on_extra_tree_scores[1], 
        [ds['test'][dataset_mask][prop_name] for ds in rfe_selectors_on_extra_tree_scores[0]],
        marker='', linestyle='-', zorder=90-i, alpha=alpha, label=label, color=color
    )
plt.legend()
plt.xlim(20+max(rfe_selectors_on_extra_tree_scores[1]), 0-20)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'rfe_selectors_on_extra_tree_scores_plot__accuracy_variations.svg'), dpi=150)
plt.show()


# In[173]:


plt.figure(figsize=(9,5))
for i, (dataset_mask, prop_name, label, alpha, color) in enumerate([
        ('all', 'accuracy', 'Accuracy (all test)', 1, 'green'),
#         ('all', 'w_accuracy', 'Weighted accuracy (all test)', 1, 'blue'), # weighted in simple terms, confusion matrix is weighted
        ('all', 'balanced_accuracy', 'Balanced accuracy (all test)', 1, 'red'), 
]):
    plt.plot(
        rfe_selectors_on_extra_tree_scores[1], 
        [ds['test'][dataset_mask][prop_name] for ds in rfe_selectors_on_extra_tree_scores[0]],
        marker='', linestyle='-', zorder=90-i, alpha=alpha, label=label, color=color
    )
plt.legend()
plt.xlim(20+max(rfe_selectors_on_extra_tree_scores[1]), 0-20)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'rfe_selectors_on_extra_tree_scores_plot__accuracy_variations_no_specificity.svg'), dpi=150)
plt.show()


# In[174]:


plt.figure(figsize=(9,5))
for i, (dataset_mask, prop_name, label, alpha, color) in enumerate([
        ('all', 'w_accuracy', 'Weighted accuracy (all test)', 1, 'blue'), # weighted in simple terms, confusion matrix is weighted
        ('all', 'balanced_accuracy', 'Balanced accuracy (all test)', 1, 'green'), 
]):
    plt.plot(
        rfe_selectors_on_extra_tree_scores[1], 
        [ds['test'][dataset_mask][prop_name] for ds in rfe_selectors_on_extra_tree_scores[0]],
        marker='', linestyle='-', zorder=90-i, alpha=alpha, label=label, color=color
    )
plt.legend()
plt.xlim(20+max(rfe_selectors_on_extra_tree_scores[1]), 0-20)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'rfe_selectors_on_extra_tree_scores_plot__accuracy_balanced_vs_w.svg'), dpi=150)
plt.show()


# In[167]:


len(rfe_selectors_on_extra_tree_scores[0])


# RFECV does this but with mutiple folds of train-test data.

# #### Performance on test set

# In[175]:


rfe_test_n_features_list = []
rfe_test_numbers_stats_list = []
rfe_test_lbl_noise_numbers_stats_list = []

for n_features_to_select, rfe_selector_on_extra_trees_cls in sorted(rfe_selector_on_extra_trees_cls_dict.items()):
    print('RFE: n_features_to_select={:d}'.format(n_features_to_select))
    print('-' * 60)
    
    y_test = learning_data__y_test
    y_test_pred = rfe_selector_on_extra_trees_cls.predict(learning_data__var_th_X_test)

    print('Confusion matrix:')
    print_confusion_matrix(y_test, y_test_pred, )

    print_accuracy_cls_report(y_test, y_test_pred)

    print('Labeled flight noise (lbl_noise_flight):')
    
    print_labeled_data_cls_stats(
        mask_arr_test=learning_data__lbl_noise_flight_mask_arr_test,
        y_test=y_test, y_test_pred=y_test_pred)
    
    rfe_test_n_features_list.append(n_features_to_select)
    rfe_test_numbers_stats_list.append(calc_cls_numbers_stats_weighted_experimental(y_test, y_test_pred))
    rfe_test_lbl_noise_numbers_stats_list.append(calc_cls_numbers_stats(y_test[learning_data__lbl_noise_flight_mask_arr_test], y_test_pred[learning_data__lbl_noise_flight_mask_arr_test]))
    
#     rfe_numbers_stats_list.append(calc_cls_numbers_stats(y_test, y_test_pred))
    
    print('=' * 60)


# In[177]:


plt.figure(figsize=(9,5))
for i, (dataset, prop_name, label, alpha, color) in enumerate([
        (rfe_test_lbl_noise_numbers_stats_list, 'specificity', 'Specificity (labeled flight noise in test)', 0.7, 'red'),
        (rfe_test_numbers_stats_list, 'specificity', 'Specificity (all test)', 1, 'green'),
        (rfe_test_numbers_stats_list, 'sensitivity', 'Sensitivity (all test)', 1, 'blue'),
        (rfe_test_numbers_stats_list, 'precision', 'Precision (all test)', 1, 'magenta'),
#         (rfe_test_numbers_stats_list, 'balanced_accuracy', 'Balanced accuracy (all test)', 1, 'black'),
]):
    plt.plot(
        rfe_test_n_features_list, 
        [entry[prop_name] for entry in dataset],
        marker='.', linestyle='-', zorder=90-i, alpha=alpha, label=label, color=color
    )
plt.legend()
plt.xlim(5+max(rfe_test_n_features_list), 0-5)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'rfe_selectors_on_extra_tree_scores_plot__feat_from_200_to_1__specificity_sensitivity_precision.svg'), dpi=150)
plt.show()


# In[151]:


plt.figure(figsize=(9,5))
for i, (dataset, prop_name, label, alpha, color) in enumerate([
        (rfe_test_lbl_noise_numbers_stats_list, 'specificity', 'Specificity (labeled flight noise in test)', 0.7, 'red'),
        (rfe_test_numbers_stats_list, 'accuracy', 'Accuracy (all test)', 1, 'black'),
        (rfe_test_numbers_stats_list, 'specificity', 'Specificity (all test)', 1, 'green'),
        (rfe_test_numbers_stats_list, 'sensitivity', 'Sensitivity (all test)', 1, 'blue'),
        (rfe_test_numbers_stats_list, 'negative_predictive_value', 'Negative predictive value (all test)', 1, 'cyan'),
        (rfe_test_numbers_stats_list, 'precision', 'Precision (all test)', 1, 'magenta'),
]):
    plt.plot(
        rfe_test_n_features_list, 
        [entry[prop_name] for entry in dataset],
        marker='.', linestyle='-', zorder=90-i, alpha=alpha, label=label, color=color
    )
plt.legend()
plt.xlim(5+max(rfe_test_n_features_list), 0-5)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'rfe_selectors_on_extra_tree_scores_plot__feat_from_200_to_1.svg'), dpi=150)
plt.show()


# In[178]:


plt.figure(figsize=(9,5))
for i, (dataset, prop_name, label, alpha, color) in enumerate([
        (rfe_test_lbl_noise_numbers_stats_list, 'specificity', 'Specificity (labeled flight noise in test)', 0.7, 'gray'),
        (rfe_test_numbers_stats_list, 'accuracy', 'Accuracy (all test)', 1, 'black'),
        (rfe_test_numbers_stats_list, 'w_accuracy', 'Weighted accuracy (all test)', 1, 'blue'), # weighted in simple terms, confusion matrix is weighted
        (rfe_test_numbers_stats_list, 'balanced_accuracy', 'Balanced accuracy (all test)', 1, 'green'), 
]):
    plt.plot(
        rfe_test_n_features_list, 
        [entry[prop_name] for entry in dataset],
        marker='.', linestyle='-', zorder=90-i, alpha=alpha, label=label, color=color
    )
    
plt.legend()
plt.xlim(5+max(rfe_test_n_features_list), 0-5)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'rfe_selectors_on_extra_tree_scores_plot__accuracy_variations__feat_from_200_to_1.svg'), dpi=150)
plt.show()


# #### Cross-validation

# In[179]:


crossval_n_features_list = []
mean_accuracy_list = []
std_accuracy_list = []
mean_accuracy_lbl_noise_list = []
std_accuracy_lbl_noise_list = []

def cross_val_calc_weights(indices, learning_data__y=learning_data__y, learning_data__source_class=learning_data__source_class): 
    return calc_learning_data_weights(learning_data__y[indices], learning_data__source_class[indices], print_info=False)

for n_features_to_select, rfe_selector_on_extra_trees_cls in sorted(rfe_selector_on_extra_trees_cls_dict.items()):
    print('RFE: n_features_to_select={:d}'.format(n_features_to_select))
    print('-' * 30)
        
    extra_trees_cls_on_train_rfe_for_crossvalidation = sklearn.ensemble.ExtraTreesClassifier(**rfe_extra_trees_params)

    # not entirely correct, feature selection should be also included in crossvalidation training

    learning_data__rfe_var_th_X =         rfe_selector_on_extra_trees_cls.transform(
            var_th_selector_on_scaled_train.transform(
                learning_data__X
            )
        )

    extra_trees_cls_on_train_rfe_for_crossvalidation_crv_results =         cross_val_score_meta_scored(
            extra_trees_cls_on_train_rfe_for_crossvalidation, 
            learning_data__rfe_var_th_X, learning_data__y, 
            meta_score_func=None,
            score_func=balanced_accuracy_score,
            cv=3, random_state=32, 
            train_sample_weight_func=cross_val_calc_weights
    )
    
    print('Cross-validation accuracy:', extra_trees_cls_on_train_rfe_for_crossvalidation_crv_results)
    print('Mean accuracy:            ', np.mean(extra_trees_cls_on_train_rfe_for_crossvalidation_crv_results))
    print('Std accuracy:             ', np.std(extra_trees_cls_on_train_rfe_for_crossvalidation_crv_results))
    
    mean_accuracy_list.append(np.mean(extra_trees_cls_on_train_rfe_for_crossvalidation_crv_results))
    std_accuracy_list.append(np.std(extra_trees_cls_on_train_rfe_for_crossvalidation_crv_results))
    
    extra_trees_cls_on_train_rfe_for_crossvalidation_lbl_noise_flight_crv_results =     cross_val_score_meta_scored(
        extra_trees_cls_on_train_rfe_for_crossvalidation,
        learning_data__rfe_var_th_X, learning_data__y, 
        cv=3, random_state=32, verbose=1,
        meta_score_func=score_masked_using_indices_lbl_noise_flight_mask_arr_all,
        train_sample_weight_func=cross_val_calc_weights
    )
        
    print('Cross-validation accuracy (lbl_noise):', extra_trees_cls_on_train_rfe_for_crossvalidation_lbl_noise_flight_crv_results)
    print('Mean accuracy (lbl_noise):            ', np.mean(extra_trees_cls_on_train_rfe_for_crossvalidation_lbl_noise_flight_crv_results))
    print('Std accuracy (lbl_noise):             ', np.std(extra_trees_cls_on_train_rfe_for_crossvalidation_lbl_noise_flight_crv_results))
    
    mean_accuracy_lbl_noise_list.append(np.mean(extra_trees_cls_on_train_rfe_for_crossvalidation_lbl_noise_flight_crv_results))
    std_accuracy_lbl_noise_list.append(np.std(extra_trees_cls_on_train_rfe_for_crossvalidation_lbl_noise_flight_crv_results))
    
    crossval_n_features_list.append(n_features_to_select)
    
    print('=' * 30)
    


# In[180]:


plt.figure(figsize=(9,5))
for i, (dataset_x, dataset_y, yerr, label, alpha, color) in enumerate([
        (rfe_test_n_features_list, [d['accuracy'] for d in rfe_test_numbers_stats_list], None, 'Accuracy (single test set)', 1, 'black'),
        (crossval_n_features_list, mean_accuracy_lbl_noise_list, std_accuracy_lbl_noise_list, 'Cross-validated specificity (labeled flight noise)',  0.7, 'gray'),
        (crossval_n_features_list, mean_accuracy_list, std_accuracy_list, 'Cross-validated balanced accuracy', 1, 'blue'), # weighted in simple terms, confusion matrix is weighted
]):
    plt.errorbar(
        dataset_x, 
        dataset_y,
        yerr=yerr,
        marker='.', linestyle='-', zorder=90-i, alpha=alpha, label=label, 
        color=color
    )
    
plt.legend()
plt.xlim(5+max(rfe_test_n_features_list), 0-5)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'rfe_selectors_on_extra_tree_scores_plot__crossvalidated_accuracy_specificity__feat_from_250_to_1.svg'), dpi=150)
plt.show()


# In[181]:


plt.figure(figsize=(9,5))
for i, (dataset_x, dataset_y, yerr, label, alpha, color) in enumerate([
        (rfe_test_n_features_list, [d['specificity'] for d in rfe_test_numbers_stats_list], None, 'Specificity (labeled flight noise in test set)', 1, 'black'),
        (crossval_n_features_list, mean_accuracy_lbl_noise_list, std_accuracy_lbl_noise_list, 'Cross-validated specificity (labeled flight noise)',  0.7, 'gray'),
]):
    plt.errorbar(
        dataset_x, 
        dataset_y,
        yerr=yerr,
        marker='.', linestyle='-', zorder=90-i, alpha=alpha, label=label, 
        color=color
    )
    
plt.legend()
plt.xlim(5+max(rfe_test_n_features_list), 0-5)
plt.xlabel('Number of features')
plt.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'rfe_selectors_on_extra_tree_scores_plot__specificity__feat_from_250_to_1.svg'), dpi=150)
plt.show()


# ### Recursive Feature Elimination with cross-validation (RFECV)

# The recursive feature elimination with cross-validation should find the optimal list of features achieving the best classification accuracy. 
# Using standard recursive feature elimination method, this method tests all possible feature counts by decreasing number of features from whole feature set to a single feature (this can be limited by a configuration). Then this test is performed several times with different train-test splits of data. Then the final number of features is decided by a maximum summed score of models fitted with same number of features but different train-test splits. Then the final run of the recursive feature elimination algorithm with number of features from the previous step finds the most optimal features.
# 
# Model trained in this step will be saved and used for classification of triggered (bgf=1) and flight events. Also sensitivity of this model will be later plotted as a function of various simulated attributes (energy, arrival angles).

# In[143]:


import rfecv_weighted


# In[144]:


load_rfecv_from_file = True
overwrite_existing_rfecv_model_files = False
recreate_rfecv_pipline_object = False


# In[145]:


extra_trees_cls_on_train_rfecv__model_pkl_pathname =     os.path.join(data_snippets_dir, 'extra_trees_cls_on_train_rfecv.pkl')
pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname =     os.path.join(data_snippets_dir, 'pipeline_from_trained_models__extr_rfecv_vth.pkl')


# In[146]:


# standard_scaler_on_train, var_th_selector_on_scaled_train, 
# k400best_f_classif_selector_on_var_th_sc_train, rfecv_selector_on_extra_trees_cls,
# extra_trees_cls_on_train_kbest400_128_est


# In[147]:


# not in the report
load_rfecv_from_file and os.path.exists(pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname)


# #### Training or loading the model

# In[148]:


len(learning_data__var_th_X_train)


# In[149]:


def cross_val_calc_weights(indices, learning_data__y=learning_data__y_train, learning_data__source_class_train=learning_data__source_class_train): 
    return calc_learning_data_weights(learning_data__y_train[indices], learning_data__source_class_train[indices], print_info=False)

# alternatively only learning_data__weights_train could ve used


pipeline_from_trained_models__extr_rfecv_vth_scale  = None

if load_rfecv_from_file and os.path.exists(pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname):
    print('Loading existing model pipeline:', pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname)
    pipeline_from_trained_models__extr_rfecv_vth =         joblib.load(pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname)
    
    rfecv_selector_on_extra_trees_cls = pipeline_from_trained_models__extr_rfecv_vth.steps[-1][1]
    extra_trees_cls_on_train_rfecv = rfecv_selector_on_extra_trees_cls.estimator_
    
else:
    X_train = learning_data__var_th_X_train
    y_train = learning_data__y_train

    X_test = learning_data__var_th_X_test
    y_test = learning_data__y_test

    extra_trees_cls_for_rfecv = sklearn.ensemble.ExtraTreesClassifier(**rfe_extra_trees_params)

    rfecv_selector_on_extra_trees_cls = rfecv_weighted.RFECV(
        estimator=extra_trees_cls_for_rfecv, 
        step=1, verbose=1, n_jobs=-1)

    rfecv_selector_on_extra_trees_cls.fit(X_train, y_train, sample_weight_func=cross_val_calc_weights)


# In[150]:


rfecv_selector_on_extra_trees__column_names =     [n for n, m in zip(learning_data__var_th_scaled_columns, rfecv_selector_on_extra_trees_cls.get_support()) if m]


# In[151]:


print(len(rfecv_selector_on_extra_trees__column_names))


# #### Saving the model

# In[152]:


if overwrite_existing_rfecv_model_files or not os.path.exists(extra_trees_cls_on_train_rfecv__model_pkl_pathname):
    print(extra_trees_cls_on_train_rfecv__model_pkl_pathname)
    joblib.dump(rfecv_selector_on_extra_trees_cls.estimator_, 
                extra_trees_cls_on_train_rfecv__model_pkl_pathname, 
                compress=1)
else:
    print("(Already exists)\t{}".format(extra_trees_cls_on_train_rfecv__model_pkl_pathname))


# In[140]:


rfecv_selector_on_extra_trees_cls.estimator_


# ##### Whole pipeline

# In[141]:


if pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname is None or         not os.path.exists(pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname) or         recreate_rfecv_pipline_object:
    pipeline_from_trained_models__extr_rfecv_vth =         sklearn.pipeline.make_pipeline(
            #standard_scaler_on_train, 
            var_th_selector_on_scaled_train, 
#             k400best_f_classif_selector_on_var_th_sc_train, 
            rfecv_selector_on_extra_trees_cls
        )
    print(pipeline_from_trained_models__extr_rfecv_vth)


# In[142]:


if pipeline_from_trained_models__extr_rfecv_vth is not None and         (overwrite_existing_rfecv_model_files or          not os.path.exists(pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname)):
    print('Saving pipeline', pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname)
    joblib.dump(pipeline_from_trained_models__extr_rfecv_vth, 
                pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname,
                compress=1)
else:
    print("(Already exists)\t{}".format(pipeline_from_trained_models__extr_rfecv_vth__model_pkl_pathname))


# ##### Creating a list of selected columns and a scaler for the columns

# In[143]:


columns_list_file_pathname = os.path.join(data_snippets_dir, 'rfecv_selector_on_extra_trees__column_names.txt')
if not os.path.exists(columns_list_file_pathname) or recreate_rfecv_pipline_object:
    print('Writing columns list into:', columns_list_file_pathname)
    with open(columns_list_file_pathname, 'w') as columns_list_file:
        for col_name in rfecv_selector_on_extra_trees__column_names:
            if col_name in common_df_columns:
                print(col_name, file=columns_list_file)
else:
    print('(Already exists) ', columns_list_file_pathname)


# In[144]:


special_columns_list_file_pathname = os.path.join(data_snippets_dir, 'rfecv_selector_on_extra_trees__column_names__special.txt')
if not os.path.exists(special_columns_list_file_pathname) or recreate_rfecv_pipline_object:
    print('Writing special columns list into:', special_columns_list_file_pathname)
    with open(special_columns_list_file_pathname, 'w') as special_columns_list_file:
        for col_name in rfecv_selector_on_extra_trees__column_names:
            if col_name not in common_df_columns:
                print(col_name, file=special_columns_list_file)
else:
    print('(Already exists) ', special_columns_list_file_pathname)


# In[145]:


np.count_nonzero(~np.isin(rfecv_selector_on_extra_trees__column_names, common_df_columns))


# ##### StandardScaler on columns selected by RFECV

# In[198]:


# column_indices_rfecv_in_analyzed_common = [analyzed_common_df_columns.index(attr) for attr in rfecv_selector_on_extra_trees__column_names]


# In[199]:


# standard_scaler_on_train_rfecv_columns = sklearn.preprocessing.StandardScaler()
# standard_scaler_on_train_rfecv_columns.fit(learning_data__X_train.T[column_indices_rfecv_in_analyzed_common].T)


# ##### Consistency check
# (not in the report)
# 
# Check if both scalers produce same resuls.

# ###### StandardScaler on columns selected by RFECV

# In[200]:


# np.sum(np.round(
#     standard_scaler_on_train_rfecv_columns.transform(
#         learning_data__X_train.T[column_indices_rfecv_in_analyzed_common].T)*10000))


# ###### StandardScaler on all columns

# In[201]:


# np.sum(np.round(
#     rfecv_selector_on_extra_trees_cls.transform(
#         k400best_f_classif_selector_on_var_th_sc_train.transform(
#             var_th_selector_on_scaled_train.transform(
#                 standard_scaler_on_train.transform(learning_data__X_train)
#             )
#         )
#     )*10000))


# ##### Check of columns count of the scaler 

# In[202]:


# len(standard_scaler_on_train_rfecv_columns.scale_)


# ##### Saving the scaler into a file
# All data should be scaled with it before classification.

# In[203]:


# standard_scaler_on_train_rfecv_columns_pathname = os.path.join(data_snippets_dir, 'standard_scaler_on_train_rfecv_columns.pkl')

# if standard_scaler_on_train_rfecv_columns is not None and \
#         (overwrite_existing_rfecv_model_files or \
#          not os.path.exists(standard_scaler_on_train_rfecv_columns_pathname)):
#     print(standard_scaler_on_train_rfecv_columns_pathname)
#     joblib.dump(standard_scaler_on_train_rfecv_columns, 
#                 standard_scaler_on_train_rfecv_columns_pathname,
#                 compress=1)
# else:
#     print("(Already exists)\t{}".format(standard_scaler_on_train_rfecv_columns_pathname))


# #### Columns selected by RFECV

# (cell not in the report)
# 
# In each iteration, the RFE.ranking_ attribute item (representing a feature) is increased if the feature is dropped -number of iterations + 1 in whitch feature was rejected - first beacause of having low importance (or coeficient), then because of not being considered.

# In[146]:


# not in the report
for n, m, sc in         sorted(
            zip(
                learning_data__var_th_scaled_columns, 
                rfecv_selector_on_extra_trees_cls.get_support(), 
                rfecv_selector_on_extra_trees_cls.ranking_
            ), 
            key=lambda x: x[2]) :
    if m:
        print("{:<100} {:<}".format(n,sc))


# The following listing shows feature importances from an extra trees classifier trained with the features selected by the RFECV procedure.

# In[177]:


# creating sorted list of columns
# learning_data__var_th_scaled_columns
importances = rfecv_selector_on_extra_trees_cls.estimator_.feature_importances_  #232
indices = np.argsort(importances)[::-1]

rfecv_selector_on_extra_trees__column_names_indices_importances__sorted =     sorted(
        [
            (rfecv_selector_on_extra_trees__column_names[indices[f]], indices[f], importances[indices[f]]) \
            for f in range(
                rfecv_selector_on_extra_trees_cls.transform(learning_data__var_th_X_train).shape[1]
            ) 
        ], 
        key=lambda e: e[2],
        reverse=True
    )

for i, (column_name, column_indice, column_importance) in         enumerate(rfecv_selector_on_extra_trees__column_names_indices_importances__sorted):
    print('{:<5d} {:<70}{:.4f}   {:.6f}  {}'.format(
        i+1, column_name,column_importance, 
        rfecv_selector_on_extra_trees_cls.grid_scores_[i],
        '|'*int(np.round((rfecv_selector_on_extra_trees_cls.grid_scores_[i]/np.max(rfecv_selector_on_extra_trees_cls.grid_scores_))*100 -80))
    ))


rfecv_selector_on_extra_trees__column_names__sorted =     [e[0] for e in rfecv_selector_on_extra_trees__column_names_indices_importances__sorted]

rfecv_selector_on_extra_trees__column_indices__sorted =     [e[1] for e in rfecv_selector_on_extra_trees__column_names_indices_importances__sorted]


# #### Checking consistency
# (not in the report)

# In[148]:


rfecv_selector_on_extra_trees_cls.transform(learning_data__var_th_X_train).shape[1]


# In[149]:


len(rfecv_selector_on_extra_trees_cls.estimator_.feature_importances_)


# In[150]:


len(rfecv_selector_on_extra_trees_cls.support_)


# In[151]:


max(rfecv_selector_on_extra_trees__column_names)


# In[152]:


max(rfecv_selector_on_extra_trees__column_indices__sorted)


# In[153]:


len(rfecv_selector_on_extra_trees__column_names)


# In[154]:


np.sum(rfecv_selector_on_extra_trees_cls.support_)


# In[155]:


len(learning_data__var_th_scaled_columns)


# In[156]:


learning_data__rfecv_selector_on_extra_trees_cls_columns_sorted_table =     make_feature_importance_table_df(rfecv_selector_on_extra_trees_cls.estimator_, rfecv_selector_on_extra_trees__column_names)

learning_data__rfecv_var_th_sc_columns_sorted_table = merge_feature_score_table_dfs(
    learning_data__all_var_th_sc_columns_sorted_table,
    learning_data__rfecv_selector_on_extra_trees_cls_columns_sorted_table,
    'rfecv_extr_tr'
).sort_values('score_rfecv_extr_tr', ascending=[False])


# In[157]:


pd.set_option("display.max_rows", 2000)
pd.set_option("display.max_colwidth", -1)


# In[158]:


learning_data__rfecv_var_th_sc_columns_sorted_table


# In[159]:


learning_data__rfecv_var_th_sc_columns_sorted_table.to_csv(
    os.path.join(data_snippets_dir, 'learning_data__rfecv_var_th_sc_columns_sorted_table.tsv'),
    sep='\t'
)


# In[160]:


for xscale in ('linear', 'log'):
    fig = plt.figure(figsize=(12,6))
    plt.plot(
        range(len(rfecv_selector_on_extra_trees_cls.grid_scores_)), 
        rfecv_selector_on_extra_trees_cls.grid_scores_,
        marker='', linestyle='-', 
    )
    plt.gca().set_xscale(xscale)
    # plt.legend()
    plt.xlim(100+len(rfecv_selector_on_extra_trees_cls.grid_scores_), 0.01)
    plt.savefig(os.path.join(data_snippets_dir, 'figures', 'rfecv_selectors_on_extra_tree_scores_plot_{}.svg'.format(xscale)), dpi=150)
    plt.show()


# #### Comparison of scores as function of feature number

# In[218]:


fig_common, ax_common = plt.subplots(1)
fig_common.set_size_inches(12,6)
ax_common.set_title('Comparison of function shapes')
ax_common.set_ylabel('Score (normalized)')
ax_common.set_xlabel('Feature number')

plot_lines = []

for (label, table_df), c in zip((
    ('Univariate - f_classif (1018 feat.)', learning_data__kbest_f_classif_var_th_sc_columns_sorted_table),
    ('Univariate - chi2 (1018 feat.)', learning_data__kbest_chi2_var_th_sc_columns_sorted_table.dropna()),
    ('Univariate - mutual_info_classif (1018 feat.)', learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table),
    ('Extra trees classifier (1018 feat.)', learning_data__extr_tr_var_th_sc_columns_sorted_table),
    ('Random forest classifier (1018 feat.)', learning_data__rndfrst_var_th_sc_columns_sorted_table),
    ('Adaboost classifier (1018 feat.)', learning_data__adabst_var_th_sc_columns_sorted_table),
    ('Extra trees classifier (RFECV feat.)', learning_data__rfecv_selector_on_extra_trees_cls_columns_sorted_table)
), plt.rcParams['axes.prop_cycle']):
    if label.startswith('Extra'):
        plt.figure(figsize=(12,6))
        plt.title("{}: Feature scores".format(label))
        plt.plot(table_df.index, table_df['score'], color=c['color'])
        plt.xlabel('Feature number')
        plt.ylabel('Score')
        plt.savefig(os.path.join(data_snippets_dir, 'figures', 'feature_importance_methods_w_rfecv_feature_score_plot_{}.svg'.format(label)), dpi=150)
        
    plot_lines.append(ax_common.plot(table_df.index, table_df['score']/np.max(table_df['score']), label=label, color=c['color']))
    
ax_common.legend()
fig_common.savefig(os.path.join(data_snippets_dir, 'figures', 'feature_importance_methods_w_rfecv_feature_score_plot_all_methods.svg'), dpi=150)
plt.show()


# #### Correlation between feature scoring approaches

# In[219]:


learning_data__rfecv_var_th_sc_columns_list =     [col for col in learning_data__rfecv_var_th_sc_columns_sorted_table.columns.values 
     if 'score' in col and 'score_std' not in col]


for sortby_column in learning_data__rfecv_var_th_sc_columns_list:

    print('Features sorted by:', sortby_column)
    
    fig_common, axs = plt.subplots(len(learning_data__rfecv_var_th_sc_columns_list), 1, sharex=True)
    fig_common.set_size_inches(15,20)
    # fig_common.suptitle('Comparison of function values')

    for i, (col, c) in enumerate(zip(learning_data__rfecv_var_th_sc_columns_list, plt.rcParams['axes.prop_cycle'])):
        df_sorted_by_col = learning_data__rfecv_var_th_sc_columns_sorted_table.sort_values(sortby_column, ascending=False)
        col_score_vals = df_sorted_by_col[col]
        axs[i].plot(range(len(df_sorted_by_col)), #df_sorted_by_col.index, 
                    col_score_vals/np.max(col_score_vals),
                    label=col.replace('score_',''), color=c['color'])
        axs[i].legend()
        axs[i].set_ylim(-0.1,1.2)
    axs[i].set_xlabel('Feature number')    
    fig_common.text(0.08, 0.5, 'Score (normalized)', va='center', rotation='vertical')
    fig_common.savefig(os.path.join(data_snippets_dir, 'figures', 
                                    'feature_importance_methods_w_rfecv_sorted_by_{}_plot.svg'.format(sortby_column)), dpi=150)
    plt.show()


# In[161]:


corrcoef_score_columns(learning_data__rfecv_var_th_sc_columns_sorted_table)


# In[221]:


plt.close('all')
corr = corrcoef_score_columns(learning_data__rfecv_var_th_sc_columns_sorted_table)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, square=True, cmap='inferno', annot=True, cbar_kws={'label': 'correlation coeff.'})
plt.gcf().set_size_inches(8,6)
plt.xticks(rotation=45)
plt.savefig(os.path.join(data_snippets_dir, 'figures', 'feature_importance_w_rfecv_methods_correlation_coeffs.svg'), dpi=150)
plt.show()


# (TODO distance metric table)

# #### Importances of interesting features

# TODO different color for each feature

# In[222]:


for feature_name_pattern in [
    'per_gtu', 'x_y_hough_peak.+width', 'x_y_hough_peak.+clusters_count', 'gtu_[xy]_hough_peak', 'x_y_clusters', 'gtu_[xy]_clusters', '^trg', '^alt', '^proc'
]:
    print(feature_name_pattern)
    for label, table_df in (
#         ('f_classif', learning_data__kbest_f_classif_var_th_sc_columns_sorted_table),
#         ('chi2', learning_data__kbest_chi2_var_th_sc_columns_sorted_table.dropna()),
#         ('mutual_classif', learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table),
        ('Univariate - f_classif (1018 feat.)', learning_data__kbest_f_classif_var_th_sc_columns_sorted_table),
#         ('Univariate - chi2', learning_data__kbest_chi2_var_th_sc_columns_sorted_table.dropna()),
#         ('Univariate - mutual_info_classif', learning_data__kbest_mutual_classif_var_th_sc_columns_sorted_table),
        ('Extra trees classifier (1018 feat.)', learning_data__extr_tr_var_th_sc_columns_sorted_table),
        ('Random forest classifier (1018 feat.)', learning_data__rndfrst_var_th_sc_columns_sorted_table),
#         ('Adaboost classifier', learning_data__adabst_var_th_sc_columns_sorted_table),),
        ('Extra trees classifier (RFECV feat.)', learning_data__rfecv_selector_on_extra_trees_cls_columns_sorted_table)
    ):
        print(' ',label)
        iii = 0
        
        indices = []
        
        for ii, r in table_df.iterrows():
            if re.search(feature_name_pattern, r['feature']):
                if iii < 10:
                    print('    {:10} {}'.format(ii, r['feature']))
                    iii += 1
                indices.append(ii)
                
        plt.hist(indices, bins=len(table_df)//20, range=(0,len(table_df)))
        plt.show()
        
        print('-'*60)
    print('='*60)


# ### gtu_x and gtu_y phi values

# In[167]:


feature_name_pattern = 'gtu_[xy].+_phi'
# learning_data__var_th_scaled_columns 
ranking_series = learning_data__adabst_var_th_sc_columns_sorted_table['feature'].tolist()

ranking_series_indices = []
learning_data_cols_indices = []
iii = 0

for ii, r in enumerate(ranking_series):
    if re.search(feature_name_pattern, r): 
        if iii < 10:
            print('    {:10} {}'.format(ii, r))
            iii += 1
            ranking_series_indices.append(ii)

learning_data_cols_indices = [
    learning_data__var_th_scaled_columns.index(ranking_series[i]) for i in ranking_series_indices]
            
learning_data_cols_indices


# In[183]:


for feat_indice in learning_data_cols_indices:
    feat_name = learning_data__var_th_scaled_columns[feat_indice]
    print(feat_name)
    fig, ax = plt.subplots(figsize=(6,4))
    for cls_val, cls_label in enumerate(['noise', 'air shower']):
        ax.hist(learning_data__var_th_X_test[:,feat_indice][learning_data__y_test==cls_val], 
                bins=180, alpha=0.5, label=cls_label)
    ax.set_xlabel(learning_data__var_th_scaled_columns[feat_indice])
    ax.set_ylabel('Number of entries')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(os.path.join(data_snippets_dir, 'figures', 'feat_{}_values_by class.svg'.format(feat_name)), dpi=150)
    plt.show()


# In[184]:


for feat_name, feat_indice, feat_impor in         rfecv_selector_on_extra_trees__column_names_indices_importances__sorted[:10]:
    print('{}: {:3.3f}'.format(feat_name, feat_impor))
    fix, ax = plt.subplots(figsize=(6,4))
    for cls_val, cls_label in enumerate(['noise', 'air shower']):
        ax.hist(learning_data__var_th_X_test[:,feat_indice][learning_data__y_test==cls_val], 
                bins=180, alpha=0.5, label=cls_label)
    ax.set_xlabel(learning_data__var_th_scaled_columns[feat_indice])
    ax.set_ylabel('Number of entries')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(os.path.join(data_snippets_dir, 'figures', 'feat_{}_values_by class.svg'.format(feat_name)), dpi=150)
    plt.show()


# ### T-SNE (RFECV features)
# *T-SNE visualization of the learning data (both training and testing)*

# Moved to ver4_machine_learning_w_labeled_flight_20190217_dimensionality_reduction.ipynb notebook

# ### Performance of the ExtraTreesClassifier model with RFECV features

# In[162]:


y_test = learning_data__y_test
y_test_pred = rfecv_selector_on_extra_trees_cls.predict(learning_data__var_th_X_test)

print(sklearn.metrics.confusion_matrix(
    y_test, 
    y_test_pred))

print_accuracy_cls_report(
    y_test, 
    y_test_pred)

labeled_data_cls_stats =     print_labeled_data_cls_stats(
        mask_arr_test=learning_data__lbl_noise_flight_mask_arr_test,
        y_test=y_test,
        y_test_pred=y_test_pred)


# #### Cross-validation

# In[173]:


def cross_val_calc_weights(indices, learning_data__y=learning_data__y, learning_data__source_class=learning_data__source_class): 
    return calc_learning_data_weights(learning_data__y[indices], learning_data__source_class[indices], print_info=False)

extra_trees_cls_on_train_rfecv_for_crossvalidation = sklearn.ensemble.ExtraTreesClassifier(**rfe_extra_trees_params)
# not entirely correct, feature selection should be also included in crossvalidation training

learning_data__rfecv_var_th_X =     rfecv_selector_on_extra_trees_cls.transform(
        var_th_selector_on_scaled_train.transform(
            learning_data__X
        )
    )

extra_trees_cls_on_train_rfecv_for_crossvalidation_crv_results =     cross_val_score_meta_scored(
        extra_trees_cls_on_train_rfecv_for_crossvalidation, 
        learning_data__rfecv_var_th_X, learning_data__y, 
        meta_score_func=None,
        score_func=balanced_accuracy_score,
        cv=3, random_state=32, 
        train_sample_weight_func=cross_val_calc_weights
)

print('Cross-validation accuracy:', extra_trees_cls_on_train_rfecv_for_crossvalidation_crv_results)
print('Mean accuracy:            ', np.mean(extra_trees_cls_on_train_rfecv_for_crossvalidation_crv_results))
print('Std accuracy:             ', np.std(extra_trees_cls_on_train_rfecv_for_crossvalidation_crv_results))

extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results = cross_val_score_meta_scored(
    extra_trees_cls_on_train_rfecv_for_crossvalidation,
    learning_data__rfecv_var_th_X, learning_data__y, 
    cv=3, random_state=32, verbose=1,
    meta_score_func=score_masked_using_indices_lbl_noise_flight_mask_arr_all,
    train_sample_weight_func=cross_val_calc_weights
)

print('Cross-validation accuracy (lbl_noise):', extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results)
print('Mean accuracy (lbl_noise):            ', np.mean(extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results))
print('Std accuracy (lbl_noise):             ', np.std(extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results))
    


# #### Cross-validation of labeled noise data

# ##### random_state = 123

# In[237]:


extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results_2 = cross_val_score_meta_scored(
    extra_trees_cls_on_train_rfecv_for_crossvalidation,
    learning_data__rfecv_var_th_X, learning_data__y, 
    cv=3, random_state=128, verbose=1,
    meta_score_func=score_masked_using_indices_lbl_noise_flight_mask_arr_all,
    train_sample_weight_func=cross_val_calc_weights
)
print('Cross-validation accuracy (lbl_noise, seed=123):', extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results_2)
print('Mean accuracy (lbl_noise, seed=123):            ', np.mean(extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results_2))
print('Std accuracy (lbl_noise, seed=123):             ', np.std(extra_trees_cls_on_train_rfecv_for_crossvalidation_lbl_noise_flight_crv_results_2))


# ## Recognition efficiency (RFECV model)

# ### Test set sensitivity as function of the energy

# In[163]:


extra_trees_cls_on_train_rfecv__test__numbers_by_energy =     score_by_column(
        rfecv_selector_on_extra_trees_cls, 
        learning_data__var_th_X_test[learning_data__simu_shower_track_mask_arr_test], 
        learning_data__y_test[learning_data__simu_shower_track_mask_arr_test], 
        calc_cls_numbers, #sklearn.metrics.accuracy_score, 
        learning_data__event_id_test[learning_data__simu_shower_track_mask_arr_test], 
        combined_simu_df, 'etruth_trueenergy')


# In[164]:


plt.close('all')
for xscale in ('linear', 'log'):
    fig, ax, errbr =         plot_efficiency_stat(
            extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
            plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 20, xscale=xscale,
            xlabel = 'True energy [MeV]', ylabel = 'Sensitivity', 
            calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
            figsize = (10,6), ylim=(0,1.2), show=False)
    fig.savefig(os.path.join(data_snippets_dir, 'figures', 
                             'Test set sensitivity as function of the energy - {}.svg'.format(xscale)), dpi=150)
    plt.show()


# ### Number of true positivie and positive samples as function of the energy

# #### Number of positive samples as function of the energy

# In[165]:


for xscale in ('linear', 'log'):
    fig, ax, errbr =         plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
                             plotted_stat='num_positive', num_steps = 20, xscale=xscale,
                             xlabel = 'True energy [MeV]', ylabel = 'Num. positive', 
                             figsize = (10,6))
    fig.savefig(os.path.join(data_snippets_dir, 'figures', 
                             'Number of positive samples as function of the energy - {}.svg'.format(xscale)), dpi=150)
    


# ##### Number of true positive samples as function of the energy

# In[166]:


for xscale in ('linear', 'log'):
    fig, ax, errbr =         plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
                             plotted_stat='num_true_positive', num_steps = 20, xscale=xscale,
                             xlabel = 'True energy [MeV]', ylabel = 'Num. true positive', 
                             figsize = (10,6), show=False)
    fig.savefig(os.path.join(data_snippets_dir, 'figures', 
                             'Number of true positive samples as function of the energy - {}.svg'.format(xscale)), dpi=150)
    plt.show()


# ##### Number of true positive or positive samples as function of the energy - comparison

# In[167]:


plt.close('all')
for xscale in ('linear', 'log'):
    fig, ax = plt.subplots()
    fig, ax, errbr_num_positive =         plot_efficiency_stat(
            extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
            plotted_stat='num_positive', num_steps = 20, xscale=xscale,
            xlabel = 'True energy [MeV]', ylabel = 'Num. positive', label='Num. positive',
            figsize = (10,6), errorbar_attrs=dict(linestyle='--', color='blue'), 
            ax=ax, show=False)
    fig, ax, errbr_num_true_positive =         plot_efficiency_stat(
            extra_trees_cls_on_train_rfecv__test__numbers_by_energy, 
            plotted_stat='num_true_positive', num_steps = 20, xscale=xscale,
            xlabel='True energy [MeV]', ylabel = 'Num. true positive', label='Num. true positive',
            figsize=(10,6), errorbar_attrs=dict(linestyle='-', color='green'),
            ax=ax, show=False)
    ax.set_ylabel('Num. samples')
    ax.legend()
    fig.savefig(os.path.join(data_snippets_dir, 'figures', 
                             'Number of true positive or positive samples as function of the energy - comparison - {}.svg'.format(xscale)), dpi=150)
    plt.show()


# #### Test set sensitivity as function of the theta (zenith angle)

# In[168]:


extra_trees_cls_on_train_rfecv__test__numbers_by_theta =     score_by_column(
        rfecv_selector_on_extra_trees_cls, 
        learning_data__var_th_X_test[learning_data__simu_shower_track_mask_arr_test], 
        learning_data__y_test[learning_data__simu_shower_track_mask_arr_test], 
        calc_cls_numbers,
        learning_data__event_id_test[learning_data__simu_shower_track_mask_arr_test], 
        combined_simu_df, 'etruth_truetheta')


# In[169]:


fig, ax, errbr =     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_theta, 
                         plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 90/2.5, 
                         xtranslate_func=np.rad2deg,
                         xlabel ='True theta [deg]', ylabel = 'Sensitivity', 
                         calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
                         figsize = (12,6), ylim=(0,1.5), show=False)
fig.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'Test set sensitivity as function of the theta (zenith angle).svg'), dpi=150)
plt.show()


# #### Test set sensitivity as function of the phi (azimuth angle)

# In[170]:


extra_trees_cls_on_train_rfecv__test__numbers_by_phi =     score_by_column(
        rfecv_selector_on_extra_trees_cls, 
        learning_data__var_th_X_test[learning_data__simu_shower_track_mask_arr_test], 
        learning_data__y_test[learning_data__simu_shower_track_mask_arr_test], 
        calc_cls_numbers,
        learning_data__event_id_test[learning_data__simu_shower_track_mask_arr_test], 
        combined_simu_df, 'etruth_truephi')


# In[171]:


fig, ax, errbr =     plot_efficiency_stat(extra_trees_cls_on_train_rfecv__test__numbers_by_phi, 
                         plotted_stat='sensitivity', plotted_yerr_stat='positive_sm_confint_beta_95', num_steps = 360/5, 
                         xtranslate_func=np.rad2deg,
                         xlabel = 'True phi [rad]', ylabel = 'Sensitivity', 
                         calc_cls_stats_from_numbers_func=calc_cls_stats_from_numbers_with_sm_proportion_confint,
                         figsize = (12,6), ylim=(0,1.5), show=False)
fig.savefig(os.path.join(data_snippets_dir, 'figures', 
                         'Test set sensitivity as function of the phi (azimuth angle).svg'), dpi=150)
plt.show()


# ### Cross-validated recognition performance

# In[174]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[175]:


pipeline_from_trained_models__extr_rfecv_vth__y_pred =     pipeline_from_trained_models__extr_rfecv_vth.predict(
        unl_flight_df[analyzed_common_df_columns].dropna().values)


# In[176]:


num_non_shower = np.count_nonzero(pipeline_from_trained_models__extr_rfecv_vth__y_pred == 0)
num_shower = np.count_nonzero(pipeline_from_trained_models__extr_rfecv_vth__y_pred == 1)
tot_entries = len(unl_flight_df[analyzed_common_df_columns].dropna().values)

print("Num. non-shower", num_non_shower)
print("Num. shower", num_shower)
print("All entries", tot_entries)
print("-"*30)
print("Fraction non-shower: {:.3f}".format(num_non_shower/tot_entries))
print("Fraction shower: {:.3f}".format(num_shower/tot_entries))


# In[ ]:





# In[ ]:


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


# In[ ]:


# joblib.dump(learning_data__X__tsne_learning_data_60_rfecv_columns_alldata, 
#                     tsne_on_learning_data_60_rfecv_columns_alldata_pathname, compress=1)


# In[ ]:


# tsne_on_learning_data_60_rfecv_columns_alldata


# In[ ]:




