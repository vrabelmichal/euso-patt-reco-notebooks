
# coding: utf-8

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
mpl.use('Agg')
import argparse
import glob
import traceback
import hashlib
import math
import collections
import functools

import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.ensemble 
import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics
import sklearn.pipeline
from sklearn.externals import joblib

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


# In[3]:


inverse_means_map = np.load('/home/spbproc/euso-spb-patt-reco-v1/resources/inverse_flat_average_directions_4m_flipud.npy')


# # Selecting the flight data

# In[4]:


model_data_snippets_dir = 'ver4_machine_learning_w_labeled_flight_20190217'
data_snippets_dir = 'ver4_machine_learning_flight_classification_tsne_cfg3_20190217'

os.makedirs(data_snippets_dir, exist_ok=True)


# In[5]:


event_processing_cls = event_processing_v4.EventProcessingV4
event_v3_storage_provider_flight = dataset_query_functions_v3.build_event_v3_storage_provider(
    event_storage_provider_config_file=os.path.join(app_base_dir,'config_w_flatmap.ini'), 
    table_names_version='ver4',
    event_storage_class=postgresql_v3_event_storage.PostgreSqlEventV3StorageProvider,
    event_processing_class=event_processing_cls
)

query_functions_flight = dataset_query_functions_v3.Ver3DatasetQueryFunctions(event_v3_storage_provider_flight)


# ## Columns

# In[6]:


rfecv_selector_on_extra_trees__column_names = []

columns_list_file_pathname = os.path.join(model_data_snippets_dir, 'rfecv_selector_on_extra_trees__column_names.txt')
print(columns_list_file_pathname)
with open(columns_list_file_pathname, 'r') as columns_list_file:
    rfecv_selector_on_extra_trees__column_names = columns_list_file.read().splitlines()


# In[7]:


rfecv_selector_on_extra_trees__column_names__special = []

special_columns_list_file_pathname = os.path.join(model_data_snippets_dir, 'rfecv_selector_on_extra_trees__column_names__special.txt')
print(special_columns_list_file_pathname)
with open(special_columns_list_file_pathname, 'r') as special_columns_list_file:
    rfecv_selector_on_extra_trees__column_names__special = special_columns_list_file.read().splitlines()


# In[8]:


# This should be empty for now
rfecv_selector_on_extra_trees__column_names__special


# In[9]:


flight_columns_for_analysis_dict = query_functions_flight.get_columns_for_classification_dict__by_excluding(
    excluded_columns_re_list=('^.+$',),
    default_excluded_columns_re_list=[],
    included_columns_re_list=[('^$','source_file_(acquisition|trigger)(_full)?|global_gtu|packet_id|gtu_in_packet|event_id|num_gtu'),] + rfecv_selector_on_extra_trees__column_names
)

classification_flight_columns_for_analysis_dict = query_functions_flight.get_columns_for_classification_dict__by_excluding(
    excluded_columns_re_list=('^.+$',),
    default_excluded_columns_re_list=[],
    included_columns_re_list=rfecv_selector_on_extra_trees__column_names
)

print_columns_dict(flight_columns_for_analysis_dict)


# WARNING: not selecting NULL trg lines

# ## Event classes

# In[10]:


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
    'noise_with_week_dot': 21,
    #
    'unclassified': -1
}


# In[11]:


len(EVENT_CLASSES)


# In[12]:


INVERSE_EVENT_CLASSES = {v: k for k, v in EVENT_CLASSES.items()}


# ## Constructing the query

# In[13]:


classification_table_name = event_v3_storage_provider_flight.database_schema_name + '.event_manual_classification'

current_columns_for_analysis_dict = flight_columns_for_analysis_dict

flight_select_clause_str, flight_tables_list =     query_functions_flight.get_query_clauses__select({
    **current_columns_for_analysis_dict,
    classification_table_name: ['class_number']
})

flight_clauses_str =     query_functions_flight.get_query_clauses__join(flight_tables_list)

flight_source_data_type_num = 1

flight_where_clauses_str = ''' 
    AND abs(gtu_in_packet-42) < 20
    AND {database_schema_name}.event_orig_x_y.count_nonzero > 256*6
'''

for table, cols_list in classification_flight_columns_for_analysis_dict.items():
    for col in cols_list:
        flight_where_clauses_str += ' AND {}.{} IS NOT NULL\n'.format(table, col)

flight_events_selection_query = query_functions_flight.get_events_selection_query_plain(
    source_data_type_num=flight_source_data_type_num,
    select_additional=flight_select_clause_str, 
    join_additional=flight_clauses_str,
    where_additional=flight_where_clauses_str,
    order_by='{data_table_name}.event_id', 
    offset=0, limit=350000,
    base_select='')


# In[14]:


print(flight_events_selection_query)


# In[15]:


flight_df = psql.read_sql(flight_events_selection_query, event_v3_storage_provider_flight.connection)


# In[16]:


flight_df.loc[flight_df['manual_classification_class_number'].isnull(), 'manual_classification_class_number'] = -1


# In[17]:


len(flight_df)


# In[18]:


flight_df['manual_classification_class_name'] = 'unclassified'
for class_name, class_number in EVENT_CLASSES.items():
    flight_df.loc[flight_df['manual_classification_class_number']==class_number, 'manual_classification_class_name'] = class_name


# In[19]:


flight_df.head()


# In[20]:


flight_df[rfecv_selector_on_extra_trees__column_names].head()


# In[21]:


np.count_nonzero(flight_df['event_id'].isnull())


# In[22]:


flight_df['had_nan_fields'] = flight_df[rfecv_selector_on_extra_trees__column_names].isnull().any(axis=1)


# In[23]:


flight_df_nonan = flight_df[~flight_df['had_nan_fields']]


# In[24]:


len(flight_df_nonan)


# In[25]:


# TODO
# SELECT COUNT(*) FROM spb_processing_v4_flatmap.event JOIN spb_processing_v4_flatmap.event_orig_x_y USING(event_id) WHERE source_data_type_num = 1 AND abs(gtu_in_packet-42) < 20 AND spb_processing_v4_flatmap.event_orig_x_y.count_nonzero > 256*6 LIMIT 5;
# SELECT COUNT( DISTINCT (source_file_acquisition, packet_id)) FROM spb_processing_v4_flatmap.event JOIN spb_processing_v4_flatmap.event_orig_x_y USING(event_id) WHERE source_data_type_num = 1 AND abs(gtu_in_packet-42) < 20 AND spb_processing_v4_flatmap.event_orig_x_y.count_nonzero > 256*6 LIMIT 5;


# # Applying models

# ## StandardScaler

# In[25]:


# standard_scaler_on_train_rfecv_columns_pathname = \
#      os.path.join(model_data_snippets_dir, 'standard_scaler_on_train_rfecv_columns.pkl')
# standard_scaler_on_train_rfecv_columns = joblib.load(standard_scaler_on_train_rfecv_columns_pathname)


# In[26]:


# flight__rfecv_columns_scaled_X = \
#     standard_scaler_on_train_rfecv_columns.transform(
#         flight_df[rfecv_selector_on_extra_trees__column_names].values)
# if np.count_nonzero(flight_df['had_nan_fields']) > 0:
#     flight_nonan__rfecv_columns_scaled_X = \
#         standard_scaler_on_train_rfecv_columns.transform(
#             flight_df_nonan[rfecv_selector_on_extra_trees__column_names].values)
# else:
#     flight_nonan__rfecv_columns_scaled_X = flight__rfecv_columns_scaled_X


# In[29]:


flight_df_nonan.shape


# In[31]:


len(rfecv_selector_on_extra_trees__column_names)


# In[28]:


np.count_nonzero(flight_df['had_nan_fields'])


# ## Extra trees classifier

# In[32]:


flight_rfecv_columns__X  = flight_df_nonan[rfecv_selector_on_extra_trees__column_names].values


# In[26]:


extra_trees_cls_on_train_rfecv__model_plk_pathname =     os.path.join(model_data_snippets_dir, 'extra_trees_cls_on_train_rfecv.pkl')
extra_trees_cls_on_train_rfecv_est = joblib.load(extra_trees_cls_on_train_rfecv__model_plk_pathname)


# In[34]:


flight_df['extra_trees_cls_on_train_rfecv_est'] =     extra_trees_cls_on_train_rfecv_est.predict(flight_rfecv_columns__X)
flight_df['extra_trees_cls_on_train_rfecv_est_dropna'] =     ((flight_df['extra_trees_cls_on_train_rfecv_est']==1) & ~flight_df['had_nan_fields']).astype('int8')


# In[35]:


np.count_nonzero(flight_df['extra_trees_cls_on_train_rfecv_est_dropna'])


# ### Percentage of selected events

# In[36]:


np.count_nonzero(flight_df['extra_trees_cls_on_train_rfecv_est_dropna'])/len(flight_df)


# In[37]:


flight_df_nonan[rfecv_selector_on_extra_trees__column_names + ['manual_classification_class_number']].head()


# # Visualization of the events

# ## Visualizing the features

# In[38]:


flight_df_nonan[rfecv_selector_on_extra_trees__column_names + ['manual_classification_class_number']].head()


# ### Radvis

# Presently not applied optimally, any useful output is not guarateed

# In[40]:


len(rfecv_selector_on_extra_trees__column_names)


# In[43]:


importances = extra_trees_cls_on_train_rfecv_est.feature_importances_
indices = np.argsort(importances)[::-1]


rfecv_selector_on_extra_trees__column_names_indices_importances__sorted =     sorted(
        [
            (rfecv_selector_on_extra_trees__column_names[indices[f]], indices[f], importances[indices[f]]) \
            for f in range(len(rfecv_selector_on_extra_trees__column_names)) 
        ], 
        key=lambda e: e[2],
        reverse=True
    )

for column_name, column_indice, column_importance in         rfecv_selector_on_extra_trees__column_names_indices_importances__sorted:
    print('{:<70}{:.4f}'.format(column_name,column_importance))


rfecv_selector_on_extra_trees__column_names__sorted =     [e[0] for e in rfecv_selector_on_extra_trees__column_names_indices_importances__sorted]

rfecv_selector_on_extra_trees__column_indices__sorted =     [e[1] for e in rfecv_selector_on_extra_trees__column_names_indices_importances__sorted]


# In[36]:


#plt.close('all')
#sns.set(style="whitegrid")


## In[46]:


#plt.close('all')
#for num_features in range(5,80,20):
    #print('Number of features:\t{}'.format(num_features))
    #f, ax = plt.subplots()
    #f.set_size_inches(20,20)
    #pd.plotting.radviz(
        #flight_df_nonan[rfecv_selector_on_extra_trees__column_names__sorted[0:num_features] + ['manual_classification_class_name']], 
        #'manual_classification_class_name', alpha=.8
    #)
    #plt.show()


# In[38]:


#mpl.rcParams.update(mpl.rcParamsDefault)


# ### T-SNE

# In[39]:


#get_ipython().magic('load_ext wurlitzer')
# from sklearn.manifold import TSNE
import MulticoreTSNE
from sklearn.externals import joblib


# In[40]:


load_existing_tsne_model = True
refit_tsne_model = False
dump_tsne_embedding = True
show_plots = False


# #### Fitting

# In[41]:


tsne_hyperparams_dict = dict(
    learning_rate=100,
    n_iter=5000,
    perplexity=50, 
    angle=0.3,
    verbose=10,
    random_state=9621
)


# In[51]:

num_columns_list = list(range(25, len(rfecv_selector_on_extra_trees__column_names)+1, 25))

if (len(rfecv_selector_on_extra_trees__column_names)-25) % 25 != 0:
        num_columns_list += [len(rfecv_selector_on_extra_trees__column_names)]

# In[52]:


num_columns_list


# In[49]:


rfecv_selector_on_extra_trees__column_names__sorted


# In[42]:


tsne_hyperparams_dict_md5str = hashlib.md5(';'.join(['{}={}'.format(k,v) for k,v in sorted(tsne_hyperparams_dict.items())]).encode()).hexdigest()

flight_rfecv_columns__X_tsne_dict = {}
tsne_on_flight_nonan_rfecv_columns_X_pathname_dict = {}

for num_columns in num_columns_list:
    
    tsne_on_flight_nonan_rfecv_columns_X_pathname_dict[num_columns] = \
        os.path.join(data_snippets_dir, 'tsne_flight_nonan__cfg_{}__{}_rfecv_columns__X.npy'.format(tsne_hyperparams_dict_md5str, num_columns))
    
    print('Fitting T-SNE of {} features'.format(num_columns))
    print('Pickled file pathname', tsne_on_flight_nonan_rfecv_columns_X_pathname_dict[num_columns])
    
    if refit_tsne_model or not os.path.exists(tsne_on_flight_nonan_rfecv_columns_X_pathname_dict[num_columns]):
        tsne_on_flight_nonan_rfecv_columns = MulticoreTSNE.MulticoreTSNE(**tsne_hyperparams_dict, n_jobs=10)
        flight_rfecv_columns__X_tsne_dict[num_columns] = tsne_on_flight_nonan_rfecv_columns.fit_transform(
            flight_rfecv_columns__X.T[rfecv_selector_on_extra_trees__column_indices__sorted[:num_columns+1]].T)
#         if dump_tsne_model: 
#             joblib.dump(tsne_on_flight_nonan_rfecv_columns, tsne_on_flight_nonan_rfecv_columns_pathname_dict[num_columns], compress=1)
        if dump_tsne_embedding:
            np.save(tsne_on_flight_nonan_rfecv_columns_X_pathname_dict[num_columns], flight_rfecv_columns__X_tsne_dict[num_columns])
        
    else:
#         tsne_on_flight_nonan_rfecv_columns = joblib.load(tsne_on_flight_nonan_rfecv_columns_pathname_dict[num_columns])
#         flight_rfecv_columns__X_tsne_dict[num_columns] = tsne_on_flight_nonan_rfecv_columns.embedding_
        flight_rfecv_columns__X_tsne_dict[num_columns] = np.load(tsne_on_flight_nonan_rfecv_columns_X_pathname_dict[num_columns])
        
        
# In[44]:


#get_ipython().system('ls ver4_machine_learning_flight_classification_tsne_cfg3/tsne_flight_nonan__cfg_79aee445682f33e378edc461ed2ef999__rfecv_columns.pkl')


# In[45]:


for num_columns, X_tsne in flight_rfecv_columns__X_tsne_dict.items():
    
    x_col_prefix = 'tsne_{}_feat__'.format(num_columns)
    
    flight_df[x_col_prefix + 'X_0'] = np.nan
    flight_df[x_col_prefix + 'X_1'] = np.nan

    flight_df.loc[flight_df['event_id'].isin(flight_df_nonan['event_id']), x_col_prefix + 'X_0'] = flight_rfecv_columns__X_tsne_dict[num_columns][:, 0]
    flight_df.loc[flight_df['event_id'].isin(flight_df_nonan['event_id']), x_col_prefix + 'X_1'] = flight_rfecv_columns__X_tsne_dict[num_columns][:, 1]

    flight_df_nonan = flight_df[~flight_df['had_nan_fields']]


# #### Visualizing results of T-SNE

# In[46]:


EVENT_CLASSES


# ##### All classes in a single plot

# In[47]:


# %load_ext autoreload
# %autoreload 1
# %aimport event_visualization
# %aimport data_analysis_utils


# In[48]:


def check_row_is_extra_trees_1(row, cls_column='extra_trees_cls_on_train_rfecv_est_dropna'):
    return r[cls_column] == 1

def add_tsne_shower_xy_axes(fig, scatter_plot_ax, df_nonan, annotation_side_frac, annotation_limit=None,
                            highlight_check_func=check_row_is_extra_trees_1, highlight_color='red', 
                            event_id_fontsize=8.5, event_id_color='red', cls_column='extra_trees_cls_on_train_rfecv_est_dropna', 
                            x_col_prefix='tsne__'):
    
    col_x0 = x_col_prefix + 'X_0'
    col_x1 = x_col_prefix + 'X_1'
    
    ax_lim_width = (scatter_plot_ax.get_xlim()[1]-scatter_plot_ax.get_xlim()[0])
    annotation_side = annotation_side_frac*ax_lim_width

    shown_images = np.array([[1., 1.]])  # just something big

    for _i, (i, r) in enumerate(df_nonan.iterrows()):

        if _i % 500 == 0:
            print('{}/{}'.format(_i, len(df_nonan)))

        dist = np.sqrt(np.sum(((r[col_x0], r[col_x1]) - shown_images) ** 2, 1))

        if np.min(dist) < annotation_side:
            continue

        shown_images = np.r_[shown_images, [ [r[col_x0], r[col_x1]] ]]

        axes_coords = [ 
            scatter_plot_ax.get_position().x0 + (- scatter_plot_ax.get_xlim()[0] + r[col_x0] - (annotation_side/2) ) * (scatter_plot_ax.get_position().x1 - scatter_plot_ax.get_position().x0)/(scatter_plot_ax.get_xlim()[1]-scatter_plot_ax.get_xlim()[0]) ,
            scatter_plot_ax.get_position().y0 + (- scatter_plot_ax.get_ylim()[0] + r[col_x1] - (annotation_side/2) ) * (scatter_plot_ax.get_position().y1 - scatter_plot_ax.get_position().y0)/(scatter_plot_ax.get_ylim()[1]-scatter_plot_ax.get_ylim()[0]) ,

            annotation_side * (scatter_plot_ax.get_position().x1 - scatter_plot_ax.get_position().x0)/(scatter_plot_ax.get_xlim()[1]-scatter_plot_ax.get_xlim()[0]), 
            annotation_side * (scatter_plot_ax.get_position().y1 - scatter_plot_ax.get_position().y0)/(scatter_plot_ax.get_ylim()[1]-scatter_plot_ax.get_ylim()[0])
        ]

        pax = fig.add_axes(axes_coords)
        if r[cls_column] == 1:
            pax.spines['bottom'].set_color(highlight_color)
            pax.spines['top'].set_color(highlight_color) 
            pax.spines['right'].set_color(highlight_color)
            pax.spines['left'].set_color(highlight_color)
        else:
            pax.set_axis_off()

        pax.set_xticks([])
        pax.set_yticks([])

        visualize_single_event(
            r['source_file_acquisition_full'], 
            r['packet_id'], r['gtu_in_packet'], r['num_gtu'],
            ax_xy=pax, vis_gtux=False, vis_gtuy=False, draw_colorbar=False, 
            xlabel_xy=None, ylabel_xy=None, zlabel_xy=None,
            inverse_means_arr=inverse_means_map
        )

        fig.text(
            scatter_plot_ax.get_position().x0 + (- scatter_plot_ax.get_xlim()[0] + r[col_x0] + (annotation_side/2) ) * (scatter_plot_ax.get_position().x1 - scatter_plot_ax.get_position().x0)/(scatter_plot_ax.get_xlim()[1]-scatter_plot_ax.get_xlim()[0]) ,
            scatter_plot_ax.get_position().y0 + (- scatter_plot_ax.get_ylim()[0] + r[col_x1] + (annotation_side/2) ) * (scatter_plot_ax.get_position().y1 - scatter_plot_ax.get_position().y0)/(scatter_plot_ax.get_ylim()[1]-scatter_plot_ax.get_ylim()[0]) ,
            str(r['event_id']),
            fontsize=event_id_fontsize, color=event_id_color, ha='right', va='top', alpha=1
        )

#             print(r['source_file_acquisition_full'], r['packet_id'], r['gtu_in_packet'], r['num_gtu'], r[col_x0], r[col_x1], axes_coords)
#             print('-'*50)

        if annotation_limit is not None and _i > annotation_limit:
            break
        
    # end for


# In[49]:


plt.close('all')

# cm = plt.cm.nipy_spectral
# cm = plt.cm.gist_ncar
cm = plt.cm.gist_rainbow
event_classes_for_cm = [v for k, v in EVENT_CLASSES.items() if k != 'unclassified']
cm_norm = mpl.colors.Normalize(min(event_classes_for_cm),max(event_classes_for_cm))
unclassified_class_color = 'black'

cls_column = 'extra_trees_cls_on_train_rfecv_est_dropna'

for num_columns in num_columns_list:

    print('-'*50)
    print('T-SNE on {} features'.format(num_columns))
    print('-'*50)
    
    filename_prefix = 'flight_data_feat_{}_'.format(num_columns)
    x_col_prefix = 'tsne_{}_feat__'.format(num_columns)
    col_x0 = x_col_prefix + 'X_0'
    col_x1 = x_col_prefix + 'X_1'

    for size_inches, do_unclassified, show_in_notebook, unclassified_class_alpha, scatter_s,         show_annotations, annotation_side_frac, annotation_limit, savefig_pathname  in (

            ((10,10), True, show_plots, .02, 10,
             False, 0.02, None, 
             os.path.join(data_snippets_dir, filename_prefix + 'tsne_10inch.png')),

            ((10,10), True, show_plots, .02, 10,
             True, 0.02, None, 
             os.path.join(data_snippets_dir, filename_prefix + 'tsne_annot_10inch.png')),

            ((40,40), True, False, .02, 60,
             False, 0.02, None, 
             os.path.join(data_snippets_dir, filename_prefix + 'tsne_40inch.png')),

            ((40,40), True, False, .02, 60,
             True, 0.02, None, 
             os.path.join(data_snippets_dir, filename_prefix + 'tsne_annot_40inch.png')),

            # following configuration is computationally demanding    
            ((200,200), True, False, .1, 80,
             True, 0.006, None, 
             os.path.join(data_snippets_dir, filename_prefix + 'tsne_annot_40inch.png')),
    ):
        f, ax = plt.subplots()
        f.set_size_inches(*size_inches)

        tsne_scatter_labels_dict = {}
        tsne_scatter_labels = []
        tsne_scatter_pathcolls = []

        for class_name, class_number in sorted(EVENT_CLASSES.items(),key=lambda x: x[1]):
            if class_name == 'unclassified' and not do_unclassified:
                continue

            flight_df_nonan_subset = flight_df_nonan[flight_df_nonan['manual_classification_class_number']==class_number]

            if len(flight_df_nonan_subset) <= 0:
                continue

            if class_name == 'unclassified':
                class_color = unclassified_class_color
                event_class_alpha = unclassified_class_alpha
            else:
                class_color = cm(cm_norm(class_number))
                event_class_alpha = 0.7


            for classification_class_num, classification_class_name, subset_marker in (
                (1, 'shower est.', 'o'), (0, 'noise est.', 'x')
            ):
                flight_df_nonan_subsubset = flight_df_nonan_subset[flight_df_nonan_subset[cls_column] == classification_class_num]

                pathcoll = ax.scatter(
                    flight_df_nonan_subsubset[col_x0], 
                    flight_df_nonan_subsubset[col_x1], 
                    c=class_color, 
                    s=80, linewidths=0,
                    alpha=event_class_alpha,
                    marker=subset_marker
                )

                pathcoll_for_legend = mpl.lines.Line2D(range(1),range(1), 
                                                       color=class_color, markersize=scatter_s, marker=subset_marker,
                                                       linewidth=0, linestyle='none') # TODO!!
                
                tsne_scatter_labels_dict[pathcoll_for_legend] =                     (len(flight_df_nonan_subset), '{} ({})'.format(class_name, classification_class_name))
                
#                 tsne_scatter_labels.append('{} ({})'.format(class_name, classification_class_name))
#                 tsne_scatter_pathcolls.append(pathcoll_for_legend)

        if show_annotations:
            add_tsne_shower_xy_axes(f, ax, flight_df_nonan, annotation_side_frac, annotation_limit=annotation_limit)

#         ax.legend(tsne_scatter_pathcolls, tsne_scatter_labels)

        if savefig_pathname is not None:
            print('Saving figure: {}'.format(savefig_pathname))
            f.savefig(savefig_pathname)

        if show_in_notebook:    
            plt.show()
        
        tsne_scatter_labels_dict = collections.OrderedDict(sorted(tsne_scatter_labels_dict.items(), key=lambda x: x[1][0], reverse=True))

        f, ax = plt.subplots()
        f.set_size_inches(15,6)

        ax.legend(tsne_scatter_labels_dict.keys(), [v[1] for v in tsne_scatter_labels_dict.values()], 
                  loc='center', fontsize='large', mode='expand', ncol=8)
        ax.set_axis_off()
            
        if savefig_pathname is not None:
            
            savefig_legend_pathname = os.path.splitext(savefig_pathname)[0]
            
            print('Saving figure: {}'.format(savefig_legend_pathname))
            f.savefig(savefig_legend_pathname)
        
        if show_in_notebook:    
            plt.show()

# flight_nonan__rfecv_columns_scaled_X__tsne


# (in the report)
# (link tsne_all_w_xy_proj_annotations.png)

# ##### Unclassified and a single class

# In[50]:


plt.close('all')

cm = plt.cm.gist_rainbow
event_classes_for_cm = [v for k, v in EVENT_CLASSES.items() if k != 'unclassified']
cm_norm = mpl.colors.Normalize(min(event_classes_for_cm),max(event_classes_for_cm))
unclassified_class_color = 'black'
single_class_event_class_alpha_dict = {'unclassified': .01}
single_class_event_class_alpha_default = 1

flight_df_nonan_subset_unclassified = flight_df_nonan[flight_df_nonan['manual_classification_class_number']==EVENT_CLASSES['unclassified']]

show_in_notebook = show_plots

for num_columns in num_columns_list:

    filename_prefix = 'flight_data_feat_{}_'.format(num_columns)
    x_col_prefix = 'tsne_{}_feat__'.format(num_columns)
    col_x0 = x_col_prefix + 'X_0'
    col_x1 = x_col_prefix + 'X_1'
    
    print('-'*50)
    print('T-SNE on {} features'.format(n_columns))
    print('-'*50)

    for class_name, class_number in sorted(EVENT_CLASSES.items(),key=lambda x: x[1]):

        if class_name == 'unclassified':
            continue

        flight_df_nonan_subset = flight_df_nonan[flight_df_nonan['manual_classification_class_number']==class_number]

        if len(flight_df_nonan_subset) <= 0:
            continue
            
            
        savefig_pathname = os.path.join(data_snippets_dir, filename_prefix + 'tsne_{}.png'.format(class_name))


        print('-'*30)
        print(class_name)
        print('-'*30)

        f, ax = plt.subplots()
        f.set_size_inches(15,15)

        tsne_scatter_labels = []
        tsne_scatter_pathcolls = []

        for t_class_name, t_class_number, t_class_color, subset_df in (
                ('unclassified', EVENT_CLASSES['unclassified'], unclassified_class_color, flight_df_nonan_subset_unclassified), 
                (class_name, class_number, cm(cm_norm(class_number)), flight_df_nonan_subset)
        ):

            if t_class_name in single_class_event_class_alpha_dict:
                event_class_alpha = single_class_event_class_alpha_dict[t_class_name]
            else:
                event_class_alpha = single_class_event_class_alpha_default

            for classification_class_num, classification_class_name, subset_marker in (
                (1, 'shower est.', 'o'), (0, 'noise est.', 'x')
            ):
                subsubset_df = subset_df[subset_df['extra_trees_cls_on_train_kbest400_128_est_dropna']==classification_class_num]
                pathcoll = ax.scatter(
                    subsubset_df[col_x0], 
                    subsubset_df[col_x1], 
                    c=t_class_color, 
                    s=80, linewidths=0,
                    alpha=event_class_alpha,
                    marker=subset_marker
                )

                tsne_scatter_labels.append('{} ({})'.format(t_class_name, classification_class_name))
                tsne_scatter_pathcolls.append(pathcoll)

        ax.legend(tsne_scatter_pathcolls, tsne_scatter_labels)
        
        if savefig_pathname is not None:
            print('Saving figure: {}'.format(savefig_pathname))
            f.savefig(savefig_pathname)
        
        
        if show_in_notebook:
            plt.show()

# flight_nonan__rfecv_columns_scaled_X__tsne


# In[57]:


# from this point is should not work


## ### T-SNE Data after classification

## In[51]:


#plt.close('all')

#cm = plt.cm.coolwarm
#event_classes_for_cm = [0,1]
#cm_norm = mpl.colors.Normalize(min(event_classes_for_cm),max(event_classes_for_cm))
## unclassified_class_color = 'black'
#unclassified_class_marker='x'

#for size_inches, show_in_notebook, unclassified_class_alpha, print_counts,     show_annotations, annotation_side_frac, annotation_limit,     savefig_pathname  in (
        #((35,35), show_plots, .01, True, False, 0.02, 10000, None),
        #((35,35), show_plots, .02, False, True, 0.02, 10000, None),
##       following configuration is computationally demanding    
        #((200,200),  False, .07, False, True, 0.006, None,
         #os.path.join(data_snippets_dir, 'tsne_cls_shower_all_w_xy_proj_annotations.png')),
#):
    #f, ax = plt.subplots()
    #f.set_size_inches(*size_inches)

    #tsne_scatter_labels = []
    #tsne_scatter_pathcolls = []

    #for class_name, class_number in sorted(EVENT_CLASSES.items(),key=lambda x: x[1]):
        #if print_counts:
            #print('Class: {}'.format(class_name))

        #for extra_trees_cls_on_train_kbest400_128_est in [0,1]:

            #flight_df_nonan_subset = flight_df_nonan[
                #(flight_df_nonan['manual_classification_class_number']==class_number) & \
                #(flight_df_nonan['extra_trees_cls_on_train_kbest400_128_est_dropna']==extra_trees_cls_on_train_kbest400_128_est)
            #]
            
            #if print_counts:
                #print('  classified {}: {} events'.format(
                    #'shower' if extra_trees_cls_on_train_kbest400_128_est == 1 else 'noise', 
                    #len(flight_df_nonan_subset)))

            #if len(flight_df_nonan_subset) <= 0:
                #continue

            #if extra_trees_cls_on_train_kbest400_128_est == 1:
                #event_class_alpha = 0.4
            #else:
            
                #if class_name == 'unclassified':
                    #class_color = unclassified_class_color
                    #event_class_alpha = unclassified_class_alpha
                #else:
                    #class_color = cm(cm_norm(class_number))
                    #event_class_alpha = 0.7

            #if class_name == 'unclassified':
                #class_marker = unclassified_class_marker
            #else:
                #class_marker = '${}$'.format(class_number)

            #class_color = cm(cm_norm(extra_trees_cls_on_train_kbest400_128_est))

            #pathcoll = ax.scatter(
                #flight_df_nonan_subset['tsne_X_0'], 
                #flight_df_nonan_subset['tsne_X_1'], 
                #c=class_color, 
                #s=80, linewidths=0,
                #alpha=event_class_alpha,
                #marker=class_marker
            #)

            #tsne_scatter_labels.append('{} ({}: {})'.format(
                #class_name, 
                #'shower' if extra_trees_cls_on_train_kbest400_128_est==1 else 'noise',
                 #len(flight_df_nonan_subset)
            #))
            #tsne_scatter_pathcolls.append(pathcoll)

    #if show_annotations:
        #add_tsne_shower_xy_axes(
            #f, ax, 
            #flight_df_nonan[flight_df_nonan['extra_trees_cls_on_train_kbest400_128_est_dropna']==1], 
            #annotation_side_frac, annotation_limit=annotation_limit)
        
    #ax.legend(tsne_scatter_pathcolls, tsne_scatter_labels)
    
    #if savefig_pathname is not None:
        #print('Saving figure: {}'.format(savefig_pathname))
        #f.savefig(savefig_pathname)
    
    #if show_in_notebook:
        #plt.show()

## flight_nonan__rfecv_columns_scaled_X__tsne


## (in the report)
## (link tsne_cls_shower_all_w_xy_proj_annotations.png)

## ### Clustering T-SNE data

## In[54]:


#import sklearn.cluster
## from sklearn import cluster, datasets, mixture


## In[55]:


#np.array([flight_df_nonan['tsne_X_0'], flight_df_nonan['tsne_X_1']]).T.shape


## In[56]:


#flight_df_nonan[['tsne_X_0','tsne_X_1']].values.shape


## #### DBSCAN clustering

## In[57]:


#dbscan_on_tsne = sklearn.cluster.DBSCAN(eps=1.1, algorithm='ball_tree')
#flight_nonan__tsne__dbscan_y_pred = dbscan_on_tsne.fit_predict(flight_df_nonan[['tsne_X_0','tsne_X_1']].values)


## ##### Identified clusters

## In[58]:


#dbscan_on_tsne_classes = np.unique(flight_nonan__tsne__dbscan_y_pred)
#print(dbscan_on_tsne_classes)


## ##### Adding to flight data DataFrame

## In[59]:


#flight_df['dbscan_tsne_y'] = np.nan

#flight_df.loc[flight_df['event_id'].isin(flight_df_nonan['event_id']), 'dbscan_tsne_y'] = flight_nonan__tsne__dbscan_y_pred

#flight_df_nonan = flight_df[~flight_df['had_nan_fields']]


## ##### Visualization of the clusters

## In[60]:


#plt.close('all')

##dbscan_cm = plt.cm.gist_rainbow
#dbscan_cm = plt.cm.prism
#dbscan_on_tsne_classes_for_cm = [v for v in dbscan_on_tsne_classes if v > 0]
#dbscan_cm_norm = mpl.colors.Normalize(min(dbscan_on_tsne_classes_for_cm),max(dbscan_on_tsne_classes_for_cm))
#dbscan_unclassified_class_color = 'black'

#f, ax = plt.subplots()
#f.set_size_inches(30,30)

#tsne_scatter_labels_dict = {}

#for cluster_number in sorted(dbscan_on_tsne_classes):
    
    #flight_df_nonan_subset = flight_df_nonan[flight_df_nonan['dbscan_tsne_y']==cluster_number]

    #if len(flight_df_nonan_subset) <= 0:
        #continue

    #event_class_alpha = 0.3

    #if cluster_number < 0:
        #class_color = dbscan_unclassified_class_color
    #else:
        #class_color = dbscan_cm(dbscan_cm_norm(cluster_number))

    #pathcoll = ax.scatter(
        #flight_df_nonan_subset['tsne_X_0'], 
        #flight_df_nonan_subset['tsne_X_1'], 
        #c=class_color, 
        #s=80, linewidths=0,
        #alpha=event_class_alpha,
        #marker='${:d}$'.format(cluster_number)
    #)

    #tsne_scatter_labels_dict[pathcoll] = (len(flight_df_nonan_subset), '{} ({})'.format(cluster_number, len(flight_df_nonan_subset)))
    
##     tsne_scatter_labels.append('{}'.format(len(flight_df_nonan_subset)))
##     tsne_scatter_pathcolls.append(pathcoll)

#plt.show()

#tsne_scatter_labels_dict = collections.OrderedDict(sorted(tsne_scatter_labels_dict.items(), key=lambda x: x[1][0], reverse=True))

#f, ax = plt.subplots()
#f.set_size_inches(15,6)

#ax.legend(tsne_scatter_labels_dict.keys(), [v[1] for v in tsne_scatter_labels_dict.values()], 
          #loc='center', fontsize='large', mode='expand', ncol=8)
#ax.set_axis_off()

#plt.show()


## ##### Cluster membership distribution

## In[61]:


#dbscan_on_tsne_classes_range = np.max(dbscan_on_tsne_classes) - np.min(dbscan_on_tsne_classes)

#for yscale in ['log','linear']:
    #f, ax = plt.subplots()
    #f.set_size_inches(24,6)
    #ax.set_yscale(yscale)
    #flight_df_nonan['dbscan_tsne_y'].hist(
        #ax=ax, bins=2*dbscan_on_tsne_classes_range+2, 
        #range=(np.min(dbscan_on_tsne_classes), np.max(dbscan_on_tsne_classes)+1))
    #plt.show()


## ##### Cluster size distibution

## In[62]:


#plt.close('all')
#f, ax = plt.subplots(figsize=(8,4))
#flight_df_nonan[['event_id', 'dbscan_tsne_y']].groupby('dbscan_tsne_y').count()['event_id'].hist(bins=200, ax=ax) 
#ax.set_xlabel('Cluster size')
#ax.set_ylabel('Frequency')
#ax.set_yscale('log')
## ax.set_xscale('log')
#plt.show()


## ##### Number of events in a cluster by a class of an event

## In[64]:


#plt.close('all')

#cm = plt.cm.gist_rainbow
#event_classes_for_cm = [v for k, v in EVENT_CLASSES.items() if k != 'unclassified']
#cm_norm = mpl.colors.Normalize(min(event_classes_for_cm),max(event_classes_for_cm))
#unclassified_class_color = 'black'

#tsne_X_0_range = np.min(flight_df_nonan['tsne_X_0']), np.max(flight_df_nonan['tsne_X_0'])
#tsne_X_1_range = np.min(flight_df_nonan['tsne_X_1']), np.max(flight_df_nonan['tsne_X_1'])

#single_class_event_class_alpha_dict = {'unclassified': .01}
#single_class_event_class_alpha_default = 1


#dbscan_on_tsne_classes_range = np.max(dbscan_on_tsne_classes) - np.min(dbscan_on_tsne_classes)

#for class_name, class_number in sorted(EVENT_CLASSES.items(),key=lambda x: x[1]):
            
    #flight_df_nonan_subset = flight_df_nonan[flight_df_nonan['manual_classification_class_number']==class_number]

    #print('-'*30)
    #print('{} ({})'.format(class_name, len(flight_df_nonan_subset)))
    #print('-'*30)
    
    #if len(flight_df_nonan_subset) <= 0:
        #print('Empty')
        #continue
    
    #unsorted_cluster_numbers_str = '  ' +         '  '.join('{:3d} ({:5d}){}'.format(cluster_number, r['event_id'], '\n' if (i+1) % 10 == 0 else '')             for i, (cluster_number, r) in enumerate(flight_df_nonan_subset[['event_id', 'dbscan_tsne_y']].groupby('dbscan_tsne_y').count().iterrows()))
    
    #sorted_cluster_numbers_str = '  ' +         '  '.join('{:3d} ({:5d}){}'.format(cluster_number, r['event_id'], '\n' if (i+1) % 10 == 0 else '')                 for i, (cluster_number, r) in                     enumerate(sorted(flight_df_nonan_subset[['event_id', 'dbscan_tsne_y']].groupby('dbscan_tsne_y').count().iterrows(), 
                           #key=lambda x: x[1]['event_id'], reverse=True)))
       
    #print(unsorted_cluster_numbers_str)
    
    #if unsorted_cluster_numbers_str != sorted_cluster_numbers_str:
        #print('  Sorted:')
        #print(sorted_cluster_numbers_str)
    
    #f, ax = plt.subplots(figsize=(18,3))
    #flight_df_nonan_subset['dbscan_tsne_y'].hist(ax=ax, bins=2*dbscan_on_tsne_classes_range+2, 
                                                 #range=(np.min(dbscan_on_tsne_classes), np.max(dbscan_on_tsne_classes)+1))
    #ax.set_xlabel('Cluster number (dbscan_tsne_y)')
    #ax.set_ylabel('Frequency')
    #ax.set_title('Cluster numbers for class "{}"'.format(class_name))
    #plt.show()
    
    #if class_name != 'unclassified':
        #f, ax = plt.subplots(figsize=(15,15))

        #ax.set_xlim(*tsne_X_0_range)
        #ax.set_ylim(*tsne_X_1_range)

        #tsne_scatter_labels = []
        #tsne_scatter_pathcolls = []

        #for cluster_number in flight_df_nonan_subset['dbscan_tsne_y'].unique():
    ##         print(cluster_number)

            #flight_df_nonan_subset_other = flight_df_nonan[(flight_df_nonan['manual_classification_class_number'] != class_number) & (flight_df_nonan['dbscan_tsne_y'] == cluster_number)]
            #flight_df_nonan_subsubset = flight_df_nonan_subset[(flight_df_nonan_subset['dbscan_tsne_y'] == cluster_number)]

            #for subset_df, t_class_name, event_class_alpha, t_class_color in (
                #(flight_df_nonan_subset_other, 'Other classes', single_class_event_class_alpha_dict['unclassified'], unclassified_class_color),
                #(flight_df_nonan_subsubset, class_name, single_class_event_class_alpha_default, cm(cm_norm(class_number)))
            #):
    ##             print('  -',t_class_name,len(subset_df))

                #if len(subset_df) <= 0:
                    #continue

                #for classification_class_num, classification_class_name, subset_marker in (
                    #(1, 'shower est.', 'o'), (0, 'noise est.', '${}$'.format(cluster_number))
                #):  
                    #subsubset_df = subset_df[subset_df['extra_trees_cls_on_train_kbest400_128_est_dropna']==classification_class_num]

    ##                 print('    -',classification_class_name, len(subsubset_df))

                    #pathcoll = ax.scatter(
                        #subsubset_df['tsne_X_0'], 
                        #subsubset_df['tsne_X_1'], 
                        #c=t_class_color, 
                        #s=80, linewidths=0,
                        #alpha=event_class_alpha,
                        #marker=subset_marker
                    #)

    ##                 if class_name != 'unclassified':
    ##                     tsne_scatter_labels.append('{}: {} ({})'.format(cluster_number, t_class_name, classification_class_name))
    ##                     tsne_scatter_pathcolls.append(pathcoll)

    ##     ax.legend(tsne_scatter_pathcolls, tsne_scatter_labels)
        #plt.show()
    


## ##### Number of event classes by a cluster

## In[65]:


#small_cluster_threshold = 5  # number of events, if less cluster is not visualized


## ###### Clusters to be visualized

## In[66]:


#class_sizes_ser =     flight_df_nonan[flight_df_nonan['manual_classification_class_number']!=EVENT_CLASSES['unclassified']]         [['event_id','manual_classification_class_number']]             .groupby('manual_classification_class_number')             .count()['event_id']
#class_sizes_ser


## In[67]:


#clusters_visualized = {}
#clusters_small = {}
#clusters_empty = []

#for cluster_number in sorted(dbscan_on_tsne_classes):
    
    #flight_df_nonan_subset = flight_df_nonan[
        #(flight_df_nonan['dbscan_tsne_y']==cluster_number) & \
        #(flight_df_nonan['manual_classification_class_number'] != EVENT_CLASSES['unclassified'])
    #]
    
    #if len(flight_df_nonan_subset) <= 0:
        #clusters_empty.append(cluster_number)
    #elif len(flight_df_nonan_subset) < small_cluster_threshold:
        #clusters_small[cluster_number] = len(flight_df_nonan_subset) 
    #else:
        #clusters_visualized[cluster_number] = len(flight_df_nonan_subset) 

#print('Num. empty clusters: ', len(clusters_empty))
#print()
#print(', '.join(str(c) for c in clusters_empty))
#print('-'*80)
#print()

#for dataset_name, dataset_dict in (('small', clusters_small), 
                                   #('visualized', clusters_visualized)):
    #print('Num. {} clusters: {}'.format(dataset_name, len(dataset_dict)))
    #print()
    
    #for cluster_number, subset_len in sorted(dataset_dict.items(), key=lambda x: x[1], reverse=True):
        #cluster_events = flight_df_nonan[flight_df_nonan['dbscan_tsne_y']==cluster_number]
        #cluster_classes_counts =             cluster_events[
                #cluster_events['manual_classification_class_number'] != EVENT_CLASSES['unclassified']
            #][['event_id', 'manual_classification_class_number']] \
                #.groupby('manual_classification_class_number').count() \
                #.sort_values('event_id',ascending=False)
        
        #print('\t{:<5d} -  {:>3} classified ev, {:>2} unique ev classes, {:>5} ev in cluster  '.format(
            #cluster_number, subset_len, len(cluster_classes_counts), len(cluster_events),
            
        #))
        #print()
        #print('\t      --   ' +
        #', '.join('{} ({})'.format(INVERSE_EVENT_CLASSES[class_number],r['event_id']) for class_number, r in \
                      #cluster_classes_counts.head(small_cluster_threshold).iterrows()),
            #'...' if len(cluster_classes_counts) > small_cluster_threshold else ''
        #)
        #print('\n')
        
    #print('-'*80)
    #print()


## In[70]:


#plt.close('all')

#cm = plt.cm.gist_rainbow
#event_classes_for_cm = [v for k, v in EVENT_CLASSES.items() if k != 'unclassified']
#cm_norm = mpl.colors.Normalize(min(event_classes_for_cm),max(event_classes_for_cm))
#unclassified_class_color = 'black'

#single_class_event_class_alpha_dict = {'unclassified': .01}
#single_class_event_class_alpha_default = 0.7

#event_class_numbers = list([v for k, v in EVENT_CLASSES.items() if k != 'unclassified'])
#event_class_numbers_range = np.max(event_class_numbers) - np.min(event_class_numbers)

#tsne_X_0_range = np.min(flight_df_nonan['tsne_X_0']), np.max(flight_df_nonan['tsne_X_0'])
#tsne_X_1_range = np.min(flight_df_nonan['tsne_X_1']), np.max(flight_df_nonan['tsne_X_1'])

#barplot_ind = np.arange(len(event_class_numbers))
## barplot_xlabels = [INVERSE_EVENT_CLASSES[v] for v in event_class_numbers]
## '#{}: {}'.format(v, INVERSE_EVENT_CLASSES[v])

#for cluster_number in sorted(dbscan_on_tsne_classes):

    #cluster_events_df = flight_df_nonan[flight_df_nonan['dbscan_tsne_y']==cluster_number]
    
    #cluster_classes_counts =         cluster_events_df[
            #cluster_events_df['manual_classification_class_number'] != EVENT_CLASSES['unclassified']
        #][['event_id', 'manual_classification_class_number']] \
            #.groupby('manual_classification_class_number').count() \
##             .sort_values('event_id',ascending=False)  # intentionally not sorted
        
    
    #flight_df_nonan_subset =         cluster_events_df[cluster_events_df['manual_classification_class_number'] != EVENT_CLASSES['unclassified']]
    
    #flight_df_nonan_subset_unclassified =         cluster_events_df[cluster_events_df['manual_classification_class_number'] == EVENT_CLASSES['unclassified']]
        
    #print('-'*80)
    #print('Cluster {:<5d} -  {:>3} classified events, {:>2} unique event classes, {:>5} events in cluster'.format(
        #cluster_number, len(flight_df_nonan_subset), len(cluster_classes_counts), len(cluster_events_df)
    #))
    #print()
    
    #if len(flight_df_nonan_subset) <= 0:
        #print('\t      --   Empty')
        #continue
    
    #print('\t      --   ' +
    #', '.join('{} (no. {}, count {})'.format(INVERSE_EVENT_CLASSES[class_number], class_number, r['event_id']) \
              #for class_number, r in cluster_classes_counts.iterrows())
    #)

    #if len(flight_df_nonan_subset) < small_cluster_threshold:
        #print('\t      --   Small cluster (IGNORED)')
        #continue

    #print('\n')
    
    #barplot_class_num_events = [0]*len(event_class_numbers)
    #barplot_class_norm_num_events = [0]*len(event_class_numbers)
    #barplot_class_norm_num_events = [0]*len(event_class_numbers)
    #barplot_class_clufrac_events = [0]*len(event_class_numbers)
    #barplot_class_norm_clufrac_events = [0]*len(event_class_numbers)
    #barplot_xlabels = []
    #for i, class_number in enumerate(event_class_numbers):
##         print('->',class_number,' ',len(cluster_classes_counts), cluster_classes_counts.index, class_number in cluster_classes_counts.index)
##         barplot_xlabels.append(INVERSE_EVENT_CLASSES[class_number])
        #if class_number not in cluster_classes_counts.index:
            #barplot_xlabels.append('')
##             barplot_xlabels[-1] += ' (0)'
            #continue
        #barplot_class_num_events[i] = cluster_classes_counts.loc[class_number]['event_id']
        ## INTENTIONALLY USING flight_df
        #barplot_class_norm_num_events[i] =             cluster_classes_counts.loc[class_number]['event_id']/len(flight_df[flight_df['manual_classification_class_number'] == class_number])
        #barplot_class_clufrac_events[i] =             cluster_classes_counts.loc[class_number]['event_id']/len(cluster_events_df)
        #barplot_class_norm_clufrac_events[i] =             barplot_class_clufrac_events[i] * class_sizes_ser.max() / class_sizes_ser[class_number]
        
        #barplot_xlabels.append(INVERSE_EVENT_CLASSES[class_number])
##         barplot_xlabels[-1] += ' ({})'.format(barplot_class_num_events[i])
    
    #for barplot_num_events_list, ylabel_extra, do_single_class in (
        #(barplot_class_num_events, '', True), 
        #(barplot_class_norm_num_events, ' (normalized)', True),
        #(barplot_class_clufrac_events, ' (normalized to all clu. events)', True),
        #(barplot_class_norm_clufrac_events, ' (wighted, normalized to all clu. events)', True)
    #):
        #if not do_single_class and np.count_nonzero(barplot_xlabels) < 2:
            #continue
        #f, ax = plt.subplots(figsize=(16,4))
        #ax.bar(barplot_ind, barplot_num_events_list)
        #ax.set_xticks(barplot_ind)
        #ax.set_xticklabels(barplot_xlabels, rotation=45, ha="right")
        #ax.set_xlabel('Class (manual_classification_class_number)')
        #ax.set_ylabel('Frequency' + ylabel_extra)
        #ax.set_title('Classes for cluster "{}"'.format(cluster_number))
        #plt.show()
    
    #f, ax = plt.subplots()
    #f.set_size_inches(16,16)
    #ax.set_xlim(*tsne_X_0_range)
    #ax.set_ylim(*tsne_X_1_range)
    
    #tsne_scatter_labels = []
    #tsne_scatter_pathcolls = []

    #for class_name, class_number in sorted(EVENT_CLASSES.items(),key=lambda x: x[1]):

        #if class_name == 'unclassified':
            #flight_df_nonan_subsubset =                 flight_df_nonan_subset_unclassified
        #else:
            #flight_df_nonan_subsubset = flight_df_nonan_subset[flight_df_nonan_subset['manual_classification_class_number']==class_number]

        #if len(flight_df_nonan_subsubset) <= 0:
            #continue
        
##         t_class_name, t_class_number, t_class_color, subset_df = \
##             class_name, class_number, cm(cm_norm(class_number)), flight_df_nonan_subsubset
        
        #if class_name in single_class_event_class_alpha_dict:
            #event_class_alpha = single_class_event_class_alpha_dict[class_name]
        #else:
            #event_class_alpha = single_class_event_class_alpha_default

        #if class_name == 'unclassified':
            #class_color = unclassified_class_color
        #else:
            #class_color = cm(cm_norm(class_number))
        
        
        #for classification_class_num, classification_class_name, subset_marker in (
            #(1, 'shower est.', 'o'), (0, 'noise est.', '${}$'.format(cluster_number))
        #):  
            #subsubset_df = flight_df_nonan_subsubset[flight_df_nonan_subsubset['extra_trees_cls_on_train_kbest400_128_est_dropna']==classification_class_num]
##                 print('    -',classification_class_name, len(subsubset_df))
            #if len(subsubset_df) <= 0:
                #continue
            #pathcoll = ax.scatter(
                #subsubset_df['tsne_X_0'], 
                #subsubset_df['tsne_X_1'], 
                #c=class_color, 
                #s=80, linewidths=0,
                #alpha=event_class_alpha,
                #marker=subset_marker
            #)

            #tsne_scatter_labels.append('{} ({})'.format(class_name, classification_class_name))
            #tsne_scatter_pathcolls.append(pathcoll)

    #ax.legend(tsne_scatter_pathcolls, tsne_scatter_labels)
    #plt.show()


## In[ ]:





## # Save classification results

## In[72]:


#flight_nonan_classified_shower_pathname = os.path.join(data_snippets_dir, 'flight_nonan_classified shower.tsv')


## In[73]:


#flight_df_nonan[flight_df_nonan['extra_trees_cls_on_train_kbest400_128_est_dropna']==1].to_csv(flight_nonan_classified_shower_pathname, sep='\t')


## In[ ]:





## In[ ]:





## In[ ]:


## TODO select clusters with positive classification (sort by the number of classifications) show distribution of event types


## In[ ]:





## In[ ]:



## flight_df_nonan_subset[['dbscan_tsne_y','manual_classification_class_number'].hist('dbscan_tsne_y', figsize=(24,4), bins=2*len(dbscan_on_tsne_classes)+1)

## plt.show()


## In[ ]:


## THIS IS NOT WHAT IS DESIRED - values should be split into features ?
## flight_nonan__cls_tsneclu_corr_df = \
##     flight_df_nonan[['dbscan_tsne_y', 'manual_classification_class_number']].corr()
## f, ax = plt.subplots(figsize=(28,22))
## plt.close('all')
## sns.heatmap(flight_nonan__cls_tsneclu_corr_df, cmap='inferno', annot=True)
## plt.show()


## In[ ]:





## In[ ]:


##     f, ax = plt.subplots()
##     f.set_size_inches(8,4)
##     flight_df_nonan_subset[['dbscan_tsne_y', 'manual_classification_class_number']].plot.bar(by='dbscan_tsne_y', ax=ax)
    


## In[ ]:


## flight_nonan__tsne__gmm_y_pred = gmm.predict(flight_df_nonan[['tsne_X_0','tsne_X_1']].values)


## In[ ]:


## flight_data__k50best_var_th_scaled_X = \
##     k50best_f_classif_selector_on_var_th_sc_train.transform(
##         var_th_selector_on_scaled_train.transform(
##             standard_scaler_on_train.transform(
##                 unl_flight_df[analyzed_common_df_columns].dropna().values)
##         )
##     )

## extra_trees_classifier_on_train_kbest50__X_flight = flight_data__k50best_var_th_scaled_X
## extra_trees_classifier_on_train_kbest50__y_flight_pred = \
##     extra_trees_classifier_on_train_kbest50.predict(extra_trees_classifier_on_train_kbest50__X_flight)


## In[ ]:




