
# coding: utf-8

# In[1]:


import sys
import os
import argparse
import getpass

app_base_dir = '/home/eusobg/EUSO-SPB/euso-spb-patt-reco-v1'
if app_base_dir not in sys.path:
    sys.path.append(app_base_dir)

import re
# import collections
import numpy as np
import psycopg2 as pg
import scipy.ndimage.filters
# import scipy.signal
# import pandas as pd
# import pandas.io.sql as psql
# import matplotlib as mpl
import matplotlib as mpl
import matplotlib.pyplot as plt
# from tqdm import tqdm
# import glob
#import ROOT
import skimage
import skimage.filters
# import skimage.restoration
# import skimage.morphology


# In[2]:

import supervised_classification as supc
import tool.acqconv

import scipy.optimize

from data_analysis_utils import get_conn
#, ensure_ext, save_csv, fig_saving_msg, print_len, save_figure


# # Loading Data

# In[4]:


def read_projections(all_rows, packet_simu_sequence_length = 35, packet_simu_data_gtu_offset = 30, max_entries=None, substract_bg=False, bgframes_length=30, bgframes_offset=0, apply_gaussian_filter=True, gaussian_filter_sigma=3):

    xy_projections = np.zeros((len(all_rows), 48, 48),dtype=np.float32)
    gtux_projections = np.zeros((len(all_rows), 48, packet_simu_sequence_length),dtype=np.float32)
    gtuy_projections = np.zeros((len(all_rows), 48, packet_simu_sequence_length),dtype=np.float32)

    for i, r in enumerate(all_rows):
        
        if max_entries is not None and max_entries >= 0 and i >= max_entries:
            break
            
        event_id, packet_id, gtu_in_packet, num_gtu, acquisition_file = r
        gtu_in_packet_corrected = max(gtu_in_packet - 4, 0)
        
        if gtu_in_packet_corrected + num_gtu < 30:
            print('#{}  ID {}  gtu_in_packet + num_gtu - 4 < 30'.format(i, event_id))
            continue
        
        if acquisition_file.endswith('.npy'):
            acquisition_arr = np.load(acquisition_file)[packet_id * 128:packet_id*128+128]
#             frames_acquisition = acquisition_arr[gtu_in_packet_corrected:packet_id * 128 + gtu_in_packet_corrected + num_gtu]                
        elif acquisition_file.endswith('.root'):
            acquisition_arr = tool.acqconv.get_frames(acquisition_file,
                                                         packet_id * 128 ,
                                                         packet_id * 128 + 127, 
                                                         entry_is_gtu_optimization=True)
        
        frames_acquisition = acquisition_arr[gtu_in_packet_corrected:gtu_in_packet_corrected + num_gtu]
        
        if substract_bg:
            avg_frames_background = np.add.reduce(acquisition_arr[bgframes_offset:bgframes_offset+bgframes_length])/min(bgframes_length,len(acquisition_arr[bgframes_offset:]))
            if apply_gaussian_filter:
                avg_frames_background = scipy.ndimage.filters.gaussian_filter(avg_frames_background, gaussian_filter_sigma)
                        
            frames_acquisition = frames_acquisition.astype(np.float32)
            
            for frame in frames_acquisition:
                frame -= avg_frames_background

        xy_proj = np.maximum.reduce(frames_acquisition)
        
        gtuy_proj = []
        for frame in acquisition_arr[packet_simu_data_gtu_offset:packet_simu_data_gtu_offset+packet_simu_sequence_length]:
            gtuy_proj.append(np.max(frame, axis=1).reshape(-1, 1))  # max in the x axis
        gtuy_proj = np.hstack(gtuy_proj)

        gtux_proj = []
        for frame in acquisition_arr[packet_simu_data_gtu_offset:packet_simu_data_gtu_offset+packet_simu_sequence_length]:
            gtux_proj.append(np.max(frame, axis=0).reshape(-1, 1))  # max the y axis
        gtux_proj = np.hstack(gtux_proj)

        xy_projections[i] = xy_proj
        gtuy_projections[i] = gtuy_proj
        gtux_projections[i] = gtux_proj

    #     m = re_acq_pathname.search(acquisition_npy)
    #     if not m:
    #         print('Unexpected source_file_acquisition_full format "{}"'.format(acquisition_npy))
    #         continue
    #     signals_npy = os.path.join(m.group(1), "simu2npy", m.group(2) + ".npy")
    #     counts_npy = os.path.join(m.group(1), "simu2npy", 'ev_{:d}_mc_1__counts.npy'.format(m.group(3)))
    #     info_txt = os.path.join(m.group(1), "simu2npy", 'ev_{:d}_mc_1__info.txt'.format(m.group(3)))

    #     if not os.path.exists(acquisition_npy):
    #         raise Exception(
    #             'Acquisition file "{}" does not exists (#{})'.format(acquisition_npy, i))
    #     if not os.path.exists(signals_npy):
    #         raise Exception('Signals file "{}" does not exists (#{}  ID {})'.format(signals_npy, i))

            
    return xy_projections, gtux_projections, gtuy_projections

def save_projections(npy_dir_pathname, prefix, xy_projections, gtux_projections, gtuy_projections):
    os.makedirs(os.path.join(npy_dir_pathname,prefix),exist_ok=True)
    np.save(os.path.join(npy_dir_pathname,prefix,'xy_projections.npy'), xy_projections)
    np.save(os.path.join(npy_dir_pathname,prefix,'gtux_projections.npy'), gtux_projections)
    np.save(os.path.join(npy_dir_pathname,prefix,'gtuy_projections.npy'), gtuy_projections)


# ## Saving projections and then loading

# In[10]:

def get_background_projections(param_visible_showers, bgframes_offset = 0, bgframes_length = 32):
    background_xy_projections = [None]*len(param_visible_showers)
    background_avg_xy_projections = [None]*len(param_visible_showers)

    # visible_showers = supc.select_events(cur, supc.get_select_simu_events_query_format(5, 999, 3, 800, 3), columns, limit=100000)[0]
    # supc.select_events(cur, supc.get_select_simu_events_query_format(5, 999, 3, 800, 3), columns, limit=5000)[0]

    print(len(param_visible_showers))
    
    for i, r in enumerate(param_visible_showers):
        event_id, packet_id, gtu_in_packet, num_gtu, acquisition_file = r
        gtu_in_packet_corrected = max(gtu_in_packet - 4, 0)

        if gtu_in_packet_corrected + num_gtu < 30:
            print('#{}  ID {}  gtu_in_packet + num_gtu - 4 < 30'.format(i, event_id))
            continue

        if acquisition_file.endswith('.npy'):
            acquisition_arr = np.load(acquisition_file)[packet_id * 128:packet_id*128+128]
    #             frames_acquisition = acquisition_arr[gtu_in_packet_corrected:packet_id * 128 + gtu_in_packet_corrected + num_gtu]                
        elif acquisition_file.endswith('.root'):
            acquisition_arr = tool.acqconv.get_frames(acquisition_file,
                                                         packet_id * 128 ,
                                                         packet_id * 128 + 127, 
                                                         entry_is_gtu_optimization=True)

        frames_acquisition = acquisition_arr[gtu_in_packet_corrected:gtu_in_packet_corrected + num_gtu]

    #     if substract_bg:
        background_avg_xy_projections[i] = np.add.reduce(acquisition_arr[bgframes_offset:bgframes_offset+bgframes_length])/min(bgframes_length,len(acquisition_arr[bgframes_offset:]))
        background_xy_projections[i] = np.maximum.reduce(acquisition_arr[bgframes_offset:bgframes_offset+bgframes_length])

        if i % 500 == 0:
            print(i)

    return background_xy_projections, background_avg_xy_projections



def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = scipy.ndimage.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


def vis_match(feats, ref_feats):
    cols = 4
    rows = int(np.ceil(len(feats)/cols))
#     fig, axs = plt.subplots(rows, cols)
#     axs_flattened = axs.flatten()
#     fig.set_size_inches(cols*5.5,rows*4*0.85)
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        title = "e: {:.5f}".format(error)
        if error < min_error:
            min_error = error
            min_i = i
            title += " | m"
        
        print(i, title)
#         axs_flattened[k].imshow( (feats - ref_feats[i, :])**2 )
#         axs_flattened[k].set_title(title)
        
    return min_i

def vis_feats(image, kernels, kernel_labels=[]):
    cols = 4
    rows = int(np.ceil(len(kernels)/cols))
    fig, axs = plt.subplots(rows, cols)
    fig.set_size_inches(cols*5.5,rows*4*0.85)
    axs_flattened = axs.flatten()
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = scipy.ndimage.convolve(image, kernel, mode='wrap')
#         feats[k, 0] = filtered.mean()
#         feats[k, 1] = filtered.var()
        axs_flattened[k].imshow(filtered)
        title = "m: {:.5f} v: {:.5f}".format(filtered.mean(), filtered.var())
        if k < len(kernel_labels):
            title += " | t: {:.3f} s: {:.3f} f: {:.3f}".format(*kernel_labels[k])
        axs_flattened[k].set_title(title)
    return fig, axs

def calc_gabor_kernels(ntheta=4, sigmas=(1,3), frequencies=(0.05, 0.25)):
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in sigmas:
            for frequency in frequencies:
                kernel = np.real(skimage.filters.gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels

def get_gabor_kernel_labels(ntheta=4, sigmas=(1,3), frequencies=(0.05, 0.25)):
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in sigmas:
            for frequency in frequencies:
                kernel = (frequency, theta, sigma)
                kernels.append(kernel)
    return kernels

def vis_gabor_kernels(kernels, labels=[]):
    cols = 4
    rows = int(np.ceil(len(kernels)/cols))
    fig, axs = plt.subplots(rows, cols)
    fig.set_size_inches(cols*5,rows*4*0.85)
    axs_flattened = axs.flatten()
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        axs_flattened[k].imshow(kernel)
        if k < len(labels):
            axs_flattened[k].set_title("t: {:.3f} s: {:.3f} f: {:.3f}".format(*labels[k]))
    return fig, axs


def classify_projections(xy_projections, ref_feats, gabor_kernels, exclude=[]):
    print("len(projections)", len(xy_projections))
    shower_classes = np.zeros(len(xy_projections)) # _bgsub
    for k, xy_projection in enumerate(xy_projections):
        if k in exclude: 
            continue
        c_xy_proj_w_bg = xy_projection * (xy_projection>0)
        c_xy_proj_w_bg_cent = (c_xy_proj_w_bg - c_xy_proj_w_bg.mean() )
        c_xy_proj_w_bg_max_norm = c_xy_proj_w_bg_cent/np.max(np.abs(c_xy_proj_w_bg_cent))
        feats = compute_feats(c_xy_proj_w_bg_max_norm, gabor_kernels)
        shower_classes[k] = match(feats, ref_feats)
        if k % 500 == 0:
            print(k)
    return shower_classes

def main(argv):
    
    args_parser = argparse.ArgumentParser(description='Gabor filter classification')
    args_parser.add_argument('-d', '--dbname', default='eusospb_data')
    args_parser.add_argument('-U', '--user', default='eusospb')
    args_parser.add_argument('--password')
    args_parser.add_argument('-s', '--host', default='localhost')
    args_parser.add_argument('-n', '--npy-dir', default='/media/node15_data2/nn_training_data', help="Numpy images are loaded from this directory (default: /media/node15_data2/nn_training_data)")
    args_parser.add_argument('-o', '--stats-dir', default='gabor_classification_stats', help="Directory where tsv stats are saved (default: gabor_classification_stats)")
    
    args_parser.add_argument('--every-nth', type=int, default=4, help='Every n-th used for training (default: 4)')
    args_parser.add_argument('--first-n', type=int, default=0, help='First n used for training (default: 4)')
    args_parser.add_argument('--last-n', type=int, default=-1, help='Last n used for training (default: 4)')
    
    args = args_parser.parse_args(argv)

    npy_dir_pathname = args.npy_dir
    stats_out_dir = args.stats_dir # '/home/eusobg/EUSO-SPB/euso-spb-patt-reco-v1/notebooks/gabor_classification_stats' 
    every_nth = args.every_nth # 4
    first_n = args.first_n # 0
    last_n = args.last_n # -1

    password = args.password
    if not password:
        password = getpass.getpass()

    con, cur = get_conn(dbname=args.dbname, user=args.user, host=args.host, password=password)
    cur = con.cursor()
    # In[5]:

    columns = ['event_id', 'packet_id', 'gtu_in_packet', 'num_gtu','source_file_acquisition_full']
    #visible_showers = supervised_classification.select_training_data__visible_showers(cur, columns);
    less34_visible_showers = supc.select_events(cur, supc.get_query__select_simu_events(3, 4, 3, 800, 3), columns, limit=100000)[0]
    visible_showers = supc.select_events(cur, supc.get_query__select_simu_events(5, 999, 3, 800, 3), columns, limit=100000)[0]
    invisible_showers = supc.select_training_data__invisible_showers(cur, columns)
    low_energy_in_pmt = supc.select_training_data__low_energy_in_pmt(cur, columns) # maybe too many gtu
    led = supc.select_training_data__led(cur, columns)

    # save_projections(npy_dir_pathname, 'less34_visible_showers', *read_projections(less34_visible_showers, max_entries=100000))
    # save_projections(npy_dir_pathname, 'visible_showers', *read_projections(visible_showers, max_entries=100000))
    # save_projections(npy_dir_pathname, 'invisible_showers', *read_projections(invisible_showers, max_entries=100000))
    # save_projections(npy_dir_pathname, 'low_energy_in_pmt', *read_projections(low_energy_in_pmt, max_entries=100000))
    # save_projections(npy_dir_pathname, 'led', *read_projections(led, max_entries=100000))

    # save_projections(npy_dir_pathname, 'visible_showers_bgsub_30gtu_sigma3', *read_projections(visible_showers, max_entries=100000, substract_bg=True))
    # save_projections(npy_dir_pathname, 'invisible_showers_bgsub_30gtu_sigma3', *read_projections(invisible_showers, max_entries=100000, substract_bg=True))
    # save_projections(npy_dir_pathname, 'low_energy_in_pmt_bgsub_30gtu_sigma3', *read_projections(low_energy_in_pmt, max_entries=100000, substract_bg=True))
    # save_projections(npy_dir_pathname, 'led_bgsub_30gtu_sigma3', *read_projections(led, max_entries=100000, substract_bg=True))
    # save_projections(npy_dir_pathname, 'visible_showers_bgsub_30gtu_nogauss', *read_projections(visible_showers, max_entries=10, substract_bg=True, apply_gaussian_filter=False))
    # save_projections(npy_dir_pathname, 'invisible_showers_bgsub_30gtu_nogauss', *read_projections(invisible_showers, max_entries=10, substract_bg=True, apply_gaussian_filter=False))
    # save_projections(npy_dir_pathname, 'low_energy_in_pmt_bgsub_30gtu_nogauss', *read_projections(low_energy_in_pmt, max_entries=10, substract_bg=True, apply_gaussian_filter=False))
    # save_projections(npy_dir_pathname, 'led_bgsub_30gtu_nogauss', *read_projections(led, max_entries=10, substract_bg=True, apply_gaussian_filter=False))

    gtux_projections_bgsub = np.load(os.path.join(npy_dir_pathname,'visible_showers_bgsub_30gtu_sigma3','gtux_projections.npy'))
    xy_projections_bgsub = np.load(os.path.join(npy_dir_pathname,'visible_showers_bgsub_30gtu_sigma3','xy_projections.npy'))
    gtuy_projections_bgsub = np.load(os.path.join(npy_dir_pathname,'visible_showers_bgsub_30gtu_sigma3','gtuy_projections.npy'))
    # gtux_projections_bgsub = np.load(os.path.join(npy_dir_pathname,'visible_showers_bgsub_30gtu_nogauss','gtux_projections.npy'))
    # xy_projections_bgsub = np.load(os.path.join(npy_dir_pathname,'visible_showers_bgsub_30gtu_nogauss','xy_projections.npy'))
    # gtuy_projections_bgsub = np.load(os.path.join(npy_dir_pathname,'visible_showers_bgsub_30gtu_nogauss','gtuy_projections.npy'))


    gtux_projections = np.load(os.path.join(npy_dir_pathname,'visible_showers','gtux_projections.npy'))
    xy_projections = np.load(os.path.join(npy_dir_pathname,'visible_showers','xy_projections.npy'))
    gtuy_projections = np.load(os.path.join(npy_dir_pathname,'visible_showers','gtuy_projections.npy'))

    less34_gtux_projections = np.load(os.path.join(npy_dir_pathname,'less34_visible_showers','gtux_projections.npy'))
    less34_xy_projections = np.load(os.path.join(npy_dir_pathname,'less34_visible_showers','xy_projections.npy'))
    less34_gtuy_projections = np.load(os.path.join(npy_dir_pathname,'less34_visible_showers','gtuy_projections.npy'))


    # background_xy_projections = [None]*len(visible_showers)
    # background_avg_xy_projections = [None]*len(visible_showers)

    # print( len(visible_showers) ,  len(less34_visible_showers) )

    background_xy_projections, background_avg_xy_projections = get_background_projections(visible_showers)
    less34_background_xy_projections, less34_background_avg_xy_projections = get_background_projections(less34_visible_showers)

    # ## Simu signals

    if last_n < 0:
        last_n = len(visible_showers)

    simu_xy_projections = [None]*len(visible_showers)
    
    os.makedirs(stats_out_dir, exist_ok=True)
    outfile_name = 'gabor_classification_{}_{}_{}.tsv'.format(first_n, last_n, every_nth)
    outfile_path = os.path.join(stats_out_dir, outfile_name)
    
    print("Outfile path: {}".format(outfile_path))

    stats_header = [
        'n',
        'positive',
        'negative',
        'true_positive',
        'true_negative',
        'false_positive',
        'false_negative',
        'sensitivity',
        'specificity',
        'precision',
        'negative_predictive_value',
        'miss_rate',
        'fall_out',
        'false_discovery_rate',
        'false_omission_rate',
        'accuracy'
    ]
    
    with open(outfile_path, 'w') as f:
        print('\t'.join(stats_header))

    for n in range(first_n, last_n):
        if n % every_nth != 0:
            continue
    
        base_xy_proj_w_bg = xy_projections[n] * (xy_projections[n]>0)
        base_xy_proj_w_bg_cent = (base_xy_proj_w_bg - base_xy_proj_w_bg.mean() )
        base_xy_proj_w_bg_max_norm = base_xy_proj_w_bg_cent/np.max(np.abs(base_xy_proj_w_bg_cent))


        bg_xy_proj = background_xy_projections[n] #np.maximum.reduce(frames_background)
        bg_xy_proj_cent = bg_xy_proj - np.mean(bg_xy_proj)
        bg_xy_proj_cent_max_norm = bg_xy_proj_cent/np.max(np.abs(bg_xy_proj_cent))
        #bg_xy_proj_cent_std_norm = bg_xy_proj_cent/np.std(bg_xy_proj_cent)

        re_acq_pathname = re.compile(r"^(.+)\/npyconv\/(ev_(\d+)_mc_1__signals).+$")
        # visible_showers = supc.select_events(cur, supc.get_select_simu_events_query_format(5, 999, 3, 800, 3), columns, limit=100000)[0]
        # supc.select_events(cur, supc.get_select_simu_events_query_format(5, 999, 3, 800, 3), columns, limit=5000)[0]

        #for i, r in enumerate(visible_showers): ....

        # gabor_kernels = calc_gabor_kernels(6,(1,2,3),(0.05, 0.25, 0.55))
        # gabor_kernel_labels = get_gabor_kernel_labels(6,(1,2,3),(0.05, 0.25, 0.55))
        gabor_kernels = calc_gabor_kernels(6,(1,2,3),(0.05, 0.25, 0.55))
        gabor_kernel_labels = get_gabor_kernel_labels(6,(1,2,3),(0.05, 0.25, 0.55))


        ## In[172]: ...

        ref_feats = np.zeros((2, len(gabor_kernels), 2), dtype=np.double)
        ref_feats[1, :, :] = compute_feats(base_xy_proj_w_bg_max_norm, gabor_kernels)  # base_xy_proj_max_norm
        ref_feats[0, :, :] = compute_feats(bg_xy_proj_cent_max_norm, gabor_kernels)

        visible_shower_classes = classify_projections(xy_projections, ref_feats, gabor_kernels, exclude=[n])

        print("visible_shower_classes")
        print("num class 0:", len(visible_shower_classes) - np.count_nonzero(visible_shower_classes) )
        print("num class 1:", np.count_nonzero(visible_shower_classes) )

        print("error: ",(len(visible_shower_classes) - np.count_nonzero(visible_shower_classes))/len(visible_shower_classes))
        print("efficiency: ",np.count_nonzero(visible_shower_classes)/len(visible_shower_classes))

        less34_visible_shower_classes = classify_projections(less34_xy_projections, ref_feats, gabor_kernels, exclude=[])

        print("less34_visible_shower_classes")
        print("num class 0:", len(less34_visible_shower_classes) - np.count_nonzero(less34_visible_shower_classes) )
        print("num class 1:", np.count_nonzero(less34_visible_shower_classes) )

        background_classes = classify_projections(background_xy_projections, ref_feats, gabor_kernels, exclude=[n])
        
        print("background_classes")
        print("num class 0:", len(background_classes) - np.count_nonzero(background_classes) )
        print("num class 1:", np.count_nonzero(background_classes) )

        less34_background_classes = classify_projections(less34_background_xy_projections, ref_feats, gabor_kernels, [])

        print("less34_background_classes")
        print("num class 0:", len(less34_background_classes) - np.count_nonzero(less34_background_classes) )
        print("num class 1:", np.count_nonzero(less34_background_classes) )
        print("error: ",(np.count_nonzero(background_classes))/len(background_classes))
        print("efficiency: ",(len(background_classes) - np.count_nonzero(background_classes))/len(background_classes))

        num_all_samples = len(visible_shower_classes)+len(background_classes)
        num_all_samples_w_less34 = num_all_samples + len(less34_visible_shower_classes)+len(less34_background_classes)

        num_valid_classifications = np.count_nonzero(visible_shower_classes) + (len(background_classes) - np.count_nonzero(background_classes))
        num_valid_classifications_w_less34 = num_valid_classifications + np.count_nonzero(less34_visible_shower_classes) + (len(less34_background_classes) - np.count_nonzero(less34_background_classes))

        # print("error: ",( np.count_nonzero(background_classes) + len(visible_shower_classes) - np.count_nonzero(visible_shower_classes))/(l)) 
        print("num_all_samples")
        print("efficiency: ",num_valid_classifications/num_all_samples )
        print("num_all_samples_w_less34")
        print("efficiency with less34: ",num_valid_classifications_w_less34/num_all_samples_w_less34 )


        # In[191]:

        positive = len(visible_shower_classes)
        negative = len(background_classes)
        true_positive = np.count_nonzero(visible_shower_classes)
        true_negative = (len(background_classes) - np.count_nonzero(background_classes))
        false_negative = positive - true_positive
        false_positive = negative - true_negative

        # sensitivity, recall, hit rate, or true positive rate (TPR)
        sensitivity = true_positive/positive  
        # specificity or true negative rate (TNR)
        specificity = true_negative/negative
        # precision or positive predictive value (PPV)
        precision = true_positive/(true_positive+false_positive)
        # negative predictive value (NPV)
        negative_predictive_value = true_negative/(true_negative+false_negative)
        # miss rate or false negative rate (FNR)
        miss_rate = false_negative/positive
        # fall-out or false positive rate (FPR)
        fall_out = false_positive/negative
        # false discovery rate (FDR)
        false_discovery_rate = false_positive / (false_positive+true_positive)
        # false omission rate (FOR)
        false_omission_rate = false_negative / (false_negative+true_negative)
        # accuracy (ACC)
        accuracy = (true_positive+true_negative)/(positive+negative)

        stats = {
            'positive': positive,
            'negative': negative,
            'true_positive': true_positive,
            'true_negative': true_negative,
            'false_positive': false_positive,
            'false_negative': false_negative,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'negative_predictive_value': negative_predictive_value,
            'miss_rate': miss_rate,
            'fall_out': fall_out,
            'false_discovery_rate': false_discovery_rate,
            'false_omission_rate': false_omission_rate,
            'accuracy': accuracy
        }
        
        for k,v in stats.items():
            print("{}\t{}".format(k,v))
        
        with open(outfile_path, 'w') as f:
            print('\t'.join([n] + [str(v) in stats.items()]), file=f)



if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])



