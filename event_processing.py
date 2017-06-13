import numpy as np
import os
import argparse
import sys
import ROOT
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from eusotrees.exptree import ExpTree
import collections
from enum import Enum
import itertools

from event_processing import *
from event_visualization import *
from event_reading import *

import skimage.transform

class EventProcessingParams(object):
    triggered_pixels_group_max_gap = 5

    triggered_pixels_ht_size = 1
    triggered_pixels_ht_phi_num_steps = 90 # 2 deg per step
    triggered_pixels_ht_rho_step = 2

    x_y_neighbour_selection_rules = [NeighbourSelectionRules(3, .3, False),
                           NeighbourSelectionRules(3, 1, True),
                           NeighbourSelectionRules(1, .9, True)]

    x_y_ht_size = .5
    x_y_ht_phi_num_steps = 90 # 2 deg per step
    x_y_ht_rho_step = 2

    x_y_ht_peak_threshold_frac_of_max = .85
    x_y_ht_peak_gap = 3
    x_y_ht_global_peak_threshold_frac_of_max = .95


def translate_struct(struct, trans_func):
    """
    Maps all Tasks in a structured data object to their .output().
    """
    if isinstance(struct, dict):
        r = {}
        for k, v in struct.items():
            r[k] = translate_struct(v, trans_func)
        return r
    elif isinstance(struct, (list, tuple)):
        try:
            s = list(struct)
        except TypeError:
            raise Exception('Cannot map %s to list' % str(struct))
        return [translate_struct(r, trans_func) for r in s]

    return trans_func(struct)


def translate_dict_keys(d, trans_func):
    od = {}
    for k,v in d.items():
        od[trans_func(k)] = v
    return od


def get_field_positions(arr, cond_func):
    o = []
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        if cond_func(arr[it.multi_index]):
            o.append(it.multi_index)
        it.iternext()
    return o

def find_n_max_values(arr, n):
    # inefficient !!!
    if n <= 0:
        raise Exception("Parameter value n has to be larger than 0")
    max_value_positions = [None]*n #collections.deque(maxlen=n)
    it = np.nditer(arr, flags=['multi_index'])
    i = 0
    while i < n and not it.finished:
        max_value_positions[i] = it.multi_index
        it.iternext()
        i += 1
    it.reset()
    while not it.finished:
        append_index = False
        smallest_val_pos = None
        for i, pos in enumerate(max_value_positions):
            if arr[it.multi_index] > arr[pos]:
                append_index = True
                if smallest_val_pos is None or arr[pos] < arr[max_value_positions[smallest_val_pos]]:
                    smallest_val_pos = i
        if append_index:
            max_value_positions[smallest_val_pos] = it.multi_index
        it.iternext()
    return max_value_positions


def split_all_filed_values_to_groups(arr):
    groups = {}

    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        if arr[it.multi_index] not in groups:
            groups[arr[it.multi_index]] = []
        groups[arr[it.multi_index]].append(it.multi_index)
        it.iternext()

    return groups


def split_values_to_groups(points, arr):
    groups = {}

    for point in points:
        if arr[point] not in groups:
            groups[arr[point]] = []
        groups[arr[point]].append(point)

    return groups


def key_vals2val_keys(in_dict, exclusive=False):
    out_dict = {}
    for k,l in in_dict.items():
        for v in l:
            if exclusive:
                out_dict[v] = k
            else:
                if v not in out_dict:
                    out_dict[v] = []
                out_dict[v].append(v)
    return out_dict


def gray_hough_line(image, size=2, phi_range=np.linspace(0, np.pi, 180), rho_step=1):
    max_distance = np.hypot(image.shape[0], image.shape[1])
    num_rho = int(np.ceil(max_distance*2/rho_step))
    rho_correction_lower = -size + max_distance
    rho_correction_upper = size  + max_distance
    #phi_range = phi_range - np.pi / 2
    acc_matrix = np.zeros((num_rho, len(phi_range)))
    # rho_acc_matrix = np.zeros((num_rho, len(phi_range)))
    # nc_acc_matrix = np.zeros((num_rho, len(phi_range)))

    # phi_corr_arr = np.ones((100,len(phi_range)))

    max_acc_matrix_val = 0

    phi_corr = 1
    for phi_index, phi in enumerate(phi_range):
        # print("hough > phi = {} ({})".format(np.rad2deg(phi), phi_index))

        phi_norm_pi_over_2 = (phi - np.floor(phi/(np.pi/2))*np.pi/2)
        if phi_norm_pi_over_2 <= np.pi/4:
            phi_corr = image.shape[1] / np.sqrt(image.shape[1] ** 2 + (image.shape[1] * np.tan( phi_norm_pi_over_2 )) ** 2)
        else:
            phi_corr = image.shape[0] / np.sqrt(image.shape[0] ** 2 + (image.shape[0] * np.tan( np.pi/2 - phi_norm_pi_over_2 )) ** 2) #np.sqrt(image.shape[0] ** 2 + (image.shape[0] / np.tan( phi_norm_pi_over_2 - np.pi/4 )) ** 2) / image.shape[1]

        # normalization vis would go here

        # phi_corr = 1 #(np.cos(phi*4) + 1)/2 + 1
        for i in range(0, len(image)): # row, y-axis
            for j in range(0, len(image[i])): # col, x-axis
                rho = j*np.cos(phi) + i*np.sin(phi)
                #
                # if rho < 0:
                #     print("rho =",rho, "phi =", phi, "phi_index =", phi_index, "i =", i, "j=", j)

                rho_index_lower = int((rho+rho_correction_lower) // rho_step)
                rho_index_upper = int((rho+rho_correction_upper) // rho_step)

                if rho_index_lower < 0:
                    print("rho_index_lower < 0 : rho_index_lower=", rho_index_lower)
                    rho_index_lower = 0

                if rho_index_upper > num_rho:
                    print("rho_index_upper > num_rho : rho_index_upper=", rho_index_upper,"num_rho=",num_rho)
                    rho_index_upper = num_rho

                for rho_index in range(rho_index_lower,rho_index_upper):
                    acc_matrix[rho_index, phi_index] += image[i,j] * phi_corr
                    # if acc_matrix[rho_index, phi_index] > max_acc_matrix_val:
                    #     max_acc_matrix_val = acc_matrix[rho_index, phi_index]
                    #     print("max_acc_matrix_val=",max_acc_matrix_val,"rho=",rho,"phi=",phi)
                    # rho += rho_step
                    # rho and nc matrixes would go hrer




    acc_matrix_max_pos = np.unravel_index(acc_matrix.argmax(), acc_matrix.shape)
    acc_matrix_max = acc_matrix[acc_matrix_max_pos]

    acc_matrix_max_rho_base = rho_step*acc_matrix_max_pos[0]
    # acc_matrix_max_rho_range = (acc_matrix_max_rho_base - rho_correction_lower, acc_matrix_max_rho_base - rho_correction_upper)
    acc_matrix_max_rho_range = [acc_matrix_max_rho_base - max_distance]
    acc_matrix_max_phi = phi_range[acc_matrix_max_pos[1]]


    print("acc_matrix: max={}, max_row={} ({}) , max_col={} ({})"
          .format(acc_matrix_max,
                  acc_matrix_max_pos[0], acc_matrix_max_rho_range[0], #acc_matrix_max_rho_range[1],
                  acc_matrix_max_pos[1], np.rad2deg(acc_matrix_max_phi) ))

    #  ({} = {}*{} - ({} = -{} + {}) + {}/2)
    #        rho_step,rho_index, rho_correction_lower, size, max_distance, size,


    # # fig2, (ax1, ax1b, ax2) = plt.subplots(3)
    # fig2, ax1 = plt.subplots(1)
    #
    # ax1.imshow(acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), -max_distance, max_distance], aspect='auto')
    # # ax1b.imshow(rho_acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), -max_distance, max_distance], aspect='auto')
    # # ax2.imshow(nc_acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), -max_distance, max_distance], aspect='auto')
    #
    # # fig3, ax3 = plt.subplots(1)
    # # cax3 = ax3.imshow(acc_matrix, aspect='auto')
    # # fig3.colorbar(cax3)
    #
    # fig4, ax4 = plt.subplots(1)
    # cax4 = ax4.imshow(image, aspect='auto', extent=[0, image.shape[1], image.shape[0], 0])
    # ax4.set_title("Hough input img (phi normalization)")
    # # y0 = (acc_matrix_max_rho - 0 * np.cos(acc_matrix_max_phi)) / np.sin(angle)
    # # y1 = (acc_matrix_max_rho - image.shape[1] * np.cos(angle)) / np.sin(angle)
    #
    # for acc_matrix_max_rho in acc_matrix_max_rho_range:
    #     print(acc_matrix_max_rho)
    #
    #     p = np.zeros((2,2))
    #
    #     p[0,1] = x0 = 0
    #     p[0,0] = y0 = acc_matrix_max_rho / np.sin(acc_matrix_max_phi)
    #
    #     p[1,1] = x1 = image.shape[0]
    #     p[1,0] = y1 = (acc_matrix_max_rho - image.shape[1] * np.cos(acc_matrix_max_phi)) / np.sin(acc_matrix_max_phi)
    #
    #     for i in range(0,len(p)):
    #         if p[i,0] < 0:
    #             p[i,0] = 0  # y
    #             p[i,1] = acc_matrix_max_rho/np.cos(acc_matrix_max_phi) # x
    #         elif p[i,0] > image.shape[0]:
    #             p[i,0] = image.shape[0] # y
    #             p[i,1] = (acc_matrix_max_rho - p[i,0]*np.sin(acc_matrix_max_phi))/np.cos(acc_matrix_max_phi) # x
    #
    #
    #     print("line (y,x) [{},{}] , [{},{}]".format(p[0,0],p[0,1],p[1,0],p[1,1]))
    #
    #     ax4.plot((p[:,1]), (p[:,0]), '-g')



    return acc_matrix, max_distance, (-max_distance, max_distance, rho_step), phi_range


def visualize_hough_space(acc_matrix, phi_range, rho_range_opts):

    #  ({} = {}*{} - ({} = -{} + {}) + {}/2)
    #        rho_step,rho_index, rho_correction_lower, size, max_distance, size,


    # fig2, (ax1, ax1b, ax2) = plt.subplots(3)
    fig2, ax1 = plt.subplots(1)

    cax1 = ax1.imshow(acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), rho_range_opts[0], rho_range_opts[1]], aspect='auto')
    # ax1b.imshow(rho_acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), -max_distance, max_distance], aspect='auto')
    # ax2.imshow(nc_acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), -max_distance, max_distance], aspect='auto')

    # fig3, ax3 = plt.subplots(1)
    # cax3 = ax3.imshow(acc_matrix, aspect='auto')
    # fig3.colorbar(cax3)

    return fig2, ax1, cax1


def hough_space_index_to_val_single(index, phi_range, rho_range_opts):
    return (rho_range_opts[0] + rho_range_opts[2] * index[0] + rho_range_opts[2], phi_range[index[1]]) # TODO justification for  `+ rho_range_opts[2]`


def hough_space_index_to_val(indexes, phi_range, rho_range_opts):
    o = []
    for index in indexes:
        o.append(hough_space_index_to_val_single(index, phi_range, rho_range_opts))
    return o


def visualize_hough_lines(image, lines, title=None, value_lines_groups=None):
    fig4, ax4 = plt.subplots(1)
    cax4 = ax4.imshow(image, aspect='auto', extent=[0, image.shape[1], image.shape[0], 0])
    if title:
        ax4.set_title(title) #"Hough input img (phi normalization)"
    # y0 = (acc_matrix_max_rho - 0 * np.cos(acc_matrix_max_phi)) / np.sin(angle)
    # y1 = (acc_matrix_max_rho - image.shape[1] * np.cos(angle)) / np.sin(angle)

    lines_colors = None
    if value_lines_groups is not None:
        # value_lines_groups_cpy = collections.OrderedDict(value_lines_groups)
        cmap = plt.get_cmap("tab20c")
        max_key = max(value_lines_groups.keys(), key=int)
        lines_colors = translate_struct(key_vals2val_keys(value_lines_groups, exclusive=True), lambda v: cmap(v/max_key))

    print(lines_colors)

    for line in lines:
        p = np.zeros((2,2))

        phi = line[1]
        rho = line[0]

        p[0, 1] = 0
        if phi != 0:
            p[0, 0] = rho / np.sin(phi)
        else:
            p[0, 0] = -1

        p[1, 1] = image.shape[0]
        if phi != 0:
            p[1, 0] = (rho - image.shape[1] * np.cos(phi)) / np.sin(phi)
        else:
            p[1, 0] = image.shape[0] + 1

        for pi in range(0, len(p)):
            if p[pi, 0] < 0:
                p[pi, 0] = 0  # y
                p[pi, 1] = rho / np.cos(phi)  # x
            elif p[pi, 0] > image.shape[0]:
                p[pi, 0] = image.shape[0]  # y
                p[pi, 1] = (rho - p[pi, 0] * np.sin(phi)) / np.cos(phi)  # x

        print("line (y,x) [{},{}] , [{},{}]".format(p[0,0],p[0,1],p[1,0],p[1,1]))

        if lines_colors is  not None and line in lines_colors:
            ax4.plot((p[:,1]), (p[:,0]), '-', color=lines_colors[line])
        else:
            ax4.plot((p[:,1]), (p[:,0]), '-r')


class NeighbourSelectionRules:
    max_gap = 3
    val_ratio_thr = 1
    grow = True

    def __init__(self, max_gap=3, val_ratio_thr=1, grow=True):
        self.max_gap = max_gap
        self.val_ratio_thr = val_ratio_thr
        self.grow = grow


def find_pixel_clusters(image, max_gap=3):
    clusters = {}

    visited_neighbourhood = np.zeros_like(image, dtype=np.bool)

    for cluster_seed_i in range(image.shape[0]):
        for cluster_seed_j in range(image.shape[1]):
            if image[cluster_seed_i, cluster_seed_j] == 0:
                continue;

            if visited_neighbourhood[cluster_seed_i,cluster_seed_j]:
                continue

            cluster_matrix = np.zeros_like(image, dtype=np.bool)
            clusters[(cluster_seed_i,cluster_seed_j)] = cluster_matrix
            # similar to select_neighbours

            seed_points = [(cluster_seed_i, cluster_seed_j)]

            while seed_points:
                seed_i, seed_j = seed_points.pop()
                i_start = max(seed_i - max_gap, 0)
                i_end = min(seed_i + max_gap, image.shape[0])
                j_start = max(seed_j - max_gap, 0)
                j_end = min(seed_j + max_gap, image.shape[1])

                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        # if i == seed_i and j == seed_j:
                        #     continue
                        if not visited_neighbourhood[i,j]:
                            # todo add option to select only neighbours of initial seeds
                            if image[i, j] != 0:
                                seed_points.append((i,j))
                            cluster_matrix[i,j] = True
                            visited_neighbourhood[i,j] = True

    return clusters


def find_minimal_dimensions(cluster_im):
    first_row = -1
    first_col = -1
    last_row = -1
    last_col = -1
    for i, j in itertools.product(range(cluster_im.shape[0]), range(cluster_im.shape[1])):
        if cluster_im[i, j]:
            if first_row < 0:
                first_row = i
            last_row = i
            if first_col < 0 or j < first_col:
                first_col = j
            if last_col < 0 or j > last_col:
                last_col = j

    assert first_row >= 0 and last_row >=0 and first_col >= 0 and last_col >= 0

    return (last_row-first_row, last_col-first_col)


# not optimal implementation
def select_neighbours(initial_seed_points, image, selections=[NeighbourSelectionRules(3, 1, True)]):
    # seed_points iterable of pairs
    # presuming 2d matrix

    # distance_counter reset - if examined point is seed point
    # similar of higher intensity increases search distance

    if not initial_seed_points:
        raise Exception("initial_seed_points cannot be empty")

    if len(image.shape) != 2:
        raise Exception("unexpected image shape")

    visited_neighbourhood = []
    for _ in selections:
        visited_neighbourhood.append(np.zeros_like(image, dtype=np.bool))
    out_neighbourhood = np.zeros_like(image, dtype=np.bool)

    # i - row
    # j - column

    individual_neighbourhoods = {}

    for seed_i, seed_j in initial_seed_points:
        out_neighbourhood[seed_i, seed_j] = True
        individual_neighbourhoods[(seed_i, seed_j)] = None

    seed_points = list(initial_seed_points)
    last_initial_seed_point = seed_points[-1]
    individual_neighbourhoods[last_initial_seed_point] = np.zeros_like(image)

    while seed_points:
        seed_i, seed_j = seed_points.pop()
        if (seed_i, seed_j) in individual_neighbourhoods and individual_neighbourhoods[(seed_i,seed_j)] is None:    #TODO
            last_initial_seed_point = (seed_i, seed_j)
            individual_neighbourhoods[last_initial_seed_point] = np.zeros_like(image)

        # 3 from seed included
        # v/this_v > thr => new seed

        # visited_neighbourhood[seed_i, seed_j] = True
        out_neighbourhood[seed_i, seed_j] = True
        individual_neighbourhoods[last_initial_seed_point][seed_i,seed_j] = True

        for si, selection in enumerate(selections):
            visited_neighbourhood[si][seed_i, seed_j] = True
            # out_neighbourhood[seed_i, seed_j] = True

            i_start = max(seed_i - selection.max_gap, 0)
            i_end = min(seed_i + selection.max_gap, image.shape[0])
            j_start = max(seed_j - selection.max_gap, 0)
            j_end = min(seed_j + selection.max_gap, image.shape[1])

            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    # if i == seed_i and j == seed_j:
                    #     continue
                    if not visited_neighbourhood[si][i,j]:
                        # todo add option to select only neighbours of initial seeds
                        if  (image[seed_i, seed_j]==0 or image[i,j]/image[seed_i, seed_j] > selection.val_ratio_thr) and (i,j) not in seed_points:
                            # print(image[i,j], image[seed_i, seed_j], image[i,j]/image[seed_i, seed_j])
                            if selection.grow:
                                seed_points.append((i,j))
                            out_neighbourhood[i,j] = True # prevent being added as a seed point again
                            # individual_neighbourhoods[last_initial_seed_point][i,j] = True
                        # # elif (seed_i,seed_j) in initial_seed_points:
                        # elif (i,j) in individual_neighbourhoods and individual_neighbourhoods[(i,j)] is None:
                        #     individual_neighbourhoods[(i,j)] = individual_neighbourhoods[last_initial_seed_point]
                        #     # individual_neighbourhoods[last_initial_seed_point][i,j] = True
                        #     # out_neighbourhood[i,j] = True
                        #     # individual_neighbourhoods[(i,j)][i, j] = True

    # fig4, ax4 = plt.subplots(1)
    # cax4 = ax4.imshow(out_neighbourhood*1, aspect='auto', extent=[0, image.shape[1], image.shape[0], 0])
    # ax4.set_title("Neighbours")

    # this seems to be pointless
    # seed_groups = []
    # k_list = list(individual_neighbourhoods.keys())
    #
    # grouped_neighbourhoods = {}
    #
    # while k_list:
    #     l1 = k_list.pop()
    #     grouped_neighbourhoods_list = []
    #     for i,l2 in enumerate(k_list):
    #         for i, j in product(range(image.shape[0]), range(image.shape[1])):
    #             if individual_neighbourhoods[l1][i,j] and individual_neighbourhoods[l1][i,j] == individual_neighbourhoods[l2][i,j]:
    #                 individual_neighbourhoods[l1] += individual_neighbourhoods[l2]
    #                 individual_neighbourhoods.pop(l2)
    #                 grouped_neighbourhoods_list.append(i)
    #                 break
    #     seeds = [l1]
    #     if grouped_neighbourhoods_list:
    #         for i in grouped_neighbourhoods_list:
    #             l2 = k_list.pop(i)
    #             seeds.append(l2)
    #     grouped_neighbourhoods[tuple(seeds)] = individual_neighbourhoods[l1]
    #     individual_neighbourhoods.pop(l1)
    #     seed_groups.append(seeds)
    # #
    # todo group if another seed is within gap



    # last_individual_neighbourhood = None
    # for seed, individual_neighbourhood in grouped_neighbourhoods.items():
    #     print(seed,individual_neighbourhood)
    #     if individual_neighbourhood is not None and last_individual_neighbourhood is not individual_neighbourhood:
    #         fig, ax = plt.subplots(1)
    #         cax = ax.imshow(individual_neighbourhood*1, aspect='auto', extent=[0, image.shape[1], image.shape[0], 0])
    #         ax.set_title("Neighbours {}".format(str(seed)))
    #         for single_seed in seed:
    #             rect = mpl_patches.Rectangle((single_seed[1], single_seed[0]), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
    #             ax.add_patch(rect)
    #         last_individual_neighbourhood = individual_neighbourhood
    #
    # for i,j in initial_seed_points:
    #     # if l1trg_ev.pix_row == 1:
    #     rect = mpl_patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
    #     ax4.add_patch(rect)

    return out_neighbourhood, individual_neighbourhoods #grouped_neighbourhoods

    # for i in range(0,len(image.shape[0]): # row

    # it = np.nditer(a, flags=['multi_index'])
    # while not it.finished:
    #     ...
    #     print
    #     "%d <%s>" % (it[0], it.multi_index),
    #     ...
    #     it.iternext()


def select_trigger_groups(trigger_points, max_gap=3):
    # seed_points iterable of pairs
    # presuming 2d matrix

    # visited_neighbourhood = np.zeros_like(image, dtype=np.bool)

    trigger_groups = []

    # i - row
    # j - column

    point_neighbours = {}
    visited_points = {}
    for trigger_point in trigger_points:
        point_neighbours[trigger_point] = []
        visited_points[trigger_point] = False

    for trigger_point in trigger_points:
        for c_trigger_point in trigger_points:
            if c_trigger_point != trigger_point and \
                    abs(trigger_point[0] - c_trigger_point[0]) <= max_gap and \
                    abs(trigger_point[1] - c_trigger_point[1]) <= max_gap:
                point_neighbours[trigger_point].append(c_trigger_point)
            # if group is None:
            #     group = [trigger_point]
            #     trigger_groups.append(group)

    for trigger_point in trigger_points:
        if visited_points[trigger_point]:
            continue

        visited_points[trigger_point] = True

        group = [trigger_point]
        trigger_groups.append(group)

        search_stack = list(point_neighbours[trigger_point])
        while search_stack:
            neighbour_point = search_stack.pop()
            if not visited_points[neighbour_point]:
                visited_points[neighbour_point] = True
                group.append(neighbour_point)
                for neighbour_neighbour_point in point_neighbours[neighbour_point]:
                    if neighbour_neighbour_point != trigger_point and not visited_points[neighbour_neighbour_point]:
                        search_stack.append(neighbour_neighbour_point)

    return trigger_groups

    # for i in range(0,len(image.shape[0]): # row

    # it = np.nditer(a, flags=['multi_index'])
    # while not it.finished:
    #     ...
    #     print
    #     "%d <%s>" % (it[0], it.multi_index),
    #     ...
    #     it.iternext()

def process_event(frames, exp_tree, proc_params=EventProcessingParams(), pixels_mask = None, do_visualization=True):
    print(len(frames))

    event_frames = []

    triggered_pixels_coords = set()

    triggered_pixel_sum_l1_frames = []
    triggered_pixel_thr_l1_frames = []
    triggered_pixel_persist_l1_frames = []

    all_event_triggers = []
    event_triggers_by_frame = [None]*len(frames)

    for frame_num, gtu_pdm_data in enumerate(frames):
        pcd = gtu_pdm_data.photon_count_data
        if len(pcd) > 0 and len(pcd[0]) > 0:
            # TODO warning now only the very first PDM is processed
            event_frames.append(pcd[0][0])
            triggered_pixel_sum_l1 = np.zeros_like(pcd[0][0])
            triggered_pixel_thr_l1 = np.zeros_like(pcd[0][0])
            triggered_pixel_persist_l1 = np.zeros_like(pcd[0][0])

            all_event_triggers += gtu_pdm_data.l1trg_events
            event_triggers_by_frame[frame_num] = gtu_pdm_data.l1trg_events

            for l1trg_ev in gtu_pdm_data.l1trg_events:
                triggered_pixel_sum_l1[l1trg_ev.pix_row, l1trg_ev.pix_col] = l1trg_ev.sum_l1
                triggered_pixel_thr_l1[l1trg_ev.pix_row, l1trg_ev.pix_col] = l1trg_ev.thr_l1
                triggered_pixel_persist_l1[l1trg_ev.pix_row, l1trg_ev.pix_col] = l1trg_ev.persist_l1
                triggered_pixels_coords.add((l1trg_ev.pix_row, l1trg_ev.pix_col))
            triggered_pixel_sum_l1_frames.append(triggered_pixel_sum_l1)
            triggered_pixel_thr_l1_frames.append(triggered_pixel_thr_l1)
            triggered_pixel_persist_l1_frames.append(triggered_pixel_persist_l1)

    pdm_max_list = [np.max(frame) for frame in event_frames]
    max_value = np.max(pdm_max_list)
    pdm_min_list = [np.max(frame) for frame in event_frames]
    min_value = np.min(pdm_min_list)

    # for frame_num, gtu_pdm_data in enumerate(frames):
    #     pcd = gtu_pdm_data.photon_count_data
    #     if len(pcd) > 0 and len(pcd[0]) > 0:
    #     # TODO warning now only the very
    #         visualize_frame(pcd[0][0], exp_tree, gtu_pdm_data.l1trg_events, "frame: {}, GTU: {}".format(frame_num, gtu_pdm_data.gtu), True, min_value, max_value)

    if len(event_frames) == 0:
        raise Exception("No event_frames to process")

    # possibly find threshold with average background (another parameter?)

    frame_num_y = []
    frame_num_x = []

    for frame in event_frames:
        frame_num_y.append(np.max(frame, axis=1).reshape(-1,1)) # summing the x axis
        frame_num_x.append(np.max(frame, axis=0).reshape(-1,1)) # summing the y axis
        # frame_num_y.append(np.sum(frame, axis=1).reshape(-1,1)) # summing the x axis
        # frame_num_x.append(np.sum(frame, axis=0).reshape(-1,1)) # summing the y axis

    frame_num_y = np.hstack(frame_num_y)
    frame_num_x = np.hstack(frame_num_x)


    if do_visualization:
        # visualize_frame(np.add.reduce(triggered_pixel_sum_l1_frames), exp_tree, all_event_triggers, "summed sum_l1", False)
        visualize_frame_num_relation(frame_num_y, event_triggers_by_frame, "pix_row", "f(frame_num) = \sum_{frame_num} x", False)
        visualize_frame_num_relation(frame_num_x, event_triggers_by_frame, "pix_col", "f(frame_num) = \sum_{frame_num} y", False)
        # visualize_frame(np.maximum.reduce(triggered_pixel_thr_l1_frames), exp_tree, all_event_triggers, "maximum thr_l1", False)
        # visualize_frame(np.maximum.reduce(triggered_pixel_persist_l1_frames), exp_tree, all_event_triggers, "maximum persist_l1", False)

    triggers_y_t_proj = set()
    triggers_x_t_proj = set()
    for frame_num, l1trg_events in enumerate(event_triggers_by_frame):
        for l1trg_ev in l1trg_events:
            triggers_y_t_proj.add((l1trg_ev.pix_row,frame_num))
            triggers_x_t_proj.add((l1trg_ev.pix_col,frame_num))


    # consider pixels mask
    if pixels_mask is None:
        pixels_mask = np.ones_like(event_frames[0])

    weights_mask = np.ones_like(event_frames[0])
    # sum_l1
    # persist_l1
    weights_mask = np.multiply(weights_mask, pixels_mask) # applying mask, should be last

    max_values_arr = np.maximum.reduce(event_frames)
    sum_values_arr = np.add.reduce(event_frames)

    if do_visualization:
        visualize_frame(max_values_arr, exp_tree, all_event_triggers, "max_values_arr", False)
        # visualize_frame(sum_values_arr, exp_tree, all_event_triggers, "sum_values_arr", False)

    #################################
    #  Triggered pixels
    #################################

    # Groups of triggered pixels
    #   Parameters:
    #   triggered_pixels_group_max_gap  - row and column-vise distance (distance diagonally is sqrt(N^2 + N^2))


    groups_of_trigger_groups = select_trigger_groups(triggered_pixels_coords, proc_params.triggered_pixels_group_max_gap)
    for triggers_group in groups_of_trigger_groups:
        trigg_im = np.zeros_like(max_values_arr)
        for trigg_point in triggers_group:
            trigg_im[trigg_point] = max_values_arr[trigg_point]

        if do_visualization:
            fig, ax = plt.subplots(1)
            ax.imshow(trigg_im)
            ax.set_title("Trigger group: {}".format(str(triggers_group)))
            # gray_hough_line(trigg_im, 1)

    ######

    print(groups_of_trigger_groups)

    ######

    # Hough transform on integrated triggered pixels
    #   Parameters:
    #   triggered_pixels_ht_size
    #   triggered_pixels_ht_phi_num_steps
    #   triggered_pixels_ht_rho_step

    integrated_triggered_pixel_sum_l1 = np.add.reduce(triggered_pixel_sum_l1_frames)

    trigg_acc_matrix, trigg_max_distance, trigg_rho_range_opts, trigg_phi_range = \
        gray_hough_line(integrated_triggered_pixel_sum_l1, proc_params.triggered_pixels_ht_size,
                        np.linspace(0,np.pi, proc_params.triggered_pixels_ht_phi_num_steps),
                        proc_params.triggered_pixels_ht_rho_step)

    #####

    if do_visualization:
        # TODO vis
        pass

    #####

    #################################
    #  X - Y integrated event (maximal pixel value)
    #################################

    # Selecting neighbouring pixels around triggered pixels
    #   Parameters:
    #   neighbour_selection_rules_x_y

    # Tested using this neighbour selection rules:
    # [NeighbourSelectionRules(3, .3, False),
    #  NeighbourSelectionRules(3, 1, True),
    #  NeighbourSelectionRules(1, .9, True)]

    trigger_neighbours, trigg_groups = \
        select_neighbours(triggered_pixels_coords, max_values_arr, proc_params.x_y_neighbour_selection_rules)

    max_values_arr_trigg = trigger_neighbours*max_values_arr

    #TODO
    # trigger_neighbours_yt = select_neighbours(triggers_y_t_proj, frame_num_y)
    # trigger_neighbours_xt = select_neighbours(triggers_x_t_proj, frame_num_x)

    # max_trigger_neighbours_yt = trigger_neighbours_yt*frame_num_y
    # max_trigger_neighbours_xt = trigger_neighbours_xt*frame_num_x

    # gray_hough_line(max_values_arr_trigg, 3)
    # gray_hough_line(max_trigger_neighbours_yt)
    # gray_hough_line(max_trigger_neighbours_xt)
    # gray_hough_line(max_values_arr)

    #####

    # Hough transform on seleceted neighbourhood in X-Y projection
    #   Parameters:
    #   x_y_ht_size = .5
    #   x_y_ht_phi_num_steps = 90 # 2 deg per step
    #   x_y_ht_rho_step = 2

    #hint: use np.argmax(cond), np.where(cond), np,select  np.argwhere

    x_y_acc_matrix, x_y_max_distance, x_y_rho_range_opts, x_y_phi_range = \
        gray_hough_line(max_values_arr_trigg, proc_params.x_y_ht_size,
                        np.linspace(0,np.pi,proc_params.x_y_ht_phi_num_steps),
                        proc_params.x_y_ht_rho_step)

    # visualize_hough_space(x_y_acc_matrix, phi_range, rho_range_opts)

    ##

    # Position of maximum value in the hough space - x_y_acc_matrix

    acc_matrix_max_pos = np.unravel_index(x_y_acc_matrix.argmax(), x_y_acc_matrix.shape)
    acc_matrix_max = x_y_acc_matrix[acc_matrix_max_pos]

    # TODO need to save/extract: rho, phi, line_rot, coord_0_x, coord_0_y, coord_1_x, coord_1_y

    ##

    # Positions of peaks in hough space
    #   Parameters:
    #   x_y_ht_peak_threshold_frac_of_max

    filter_func = np.vectorize(lambda v1,v2: 0 if v1<v2 else v1)
    perc_max_peaks_arr = filter_func(x_y_acc_matrix, acc_matrix_max * proc_params.x_y_ht_peak_threshold_frac_of_max)
    perc_max_peaks_pos = get_field_positions(perc_max_peaks_arr, lambda v: v > 0) # np.where ?

    # Find clusters in the hough space
    #   Parameters:
    #   x_y_ht_peak_gap

    cluster_with_maximum = None

    perc_max_peaks_arr_clusters = find_pixel_clusters(perc_max_peaks_arr, proc_params.x_y_ht_peak_gap)
    for cluster_seed, cluster_im in perc_max_peaks_arr_clusters.items():
        cluster_dimensions = find_minimal_dimensions(cluster_im)

        if cluster_im[acc_matrix_max_pos] != 0:
            cluster_with_maximum = cluster_im[acc_matrix_max_pos]

        # todo convert from indexes to rho and phi
        print("Hough space cluster {} dimensions {}".format(str(cluster_seed), str(cluster_dimensions)))

    assert cluster_with_maximum is not None

    # TODO need to save: clusers - dimensions, size, counts sum;

    if do_visualization:
        fig_max_peaks, ax_max_peaks = plt.subplots(1)
        ax_max_peaks.imshow(perc_max_peaks_arr)
        ax_max_peaks.set_title("{}% of maximum peak",proc_params.x_y_ht_peak_threshold_frac_of_max*100)

    ##

    # Absolute peak - only small fluctuation from the maximal peak is allowed
    #   Parameters:
    #   x_y_ht_global_peak_threshold_frac_of_max

    perc_global_peaks_arr = filter_func(x_y_acc_matrix, acc_matrix_max * proc_params.x_y_ht_global_peak_threshold_frac_of_max)
    perc_max_peaks_pos = get_field_positions(perc_max_peaks_arr, lambda v: v > 0) # np.where ?

    # TODO need to save phi, rho, line_rot, coord_0_x, coord_0_y, coord_1_x, coord_1_y

    # TODO x_gtu  y_gtu

    # num_non_zero = np.count_nonzero(x_y_acc_matrix)
    # print("num_non_zero", num_non_zero)
    # n_max_peaks_pos = find_n_max_values(x_y_acc_matrix, int(np.ceil(num_non_zero*.1)))

    # rho_range = np.arange(rho_range_opts[0],rho_range_opts[1],rho_range_opts[2])
    # peaks_hspace, peaks_angles, peaks_dists = skimage.transform.hough_line_peaks(x_y_acc_matrix, phi_range, rho_range) #, 2, 2
    #
    # print("peaks_angles", peaks_angles)
    # print("peaks_dists", peaks_dists)
    # fig, ax = plt.subplots(1)
    # ax.imshow(peaks_hspace)
    # ax.set_title("skimage.transform.hough_line_peaks hspace")
    # plt.show()

    # skimage_max_lines = np.hstack( (np.array(peaks_dists).reshape(-1,1) , np.array(peaks_angles).reshape(-1,1) , np.array(peaks_hspace).reshape(-1,1)  ))

    # print("skimage max lines")
    # print(skimage_max_lines)
    #
    # fig_hist2d, ax_hist2d = plt.subplots(1)
    # ax_hist2d.hist2d(skimage_max_lines[:,1], skimage_max_lines[:,0], bins=40)
    # ax_hist2d.set_ylabel("rho")
    # ax_hist2d.set_xlabel("phi")

    # value_points_groups = split_values_to_groups(max_lines_pos, x_y_acc_matrix)
    # value_lines_groups = {}
    # for k,l in value_points_groups.items():
    #     value_lines_groups[k] = [hough_space_max2val_single(v, phi_range, rho_range_opts) for v in l]

    # visualize_hough_lines(max_values_arr_trigg, skimage_max_lines[:,:2], "Lines selected from max_values_arr")


    # n_max_lines_pos_ndarray = np.array(n_max_peaks_pos)

    # fig_hist2d, ax_hist2d = plt.subplots(1)
    # ax_hist2d.hist2d(n_max_lines_pos_ndarray[:,1], n_max_lines_pos_ndarray[:,0], bins=40)
    # ax_hist2d.set_ylabel("rho")
    # ax_hist2d.set_xlabel("phi")
    # ax_hist2d.set_title("Max 10% of points")

    #
    # print("max_lines_pos", max_lines_pos)

    perc_max_lines = hough_space_index_to_val(perc_max_peaks_pos, x_y_phi_range, x_y_rho_range_opts)
    # max_lines = hough_space_max2val(max_lines_pos, phi_range, rho_range_opts)
    # print("max_lines", max_lines)
    #
    #
    # value_points_groups = split_values_to_groups(max_lines_pos, x_y_acc_matrix)

    value_points_groups = split_all_filed_values_to_groups(perc_max_peaks_arr)
    value_lines_groups = {}
    for k,l in value_points_groups.items():
        value_lines_groups[k] = [hough_space_index_to_val_single(v, x_y_phi_range, x_y_rho_range_opts) for v in l]

    visualize_hough_lines(max_values_arr_trigg, perc_max_lines, "Lines selected from max_values_arr", value_lines_groups)


    # rho_acc_matrix visulaization would go here





    print(len(frames))

    # hough_line()
    # hough_line_peaks()

    plt.show()

    return None