import numpy as np
import itertools
import utility_funtions as utl
# import collections
import event_visualization as vis

class NeighbourSelectionRules:
    max_gap = 3
    val_ratio_thr = 1.0
    grow = True

    def __init__(self, max_gap=3, val_ratio_thr=1, grow=True):
        self.max_gap = int(max_gap)
        self.val_ratio_thr = float(val_ratio_thr)
        self.grow = bool(grow)

    def __str__(self):
        return "{:d},{:.2f},{}".format(int(self.max_gap), float(self.val_ratio_thr), bool(self.grow))

    @classmethod
    def from_str(cls,s):
        p = s.split(",")
        if len(p) != 3:
            raise Exception('Unexpected string format')
        return NeighbourSelectionRules(int(p[0].strip()), float(p[1].strip()), utl.str2bool(p[2].strip()))


def parse_x_y_neighbour_selection_rules_str(conf_attr_str):
    for sep in (", ",";"):
        if sep in conf_attr_str:
            return [NeighbourSelectionRules.from_str(s.strip()) for s in conf_attr_str.split(';')]
    return None


def gray_hough_line(input_image, line_thicknes=2, phi_range=np.linspace(0, np.pi, 180)[:-1], rho_step=1, fix_negative_r=False, flip_image=False):
    image = None

    if flip_image:
        image = np.flipud(input_image)
    else:
        image = input_image


    max_distance = np.hypot(image.shape[0], image.shape[1])
    num_rho = int(np.ceil(max_distance*2/rho_step))
    rho_correction_lower = -line_thicknes + max_distance
    rho_correction_upper = line_thicknes + max_distance
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

                # rho_index_lower = int((rho+rho_correction_lower) // rho_step)
                # rho_index_upper = int((rho+rho_correction_upper) // rho_step)
                rho_index_lower = int(np.round((rho+rho_correction_lower) / rho_step))
                rho_index_upper = int(np.round((rho+rho_correction_upper) / rho_step))

                if rho_index_lower < 0:
                    # print("rho_index_lower < 0 : rho_index_lower=", rho_index_lower)
                    rho_index_lower = 0

                if rho_index_upper > num_rho:
                    # print("rho_index_upper > num_rho : rho_index_upper=", rho_index_upper,"num_rho=",num_rho)
                    rho_index_upper = num_rho

                for rho_index in range(rho_index_lower,rho_index_upper):
                    acc_matrix[rho_index, phi_index] += image[i,j] * phi_corr

    if not fix_negative_r:
        return acc_matrix, max_distance, (-max_distance, max_distance, rho_step), phi_range
    else:
        if len(phi_range) < 2:
            raise Exception('Insufficient number of phi steps')

        zero_rho_index = int((0 + max_distance) // rho_step)
        zero_phi_index = len(phi_range) // 2

        acc_matrix_positive_r = acc_matrix[zero_rho_index+1:,:]
        # acc_matrix_negative_r_right = np.flipud(acc_matrix[zero_rho_index+1:,zero_phi_index:])
        # acc_matrix_negative_r_left = np.flipud(acc_matrix[zero_rho_index+1:,:zero_phi_index])
        acc_matrix_negative_r_flipped = np.flipud(acc_matrix[0:zero_rho_index])
        # acc_matrix_negative_r_right = np.flipud(acc_matrix[zero_rho_index+1:,:])
        # acc_matrix_negative_r_left = np.flipud(acc_matrix[zero_rho_index+1:,:])

        # z[0:3,1:4] = r[:3,:3]

        # print(acc_matrix.shape, acc_matrix_positive_r.shape, acc_matrix_negative_r_flipped.shape, acc_matrix_negative_r_flipped.shape)

        # vis.visualize_hough_space(acc_matrix_positive_r, phi_range, (0, max_distance, rho_step), r"Hough space with positive $\rho$ part BEFORE CORRECTION")

        max_height = max([acc_matrix_positive_r.shape[0], acc_matrix_negative_r_flipped.shape[0], acc_matrix_negative_r_flipped.shape[0]])
        if acc_matrix_positive_r.shape[0] < max_height:
            new_acc_matrix_positive_r = np.zeros((max_height, acc_matrix_positive_r.shape[1]))
            new_acc_matrix_positive_r[0:acc_matrix_positive_r.shape[0],:] = acc_matrix_positive_r
            # old = acc_matrix_positive_r
            acc_matrix_positive_r = new_acc_matrix_positive_r
            # import matplotlib.pyplot as plt
            # fig,ax = plt.subplots(1)
            # ax.imshow(old)
            # fig,ax = plt.subplots(1)
            # ax.imshow(acc_matrix_positive_r)
        elif acc_matrix_negative_r_flipped.shape[1] < max_height:
            new_acc_matrix_negative_r = np.zeros((max_height, acc_matrix_negative_r_flipped.shape[1]))
            new_acc_matrix_negative_r[0:acc_matrix_negative_r_flipped.shape[0], :] = acc_matrix_negative_r_flipped
            acc_matrix_negative_r_flipped = new_acc_matrix_negative_r

        vis.visualize_hough_space(acc_matrix, phi_range, (-max_distance, max_distance, rho_step), r"Hough space with negative and positive $\rho$")

        # vis.visualize_hough_space(np.flipud(acc_matrix_negative_r_flipped),phi_range, (-max_distance, 0, rho_step), r"Hough space with negative $\rho$ part")

        # vis.visualize_hough_space(acc_matrix_positive_r, phi_range, (0, max_distance, rho_step), r"Hough space with positive $\rho$ part")

        print(acc_matrix_positive_r[:,-1:].shape, acc_matrix_positive_r[:,:1].shape)

        add_to_right = True in (acc_matrix_positive_r[:,-1:]>0)
        add_to_left = True in (acc_matrix_positive_r[:,:1]>0)     # should be mutually exclusive to previous
        l = []
        new_phi_range = phi_range
        if add_to_left:
            # l.append(acc_matrix_negative_r_flipped[:,:-1])
            # new_phi_range = np.hstack(((phi_range[:-1]-phi_range[-1]),phi_range))
            l.append(acc_matrix_negative_r_flipped)
            new_phi_range = np.hstack((phi_range-(2*phi_range[-1]-phi_range[-2]),phi_range))
        l.append(acc_matrix_positive_r)
        if add_to_right:
            # l.append(acc_matrix_negative_r_flipped[:,1:]) # phi range should not contain pi and this might be unnecessary
            # new_phi_range = np.hstack((phi_range,(phi_range[1:]+phi_range[-1])))
            l.append(acc_matrix_negative_r_flipped)
            new_phi_range = np.hstack((phi_range,phi_range+(2*phi_range[-1]-phi_range[-2]))) # considering pi_range[0] == 0

        acc_matrix_fixed = np.hstack(l)

        #print(acc_matrix_fixed.shape[1] == len(new_phi_range))
        assert acc_matrix_fixed.shape[1] == len(new_phi_range)

        return acc_matrix_fixed, max_distance, (0, max_distance, rho_step), new_phi_range


def hough_space_rho_index_to_val(index, rho_range_opts):
    return rho_range_opts[0] + rho_range_opts[2] * index + rho_range_opts[2] // 2 # TODO justification for  `+ rho_range_opts[2]`


def hough_space_index_to_val_single(index, phi_range, rho_range_opts):
    return (hough_space_rho_index_to_val(index[0], rho_range_opts), phi_range[index[1]])


def hough_space_index_to_val(indexes, phi_range, rho_range_opts):
    o = []
    for index in indexes:
        o.append(hough_space_index_to_val_single(index, phi_range, rho_range_opts))
    return o


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

def find_minimal_rectangle(cluster_im):
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
    return (first_row, last_row), (first_col, last_col)

def find_minimal_dimensions(cluster_im):
    p1, p2 = find_minimal_rectangle(cluster_im)
    first_row, last_row = p1
    first_col, last_col = p2
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
        for c_trigger_point in trigger_points: # very inefficient
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
