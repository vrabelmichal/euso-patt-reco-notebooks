import os
import sys
import pickle
import hashlib
import pandas as pd
import numpy as np

import tool.acqconv

def count_num_max_pix_on_pmt_and_ec(df, fractions=[0.6, 0.8, 0.9], save_npy_dir=None, npy_file_key=None,
                                    debug_messages=False, hashstr=None):
    events_num_max_pix_on_pmt = {}
    events_num_max_pix_on_ec = {}

    pickled_events_events_num_max_pix_on_pmt = {}
    pickled_events_num_max_pix_on_ec = {}

    if hashstr is None:
        hashstr = hashlib.md5(pickle.dumps(df.values, protocol=0)).hexdigest()

    def get_npy_pathname(basename, frac, i, j, hashstr=hashstr):
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
                try:
                    dest[frac][(i, j)] = np.load(pkl_pathname)
                    return True
                except:
                    if debug_messages:
                        print("#### {} DOES NOT EXIST".format(pkl_pathname))
                    pass
            elif debug_messages:
                print("#### {} DOES NOT EXIST".format(pkl_pathname))
        return False

    if save_npy_dir:
        os.makedirs(save_npy_dir, exist_ok=True)

    load_files = False

    for frac in fractions:

        for i in range(6):
            for j in range(6):
                load_successful = try_load_npy_file(pickled_events_events_num_max_pix_on_pmt, 'num_max_pix_on_pmt',
                                                    frac, i, j)
        #                 if not load_successful:
        #                     events_num_max_pix_on_pmt[frac][(i,j)] = np.zeros((len(df), 2))

        for i in range(3):
            for j in range(3):
                load_successful = try_load_npy_file(pickled_events_num_max_pix_on_ec, 'num_max_pix_on_ec', frac, i, j)
        #                 if not load_successful:
        #                     events_num_max_pix_on_ec[frac][(i,j)] = np.zeros((len(df), 2))

        if frac not in pickled_events_num_max_pix_on_ec:
            load_files = True
            print(">>> pickled_events_num_max_pix_on_ec[{:.1f}] IS NOT LOADED - READING ALL EVENTS".format(frac))
            break
        elif frac not in pickled_events_events_num_max_pix_on_pmt:
            load_files = True
            print(
                ">>> pickled_events_events_num_max_pix_on_pmt[{:.1f}] IS NOT LOADED - READING ALL EVENTS".format(frac))
            break
        if not load_files:
            for i in range(6):
                for j in range(6):
                    if (i, j) not in pickled_events_events_num_max_pix_on_pmt[frac]:
                        print(
                            ">>> pickled_events_events_num_max_pix_on_pmt[{:.1f}][({:d},{:d})] IS NOT LOADED - READING ALL EVENTS".format(
                                frac, i, j))
                        load_files = True
                        break
        if not load_files:
            for i in range(3):
                for j in range(3):
                    if (i, j) not in pickled_events_num_max_pix_on_ec[frac]:
                        print(
                            ">>> pickled_events_num_max_pix_on_ec[{:.1f}][({:d},{:d})] IS NOT LOADED - READING ALL EVENTS".format(
                                frac, i, j))
                        load_files = True
                        break

    if load_files:
        for frac in fractions:
            events_num_max_pix_on_pmt[frac] = {}
            events_num_max_pix_on_ec[frac] = {}

            for i in range(6):
                for j in range(6):
                    events_num_max_pix_on_pmt[frac][(i, j)] = np.zeros((len(df), 2))

            for i in range(3):
                for j in range(3):
                    events_num_max_pix_on_ec[frac][(i, j)] = np.zeros((len(df), 2))

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
                                     row['packet_id'] * 128 + row['gtu_in_packet'] - 4:row['packet_id'] * 128 + row[
                                         'gtu_in_packet'] - 4 + row['num_gtu']]
            elif row['source_file_acquisition_full'].endswith('.root'):
                frames_acquisition = tool.acqconv.get_frames(row['source_file_acquisition_full'],
                                                             row['packet_id'] * 128 + row['gtu_in_packet'] - 4,
                                                             row['packet_id'] * 128 + row['gtu_in_packet'] - 4 + row[
                                                                 'num_gtu'], entry_is_gtu_optimization=True)
            else:
                raise RuntimeError(
                    'Unexpected source_file_acquisition_full "{}"'.format(row['source_file_acquisition_full']))

            ev_integrated = np.maximum.reduce(frames_acquisition)
            ev_integrated_max = np.max(ev_integrated)

            max_positions = {}

            for frac in fractions:
                max_positions = np.transpose(np.where(ev_integrated > ev_integrated_max * frac))

                for pos_y, pos_x in max_positions:
                    pmt_y = pos_y // 8
                    pmt_x = pos_x // 8

                    ec_y = pos_y // 16
                    ec_x = pos_x // 16

                    pmt_key = (pmt_y, pmt_x)
                    ec_key = (ec_y, ec_x)

                    events_num_max_pix_on_pmt[frac][pmt_key][row_i, 0] += 1
                    events_num_max_pix_on_ec[frac][ec_key][row_i, 0] += 1

        for k in range(len(df)):
            for frac in fractions:
                for i in range(3):
                    for j in range(3):
                        for ii in range(3):
                            for jj in range(3):
                                if jj == j and ii == i:
                                    continue
                                events_num_max_pix_on_ec[frac][(i, j)][k, 1] += \
                                events_num_max_pix_on_ec[frac][(ii, jj)][k, 0]

        if save_npy_dir and npy_file_key:
            for frac in fractions:
                for i in range(3):
                    for j in range(3):
                        np.save(get_npy_pathname('num_max_pix_on_ec', frac, i, j),
                                events_num_max_pix_on_ec[frac][(i, j)])

        for k in range(len(df)):
            for frac in fractions:
                for i in range(6):
                    for j in range(6):
                        for ii in range(6):
                            for jj in range(6):
                                if jj == j and ii == i:
                                    continue
                                events_num_max_pix_on_pmt[frac][(i, j)][k, 1] += \
                                events_num_max_pix_on_pmt[frac][(ii, jj)][k, 0]

        if save_npy_dir and npy_file_key:
            for frac in fractions:
                for i in range(6):
                    for j in range(6):
                        np.save(get_npy_pathname('num_max_gitpix_on_pmt', frac, i, j),
                                events_num_max_pix_on_pmt[frac][(i, j)])

        return events_num_max_pix_on_pmt, events_num_max_pix_on_ec
    else:
        return pickled_events_events_num_max_pix_on_pmt, pickled_events_num_max_pix_on_ec


def extend_df_with_num_max_pix(events_within_cond, events_num_max_pix_on_pmt, events_num_max_pix_on_ec):
    events_within_cond_cp = events_within_cond.copy()  # this is NOT creating the copy
    k = 0

    for frac, frac_dict in events_num_max_pix_on_pmt.items():
        for pmt_pos, pmt_counts in frac_dict.items():
            i, j = pmt_pos
            col = 'pmt_{:d}_{:d}'.format(i, j) + '_frac{:.1f}'.format(frac).replace('.', '')

            events_within_cond_cp[col + '_in'] = pd.Series(pmt_counts[:, 0])
            events_within_cond_cp[col + '_out'] = pd.Series(pmt_counts[:, 1])

    # for frac in fractions:
    #     for i in range(3):
    #         for j in range(3):
    for frac, frac_dict in events_num_max_pix_on_ec.items():
        for ec_pos, ec_counts in frac_dict.items():
            i, j = ec_pos
            col = 'ec_{:d}_{:d}'.format(i, j) + '_frac{:.1f}'.format(frac).replace('.', '')
            events_within_cond_cp[col + '_in'] = pd.Series(ec_counts[:, 0])
            events_within_cond_cp[col + '_out'] = pd.Series(ec_counts[:, 1])

    print(len(events_within_cond_cp.columns))
    return events_within_cond_cp


def filter_out_by_fraction(events_within_cond_cp, ec_0_0_frac_lt=0.5, ec_in_column='ec_0_0_frac06_in',
                           ec_out_column='ec_0_0_frac06_out'):
    ec_0_0_frac = events_within_cond_cp[ec_in_column] / events_within_cond_cp[ec_out_column]
    # filtered_events_within_cond = events_within_cond_cp[ (events_within_cond_cp['ec_0_0_frac06_out'] == 0) ]
    filtered_events_within_cond = events_within_cond_cp[
        (events_within_cond_cp[ec_out_column] != 0) & (ec_0_0_frac < ec_0_0_frac_lt)]
    return filtered_events_within_cond

def angle_difference(a1, a2):
    r = np.abs(a1 - a2) % (2*np.pi)
    if r >= np.pi:
        r = 2*np.pi - r
    return r

def smaller_angle_difference(a1, a2):
    d = angle_difference(a1,a2)
    if d > np.pi/2:
        d = np.pi - d
    return normalize_phi(d)

def normalize_phi(phi):
    norm_phi = np.arctan2(np.sin(phi), np.cos(phi))
    if norm_phi < 0:
        norm_phi = (2*np.pi + norm_phi)
    # norm_phi = (2*np.pi + norm_phi) * (norm_phi < 0) + norm_phi * (norm_phi > 0)
    return norm_phi

# def angle_difference_(a1, a2):
#     r = np.abs(a1 - a2) % (2*np.pi)
#     if r >= np.pi:
#         r = 2*np.pi - r
#     return r