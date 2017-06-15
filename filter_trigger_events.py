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

import time

from event_processing import *
from event_visualization import *
from event_reading import *
from base_classes import *
import processing_config
from tsv_event_storage import *
from sqllite_event_storage import *


def main(argv):
    parser = argparse.ArgumentParser(description='Find patterns inside triggered pixes')
    # parser.add_argument('files', nargs='+', help='List of files to convert')
    parser.add_argument('-a', '--acquisition-file', help="ACQUISITION root file in \"Lech\" format")
    parser.add_argument('-k', '--kenji-l1trigger-file', help="L1 trigger root file in \"Kenji\" format")
    parser.add_argument('-o', '--outfile', default=None, help="Output specification")
    parser.add_argument('-f', '--output-type', default="tsv", help="Output type - tsv, stdout, sqlite")
    parser.add_argument('-c', '--corr-map-file', default=None, help="Corrections map .npy file")
    parser.add_argument('--gtu-before-trigger', type=int, default=4, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--gtu-after-trigger', type=int, default=4, help="Number of GTU included in track finding data before the trigger")
    # parser.add_argument('--trigger-persistency', type=int, default=2, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--packet-size', type=int, default=128, help="Number of GTU in packet")

    parser.add_argument("--visualize", type=str2bool_for_argparse, default=False, help="If this option is true, matplotlib plots are shown.")

    parser.add_argument('--first-gtu', type=int, default=0, help="GTU before will be skipped")
    parser.add_argument('--last-gtu', type=int, default=sys.maxsize, help="GTU after will be skipped")

    parser.add_argument('--filter-n-persist-gt', type=int, default=-1, help="Accept only events with at least one GTU with nPersist more than this value.")
    parser.add_argument('--filter-sum-l1-pdm-gt', type=int, default=-1, help="Accept only events with at least one GTU with sumL1PDM more than this value.")
    parser.add_argument('--filter-sum-l1-ec-one-gt', type=int, default=-1, help="Accept only events with at least one GTU with at leas one EC sumL1PDM more than this value.")
    parser.add_argument('--filter-sum-l1-pmt-one-gt', type=int, default=-1, help="Accept only events with at least one GTU with at leas one PMT sumL1PMT more than this value.")

    args = parser.parse_args(argv)

    ack_l1_reader = AckL1EventReader(args.acquisition_file, args.kenji_l1trigger_file)

    filter_options = EventFilterOptions()
    filter_options.n_persist = args.filter_n_persist_gt
    filter_options.sum_l1_pdm = args.filter_sum_l1_pdm_gt
    filter_options.sum_l1_ec_one = args.filter_sum_l1_ec_one_gt
    filter_options.sum_l1_pmt_one = args.filter_sum_l1_pmt_one_gt

    if args.output_type == "tsv" or args.output_type == "csv":
        output_storage_provider = TsvEventStorageProvider()
    elif args.output_type == "sqlite":
        output_storage_provider = SqlLite3EventStorageProvider()
    else:
        output_storage_provider = EventStorageProvider()

    output_storage_provider.initialize(args.outfile)

    read_and_process_events(ack_l1_reader,
                            args.first_gtu, args.last_gtu, args.gtu_before_trigger, args.gtu_after_trigger,
                            args.packet_size, filter_options, output_storage_provider, args)

    output_storage_provider.finalize()


def read_and_process_events(ack_l1_reader, first_gtu, last_gtu, gtu_before_trigger, gtu_after_trigger, packet_size, filter_options, output_storage_provider, context):

    before_trg_frames_circ_buffer = collections.deque(maxlen=gtu_before_trigger+1) # +1 -> the last gtu in the buffer is triggered
    frame_buffer = []
    gtu_after_trigger -= 1 # -1 -> when counter is 0 the frame is still saved

    process_event_down_counter = np.inf
    packet_id = -1

    #TODO ???? why ????
    event_start_gtu = -1

    save_config_info_result = output_storage_provider.save_config_info(processing_config.proc_params)

    for gtu_pdm_data in ack_l1_reader.iter_gtu_pdm_data():

        gtu_in_packet = gtu_pdm_data.gtu % packet_size
        if gtu_in_packet == 0:
            packet_id += 1 # starts at -1
            before_trg_frames_circ_buffer.clear()
            frame_buffer.clear()
            process_event_down_counter = np.inf
            event_start_gtu = -1

        if np.isinf(process_event_down_counter):
            before_trg_frames_circ_buffer.append(gtu_pdm_data)
        else:
            frame_buffer.append(gtu_pdm_data)

        print_frame_info(gtu_pdm_data)

        if len(gtu_pdm_data.l1trg_events) > 0:

            if np.isinf(process_event_down_counter):
                frame_buffer.extend(before_trg_frames_circ_buffer)
                before_trg_frames_circ_buffer.clear()
                event_start_gtu = gtu_pdm_data.gtu

            process_event_down_counter = gtu_after_trigger

            for l1trg_ev in gtu_pdm_data.l1trg_events:
                if l1trg_ev.packet_id != packet_id:
                    raise Exception("Unexpected L1 trigger event's packet id (actual: {}, expected: {})".format(l1trg_ev.packet_id, packet_id))
                if l1trg_ev.gtu_in_packet != gtu_in_packet:
                    raise Exception("Unexpected L1 trigger event's gtu in packet (actual: {}, expected: {})".format(l1trg_ev.gtu_in_packet, gtu_in_packet))

            # pcd = gtu_pdm_data.photon_count_data
            # if len(pcd) > 0 and len(pcd[0]) > 0:
            #     visualize_frame(gtu_pdm_data, ack_l1_reader.exp_tree)

        if not np.isinf(process_event_down_counter):
            if process_event_down_counter == 0 or gtu_in_packet == 127:
                if gtu_pdm_data.gtu >= first_gtu and gtu_pdm_data.gtu <= last_gtu and filter_options.check_pdm_gtu(gtu_pdm_data):

                    ########################################################################################

                    ev = TriggerEventAnalysisRecord()
                    ev.source_file_acquisition = context.acquisition_file
                    ev.source_file_trigger = context.kenji_l1trigger_file
                    ev.exp_tree = ack_l1_reader.exp_tree
                    ev.global_gtu = event_start_gtu
                    ev.packet_id = packet_id
                    ev.gtu_in_packet = event_start_gtu % packet_size
                    ev.gtu_data = frame_buffer

                    process_event(trigger_event_record=ev,
                                  proc_params=processing_config.proc_params, do_visualization=context.visualize)

                    ########################################################################################

                    # print(str(ev), file=outfile)
                    # outfile.flush()

                    output_storage_provider.save_row(ev, save_config_info_result)

                process_event_down_counter = np.inf
                before_trg_frames_circ_buffer.extend(frame_buffer)
                frame_buffer.clear()
                event_start_gtu = -1

            elif process_event_down_counter > 0:
                if len(gtu_pdm_data.l1trg_events) == 0: # TODO this might require increase size of the circular buffer
                    process_event_down_counter -= 1
            else:
                raise Exception("Unexpected value of process_event_down_counter")


def str2bool_for_argparse(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
