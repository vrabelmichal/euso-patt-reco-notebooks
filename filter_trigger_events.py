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

import event_processing
from trigger_event_analysis_record import *
from event_visualization import *
from event_reading import *
from base_classes import *
import processing_config
from tsv_event_storage import *
from sqlite_event_storage import *
from postgresql_event_storage import *
import web_config


def main(argv):
    parser = argparse.ArgumentParser(description='Find patterns inside triggered pixes')
    # parser.add_argument('files', nargs='+', help='List of files to convert')
    parser.add_argument('-a', '--acquisition-file', default=None, help="ACQUISITION root file in \"Lech\" format")
    parser.add_argument('-k', '--kenji-l1trigger-file', default=None, help="L1 trigger root file in \"Kenji\" format")
    parser.add_argument('-o', '--out', default=None, help="Output specification")
    parser.add_argument('-f', '--output-type', default="stdout", help="Output type - tsv, stdout, sqlite, prostgresql")
    parser.add_argument('-c', '--corr-map-file', default=None, help="Corrections map .npy file")
    parser.add_argument('--gtu-before-trigger', type=int, default=None, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--gtu-after-trigger', type=int, default=None, help="Number of GTU included in track finding data before the trigger")
    # parser.add_argument('--trigger-persistency', type=int, default=2, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--packet-size', type=int, default=128, help="Number of GTU in packet")

    parser.add_argument('--run-again', default=None, help="Specification defining events to be run again")
    parser.add_argument('--run-again-input-type', default="postgresql", help="Input type - postgresql")
    parser.add_argument('--run-again-limit', type=int, default=10000, help="Maximal number of events to be rerun based on --run-again specification")

    parser.add_argument("--save-figures-base-dir", default=None, help="If this option is set, matplotlib figures are saved to this directory in format defined by --figure-pathname-format option.")
    parser.add_argument('--figure-name-format',
                        default="{program_version:.2f}"
                                "/{triggered_pixels_group_max_gap:d}_{triggered_pixels_ht_line_thickness:.2f}_{triggered_pixels_ht_phi_num_steps:d}_{x_y_neighbour_selection_rules}"
                                "_{x_y_ht_line_thickness:.2f}_{x_y_ht_phi_num_steps:d}_{x_y_ht_rho_step:.2f}_{x_y_ht_peak_threshold_frac_of_max:.2f}_{x_y_ht_peak_gap:d}"
                                "_{x_y_ht_global_peak_threshold_frac_of_max:.2f}"
                                "/{acquisition_file_basename}/{kenji_l1trigger_file_basename}"
                                "/{gtu_global:d}_{packet_id:d}_{gtu_in_packet:d}/{name}.png",
                        help="Format of a saved matplotib figure pathname relative to base directory.")
    parser.add_argument("--generate-web-figures", type=str2bool_argparse, default=False, help="If this option is true, matplotlib figures are generated in web directory.")
    parser.add_argument("--visualize", type=str2bool_argparse, default=False, help="If this option is true, matplotlib figures are shown.")
    parser.add_argument("--no-overwrite--weak-check", type=str2bool_argparse, default=True, help="If this option is true, the existnece of records with same files and processing version is chceked BEFORE event is processed.")
    parser.add_argument("--no-overwrite--strong-check", type=str2bool_argparse, default=False, help="If this option is true, the existnece of records with same parameters is chceked AFTER event is processed")
    parser.add_argument("--update", type=str2bool_argparse, default=False, help="If this option is true, the existnece of records with same files and processing version is chceked AFTER event is processed."
                                                                                    "If event is found event other parameters are updated.")
    parser.add_argument('--first-gtu', type=int, default=0, help="GTU before will be skipped")
    parser.add_argument('--last-gtu', type=int, default=sys.maxsize, help="GTU after will be skipped")

    parser.add_argument('--filter-n-persist-gt', type=int, default=-1, help="Accept only events with at least one GTU with nPersist more than this value.")
    parser.add_argument('--filter-sum-l1-pdm-gt', type=int, default=-1, help="Accept only events with at least one GTU with sumL1PDM more than this value.")
    parser.add_argument('--filter-sum-l1-ec-one-gt', type=int, default=-1, help="Accept only events with at least one GTU with at leas one EC sumL1PDM more than this value.")
    parser.add_argument('--filter-sum-l1-pmt-one-gt', type=int, default=-1, help="Accept only events with at least one GTU with at leas one PMT sumL1PMT more than this value.")

    args = parser.parse_args(argv)


    filter_options = EventFilterOptions()
    filter_options.n_persist = args.filter_n_persist_gt
    filter_options.sum_l1_pdm = args.filter_sum_l1_pdm_gt
    filter_options.sum_l1_ec_one = args.filter_sum_l1_ec_one_gt
    filter_options.sum_l1_pmt_one = args.filter_sum_l1_pmt_one_gt

    if args.output_type == "tsv" or args.output_type == "csv":
        output_storage_provider = TsvEventStorageProvider(args.out)
    elif args.output_type == "sqlite":
        output_storage_provider = Sqlite3EventStorageProvider(args.out)
    elif args.output_type == "postgresql":
        output_storage_provider = PostgreSqlEventStorageProvider(args.out)
    else:
        output_storage_provider = EventStorageProvider()

    if args.generate_web_figures:
        args.save_figures_base_dir = web_config.base_image_storage_directory
        args.figure_name_format = web_config.figure_name_format

    processing_runs = []

    acq_trg_params_added_to_processing_runs = False

    if args.run_again:
        if args.run_again_input_type != 'postgresql':
            raise Exception('Unsupported run again input type')
        if not args.run_again:
            raise Exception('run_again is expected to contain conditions')

        run_again_storage_provider = output_storage_provider
        if args.output_type != args.run_again_input_type or args.run_again != args.out:
            run_again_storage_provider = PostgreSqlEventStorageProvider(args.run_again)

        processed_files_configs__gtus = collections.OrderedDict()
        # event_ids__processing_configs = collections.OrderedDict()
        processing_config_ids = set()

        run_again_events_query = 'SELECT {data_table_pk}, {config_info_table_pk}, {data_table_source_file_acquisition_column}, ' \
                                 '{data_table_source_file_trigger_column}, {data_table_global_gtu_column}' \
                                 'FROM {data_table} WHERE '+args.run_again+' OFFSET 0 LIMIT {:d}'.format(args.run_again_limit) # could be unsafe
        trigger_analysis_records, all_cols = run_again_storage_provider.fetch_trigger_analysis_records(run_again_events_query)

        for id, config_info_id, trigger_analysis_record in trigger_analysis_records:
            key_tuple = (trigger_analysis_record.source_file_acquisition, trigger_analysis_record.source_file_trigger, config_info_id)
            if key_tuple not in processed_files_configs__gtus:
                processed_files_configs__gtus[key_tuple] = []
            processed_files_configs__gtus[key_tuple].append(
                # getattr(trigger_analysis_record, run_again_storage_provider.data_table_pk),
                getattr(trigger_analysis_record, run_again_storage_provider.data_table_global_gtu_column)
            )
            #event_ids__processing_configs[trigger_analysis_record.event_id] = config_info_id # todo translate_struct ?
            processing_config_ids.add(config_info_id)

        processing_configs = run_again_storage_provider.fetch_config_info_records(
            "SELECT * FROM {config_info_table} WHERE {config_info_table_pk} IN ({})".format(
                ", ".join([str(cid) for cid in list(processing_config_ids)] )
            )
        )

        #translate_struct(event_ids__processing_configs, lambda config_info_id: processing_configs[config_info_id])

        if len(trigger_analysis_records) == int(args.run_again_limit):
            m = "WARNING only first {:d} events will be rerun".format(args.run_again_limit)
            print(m,file=sys.stderr)

        for k, gtus in processed_files_configs__gtus.items():
            source_file_acquisition, source_file_trigger, config_info_id = k

            if config_info_id not in processing_configs:
                raise Exception('Unexpected state config info not fetched from processing config_infor table')
            if source_file_acquisition == args.acquisition_file and args.kenji_l1trigger_file == source_file_trigger \
                and  processing_configs[config_info_id] == processing_config.proc_params:
                processing_runs.append((source_file_acquisition, source_file_trigger, processing_configs.proc_params, sorted(gtus), False)) # todo event_ids is wrong
                acq_trg_params_added_to_processing_runs = True
            else:
                processing_runs.append((source_file_acquisition, source_file_trigger, processing_configs[config_info_id], sorted(gtus), True)) # discarding config_info_id creates overhead, but gtu_before_triggerer might be changed

    if not acq_trg_params_added_to_processing_runs and args.acquisition_file and args.kenji_l1trigger_file:
        processing_runs((args.acquisition_file, args.kenji_l1trigger_file, processing_configs.proc_params, None, False))

    for source_file_acquisition, source_file_trigger, proc_params, gtus, process_only_selected in processing_runs:

        ack_l1_reader = AckL1EventReader(source_file_acquisition, source_file_trigger)

        if args.gtu_before_trigger is not None:
            proc_params.gtu_before_trigger = args.gtu_before_trigger
        if args.gtu_after_trigger is not None:
            proc_params.gtu_after_trigger = args.gtu_after_trigger

        read_and_process_events(ack_l1_reader,
                                args.first_gtu, args.last_gtu, args.packet_size, filter_options,
                                output_storage_provider, args,
                                args.no_overwrite__weak_check, args.no_overwrite__strong_check, args.update,
                                args.save_figures_base_dir, args.figure_name_format, gtus, process_only_selected,
                                proc_params)

    output_storage_provider.finalize()


def read_and_process_events(ack_l1_reader, first_gtu, last_gtu, packet_size,
                            filter_options, output_storage_provider,
                            context,
                            no_overwrite__weak_check=True, no_overwrite__strong_check=False, update=True,
                            figure_img_base_dir=None, figure_img_name_format=None,
                            run_again_gtus=None, run_again_exclusively=False,
                            proc_params=None,
                            log_file=sys.stdout):

    if run_again_gtus is not None and run_again_exclusively:
        raise Exception("run_again_gtus is None but run_again_exclusively is True")

    if proc_params is None:
        proc_params = processing_config.proc_params

    gtu_before_trigger = proc_params.gtu_before_trigger
    gtu_after_trigger = proc_params.gtu_after_trigger

    before_trg_frames_circ_buffer = collections.deque(maxlen=gtu_before_trigger+1) # +1 -> the last gtu in the buffer is triggered
    frame_buffer = []
    gtu_after_trigger -= 1 # -1 -> when counter is 0 the frame is still saved

    process_event_down_counter = np.inf
    packet_id = -1

    event_start_gtu = -1

    save_config_info_result = output_storage_provider.save_config_info(proc_params)

    for gtu_pdm_data in ack_l1_reader.iter_gtu_pdm_data():

        last_gtu_in_packet = gtu_pdm_data.gtu % packet_size
        if last_gtu_in_packet == 0:
            packet_id += 1 # starts at -1
            before_trg_frames_circ_buffer.clear()
            frame_buffer.clear()
            process_event_down_counter = np.inf
            event_start_gtu = -1

        if np.isinf(process_event_down_counter):
            before_trg_frames_circ_buffer.append(gtu_pdm_data)
        else:
            frame_buffer.append(gtu_pdm_data)

        if len(gtu_pdm_data.l1trg_events) > 0:

            if np.isinf(process_event_down_counter):
                frame_buffer.extend(before_trg_frames_circ_buffer)
                before_trg_frames_circ_buffer.clear()
                event_start_gtu = gtu_pdm_data.gtu

            process_event_down_counter = gtu_after_trigger

            for l1trg_ev in gtu_pdm_data.l1trg_events:
                if l1trg_ev.packet_id != packet_id:
                    raise Exception("Unexpected L1 trigger event's packet id (actual: {}, expected: {})".format(l1trg_ev.packet_id, packet_id))
                if l1trg_ev.gtu_in_packet != last_gtu_in_packet:
                    raise Exception("Unexpected L1 trigger event's gtu in packet (actual: {}, expected: {})".format(l1trg_ev.gtu_in_packet, last_gtu_in_packet))

            # pcd = gtu_pdm_data.photon_count_data
            # if len(pcd) > 0 and len(pcd[0]) > 0:
            #     visualize_frame(gtu_pdm_data, ack_l1_reader.exp_tree)

        if not np.isinf(process_event_down_counter):
            if process_event_down_counter == 0 or last_gtu_in_packet+1 == packet_size:
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

                    acquisition_file_basename = os.path.basename(context.acquisition_file)
                    kenji_l1trigger_file_basename = os.path.basename(context.kenji_l1trigger_file)

                    event_start_msg = "GLB.GTU: {} ; PCK: {} ; PCK.GTU: {}".format(event_start_gtu, packet_id, ev.gtu_in_packet)
                    event_files_msg = "ACK: {} ; TRG: {}".format(acquisition_file_basename, kenji_l1trigger_file_basename)

                    log_file.write("{} ; LAST.GTU: {} ; LAST.PCK.GTU: {} ; {}".format(event_start_msg,
                                                                             gtu_pdm_data.gtu, last_gtu_in_packet,
                                                                             event_files_msg))
                    run_event = True
                    if run_again_exclusively:
                        run_event = event_start_gtu in run_again_gtus
                    elif no_overwrite__weak_check:
                        if output_storage_provider.check_event_exists_weak(ev, event_processing.program_version):
                            # True - event does exist
                            run_event = run_again_gtus is not None and event_start_gtu in run_again_gtus

                    if run_event:

                        log_file.write(" ; PROCESSING")
                        log_file.flush()

                        event_watermark = "{}\n{}".format(
                            event_start_msg,
                            event_files_msg
                        )
                        save_fig_pathname_format = None

                        if figure_img_base_dir and figure_img_name_format:
                            proc_params_dict = proc_params.get_dict_of_str()
                            # "{program_version}"
                            # "/{triggered_pixels_group_max_gap}_{triggered_pixels_group_max_gap}_{triggered_pixels_ht_phi_num_steps}_{x_y_neighbour_selection_rules}"
                            # "_{x_y_ht_line_thickness}_{x_y_ht_phi_num_steps}_{x_y_ht_rho_step}_{x_y_ht_peak_threshold_frac_of_max}_{x_y_ht_peak_gap}"
                            # "_{x_y_ht_global_peak_threshold_frac_of_max}"
                            # "/{acquisition_file_basename}/{kenji_l1trigger_file_basename}/{gtu_global}_{packet_id}_{gtu_in_packet}/{name}.png"

                            save_fig_pathname_format = os.path.join(figure_img_base_dir, figure_img_name_format.format(
                                program_version="{program_version}",
                                name='{name}',
                                acquisition_file_basename=acquisition_file_basename,
                                kenji_l1trigger_file_basename=kenji_l1trigger_file_basename,
                                gtu_global=event_start_gtu,
                                packet_id=packet_id,
                                gtu_in_packet=ev.gtu_in_packet,
                                packet_size=packet_size,
                                num_gtu=len(frame_buffer),
                                last_gtu=gtu_pdm_data.gtu,
                                last_gtu_in_packet=last_gtu_in_packet,
                                **proc_params_dict
                            ))

                        event_processing.process_event(trigger_event_record=ev,
                                      proc_params=proc_params,
                                      do_visualization=context.visualize,
                                      save_fig_pathname_format=save_fig_pathname_format,
                                      watermark_text=event_watermark)

                        log_file.write(" ; SAVING")
                        log_file.flush()

                    ########################################################################################

                        if not no_overwrite__strong_check or not output_storage_provider.check_event_exists_strong(ev):
                            save_result = output_storage_provider.save_event(ev, save_config_info_result, update)
                            if not save_result:
                                log_file.write(" ; FAILED")
                            elif save_result=='update':
                                log_file.write(" ; UPDATED")
                            else:
                                log_file.write(" ; SAVED")
                        else:
                            log_file.write(" ; EVENT ALREADY EXISTS (STRONG CHECK)")

                    else:
                        log_file.write(" ; EVENT ALREADY EXISTS (WEAK CHECK)")

                    log_file.write(" ; EVENT FINISHED\n")
                    log_file.flush()

                process_event_down_counter = np.inf
                before_trg_frames_circ_buffer.extend(frame_buffer)
                frame_buffer.clear()
                event_start_gtu = -1

            elif process_event_down_counter > 0:
                if len(gtu_pdm_data.l1trg_events) == 0: # TODO this might require increase size of the circular buffer
                    process_event_down_counter -= 1
            else:
                raise Exception("Unexpected value of process_event_down_counter")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
