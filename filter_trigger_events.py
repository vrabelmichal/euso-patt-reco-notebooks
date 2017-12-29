import numpy as np
import os
import argparse
import sys
import re
import collections
import configparser
import json

# Workaround to use Matplotlib without DISPLAY
visualize_option_argv_key = '--visualize'
use_agg = False
if visualize_option_argv_key not in sys.argv:
    use_agg = True
else:
    visualize_option_value = sys.argv.index(visualize_option_argv_key)+1
    if len(sys.argv) > visualize_option_value and sys.argv[visualize_option_value][0] in "0nNfF":
        use_agg = True
if use_agg:
    import matplotlib as mpl
    mpl.use('Agg')

import event_processing_v1
import event_processing_v2
import event_processing_v3
import base_classes
import safe_termination
import processing_sync
#import processing_config
from postgresql_event_storage import PostgreSqlEventStorageProvider
from sqlite_event_storage import Sqlite3EventStorageProvider
from utility_funtions import str2bool_argparse
from event_reading import AcqL1EventReader, EventFilterOptions
from tsv_event_storage import TsvEventStorageProvider
from npy_l1_event_reader import NpyL1EventReader
import query_functions

def main(argv):
    # TODO replace default=None
    parser = argparse.ArgumentParser(description='Find patterns inside triggered pixes')
    # parser.add_argument('files', nargs='+', help='List of files to convert')
    parser.add_argument('-a', '--acquisition-file', default=None, help="ACQUISITION root file in \"Lech\" format (.root), or numpy ndarray (.npy created by simu2npy and npyconv) if --event-reader option is \"npy\".")
    parser.add_argument('-k', '--kenji-l1trigger-file', default=None, help="L1 trigger root file in \"Kenji\" format")
    parser.add_argument('-r', '--event-reader', default='acq', help="Event reader. Available event readers acq (default, processes acquisitions in \"Lech\" format), npy (processes acquisitions in npy format).")
    parser.add_argument('-o', '--out', default=None, help="Output specification")
    parser.add_argument('-f', '--output-type', default="stdout", help="Output type - tsv, stdout, sqlite, prostgresql")
    parser.add_argument('-c', '--calibration-map', default=None, help="Flat field map .npy file, orientation as kenji's L1 trigger")
    parser.add_argument('--gtu-before-trigger', type=int, default=None, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--gtu-after-trigger', type=int, default=None, help="Number of GTU included in track finding data before the trigger")
    # parser.add_argument('--trigger-persistency', type=int, default=2, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--packet-size', type=int, default=128, help="Number of GTU in packet")
    parser.add_argument('--config', default=os.path.realpath(os.path.join(os.path.dirname(__file__),'config.ini')), help="Configuration file")

    parser.add_argument('--run-again', default='', help="Selector defining events to be run again")
    parser.add_argument('--run-again-spec', default='', help="Specification defining events to be run again")
    parser.add_argument('--run-again-input-type', default="postgresql", help="Input type - postgresql")
    parser.add_argument('--run-again-limit', type=int, default=10000, help="Maximal number of events to be rerun based on --run-again specification")
    parser.add_argument('--run-again-offset', type=int, default=0, help="Offset of rerun events selection based on --run-again specification")

    parser.add_argument("--save-figures-base-dir", default='', help="If this option is set, matplotlib figures are saved to this directory in format defined by --figure-pathname-format option.")
    parser.add_argument('--figure-name-format',
                        default="{program_version:.2f}/config_{config_info_id}/event_{event_id}/{acquisition_file_basename}/{kenji_l1trigger_file_basename}/"\
                                "pck_{packet_id:d}/inpck_{gtu_in_packet:d}__gtu_{gtu_global:d}/{name}.png",
                        help="Format of a saved matplotib figure pathname relative to base directory.")
    parser.add_argument("--generate-web-figures", type=str2bool_argparse, default=False, help="If this option is true, matplotlib figures are generated in web directory.")
    parser.add_argument(visualize_option_argv_key, type=str2bool_argparse, default=False, help="If this option is true, matplotlib figures are shown.")
    # parser.add_argument("--savefig", type=str2bool_argparse, default=True, help="If this option is true and directories are provided, matplotlib figures are saved (default: True).")
    parser.add_argument("--no-overwrite--weak-check", type=str2bool_argparse, default=True, help="If this option is true, the existnece of records with same files and processing version is chceked BEFORE event is processed.")
    parser.add_argument("--no-overwrite--strong-check", type=str2bool_argparse, default=False, help="If this option is true, the existnece of records with same parameters is chceked AFTER event is processed")
    parser.add_argument("--dry-run", type=str2bool_argparse, default=False, help="If this option is true, the results are not saved")
    parser.add_argument("--debug", type=str2bool_argparse, default=False, help="If this option is true, various debug information might be printed")
    parser.add_argument("--update", type=str2bool_argparse, default=False, help="If this option is true, the existnece of records with same files and processing version is chceked AFTER event is processed."
                                                                                    "If event is found event other parameters are updated.")
    parser.add_argument('--first-gtu', type=int, default=0, help="GTU before will be skipped")
    parser.add_argument('--last-gtu', type=int, default=sys.maxsize, help="GTU after will be skipped")

    parser.add_argument('--save-source-data-type-num', type=str2bool_argparse, default=False, help="If true, number in --data-type-num parameter is saved in the output.")  # TODO implement
    parser.add_argument('--source-data-type-num', type=int, default=-1, help="Number indicating type of processed data in the output / database (default: -1)")
    #parser.add_argument('--additional-img-path-segment', default='', help="Additional arbitrary path segment that might be included in path given _figure_name_format config option (empty by default)")

    parser.add_argument('--filter-n-persist-gt', type=int, default=-1, help="Accept only events with at least one GTU with nPersist more than this value.")
    parser.add_argument('--filter-sum-l1-pdm-gt', type=int, default=-1, help="Accept only events with at least one GTU with sumL1PDM more than this value.")
    parser.add_argument('--filter-sum-l1-ec-one-gt', type=int, default=-1, help="Accept only events with at least one GTU with at leas one EC sumL1PDM more than this value.")
    parser.add_argument('--filter-sum-l1-pmt-one-gt', type=int, default=-1, help="Accept only events with at least one GTU with at leas one PMT sumL1PMT more than this value.")

    parser.add_argument('--algorithm', default='ver2', help="Version of the processing algorithm used")

    parser.add_argument('--lockfile-dir', default="/tmp/spb_file_processing_sync", help="Path to the directory where synchronization lockfiles are stored")

    parser.add_argument('--visualization-options', default='', help='Additional visualization options (default: "")')
    parser.add_argument('--savefig-options', default='', help='Additional saved figure options (default: "")')

    args = parser.parse_args(argv)
    config = configparser.ConfigParser()
    config.read(args.config)

    filter_options = EventFilterOptions()
    filter_options.n_persist = args.filter_n_persist_gt
    filter_options.sum_l1_pdm = args.filter_sum_l1_pdm_gt
    filter_options.sum_l1_ec_one = args.filter_sum_l1_ec_one_gt
    filter_options.sum_l1_pmt_one = args.filter_sum_l1_pmt_one_gt

    if args.event_reader == 'acq':
        event_reader = AcqL1EventReader
    elif args.event_reader == 'npy':
        event_reader = NpyL1EventReader
    else:
        raise Exception('Unknown event reader')

    if args.algorithm == 'ver1':
        event_processing = event_processing_v1.EventProcessingV1()
    elif args.algorithm == 'ver2':
        event_processing = event_processing_v2.EventProcessingV2()
    elif args.algorithm == 'ver3':
        event_processing = event_processing_v3.EventProcessingV3()
    else:
        raise Exception('Unknown algorithm')

    proc_params = event_processing.event_processing_params_class().from_global_config(config)

    flat_field_ndarray = None
    if args.calibration_map:
        flat_field_ndarray = np.load(args.calibration_map)
        proc_params.calibration_map_path = args.calibration_map

    if args.output_type == "tsv" or args.output_type == "csv":
        output_storage_provider = TsvEventStorageProvider(args.out) # TODO probably not working
    elif args.output_type == "sqlite":
        output_storage_provider = Sqlite3EventStorageProvider(args.out) # TODO probably not working
    elif args.output_type == "postgresql":
        args_list = [None, event_processing.event_processing_params_class, event_processing.event_analysis_record_class,
                     event_processing.column_info, False]
        if args.out:
            print('WARNING: pareter output specification for postgresql not supported. Using configuration file "{}"'.format(args.config))
            # args_list[0] = args.out
            # output_storage_provider = PostgreSqlEventStorageProvider(*args_list)
        # else:
        args_list[0] = config
        args_list.append(args.algorithm)
        output_storage_provider = PostgreSqlEventStorageProvider.from_global_config(*args_list)
    else:
        output_storage_provider = base_classes.BaseEventStorageProvider()

    visualization_options = None if not args.visualization_options else json.loads(args.visualization_options)
    savefig_options = None if not args.visualization_options else json.loads(args.savefig_options)

    if args.generate_web_figures:
        if "DatabaseBrowserWeb" not in config:
            raise Exception("Missing DatabaseBrowserWeb configuration section")

        figure_name_format_key = '{}_figure_name_format'.format(args.algorithm)
        for opt in ['base_figures_storage_directory', figure_name_format_key]:
            if opt not in config["DatabaseBrowserWeb"]:
                raise Exception('Missing config option "{}"'.format(opt))

        args.save_figures_base_dir = \
            os.path.realpath(config['DatabaseBrowserWeb']['base_figures_storage_directory']\
                .format(basedir=os.path.dirname(os.path.abspath(__file__))))

        args.figure_name_format = config['DatabaseBrowserWeb'][figure_name_format_key]

    processing_runs = []

    acq_trg_params_added_to_processing_runs = False

    if args.run_again:
        if args.run_again_input_type != 'postgresql':
            raise Exception('Unsupported run again input type')
        if not args.run_again:
            raise Exception('run_again is expected to contain conditions')

        run_again_storage_provider = output_storage_provider
        if args.output_type != args.run_again_input_type:
            args_list = [None, event_processing.event_processing_params_class, event_processing.event_analysis_record_class,
                         event_processing.column_info, True]
            if not args.run_again_spec:
                args_list[0] = config
                args_list.append(args.algorithm)
                run_again_storage_provider = PostgreSqlEventStorageProvider.from_global_config(*args_list)
            else:
                args_list[0] = args.run_again_spec
                run_again_storage_provider = PostgreSqlEventStorageProvider(*args_list)

        processed_files_configs__gtus_ids = collections.OrderedDict()
        # event_ids__processing_configs = collections.OrderedDict()
        processing_config_ids = set()

        run_again_events_query = 'SELECT {data_table_pk}, {config_info_table_pk}, {data_table_source_file_acquisition_full_column}, ' \
                                 '{data_table_source_file_trigger_full_column}, {data_table_global_gtu_column} ' \
                                 'FROM {data_table} '+args.run_again+' OFFSET {:d} LIMIT {:d}'.format(args.run_again_offset, args.run_again_limit) # could be unsafe
        trigger_analysis_records, all_cols = run_again_storage_provider.fetch_trigger_analysis_records(run_again_events_query)

        if args.debug:
            print("run_again_events_query", run_again_events_query)
            print("len(trigger_analysis_records)", len(trigger_analysis_records))

        for id, config_info_id, timestamp, trigger_analysis_record in trigger_analysis_records:
            key_tuple = (trigger_analysis_record.source_file_acquisition_full, trigger_analysis_record.source_file_trigger_full, config_info_id)
            if key_tuple not in processed_files_configs__gtus_ids:
                processed_files_configs__gtus_ids[key_tuple] = []
            processed_files_configs__gtus_ids[key_tuple].append(
                # getattr(trigger_analysis_record, run_again_storage_provider.data_table_pk),
                (getattr(trigger_analysis_record, run_again_storage_provider.data_table_global_gtu_column), id)
            )
            #event_ids__processing_configs[trigger_analysis_record.event_id] = config_info_id # todo translate_struct ?
            processing_config_ids.add(config_info_id)

        processing_config_info_records = {}
        #processing_config_info_columns = []
        if processing_config_ids:
            processing_config_info_records, processing_config_info_columns  = run_again_storage_provider.fetch_config_info_records(
                "SELECT * FROM {{config_info_table}} WHERE {{config_info_table_pk}} IN ({})".format(
                    ", ".join([str(cid) for cid in list(processing_config_ids)] )
                )
            )

        #translate_struct(event_ids__processing_configs, lambda config_info_id: processing_configs[config_info_id])

        if len(trigger_analysis_records) == int(args.run_again_limit):
            m = "WARNING only first {:d} events will be rerun".format(args.run_again_limit)
            print(m,file=sys.stderr)

        for k, gtus_ids in processed_files_configs__gtus_ids.items():
            source_file_acquisition, source_file_trigger, config_info_id = k

            if config_info_id not in processing_config_info_records:
                raise Exception('Unexpected state config info not fetched from processing config_infor table')

            sorted_gtus_ids = sorted(gtus_ids, key=lambda v: v[0])

            if source_file_acquisition == args.acquisition_file and args.kenji_l1trigger_file == source_file_trigger \
                and processing_config_info_records[config_info_id] == proc_params:
                processing_runs.append((source_file_acquisition, source_file_trigger, proc_params, sorted_gtus_ids, False)) # todo event_ids is wrong
                acq_trg_params_added_to_processing_runs = True
            else:
                processing_runs.append((source_file_acquisition, source_file_trigger, processing_config_info_records[config_info_id], sorted_gtus_ids, True)) # discarding config_info_id creates overhead, but gtu_before_triggerer might be changed

    if not acq_trg_params_added_to_processing_runs and args.acquisition_file and args.kenji_l1trigger_file:
        proc_params.set_config_info_id_hash()
        processing_runs.append((args.acquisition_file, args.kenji_l1trigger_file, proc_params, None, False))

    for source_file_acquisition, source_file_trigger, run_proc_params, gtus_ids, process_only_selected in processing_runs:
        if args.gtu_before_trigger is not None:
            run_proc_params.gtu_before_trigger = args.gtu_before_trigger
        if args.gtu_after_trigger is not None:
            run_proc_params.gtu_after_trigger = args.gtu_after_trigger

        read_and_process_events(source_file_acquisition, source_file_trigger,
                                args.first_gtu, args.last_gtu, args.packet_size, filter_options,
                                event_reader, output_storage_provider, event_processing,
                                args.visualize,
                                args.no_overwrite__weak_check, args.no_overwrite__strong_check, args.update,
                                args.save_figures_base_dir, args.figure_name_format, gtus_ids, process_only_selected,
                                run_proc_params if run_proc_params is not None else proc_params,
                                False, sys.stdout, args.lockfile_dir, flat_field_ndarray,
                                visualization_options, savefig_options)

        if safe_termination.terminate_flag:
            break

    output_storage_provider.finalize()


def read_and_process_events(source_file_acquisition, source_file_trigger, first_gtu, last_gtu, packet_size,
                            filter_options,
                            event_reader_cls, output_storage_provider, event_processing,
                            do_visualization=True,
                            no_overwrite__weak_check=True, no_overwrite__strong_check=False, update=True,
                            figure_img_base_dir=None, figure_img_name_format=None,
                            run_again_gtus_ids=None, run_again_exclusively=False,
                            proc_params=None, dry_run=False,
                            log_file=sys.stdout, lockfile_dir="/tmp/trigger-events-processing", flat_field_ndarray=None,
                            visualization_options=None, savefig_options=None
                            ):

    if run_again_gtus_ids is None and run_again_exclusively:
        raise Exception("run_again_gtus is None but run_again_exclusively is True")

    figure_img_name_format_nested = re.sub(r'\{(program_version|name)(:[^}]+)?\}', r'{\g<0>}', figure_img_name_format)

    gtu_before_trigger = proc_params.gtu_before_trigger
    gtu_after_trigger = proc_params.gtu_after_trigger

    before_trg_frames_circ_buffer = collections.deque(maxlen=gtu_before_trigger+1) # +1 -> the last gtu in the buffer is triggered
    frame_buffer = []
    gtu_after_trigger -= 1 # -1 -> when counter is 0 the frame is still saved

    process_event_down_counter = np.inf
    packet_id = -1

    event_start_gtu = -1

    proc_params_id = output_storage_provider.save_config_info(proc_params) # should also
    proc_params_str = str(proc_params)

    packet_frames = [] # !!!!!!!

    with event_reader_cls(source_file_acquisition, source_file_trigger) as event_reader:

        for gtu_pdm_data in event_reader.iter_gtu_pdm_data():

            last_gtu_in_packet = gtu_pdm_data.gtu % packet_size
            if last_gtu_in_packet == 0:
                packet_id += 1 # starts at -1
                before_trg_frames_circ_buffer.clear()
                frame_buffer.clear()
                process_event_down_counter = np.inf
                event_start_gtu = -1

                packet_frames.clear()

            packet_frames.append(gtu_pdm_data)

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
                    if event_start_gtu >= first_gtu and event_start_gtu <= last_gtu and filter_options.check_pdm_gtu(gtu_pdm_data):

                        ########################################################################################

                        ev = output_storage_provider.event_processing_analysis_record_class()
                        ev.program_version = event_processing.program_version
                        ev.source_file_acquisition_full = source_file_acquisition
                        ev.source_file_trigger_full = source_file_trigger
                        ev.exp_tree = event_reader.exp_tree
                        ev.global_gtu = event_start_gtu
                        ev.packet_id = packet_id
                        ev.gtu_in_packet = event_start_gtu % packet_size
                        ev.gtu_data = frame_buffer
                        ev.config_info = proc_params

                        acquisition_file_basename = os.path.basename(source_file_acquisition)
                        kenji_l1trigger_file_basename = os.path.basename(source_file_trigger)

                        event_start_msg = "GLB.GTU: {} ; PCK: {} ; PCK.GTU: {}".format(event_start_gtu, packet_id, ev.gtu_in_packet)
                        event_files_msg = "ACK: {} ; TRG: {}".format(acquisition_file_basename, kenji_l1trigger_file_basename)

                        log_file.write("{} ; LAST.GTU: {} ; LAST.PCK.GTU: {} ; {}".format(event_start_msg,
                                                                                 gtu_pdm_data.gtu, last_gtu_in_packet,
                                                                                 event_files_msg))
                        run_event = True
                        run_event_id = None
                        not_run_reason = ''

                        processing_lock = None

                        if lockfile_dir is not None:
                            processing_lock = processing_sync.acquire_lock(lockfile_dir,
                                    [source_file_acquisition, source_file_trigger, proc_params_str, event_start_gtu,
                                     output_storage_provider.get_id_str()],
                                    True)
                            if not processing_lock:
                                run_event = False
                                not_run_reason = 'LOCK NOT ACQUIRED'

                        if run_event:
                            if run_again_exclusively:

                                run_again_entry_index = next((i for i, v in enumerate(run_again_gtus_ids) if v[0] == event_start_gtu), -1)
                                run_event = run_again_entry_index >= 0
                                # run_event = event_start_gtu in run_again_gtus

                                if run_event:
                                    run_event_id = run_again_gtus_ids[run_again_entry_index][1]
                                    log_file.write(" ; GLB.GTU IN EXCLUSIVE RUN AGAIN LIST")
                                else:
                                    not_run_reason = 'GLB.GTU NOT IN EXCLUSIVE RUN AGAIN LIST'
                            elif no_overwrite__weak_check:
                                if output_storage_provider.check_event_exists_weak(ev):
                                    # True - event does exist
                                    if run_again_gtus_ids is not None:

                                        run_again_entry_index = next((i for i, v in enumerate(run_again_gtus_ids) if v[0] == event_start_gtu), -1)
                                        run_event = run_again_entry_index >= 0
                                        # run_event = event_start_gtu in run_again_gtus

                                        if run_event:
                                            run_event_id = run_again_gtus_ids[run_again_entry_index][1]
                                            log_file.write(" ; GLB.GTU IN EXCLUSIVE RUN AGAIN LIST")
                                        else:
                                            not_run_reason = 'GLB.GTU NOT IN RUN AGAIN LIST'
                                    else:
                                        run_event = False
                                        not_run_reason = 'EVENT ALREADY EXISTS (WEAK CHECK)'

                        if run_event:
                            if run_event_id is not None and run_event_id >= 0:
                                ev.event_id = run_event_id
                            else:
                                ev.set_event_id_hash()

                            # in the current implementation the event_id should be -1 if run_event_id is not available

                            log_file.write(" ; PROCESSING")
                            log_file.flush()

                            event_watermark = "{}\n{}".format(
                                event_start_msg,
                                event_files_msg
                            )
                            save_fig_pathname_format = None

                            if figure_img_base_dir and figure_img_name_format:
                                save_fig_pathname_format = query_functions.get_figure_pathname(figure_img_base_dir, figure_img_name_format, ev, '{name}')

                            event_processing.process_event(trigger_event_record=ev,
                                                           proc_params=proc_params,
                                                           do_visualization=do_visualization,
                                                           do_savefig=save_fig_pathname_format is not None,
                                                           save_fig_pathname_format=save_fig_pathname_format,
                                                           watermark_text=event_watermark,
                                                           packet_gtu_data=packet_frames,
                                                           calibration_arr=flat_field_ndarray,
                                                           visualization_options=visualization_options,
                                                           savefig_options=savefig_options)

                            log_file.write(" ; SAVING")
                            log_file.flush()

                        ########################################################################################

                            if not no_overwrite__strong_check or not output_storage_provider.check_event_exists_strong(ev):
                                if not dry_run:
                                    save_result = output_storage_provider.save_event(ev, update)
                                    if not save_result:
                                        log_file.write(" ; FAILED")
                                    elif save_result=='update':
                                        log_file.write(" ; UPDATED")
                                    else:
                                        log_file.write(" ; SAVED")
                                else:
                                    log_file.write(" ; DRY RUN - NOT SAVED")
                            else:
                                log_file.write(" ; EVENT ALREADY EXISTS (STRONG CHECK)")

                        else:
                            log_file.write(' ; '+not_run_reason)


                        if lockfile_dir is not None and processing_lock:
                            processing_sync.release_lock(processing_lock)

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

            if safe_termination.terminate_flag:
                return


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
