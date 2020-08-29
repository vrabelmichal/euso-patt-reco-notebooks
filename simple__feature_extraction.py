import numpy as np
import os
import argparse
import sys
import collections
import configparser
import json
import time
import traceback

# Workaround to use Matplotlib without DISPLAY
_feature_extraction_visualize_option_argv_key = '--visualize'
_feature_extraction_use_agg = False

def _feature_extraction_is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

if 'FEATURE_EXTRACION_KEEP_MPL_BACKEND' in globals() and not globals()['FEATURE_EXTRACION_KEEP_MPL_BACKEND']:
    _feature_extraction_use_agg = True
elif not _feature_extraction_is_interactive():
    if _feature_extraction_visualize_option_argv_key not in sys.argv:
        _feature_extraction_use_agg = True
    else:
        visualize_option_value = sys.argv.index(_feature_extraction_visualize_option_argv_key) + 1
        if len(sys.argv) > visualize_option_value and sys.argv[visualize_option_value][0] in "0nNfF":
            _feature_extraction_use_agg = True

# Another check possibility
# mpl.rcParams['backend'].startswith('module://ipykernel.pylab.backend')

if _feature_extraction_use_agg:
    import matplotlib as mpl
    mpl.use('Agg')

del _feature_extraction_use_agg
del _feature_extraction_visualize_option_argv_key

from event_processing import base_classes
from utils import processing_sync, query_functions, safe_termination
from utils.utility_functions import merge_config_and_run_args

from utils.utility_functions import str2bool_argparse
from event_reading import AcqL1EventReader, EventFilterOptions
from event_storage.tsv_event_storage import TsvEventStorageProvider
from event_reading.npy_l1_event_reader import NpyL1EventReader
from event_reading.npyconv_event_reader import NpyconvEventReader
from event_storage.stdout_v3_event_storage import StdoutV3EventStorageProvider
from event_storage.dumpfile_v3_event_storage import DumpfileV3EventStorageProvider

OBJ_CACHE = {}


class FeatureExtractionParams(base_classes.BaseProcessingParams):
    acquisition_file = None
    trigger_file = None
    output_type = 'stdout'
    event_reader = 'acq'
    out = None,     # output specification

    calibration_map_path = None
    calibration_map_is_inversed = False
    _calibration_map_arr = None

    gtu_before_trigger = 4
    gtu_after_trigger = 4
    packet_size = 128
    algorithm = 'workshop'
    first_gtu = 0
    last_gtu = sys.maxsize
    lockfile_dir = '/tmp/spb_file_processing_sync'
    filter_n_persist_gt = None #(int,None)
    filter_sum_l1_pdm_gt = None
    filter_sum_l1_ec_one_gt = None
    filter_sum_l1_pmt_one_gt = None
    save_figures_base_dir = None
    figure_name_format = "{program_version:.2f}/config_{config_info_id}/{source_files_combined}/"\
                                "pck_{packet_id:d}/inpck_{gtu_in_packet:d}__gtu_{gtu_global:d}/event_{event_id}/{name}.{ext}"
    no_overwrite__weak_check = True
    no_overwrite__strong_check = False
    visualize = False
    dry_run = False
    debug = 0
    update = False
    source_data_type_num = -1
    visualization_options = None
    savefig_options = None
    acq_base_infile_path = ''
    trg_base_infile_path = ''
    cache_storage_provider = False
    raise_on_proc_error = False
    always_triggered_frames_first = None
    always_triggered_frames_last = None
    npyconv_fix_offset = 30
    weak_check__checked_columns = 'default' # default, default_w_src_dt_n
    storage_provider_init_params = None

    proc_params_config_file = os.path.realpath(os.path.join(os.path.dirname(__file__), 'config.ini'))
    proc_params_config_dump_file = None

    skip_process_single_projection = False
    dump_npy = False

    _attribute_type_hints = dict(
        acquisition_file=(str,None),
        trigger_file=(str,None),
        out=(str,None),
        event_reader=str,
        output_type=str,
        calibration_map_path=(str,None),
        calibration_map_is_inversed=bool,
        gtu_before_trigger = int,
        gtu_after_trigger = int,
        packet_size = int,
        algorithm = str,
        first_gtu = int,
        last_gtu = int,
        lockfile_dir = str,
        filter_n_persist_gt = (int,None),
        filter_sum_l1_pdm_gt = (int, None),
        filter_sum_l1_ec_one_gt = (int, None),
        filter_sum_l1_pmt_one_gt = (int, None),
        save_figures_base_dir = (str,None),
        figure_name_format = str,
        no_overwrite__weak_check=bool,
        no_overwrite__strong_check=bool,
        visualize=bool,
        dry_run=bool,
        debug=int,
        update=bool,
        source_data_type_num=int,
        visualization_options=(str,None),
        savefig_options=(str,None),
        acq_base_infile_path=str,
        trg_base_infile_path=str,
        cache_storage_provider=bool,
        proc_params_config_file=str,
        raise_on_proc_error=bool,
        always_triggered_frames_first=(int,None),
        always_triggered_frames_last=(int,None),
        npyconv_fix_offset=int,
        weak_check__checked_columns=str,
        storage_provider_init_params=(str,None),
        proc_params_config_dump_file=(str,None),
        skip_process_single_projection=bool
    )


class FeatureExtractionRunAgainParams(base_classes.BaseProcessingParams):
    selector = ''
    spec = ''
    input_type = 'none'  # global_gtu_list
    limit = 10000
    offset = 0
    data = None
    do_weak_check = True

def main(argv):

    parser = argparse.ArgumentParser(description='Find patterns inside triggered pixes')
    parser.add_argument('--config', default=os.path.realpath(os.path.join(os.path.dirname(__file__),'config.ini')), help="Configuration file")

    parser.add_argument('-a', '--acquisition-file', required=True, help="ACQUISITION root file in \"Lech\" format (.root), or numpy ndarray (.npy created by simu2npy and npyconv) if --event-reader option is \"npy\".")
    parser.add_argument('-k', '--trigger-file', help="L1 trigger root file in \"Kenji\" format")
    parser.add_argument('-r', '--event-reader', help="Event reader. Available event readers acq (default, processes acquisitions in \"Lech\" format), npy (processes acquisitions in npy format).")
    parser.add_argument('-o', '--out', help="Output specification")
    parser.add_argument('-f', '--output-type', help="Output type - tsv, stdout, sqlite, prostgresql")
    parser.add_argument('-c', '--calibration-map', help="Flat field map .npy file, orientation in \"L1 trigger\" format")
    parser.add_argument('--calibration-map-is-inversed', help="If true, calibration map is presumed to be inversed values")
    parser.add_argument('--gtu-before-trigger', type=int, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--gtu-after-trigger', type=int, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--packet-size', type=int, help="Number of GTU in packet")

    parser.add_argument("--save-figures-base-dir", default='', help="If this option is set, matplotlib figures are saved to this directory in format defined by --figure-pathname-format option.")
    parser.add_argument('--figure-name-format',
                        default="{program_version:.2f}/config_{config_info_id}/{source_files_combined}/"\
                                "pck_{packet_id:d}/inpck_{gtu_in_packet:d}__gtu_{gtu_global:d}/event_{event_id}/{name}.{ext}",
                        help="Format of a saved matplotib figure pathname relative to base directory.")
    parser.add_argument("--generate-web-figures", type=str2bool_argparse, help="If this option is true, matplotlib figures are generated in web directory.")  # default=False,
    parser.add_argument(_feature_extraction_visualize_option_argv_key, type=str2bool_argparse, help="If this option is true, matplotlib figures are shown.")  # default=False,

    parser.add_argument("--no-overwrite--weak-check", type=str2bool_argparse, help="If this option is true, the existnece of records with same files and processing version is chceked BEFORE event is processed.")  # default=True,
    parser.add_argument("--no-overwrite--strong-check", type=str2bool_argparse, help="If this option is true, the existnece of records with same parameters is chceked AFTER event is processed")  # default=False,
    parser.add_argument("--dry-run", type=str2bool_argparse, help="If this option is true, the results are not saved")  # default=False,
    parser.add_argument("--debug", type=int, help="If this option is greater than 0, various debug information might be printed")  # default=False,
    parser.add_argument("--update", type=str2bool_argparse, help="If this option is true, the existnece of records with same files and processing version is chceked AFTER event is processed. If event is found event other parameters are updated.")  # default=False,
    parser.add_argument('--first-gtu', type=int, help="GTU before will be skipped")  # default=0,
    parser.add_argument('--last-gtu', type=int, help="GTU after will be skipped")  # default=sys.maxsize,

    parser.add_argument('--source-data-type-num', type=int, help="Number indicating type of processed data in the output / database (default: -1)")  # default=-1,

    parser.add_argument('--filter-n-persist-gt', type=int, help="Accept only events with at least one GTU with nPersist more than this value.")  # default=-1,
    parser.add_argument('--filter-sum-l1-pdm-gt', type=int, help="Accept only events with at least one GTU with sumL1PDM more than this value.")  # default=-1,
    parser.add_argument('--filter-sum-l1-ec-one-gt', type=int, help="Accept only events with at least one GTU with at leas one EC sumL1PDM more than this value.")  # default=-1,
    parser.add_argument('--filter-sum-l1-pmt-one-gt', type=int, help="Accept only events with at least one GTU with at leas one PMT sumL1PMT more than this value.")  # default=-1,

    parser.add_argument('--algorithm', help="Version of the processing algorithm used")  # default='workshop',

    parser.add_argument('--lockfile-dir', help="Path to the directory where synchronization lockfiles are stored")  # default="/tmp/spb_file_processing_sync",

    parser.add_argument('--visualization-options', help='Additional visualization options (JSON expected, default: EMPTY)')  # default='',
    parser.add_argument('--savefig-options', help='Additional saved figure options (JSON expected, default: EMPTY)')  # default='',
    parser.add_argument('--storage-provider-init-params', help='Parameters for in (JSON expected, default: EMPTY)')  # default='',

    parser.add_argument('--acq-base-infile-path', help='Common part of path used to drive unique acquisition file name, if empty, file basename is used (default: "")')  # default='',
    parser.add_argument('--trg-base-infile-path', help='Common part of path used to drive unique trigger file name, if empty, file basename is used (default: "")')  # default='',
    parser.add_argument("--cache-storage-provider", type=str2bool_argparse, help="If this option is true, storage provider is cached and not finalized.")  # default=False,

    parser.add_argument("--raise-on-proc-error", type=str2bool_argparse, help="If this option is true, storage provider is cached and not finalized.")  # default=False,
    parser.add_argument("--skip-process-single-projection", type=str2bool_argparse, help="For dubugging purposes.")  # default=False,
    parser.add_argument("--dump-npy", type=str2bool_argparse, help="For dubugging purposes.")  # default=False,

    parser.add_argument('--run-again-selector', help="Selector defining events to be run again")  # default='',
    parser.add_argument('--run-again-spec', help="Specification defining events to be run again")  # default='',
    parser.add_argument('--run-again-input-type', help="Input type - postgresql")  # default="postgresql",
    parser.add_argument('--run-again-limit', type=int, help="Maximal number of events to be rerun based on --run-again specification")  # default=10000,
    parser.add_argument('--run-again-offset', type=int, help="Offset of rerun events selection based on --run-again specification")  # default=0,

    parser.add_argument('--npyconv-fix-offset', type=int, help="")
    parser.add_argument('--always-triggered-frames-first', type=int, help="")
    parser.add_argument('--always-triggered-frames-last', type=int, help="")

    parser.add_argument('--weak-check--checked-columns', help="Properties selected for the weak check of entry existence (Available values: default, default_w_src_dt_n)")

    parser.add_argument('--config-dump-file', type=str, help="If provided, processing parameters config dump file is saved here")

    run_args = parser.parse_args(argv)

    config = configparser.ConfigParser()
    if not config.read(run_args.config):
        raise RuntimeError('Unable to load config file')

    fe_params = FeatureExtractionParams()
    run_again_params = FeatureExtractionRunAgainParams()

    fe_params.read_from_global_config(config)
    run_again_params.read_from_global_config(config)

    merge_config_and_run_args(run_args, fe_params,
                              ('acquisition_file', 'trigger_file', 'out', 'output_type', ('config','proc_params_config_file'),
                               'event_reader',
                               ('calibration_map','calibration_map_path'), 'calibration_map_is_inversed',
                               'gtu_before_trigger', 'gtu_after_trigger',
                               'packet_size', 'algorithm', 'lockfile_dir',
                               'first_gtu', 'last_gtu',
                               'filter_n_persist_gt', 'filter_sum_l1_pdm_gt', 'filter_sum_l1_ec_one_gt',
                               'filter_sum_l1_pmt_one_gt',
                               'save_figures_base_dir', 'figure_name_format',
                               'no_overwrite__weak_check', 'no_overwrite__strong_check',
                               _feature_extraction_visualize_option_argv_key[2:], 'dry_run', 'debug', 'update',
                               'source_data_type_num',
                               'visualization_options', 'savefig_options', 'storage_provider_init_params',
                               'acq_base_infile_path', 'trg_base_infile_path',
                               'cache_storage_provider', 'raise_on_proc_error', 'skip_process_single_projection', 'dump_npy',
                               'npyconv_fix_offset', 'always_triggered_frames_first', 'always_triggered_frames_last',
                               'weak_check__checked_columns', ('config_dump_file','proc_params_config_dump_file')), '')

    merge_config_and_run_args(run_args, run_again_params, ('selector', 'spec', 'input_type', 'limit', 'offset'), 'run_again_')

    feature_extraction(fe_params, run_again_params, config)


def feature_extraction(params, run_again_params=None, global_config=None):

    global OBJ_CACHE

    def get_global_config():
        nonlocal global_config
        if global_config is None:
            global_config = configparser.ConfigParser()
            global_config.read(params.proc_params_config_file)
        return global_config

    #####################################################
    # EventFilterOptions

    filter_options = EventFilterOptions()
    filter_options.n_persist = params.filter_n_persist_gt
    filter_options.sum_l1_pdm = params.filter_sum_l1_pdm_gt
    filter_options.sum_l1_ec_one = params.filter_sum_l1_ec_one_gt
    filter_options.sum_l1_pmt_one = params.filter_sum_l1_pmt_one_gt

    #####################################################
    # new_event_reader_func

    source_file_trigger_required = True

    if params.event_reader == 'acq':
        new_event_reader_func = lambda source_file_acquisition_full, source_file_trigger_full: \
            AcqL1EventReader(source_file_acquisition_full, source_file_trigger_full)
    elif params.event_reader == 'npy':
        new_event_reader_func = lambda source_file_acquisition_full, source_file_trigger_full: \
            NpyL1EventReader(source_file_acquisition_full, source_file_trigger_full)
    elif params.event_reader == 'npyconv':
        source_file_trigger_required = False
        new_event_reader_func = lambda source_file_acquisition_full, source_file_trigger_full: \
            NpyconvEventReader(source_file_acquisition_full, source_file_trigger_full,
                             background_source=None,
                             fix_length=128, fix_offset=params.npyconv_fix_offset)
    else:
        raise Exception('Unknown event reader')

    #####################################################
    # event_processing

    if isinstance(params.algorithm, base_classes.BaseEventProcessing):
        event_processing = params.algorithm
    elif params.algorithm == 'workshop':
        pass
    else:
        raise RuntimeError('Unsupported event processing algorithm')

    #####################################################
    # calibration_map

    calibration_map = None
    if params.calibration_map_path:
        if 'calibration_map' in OBJ_CACHE and OBJ_CACHE['calibration_map'][0] == params.calibration_map_path:
            calibration_map = OBJ_CACHE['calibration_map'][1]
        else:
            calibration_map = np.load(params.calibration_map_path)
            OBJ_CACHE['calibration_map'] = (params.calibration_map_path, calibration_map)

    #####################################################
    # proc_params

    proc_params_cache_key = (params.algorithm, params.proc_params_config_file, params.calibration_map_path)

    if 'proc_params' in OBJ_CACHE and OBJ_CACHE['proc_params'][0] == proc_params_cache_key:
        proc_params = OBJ_CACHE['proc_params'][1]
    else:
        proc_params = event_processing.event_processing_params_class().from_global_config(get_global_config())
        proc_params.calibration_map_path = params.calibration_map_path
        proc_params._calibration_map_arr = calibration_map
        proc_params._calibration_map_is_inversed = params.calibration_map_is_inversed
        proc_params.set_config_info_id_hash()

        OBJ_CACHE['proc_params'] = (proc_params_cache_key, proc_params)

    proc_params._skip_process_single_projection = params.skip_process_single_projection
    proc_params._dump_npy = params.dump_npy

    #####################################################
    # output_storage_provider

    if isinstance(params.out, str) and len(params.out) > 0:
        params.out = json.loads(params.out)

    osp_cache_key = (params.output_type, params.proc_params_config_file, params.out,
                     event_processing.event_processing_params_class, event_processing.event_analysis_record_class,
                     event_processing.column_info)

    if params.cache_storage_provider and 'output_storage_provider' in OBJ_CACHE and OBJ_CACHE['output_storage_provider'][0] == osp_cache_key:
        output_storage_provider = OBJ_CACHE['output_storage_provider'][1]
    else:
        base_osp_initialize_params = dict(
            proc_params=proc_params,
            attr_max_numbering=event_processing.get_attr_numbering_dict(proc_params))

        osp_initialize_params = base_osp_initialize_params \
            if not params.storage_provider_init_params else \
            {**json.loads(params.storage_provider_init_params), **base_osp_initialize_params}

        if params.output_type == "tsv" or params.output_type == "csv":
            output_storage_provider = TsvEventStorageProvider(params.out)                                                  # TODO probably not working !!!!!!!!!!!!!
        else:
            if params.output_type == 'dumpfile':
                output_storage_provider_class = DumpfileV3EventStorageProvider
            else:
                output_storage_provider_class = StdoutV3EventStorageProvider

            output_storage_provider = output_storage_provider_class(
                event_processing.event_processing_params_class, event_processing.event_analysis_record_class)

            try:
                # order is intentional
                osp_initialize_params = {
                    **dict(get_global_config()[output_storage_provider.__class__.__name__]),
                    **osp_initialize_params
                }
            except KeyError:
                pass

        output_storage_provider.initialize(**osp_initialize_params)
        if params.cache_storage_provider:
            if 'output_storage_provider' in OBJ_CACHE and OBJ_CACHE['output_storage_provider'] is not None:
                OBJ_CACHE['output_storage_provider'][1].finalize()
            OBJ_CACHE['output_storage_provider'] = (osp_cache_key, output_storage_provider)

    #####################################################
    # visualization_options and savefig_options

    visualization_options = None if not params.visualization_options else json.loads(params.visualization_options)
    savefig_options = None if not params.savefig_options else json.loads(params.savefig_options)

    #####################################################

    processing_runs = []

    #####################################################

    # run again params

    acq_trg_params_added_to_processing_runs = False

    if run_again_params and run_again_params.input_type != 'none' and run_again_params.input_type is not None:

        if run_again_params.input_type == 'global_gtu_list':
            processing_runs.append((params.acquisition_file, params.trigger_file, proc_params, run_again_params.data, True, False,
                                    lambda v: v, None, run_again_params.do_weak_check)) # TODO run_again_params.do_weak_check is not entirely correct
            acq_trg_params_added_to_processing_runs = True
        # elif run_again_params.input_type == 'postgresql': pass
        else:
            raise RuntimeError('Unsupported run again input type')

    if not acq_trg_params_added_to_processing_runs and params.acquisition_file and (params.trigger_file or not source_file_trigger_required):
        processing_runs.append((params.acquisition_file, params.trigger_file, proc_params, None, False, False, None, None, True))

    #####################################################

    acq_base_infile_path = '' if not params.acq_base_infile_path else os.path.normpath(params.acq_base_infile_path)

    trg_base_infile_path = '' if not params.trg_base_infile_path else os.path.normpath(params.trg_base_infile_path)

    #####################################################
    # calling read_and_process_events

    for source_file_acquisition, source_file_trigger, run_proc_params, gtus_ids, \
        process_only_selected, run_again_consider_event_id, run_again__get_gtu, run_again__get_event_id, run_again__do_weak_check \
            in processing_runs:

        if params.gtu_before_trigger is not None:
            run_proc_params.gtu_before_trigger = params.gtu_before_trigger
        if params.gtu_after_trigger is not None:
            run_proc_params.gtu_after_trigger = params.gtu_after_trigger

        read_and_process_events(source_file_acquisition, source_file_trigger,
                                params.first_gtu, params.last_gtu, params.packet_size, filter_options,
                                new_event_reader_func, output_storage_provider, event_processing,
                                params.visualize,
                                params.no_overwrite__weak_check, params.no_overwrite__strong_check, params.update,
                                params.save_figures_base_dir, params.figure_name_format,
                                gtus_ids, process_only_selected,
                                run_proc_params if run_proc_params is not None else proc_params,
                                False, sys.stdout, params.lockfile_dir,
                                visualization_options, savefig_options,
                                acq_base_infile_path, trg_base_infile_path, params.source_data_type_num, params.raise_on_proc_error,
                                params.always_triggered_frames_first, params.always_triggered_frames_last,
                                params.weak_check__checked_columns,
                                run_again_consider_event_id, run_again__get_gtu, run_again__get_event_id, run_again__do_weak_check,
                                params.debug, params.proc_params_config_dump_file
                                )

        if safe_termination.terminate_flag:
            raise safe_termination.SafelyTerminated('After read_and_process_events', __file__)

    #####################################################

    if not params.cache_storage_provider:
        output_storage_provider.finalize()


def read_and_process_events(source_file_acquisition_full, source_file_trigger_full, first_gtu, last_gtu, packet_size,
                            filter_options,
                            new_event_reader_func, output_storage_provider, event_processing,
                            do_visualization=True,
                            no_overwrite__weak_check=True, no_overwrite__strong_check=False, update=True,
                            figure_img_base_dir=None, figure_img_name_format=None,
                            run_again__gtus_ids=None, run_again__only_selected=False,
                            proc_params=None, dry_run=False,
                            log_file=sys.stdout, lockfile_dir="/tmp/trigger-events-processing",
                            visualization_options=None, savefig_options=None,
                            acq_base_infile_path='', trg_base_infile_path='', source_data_type_num=None, raise_on_processing_error=True,
                            always_triggered_frames_first=None, always_triggered_frames_last=None, weak_check__checked_columns='default',
                            run_again__consider_event_id=False,
                            run_again__get_gtu=lambda v: v[0], run_again__get_event_id=lambda v: v[1],
                            run_again__do_weak_check=True, debug=False, config_dump_file_path=None):

    if run_again__gtus_ids is None and run_again__only_selected:
        raise Exception("run_again_gtus is None but run_again_exclusively is True")

    #  TODO should be following line used?
    # figure_img_name_format_nested = re.sub(r'\{(program_version|name)(:[^}]+)?\}', r'{\g<0>}', figure_img_name_format)

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

    # event_exists_weak_check_conds = dict(output_storage_provider.default_event_exists_weak_check_conds)
    # event_exists_weak_check_conds['{source_file_acquisition_full_column}'] = \
    #     ('LIKE', lambda trigger_event: '%{}'.format(trigger_event.source_file_acquisition_full))
    # event_exists_weak_check_conds['{source_file_trigger_full_column}'] = \
    #     ('LIKE', lambda trigger_event: '%{}'.format(trigger_event.source_file_trigger_full))

    with new_event_reader_func(source_file_acquisition_full, source_file_trigger_full) as event_reader:
        if not acq_base_infile_path:
            source_file_acquisition = os.path.basename(source_file_acquisition_full)
        elif source_file_acquisition_full.startswith(acq_base_infile_path):
            source_file_acquisition = source_file_acquisition_full[len(acq_base_infile_path):]
        else:
            source_file_acquisition = source_file_acquisition_full

        if source_file_trigger_full is not None:
            if not trg_base_infile_path:
                source_file_trigger = os.path.basename(source_file_trigger_full)
            elif source_file_trigger_full.startswith(trg_base_infile_path):
                source_file_trigger = source_file_trigger_full[len(trg_base_infile_path):]
            else:
                source_file_trigger = source_file_trigger_full
        else:
            source_file_trigger = None

        if source_file_acquisition[0] == '/':  # unix specific
            source_file_acquisition = source_file_acquisition[1:]

        if source_file_trigger is not None and source_file_trigger[0] == '/':  # unix specific
            source_file_trigger = source_file_trigger[1:]

        if config_dump_file_path is not None:
            config_dump_file_path = _format_config_dump_pathname(
                config_dump_file_path, source_file_acquisition, source_file_trigger, event_processing, proc_params)
            os.makedirs(os.path.dirname(config_dump_file_path), exist_ok=True)
            with open(config_dump_file_path, 'w') as config_dump_file:
                config_dump_file.write('[{}]\n{}'.format(proc_params.__class__.__name__, proc_params_str))

        for gtu_pdm_data in event_reader.iter_gtu_pdm_data():
            # print('frame loop')
            last_gtu_in_packet = gtu_pdm_data.gtu % packet_size
            if last_gtu_in_packet == 0:
                packet_id += 1 # starts at -1
                before_trg_frames_circ_buffer.clear()
                frame_buffer.clear()
                process_event_down_counter = np.inf
                event_start_gtu = -1

                # packet_frames.clear()
                packet_frames = [] # to assure last_packet_gtu_data is not packet_gtu_data

            packet_frames.append(gtu_pdm_data)

            if np.isinf(process_event_down_counter):
                before_trg_frames_circ_buffer.append(gtu_pdm_data)
            else:
                frame_buffer.append(gtu_pdm_data)

            if len(gtu_pdm_data.l1trg_events) > 0 or \
                    (
                        (always_triggered_frames_first is not None or always_triggered_frames_last is not None) and \
                        (always_triggered_frames_first is None or always_triggered_frames_first <= gtu_pdm_data.gtu) and \
                        (always_triggered_frames_last is None or always_triggered_frames_last >= gtu_pdm_data.gtu)
                    ):

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
                        ev.source_file_acquisition_full = source_file_acquisition_full
                        ev.source_file_trigger_full = source_file_trigger_full
                        ev.exp_tree = event_reader.exp_tree
                        ev.global_gtu = event_start_gtu
                        ev.packet_id = packet_id
                        ev.gtu_in_packet = event_start_gtu % packet_size
                        ev.gtu_data = frame_buffer
                        ev.config_info = proc_params
                        ev.timestamp = time.time()
                        ev.source_file_acquisition = source_file_acquisition
                        ev.source_file_trigger = source_file_trigger
                        if source_data_type_num is not None:
                            ev.source_data_type_num = source_data_type_num

                        event_start_msg = "GLB.GTU: {} ; PCK: {} ; PCK.GTU: {}".format(event_start_gtu, packet_id, ev.gtu_in_packet)
                        event_files_msg = "ACK: {} ; TRG: {}".format(source_file_acquisition, source_file_trigger)

                        log_file.write("{} ; LAST.GTU: {} ; LAST.PCK.GTU: {} ; {}".format(event_start_msg, gtu_pdm_data.gtu, last_gtu_in_packet, event_files_msg))
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

                        weak_check_required = True
                        run_again_entry_index = -1

                        if run_event and run_again__gtus_ids is not None:
                            run_again_entry_index = next((i for i, v in enumerate(run_again__gtus_ids) if
                                                          run_again__get_gtu(v) == event_start_gtu), -1)
                            if run_again_entry_index < 0:
                                if run_again__only_selected:
                                    run_event = False
                                    not_run_reason = 'GLB.GTU NOT IN EXCLUSIVE RUN AGAIN LIST'
                            else:
                                if run_again__only_selected:
                                    weak_check_required = run_again__do_weak_check
                                    log_file.write(" ; GLB.GTU IN EXCLUSIVE RUN AGAIN LIST")
                                else:
                                    weak_check_required = False # event is getting processed anyway
                                    log_file.write(" ; GLB.GTU IN RUN AGAIN LIST")

                        if run_event and no_overwrite__weak_check and weak_check_required:
                            if output_storage_provider.check_event_exists_weak(ev, checked_columns_dict=weak_check__checked_columns):
                                # True - event does exist
                                run_event = False
                                not_run_reason = 'EVENT ALREADY EXISTS (WEAK CHECK)'

                        if run_event and run_again__consider_event_id and run_again_entry_index >= 0:
                            run_event_id = run_again__get_event_id(run_again__gtus_ids[run_again_entry_index])  # [1]

                        if run_event:
                            if run_event_id is not None and run_event_id >= 0:
                                ev.event_id = run_event_id
                            else:
                                ev.set_event_id_hash()

                            # in the current implementation the event_id should be -1 if run_event_id is not available

                            log_file.write(" ; PROCESSING ")
                            log_file.flush()

                            event_watermark = "{}\n{}".format(
                                event_start_msg,
                                event_files_msg
                            )
                            save_fig_pathname_format = None

                            if figure_img_base_dir and figure_img_name_format:
                                save_fig_pathname_format = query_functions.get_figure_pathname(figure_img_base_dir, figure_img_name_format, ev, '{name}')

                            ret_ev = None
                            ret_ev_list = None

                            try:
                                ret_ev = event_processing.process_event(pdm_analysis_record=ev,
                                                                        proc_params=proc_params,
                                                                        do_visualization=do_visualization,
                                                                        do_savefig=save_fig_pathname_format is not None,
                                                                        save_fig_pathname_format=save_fig_pathname_format,
                                                                        watermark_text=event_watermark,
                                                                        packet_gtu_data=packet_frames,
                                                                        visualization_options=visualization_options,
                                                                        savefig_options=savefig_options,
                                                                        debug=debug)

                                if ret_ev is None:
                                    raise RuntimeError('process_event returned None')
                                elif isinstance(ret_ev, (tuple, list)):
                                    if len(ret_ev) == 0:
                                        raise RuntimeError('process_event returned empty list')
                                    ret_ev_list = ret_ev
                                else:
                                    ret_ev_list = [ret_ev]

                                log_file.write(" ; DONE ({:.4f}) ; SAVING".format(time.time() - ev.timestamp))
                            except safe_termination.SafelyTerminated:
                                raise
                            except Exception as e:
                                log_file.write("; FAILED ({})".format(e.__class__.__name__))
                                if raise_on_processing_error:
                                    raise
                                traceback.print_exc()
                            log_file.flush()

                        ########################################################################################

                            strong_check_pass = True
                            if not no_overwrite__strong_check:
                                for t_ret_ev in ret_ev_list:
                                    # THIS MIGHT NOT BE IDEAL, SINGLE STRONG CHECK FAILURE SKIPS ALL SUB-EVENTS
                                    if not output_storage_provider.check_event_exists_strong(t_ret_ev):
                                        strong_check_pass = False
                                        break

                            if strong_check_pass:
                                if not dry_run:
                                    msg_ext_format = '' if len(ret_ev_list) == 0 else ' ({:d})'
                                    for t_rev_ev_i, t_ret_ev in enumerate(ret_ev_list):
                                        save_result = output_storage_provider.save_event(t_ret_ev, update)

                                        if not save_result:
                                            log_file.write(" ; FAILED" + msg_ext_format.format(t_rev_ev_i))
                                        elif save_result=='update':
                                            log_file.write(" ; UPDATED" + msg_ext_format.format(t_rev_ev_i))
                                        else:
                                            log_file.write(" ; SAVED" + msg_ext_format.format(t_rev_ev_i))
                                else:
                                    log_file.write(" ; DRY RUN - NOT SAVED")
                            else:
                                log_file.write(" ; EVENT ALREADY EXISTS (STRONG CHECK)")

                        else:
                            log_file.write(' ; '+not_run_reason)

                        if lockfile_dir is not None and processing_lock:
                            processing_sync.release_lock(processing_lock)

                        log_file.write(" ; EVENT FINISHED ({:.4f}) \n".format(time.time() - ev.timestamp))
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
                raise safe_termination.SafelyTerminated('In gtu_pdm_data iteration', __file__)

def _format_config_dump_pathname(pathname_format, source_file_acquisition, source_file_trigger, event_processing, config_info):
    proc_params_dict = config_info.get_dict_of_str()
    if source_file_trigger is not None:
        source_files_acq_trigger_common = os.path.commonprefix((source_file_acquisition, source_file_trigger))
        source_file_trigger_base = source_file_trigger[len(source_files_acq_trigger_common):]
    else:
        source_file_trigger_base = '_'

    if source_file_trigger_base[0] == '/':
        source_file_trigger_base = source_file_trigger_base[1:]
    source_files_acq_trigger = os.path.join(source_file_acquisition+"/",source_file_trigger_base)

    return pathname_format.format(
        program_version=event_processing.program_version,
        source_files_combined=source_files_acq_trigger,
        source_file_acquisition=source_file_acquisition,
        source_file_trigger=source_file_trigger,
        **proc_params_dict)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
