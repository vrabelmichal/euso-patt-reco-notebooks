from _ast import Gt

import numpy as np
import base_classes
import enum

# import processing_config

trigger_event_string_column_order = \
    ["source_file_acquisition", "source_file_trigger",
     "global_gtu", "packet_id", "gtu_in_packet", "num_gtu",
     "num_triggered_pixels", "num_triggered_pixel_groups", "max_triggered_pixel_group_size",
     "triggered_pixels_x_y_hough_transform__max_peak_line_rot",
     "triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_x",
     "triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_y",
     "triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_x",
     "triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_y",
     "triggers_x_y_neighbourhood_size",
     "triggers_x_y_neighbourhood_area",  # todo wrong
     "triggers_x_y_neighbourhood_width",  # todo wrong
     "triggers_x_y_neighbourhood_height",  # todo wrong
     "hough_transform_x_y__num_clusters_above_thr",
     "hough_transform_x_y__max_cluster_size",
     "hough_transform_x_y__avg_cluster_size",
     "hough_transform_x_y__max_cluster_height",
     "hough_transform_x_y__avg_cluster_height",
     "hough_transform_x_y__max_cluster_width",
     "hough_transform_x_y__avg_cluster_width",
     "hough_transform_x_y__max_cluster_area",
     "hough_transform_x_y__avg_cluster_area",
     "hough_transform_x_y__max_cluster_counts_sum",
     "hough_transform_x_y__avg_cluster_counts_sum",
     "hough_transform_x_y__max_peak_line_rot",
     "hough_transform_x_y__max_peak_line_coord_0_x",
     "hough_transform_x_y__max_peak_line_coord_0_y",
     "hough_transform_x_y__max_peak_line_coord_1_x",
     "hough_transform_x_y__max_peak_line_coord_1_y",
     # "triggers_x_gtu_neighbourhood_size",
     # "triggers_x_gtu_neighbourhood_dimensions",
     # "hough_transform_x_gtu__num_clusters_above_thr",
     # "hough_transform_x_gtu__max_cluster_size",
     # "hough_transform_x_gtu__avg_cluster_size",
     # "hough_transform_x_gtu__max_cluster_dimensions",
     # "hough_transform_x_gtu__avg_cluster_dimensions",
     # "hough_transform_x_gtu__max_cluster_counts_sum",
     # "hough_transform_x_gtu__avg_cluster_counts_sum",
     # "hough_transform_x_gtu__max_peak_line_rot",
     # "hough_transform_x_gtu__max_peak_line_coord_0_x",
     # "hough_transform_x_gtu__max_peak_line_coord_0_y",
     # "hough_transform_x_gtu__max_peak_line_coord_1_x",
     # "hough_transform_x_gtu__max_peak_line_coord_1_y",
     # "triggers_y_gtu_neighbourhood_size",
     # "triggers_y_gtu_neighbourhood_dimensions",
     # "hough_transform_y_gtu__num_clusters_above_thr",
     # "hough_transform_y_gtu__max_cluster_size",
     # "hough_transform_y_gtu__avg_cluster_size",
     # "hough_transform_y_gtu__max_cluster_dimensions",
     # "hough_transform_y_gtu__avg_cluster_dimensions",
     # "hough_transform_y_gtu__max_cluster_counts_sum",
     # "hough_transform_y_gtu__avg_cluster_counts_sum",
     # "hough_transform_y_gtu__max_peak_line_rot",
     # "hough_transform_y_gtu__max_peak_line_coord_0_x",
     # "hough_transform_y_gtu__max_speak_line_coord_0_y",
     # "hough_transform_y_gtu__max_peak_line_coord_1_x",
     # "hough_transform_y_gtu__max_peak_line_coord_1_y"
     ]

trigger_event_str_prop_separator = "\n"


class TriggerEventAnalysisRecordV2(base_classes.BaseEventAnalysisRecord):
    extra_attr_method_mapping = {}

    #############################################
    # Should be provided before processing

    source_file_acquisition = ""
    source_file_trigger = ""

    exp_tree = None

    global_gtu = -1
    packet_id = -1
    gtu_in_packet = -1

    gtu_data = None  # list of GtuPdmData objects

    ###########################################

    program_version = 0.0

    ###########################################

    triggered_pixels_coords = None
    # num

    ###########################################

    x_y_projection = None
    x_gtu_projection = None
    y_gtu_projection = None

    ##########################################

    triggered_pixels_x_y_integrated_sum3x3 = None # ndarray
    # sum
    # avg
    # normalized sum

    triggered_pixels_x_y_groups = None
    # num
    # max size
    # avg size

    triggered_pixels_x_y_groups_integrated_sum3x3 = None
    # sum
    # avg
    # normalized sum
    # normalized avg


    triggered_pixels_x_y_neighbourhood = None  # image
    # size
    # sum
    # normalized sum
    # avg
    # normalized avg

    triggered_pixels_x_y_neighbourhood_dimensions = None  # image
    # width
    # height
    # area
    # size


    triggered_pixels_x_y_hough_transform = None

    triggered_pixels_x_y_hough_transform__clusters_above_thr1 = None
    triggered_pixels_x_y_hough_transform__clusters_above_thr2 = None
    # size !!! TODO more important than area

    triggered_pixels_x_y_hough_transform__clusters_above_thr1_dimensions = None
    triggered_pixels_x_y_hough_transform__clusters_above_thr2_dimensions = None

    # todo might be not necessary
    triggered_pixels_x_y_hough_transform__clusters_above_thr1_counts_sums = None
    triggered_pixels_x_y_hough_transform__clusters_above_thr2_counts_sums = None

    triggered_pixels_x_y_hough_transform__clusters_above_thr1_counts_sums_normalized = None
    triggered_pixels_x_y_hough_transform__clusters_above_thr2_counts_sums_normalized = None
    # # # #


    # TODO fix filpped of hough space before setting the value
    triggered_pixels_x_y_hough_transform__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_x_y_hough_transform__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_x_y_hough_transform__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space
    # dynamic
    # triggered_pixels_x_y_hough_transform__max_peak_line_rot  = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_x = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_y = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_x = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_y = -1 # peak determined only from the maximal point of the hough space

    triggered_pixels_x_y_hough_transform__peak_thr1_rho = -1
    triggered_pixels_x_y_hough_transform__peak_thr1_phi = -1
    triggered_pixels_x_y_hough_transform__peak_thr1_line_coords = None

    triggered_pixels_x_y_hough_transform__peak_thr2_rho = -1
    triggered_pixels_x_y_hough_transform__peak_thr2_phi = -1
    triggered_pixels_x_y_hough_transform__peak_thr2_line_coords = None



    hough_transform_x_y = None

    hough_transform_x_y__clusters_above_thr1 = None
    hough_transform_x_y__clusters_above_thr2 = None
    hough_transform_x_y__clusters_above_thr3 = None
    hough_transform_x_y__clusters_above_thr4 = None
    hough_transform_x_y__clusters_above_thr5 = None
    # size
    # dynamic
    # hough_transform_x_y__max_cluster_size = -1
    # hough_transform_x_y__avg_cluster_size = -1

    hough_transform_x_y__clusters_above_thr1_dimensions = None
    hough_transform_x_y__clusters_above_thr2_dimensions = None
    hough_transform_x_y__clusters_above_thr3_dimensions = None
    hough_transform_x_y__clusters_above_thr4_dimensions = None
    hough_transform_x_y__clusters_above_thr5_dimensions = None
    # dynamic from cluster_dimensions
    # hough_transform_x_y__max_cluster_dimensions = -1
    # hough_transform_x_y__avg_cluster_dimensions = -1


    hough_transform_x_y__clusters_above_thr1_counts_sums = None
    hough_transform_x_y__clusters_above_thr2_counts_sums = None
    hough_transform_x_y__clusters_above_thr3_counts_sums = None
    hough_transform_x_y__clusters_above_thr4_counts_sums = None
    hough_transform_x_y__clusters_above_thr5_counts_sums = None

    hough_transform_x_y__clusters_above_thr1_counts_sums_normalized = None
    hough_transform_x_y__clusters_above_thr2_counts_sums_normalized = None
    hough_transform_x_y__clusters_above_thr3_counts_sums_normalized = None
    hough_transform_x_y__clusters_above_thr4_counts_sums_normalized = None
    hough_transform_x_y__clusters_above_thr5_counts_sums_normalized = None

    hough_transform_x_y__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_y__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_y__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space
    # dynamic
    # hough_transform_x_y__max_peak_line_rot  = -1 # peak determined only from the maximal point of the hough space

    hough_transform_x_y__peak_thr1_rho = -1  
    hough_transform_x_y__peak_thr1_phi = -1  
    hough_transform_x_y__peak_thr1_line_coords = None  
    # dynamic
    # hough_transform_x_y__peak_thr1_line_rot  = -1 

    hough_transform_x_y__peak_thr2_rho = -1  
    hough_transform_x_y__peak_thr2_phi = -1  
    hough_transform_x_y__peak_thr2_line_coords = None  
    # dynamic
    # hough_transform_x_y__peak_thr2_line_rot  = -1 

    hough_transform_x_y__peak_thr3_rho = -1  
    hough_transform_x_y__peak_thr3_phi = -1  
    hough_transform_x_y__peak_thr3_line_coords = None  
    # dynamic
    # hough_transform_x_y__peak_thr3_line_rot  = -1 

    hough_transform_x_y__peak_thr4_rho = -1  
    hough_transform_x_y__peak_thr4_phi = -1  
    hough_transform_x_y__peak_thr4_line_coords = None  
    # dynamic
    # hough_transform_x_y__peak_thr4_line_rot  = -1 

    hough_transform_x_y__peak_thr5_rho = -1  
    hough_transform_x_y__peak_thr5_phi = -1  
    hough_transform_x_y__peak_thr5_line_coords = None  
    # dynamic
    # hough_transform_x_y__peak_thr5_line_rot  = -1 
    


    ##########################################

    triggered_pixels_x_gtu_groups = None
    # num
    # max size
    # avg size
    # sum
    # avg
    # normalized sum
    # normalized avg

    triggered_pixels_x_gtu_neighbourhood = None  # image
    # size
    # sum
    # normalized sum
    # avg
    # normalized avg

    triggered_pixels_x_gtu_neighbourhood_dimensions = None  # image
    # max area
    # min area
    # max width
    # max height
    # width of max area
    # height of max area
    # width of min area
    # height of min area


    triggered_pixels_x_gtu_hough_transform = None

    triggered_pixels_x_gtu_hough_transform__clusters_above_thr1 = None
    triggered_pixels_x_gtu_hough_transform__clusters_above_thr2 = None
    # size !!! TODO more important than area

    triggered_pixels_x_gtu_hough_transform__clusters_above_thr1_dimensions = None
    triggered_pixels_x_gtu_hough_transform__clusters_above_thr2_dimensions = None

    triggered_pixels_x_gtu_hough_transform__clusters_above_thr1_counts_sums = None
    triggered_pixels_x_gtu_hough_transform__clusters_above_thr2_counts_sums = None

    triggered_pixels_x_gtu_hough_transform__clusters_above_thr1_counts_sums_normalized = None
    triggered_pixels_x_gtu_hough_transform__clusters_above_thr2_counts_sums_normalized = None


    # TODO fix filpped of hough space before setting the value
    triggered_pixels_x_gtu_hough_transform__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_x_gtu_hough_transform__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_x_gtu_hough_transform__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space
    # dynamic
    # triggered_pixels_x_gtu_hough_transform__max_peak_line_rot  = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_gtu_hough_transform__max_peak_line_coord_0_x = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_gtu_hough_transform__max_peak_line_coord_0_y = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_gtu_hough_transform__max_peak_line_coord_1_x = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_gtu_hough_transform__max_peak_line_coord_1_y = -1 # peak determined only from the maximal point of the hough space

    triggered_pixels_x_gtu_hough_transform__peak_thr1_rho = -1
    triggered_pixels_x_gtu_hough_transform__peak_thr1_phi = -1
    triggered_pixels_x_gtu_hough_transform__peak_thr1_line_coords = None

    triggered_pixels_x_gtu_hough_transform__peak_thr2_rho = -1
    triggered_pixels_x_gtu_hough_transform__peak_thr2_phi = -1
    triggered_pixels_x_gtu_hough_transform__peak_thr2_line_coords = None



    hough_transform_x_gtu = None

    hough_transform_x_gtu__clusters_above_thr1 = None
    hough_transform_x_gtu__clusters_above_thr2 = None
    hough_transform_x_gtu__clusters_above_thr3 = None
    hough_transform_x_gtu__clusters_above_thr4 = None
    hough_transform_x_gtu__clusters_above_thr5 = None
    # size
    # dynamic
    # hough_transform_x_gtu__max_cluster_size = -1
    # hough_transform_x_gtu__avg_cluster_size = -1

    hough_transform_x_gtu__clusters_above_thr1_dimensions = None
    hough_transform_x_gtu__clusters_above_thr2_dimensions = None
    hough_transform_x_gtu__clusters_above_thr3_dimensions = None
    hough_transform_x_gtu__clusters_above_thr4_dimensions = None
    hough_transform_x_gtu__clusters_above_thr5_dimensions = None
    # dynamic from cluster_dimensions
    # hough_transform_x_gtu__max_cluster_dimensions = -1
    # hough_transform_x_gtu__avg_cluster_dimensions = -1


    hough_transform_x_gtu__clusters_above_thr1_counts_sums = None
    hough_transform_x_gtu__clusters_above_thr2_counts_sums = None
    hough_transform_x_gtu__clusters_above_thr3_counts_sums = None
    hough_transform_x_gtu__clusters_above_thr4_counts_sums = None
    hough_transform_x_gtu__clusters_above_thr5_counts_sums = None

    hough_transform_x_gtu__clusters_above_thr1_counts_sums_normalized = None
    hough_transform_x_gtu__clusters_above_thr2_counts_sums_normalized = None
    hough_transform_x_gtu__clusters_above_thr3_counts_sums_normalized = None
    hough_transform_x_gtu__clusters_above_thr4_counts_sums_normalized = None
    hough_transform_x_gtu__clusters_above_thr5_counts_sums_normalized = None

    hough_transform_x_gtu__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_gtu__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_gtu__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space
    # dynamic
    # hough_transform_x_gtu__max_peak_line_rot  = -1 # peak determined only from the maximal point of the hough space

    hough_transform_x_gtu__peak_thr1_rho = -1  
    hough_transform_x_gtu__peak_thr1_phi = -1  
    hough_transform_x_gtu__peak_thr1_line_coords = None  
    # dynamic
    # hough_transform_x_gtu__peak_thr1_line_rot  = -1 

    hough_transform_x_gtu__peak_thr2_rho = -1  
    hough_transform_x_gtu__peak_thr2_phi = -1  
    hough_transform_x_gtu__peak_thr2_line_coords = None  
    # dynamic
    # hough_transform_x_gtu__peak_thr2_line_rot  = -1 

    hough_transform_x_gtu__peak_thr3_rho = -1  
    hough_transform_x_gtu__peak_thr3_phi = -1  
    hough_transform_x_gtu__peak_thr3_line_coords = None  
    # dynamic
    # hough_transform_x_gtu__peak_thr3_line_rot  = -1 

    hough_transform_x_gtu__peak_thr4_rho = -1  
    hough_transform_x_gtu__peak_thr4_phi = -1  
    hough_transform_x_gtu__peak_thr4_line_coords = None  
    # dynamic
    # hough_transform_x_gtu__peak_thr4_line_rot  = -1 

    hough_transform_x_gtu__peak_thr5_rho = -1  
    hough_transform_x_gtu__peak_thr5_phi = -1  
    hough_transform_x_gtu__peak_thr5_line_coords = None  
    # dynamic
    # hough_transform_x_gtu__peak_thr5_line_rot  = -1 
    

    ##########################################

    triggered_pixels_y_gtu_groups = None
    # num
    # max size
    # avg size
    # sum
    # avg
    # normalized sum
    # normalized avg

    triggered_pixels_y_gtu_neighbourhood = None  # image
    # size
    # sum
    # normalized sum
    # avg
    # normalized avg

    triggered_pixels_y_gtu_neighbourhood_dimensions = None  # image
    # max area
    # min area
    # max width
    # max height
    # width of max area
    # height of max area
    # width of min area
    # height of min area


    triggered_pixels_y_gtu_hough_transform = None

    triggered_pixels_y_gtu_hough_transform__clusters_above_thr1 = None
    triggered_pixels_y_gtu_hough_transform__clusters_above_thr2 = None
    # size !!! TODO more important than area

    triggered_pixels_y_gtu_hough_transform__clusters_above_thr1_dimensions = None
    triggered_pixels_y_gtu_hough_transform__clusters_above_thr2_dimensions = None

    triggered_pixels_y_gtu_hough_transform__clusters_above_thr1_counts_sums = None
    triggered_pixels_y_gtu_hough_transform__clusters_above_thr2_counts_sums = None

    triggered_pixels_y_gtu_hough_transform__clusters_above_thr1_counts_sums_normalized = None
    triggered_pixels_y_gtu_hough_transform__clusters_above_thr2_counts_sums_normalized = None


    # TODO fix filpped of hough space before setting the value
    triggered_pixels_y_gtu_hough_transform__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_y_gtu_hough_transform__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_y_gtu_hough_transform__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space
    # dynamic
    # triggered_pixels_y_gtu_hough_transform__max_peak_line_rot  = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_y_gtu_hough_transform__max_peak_line_coord_0_x = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_y_gtu_hough_transform__max_peak_line_coord_0_y = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_y_gtu_hough_transform__max_peak_line_coord_1_x = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_y_gtu_hough_transform__max_peak_line_coord_1_y = -1 # peak determined only from the maximal point of the hough space

    triggered_pixels_y_gtu_hough_transform__peak_thr1_rho = -1
    triggered_pixels_y_gtu_hough_transform__peak_thr1_phi = -1
    triggered_pixels_y_gtu_hough_transform__peak_thr1_line_coords = None

    triggered_pixels_y_gtu_hough_transform__peak_thr2_rho = -1
    triggered_pixels_y_gtu_hough_transform__peak_thr2_phi = -1
    triggered_pixels_y_gtu_hough_transform__peak_thr2_line_coords = None



    hough_transform_y_gtu = None

    hough_transform_y_gtu__clusters_above_thr1 = None
    hough_transform_y_gtu__clusters_above_thr2 = None
    hough_transform_y_gtu__clusters_above_thr3 = None
    hough_transform_y_gtu__clusters_above_thr4 = None
    hough_transform_y_gtu__clusters_above_thr5 = None
    # size
    # dynamic
    # hough_transform_y_gtu__max_cluster_size = -1
    # hough_transform_y_gtu__avg_cluster_size = -1

    hough_transform_y_gtu__clusters_above_thr1_dimensions = None
    hough_transform_y_gtu__clusters_above_thr2_dimensions = None
    hough_transform_y_gtu__clusters_above_thr3_dimensions = None
    hough_transform_y_gtu__clusters_above_thr4_dimensions = None
    hough_transform_y_gtu__clusters_above_thr5_dimensions = None
    # dynamic from cluster_dimensions
    # hough_transform_y_gtu__max_cluster_dimensions = -1
    # hough_transform_y_gtu__avg_cluster_dimensions = -1


    hough_transform_y_gtu__clusters_above_thr1_counts_sums = None
    hough_transform_y_gtu__clusters_above_thr2_counts_sums = None
    hough_transform_y_gtu__clusters_above_thr3_counts_sums = None
    hough_transform_y_gtu__clusters_above_thr4_counts_sums = None
    hough_transform_y_gtu__clusters_above_thr5_counts_sums = None

    hough_transform_y_gtu__clusters_above_thr1_counts_sums_normalized = None
    hough_transform_y_gtu__clusters_above_thr2_counts_sums_normalized = None
    hough_transform_y_gtu__clusters_above_thr3_counts_sums_normalized = None
    hough_transform_y_gtu__clusters_above_thr4_counts_sums_normalized = None
    hough_transform_y_gtu__clusters_above_thr5_counts_sums_normalized = None

    hough_transform_y_gtu__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    hough_transform_y_gtu__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    hough_transform_y_gtu__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space
    # dynamic
    # hough_transform_y_gtu__max_peak_line_rot  = -1 # peak determined only from the maximal point of the hough space

    hough_transform_y_gtu__peak_thr1_rho = -1  
    hough_transform_y_gtu__peak_thr1_phi = -1  
    hough_transform_y_gtu__peak_thr1_line_coords = None  
    # dynamic
    # hough_transform_y_gtu__peak_thr1_line_rot  = -1 

    hough_transform_y_gtu__peak_thr2_rho = -1  
    hough_transform_y_gtu__peak_thr2_phi = -1  
    hough_transform_y_gtu__peak_thr2_line_coords = None  
    # dynamic
    # hough_transform_y_gtu__peak_thr2_line_rot  = -1 

    hough_transform_y_gtu__peak_thr3_rho = -1  
    hough_transform_y_gtu__peak_thr3_phi = -1  
    hough_transform_y_gtu__peak_thr3_line_coords = None  
    # dynamic
    # hough_transform_y_gtu__peak_thr3_line_rot  = -1 

    hough_transform_y_gtu__peak_thr4_rho = -1  
    hough_transform_y_gtu__peak_thr4_phi = -1  
    hough_transform_y_gtu__peak_thr4_line_coords = None  
    # dynamic
    # hough_transform_y_gtu__peak_thr4_line_rot  = -1 

    hough_transform_y_gtu__peak_thr5_rho = -1  
    hough_transform_y_gtu__peak_thr5_phi = -1  
    hough_transform_y_gtu__peak_thr5_line_coords = None  
    # dynamic
    # hough_transform_y_gtu__peak_thr5_line_rot  = -1 
    
    class _GetterType(enum.Enum):
        LEN=1
        NP_COUNT_NONZERO=2
        NP_SUM=3
        NP_SUM_NORMALIZED=4
        NP_MAX=13
        MAX_LEN=14
        SUM_LEN=15
        AVG_LEN=16
        SUM_NP_SUM=6
        MAX_NP_SUM=7
        AVG_NP_SUM_USING_LIST_LEN=8
        INDEX_0=9
        INDEX_1=10
        MULT_INDEX_0_INDEX_1=11
        AVG_USING_NONZERO=12
        INDEX_0_0=17
        INDEX_0_1=18
        INDEX_1_0=19
        INDEX_1_1=20
        PLUS_PI_OVER_2=21


    @classmethod
    def gen_getter(cls, type, attr, **kwargs):
        Gtrt = TriggerEventAnalysisRecordV2._GetterType
        lk = None
        if 'list' in kwargs:
            lk = kwargs[list]
        if
        if type == Gtrt.LEN:
            return lambda o: len(getattr(o, attr)) if getattr(o, attr) is not None else -1
        if type == Gtrt.NP_COUNT_NONZERO:
            return lambda o: np.count_nonzero(getattr(o,attr)) if getattr(o,attr) is not None else -1
        if type == Gtrt.NP_SUM:
            return lambda o: np.sum(getattr(o,attr)) if getattr(o,attr) is not None else -1
        if type == Gtrt.NP_MAX:
            return lambda o: np.max(getattr(o,attr)) if getattr(o,attr) is not None else -1
        if type == Gtrt.NP_SUM_NORMALIZED:
            return lambda o: np.sum(getattr(o,attr)/np.max(getattr(o,attr))) if getattr(o,attr) is not None else -1 # and 'list' in kwargs and kwargs['list'] is not None else -1
        if type == Gtrt.MAX_LEN:
            return lambda o: np.max([len(g) for g in getattr(o,attr)]) if getattr(o,attr) is not None else -1
        if type == Gtrt.SUM_LEN:
            return lambda o: np.max([len(g) for g in getattr(o,attr)]) if getattr(o,attr) is not None else -1
        if type == Gtrt.AVG_LEN:
            return lambda o: np.sum([len(g) for g in getattr(o,attr)]) / len(getattr(o,attr)) if getattr(o,attr) is not None and len(getattr(o,attr)) > 0 else -1. # np.avg ?
        if type == Gtrt.SUM_NP_SUM:
            return lambda o: sum([np.sum(g) for g in getattr(o,attr)]) if getattr(o,attr) is not None else -1
        if type == Gtrt.MAX_NP_SUM:
            return lambda o: max([np.sum(g) for g in getattr(o,attr)]) if getattr(o,attr) is not None else -1
        if type == Gtrt.AVG_NP_SUM_USING_LIST_LEN:
            return lambda o: sum([np.sum(g) for g in getattr(o,attr)])/len(getattr(o, lk)) if getattr(o,attr) is not None and lk is not None and getattr(o,lk) is not None and len(getattr(o,lk)) > 0 else -1,
        if type == Gtrt.INDEX_0:
            return lambda o: getattr(o,attr)[0] if getattr(o,attr) is not None and len(getattr(o,attr)) > 1 else -1
        if type == Gtrt.INDEX_1:
            return lambda o: getattr(o,attr)[1] if getattr(o,attr) is not None and len(getattr(o,attr)) > 1 else -1
        if type == Gtrt.MULT_INDEX_0_INDEX_1:
            return lambda o: getattr(o,attr)[1] * getattr(o,attr)[0] if getattr(o,attr) is not None and len(getattr(o, attr)) > 1 else -1
        if type == Gtrt.AVG_USING_NONZERO:
            lambda o: np.sum(getattr(o,attr) / np.max(getattr(o,attr))) if getattr(o,attr) is not None else -1
        if type == Gtrt.INDEX_0_0:
            return lambda o: getattr(o,attr)[0,0] if getattr(o,attr) is not None and getattr(o,attr).shape[0] > 0 and getattr(o,attr).shape[1] > 0 else -1
        if type == Gtrt.INDEX_0_1:
            return lambda o: getattr(o,attr)[0,1] if getattr(o,attr) is not None and getattr(o,attr).shape[0] > 0 and getattr(o,attr).shape[1] > 1 else -1
        if type == Gtrt.INDEX_0_1:
            return lambda o: getattr(o,attr)[1,0] if getattr(o,attr) is not None and getattr(o,attr).shape[0] > 1 and getattr(o,attr).shape[1] > 0 else -1
        if type == Gtrt.INDEX_1_1:
            return lambda o: getattr(o,attr)[1,1] if getattr(o,attr) is not None and getattr(o,attr).shape[1] > 1 and getattr(o,attr).shape[1] > 0 else -1
        if type == Gtrt.PLUS_PI_OVER_2:
            return lambda o: getattr(o,attr) + np.pi/2 if getattr(o,attr) is not None else -1


    def __init__(self):
        self.gtu_data = []
        self.triggered_pixels_coords = []
        self.triggered_pixel_groups = []
        self.triggered_pixels_x_y_hough_transform = []
        self.triggered_pixels_x_y_hough_transform__max_peak_line_coords = [(None, None), (None, None)]
        self.triggered_pixels_x_y_neighbourhood = []
        self.triggered_pixels_x_y_neighbourhood_dimensions = []
        self.hough_transform_x_y = []
        self.hough_transform_x_y__clusters_above_thr = []
        self.hough_transform_x_y__cluster_dimensions = []
        self.hough_transform_x_y__cluster_counts_sums = []
        self.hough_transform_x_y__max_peak_line_coords = [(None, None), (None, None)]
        self.hough_transform_x_y__thr_peak_line_coords = [(None, None), (None, None)]
        self.triggers_x_gtu_neighbourhood = []
        self.triggers_x_gtu_neighbourhood_size = []
        self.triggers_x_gtu_neighbourhood_dimensions = []
        self.hough_transform_x_gtu = []
        self.hough_transform_x_gtu__clusters_above_thr = []
        self.hough_transform_x_gtu__num_clusters_above_thr = []
        self.triggers_x_gtu_neighbourhood = []
        self.triggers_x_gtu_neighbourhood_size = []
        self.triggers_x_gtu_neighbourhood_dimensions = []
        self.hough_transform_y_gtu_ = []
        self.hough_transform_y_gtu__clusters_above_thr = []
        self.hough_transform_y_gtu__num_clusters_above_thr = []

        if not self.__class__.extra_attr_method_mapping:
            # 'extra_attr_method_mapping' not in self.__class__.__dict__ or

            Gtrt = TriggerEventAnalysisRecordV2._GetterType
            gg = self.gen_getter

            getters_def = {
                'num_gtu': gg(Gtrt.LEN, 'gtu_data'),
                'triggered_pixels_num': gg(Gtrt.LEN, 'triggered_pixels_coords'),
            }

            for dim in ('x_y', 'x_gtu', 'y_gtu'):
                dim_getter_def = {
                    '{d}_active_pixels': (Gtrt.NP_COUNT_NONZERO, '{d}_projection'),
                    'triggered_pixels_{d}_sum3x3_sum': (Gtrt.NP_COUNT_NONZERO, 'triggered_pixels_{d}_integrated_sum3x3'),
                    'triggered_pixels_{d}_sum3x3_norm_sum': (Gtrt.NP_SUM_NORMALIZED ,'triggered_pixels_{d}_integrated_sum3x3'),
                    'triggered_pixels_{d}_sum3x3_avg': (Gtrt.AVG_NP_SUM_USING_LIST_LEN , 'triggered_pixels_x_y_integrated_sum3x3', {'list': 'triggered_pixels_coords'}),
                    'triggered_pixels_{d}_groups_num': (Gtrt.LEN ,'triggered_pixel_groups'),
                    'triggered_pixels_{d}_groups_max_size': (Gtrt.MAX_LEN ,'triggered_pixel_groups'),
                    'triggered_pixels_{d}_groups_avg_size': (Gtrt.AVG_LEN ,'triggered_pixel_groups'),
                    'triggered_pixels_{d}_groups_sum_sum_sum3x3': (Gtrt.SUM_NP_SUM, 'triggered_pixels_{d}_groups_integrated_sum3x3'),
                    'triggered_pixels_{d}_groups_max_sum_sum3x3': (Gtrt.MAX_NP_SUM,'triggered_pixels_{d}_groups_integrated_sum3x3'),
                    'triggered_pixels_{d}_groups_avg_sum_sum3x3': (Gtrt.AVG_NP_SUM_USING_LIST_LEN, 'triggered_pixel_groups', {'list': 'triggered_pixels_{d}_groups'}),
                    'triggers_{d}_neighbourhood_size': (Gtrt.NP_COUNT_NONZERO, 'triggers_{d}_neighbourhood'),
                    'triggers_{d}_neighbourhood_width': (Gtrt.INDEX_1, 'triggers_{d}_neighbourhood_dimensions'),
                    'triggers_{d}_neighbourhood_height': (Gtrt.INDEX_0, 'triggers_{d}_neighbourhood_dimensions'),
                    'triggers_{d}_neighbourhood_area': (Gtrt.MULT_INDEX_0_INDEX_1, 'triggers_{d}_neighbourhood_dimensions[1'),
                    'triggers_{d}_neighbourhood_counts_sum': (Gtrt.NP_SUM, 'triggers_{d}_neighbourhood'),
                    'triggers_{d}_neighbourhood_counts_avg': (Gtrt.AVG_USING_NONZERO, 'triggers_{d}_neighbourhood'),
                    'triggers_{d}_neighbourhood_counts_norm_sum': (Gtrt.NP_SUM_NORMALIZED, 'triggers_{d}_neighbourhood'),
                    'hough_transform_{d}__max_cluster_width': (Gtrt., 'hough_transform_{d}__cluster_dimensions'),
                    'hough_transform_{d}__avg_cluster_width': (Gtrt., 'hough_transform_{d}__cluster_dimensions'),
                    'hough_transform_{d}__max_cluster_height': (Gtrt., 'hough_transform_{d}__cluster_dimensions'),
                    'hough_transform_{d}__avg_cluster_height': (Gtrt., 'hough_transform_{d}__cluster_dimensions'),
                    'hough_transform_{d}__max_cluster_area': (Gtrt., 'hough_transform_{d}__cluster_dimensions'),
                    'hough_transform_{d}__avg_cluster_area': (Gtrt., 'hough_transform_{d}__cluster_dimensions'),
                    'hough_transform_{d}__max_cluster_size': (Gtrt., 'hough_transform_{d}__clusters_above_thr'),
                    'hough_transform_{d}__avg_cluster_size': (Gtrt., 'hough_transform_{d}__clusters_above_thr'),
                    'hough_transform_{d}__max_cluster_counts_sum': (Gtrt., 'hough_transform_{d}__cluster_counts_sums'),
                    'hough_transform_{d}__avg_cluster_counts_sum': (Gtrt., 'hough_transform_{d}__cluster_counts_sums'),
                    'hough_transform_{d}__max_peak_line_rot': (Gtrt., 'hough_transform_{d}__max_peak_phi'),
                    'hough_transform_{d}__max_peak_line_coord_0_x': (Gtrt., 'hough_transform_{d}__max_peak_line_coords[0'),
                    'hough_transform_{d}__max_peak_line_coord_0_y': (Gtrt., 'hough_transform_{d}__max_peak_line_coords[0'),
                    'hough_transform_{d}__max_peak_line_coord_1_x': (Gtrt., 'hough_transform_{d}__max_peak_line_coords[1'),
                    'hough_transform_{d}__max_peak_line_coord_1_y': (Gtrt., 'hough_transform_{d}__max_peak_line_coords[1'),
                    'hough_transform_{d}__thr_peak_line_rot': (Gtrt., 'hough_transform_{d}__thr_peak_phi'),
                    'hough_transform_{d}__thr_peak_line_coord_0_x': (Gtrt., 'hough_transform_{d}__thr_peak_line_coords[0'),
                    'hough_transform_{d}__thr_peak_line_coord_0_y': (Gtrt., 'hough_transform_{d}__thr_peak_line_coords[0'),
                    'hough_transform_{d}__thr_peak_line_coord_1_x': (Gtrt., 'hough_transform_{d}__thr_peak_line_coords[1'),
                    'hough_transform_{d}__thr_peak_line_coord_1_y': (Gtrt., 'hough_transform_{d}__thr_peak_line_coords[1'),
                }




            self.gen_getter(Gtrt.LEN, 'triggered_pixels_num')

            d = {
                'num_gtu': lambda o: len(o.gtu_data),

                'triggered_pixels_num': lambda o: len(o.triggered_pixels_coords),

                'x_y_active_pixels': lambda o: np.count_nonzero(o.x_y_projection) if o.x_y_projection is not None else -1,
                'x_gtu_active_pixels': lambda o: np.count_nonzero(o.x_gtu_projection) if o.x_gtu_projection is not None else -1,
                'y_gtu_active_pixels': lambda o: np.count_nonzero(o.y_gtu_projection) if o.y_gtu_projection is not None else -1,

                'triggered_pixels_x_y_sum3x3_sum': lambda o: np.sum(o.triggered_pixels_x_y_integrated_sum3x3) if o.triggered_pixels_x_y_integrated_sum3x3 is not None else -1,
                'triggered_pixels_x_y_sum3x3_norm_sum': lambda o: np.sum(o.triggered_pixels_x_y_integrated_sum3x3/np.max(o.triggered_pixels_x_y_integrated_sum3x3)) if o.triggered_pixels_x_y_integrated_sum3x3 is not None else -1,
                'triggered_pixels_x_y_sum3x3_avg': lambda o: np.sum(o.triggered_pixels_x_y_integrated_sum3x3)/len(o.num_triggered_pixels) if o.triggered_pixels_x_y_integrated_sum3x3 is not None and o.num_triggered_pixels is not None and len(o.num_triggered_pixels) > 0 else -1,

                'triggered_pixels_x_y_groups_num': lambda o: len(o.triggered_pixel_groups) if o.triggered_pixel_groups is not None else -1,
                'triggered_pixels_x_y_groups_max_size': lambda o: max([len(g) for g in o.triggered_pixel_groups]) if o.triggered_pixel_groups is not None else -1,
                'triggered_pixels_x_y_groups_avg_size': lambda o: sum([len(g) for g in o.triggered_pixel_groups]) / len(o.triggered_pixel_groups) if o.triggered_pixel_groups is not None and len(o.triggered_pixel_groups) > 0 else -1.,

                'triggered_pixels_x_y_groups_sum_sum_sum3x3': lambda o: sum([np.sum(g) for g in o.triggered_pixels_x_y_groups_integrated_sum3x3]) if o.triggered_pixels_x_y_groups_integrated_sum3x3 is not None else -1,
                'triggered_pixels_x_y_groups_max_sum_sum3x3': lambda o: max([np.sum(g) for g in o.triggered_pixels_x_y_groups_integrated_sum3x3]) if o.triggered_pixels_x_y_groups_integrated_sum3x3 is not None else -1,
                'triggered_pixels_x_y_groups_avg_sum_sum3x3': lambda o: sum([np.sum(g) for g in o.triggered_pixels_x_y_groups_integrated_sum3x3])/len(o.triggered_pixel_groups) if o.triggered_pixels_x_y_groups_integrated_sum3x3 is not None and o.triggered_pixel_groups is not None and len(o.triggered_pixel_groups) > 0 else -1,

                'triggers_x_y_neighbourhood_size': lambda o: np.count_nonzero(o.triggers_x_y_neighbourhood) if o.triggers_x_y_neighbourhood is not None else -1,
                # 'max_triggered_pixels_group_sum': lambda o: max([sum(g) for g in o.triggered_pixel_groups]) if len(o.triggered_pixel_groups) > 0 else -1,
                # 'avg_triggered_pixels_group_group_sum': lambda o: sum([sum(g) for g in o.triggered_pixel_groups]) / len(o.triggered_pixel_groups) if len(o.triggered_pixel_groups) > 0 else -1.,
                'triggers_x_y_neighbourhood_width': lambda o: o.triggers_x_y_neighbourhood_dimensions[1] if len(
                    o.triggers_x_y_neighbourhood_dimensions) > 1 else -1,
                'triggers_x_y_neighbourhood_height': lambda o: o.triggers_x_y_neighbourhood_dimensions[0] if len(
                    o.triggers_x_y_neighbourhood_dimensions) > 0 else -1,
                'triggers_x_y_neighbourhood_area': lambda o: o.triggers_x_y_neighbourhood_dimensions[1] *
                                                             o.triggers_x_y_neighbourhood_dimensions[0] if o.triggers_x_y_neighbourhood_dimensions is not None and len(
                    o.triggers_x_y_neighbourhood_dimensions) > 1 else -1,
                'triggers_x_y_neighbourhood_counts_sum': lambda o: np.sum(o.triggers_x_y_neighbourhood),
                'triggers_x_y_neighbourhood_counts_avg': lambda o: np.sum(o.triggers_x_y_neighbourhood)/np.count_nonzero(o.triggers_x_y_neighbourhood) if o.triggers_x_y_neighbourhood is not None else -1,
                'triggers_x_y_neighbourhood_counts_norm_sum': lambda o: np.sum(o.triggers_x_y_neighbourhood/np.max(o.triggers_x_y_neighbourhood)) if o.triggers_x_y_neighbourhood is not None else -1,




                'hough_transform_x_y__max_cluster_width': lambda o: max(
                    [c[1] for c in o.hough_transform_x_y__cluster_dimensions]) if len(
                    o.hough_transform_x_y__cluster_dimensions) > 0 else -1,
                'hough_transform_x_y__avg_cluster_width': lambda o: sum(
                    [c[1] for c in o.hough_transform_x_y__cluster_dimensions]) / len(
                    o.hough_transform_x_y__cluster_dimensions) if len(
                    o.hough_transform_x_y__cluster_dimensions) else -1.,
                'hough_transform_x_y__max_cluster_height': lambda o: max(
                    [c[0] for c in o.hough_transform_x_y__cluster_dimensions]) if len(
                    o.hough_transform_x_y__cluster_dimensions) > 0 else -1,
                'hough_transform_x_y__avg_cluster_height': lambda o: sum(
                    [c[0] for c in o.hough_transform_x_y__cluster_dimensions]) / len(
                    o.hough_transform_x_y__cluster_dimensions) if len(
                    o.hough_transform_x_y__cluster_dimensions) > 0 else -1.,
                'hough_transform_x_y__max_cluster_area': lambda o: max(
                    [c[1] * c[0] for c in o.hough_transform_x_y__cluster_dimensions]) if len(
                    o.hough_transform_x_y__cluster_dimensions) > 0 else -1,
                'hough_transform_x_y__avg_cluster_area': lambda o: sum(
                    [c[1] * c[0] for c in o.hough_transform_x_y__cluster_dimensions]) / len(
                    o.hough_transform_x_y__cluster_dimensions) if len(
                    o.hough_transform_x_y__cluster_dimensions) > 0 else -1.,
                'hough_transform_x_y__max_cluster_size': lambda o: max(
                    [np.count_nonzero(c) for c in o.hough_transform_x_y__clusters_above_thr]) if len(
                    o.hough_transform_x_y__clusters_above_thr) > 0 else -1,
                'hough_transform_x_y__avg_cluster_size': lambda o: sum(
                    [np.count_nonzero(c) for c in o.hough_transform_x_y__clusters_above_thr]) / len(
                    o.hough_transform_x_y__clusters_above_thr) if len(
                    o.hough_transform_x_y__clusters_above_thr) > 0 else -1.,
                'hough_transform_x_y__max_cluster_counts_sum': lambda o: max(
                    [c for c in o.hough_transform_x_y__cluster_counts_sums]) if len(
                    o.hough_transform_x_y__cluster_counts_sums) > 0 else -1,
                'hough_transform_x_y__avg_cluster_counts_sum': lambda o: sum(
                    [c for c in o.hough_transform_x_y__cluster_counts_sums]) / len(
                    o.hough_transform_x_y__cluster_counts_sums) if len(
                    o.hough_transform_x_y__cluster_counts_sums) > 0 else -1.,



                'hough_transform_x_y__max_peak_line_rot': lambda o: o.hough_transform_x_y__max_peak_phi + np.pi / 2,
                'hough_transform_x_y__max_peak_line_coord_0_x': lambda o:
                o.hough_transform_x_y__max_peak_line_coords[0][1],
                'hough_transform_x_y__max_peak_line_coord_0_y': lambda o:
                o.hough_transform_x_y__max_peak_line_coords[0][0],
                'hough_transform_x_y__max_peak_line_coord_1_x': lambda o:
                o.hough_transform_x_y__max_peak_line_coords[1][1],
                'hough_transform_x_y__max_peak_line_coord_1_y': lambda o:
                o.hough_transform_x_y__max_peak_line_coords[1][0],
                'hough_transform_x_y__thr_peak_line_rot': lambda o: o.hough_transform_x_y__thr_peak_phi + np.pi / 2,
                'hough_transform_x_y__thr_peak_line_coord_0_x': lambda o:
                o.hough_transform_x_y__thr_peak_line_coords[0][1],
                'hough_transform_x_y__thr_peak_line_coord_0_y': lambda o:
                o.hough_transform_x_y__thr_peak_line_coords[0][0],
                'hough_transform_x_y__thr_peak_line_coord_1_x': lambda o:
                o.hough_transform_x_y__thr_peak_line_coords[1][1],
                'hough_transform_x_y__thr_peak_line_coord_1_y': lambda o:
                o.hough_transform_x_y__thr_peak_line_coords[1][0],
            }

            # self.__class__.__dict__[''] =
            setattr(self.__class__, "extra_attr_method_mapping", d)


if __name__ == "__main__":
    # execute only if run as a script
    r = TriggerEventAnalysisRecordV1()
    r.triggered_pixels_x_y_hough_transform__max_peak_phi = 34

    print(r.triggered_pixels_x_y_hough_transform__max_peak_line_rot)
