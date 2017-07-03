import numpy as np
import base_classes
import enum
import collections

# import processing_config


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

    triggered_pixels_x_y_coords = None
    triggered_pixels_x_gtu_coords = None
    triggered_pixels_y_gtu_coords = None
    # num

    ###########################################

    x_y_integrated = None
    x_gtu_integrated = None
    y_gtu_integrated = None

    ##########################################

    triggered_pixels_x_y_integrated = None # ndarray
    # sum
    # avg
    # normalized sum

    triggered_pixels_x_y_groups = None
    # num
    # max size
    # avg size

    triggered_pixels_x_y_groups_integrated = None
    # sum
    # avg
    # normalized sum
    # normalized avg


    x_y_neighbourhood = None  # image
    # size
    # sum
    # normalized sum
    # avg
    # normalized avg

    x_y_neighbourhood_dimensions = None  # image
    # width
    # height
    # area
    # size


    triggered_pixels_x_y_hough_transform = None

    triggered_pixels_x_y_hough_transform__clusters_above_thr1 = None # ndarray
    triggered_pixels_x_y_hough_transform__clusters_above_thr2 = None # ndarray
    # size !!! TODO more important than area

    triggered_pixels_x_y_hough_transform__clusters_above_thr1_dimensions = None
    triggered_pixels_x_y_hough_transform__clusters_above_thr2_dimensions = None

    # TODO fix filpped of hough space before setting the value
    triggered_pixels_x_y_hough_transform__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_x_y_hough_transform__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_x_y_hough_transform__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space
    # rot
    # coord_0_x ...

    triggered_pixels_x_y_hough_transform__peak_thr1_rho = -1
    triggered_pixels_x_y_hough_transform__peak_thr1_phi = -1
    triggered_pixels_x_y_hough_transform__peak_thr1_line_coords = None

    triggered_pixels_x_y_hough_transform__peak_thr2_rho = -1
    triggered_pixels_x_y_hough_transform__peak_thr2_phi = -1
    triggered_pixels_x_y_hough_transform__peak_thr2_line_coords = None


    x_y_hough_transform = None

    x_y_hough_transform__clusters_above_thr1 = None
    x_y_hough_transform__clusters_above_thr2 = None
    x_y_hough_transform__clusters_above_thr3 = None
    # x_y_hough_transform__clusters_above_thr4 = None
    # x_y_hough_transform__clusters_above_thr5 = None

    x_y_hough_transform__clusters_above_thr1_dimensions = None
    x_y_hough_transform__clusters_above_thr2_dimensions = None
    x_y_hough_transform__clusters_above_thr3_dimensions = None
    # x_y_hough_transform__clusters_above_thr4_dimensions = None
    # x_y_hough_transform__clusters_above_thr5_dimensions = None

    x_y_hough_transform__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    x_y_hough_transform__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    x_y_hough_transform__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space
    # dynamic
    # x_y_hough_transform__max_peak_line_rot  = -1 # peak determined only from the maximal point of the hough space

    x_y_hough_transform__peak_thr1_rho = -1  
    x_y_hough_transform__peak_thr1_phi = -1  
    x_y_hough_transform__peak_thr1_line_coords = None  
    # rot
    x_y_hough_transform__peak_thr2_rho = -1  
    x_y_hough_transform__peak_thr2_phi = -1  
    x_y_hough_transform__peak_thr2_line_coords = None  
    # rot
    x_y_hough_transform__peak_thr3_rho = -1  
    x_y_hough_transform__peak_thr3_phi = -1  
    x_y_hough_transform__peak_thr3_line_coords = None  
    # rot
    # x_y_hough_transform__peak_thr4_rho = -1
    # x_y_hough_transform__peak_thr4_phi = -1
    # x_y_hough_transform__peak_thr4_line_coords = None
    # # rot
    # x_y_hough_transform__peak_thr5_rho = -1
    # x_y_hough_transform__peak_thr5_phi = -1
    # x_y_hough_transform__peak_thr5_line_coords = None
    # # rot


    ##########################################

    triggered_pixels_x_gtu_groups = None
    # num
    # max size
    # avg size
    # sum
    # avg
    # normalized sum
    # normalized avg

    x_gtu_neighbourhood = None  # image
    # size
    # sum
    # normalized sum
    # avg
    # normalized avg

    x_gtu_neighbourhood_dimensions = None  # image
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


    # TODO fix filpped of hough space before setting the value
    triggered_pixels_x_gtu_hough_transform__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_x_gtu_hough_transform__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_x_gtu_hough_transform__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space

    triggered_pixels_x_gtu_hough_transform__peak_thr1_rho = -1
    triggered_pixels_x_gtu_hough_transform__peak_thr1_phi = -1
    triggered_pixels_x_gtu_hough_transform__peak_thr1_line_coords = None

    triggered_pixels_x_gtu_hough_transform__peak_thr2_rho = -1
    triggered_pixels_x_gtu_hough_transform__peak_thr2_phi = -1
    triggered_pixels_x_gtu_hough_transform__peak_thr2_line_coords = None



    x_gtu_hough_transform = None

    x_gtu_hough_transform__clusters_above_thr1 = None
    x_gtu_hough_transform__clusters_above_thr2 = None
    x_gtu_hough_transform__clusters_above_thr3 = None
    # x_gtu_hough_transform__clusters_above_thr4 = None
    # x_gtu_hough_transform__clusters_above_thr5 = None
    # size

    x_gtu_hough_transform__clusters_above_thr1_dimensions = None
    x_gtu_hough_transform__clusters_above_thr2_dimensions = None
    x_gtu_hough_transform__clusters_above_thr3_dimensions = None
    # x_gtu_hough_transform__clusters_above_thr4_dimensions = None
    # x_gtu_hough_transform__clusters_above_thr5_dimensions = None

    x_gtu_hough_transform__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    x_gtu_hough_transform__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    x_gtu_hough_transform__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space

    x_gtu_hough_transform__peak_thr1_rho = -1  
    x_gtu_hough_transform__peak_thr1_phi = -1  
    x_gtu_hough_transform__peak_thr1_line_coords = None  

    x_gtu_hough_transform__peak_thr2_rho = -1  
    x_gtu_hough_transform__peak_thr2_phi = -1  
    x_gtu_hough_transform__peak_thr2_line_coords = None  

    x_gtu_hough_transform__peak_thr3_rho = -1  
    x_gtu_hough_transform__peak_thr3_phi = -1  
    x_gtu_hough_transform__peak_thr3_line_coords = None  

    # x_gtu_hough_transform__peak_thr4_rho = -1
    # x_gtu_hough_transform__peak_thr4_phi = -1
    # x_gtu_hough_transform__peak_thr4_line_coords = None
    #
    # x_gtu_hough_transform__peak_thr5_rho = -1
    # x_gtu_hough_transform__peak_thr5_phi = -1
    # x_gtu_hough_transform__peak_thr5_line_coords = None


    ##########################################

    triggered_pixels_y_gtu_groups = None
    # num
    # max size
    # avg size
    # sum
    # avg
    # normalized sum
    # normalized avg

    y_gtu_neighbourhood = None  # image
    # size
    # sum
    # normalized sum
    # avg
    # normalized avg

    y_gtu_neighbourhood_dimensions = None  # image
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

    # TODO fix filpped of hough space before setting the value
    triggered_pixels_y_gtu_hough_transform__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_y_gtu_hough_transform__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    triggered_pixels_y_gtu_hough_transform__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space

    triggered_pixels_y_gtu_hough_transform__peak_thr1_rho = -1
    triggered_pixels_y_gtu_hough_transform__peak_thr1_phi = -1
    triggered_pixels_y_gtu_hough_transform__peak_thr1_line_coords = None

    triggered_pixels_y_gtu_hough_transform__peak_thr2_rho = -1
    triggered_pixels_y_gtu_hough_transform__peak_thr2_phi = -1
    triggered_pixels_y_gtu_hough_transform__peak_thr2_line_coords = None



    y_gtu_hough_transform = None

    y_gtu_hough_transform__clusters_above_thr1 = None
    y_gtu_hough_transform__clusters_above_thr2 = None
    y_gtu_hough_transform__clusters_above_thr3 = None
    # y_gtu_hough_transform__clusters_above_thr4 = None
    # y_gtu_hough_transform__clusters_above_thr5 = None

    y_gtu_hough_transform__clusters_above_thr1_dimensions = None
    y_gtu_hough_transform__clusters_above_thr2_dimensions = None
    y_gtu_hough_transform__clusters_above_thr3_dimensions = None
    # y_gtu_hough_transform__clusters_above_thr4_dimensions = None
    # y_gtu_hough_transform__clusters_above_thr5_dimensions = None

    y_gtu_hough_transform__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    y_gtu_hough_transform__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    y_gtu_hough_transform__max_peak_line_coords = None  # peak determined only from the maximal point of the hough space

    y_gtu_hough_transform__peak_thr1_rho = -1  
    y_gtu_hough_transform__peak_thr1_phi = -1  
    y_gtu_hough_transform__peak_thr1_line_coords = None

    y_gtu_hough_transform__peak_thr2_rho = -1  
    y_gtu_hough_transform__peak_thr2_phi = -1  
    y_gtu_hough_transform__peak_thr2_line_coords = None

    y_gtu_hough_transform__peak_thr3_rho = -1  
    y_gtu_hough_transform__peak_thr3_phi = -1  
    y_gtu_hough_transform__peak_thr3_line_coords = None

    # y_gtu_hough_transform__peak_thr4_rho = -1
    # y_gtu_hough_transform__peak_thr4_phi = -1
    # y_gtu_hough_transform__peak_thr4_line_coords = None
    # # dynamic
    # # y_gtu_hough_transform__peak_thr4_line_rot  = -1
    #
    # y_gtu_hough_transform__peak_thr5_rho = -1
    # y_gtu_hough_transform__peak_thr5_phi = -1
    # y_gtu_hough_transform__peak_thr5_line_coords = None
    # # dynamic
    # # y_gtu_hough_transform__peak_thr5_line_rot  = -1
    
    class _GetterType(enum.Enum):
        LEN=1
        NP_COUNT_NONZERO=2
        NP_SUM=3
        NP_SUM_NORMALIZED=4
        NP_MAX=5
        MAX_LEN=6
        SUM_LEN=7
        AVG_LEN=8
        SUM_NP_SUM=9
        MAX_NP_SUM=10
        AVG_NP_SUM_USING_LIST_LEN=11
        AVG_NP_SUM=12
        INDEX_0=13
        INDEX_1=14
        MULT_INDEX_0_INDEX_1=15
        AVG_USING_NONZERO=16
        INDEX_0_0=17
        INDEX_0_1=18
        INDEX_1_0=19
        INDEX_1_1=20
        PLUS_PI_OVER_2=21
        AVG_INDEX_0_USING_LEN=22
        AVG_INDEX_1_USING_LEN=23
        MAX_INDEX_0=24
        MAX_INDEX_1=25
        INDEX_0_OF_MAX_MULT_INDEX_0_1=26
        INDEX_1_OF_MAX_MULT_INDEX_0_1=27
        MAX_MULT_INDEX_0_INDEX_1=28
        AVG_MULT_INDEX_0_INDEX_1=29
        INDEX_0_OF_MAX_LEN=30
        INDEX_1_OF_MAX_LEN=31
        SUM_NP_COUNT_NONZERO=32
        MAX_NP_COUNT_NONZERO=33
        AVG_NP_COUNT_NONZERO=34
        INDEX_0_OF_MAX_USING_NDARRAY_MAX_LIST=35
        INDEX_1_OF_MAX_USING_NDARRAY_MAX_LIST=36
        INDEX_0_OF_MAX_USING_NDARRAY_COUNT_NONZERO_LIST=37
        INDEX_1_OF_MAX_USING_NDARRAY_COUNT_NONZERO_LIST=38
        INDEX_0_OF_MAX_USING_NDARRAY_SUM_LIST = 39
        INDEX_1_OF_MAX_USING_NDARRAY_SUM_LIST = 40

    @classmethod
    def gen_getter(cls, type, attr, **kwargs):
        Gtrt = TriggerEventAnalysisRecordV2._GetterType
        lk = None
        if 'list' in kwargs:
            lk = kwargs['list']
        ndark = None
        if 'ndarray' in kwargs:
            ndark = kwargs['ndarray']
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
            return lambda o: sum([np.sum(g) for g in getattr(o,attr)])/len(getattr(o, lk)) if getattr(o,attr) is not None and lk is not None and getattr(o,lk) is not None and len(getattr(o,lk)) > 0 else -1.
        if type == Gtrt.AVG_NP_SUM:
            return lambda o: sum([np.sum(g) for g in getattr(o,attr)])/len(getattr(o, attr)) if getattr(o,attr) is not None and len(getattr(o,attr)) > 0 else -1.
        if type == Gtrt.SUM_NP_COUNT_NONZERO:
            return lambda o: sum([np.count_nonzero(g) for g in getattr(o,attr)]) if getattr(o,attr) is not None else -1
        if type == Gtrt.MAX_NP_COUNT_NONZERO:
            return lambda o: max([np.count_nonzero(g) for g in getattr(o,attr)]) if getattr(o,attr) is not None else -1
        if type == Gtrt.AVG_NP_COUNT_NONZERO:
            return lambda o: sum([np.count_nonzero(g) for g in getattr(o,attr)])/len(getattr(o, attr)) if getattr(o,attr) is not None and len(getattr(o,attr)) > 0 else -1.
        if type == Gtrt.INDEX_0:
            return lambda o: getattr(o,attr)[0] if getattr(o,attr) is not None and len(getattr(o,attr)) > 1 else -1
        if type == Gtrt.INDEX_1:
            return lambda o: getattr(o,attr)[1] if getattr(o,attr) is not None and len(getattr(o,attr)) > 1 else -1
        if type == Gtrt.MULT_INDEX_0_INDEX_1:
            return lambda o: getattr(o,attr)[1] * getattr(o,attr)[0] if getattr(o,attr) is not None and len(getattr(o, attr)) > 1 else -1
        if type == Gtrt.AVG_USING_NONZERO:
            return lambda o: np.sum(getattr(o,attr))/np.count_nonzero(getattr(o,attr) if getattr(o,attr) is not None else -1.
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
        if type == Gtrt.AVG_INDEX_0_USING_LEN:
            return lambda o: np.sum([float(g[0]) for g in getattr(o,attr)]) / len(getattr(o,attr)) if getattr(o,attr) is not None and len(getattr(o,attr)) > 0 else -1. # np.avg ?
        if type == Gtrt.AVG_INDEX_1_USING_LEN:
            return lambda o: np.sum([float(g[0]) for g in getattr(o,attr)]) / len(getattr(o,attr)) if getattr(o,attr) is not None and len(getattr(o,attr)) > 0 else -1. # np.avg ?
        if type == Gtrt.MAX_INDEX_0:
            return lambda o: np.max(getattr(o,attr)[0]) if getattr(o,attr) is not None and len(getattr(o,attr)) > 0 and getattr(o,attr)[0] is not None else -1
        if type == Gtrt.MAX_INDEX_1:
            return lambda o: np.max(getattr(o,attr)[0]) if getattr(o,attr) is not None and len(getattr(o,attr)) > 0 and getattr(o,attr)[0] is not None else -1
        if type == Gtrt.INDEX_0_OF_MAX_MULT_INDEX_0_1:
            return lambda o: getattr(o, attr)[np.argmax([c[1] * c[0] for c in getattr(o, attr)])][0] if getattr(o, attr) is not None and len(getattr(o, attr)) > 0 else -1
        if type == Gtrt.INDEX_1_OF_MAX_MULT_INDEX_0_1:
            return lambda o: getattr(o, attr)[np.argmax([c[1] * c[0] for c in getattr(o, attr)])][1] if getattr(o, attr) is not None and len(getattr(o, attr)) > 0 else -1
        if type == Gtrt.MAX_MULT_INDEX_0_INDEX_1:
            return lambda o: max([c[1] * c[0] for c in getattr(o, attr)]) if getattr(o, attr) is not None and len(getattr(o, attr)) > 0 else -1
        if type == Gtrt.AVG_MULT_INDEX_0_INDEX_1:
            return lambda o: np.sum([c[1] * c[0] for c in getattr(o, attr)])/len(getattr(o, attr)) if getattr(o, attr) is not None and len(getattr(o, attr)) > 0 else -1
        if type == Gtrt.INDEX_0_OF_MAX_USING_NDARRAY_MAX_LIST:
            return lambda o: getattr(o, attr)[ np.argmax([np.max(c) for c in getattr(o, ndark)]) ][0]  if getattr(o, attr) is not None and getattr(o, ndark) is not None and len(getattr(o, attr)) == len(getattr(o, ndark)) else -1
        if type == Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_MAX_LIST:
            return lambda o: getattr(o, attr)[ np.argmax([np.max(c) for c in getattr(o, ndark)]) ][1]  if getattr(o, attr) is not None and getattr(o, ndark) is not None and len(getattr(o, attr)) == len(getattr(o, ndark)) else -1
        if type == Gtrt.INDEX_0_OF_MAX_USING_NDARRAY_COUNT_NONZERO_LIST:
            return lambda o: getattr(o, attr)[ np.argmax([np.count_nonzero(c) for c in getattr(o, ndark)]) ][0]  if getattr(o, attr) is not None and getattr(o, ndark) is not None and len(getattr(o, attr)) == len(getattr(o, ndark)) else -1
        if type == Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_COUNT_NONZERO_LIST:
            return lambda o: getattr(o, attr)[ np.argmax([np.count_nonzero(c) for c in getattr(o, ndark)]) ][1]  if getattr(o, attr) is not None and getattr(o, ndark) is not None and len(getattr(o, attr)) == len(getattr(o, ndark)) else -1
        if type == Gtrt.INDEX_0_OF_MAX_USING_NDARRAY_SUM_LIST:
            return lambda o: getattr(o, attr)[ np.argmax([np.sum(c) for c in getattr(o, ndark)]) ][0]  if getattr(o, attr) is not None and getattr(o, ndark) is not None and len(getattr(o, attr)) == len(getattr(o, ndark)) else -1
        if type == Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_SUM_LIST:
            return lambda o: getattr(o, attr)[ np.argmax([np.sum(c) for c in getattr(o, ndark)]) ][1]  if getattr(o, attr) is not None and getattr(o, ndark) is not None and len(getattr(o, attr)) == len(getattr(o, ndark)) else -1


    def __init__(self):
        self.gtu_data = []
        self.triggered_pixels_x_y_coords = []
        self.triggered_pixel_groups = []
        self.triggered_pixels_x_y_hough_transform = []
        self.triggered_pixels_x_y_hough_transform__max_peak_line_coords = [(None, None), (None, None)]
        self.x_y_neighbourhood = []
        self.x_y_neighbourhood_dimensions = []
        self.x_y_hough_transform = []
        self.x_y_hough_transform__clusters_above_thr = []
        self.x_y_hough_transform__cluster_dimensions = []
        self.x_y_hough_transform__cluster_counts_sums = []
        self.x_y_hough_transform__max_peak_line_coords = [(None, None), (None, None)]
        self.x_y_hough_transform__thr_peak_line_coords = [(None, None), (None, None)]
        self.x_gtu_neighbourhood = []
        self.x_gtu_neighbourhood_size = []
        self.x_gtu_neighbourhood_dimensions = []
        self.x_gtu_hough_transform = []
        self.x_gtu_hough_transform__clusters_above_thr = []
        self.x_gtu_hough_transform__num_clusters_above_thr = []
        self.x_gtu_neighbourhood = []
        self.y_gtu_neighbourhood_size = []
        self.y_gtu_neighbourhood_dimensions = []
        self.y_gtu_hough_transform = []
        self.y_gtu_hough_transform__clusters_above_thr = []
        self.y_gtu_hough_transform__num_clusters_above_thr = []

        if not self.__class__.extra_attr_method_mapping:
            # 'extra_attr_method_mapping' not in self.__class__.__dict__ or

            Gtrt = TriggerEventAnalysisRecordV2._GetterType

            getters_def = {
                'num_gtu': (Gtrt.LEN, 'gtu_data'),
                'max_trg_box_per_gtu': lambda o: max([gtu_pdm_data.trg_box_per_gtu for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'sum_trg_box_per_gtu': lambda o: max([gtu_pdm_data.trg_box_per_gtu for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'avg_trg_box_per_gtu': lambda o: max([gtu_pdm_data.trg_box_per_gtu for gtu_pdm_data in o.gtu_data])/len(o.gtu_data) if o.gtu_data is not None and len(o.gtu_data) > 0 else -1,
                'max_trg_pmt_per_gtu': lambda o: max([gtu_pdm_data.trg_pmt_per_gtu for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'sum_trg_pmt_per_gtu': lambda o: max([gtu_pdm_data.trg_pmt_per_gtu for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'avg_trg_pmt_per_gtu': lambda o: max([gtu_pdm_data.trg_pmt_per_gtu for gtu_pdm_data in o.gtu_data])/len(o.gtu_data) if o.gtu_data is not None and len(o.gtu_data) > 0 else -1,
                'max_trg_ec_per_gtu': lambda o: max([gtu_pdm_data.trg_ec_per_gtu for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'sum_trg_ec_per_gtu': lambda o: max([gtu_pdm_data.trg_ec_per_gtu for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'avg_trg_ec_per_gtu': lambda o: max([gtu_pdm_data.trg_ec_per_gtu for gtu_pdm_data in o.gtu_data])/len(o.gtu_data) if o.gtu_data is not None and len(o.gtu_data) > 0 else -1,
                'max_n_persist': lambda o: max([gtu_pdm_data.n_persist for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'sum_n_persist': lambda o: max([gtu_pdm_data.n_persist for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'avg_n_persist': lambda o: max([gtu_pdm_data.n_persist for gtu_pdm_data in o.gtu_data])/len(o.gtu_data) if o.gtu_data is not None and len(o.gtu_data) > 0 else -1,
                'max_sum_l1_pdm': lambda o: max([gtu_pdm_data.sum_l1_pdm for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'sum_sum_l1_pdm': lambda o: max([gtu_pdm_data.sum_l1_pdm for gtu_pdm_data in o.gtu_data]) if o.gtu_data is not None else -1,
                'avg_sum_l1_pdm': lambda o: max([gtu_pdm_data.sum_l1_pdm for gtu_pdm_data in o.gtu_data])/len(o.gtu_data) if o.gtu_data is not None and len(o.gtu_data) > 0 else -1,
            }

            dim_getter_def = {
                '{d}_active_pixels': (Gtrt.NP_COUNT_NONZERO, '{d}_projection'),
                'triggered_pixels_{d}_num': (Gtrt.LEN, 'triggered_pixels_coords'),
                'triggered_pixels_{d}_sum3x3_sum': (Gtrt.NP_COUNT_NONZERO, 'triggered_pixels_{d}_integrated_sum3x3'),
                'triggered_pixels_{d}_sum3x3_norm_sum': (Gtrt.NP_SUM_NORMALIZED ,'triggered_pixels_{d}_integrated_sum3x3'),
                'triggered_pixels_{d}_sum3x3_avg': (Gtrt.AVG_NP_SUM_USING_LIST_LEN , 'triggered_pixels_x_y_integrated', {'list': 'triggered_pixels_coords'}),
                'triggered_pixels_{d}_groups_num': (Gtrt.LEN ,'triggered_pixels_{d}_groups'),
                'triggered_pixels_{d}_groups_max_size': (Gtrt.MAX_LEN ,'triggered_pixels_{d}_groups'),
                'triggered_pixels_{d}_groups_avg_size': (Gtrt.AVG_LEN ,'triggered_pixels_{d}_groups'),
                'triggered_pixels_{d}_groups_sum_sum_sum3x3': (Gtrt.SUM_NP_SUM, 'triggered_pixels_{d}_groups_integrated_sum3x3'),
                'triggered_pixels_{d}_groups_max_sum_sum3x3': (Gtrt.MAX_NP_SUM,'triggered_pixels_{d}_groups_integrated_sum3x3'),
                'triggered_pixels_{d}_groups_avg_sum_sum3x3': (Gtrt.AVG_NP_SUM_USING_LIST_LEN, 'triggered_pixels_{d}_groups', {'list': 'triggered_pixels_{d}_groups'}),

                '{d}_neighbourhood_size': (Gtrt.NP_COUNT_NONZERO, '{d}_neighbourhood'),
                '{d}_neighbourhood_width': (Gtrt.INDEX_1, '{d}_neighbourhood_dimensions'),
                '{d}_neighbourhood_height': (Gtrt.INDEX_0, '{d}_neighbourhood_dimensions'),
                '{d}_neighbourhood_area': (Gtrt.MULT_INDEX_0_INDEX_1, '{d}_neighbourhood_dimensions[1'),
                '{d}_neighbourhood_counts_sum': (Gtrt.NP_SUM, '{d}_neighbourhood'),
                '{d}_neighbourhood_counts_avg': (Gtrt.AVG_USING_NONZERO, '{d}_neighbourhood'),
                '{d}_neighbourhood_counts_norm_sum': (Gtrt.NP_SUM_NORMALIZED, '{d}_neighbourhood'),

                '{d}_hough_transform__max_peak_line_rot': (Gtrt.PLUS_PI_OVER_2, '{d}_hough_transform__max_peak_phi'),
                '{d}_hough_transform__max_peak_line_coord_0_x': (Gtrt.INDEX_0_1, '{d}_hough_transform__max_peak_line_coords'),
                '{d}_hough_transform__max_peak_line_coord_0_y': (Gtrt.INDEX_0_0, '{d}_hough_transform__max_peak_line_coords'),
                '{d}_hough_transform__max_peak_line_coord_1_x': (Gtrt.INDEX_1_1, '{d}_hough_transform__max_peak_line_coords'),
                '{d}_hough_transform__max_peak_line_coord_1_y': (Gtrt.INDEX_1_0, '{d}_hough_transform__max_peak_line_coords'),
            }
            
            trigg_thr_def = {
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__height_of_max_cluster_area': (Gtrt.INDEX_1_OF_MAX_MULT_INDEX_0_1, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}_dimensions'),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__height_of_max_cluster_size': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_COUNT_NONZERO_LIST, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}_dimensions', {'ndarray': 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}'}),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__height_of_max_cluster_counts_sum': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_SUM_LIST, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}_dimensions', {'ndarray': 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}'}),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__height_of_max_peak_cluster': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_MAX_LIST, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}_dimensions', {'ndarray': 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}'}),

                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__width_of_max_cluster_area': (Gtrt.INDEX_1_OF_MAX_MULT_INDEX_0_1, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}_dimensions', {'ndarray': 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}'}),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__width_of_max_cluster_size': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_COUNT_NONZERO_LIST, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}_dimensions', {'ndarray': 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}'}),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__width_of_max_cluster_counts_sum': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_SUM_LIST, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}_dimensions', {'ndarray': 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}'}),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__width_of_max_peak_cluster': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_MAX_LIST, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}_dimensions', {'ndarray': 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}'}),

                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__line_rot': (Gtrt.PLUS_PI_OVER_2, 'triggered_pixels_{{d}}_hough_transform__peak_thr{n}_phi'),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__line_coord_0_x': (Gtrt.INDEX_0_1, 'triggered_pixels_{{d}}_hough_transform__peak_thr{n}_line_coords'),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__line_coord_0_y': (Gtrt.INDEX_0_0, 'triggered_pixels_{{d}}_hough_transform__peak_thr{n}_line_coords'),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__line_coord_1_x': (Gtrt.INDEX_1_1, 'triggered_pixels_{{d}}_hough_transform__peak_thr{n}_line_coords'),
                'triggered_pixels_{{d}}_hough_transform__peak_thr{n}__line_coord_1_y': (Gtrt.INDEX_1_0, 'triggered_pixels_{{d}}_hough_transform__peak_thr{n}_line_coords'),
            }

            pix_thr_def = {
                '{{d}}_hough_transform__peak_thr{n}__max_cluster_width': (Gtrt.MAX_INDEX_1, '{{d}}_hough_transform__clusters_above_thr{n}_dimensions'),
                '{{d}}_hough_transform__peak_thr{n}__avg_cluster_width': (Gtrt.AVG_INDEX_1_USING_LEN, '{{d}}_hough_transform__clusters_above_thr{n}_dimensions'),
                '{{d}}_hough_transform__peak_thr{n}__max_cluster_height': (Gtrt.MAX_INDEX_1, '{{d}}_hough_transform__clusters_above_thr{n}_dimensions'),
                '{{d}}_hough_transform__peak_thr{n}__avg_cluster_height': (Gtrt.AVG_INDEX_1_USING_LEN, '{{d}}_hough_transform__clusters_above_thr{n}_dimensions'),
                '{{d}}_hough_transform__peak_thr{n}__max_cluster_area': (Gtrt.MAX_MULT_INDEX_0_INDEX_1, '{{d}}_hough_transform__clusters_above_thr{n}_dimensions'),
                '{{d}}_hough_transform__peak_thr{n}__avg_cluster_area': (Gtrt.AVG_MULT_INDEX_0_INDEX_1, '{{d}}_hough_transform__clusters_above_thr{n}_dimensions'),
                '{{d}}_hough_transform__peak_thr{n}__max_cluster_size': (Gtrt.MAX_NP_COUNT_NONZERO, '{{d}}_hough_transform__clusters_above_thr{n}'),                             #
                '{{d}}_hough_transform__peak_thr{n}__avg_cluster_size': (Gtrt.AVG_NP_COUNT_NONZERO, '{{d}}_hough_transform__clusters_above_thr{n}'),
                '{{d}}_hough_transform__peak_thr{n}__max_cluster_counts_sum': (Gtrt.MAX_NP_SUM, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}'),
                '{{d}}_hough_transform__peak_thr{n}__avg_cluster_counts_sum': (Gtrt.AVG_NP_SUM, 'triggered_pixels_{{d}}_hough_transform__clusters_above_thr{n}'),

                '{{d}}_hough_transform__peak_thr{n}__height_of_max_cluster_area': (Gtrt.INDEX_1_OF_MAX_MULT_INDEX_0_1, '{{d}}_hough_transform__clusters_above_thr{n}_dimensions'),
                '{{d}}_hough_transform__peak_thr{n}__height_of_max_cluster_size': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_COUNT_NONZERO_LIST, '{{d}}_hough_transform__clusters_above_thr{n}'),
                '{{d}}_hough_transform__peak_thr{n}__height_of_max_cluster_counts_sum': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_SUM_LIST, '{{d}}_hough_transform___clusters_above_thr{n}'),
                '{{d}}_hough_transform__peak_thr{n}__height_of_max_peak_cluster': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_MAX_LIST, '{{d}}_hough_transform__clusters_above_thr{n}'),

                '{{d}}_hough_transform__peak_thr{n}__width_of_max_cluster_area': (Gtrt.INDEX_1_OF_MAX_MULT_INDEX_0_1, '{{d}}_hough_transform__clusters_above_thr{n}_dimensions'),
                '{{d}}_hough_transform__peak_thr{n}__width_of_max_cluster_size': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_COUNT_NONZERO_LIST, '{{d}}_hough_transform__clusters_above_thr{n}'),
                '{{d}}_hough_transform__peak_thr{n}__width_of_max_cluster_counts_sum': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_SUM_LIST, '{{d}}_hough_transform___clusters_above_thr{n}'),
                '{{d}}_hough_transform__peak_thr{n}__width_of_max_peak_cluster': (Gtrt.INDEX_1_OF_MAX_USING_NDARRAY_MAX_LIST, '{{d}}_hough_transform__clusters_above_thr{n}'),

                '{{d}}_hough_transform__peak_thr{n}__line_rot': (Gtrt.PLUS_PI_OVER_2, '{{d}}_hough_transform__peak_thr{n}_phi'),
                '{{d}}_hough_transform__peak_thr{n}__line_coord_0_x': (Gtrt.INDEX_0_1, '{{d}}_hough_transform__peak_thr{n}_line_coords'),
                '{{d}}_hough_transform__peak_thr{n}__line_coord_0_y': (Gtrt.INDEX_0_0, '{{d}}_hough_transform__peak_thr{n}_line_coords'),
                '{{d}}_hough_transform__peak_thr{n}__line_coord_1_x': (Gtrt.INDEX_1_1, '{{d}}_hough_transform__peak_thr{n}_line_coords'),
                '{{d}}_hough_transform__peak_thr{n}__line_coord_1_y': (Gtrt.INDEX_1_0, '{{d}}_hough_transform__peak_thr{n}_line_coords'),
            }

            def format_n(src, dest, r, varname='n'):
                for n in r:
                    for k, v in src.items():
                        v1 = v[1].format(**{varname:n})
                        v2 = None
                        if len(v) > 2 and isinstance(v[2],dict):
                            nd = {}
                            for dk, dv in v[2]:
                                nd[dk] = dv.format(n=n)
                            v2 = nd
                        dest[k.format(n=n)] = (v[0], v1, v2)

            format_n(trigg_thr_def, dim_getter_def, range(1,2))
            format_n(pix_thr_def, dim_getter_def, range(1,4))
            format_n(dim_getter_def, getters_def, ('x_y', 'x_gtu', 'y_gtu'), 'd')

            for k,v in getters_def.items():
                if isinstance(v,(tuple, list)):
                    getters_def[k] = self.gen_getter(*v)

            setattr(self.__class__, "extra_attr_method_mapping", getters_def)


if __name__ == "__main__":
    # execute only if run as a script
    r = TriggerEventAnalysisRecordV2()
    r.triggered_pixels_x_y_hough_transform__max_peak_phi = 34

    print(r.triggered_pixels_x_y_hough_transform__max_peak_line_rot)
