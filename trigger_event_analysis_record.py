import numpy as np

class TriggerEventAnalysisRecord(object):

    source_file_acquisition = ""
    source_file_trigger = ""

    exp_tree = None

    global_gtu = -1
    packet_id = -1
    gtu_in_packet = -1

    gtu_data = None # list of GtuPdmData objects

    triggered_pixels = None
    # num
    triggered_pixel_groups = None
    # num
    # max size
    # avg size


    # value in th hough transform for the trigger pixel should be its weight
    triggered_pixels_x_y_hough_transform = None
    triggered_pixels_x_y_hough_transform__max_peak_rho = -1 # peak determined only from the maximal point of the hough space
    triggered_pixels_x_y_hough_transform__max_peak_phi = -1 # peak determined only from the maximal point of the hough space

    # dynamic TODO
    # triggered_pixels_x_y_hough_transform__max_peak_line_rot  = -1 # peak determined only from the maximal point of the hough space

    triggered_pixels_x_y_hough_transform__max_peak_line_coords = -1 # peak determined only from the maximal point of the hough space

    # dynamic TODO
    # triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_x = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_y = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_x = -1 # peak determined only from the maximal point of the hough space
    # triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_y = -1 # peak determined only from the maximal point of the hough space

    triggers_x_y_neighbourhood = None # image
    triggers_x_y_neighbourhood_size = None # image
    triggers_x_y_neighbourhood_dimensions = None # image

    hough_transform_x_y = None
    hough_transform_x_y__clusters_above_thr = None
    # hough_transform_x_y__num_clusters_above_thr = None

    # dynamic
    #hough_transform_x_y__max_cluster_size = -1
    #hough_transform_x_y__avg_cluster_size = -1

    hough_transform_x_y__cluster_dimensions = None
    # dynamic from cluster_dimensions
    # hough_transform_x_y__max_cluster_dimensions = -1
    # hough_transform_x_y__avg_cluster_dimensions = -1
    
    hough_transform_x_y__cluster_counts_sums = None
    #dynamic
    # hough_transform_x_y__max_cluster_counts_sum = -1
    # hough_transform_x_y__avg_cluster_counts_sum = -1

    hough_transform_x_y__max_peak_rho = -1 # peak determined only from the maximal point of the hough space
    hough_transform_x_y__max_peak_phi = -1 # peak determined only from the maximal point of the hough space
    # dynamic
    # hough_transform_x_y__max_peak_line_rot  = -1 # peak determined only from the maximal point of the hough space

    hough_transform_x_y__max_peak_coords = -1 # peak determined only from the maximal point of the hough space

    #dynamic
    # hough_transform_x_y__max_peak_line_coord_0_x = -1 # peak determined only from the maximal point of the hough space
    # hough_transform_x_y__max_peak_line_coord_0_y = -1 # peak determined only from the maximal point of the hough space
    # hough_transform_x_y__max_peak_line_coord_1_x = -1 # peak determined only from the maximal point of the hough space
    # hough_transform_x_y__max_peak_line_coord_1_y = -1 # peak determined only from the maximal point of the hough space
    
    hough_transform_x_y__thr_peak_rho = -1 # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_x_y__thr_peak_phi = -1 # peak is average rho and phi of points over a threshold relative to the maximum peak value
    #dynamic
    #hough_transform_x_y__thr_peak_line_rot = -1 # peak is average rho and phi of points over a threshold relative to the maximum peak value

    hough_transform_x_y__thr_peak_line_coords = None # peak is average rho and phi of points over a threshold relative to the maximum peak value

    #dynamic
    #hough_transform_x_y__thr_peak_line_coord_0_x = -1 # peak is average rho and phi of points over a threshold relative to the maximum peak value
    #hough_transform_x_y__thr_peak_line_coord_0_y = -1 # peak is average rho and phi of points over a threshold relative to the maximum peak value
    #hough_transform_x_y__thr_peak_line_coord_1_x = -1 # peak is average rho and phi of points over a threshold relative to the maximum peak value
    #hough_transform_x_y__thr_peak_line_coord_1_y = -1 # peak is average rho and phi of points over a threshold relative to the maximum peak value



    triggers_x_gtu_neighbourhood = None # image
    triggers_x_gtu_neighbourhood_size = None # image
    triggers_x_gtu_neighbourhood_dimensions = None # image

    hough_transform_x_gtu = None
    hough_transform_x_gtu__clusters_above_thr = None
    hough_transform_x_gtu__num_clusters_above_thr = None

    hough_transform_x_gtu__max_cluster_size = -1
    hough_transform_x_gtu__avg_cluster_size = -1

    hough_transform_x_gtu__max_cluster_dimensions = -1
    hough_transform_x_gtu__avg_cluster_dimensions = -1
    
    hough_transform_x_gtu__max_cluster_counts_sum = -1
    hough_transform_x_gtu__avg_cluster_counts_sum = -1

    hough_transform_x_gtu__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_gtu__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_gtu__max_peak_line_rot = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_gtu__max_peak_line_coord_0_x = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_gtu__max_peak_line_coord_0_y = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_gtu__max_peak_line_coord_1_x = -1  # peak determined only from the maximal point of the hough space
    hough_transform_x_gtu__max_peak_line_coord_1_y = -1  # peak determined only from the maximal point of the hough space

    hough_transform_x_gtu__thr_peak_rho = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_x_gtu__thr_peak_phi = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_x_gtu__thr_peak_line_rot = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_x_gtu__thr_peak_line_coord_0_x = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_x_gtu__thr_peak_line_coord_0_y = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_x_gtu__thr_peak_line_coord_1_x = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_x_gtu__thr_peak_line_coord_1_y = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value

    triggers_x_gtu_neighbourhood = None # image
    triggers_x_gtu_neighbourhood_size = None # image
    triggers_x_gtu_neighbourhood_dimensions = None # image
    
    hough_transform_y_gtu_ = None
    hough_transform_y_gtu__clusters_above_thr = None
    hough_transform_y_gtu__num_clusters_above_thr = None

    hough_transform_y_gtu__max_cluster_size = -1
    hough_transform_y_gtu__avg_cluster_size = -1

    hough_transform_y_gtu__max_cluster_dimensions = -1
    hough_transform_y_gtu__avg_cluster_dimensions = -1
    
    hough_transform_y_gtu__max_cluster_counts_sum = -1
    hough_transform_y_gtu__avg_cluster_counts_sum = -1

    hough_transform_y_gtu__max_peak_rho = -1  # peak determined only from the maximal point of the hough space
    hough_transform_y_gtu__max_peak_phi = -1  # peak determined only from the maximal point of the hough space
    hough_transform_y_gtu__max_peak_line_rot = -1  # peak determined only from the maximal point of the hough space
    hough_transform_y_gtu__max_peak_line_coord_0_x = -1  # peak determined only from the maximal point of the hough space
    hough_transform_y_gtu__max_peak_line_coord_0_y = -1  # peak determined only from the maximal point of the hough space

    hough_transform_y_gtu__thr_peak_rho = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_y_gtu__thr_peak_phi = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_y_gtu__thr_peak_line_rot = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_y_gtu__thr_peak_line_coord_0_x = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_y_gtu__thr_peak_line_coord_0_y = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_y_gtu__thr_peak_line_coord_1_x = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    hough_transform_y_gtu__thr_peak_line_coord_1_y = -1  # peak is average rho and phi of points over a threshold relative to the maximum peak value
    
    _output_column_order = ["source_file_acquisition", "source_file_trigger",
                            "global_gtu", "packet_id", "gtu_in_packet", "num_gtu",
                            "num_triggered_pixels", "num_triggered_pixel_groups", "max_triggered_pixel_group_size",
                            "triggered_pixels_hough_transform_x_y__max_peak_line_rot",
                            "triggered_pixels_hough_transform_x_y__max_peak_line_coord_0_x",
                            "triggered_pixels_hough_transform_x_y__max_peak_line_coord_0_y",
                            "triggered_pixels_hough_transform_x_y__max_peak_line_coord_1_x",
                            "triggered_pixels_hough_transform_x_y__max_peak_line_coord_1_y",
                            "triggers_x_y_neighbourhood_size",
                            "triggers_x_y_neighbourhood_dimensions",
                            "hough_transform_x_y__num_clusters_above_thr",
                            "hough_transform_x_y__max_cluster_size",
                            "hough_transform_x_y__avg_cluster_size",
                            "hough_transform_x_y__max_cluster_dimensions",
                            "hough_transform_x_y__avg_cluster_dimensions",
                            "hough_transform_x_y__max_cluster_counts_sum",
                            "hough_transform_x_y__avg_cluster_counts_sum",
                            "hough_transform_x_y__max_peak_line_rot",
                            "hough_transform_x_y__max_peak_line_coord_0_x",
                            "hough_transform_x_y__max_peak_line_coord_0_y",
                            "hough_transform_x_y__max_peak_line_coord_1_x",
                            "hough_transform_x_y__max_peak_line_coord_1_y",
                            "triggers_x_gtu_neighbourhood_size",
                            "triggers_x_gtu_neighbourhood_dimensions",
                            "hough_transform_x_gtu__num_clusters_above_thr",
                            "hough_transform_x_gtu__max_cluster_size",
                            "hough_transform_x_gtu__avg_cluster_size",
                            "hough_transform_x_gtu__max_cluster_dimensions",
                            "hough_transform_x_gtu__avg_cluster_dimensions",
                            "hough_transform_x_gtu__max_cluster_counts_sum",
                            "hough_transform_x_gtu__avg_cluster_counts_sum",
                            "hough_transform_x_gtu__max_peak_line_rot",
                            "hough_transform_x_gtu__max_peak_line_coord_0_x",
                            "hough_transform_x_gtu__max_peak_line_coord_0_y",
                            "hough_transform_x_gtu__max_peak_line_coord_1_x",
                            "hough_transform_x_gtu__max_peak_line_coord_1_y",
                            "triggers_y_gtu_neighbourhood_size",
                            "triggers_y_gtu_neighbourhood_dimensions",
                            "hough_transform_y_gtu__num_clusters_above_thr",
                            "hough_transform_y_gtu__max_cluster_size",
                            "hough_transform_y_gtu__avg_cluster_size",
                            "hough_transform_y_gtu__max_cluster_dimensions",
                            "hough_transform_y_gtu__avg_cluster_dimensions",
                            "hough_transform_y_gtu__max_cluster_counts_sum",
                            "hough_transform_y_gtu__avg_cluster_counts_sum",
                            "hough_transform_y_gtu__max_peak_line_rot",
                            "hough_transform_y_gtu__max_peak_line_coord_0_x",
                            "hough_transform_y_gtu__max_peak_line_coord_0_y",
                            "hough_transform_y_gtu__max_peak_line_coord_1_x",
                            "hough_transform_y_gtu__max_peak_line_coord_1_y"
                            ]

    def __init__(self, triggered_pixels=[], trigger_groups=[], ):
        self.triggered_pixel_groups = trigger_groups
        self.triggered_pixels = triggered_pixels
        if 'extra_attr_method_mapping' not in self.__class__.__dict__:
            self.__class__.__dict__['extra_attr_method_mapping'] = {
                'num_triggered_pixels': lambda o: len(o.triggered_pixels),
                'num_triggered_pixel_groups': lambda o: len(o.trigger_groups),
                'max_triggered_pixel_group_size:': lambda o: max([len(g) for g in o.trigger_groups]),
                'avg_trigger_group_size:': lambda o: sum([len(g) for g in o.trigger_groups])/len(trigger_groups),
                'triggered_pixels_x_y_hough_transform__max_peak_line_rot': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_phi + np.pi,
                'triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_x': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_line_coords[0][1],
                'triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_y': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_line_coords[0][0],
                'triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_x': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_line_coords[1][1],
                'triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_y': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_line_coords[1][0],
                'hough_transform_x_y__num_clusters_above_thr': lambda o: len(o.hough_transform_x_y__num_clusters_above_thr),
                'hough_transform_x_y__max_cluster_size': lambda o: max([len(c) for c in o.hough_transform_x_y__clusters_above_thr]),
                'hough_transform_x_y__avg_cluster_size': lambda o: sum([len(c) for c in o.hough_transform_x_y__clusters_above_thr])/len(o.hough_transform_x_y__clusters_above_thr),
                'hough_transform_x_y__max_cluster_counts_sum': lambda o: max([c for c in o.hough_transform_x_y__cluster_counts_sums]),
                'hough_transform_x_y__avg_cluster_counts_sum': lambda o: sum([c for c in o.hough_transform_x_y__cluster_counts_sums])/len(o.hough_transform_x_y__cluster_counts_sums),
                'hough_transform_x_y__max_peak_line_rot': lambda o: o.hough_transform_x_y__max_peak_phi + np.pi,      
                'hough_transform_x_y__max_peak_line_coord_0_x': lambda o: o.hough_transform_x_y__max_peak_coords[0][1],
                'hough_transform_x_y__max_peak_line_coord_0_y': lambda o: o.hough_transform_x_y__max_peak_coords[0][0],
                'hough_transform_x_y__max_peak_line_coord_1_x': lambda o: o.hough_transform_x_y__max_peak_coords[1][1],
                'hough_transform_x_y__max_peak_line_coord_1_y': lambda o: o.hough_transform_x_y__max_peak_coords[1][0],
                'hough_transform_x_y__thr_peak_line_rot': lambda o: o.hough_transform_x_y__thr_peak_phi + np.pi,      
                'hough_transform_x_y__thr_peak_line_coord_0_x': lambda o: o.hough_transform_x_y__thr_peak_coords[0][1],
                'hough_transform_x_y__thr_peak_line_coord_0_y': lambda o: o.hough_transform_x_y__thr_peak_coords[0][0],
                'hough_transform_x_y__thr_peak_line_coord_1_x': lambda o: o.hough_transform_x_y__thr_peak_coords[1][1],
                'hough_transform_x_y__thr_peak_line_coord_1_y': lambda o: o.hough_transform_x_y__thr_peak_coords[1][0],
                
                
            }

    def __str__(self):
        return super().__str__()

    def __getattr__(self, item):
        d = self.__class__.__dict__['extra_attr_method_mapping']
        if item not in d:
            raise Exception('Attribute "{}" is not available'.format(item))
        return d[item](self)
