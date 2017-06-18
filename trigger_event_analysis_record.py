import numpy as np
import processing_config

try:
    trigger_event_string_column_order = processing_config.trigger_event_string_column_order
except AttributeError:
    trigger_event_string_column_order = \
        ["source_file_acquisition", "source_file_trigger",
        "global_gtu", "packet_id", "gtu_in_packet", "num_gtu"]

try:
    trigger_event_str_prop_separator = processing_config.trigger_event_str_prop_separator
except AttributeError:
    trigger_event_str_prop_separator = "\n"


class TriggerEventAnalysisRecord(object):

    extra_attr_method_mapping = {}

    #############################################
    # Should be provided before processing

    source_file_acquisition = ""
    source_file_trigger = ""

    exp_tree = None

    global_gtu = -1
    packet_id = -1
    gtu_in_packet = -1

    gtu_data = None # list of GtuPdmData objects

    ###########################################

    program_version = 0

    ###########################################

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
    # triggers_x_y_neighbourhood_size = None # image
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

    hough_transform_x_y__max_peak_line_coords = -1 # peak determined only from the maximal point of the hough space

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

    def __init__(self):

        self.triggered_pixels = []
        self.triggered_pixel_groups = []
        self.triggered_pixels_x_y_hough_transform = []
        self.triggers_x_y_neighbourhood = []
        self.triggers_x_y_neighbourhood_dimensions = []
        self.hough_transform_x_y = []
        self.hough_transform_x_y__clusters_above_thr = []
        self.hough_transform_x_y__cluster_dimensions = []
        self.hough_transform_x_y__cluster_counts_sums = []
        self.hough_transform_x_y__thr_peak_line_coords = []
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
            #'extra_attr_method_mapping' not in self.__class__.__dict__ or
            d = {
                'num_gtu': lambda o: len(o.gtu_data),
                'num_triggered_pixels': lambda o: len(o.triggered_pixels),
                'num_triggered_pixel_groups': lambda o: len(o.triggered_pixel_groups),
                'max_triggered_pixel_group_size': lambda o: max([len(g) for g in o.triggered_pixel_groups]) if len(o.triggered_pixel_groups)>0 else -1,
                'avg_trigger_group_size': lambda o: sum([len(g) for g in o.triggered_pixel_groups])/len(o.triggered_pixel_groups) if len(o.triggered_pixel_groups)>0 else -1,
                'triggered_pixels_x_y_hough_transform__max_peak_line_rot': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_phi + np.pi/2,
                'triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_x': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_line_coords[0][1],
                'triggered_pixels_x_y_hough_transform__max_peak_line_coord_0_y': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_line_coords[0][0],
                'triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_x': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_line_coords[1][1],
                'triggered_pixels_x_y_hough_transform__max_peak_line_coord_1_y': lambda o: o.triggered_pixels_x_y_hough_transform__max_peak_line_coords[1][0],
                'triggers_x_y_neighbourhood_width': lambda o: o.triggers_x_y_neighbourhood_dimensions[1],
                'triggers_x_y_neighbourhood_height': lambda o: o.triggers_x_y_neighbourhood_dimensions[0],
                'triggers_x_y_neighbourhood_area': lambda o: o.triggers_x_y_neighbourhood_dimensions[1] * o.triggers_x_y_neighbourhood_dimensions[0],
                'triggers_x_y_neighbourhood_size': lambda o: np.count_nonzero(o.triggers_x_y_neighbourhood),
                'hough_transform_x_y__num_clusters_above_thr': lambda o: len(o.hough_transform_x_y__clusters_above_thr),
                'hough_transform_x_y__max_cluster_width': lambda o: max([c[1] for c in o.hough_transform_x_y__cluster_dimensions]) if len(o.hough_transform_x_y__cluster_dimensions)>0 else -1,
                'hough_transform_x_y__avg_cluster_width': lambda o: sum([c[1] for c in o.hough_transform_x_y__cluster_dimensions])/len(o.hough_transform_x_y__cluster_dimensions),
                'hough_transform_x_y__max_cluster_height': lambda o: max([c[0] for c in o.hough_transform_x_y__cluster_dimensions]) if len(o.hough_transform_x_y__cluster_dimensions)>0 else -1,
                'hough_transform_x_y__avg_cluster_height': lambda o: sum([c[0] for c in o.hough_transform_x_y__cluster_dimensions])/len(o.hough_transform_x_y__cluster_dimensions),
                'hough_transform_x_y__max_cluster_area': lambda o: max([c[1]*c[0] for c in o.hough_transform_x_y__cluster_dimensions]) if len(o.hough_transform_x_y__cluster_dimensions)>0 else -1,
                'hough_transform_x_y__avg_cluster_area': lambda o: sum([c[1]*c[0] for c in o.hough_transform_x_y__cluster_dimensions])/len(o.hough_transform_x_y__cluster_dimensions),
                'hough_transform_x_y__max_cluster_size': lambda o: max([np.count_nonzero(c) for c in o.hough_transform_x_y__clusters_above_thr]) if len(o.hough_transform_x_y__clusters_above_thr)>0 else -1,
                'hough_transform_x_y__avg_cluster_size': lambda o: sum([np.count_nonzero(c) for c in o.hough_transform_x_y__clusters_above_thr])/len(o.hough_transform_x_y__clusters_above_thr),
                'hough_transform_x_y__max_cluster_counts_sum': lambda o: max([c for c in o.hough_transform_x_y__cluster_counts_sums]) if len(o.hough_transform_x_y__cluster_counts_sums)>0 else -1,
                'hough_transform_x_y__avg_cluster_counts_sum': lambda o: sum([c for c in o.hough_transform_x_y__cluster_counts_sums])/len(o.hough_transform_x_y__cluster_counts_sums),
                'hough_transform_x_y__max_peak_line_rot': lambda o: o.hough_transform_x_y__max_peak_phi + np.pi/2,
                'hough_transform_x_y__max_peak_line_coord_0_x': lambda o: o.hough_transform_x_y__max_peak_line_coords[0][1],
                'hough_transform_x_y__max_peak_line_coord_0_y': lambda o: o.hough_transform_x_y__max_peak_line_coords[0][0],
                'hough_transform_x_y__max_peak_line_coord_1_x': lambda o: o.hough_transform_x_y__max_peak_line_coords[1][1],
                'hough_transform_x_y__max_peak_line_coord_1_y': lambda o: o.hough_transform_x_y__max_peak_line_coords[1][0],
                'hough_transform_x_y__thr_peak_line_rot': lambda o: o.hough_transform_x_y__thr_peak_phi + np.pi/2,
                'hough_transform_x_y__thr_peak_line_coord_0_x': lambda o: o.hough_transform_x_y__thr_peak_line_coords[0][1],
                'hough_transform_x_y__thr_peak_line_coord_0_y': lambda o: o.hough_transform_x_y__thr_peak_line_coords[0][0],
                'hough_transform_x_y__thr_peak_line_coord_1_x': lambda o: o.hough_transform_x_y__thr_peak_line_coords[1][1],
                'hough_transform_x_y__thr_peak_line_coord_1_y': lambda o: o.hough_transform_x_y__thr_peak_line_coords[1][0],
            }

            #self.__class__.__dict__[''] =
            setattr(self.__class__, "extra_attr_method_mapping", d)

    def __str__(self):
        return self.to_str()

    def to_str(self, separator=None, columns=None):
        global trigger_event_string_column_order
        global trigger_event_str_prop_separator
        if separator is None:
            separator = trigger_event_str_prop_separator
        if columns is None:
            columns = trigger_event_string_column_order
        return separator.join([str(getattr(self, n)) for n in columns])

    def __getattr__(self, item):
        d = self.__class__.__dict__['extra_attr_method_mapping']
        if item not in d:
            raise Exception("Attribute \"{}\" is not available".format(item))
        return d[item](self)

    @classmethod
    def get_string_column_order(self):
        global trigger_event_string_column_order
        return trigger_event_string_column_order


if __name__ == "__main__":
    # execute only if run as a script
    r = TriggerEventAnalysisRecord()
    r.triggered_pixels_x_y_hough_transform__max_peak_phi = 34

    print(r.triggered_pixels_x_y_hough_transform__max_peak_line_rot)
