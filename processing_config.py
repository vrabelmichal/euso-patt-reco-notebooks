from base_classes import *
proc_params = EventProcessingParams()

proc_params.triggered_pixels_group_max_gap = 5

proc_params.triggered_pixels_ht_size = 1
proc_params.triggered_pixels_ht_phi_num_steps = 90  # 2 deg per step
proc_params.triggered_pixels_ht_rho_step = 2

proc_params.x_y_neighbour_selection_rules = [NeighbourSelectionRules(3, .3, False),
                                             NeighbourSelectionRules(3, 1, True),
                                             NeighbourSelectionRules(1, .9, True)]

proc_params.x_y_ht_size = .5
proc_params.x_y_ht_phi_num_steps = 90  # 2 deg per step
proc_params.x_y_ht_rho_step = 2

proc_params.x_y_ht_peak_threshold_frac_of_max = .85
proc_params.x_y_ht_peak_gap = 3
# proc_params.x_y_ht_global_peak_threshold_frac_of_max = .95
proc_params.x_y_ht_global_peak_threshold_frac_of_max = .98

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
    "triggers_x_y_neighbourhood_area", # todo wrong
    "triggers_x_y_neighbourhood_width", # todo wrong
    "triggers_x_y_neighbourhood_height", # todo wrong
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
