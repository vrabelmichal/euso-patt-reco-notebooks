from base_classes import BaseAnalysisRecordColumnInfo

class EventAnalysisRecordV3ColumnInfo(BaseAnalysisRecordColumnInfo):

    # PostgreSQL's Max Identifier Length Is 63 Bytes
    config_info_table_columns = [

        ('gtu_before_trigger', 'integer NOT NULL'),
        ('gtu_after_trigger', 'integer NOT NULL'),

        # ('calibration_map', 'text'),

        ('substract_background', 'boolean NOT NULL'),
        ('background_offset', 'integer NOT NULL'),
        ('background_length', 'integer NOT NULL'),
        ('background_gauss_sigma', 'integer NOT NULL'),

        ('thresholding_method', 'varchar(32) NOT NULL'),
        ('threshold_n_factor', 'integer NOT NULL'),

        ('trigg_x_y__group_max_gap', 'integer NOT NULL'),
        ('trigg_gtu_x__group_max_gap', 'integer NOT NULL'),
        ('trigg_gtu_y__group_max_gap', 'integer NOT NULL'),

        ('trigg_x_y_hough__line_thickness', 'real NOT NULL'),
        ('trigg_x_y_hough__phi_num_steps', 'integer NOT NULL'),
        ('trigg_x_y_hough__rho_step', 'real NOT NULL'),

        ('trigg_gtu_x_hough__line_thickness', 'real NOT NULL'),
        ('trigg_gtu_x_hough__phi_num_steps', 'integer NOT NULL'),
        ('trigg_gtu_x_hough__rho_step', 'real NOT NULL'),

        ('trigg_gtu_y_hough__line_thickness', 'real NOT NULL'),
        ('trigg_gtu_y_hough__phi_num_steps', 'integer NOT NULL'),
        ('trigg_gtu_y_hough__rho_step', 'real NOT NULL'),

        ('trigg_x_y_hough__pixel_peak_gap', 'integer NOT NULL'),
        ('trigg_x_y_hough__pi_norm_peak_gap', 'real NOT NULL'),
        ('trigg_x_y_hough__peak_thr1', 'real NOT NULL'),
        ('trigg_x_y_hough__peak_thr2', 'real NOT NULL'),
        ('trigg_x_y_hough__peak_thr3', 'real NOT NULL'),

        ('trigg_x_y_thr_num_of_passed_phi_range_clue', 'integer NOT NULL'),

        ('trigg_gtu_x_hough__pixel_peak_gap', 'integer NOT NULL'),
        ('trigg_gtu_x_hough__pi_norm_peak_gap', 'real NOT NULL'),
        ('trigg_gtu_x_hough__peak_thr1', 'real NOT NULL'),
        ('trigg_gtu_x_hough__peak_thr2', 'real NOT NULL'),
        ('trigg_gtu_x_hough__peak_thr3', 'real NOT NULL'),

        ('trigg_gtu_x_thr_num_of_passed_phi_range_clue', 'integer NOT NULL'),

        ('trigg_gtu_y_hough__pixel_peak_gap', 'integer NOT NULL'),
        ('trigg_gtu_y_hough__pi_norm_peak_gap', 'real NOT NULL'),
        ('trigg_gtu_y_hough__peak_thr1', 'real NOT NULL'),
        ('trigg_gtu_y_hough__peak_thr2', 'real NOT NULL'),
        ('trigg_gtu_y_hough__peak_thr3', 'real NOT NULL'),

        ('trigg_gtu_y_thr_num_of_passed_phi_range_clue', 'integer NOT NULL'),

        ('x_y_trigger_neighbour_selection_rules', 'text NOT NULL'),
        ('x_y_trigger_neighbour_selection_zero_value_seeds', 'boolean NOT NULL'),
        ('x_y_trigger_neighbour_selection_pixel_value_border', 'integer NOT NULL'),
        ('x_y_trigger_neighbour_selection_zero_values_in_mean', 'integer NOT NULL'),

        ('x_y_hough__min_phi_range_sep', 'real NOT NULL'),

        ('x_y_hough__line_thickness', 'real NOT NULL'),
        ('x_y_hough__phi_num_steps', 'integer NOT NULL'),
        ('x_y_hough__rho_step', 'real NOT NULL'),

        ('x_y_counts_weight_from_trig__enabled', 'boolean NOT NULL'),
        ('x_y_counts_weight_from_trig__gauss_sigma', 'real NOT NULL'),

        ('x_y_hough__pixel_peak_gap', 'integer NOT NULL'),
        ('x_y_hough__pi_norm_peak_gap', 'real NOT NULL'),
        ('x_y_hough__peak_thr1', 'real NOT NULL'),
        ('x_y_hough__peak_thr2', 'real NOT NULL'),
        ('x_y_hough__peak_thr3', 'real NOT NULL'),
        ('x_y_hough__peak_thr4', 'real NOT NULL'),

        ('gtu_x_trigger_neighbour_selection_rules', 'text NOT NULL'),
        ('gtu_x_trigger_neighbour_selection_zero_value_seeds', 'boolean NOT NULL'),
        ('gtu_x_trigger_neighbour_selection_pixel_value_border', 'integer NOT NULL'),
        ('gtu_x_trigger_neighbour_selection_zero_values_in_mean', 'integer NOT NULL'),

        ('gtu_x_hough__min_phi_range_sep', 'real NOT NULL'),

        ('gtu_x_hough__line_thickness', 'real NOT NULL'),
        ('gtu_x_hough__phi_num_steps', 'integer NOT NULL'),
        ('gtu_x_hough__rho_step', 'real NOT NULL'),

        ('gtu_x_counts_weight_map_gauss__enabled', 'boolean NOT NULL'),
        ('gtu_x_counts_weight_map_gauss__gauss_sigma', 'real NOT NULL'),

        ('gtu_x_hough__pixel_peak_gap', 'integer NOT NULL'),
        ('gtu_x_hough__pi_norm_peak_gap', 'real NOT NULL'),
        ('gtu_x_hough__peak_thr1', 'real NOT NULL'),
        ('gtu_x_hough__peak_thr2', 'real NOT NULL'),
        ('gtu_x_hough__peak_thr3', 'real NOT NULL'),
        ('gtu_x_hough__peak_thr4', 'real NOT NULL'),

        ('gtu_y_trigger_neighbour_selection_rules', 'text NOT NULL'),
        ('gtu_y_trigger_neighbour_selection_zero_value_seeds', 'boolean NOT NULL'),
        ('gtu_y_trigger_neighbour_selection_pixel_value_border', 'integer NOT NULL'),
        ('gtu_y_trigger_neighbour_selection_zero_values_in_mean', 'integer NOT NULL'),

        ('gtu_y_hough__min_phi_range_sep', 'real NOT NULL'),

        ('gtu_y_hough__line_thickness', 'real NOT NULL'),
        ('gtu_y_hough__phi_num_steps', 'integer NOT NULL'),
        ('gtu_y_hough__rho_step', 'real NOT NULL'),

        ('gtu_y_counts_weight_from_trig__enabled', 'boolean NOT NULL'),
        ('gtu_y_counts_weight_from_trig__gauss_sigma', 'real NOT NULL'),

        ('gtu_y_hough__pixel_peak_gap', 'integer NOT NULL'),
        ('gtu_y_hough__pi_norm_peak_gap', 'real NOT NULL'),
        ('gtu_y_hough__peak_thr1', 'real NOT NULL'),
        ('gtu_y_hough__peak_thr2', 'real NOT NULL'),
        ('gtu_y_hough__peak_thr3', 'real NOT NULL'),
        ('gtu_y_hough__peak_thr4', 'real NOT NULL'),

        ('dbscan_eps', 'real NOT NULL'),
        ('dbscan_min_samples', 'integer NOT NULL'),
    ]


    data_table_columns = [
        ('{data_table_program_version_column}', 'real NOT NULL'),
        ('{data_table_source_file_acquisition_column}', 'text NOT NULL'),
        ('{data_table_source_file_trigger_column}', 'text NOT NULL'),
        ('{data_table_source_file_acquisition_full_column}', 'text NOT NULL'),
        ('{data_table_source_file_trigger_full_column}', 'text NOT NULL'),
        ('{data_table_global_gtu_column}', 'integer NOT NULL'),
        ('{data_table_source_data_type_num_column}', 'integer'),                                                # can be NULL
        ('packet_id', 'integer NOT NULL'),
        ('gtu_in_packet', 'integer NOT NULL'),
        ('num_gtu', 'integer NOT NULL'),
        ('num_triggered_pixels', 'integer NOT NULL'),                                                # wasn't NULL ----v
        ('max_trg_box_per_gtu', 'real NOT NULL'),
        ('sum_trg_box_per_gtu', 'real NOT NULL'),
        ('avg_trg_box_per_gtu', 'real NOT NULL'),
        ('max_trg_pmt_per_gtu', 'real NOT NULL'),
        ('sum_trg_pmt_per_gtu', 'real NOT NULL'),
        ('avg_trg_pmt_per_gtu', 'real NOT NULL'),
        ('max_trg_ec_per_gtu', 'real NOT NULL'),
        ('sum_trg_ec_per_gtu', 'real NOT NULL'),
        ('avg_trg_ec_per_gtu', 'real NOT NULL'),
        ('max_n_persist', 'real NOT NULL'), # integer?
        ('sum_n_persist', 'real NOT NULL'),
        ('avg_n_persist', 'real NOT NULL'),
        ('max_sum_l1_pdm', 'real NOT NULL'),
        ('sum_sum_l1_pdm', 'real NOT NULL'),
        ('avg_sum_l1_pdm', 'real NOT NULL'),
    ]

    data_table_columns_per_dimension = [
        ('trigg_{dims}_hough__peak_thr1__line_dbscan_num_clusters', 'integer NOT NULL'),     # TODO
        ('trigg_{dims}_hough__peak_thr2__line_dbscan_num_clusters', 'integer NOT NULL'),
        ('trigg_{dims}_hough__max_peak_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__max_peak_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__major_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__major_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__major_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__major_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr3__major_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr3__major_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr3__avg_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr3__avg_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_major_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_major_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_major_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_major_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr3__max_clu_major_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr3__max_clu_major_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_avg_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_avg_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_avg_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_avg_phi', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr3__max_clu_avg_rho', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr3__max_clu_avg_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__line_dbscan_num_clusters', 'real NOT NULL'),      # TODO
        ('{dims}_hough__peak_thr2__line_dbscan_num_clusters', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__line_dbscan_num_clusters', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__line_dbscan_num_clusters', 'real NOT NULL'),
        ('{dims}_hough__max_peak_rho', 'real NOT NULL'),
        ('{dims}_hough__max_peak_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__major_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__major_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__major_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__major_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__major_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__major_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__major_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__major_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_clu_major_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_clu_major_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_major_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_major_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_major_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_major_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_major_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_major_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_clu_avg_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_clu_avg_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_avg_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_avg_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_avg_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_avg_phi', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_avg_rho', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_avg_phi', 'real NOT NULL'),

        ('{dims}_active_pixels_num', 'integer NOT NULL'),
        ('trigg_{dims}_sum3x3_sum', 'real NOT NULL'),
        ('trigg_{dims}_sum3x3_norm_sum', 'real NOT NULL'),
        ('trigg_{dims}_sum3x3_avg', 'real NOT NULL'),
        ('trigg_{dims}_groups_num', 'real NOT NULL'),
        ('trigg_{dims}_groups_max_size', 'real NOT NULL'),
        ('trigg_{dims}_groups_avg_size', 'real NOT NULL'),
        ('trigg_{dims}_groups_sum_sum_sum3x3', 'real NOT NULL'),
        ('trigg_{dims}_groups_max_sum_sum3x3', 'real NOT NULL'),
        ('trigg_{dims}_groups_avg_sum_sum3x3', 'real NOT NULL'),
        ('trigg_{dims}_hough__max_peak_line_rot', 'real NOT NULL'),
        ('trigg_{dims}_hough__max_peak_line_coord_0_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__max_peak_line_coord_0_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__max_peak_line_coord_1_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__max_peak_line_coord_1_y', 'real NOT NULL'),
        ('{dims}_trigger_neighbourhood_size', 'real NOT NULL'),
        ('{dims}_trigger_neighbourhood_width', 'real NOT NULL'),
        ('{dims}_trigger_neighbourhood_height', 'real NOT NULL'),
        ('{dims}_trigger_neighbourhood_area', 'real NOT NULL'),
        ('{dims}_trigger_neighbourhood_counts_sum', 'real NOT NULL'),
        ('{dims}_trigger_neighbourhood_counts_avg', 'real NOT NULL'),
        ('{dims}_trigger_neighbourhood_counts_norm_sum', 'real NOT NULL'),
        ('{dims}_hough__max_peak_line_rot', 'real NOT NULL'),
        ('{dims}_hough__max_peak_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__max_peak_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__max_peak_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__max_peak_line_coord_1_y', 'real NOT NULL'),

        ('trigg_{dims}_hough__peak_thr1__num_clusters', 'integer NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_cluster_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_cluster_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_area', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_cluster_area', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_size', 'integer NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_cluster_size', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_counts_sum', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_cluster_counts_sum', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_area_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_area_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_size_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_size_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_counts_sum_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_cluster_counts_sum_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_peak_cluster_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_peak_cluster_height', 'real NOT NULL'),
        # ('trigg_{dims}_hough__peak_thr1__avg_line_rot', 'real NOT NULL'),

        ('trigg_{dims}_hough__peak_thr1__major_line_coord_0_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__major_line_coord_0_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__major_line_coord_1_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__major_line_coord_1_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_line_coord_0_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_line_coord_0_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_line_coord_1_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__avg_line_coord_1_y', 'real NOT NULL'),
        # ('trigg_{dims}_hough__peak_thr1__max_clu_line_rot', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_major_line_coord_0_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_major_line_coord_0_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_major_line_coord_1_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_major_line_coord_1_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_avg_line_coord_0_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_avg_line_coord_0_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_avg_line_coord_1_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr1__max_clu_avg_line_coord_1_y', 'real NOT NULL'),

        ('trigg_{dims}_hough__peak_thr2__num_clusters', 'integer NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_cluster_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_cluster_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_area', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_cluster_area', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_size', 'integer NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_cluster_size', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_counts_sum', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_cluster_counts_sum', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_area_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_area_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_size_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_size_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_counts_sum_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_cluster_counts_sum_height', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_peak_cluster_width', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_peak_cluster_height', 'real NOT NULL'),
        # ('trigg_{dims}_hough__peak_thr2__avg_line_rot', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__major_line_coord_0_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__major_line_coord_0_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__major_line_coord_1_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__major_line_coord_1_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_line_coord_0_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_line_coord_0_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_line_coord_1_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__avg_line_coord_1_y', 'real NOT NULL'),
        # ('trigg_{dims}_hough__peak_thr2__max_clu_line_rot', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_major_line_coord_0_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_major_line_coord_0_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_major_line_coord_1_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_major_line_coord_1_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_avg_line_coord_0_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_avg_line_coord_0_y', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_avg_line_coord_1_x', 'real NOT NULL'),
        ('trigg_{dims}_hough__peak_thr2__max_clu_avg_line_coord_1_y', 'real NOT NULL'),

        ('{dims}_hough__peak_thr1__num_clusters', 'integer NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_cluster_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_area', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_cluster_area', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_size', 'integer NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_cluster_size', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_counts_sum', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_cluster_counts_sum', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_area_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_area_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_size_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_size_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_counts_sum_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_cluster_counts_sum_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_peak_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_peak_cluster_height', 'real NOT NULL'),
        # '{dims}_hough__peak_thr1__avg_line_rot', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__avg_line_coord_1_y', 'real NOT NULL'),
        # ('{dims}_hough__peak_thr1__max_clu_line_rot', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_clu_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_clu_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_clu_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr1__max_clu_line_coord_1_y', 'real NOT NULL'),

        ('{dims}_hough__peak_thr2__num_clusters', 'integer NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_cluster_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_area', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_cluster_area', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_size', 'integer NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_cluster_size', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_counts_sum', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_cluster_counts_sum', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_area_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_area_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_size_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_size_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_counts_sum_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_cluster_counts_sum_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_peak_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_peak_cluster_height', 'real NOT NULL'),
        # ('{dims}_hough__peak_thr2__avg_line_rot', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__major_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__major_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__major_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__major_line_coord_1_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__avg_line_coord_1_y', 'real NOT NULL'),
        # ('{dims}_hough__peak_thr2__max_clu_line_rot', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_major_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_major_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_major_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_major_line_coord_1_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_avg_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_avg_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_avg_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr2__max_clu_avg_line_coord_1_y', 'real NOT NULL'),

        ('{dims}_hough__peak_thr3__num_clusters', 'integer NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_cluster_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_area', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_cluster_area', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_size', 'integer NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_cluster_size', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_counts_sum', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_cluster_counts_sum', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_area_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_area_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_size_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_size_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_counts_sum_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_cluster_counts_sum_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_peak_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_peak_cluster_height', 'real NOT NULL'),
        # ('{dims}_hough__peak_thr3__avg_line_rot', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__major_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__major_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__major_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__major_line_coord_1_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__avg_line_coord_1_y', 'real NOT NULL'),
        # ('{dims}_hough__peak_thr3__max_clu_line_rot', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_major_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_major_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_major_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_major_line_coord_1_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_avg_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_avg_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_avg_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr3__max_clu_avg_line_coord_1_y', 'real NOT NULL'),

        ('{dims}_hough__peak_thr4__num_clusters', 'integer NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_cluster_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_area', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_cluster_area', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_size', 'integer NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_cluster_size', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_counts_sum', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_cluster_counts_sum', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_area_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_area_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_size_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_size_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_counts_sum_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_cluster_counts_sum_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_peak_cluster_width', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_peak_cluster_height', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__major_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__major_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__major_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__major_line_coord_1_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__avg_line_coord_1_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_major_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_major_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_major_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_major_line_coord_1_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_avg_line_coord_0_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_avg_line_coord_0_y', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_avg_line_coord_1_x', 'real NOT NULL'),
        ('{dims}_hough__peak_thr4__max_clu_avg_line_coord_1_y', 'real NOT NULL'),
    ]