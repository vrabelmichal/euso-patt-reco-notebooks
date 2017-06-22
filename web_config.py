import os.path as op
base_image_storage_directory = op.realpath(op.join(op.dirname(__file__),'../event_figures'))
base_image_web_static_path = 'event_figures'
figure_name_format = "{program_version}" \
                    "/{triggered_pixels_group_max_gap}_{triggered_pixels_ht_line_thickness}_{triggered_pixels_ht_phi_num_steps}_{x_y_neighbour_selection_rules}" \
                    "_{x_y_ht_line_thickness}_{x_y_ht_phi_num_steps}_{x_y_ht_rho_step}_{x_y_ht_peak_threshold_frac_of_max}_{x_y_ht_peak_gap}" \
                    "_{x_y_ht_global_peak_threshold_frac_of_max}" \
                    "/{acquisition_file_basename}/{kenji_l1trigger_file_basename}" \
                    "/{gtu_global}_{packet_id}_{gtu_in_packet}/{name}.png"

packet_size = 128