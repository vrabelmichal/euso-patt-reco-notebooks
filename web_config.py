import os.path as op
base_image_storage_directory = op.realpath(op.join(op.dirname(__file__),'../event_figures'))
base_image_web_static_path = 'event_figures'
figure_name_format =    '{program_version:.2f}' \
                        '/{triggered_pixels_group_max_gap:d}_{triggered_pixels_ht_line_thickness:.2f}_{triggered_pixels_ht_phi_num_steps:d}_{x_y_neighbour_selection_rules}' \
                        '_{x_y_ht_line_thickness:.2f}_{x_y_ht_phi_num_steps:d}_{x_y_ht_rho_step:.2f}_{x_y_ht_peak_threshold_frac_of_max:.2f}_{x_y_ht_peak_gap:d}' \
                        '_{x_y_ht_global_peak_threshold_frac_of_max:.2f}' \
                        '/{acquisition_file_basename}/{kenji_l1trigger_file_basename}' \
                        '/{gtu_global:d}_{packet_id:d}_{gtu_in_packet:d}/{name}.png' 

packet_size = 128