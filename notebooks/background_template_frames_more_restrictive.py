
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import psycopg2 as pg
import pandas as pd
import pandas.io.sql as psql
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['figure.dpi'] = 150
import matplotlib.pyplot as plt
import numpy as np
import tool.acqconv
from tqdm import tqdm

if len(sys.argv) < 3:
	sys.exit(1)

active_pixels_restriction = 2304 - 64

con = pg.connect("dbname=eusospb_data user=eusospb password=1e20eVjemeuso host=localhost")

dataframe_9ec = psql.read_sql(
    "SELECT " \
    "event_id, global_gtu, gtu_in_packet, packet_id, num_triggered_pixels, source_file_acquisition_full, x_y_active_pixels_num "\
    "FROM spb_processing_event_ver2 " \
    "WHERE source_file_acquisition LIKE 'allpackets-SPBEUSO-ACQUISITION-2017%' AND source_file_acquisition NOT LIKE '%sqz.root' AND " \
    " x_y_active_pixels_num > {restriction} ORDER BY x_y_active_pixels_num DESC LIMIT {limit:d} OFFSET {offset:d}".format(limit=int(sys.argv[1]),offset=int(sys.argv[2]), restriction=active_pixels_restriction),con)

#dataframe_9ec.head(100)

i=0
    
background_template_event_ids = []

plot_displayed = False

with open('bg_tmp_analysis_restriction_{}_log_limit_{}_offset_{}.txt'.format(active_pixels_restriction,sys.argv[1],sys.argv[2]),"w") as logfile:
    i=0
    ax_i = 0
    for idx, row in tqdm(dataframe_9ec.iterrows()):
        frames = tool.acqconv.get_frames(row.source_file_acquisition_full, 128*row.packet_id, 128*row.packet_id+128, entry_is_gtu_optimization=True)
        max_integrated = np.maximum.reduce(frames)
        max_integrated_32 = np.maximum.reduce(frames[:32])

        #non_zero_in_max_integrated = np.count_nonzero(max_integrated)
        non_zero_in_max_integrated_32 = np.count_nonzero(max_integrated_32)

        if non_zero_in_max_integrated_32 > active_pixels_restriction:
            background_template_event_ids.append(row.event_id)

        i += 1
            
        if i % 1000 == 0:
                
            with open('bg_tmp_analysis_restriction_{}_results_limit_{}_offset_{}.txt'.format(active_pixels_restriction, sys.argv[1],sys.argv[2]),'w') as intermediate_result:
                intermediate_result.write("\n".join([str(evid) for evid in background_template_event_ids]))
            logmsg = "{}\t{}\t{}\n".format(i, len(background_template_event_ids), row.source_file_acquisition_full)
            logfile.write(logmsg)
            logfile.flush()
            print("Log: "+logmsg)
            
  
with open('bg_tmp_analysis_restriction_{}_results_limit_{}_offset_{}.txt'.format(active_pixels_restriction, sys.argv[1],sys.argv[2]),'w') as intermediate_result:
    intermediate_result.write("\n".join([str(evid) for evid in background_template_event_ids]))          

print("done")
