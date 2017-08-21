import sys
import os
import getpass
import argparse

app_base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

if app_base_dir not in sys.path:
    sys.path.append(app_base_dir)

import collections
import numpy as np
import psycopg2 as pg
import pandas as pd
import pandas.io.sql as psql
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['figure.dpi'] = 150
import matplotlib.pyplot as plt
import npy_l1_event_reader
import tool.npy_frames_visualization as npy_vis


def vis_triggers(ev_npy_pathname, tbgf05_pathname=None, tbgf075_pathname=None, tbgf10_pathname=None):

#     ev_npy_frames = np.load(ev_npy_pathname)
#     ev_integrated = np.maximum.reduce(ev_npy_frames[128:])

    fig, axs = plt.subplots(1,bool(tbgf05_pathname)+bool(tbgf075_pathname)+bool(tbgf10_pathname))
    fig.set_size_inches(15,5)

    bgfactors = [("bgfactor 0.5", tbgf05_pathname), ("bgfactor 0.75", tbgf075_pathname), ("bgfactor 1.0", tbgf10_pathname)]
    
    i=0
    for title, trigger_pathname in bgfactors:
        if trigger_pathname:
            print(title)
            with npy_l1_event_reader.NpyL1EventReader(ev_npy_pathname, trigger_pathname) as event_reader:
#                 ev_integrated = np.maximum.reduce(ev_npy_frames[128:]) 
                ev_integrated = np.maximum.reduce(event_reader._current_acquisition_ndarray[128:]) 
                l1trg_events = []
                for gtu_pdm_data in event_reader.iter_gtu_pdm_data():
                    if len(gtu_pdm_data.l1trg_events) > 0:
                        print(gtu_pdm_data.gtu)
                        for l1trg_ev in gtu_pdm_data.l1trg_events:
                            print("  {} {}".format(l1trg_ev.pix_col, l1trg_ev.pix_row))
                        l1trg_events += gtu_pdm_data.l1trg_events
                npy_vis.visualize_frame(ev_integrated, event_reader.exp_tree, l1trg_events, title=title, ax=axs[i], show=False)
                i += 1


    plt.show()
    
# vis_triggers(
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_24000000.00/energy_6.85e+12/simu.2017-07-22-19h51m20s/npyconv/ev_66_mc_1__signals_p128_a0_g30_f128_b20170427-105115-001.001_k1_s0_d32_n27_m128.npy",
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_24000000.00/energy_6.85e+12/simu.2017-07-22-19h51m20s/l1_tigger_kenji/trn_ev_66_mc_1__signals_p128_a0_g30_f128_b20170427-105115-001.001_k1_s0_d32_n27_m128_ec_asc_bgf_0.50.root",
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_24000000.00/energy_6.85e+12/simu.2017-07-22-19h51m20s/l1_tigger_kenji/trn_ev_66_mc_1__signals_p128_a0_g30_f128_b20170427-105115-001.001_k1_s0_d32_n27_m128_ec_asc_bgf_0.75.root",
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_24000000.00/energy_6.85e+12/simu.2017-07-22-19h51m20s/l1_tigger_kenji/trn_ev_66_mc_1__signals_p128_a0_g30_f128_b20170427-105115-001.001_k1_s0_d32_n27_m128_ec_asc_bgf_1.00.root"
# );


# vis_triggers(
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_24000000.00/energy_6.85e+12/simu.2017-07-23-00h14m11s/npyconv/ev_82_mc_1__signals_p128_a0_g30_f128_b20170427-105115-001.001_k1_s0_d32_n42_m128.npy",
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_24000000.00/energy_6.85e+12/simu.2017-07-23-00h14m11s/l1_tigger_kenji/trn_ev_82_mc_1__signals_p128_a0_g30_f128_b20170427-105115-001.001_k1_s0_d32_n42_m128_ec_asc_bgf_0.50.root",
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_24000000.00/energy_6.85e+12/simu.2017-07-23-00h14m11s/l1_tigger_kenji/trn_ev_82_mc_1__signals_p128_a0_g30_f128_b20170427-105115-001.001_k1_s0_d32_n42_m128_ec_asc_bgf_0.75.root",
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_24000000.00/energy_6.85e+12/simu.2017-07-23-00h14m11s/l1_tigger_kenji/trn_ev_82_mc_1__signals_p128_a0_g30_f128_b20170427-105115-001.001_k1_s0_d32_n42_m128_ec_asc_bgf_1.00.root"
# )

# vis_triggers(
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_30000000.00/energy_5.01e+12/thousnd30E14.2017-07-25-09h58m06s/npyconv/ev_80_mc_1__signals_p128_a0_g30_f128_b20170429-070031-019.001_k1_s0_d32_n32_m128.npy",
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_30000000.00/energy_5.01e+12/thousnd30E14.2017-07-25-09h58m06s/l1_tigger_kenji/trn_ev_80_mc_1__signals_p128_a0_g30_f128_b20170429-070031-019.001_k1_s0_d32_n32_m128_ec_asc_bgf_0.50.root",
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_30000000.00/energy_5.01e+12/thousnd30E14.2017-07-25-09h58m06s/l1_tigger_kenji/trn_ev_80_mc_1__signals_p128_a0_g30_f128_b20170429-070031-019.001_k1_s0_d32_n32_m128_ec_asc_bgf_0.75.root",
#     "/home/eusobg/EUSO-SPB/SPBDATA_processed/spb_simu/posz_30000000.00/energy_5.01e+12/thousnd30E14.2017-07-25-09h58m06s/l1_tigger_kenji/trn_ev_80_mc_1__signals_p128_a0_g30_f128_b20170429-070031-019.001_k1_s0_d32_n32_m128_ec_asc_bgf_1.00.root"
# )


# q = '''SELECT  dt2.event_id, dt2.source_file_acquisition_full, num_triggered_pixels, simu_event.edetector_numfee, simu_event.etruth_truetheta /*, etruth_trueenergy*/ 
# FROM simu_event_spb_proc JOIN simu_event USING(simu_event_id) JOIN spb_processing_event_ver2 AS dt2 USING(event_id) /*JOIN spb_processing_event_ver2 AS dt2 USING(source_file_acquisition_full) 
#     WHERE dt1.source_data_type_num=3 */  WHERE /*num_triggered_pixels = 3  AND*/ dt2.source_data_type_num=3 AND/**/ /*edetector_numfee > 35000 AND  AND*/ etruth_truetheta < 0.0872665 OFFSET 0 LIMIT 140;'''

def visualize_events(con, outdir='.', max_pages=100, custom_query=None, show=False):
    # con.rollback()
    outdir = os.path.realpath(outdir)
    os.makedirs(outdir, exist_ok=True)
    cur = con.cursor()
    visualized_pages = 0
    while visualized_pages < max_pages:
        if not custom_query:
            q = '''SELECT  
            t1.event_id, t1.source_file_acquisition_full,
            t1.packet_id, t1.gtu_in_packet, t1.num_gtu, 
            t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width, 
            t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width,
            t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width,
            t1.sum_n_persist,
            num_triggered_pixels, simu_event.edetector_numfee, simu_event.etruth_truetheta /*, etruth_trueenergy*/ 
            FROM simu_event_spb_proc JOIN simu_event USING(simu_event_id) 
            JOIN spb_processing_event_ver2 AS t1 USING(event_id) 
                WHERE 
                /*etruth_truetheta < 0.0872665 
                AND */ t1.trigg_x_y_hough__peak_thr1__max_cluster_counts_sum_width < 75 
                AND t1.trigg_gtu_x_hough__peak_thr1__max_cluster_counts_sum_width < 31 
                AND t1.trigg_gtu_y_hough__peak_thr1__max_cluster_counts_sum_width < 31
                AND num_triggered_pixels > 10
                AND t1.source_data_type_num = 3
                AND gtu_in_packet >= 30
                AND simu_event.edetector_numfee > 15000
                
                AND trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 5
                /*AND sum_n_persist < 25*/
                
            ORDER BY num_triggered_pixels ASC, t1.event_id ASC 
            OFFSET {offset} LIMIT {limit};'''

            # sum_trg_ec_per_gtu < 10 /* ??? */
            # sum_n_persist < 25 
            # trigg_x_y_hough__dbscan_num_clusters_above_thr1 < 5

            # TODO try to consider num triggered in the simulation
        else:
            q = custom_query

        offset = visualized_pages*160
        limit = 160

        q = q.format(offset=offset, limit=limit)

        cur.execute(q)
        
        all_entries = cur.fetchall()

        if not all_entries:
            break

        fig, axs = plt.subplots(10, 16)
        axs_flattned = axs.flatten()
        fig.set_size_inches(16*80/16, 10*20/4)

        print("-"*20)
        print("PAGE {}".format(visualized_pages+1))
        print("-"*20)

        for i, r in enumerate(all_entries):
            event_id, source_file_acquisition_full, packet_id, gtu_in_packet, num_gtu, xy_peak_w, gtux_peak_w, gtuy_peak_w, sum_n_persist, num_triggered_pixels, edetector_numfee, etruth_etruetheta = r
            #print(i, source_file_acquisition_full, num_triggered_pixels, edetector_numfee, etruth_etruetheta)
            sys.stdout.write("#{} (ID {}, pck {}, gtu_in_pck {}, {} gtu, {} trig, {} fee, {} deg, xy_peak_w {}, gtux_peak_w {}, gtuy_peak_w {}, sum_n_persist {})\n".format(
                i, event_id, packet_id, gtu_in_packet, num_gtu, num_triggered_pixels, edetector_numfee, np.rad2deg(etruth_etruetheta), 
                            xy_peak_w, gtux_peak_w, gtuy_peak_w, sum_n_persist))
            sys.stdout.flush()
            frames = np.load(source_file_acquisition_full)
            ev_integrated = np.maximum.reduce(frames[packet_id*128+gtu_in_packet:packet_id*128+gtu_in_packet+num_gtu]) 
            npy_vis.visualize_frame(ev_integrated, ax=axs_flattned[i], title="#{} ID {}, {} fee, {:.1f} deg".format(i, event_id, edetector_numfee,np.rad2deg(etruth_etruetheta)), show=False)
            
            if i==limit-1:
                break

        if show:
            plt.show()
        else:
            fname = "events_{}_{}.png".format(offset, limit)
            plt.savefig(os.path.join(outdir, fname))
        
        visualized_pages += 1


def main(argv):
    parser = argparse.ArgumentParser(description='Draw projections of x - y dimensionsdscdf')
    parser.add_argument('-d','--dbname',default='eusospb_data')
    parser.add_argument('-U','--user',default='eusospb')
    parser.add_argument('--password')
    parser.add_argument('-s','--host',default='localhost')
    parser.add_argument('-o', '--odir', default='.')
    parser.add_argument('--max-pages', type=int, default=100)
    parser.add_argument('--custom-query', default='')

    args = parser.parse_args(argv)

    if not args.password:
        args.password = getpass.getpass()

    con = pg.connect(dbname=args.dbname, user=args.user, password=args.password, host=args.host)

    visualize_events(con, args.odir, args.max_pages, args.custom_query, False)


if __name__ == '__main__':
    main(sys.argv[1:])

