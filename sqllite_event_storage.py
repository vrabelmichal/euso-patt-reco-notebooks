import sqlite3
import time

from base_classes import *


class SqlLite3EventStorageProvider(EventStorageProvider):
    _conn = None
    _cursor = None

    run_timestamp = None

    run_info_table_name = 'spb_run_info'
    data_table_name = 'spb_trigger_event'

    run_info_table_columns = {
        'run_info_id': 'integer PRIMARY KEY',
        'triggered_pixels_group_max_gap': 'integer NOT NULL',

        'triggered_pixels_ht_size': 'integer NOT NULL',
        'triggered_pixels_ht_phi_num_steps': 'integer NOT NULL',
        'triggered_pixels_ht_rho_step': 'real NOT NULL',

        'x_y_neighbour_selection_rules': 'text NOT NULL',

        'x_y_ht_size': 'integer NOT NULL',
        'x_y_ht_phi_num_steps': 'integer NOT NULL',
        'x_y_ht_rho_step': 'real NOT NULL',

        'x_y_ht_peak_threshold_frac_of_max': 'real NOT NULL',
        'x_y_ht_peak_gap': 'real NOT NULL',
        'x_y_ht_global_peak_threshold_frac_of_max': 'real NOT NULL',
    }

    data_table_columns = {
        'processed_event_id': 'integer PRIMARY KEY',
        'run_info_id': 'integer',
        'run_timestamp': 'real',
        'source_file_acquisition': 'text NOT NULL',
        'source_file_trigger': 'text NOT NULL',
        'global_gtu': 'integer NOT NULL',
        'packet_id': 'integer NOT NULL',
        'gtu_in_packet': 'integer NOT NULL',
        'num_gtu': 'integer NOT NULL',
        'num_triggered_pixels': 'integer',
        'num_triggered_pixel_groups': 'integer',
        'max_triggered_pixel_group_size': 'integer',
        'avg_trigger_group_size': 'real',
    }

    data_table_columns_hough = {
        'triggered_pixels_{dims}_hough_transform__max_peak_rho': 'real',
        'triggered_pixels_{dims}_hough_transform__max_peak_phi': 'real',
        'triggered_pixels_{dims}_hough_transform__max_peak_line_rot': 'real',
        'triggered_pixels_{dims}_hough_transform__max_peak_line_coord_0_x': 'real',
        'triggered_pixels_{dims}_hough_transform__max_peak_line_coord_0_y': 'real',
        'triggered_pixels_{dims}_hough_transform__max_peak_line_coord_1_x': 'real',
        'triggered_pixels_{dims}_hough_transform__max_peak_line_coord_1_y': 'real',
        'triggers_{dims}_neighbourhood_width': 'real',
        'triggers_{dims}_neighbourhood_height': 'real',
        'triggers_{dims}_neighbourhood_size': 'integer',
        'hough_transform_{dims}__num_clusters_above_thr': 'integer',
        'hough_transform_{dims}__max_cluster_width': 'integer',
        'hough_transform_{dims}__avg_cluster_width': 'real',
        'hough_transform_{dims}__max_cluster_height': 'real',
        'hough_transform_{dims}__avg_cluster_height': 'real',
        'hough_transform_{dims}__max_cluster_area': 'real',
        'hough_transform_{dims}__avg_cluster_area': 'real',
        'hough_transform_{dims}__max_cluster_size': 'real',
        'hough_transform_{dims}__avg_cluster_size': 'real',
        'hough_transform_{dims}__max_cluster_counts_sum': 'int',
        'hough_transform_{dims}__avg_cluster_counts_sum': 'real',
        'hough_transform_{dims}__clusters_above_thr': 'integer',
        'hough_transform_{dims}__max_peak_rho': 'real',
        'hough_transform_{dims}__max_peak_phi': 'real',
        'hough_transform_{dims}__max_peak_line_rot': 'real',
        'hough_transform_{dims}__max_peak_line_coord_0_x': 'real',
        'hough_transform_{dims}__max_peak_line_coord_0_y': 'real',
        'hough_transform_{dims}__max_peak_line_coord_1_x': 'real',
        'hough_transform_{dims}__max_peak_line_coord_1_y': 'real',
        'hough_transform_{dims}__thr_peak_rho': 'real',
        'hough_transform_{dims}__thr_peak_phi': 'real',
        'hough_transform_{dims}__thr_peak_line_rot': 'real',
        'hough_transform_{dims}__thr_peak_line_coord_0_x': 'real',
        'hough_transform_{dims}__thr_peak_line_coord_0_y': 'real',
        'hough_transform_{dims}__thr_peak_line_coord_1_x': 'real',
        'hough_transform_{dims}__thr_peak_line_coord_1_y': 'real',
    }

    all_hough_transform_dims = ["x_y", "x_gtu", "y_gtu"]
    active_hough_transform_dims = ["x_y"]

    def initialize(self, output_spec=None):
        self._conn = sqlite3.connect(output_spec, isolation_level=None)
        self._cursor = self._conn.cursor()
        self.run_timestamp = time.time()

        create_info_table_query = 'CREATE TABLE IF NOT EXISTS {table_name} ({columns}, UNIQUE ({uniq_columns}) )' \
            .format(table_name=self.run_info_table_name,
                    columns=', '.join(["{} {}".format(c, t) for c, t in self.run_info_table_columns.items()]),
                    uniq_columns=', '.join([c for c, t in self.run_info_table_columns.items() if c != 'run_info_id']))

        self._cursor.execute(create_info_table_query)

        all_data_table_columns = dict(self.data_table_columns)

        for dims in self.all_hough_transform_dims:
            for c, t in self.data_table_columns_hough.items():
                all_data_table_columns[c.format(dims=dims)] = t

        create_data_table_query = 'CREATE TABLE IF NOT EXISTS {data_table} ( {data_table_columns}, '  \
                                  'FOREIGN KEY ({run_info_fk}) REFERENCES {run_info_table} ({run_info_id_column}) '  \
                                  'ON DELETE CASCADE ON UPDATE NO ACTION )' \
                                      .format(data_table=self.data_table_name,
                                              data_table_columns=', '.join(
                                                  ["{} {}".format(c, t) for c, t in all_data_table_columns.items()]),
                                              run_info_fk='run_info_id',
                                              run_info_table=self.run_info_table_name,
                                              run_info_id_column='run_info_id'
                                              )

        self._cursor.execute(create_data_table_query)

    def save_row(self, trigger_event, save_config_result=None):
        if self._conn is None or self._cursor is None:
            raise Exception("SqlLite3EventStorageProvider has not been initialized")
        if not isinstance(save_config_result, int) or save_config_result < 0:
            raise Exception("save_config_result has an unexpected data type or value")


        attrs = [c for c, t in self.data_table_columns.items() if c != 'processed_event_id' and c!='run_info_id' and c!='run_timestamp'] # todo remove hardcoded
        for dims in self.active_hough_transform_dims:
            for col in self.data_table_columns_hough:
                attrs.append(col.format(dims=dims))

        vals = []
        for c in attrs:
            v = getattr(trigger_event, c)
            if isinstance(v,(float,int)):
                vals.append('{}'.format(v))
            elif isinstance(v,(tuple,list)):
                vals.append('"' + ", ".join([str(sv) for sv in v]) + '"' )
            else:
                vals.append('"{}"'.format(v))

        attrs.append("run_info_id")
        vals.append(str(save_config_result))

        attrs.append("run_timestamp")
        vals.append(str(self.run_timestamp))

        q = "INSERT INTO {data_table}({columns}) VALUES({values})" \
            .format(data_table=self.data_table_name, columns=", ".join(attrs), values=", ".join(vals))

        self._cursor.execute(q)

    def save_config_info(self, config_info):  # instance of Proc
        if self._conn is None or self._cursor is None:
            raise Exception("SqlLite3EventStorageProvider has not been initialized")

        attrs = [c for c, t in self.run_info_table_columns.items() if c != 'run_info_id']
            # ['triggered_pixels_group_max_gap',
            #      'triggered_pixels_ht_size',
            #      'triggered_pixels_ht_phi_num_steps',
            #      'triggered_pixels_ht_rho_step',
            #      'x_y_neighbour_selection_rules',
            #      'x_y_ht_size',
            #      'x_y_ht_phi_num_steps',
            #      'x_y_ht_rho_step',
            #      'x_y_ht_peak_threshold_frac_of_max',
            #      'x_y_ht_peak_gap',
            #      'x_y_ht_global_peak_threshold_frac_of_max']

        #vals = [getattr(config_info,attr) for attr in attrs]
        vals =[]
        for c in attrs:
            v = getattr(config_info, c)
            if isinstance(v,(float,int)):
                vals.append('{}'.format(v))
            elif isinstance(v,(tuple,list)):
                vals.append('"' + ", ".join([str(sv) for sv in v]) + '"' )
            else:
                vals.append('"{}"'.format(v))

        try:
            q = "INSERT INTO {run_info_table}({columns}) VALUES({values})" \
                                .format(run_info_table=self.run_info_table_name, columns=", ".join(attrs), values=", ".join(vals))
            self._cursor.execute(q)

            return self._cursor.lastrowid

        except sqlite3.IntegrityError:

            conds = ['{}={}'.format(a,v) for a,v in zip(attrs, vals)]

            q = "SELECT {run_info_id} FROM {run_info_table} WHERE {conds}"\
                                .format(run_info_id='run_info_id', run_info_table=self.run_info_table_name,
                                        conds=" AND ".join(conds))
            self._cursor.execute(q)

            return self._cursor.fetchone()[0]

    def finalize(self):
        self._conn.close()
