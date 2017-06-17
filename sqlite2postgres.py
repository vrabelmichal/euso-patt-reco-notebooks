import sys
import os
import argparse
import postgresql_event_storage
import sqlite_event_storage
import event_processing

def main(argv):
    parser = argparse.ArgumentParser(description='Sqlite3 to PostgreSql')
    parser.add_argument('-i', '--input-files', nargs='+')
    parser.add_argument('-o', '--output')

    args = parser.parse_args(argv)

    postgresql_storage = postgresql_event_storage.PostgreSqlEventStorageProvider()
    postgresql_storage.initialize(args.output)

    for input_file_path in args.input_files:


        sqlite_storage = sqlite_event_storage.Sqlite3EventStorageProvider()
        sqlite_storage.config_info_table_name = "spb_run_info"
        sqlite_storage.config_info_table_pk = "run_info_id"

        sqlite_storage.data_table_name = "spb_trigger_event"
        sqlite_storage.data_table_pk = 'processed_event_id'

        sqlite_storage.initialize(input_file_path, True)

        config_info_records = sqlite_storage.fetch_config_info_records()
        trigger_analysis_records = sqlite_storage.fetch_trigger_analysis_records()

        config_info_record_mapping = {}

        for config_info_record in config_info_records:
            config_info_record_mapping[config_info_record.id] = \
                postgresql_storage.save_config_info(config_info_record)
            if config_info_record.id != config_info_record_mapping[config_info_record.id]:
                print("Inserted config info: "+str(config_info_record))

        for trigger_analysis_record in trigger_analysis_records:
            if postgresql_storage.check_event_exists_weak(trigger_analysis_record, event_processing.program_version):
                postgresql_storage.save_event(trigger_analysis_record)
                pgrint("Trigger analysis record saved ({} ; {})".format(trigger_analysis_record.source_file_acquisition,
                                                                       trigger_analysis_record.source_file_trigger))

        sqlite_storage.finalize()

    postgresql_storage.finalize()

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
