from event_reading import *
import fileinput

def main(argv):

    for root_file_pathname in fileinput.input():
        success = True
        root_file_pathname = root_file_pathname.strip()
        try:
            f, t_texp, t_tevent = AcqL1EventReader.open_acquisition(root_file_pathname)
            success = bool(f and t_texp and t_tevent)
        except Exception:
            success = False

        print("{}\t{}".format(root_file_pathname, success))


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
