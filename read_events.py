#!/usr/bin/python3

import numpy as np
import os
import argparse
import sys
import ROOT
import matplotlib.pyplot as plt
from eusotrees.exptree import ExpTree

class GtuPdmData:
    photon_count_data = None
    gtu = -1
    gtu_time = -1
    gtu_time1 = -1

    l1trg_events = []

    def __init__(self, photon_count_data, gtu, gtu_time, gtu_time1):
        self.photon_count_data = photon_count_data
        self.gtu = np.asscalar(gtu) if isinstance(gtu, np.ndarray) else gtu
        self.gtu_time = np.asscalar(gtu_time) if isinstance(gtu_time, np.ndarray) else gtu_time
        self.gtu_time1 = np.asscalar(gtu_time1) if isinstance(gtu_time1, np.ndarray) else gtu_time1


class L1TrgEvent:
    gtu_pdm_data = None
    ec_id = -1
    pmt_row = -1  # should be converted to Lech
    pmt_col = -1  # sholud be converted to Lech
    pix_row = -1
    pix_col = -1
    sum_l1 = -1
    thr_l1 = -1
    persist_l1 = -1
    
    def __init__(self, gtu_pdm_data, ec_id, pmt_row, pmt_col, pix_row, pix_col, sum_l1, thr_l1, persist_l1):
        self.gtu_pdm_data = gtu_pdm_data
        self.ec_id = np.asscalar(ec_id) if isinstance(ec_id, np.ndarray) else ec_id
        self.pmt_row = np.asscalar(pmt_row) if isinstance(pmt_row, np.ndarray) else pmt_row
        self.pmt_col = np.asscalar(pmt_col) if isinstance(pmt_col, np.ndarray) else pmt_col
        self.pix_row = np.asscalar(pix_row) if isinstance(pix_row, np.ndarray) else pix_row
        self.pix_col = np.asscalar(pix_col) if isinstance(pix_col, np.ndarray) else pix_col
        self.sum_l1 = np.asscalar(sum_l1) if isinstance(sum_l1, np.ndarray) else sum_l1
        self.thr_l1 = np.asscalar(thr_l1) if isinstance(thr_l1, np.ndarray) else thr_l1
        self.persist_l1 = np.asscalar(persist_l1) if isinstance(persist_l1, np.ndarray) else persist_l1

    @classmethod
    def from_mario_format(cls, gtu_pdm_data, ec_id, pmt_row, pmt_col, pix_row, pix_col, sum_l1, thr_l1, persist_l1):
        e = L1TrgEvent(gtu_pdm_data, ec_id, pmt_row, pmt_col, pix_row, pix_col, sum_l1, thr_l1, persist_l1)
        e.pmt_col, e.pmt_row, e.pix_col, e.pix_row = cls.mario2pdm(e.pmt_col, e.pmt_row, e.pix_col, e.pix_row)
        return e

    # rewritten from kenji's c++ version
    @classmethod
    def mario2pdm(cls, mpmt_col=0, mpmt_row=0, pix_x=0, pix_y=0):
        abs_pmtx = 0
        abs_pmty = 0

        if mpmt_row >= 18 or mpmt_row < 0:
            raise Exception("Something is rotten in the state of row#. mario")
        
        if mpmt_col >= 2 or mpmt_col < 0:
            raise Exception("Something is rotten in the state of col#. mario")

        # EC1
        if mpmt_col == 0 and mpmt_row == 0:
            abs_pmtx = 0;abs_pmty = 5;
        if mpmt_col == 1 and mpmt_row == 0 :
            abs_pmtx = 1;abs_pmty = 5;
        if mpmt_col == 0 and mpmt_row == 1 :
            abs_pmtx = 0;abs_pmty = 4;
        if mpmt_col == 1 and mpmt_row == 1 :
            abs_pmtx = 1;abs_pmty = 4;

        # EC2
        if mpmt_col == 0 and mpmt_row == 2 :
            abs_pmtx = 2;abs_pmty = 5;
        if mpmt_col == 1 and mpmt_row == 2 :
            abs_pmtx = 3;abs_pmty = 5;
        if mpmt_col == 0 and mpmt_row == 3 :
            abs_pmtx = 2;abs_pmty = 4;
        if mpmt_col == 1 and mpmt_row == 3 :
            abs_pmtx = 3;abs_pmty = 4;

        # EC3
        if mpmt_col == 0 and mpmt_row == 4 :
            abs_pmtx = 4;abs_pmty = 5;
        if mpmt_col == 1 and mpmt_row == 4 :
            abs_pmtx = 5;abs_pmty = 5;
        if mpmt_col == 0 and mpmt_row == 5 :
            abs_pmtx = 4;abs_pmty = 4;
        if mpmt_col == 1 and mpmt_row == 5 :
            abs_pmtx = 5;abs_pmty = 4;

        # EC4
        if mpmt_col == 0 and mpmt_row == 6 :
            abs_pmtx = 0;abs_pmty = 3;
        if mpmt_col == 1 and mpmt_row == 6 :
            abs_pmtx = 1;abs_pmty = 3;
        if mpmt_col == 0 and mpmt_row == 7 :
            abs_pmtx = 0;abs_pmty = 2;
        if mpmt_col == 1 and mpmt_row == 7 :
            abs_pmtx = 1;abs_pmty = 2;

        # EC5
        if mpmt_col == 0 and mpmt_row == 8 :
            abs_pmtx = 2;abs_pmty = 3;
        if mpmt_col == 1 and mpmt_row == 8 :
            abs_pmtx = 3;abs_pmty = 3;
        if mpmt_col == 0 and mpmt_row == 9 :
            abs_pmtx = 2;abs_pmty = 2;
        if mpmt_col == 1 and mpmt_row == 9 :
            abs_pmtx = 3;abs_pmty = 2;

        # EC6
        if mpmt_col == 0 and mpmt_row == 10 :
            abs_pmtx = 4;abs_pmty = 3;
        if mpmt_col == 1 and mpmt_row == 10 :
            abs_pmtx = 5;abs_pmty = 3;
        if mpmt_col == 0 and mpmt_row == 11 :
            abs_pmtx = 4;abs_pmty = 2;
        if mpmt_col == 1 and mpmt_row == 11 :
            abs_pmtx = 5;abs_pmty = 2;

        # EC7
        if mpmt_col == 0 and mpmt_row == 12 :
            abs_pmtx = 0;abs_pmty = 1;
        if mpmt_col == 1 and mpmt_row == 12 :
            abs_pmtx = 1;abs_pmty = 1;
        if mpmt_col == 0 and mpmt_row == 13 :
            abs_pmtx = 0;abs_pmty = 0;
        if mpmt_col == 1 and mpmt_row == 13 :
            abs_pmtx = 1;abs_pmty = 0;

        # EC8
        if mpmt_col == 0 and mpmt_row == 14 :
            abs_pmtx = 2;abs_pmty = 1;
        if mpmt_col == 1 and mpmt_row == 14 :
            abs_pmtx = 3;abs_pmty = 1;
        if mpmt_col == 0 and mpmt_row == 15 :
            abs_pmtx = 2;abs_pmty = 0;
        if mpmt_col == 1 and mpmt_row == 15 :
            abs_pmtx = 3;abs_pmty = 0;

        # EC9
        if mpmt_col == 0 and mpmt_row == 16 :
            abs_pmtx = 4;abs_pmty = 1;
        if mpmt_col == 1 and mpmt_row == 16 :
            abs_pmtx = 5;abs_pmty = 1;
        if mpmt_col == 0 and mpmt_row == 17 :
            abs_pmtx = 4;abs_pmty = 0;
        if mpmt_col == 1 and mpmt_row == 17 :
            abs_pmtx = 5;abs_pmty = 0;

        abspixx = abs_pmtx * 8 + pix_x
        abspixy = abs_pmty * 8 + (8-pix_y);  # Top-to-bottom

        return mpmt_row, mpmt_col, abspixx, abspixy
        #return abspixy * 100 + abspixx
        

class AckL1EventReader:
    acquisition_file = None
    kenji_l1_file = None

    t_texp = None
    t_tevent = None

    t_l1trg = None
    t_gtusry = None
    t_thrtable = None

    ExpTree = None

    kenji_l1trg_entries = -1
    t_gtusry_entries = -1
    t_thrtable_entries = -1
    texp_entries = -1
    tevent_entries = -1

    _current_l1trg_entry = -1
    _current_tevent_entry = -1

    _l1trg_ecID = np.array([-1], dtype=np.int32)
    _l1trg_pmtRow = np.array([-1], dtype=np.int32)
    _l1trg_pmtCol = np.array([-1], dtype=np.int32)
    _l1trg_pixRow = np.array([-1], dtype=np.int32)
    _l1trg_pixCol = np.array([-1], dtype=np.int32)
    _l1trg_gtuGlobal = np.array([-1], dtype=np.int32)
    _l1trg_packetID = np.array([-1], dtype=np.int32)
    _l1trg_gtuInPacket = np.array([-1], dtype=np.int32)
    _l1trg_sumL1 = np.array([-1], dtype=np.int32)
    _l1trg_thrL1 = np.array([-1], dtype=np.int32)
    _l1trg_persistL1 = np.array([-1], dtype=np.int32)

    _tevent_photon_count_data = None
    _tevent_gtu = np.array([-1], dtype=np.int32)
    _tevent_gtu_time = np.array([-1], dtype=np.double)
    _tevent_gtu_time1 = np.array([-1], dtype=np.double)
    # ...

    last_gtu_pdm_data = None

    @classmethod
    def _get_branch_or_raise(cls, file, tree, name):
        br = tree.GetBranch(name)
        if br is None:
            raise Exception("{} > {} is missing branch \"{}\"".format(file, tree.GetName(), name))
        return br

    def __init__(self, acquisition_pathname, kenji_l1_pathname):
        self.acquisition_file, self.t_texp, self.t_tevent = self.open_acquisition(acquisition_pathname)
        self.kenji_l1_file, self.t_l1trg, self.t_gtusry, self.t_thrtable = self.open_kenji_l1(kenji_l1_pathname)

        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "ecID").SetAddress(self._l1trg_ecID)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "pmtRow").SetAddress(self._l1trg_pmtRow)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "pmtCol").SetAddress(self._l1trg_pmtCol)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "pixRow").SetAddress(self._l1trg_pixRow)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "pixCol").SetAddress(self._l1trg_pixCol)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "gtuGlobal").SetAddress(self._l1trg_gtuGlobal)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "packetID").SetAddress(self._l1trg_packetID)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "gtuInPacket").SetAddress(self._l1trg_gtuInPacket)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "sumL1").SetAddress(self._l1trg_sumL1)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "thrL1").SetAddress(self._l1trg_thrL1)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_l1trg, "persistL1").SetAddress(self._l1trg_persistL1)

        # l1trg->Branch("ecID", & ecID, "ecID/I");
        # l1trg->Branch("pmtRow", & pmtRow, "pmtRow/I");
        # l1trg->Branch("pmtCol", & pmtCol, "pmtCol/I");
        # l1trg->Branch("pixRow", & pixRow, "pixRow/I");
        # l1trg->Branch("pixCol", & pixCol, "pixCol/I");
        # l1trg->Branch("gtuGlobal", & gtuGlobal, "gtuGlobal/I");
        # l1trg->Branch("packetID", & packetID, "packetID/I");
        # l1trg->Branch("gtuInPacket", & gtuInPacket, "gtuInPacket/I");
        # l1trg->Branch("sumL1", & sumL1, "sumL1/I");
        # l1trg->Branch("thrL1", & thrL1, "thrL1/I");
        # l1trg->Branch("persistL1", & persistL1, "persistL1/I");


        # thrtable->Branch("triggerThresholds", triggerThresholds, "triggerThresholds[100][5]/F");

        # TODO gtusry

        # gtusry->Branch("gtuGlobal", & gtuGlobal, "gtuGlobal/I");
        # gtusry->Branch("trgBoxPerGTU", & trgBoxPerGTU, "trgBoxPerGTU/I");
        # gtusry->Branch("trgPMTPerGTU", & trgPMTPerGTU, "trgPMTPerGTU/I");
        # gtusry->Branch("trgECPerGTU", & trgECPerGTU, "trgECPerGTU/I");
        # gtusry->Branch("nPersist", & nPersist, "&nPersist/I");
        # gtusry->Branch("gtuInPersist", & gtuInPersist, "&gtuInPersist/I");
        #
        # gtusry->Branch("sumL1PDM", & sumL1PDM, "sumL1PDM/I");
        # gtusry->Branch("sumL1EC", sumL1EC, "sumL1EC[9]/I");
        # gtusry->Branch("sumL1PMT", sumL1PMT, "sumL1PMT[18][2]/I");
        #
        # Int_t
        # trgPMT[18][2];
        # gtusry->Branch("trgPMT", trgPMT, "trgPMT[18][2]/I");

        self.ExpTree = ExpTree(self.t_texp, self.acquisition_file)

        self._tevent_photon_count_data = np.zeros((self.ExpTree.ccbCount, self.ExpTree.pdmCount,
                                                   self.ExpTree.pmtCountX * self.ExpTree.pixelCountX,
                                                   self.ExpTree.pmtCountY * self.ExpTree.pixelCountY), dtype=np.ubyte)

        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "photon_count_data").SetAddress(self._tevent_photon_count_data)
        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "gtu").SetAddress(self._tevent_gtu)
        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "gtu_time").SetAddress(self._tevent_gtu_time)
        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "gtu_time1").SetAddress(self._tevent_gtu_time1)

        self.kenji_l1trg_entries = self.t_l1trg.GetEntries()     # 23331
        self.t_gtusry_entries = self.t_gtusry.GetEntries()       # 16512
        self.t_thrtable_entries = self.t_thrtable.GetEntries()   # 1
        self.texp_entries = self.t_texp.GetEntries()             # 1
        self.tevent_entries = self.t_tevent.GetEntries()         # 16512


    @classmethod
    def open_acquisition(cls, pathname):
        f = ROOT.TFile.Open(pathname, "read")
        t_texp = f.Get("texp")
        t_tevent = f.Get("tevent")
        return f, t_texp, t_tevent

        #self.tree.SetBranchAddress("photon_count_data", self.pcd)
        #
        # for e in t :
        #     print(e.__dir__())

    @classmethod
    def open_kenji_l1(cls, pathname):
        f = ROOT.TFile.Open(pathname, "read")
        t_l1trg = f.Get("l1trg")                # each PDM is analyzed
        t_gtusry = f.Get("gtusry")              # each GTU is saved (records / 2304)
        t_thrtable = f.Get("thrtable")          # 1 entry - single table
        return f, t_l1trg, t_gtusry, t_thrtable

    def _search_for_tevent_by_gtu(self, gtu):
        while (self._current_tevent_entry < 0 or self._tevent_gtu != gtu) and \
                        self._current_tevent_entry < self.tevent_entries:
            self._current_tevent_entry += 1
            self.t_tevent.GetEntry(self._current_tevent_entry)

    # TODO shold be able to iterate either over GTU or over trigger events
    def __iter__(self):
        self._current_l1trg_entry = -1
        self._current_tevent_entry = -1
        return self

    # iterate over trigger events
    def __next__(self):
        self._current_l1trg_entry += 1

        if self._current_l1trg_entry >= self.kenji_l1trg_entries:
            raise StopIteration

        self.t_l1trg.GetEntry(self._current_l1trg_entry)

        if self.last_gtu_pdm_data is None or self.last_gtu_pdm_data.gtu != self._l1trg_gtuGlobal:

            self._search_for_tevent_by_gtu(self._l1trg_gtuGlobal)

            if self._tevent_gtu != self._l1trg_gtuGlobal:
                self._current_tevent_entry = -1
                self._search_for_tevent_by_gtu(self._l1trg_gtuGlobal)
                if self._tevent_gtu != self._l1trg_gtuGlobal:
                    raise Exception("GTU {} from trigger data file was not found in acquisition file")

            self.last_gtu_pdm_data = GtuPdmData(self._tevent_photon_count_data, self._tevent_gtu, self._tevent_gtu_time, self._tevent_gtu_time1)

        return L1TrgEvent.from_mario_format(self.last_gtu_pdm_data, self._l1trg_ecID,
                          self._l1trg_pmtRow, self._l1trg_pmtCol,
                          self._l1trg_pixRow, self._l1trg_pixCol, self._l1trg_sumL1, self._l1trg_thrL1, self. _l1trg_persistL1 )

        #
        # def next(self):
        #     kenji_l1trg_entries = self.t_l1trg.GetEntries()     # 23331
        #     t_gtusry_entries = self.t_gtusry.GetEntries()       # 16512
        #     t_thrtable_entries = self.t_thrtable.GetEntries()   # 1
        #     texp_entries = self.t_texp.GetEntries()             # 1
        #     tevent_entries = self.t_tevent.GetEntries()         # 16512
        #
        #     print(kenji_l1trg_entries)                          # 23331
        #     print(tevent_entries)
        #
        #     pass


def main(argv):
    parser = argparse.ArgumentParser(description='Find patterns inside triggered pixes')
    # parser.add_argument('files', nargs='+', help='List of files to convert')
    parser.add_argument('-a', '--acquisition-file', help="ACQUISITION root file in \"Lech\" format")
    parser.add_argument('-k', '--kenji-l1trigger-file', help="L1 trigger root file in \"Kenji\" format")

    args = parser.parse_args()

    ack_l1_reader = AckL1EventReader(args.acquisition_file, args.kenji_l1trigger_file)
    # e = next(ack_l1_reader)

    for e in ack_l1_reader:

        print(e.pix_col, e.pix_row, e.pmt_col, e.pmt_row, e.__dict__, e.gtu_pdm_data.photon_count_data[0][0])
        plt.imshow(np.transpose(e.gtu_pdm_data.photon_count_data[0][0]))
        # plt.imshow(np.reshape(e.gtu_pdm_data.photon_count_data[0][0], e.gtu_pdm_data.photon_count_data[0][0].shape, 'C'))
        plt.colorbar()
        plt.show()

    print(e)

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)

