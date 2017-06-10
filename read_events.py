#!/usr/bin/python3

import numpy as np
import os
import argparse
import sys
import ROOT
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from eusotrees.exptree import ExpTree
import collections
from enum import Enum

from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line


class GtuPdmData:
    photon_count_data = None
    gtu = -1
    gtu_time = -1
    # gtu_time1 = -1

    # gtu_global = -1         #  "gtuGlobal/I"
    trg_box_per_gtu = -1    #  "trgBoxPerGTU/I"
    trg_pmt_per_gtu = -1    #  "trgPMTPerGTU/I"
    trg_ec_per_gtu = -1     #  "trgECPerGTU/I"
    n_persist = -1          #  "&nPersist/I"
    gtu_in_persist = -1     #  "&gtuInPersist/I"
    sum_l1_pdm = -1         #  "sumL1PDM/I"
    sum_l1_ec = None                #  "sumL1EC[9]/I"
    sum_l1_pmt = None   #  "sumL1PMT[18][2]/I"

    l1trg_events = None

    def __init__(self, photon_count_data, gtu, gtu_time, #gtu_time1,
                 trg_box_per_gtu, trg_pmt_per_gtu, trg_ec_per_gtu,
                 n_persist, gtu_in_persist, sum_l1_pdm, sum_l1_ec, sum_l1_pmt,
                 l1trg_events=[]):

        self.photon_count_data = np.zeros_like(photon_count_data)

        for ccb_index in range(0,len(photon_count_data)):
            for pdm_index in range(0,len(photon_count_data[ccb_index])):
                self.photon_count_data[ccb_index, pdm_index] = np.transpose(np.fliplr(photon_count_data[ccb_index, pdm_index])) # np.fliplr(np.transpose(photon_count_data[ccb_index, pdm_index]))

        self.gtu = np.asscalar(gtu) if isinstance(gtu, np.ndarray) else gtu
        self.gtu_time = np.asscalar(gtu_time) if isinstance(gtu_time, np.ndarray) else gtu_time
        # self.gtu_time1 = np.asscalar(gtu_time1) if isinstance(gtu_time1, np.ndarray) else gtu_time1

        self.trg_box_per_gtu = np.asscalar(trg_box_per_gtu) if isinstance(trg_box_per_gtu, np.ndarray) else trg_box_per_gtu 
        self.trg_pmt_per_gtu = np.asscalar(trg_pmt_per_gtu) if isinstance(trg_pmt_per_gtu, np.ndarray) else trg_pmt_per_gtu
        self.trg_ec_per_gtu = np.asscalar(trg_ec_per_gtu) if isinstance(trg_ec_per_gtu, np.ndarray) else trg_ec_per_gtu
        self.n_persist = np.asscalar(n_persist) if isinstance(n_persist, np.ndarray) else n_persist
        self.gtu_in_persist = np.asscalar(gtu_in_persist) if isinstance(gtu_in_persist, np.ndarray) else gtu_in_persist
        self.sum_l1_pdm = np.asscalar(sum_l1_pdm) if isinstance(sum_l1_pdm, np.ndarray) else sum_l1_pdm
        self.sum_l1_ec = sum_l1_ec
        self.sum_l1_pmt = sum_l1_pmt

        self.l1trg_events = l1trg_events

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

    packet_id = -1      # ideally this should be in GtuPdmData, but this is from l1trg tree
    gtu_in_packet = -1

    def __init__(self, gtu_pdm_data, ec_id, pmt_row, pmt_col, pix_row, pix_col, sum_l1, thr_l1, persist_l1,
                 packet_id = -1, gtu_in_packet = -1):
        self.gtu_pdm_data = gtu_pdm_data
        self.ec_id = np.asscalar(ec_id) if isinstance(ec_id, np.ndarray) else ec_id
        self.pmt_row = np.asscalar(pmt_row) if isinstance(pmt_row, np.ndarray) else pmt_row
        self.pmt_col = np.asscalar(pmt_col) if isinstance(pmt_col, np.ndarray) else pmt_col
        self.pix_row = np.asscalar(pix_row) if isinstance(pix_row, np.ndarray) else pix_row
        self.pix_col = np.asscalar(pix_col) if isinstance(pix_col, np.ndarray) else pix_col
        self.sum_l1 = np.asscalar(sum_l1) if isinstance(sum_l1, np.ndarray) else sum_l1
        self.thr_l1 = np.asscalar(thr_l1) if isinstance(thr_l1, np.ndarray) else thr_l1
        self.persist_l1 = np.asscalar(persist_l1) if isinstance(persist_l1, np.ndarray) else persist_l1

        self.packet_id = np.asscalar(packet_id) if isinstance(packet_id, np.ndarray) else packet_id
        self.gtu_in_packet = np.asscalar(gtu_in_packet) if isinstance(gtu_in_packet, np.ndarray) else gtu_in_packet

    @classmethod
    def from_mario_format(cls, gtu_pdm_data, ec_id, pmt_row, pmt_col, pix_row, pix_col, sum_l1, thr_l1, persist_l1,
                          packet_id = -1, gtu_in_packet = -1):
        e = L1TrgEvent(gtu_pdm_data, ec_id, pmt_row, pmt_col, pix_row, pix_col, sum_l1, thr_l1, persist_l1, packet_id, gtu_in_packet)
        e.o_pmt_col = e.pmt_col; e.o_pmt_row = e.pmt_row; e.o_pix_col = e.pix_col; e.o_pix_row = e.pix_row
        e.pmt_col, e.pmt_row, e.pix_col, e.pix_row = cls.mario2pdm_for_mpl(e.pmt_col, e.pmt_row, e.pix_col, e.pix_row)
        return e

    @classmethod
    def mario2pdm_for_mpl(cls, mpmt_col=0, mpmt_row=0, pix_x=0, pix_y=0):
        if mpmt_row >= 18 or mpmt_row < 0:
            raise Exception("Something is rotten in the state of row#. mario")

        if mpmt_col >= 2 or mpmt_col < 0:
            raise Exception("Something is rotten in the state of col#. mario")

        ec_index = mpmt_row // 2
        pmt_row_in_ec = mpmt_row % 2
        lech_pmt_row = (ec_index // 3) * 2 + pmt_row_in_ec
        lech_pmt_col = (ec_index % 3) * 2 + mpmt_col

        abspix_col = lech_pmt_col*8 + pix_x
        abspix_row = lech_pmt_row*8 + pix_y

        return lech_pmt_col, lech_pmt_row, abspix_col, abspix_row


class AckL1EventReader:
    acquisition_file = None
    kenji_l1_file = None

    t_texp = None
    t_tevent = None

    t_l1trg = None
    t_gtusry = None
    t_thrtable = None

    exp_tree = None

    kenji_l1trg_entries = -1
    t_gtusry_entries = -1
    t_thrtable_entries = -1
    texp_entries = -1
    tevent_entries = -1

    _current_l1trg_entry = -1
    _current_tevent_entry = -1
    _current_gtusry_entry = -1

    _l1trg_ecID = None # np.array([-1], dtype=np.int32)
    _l1trg_pmtRow = None # np.array([-1], dtype=np.int32)
    _l1trg_pmtCol = None # np.array([-1], dtype=np.int32)
    _l1trg_pixRow = None # np.array([-1], dtype=np.int32)
    _l1trg_pixCol = None # np.array([-1], dtype=np.int32)
    _l1trg_gtuGlobal = None # np.array([-1], dtype=np.int32)
    _l1trg_packetID = None # np.array([-1], dtype=np.int32)
    _l1trg_gtuInPacket = None # np.array([-1], dtype=np.int32)
    _l1trg_sumL1 = None # np.array([-1], dtype=np.int32)
    _l1trg_thrL1 = None # np.array([-1], dtype=np.int32)
    _l1trg_persistL1 = None # np.array([-1], dtype=np.int32)

    _gtusry_gtuGlobal = None # np.array([-1], dtype=np.int32) #  "gtuGlobal/I"
    _gtusry_trgBoxPerGTU = None # np.array([-1], dtype=np.int32) #  "trgBoxPerGTU/I"
    _gtusry_trgPMTPerGTU = None # np.array([-1], dtype=np.int32) #  "trgPMTPerGTU/I"
    _gtusry_trgECPerGTU = None # np.array([-1], dtype=np.int32) #  "trgECPerGTU/I"
    _gtusry_nPersist = None # np.array([-1], dtype=np.int32) #  "&nPersist/I"
    _gtusry_gtuInPersist = None # np.array([-1], dtype=np.int32) #  "&gtuInPersist/I"

    _gtusry_sumL1PDM = None # np.array([-1], dtype=np.int32) #  "sumL1PDM/I"
    _gtusry_sumL1EC = None # np.array([-1]*9, dtype=np.int32) #  "sumL1EC[9]/I"
    _gtusry_sumL1PMT = None # np.negative(np.ones((18,2), dtype=np.int32)) #  "sumL1PMT[18][2]/I"
    _gtusry_trgPMT = None # np.negative(np.ones((18,2), dtype=np.int32)) #  "sumL1PMT[18][2]/I"

    _tevent_photon_count_data = None
    _tevent_gtu = None # np.array([-1], dtype=np.int32)
    _tevent_gtu_time = None # np.array([-1], dtype=np.double)
    #_tevent_gtu_time1 = None # np.array([-1], dtype=np.double)
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

        self._l1trg_ecID = np.array([-1], dtype=np.int32)
        self._l1trg_pmtRow = np.array([-1], dtype=np.int32)
        self._l1trg_pmtCol = np.array([-1], dtype=np.int32)
        self._l1trg_pixRow = np.array([-1], dtype=np.int32)
        self._l1trg_pixCol = np.array([-1], dtype=np.int32)
        self._l1trg_gtuGlobal = np.array([-1], dtype=np.int32)
        self._l1trg_packetID = np.array([-1], dtype=np.int32)
        self._l1trg_gtuInPacket = np.array([-1], dtype=np.int32)
        self._l1trg_sumL1 = np.array([-1], dtype=np.int32)
        self._l1trg_thrL1 = np.array([-1], dtype=np.int32)
        self._l1trg_persistL1 = np.array([-1], dtype=np.int32)

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

        # !!!
        #
        # thrtable->Branch("triggerThresholds", triggerThresholds, "triggerThresholds[100][5]/F");
        #
        # !!!

        self._gtusry_gtuGlobal = np.array([-1], dtype=np.int32)  # "gtuGlobal/I"
        self._gtusry_trgBoxPerGTU = np.array([-1], dtype=np.int32)  # "trgBoxPerGTU/I"
        self._gtusry_trgPMTPerGTU = np.array([-1], dtype=np.int32)  # "trgPMTPerGTU/I"
        self._gtusry_trgECPerGTU = np.array([-1], dtype=np.int32)  # "trgECPerGTU/I"
        self._gtusry_nPersist = np.array([-1], dtype=np.int32)  # "&nPersist/I"
        self._gtusry_gtuInPersist = np.array([-1], dtype=np.int32)  # "&gtuInPersist/I"

        self._gtusry_sumL1PDM = np.array([-1], dtype=np.int32)  # "sumL1PDM/I"
        self._gtusry_sumL1EC = np.array([-1] * 9, dtype=np.int32)  # "sumL1EC[9]/I"
        self._gtusry_sumL1PMT = np.negative(np.ones((18, 2), dtype=np.int32))  # "sumL1PMT[18][2]/I"
        self._gtusry_trgPMT = np.negative(np.ones((18, 2), dtype=np.int32))  # "sumL1PMT[18][2]/I"

        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "gtuGlobal").SetAddress(self._gtusry_gtuGlobal)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "trgBoxPerGTU").SetAddress(self._gtusry_trgBoxPerGTU)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "trgPMTPerGTU").SetAddress(self._gtusry_trgPMTPerGTU)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "trgECPerGTU").SetAddress(self._gtusry_trgECPerGTU)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "nPersist").SetAddress(self._gtusry_nPersist)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "gtuInPersist").SetAddress(self._gtusry_gtuInPersist)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "sumL1PDM").SetAddress(self._gtusry_sumL1PDM)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "sumL1EC").SetAddress(self._gtusry_sumL1EC)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "sumL1PMT").SetAddress(self._gtusry_sumL1PMT)
        self._get_branch_or_raise(kenji_l1_pathname, self.t_gtusry, "trgPMT").SetAddress(self._gtusry_trgPMT)

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
        # gtusry->Branch("trgPMT", trgPMT, "trgPMT[18][2]/I");

        self.exp_tree = ExpTree(self.t_texp, self.acquisition_file)

        self._tevent_photon_count_data = np.zeros((self.exp_tree.ccbCount, self.exp_tree.pdmCount,
                                                   self.exp_tree.pmtCountX * self.exp_tree.pixelCountX,
                                                   self.exp_tree.pmtCountY * self.exp_tree.pixelCountY), dtype=np.ubyte)

        self._tevent_gtu = np.array([-1], dtype=np.int32)
        self._tevent_gtu_time = np.array([-1], dtype=np.double)
        #self._tevent_gtu_time1 = np.array([-1], dtype=np.double)

        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "photon_count_data").SetAddress(self._tevent_photon_count_data)
        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "gtu").SetAddress(self._tevent_gtu)
        self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "gtu_time").SetAddress(self._tevent_gtu_time)
        # self._get_branch_or_raise(acquisition_pathname, self.t_tevent, "gtu_time1").SetAddress(self._tevent_gtu_time1)

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

    # def get_tevent_entry(self, num = None):
    #     if num is None:
    #         self.t_tevent.GetEntry(self._current_tevent_entry)
    #     else:
    #         self.t_tevent.GetEntry(num)
    #         self._current_l1trg_entry = num # practically unnecesarry
    #     self._tevent_photon_count_data = np.filplr(np.transpose(self._tevent_photon_count_data))


    def _search_for_tevent_by_gtu(self, gtu):
        while (self._current_tevent_entry < 0 or self._tevent_gtu != gtu) and \
                        self._current_tevent_entry < self.tevent_entries:
            self._current_tevent_entry += 1
            self.t_tevent.GetEntry(self._current_tevent_entry)

    def _search_for_gtusry_by_gtu(self, gtu):
        while (self._current_gtusry_entry < 0 or self._gtusry_gtuGlobal != gtu) and \
                        self._current_gtusry_entry < self.t_gtusry_entries:
            self._current_gtusry_entry += 1
            self.t_gtusry.GetEntry(self._current_gtusry_entry)

    def _search_for_l1trg_events_by_gtu(self, gtu, gtu_pdm_data=None, presume_sorted=True):
        if not presume_sorted: # or self._l1trg_gtuGlobal > gtu:
            self._current_l1trg_entry = -1

        events_list = []

        while self._current_l1trg_entry < self.kenji_l1trg_entries:
            if self._current_l1trg_entry == -1 or (presume_sorted and self._l1trg_gtuGlobal < gtu) or (not presume_sorted and self._l1trg_gtuGlobal != gtu) :
                self._current_l1trg_entry += 1
                self.t_l1trg.GetEntry(self._current_l1trg_entry)

            # if self._current_l1trg_entry == 168:
            #     print(">"*30, "Entry 168 read", "<"*30)

            if self._l1trg_gtuGlobal == gtu:
                events_list.append(L1TrgEvent.from_mario_format(gtu_pdm_data, self._l1trg_ecID,
                      self._l1trg_pmtRow, self._l1trg_pmtCol,
                      self._l1trg_pixRow, self._l1trg_pixCol, self._l1trg_sumL1, self._l1trg_thrL1, self. _l1trg_persistL1,
                      self._l1trg_packetID, self._l1trg_gtuInPacket))

                # _l1trg_packetID = None  # np.array([-1], dtype=np.int32)
                # _l1trg_gtuInPacket = None  # np.array([-1], dtype=np.int32)

                self._current_l1trg_entry += 1
                self.t_l1trg.GetEntry(self._current_l1trg_entry)
            elif presume_sorted and self._l1trg_gtuGlobal > gtu:
                break

        return events_list

    class L1TrgEventIterator:
        ack_ev_reader = None

        def __init__(self, ack_ev_reader):
            self.ack_ev_reader = ack_ev_reader

        def __iter__(self):
            self.ack_ev_reader._current_l1trg_entry = -1
            self.ack_ev_reader._current_tevent_entry = -1
            self.ack_ev_reader._current_gtusry_entry = -1
            return self

        def __next__(self):
            aer = self.ack_ev_reader

            aer._current_l1trg_entry += 1

            if aer._current_l1trg_entry >= aer.kenji_l1trg_entries:
                raise StopIteration

            aer.t_l1trg.GetEntry(aer._current_l1trg_entry)

            if aer.last_gtu_pdm_data is None or aer.last_gtu_pdm_data.gtu != aer._l1trg_gtuGlobal:

                aer._search_for_tevent_by_gtu(aer._l1trg_gtuGlobal)

                if aer._tevent_gtu != aer._l1trg_gtuGlobal:
                    aer._current_tevent_entry = -1
                    aer._search_for_tevent_by_gtu(aer._l1trg_gtuGlobal)
                    if aer._tevent_gtu != aer._l1trg_gtuGlobal:
                        raise Exception("GTU {} from trigger data file (tree l1trg) was not found in acquisition file (tree tevent)".format(aer._l1trg_gtuGlobal))

                aer._search_for_gtusry_by_gtu(aer._l1trg_gtuGlobal)

                if aer._gtusry_gtuGlobal != aer._l1trg_gtuGlobal:
                    aer._current_gtusry_entry = -1
                    aer._search_for_gtusry_by_gtu(aer._l1trg_gtuGlobal)
                    if aer._gtusry_gtuGlobal != aer._l1trg_gtuGlobal:
                        raise Exception("GTU {} from trigger data file (tree l1trg) was not found in trigger data file (tree gtusry)".format(aer._l1trg_gtuGlobal))

                aer.last_gtu_pdm_data = GtuPdmData(aer._tevent_photon_count_data, aer._tevent_gtu, aer._tevent_gtu_time, #aer._tevent_gtu_time1,
                                                    aer._gtusry_trgBoxPerGTU, aer._gtusry_trgPMTPerGTU, aer._gtusry_trgECPerGTU,
                                                    aer._gtusry_nPersist, aer._gtusry_gtuInPersist,
                                                    aer._gtusry_sumL1PDM, aer._gtusry_sumL1EC, aer._gtusry_sumL1PMT)

            l1trg_ev = L1TrgEvent.from_mario_format(aer.last_gtu_pdm_data, aer._l1trg_ecID,
                          aer._l1trg_pmtRow, aer._l1trg_pmtCol,
                          aer._l1trg_pixRow, aer._l1trg_pixCol, aer._l1trg_sumL1, aer._l1trg_thrL1, aer. _l1trg_persistL1)
            aer.last_gtu_pdm_data.l1trg_events.append(l1trg_ev) # not very correct in this form - not all events are going to be associated to the GTU

            return l1trg_ev

    class GtuPdmDataIterator:
        ack_ev_reader = None
        presume_sorted = True

        def __init__(self, ack_ev_reader, presume_sorted = True):
            self.ack_ev_reader = ack_ev_reader
            self.presume_sorted = presume_sorted

        def __iter__(self):
            self.ack_ev_reader._current_l1trg_entry = -1
            self.ack_ev_reader._current_tevent_entry = -1
            self.ack_ev_reader._current_gtusry_entry = -1
            return self

        # TODO iterate over gtu
        def __next__(self):
            aer = self.ack_ev_reader
            aer._current_tevent_entry += 1

            if aer._current_tevent_entry >= aer.tevent_entries:
                raise StopIteration

            aer.t_tevent.GetEntry(aer._current_tevent_entry)

            aer._search_for_gtusry_by_gtu(aer._tevent_gtu)

            if aer._gtusry_gtuGlobal != aer._tevent_gtu:
                aer._current_gtusry_entry = -1
                aer._search_for_gtusry_by_gtu(aer._tevent_gtu)
                if aer._gtusry_gtuGlobal != aer._tevent_gtu:
                    raise Exception(
                        "GTU {} from acquisition data file (tree tevent) was not found in trigger data file (tree gtusry)".format(aer._tevent_gtu))

            gtu_pdm_data = GtuPdmData(aer._tevent_photon_count_data, aer._tevent_gtu, aer._tevent_gtu_time, #aer._tevent_gtu_time1,
                                        aer._gtusry_trgBoxPerGTU, aer._gtusry_trgPMTPerGTU, aer._gtusry_trgECPerGTU,
                                        aer._gtusry_nPersist, aer._gtusry_gtuInPersist,
                                        aer._gtusry_sumL1PDM, aer._gtusry_sumL1EC, aer._gtusry_sumL1PMT)

            l1trg_events = aer._search_for_l1trg_events_by_gtu(aer._tevent_gtu, gtu_pdm_data)

            gtu_pdm_data.l1trg_events = l1trg_events

            return gtu_pdm_data

    def iter_l1trg_events(self):
        return self.L1TrgEventIterator(self)

    def iter_gtu_pdm_data(self):
        return self.GtuPdmDataIterator(self)

class EventFilterOptions:
    class Cond(Enum):
        lt = 1
        le = 2
        eq = 3
        ge = 4
        gt = 5

    n_persist = -1
    n_persist_cond = Cond.lt
    sum_l1_pdm = -1
    sum_l1_pdm_cond = Cond.lt
    sum_l1_ec_one = -1
    sum_l1_ec_one_cond = Cond.lt
    sum_l1_pmt_one = -1
    sum_l1_pmt_one_cond = Cond.lt

    def cmp(self, val1, val2, cmp_type):
        if cmp_type == EventFilterOptions.Cond.lt:
            return val1 < val2
        elif cmp_type == EventFilterOptions.Cond.le:
            return val1 <= val2
        elif cmp_type == EventFilterOptions.Cond.eq:
            return val1 == val2
        elif cmp_type == EventFilterOptions.Cond.ge:
            return val1 >= val2
        else:
            val1 > val2

    def has_one_cell_valid(self, v, m, cond):
        for i in self.m.shape[0]:
            for j in self.m.shape[1]:
                if self.cmp(v, m[i][j], cond): #TODO one or all?
                    return True
        return False

    def check_pdm_gtu(self, pdm_gtu_data):
        if not (self.cmp(self.n_persist, pdm_gtu_data.n_persist, self.n_persist_cond) and self.cmp(self.sum_l1_pdm, pdm_gtu_data.sum_l1_pdm, self.sum_l1_pdm_cond)):
            return False
        if not self.has_one_cell_valid(self.sum_l1_ec_one, pdm_gtu_data.sum_l1_ec, self.sum_l1_ec_one_cond):
            return False
        if not self.has_one_cell_valid(self.sum_l1_pmt_one, pdm_gtu_data.sum_l1_pmt, self.sum_l1_pmt_one_cond):
            return False
        return True


def visualize_frame(pcd, exp_tree, l1trg_events=[], title=None, show=True, vmin=None, vmax=None):
    fig, ax = plt.subplots(1)

    det_width =  exp_tree.pmtCountX * exp_tree.pixelCountX
    det_height = exp_tree.pmtCountY * exp_tree.pixelCountY
    pmt_width =  exp_tree.pdmPixelCountX / exp_tree.pmtCountX
    pmt_height = exp_tree.pdmPixelCountY / exp_tree.pmtCountY

    # det_array = np.zeros_like(pcd)
    # pdm_pcd = pcd[0,0]
    # for l1trg_ev in gtu_pdm_data.l1trg_events:
    #     det_array[0][0][l1trg_ev.pix_row, l1trg_ev.pix_col] = gtu_pdm_data.photon_count_data[0][0][l1trg_ev.pix_row, l1trg_ev.pix_col]

    if title is not None:
        ax.set_title(title)

    cax = ax.imshow(pcd, extent=[0, det_width, det_height, 0], vmin=vmin, vmax=vmax)
    fig.colorbar(cax)

    for l1trg_ev in l1trg_events:
        # if l1trg_ev.pix_row == 1:
        rect = mpl_patches.Rectangle((l1trg_ev.pix_col, l1trg_ev.pix_row), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    for tmp_pmt_x in range(0, exp_tree.pmtCountX):
        for tmp_pmt_y in range(0, exp_tree.pmtCountY):
            rect = mpl_patches.Rectangle((tmp_pmt_x*pmt_width, tmp_pmt_y*pmt_height), pmt_width, pmt_height, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    if show:
        plt.show()


def visualize_frame_num_relation(frame_num_x, l1trg_events_by_frame_num=[], att_name="pix_col", title=None, show=True, vmin=None, vmax=None):
    fig, ax = plt.subplots(1)

    if title is not None:
        ax.set_title(title)

    cax = ax.imshow(frame_num_x, extent=[0, frame_num_x.shape[1], frame_num_x.shape[0], 0], vmin=vmin, vmax=vmax)
    fig.colorbar(cax)

    for frame_num, l1trg_events in enumerate(l1trg_events_by_frame_num):
        # if l1trg_ev.pix_row == 1:
        for l1trg_ev in l1trg_events:
            rect = mpl_patches.Rectangle((frame_num, getattr(l1trg_ev, att_name)), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    if show:
        plt.show()


def print_frame_info(gtu_pdm_data):
    print("GTU {} ({}); trgBoxPerGTU: {}, trgPmtPerGTU: {}, trgPmtPerGTU: {}; nPersist: {}, gtuInPersist: {}"
          .format(gtu_pdm_data.gtu, gtu_pdm_data.gtu_time,
                  gtu_pdm_data.trg_box_per_gtu, gtu_pdm_data.trg_pmt_per_gtu, gtu_pdm_data.trg_ec_per_gtu,
                  gtu_pdm_data.n_persist, gtu_pdm_data.gtu_in_persist))
    for l1trg_ev in gtu_pdm_data.l1trg_events:
        print("    pix: {},{}; PMT: {},{}; EC: {}; sumL1: {}, thrL1: {}, persistL1: {} ".format(l1trg_ev.pix_col,
                                                                                                l1trg_ev.pix_row,
                                                                                                l1trg_ev.pmt_col,
                                                                                                l1trg_ev.pmt_row,
                                                                                                l1trg_ev.ec_id,
                                                                                                l1trg_ev.sum_l1,
                                                                                                l1trg_ev.thr_l1,
                                                                                                l1trg_ev.persist_l1),
              " orig: ", l1trg_ev.o_pmt_col, l1trg_ev.o_pmt_row, "; ", l1trg_ev.o_pix_col, l1trg_ev.o_pix_row)


def read_l1trg_events(ack_l1_reader):
    for l1trg_ev in ack_l1_reader.iter_l1trg_events():
        print_frame_info(l1trg_ev.gtu_pdm_data)
        # print("GTU {} ({}; {}); trgBoxPerGTU: {}, trgPmtPerGTU: {}, trgPmtPerGTU: {}; nPersist: {}, gtuInPersist: {}"
        # .format(l1trg_ev.gtu_pdm_data.gtu, l1trg_ev.gtu_pdm_data.gtu_time, l1trg_ev.gtu_pdm_data.gtu_time1,
        #         l1trg_ev.gtu_pdm_data.trg_box_per_gtu, l1trg_ev.gtu_pdm_data.trg_pmt_per_gtu, l1trg_ev.gtu_pdm_data.trg_ec_per_gtu,
        #         l1trg_ev.gtu_pdm_data.n_persist, l1trg_ev.gtu_pdm_data.gtu_in_persist))
        # print("    pix: {},{}; PMT: {},{}; EC: {}; sumL1: {}, thrL1: {}, persistL1: {} ".format(l1trg_ev.pix_col,
        #                                                                                         l1trg_ev.pix_row,
        #                                                                                         l1trg_ev.pmt_col,
        #                                                                                         l1trg_ev.pmt_row,
        #                                                                                         l1trg_ev.ec_id,
        #                                                                                         l1trg_ev.sum_l1,
        #                                                                                         l1trg_ev.thr_l1,
        #                                                                                         l1trg_ev.persist_l1))
        # plt.imshow(np.transpose(l1trg_ev.gtu_pdm_data.photon_count_data[0][0]))
        # plt.colorbar()
        # plt.show()

def gray_hough_line(image, size=2, phi_range=np.linspace(0, np.pi, 90), rho_step=1):
    max_distance = np.hypot(image.shape[0], image.shape[1])
    num_rho = int(np.ceil(max_distance*2/rho_step))
    rho_correction_lower = -size + max_distance
    rho_correction_upper = size  + max_distance
    #phi_range = phi_range - np.pi / 2
    acc_matrix = np.zeros((num_rho, len(phi_range)))
    nc_acc_matrix = np.zeros((num_rho, len(phi_range)))

    phi_corr_arr = np.ones((100,len(phi_range)))

    max_acc_matrix_val = 0

    phi_corr = 1
    for phi_index, phi in enumerate(phi_range):
        # print("hough > phi = {} ({})".format(np.rad2deg(phi), phi_index))
        phi_norm_pi_over_2 = (phi - np.floor(phi/(np.pi/2))*np.pi/2)
        if phi_norm_pi_over_2 <= np.pi/4:
            phi_corr = image.shape[1] / np.sqrt(image.shape[1] ** 2 + (image.shape[1] * np.tan( phi_norm_pi_over_2 )) ** 2)
        else:
            phi_corr = image.shape[0] / np.sqrt(image.shape[0] ** 2 + (image.shape[0] * np.tan( np.pi/2 - phi_norm_pi_over_2 )) ** 2) #np.sqrt(image.shape[0] ** 2 + (image.shape[0] / np.tan( phi_norm_pi_over_2 - np.pi/4 )) ** 2) / image.shape[1]

        # for l in range(0,len(phi_corr_arr)):
        #     phi_corr_arr[l,phi_index] = phi_corr

        # phi_corr = 1 #(np.cos(phi*4) + 1)/2 + 1
        for i in range(0, len(image)): # row, y-axis
            for j in range(0, len(image[i])): # col, x-axis
                rho = j*np.cos(phi) + i*np.sin(phi)
                #
                # if rho < 0:
                #     print("rho =",rho, "phi =", phi, "phi_index =", phi_index, "i =", i, "j=", j)

                rho_index_lower = int((rho+rho_correction_lower) // rho_step)
                rho_index_upper = int((rho+rho_correction_upper) // rho_step + 1)

                if rho_index_lower < 0:
                    rho_index_lower = 0

                if rho_index_upper > num_rho:
                    rho_index_upper = num_rho

                for rho_index in range(rho_index_lower,rho_index_upper):
                    acc_matrix[rho_index, phi_index] +=  image[i,j] * phi_corr

                    if acc_matrix[rho_index, phi_index] > max_acc_matrix_val:
                        max_acc_matrix_val = acc_matrix[rho_index, phi_index]
                        print("new max val [{},{}] = {} | rho={} ({},{})  phi={} ({} rad)".format(rho_index, phi_index, max_acc_matrix_val, rho, rho_index_lower, rho_index_upper, np.rad2deg(phi), phi))

                    nc_acc_matrix[rho_index, phi_index] +=  image[i,j]



    # fig1, ax = plt.subplots(1)
    # cax=ax.imshow(phi_corr_arr, aspect='auto')
    # fig1.colorbar(cax)

    acc_matrix_max_pos = np.unravel_index(acc_matrix.argmax(), acc_matrix.shape)
    acc_matrix_max = acc_matrix[acc_matrix_max_pos]

    acc_matrix_max_rho_base = rho_step*acc_matrix_max_pos[0]
    acc_matrix_max_rho_range = (acc_matrix_max_rho_base - rho_correction_lower, acc_matrix_max_rho_base - rho_correction_upper)
    acc_matrix_max_phi = phi_range[acc_matrix_max_pos[1]]


    print("acc_matrix: max={}, max_row={} ({},{}) , max_col={} ({})"
          .format(acc_matrix_max,
                  acc_matrix_max_pos[0], acc_matrix_max_rho_range[0], acc_matrix_max_rho_range[1],
                  acc_matrix_max_pos[1], np.rad2deg(acc_matrix_max_phi) ))

    #  ({} = {}*{} - ({} = -{} + {}) + {}/2)
    #        rho_step,rho_index, rho_correction_lower, size, max_distance, size,

    ###

    nc_acc_matrix_max_pos = np.unravel_index(nc_acc_matrix.argmax(), nc_acc_matrix.shape)
    nc_acc_matrix_max = nc_acc_matrix[nc_acc_matrix_max_pos]

    nc_acc_matrix_max_rho = rho_step*nc_acc_matrix_max_pos[0] - max_distance # TODO this should be range !!!
    nc_acc_matrix_max_phi = phi_range[nc_acc_matrix_max_pos[1]]

    print("nc_acc_matrix: max={}, max_row={} ({}), max_col={} ({})".format(nc_acc_matrix_max, nc_acc_matrix_max_pos[0], nc_acc_matrix_max_rho, nc_acc_matrix_max_pos[1], np.rad2deg(nc_acc_matrix_max_phi) ))

    ###

    fig2, (ax1, ax2) = plt.subplots(2)

    ax1.imshow(acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), -max_distance, max_distance], aspect='auto')
    ax2.imshow(nc_acc_matrix, extent=[np.rad2deg(phi_range[0]),np.rad2deg(phi_range[-1]), -max_distance, max_distance], aspect='auto')

    fig3, ax3 = plt.subplots(1)
    cax3 = ax3.imshow(acc_matrix, aspect='auto')
    fig3.colorbar(cax3)

    fig4, ax4 = plt.subplots(1)
    cax4 = ax4.imshow(image, aspect='auto', extent=[0, image.shape[1], image.shape[0], 0])
    ax4.set_title("Hough input img")
    # y0 = (acc_matrix_max_rho - 0 * np.cos(acc_matrix_max_phi)) / np.sin(angle)
    # y1 = (acc_matrix_max_rho - image.shape[1] * np.cos(angle)) / np.sin(angle)

    for acc_matrix_max_rho in acc_matrix_max_rho_range:
        p = np.zeros((2,2))

        p[0,1] = x0 = 0
        p[0,0] = y0 = acc_matrix_max_rho / np.sin(acc_matrix_max_phi)

        p[1,1] = x1 = image.shape[0]
        p[1,0] = y1 = (acc_matrix_max_rho - image.shape[1] * np.cos(acc_matrix_max_phi)) / np.sin(acc_matrix_max_phi)

        for i in range(0,len(p)):
            if p[i,0] < 0:
                p[i,0] = 0  # y
                p[i,1] = acc_matrix_max_rho/np.cos(acc_matrix_max_phi) # x
            elif p[i,0] > image.shape[0]:
                p[i,0] = image.shape[0] # y
                p[i,1] = (acc_matrix_max_rho - p[i,0]*np.sin(acc_matrix_max_phi))/np.cos(acc_matrix_max_phi) # x


        print("line (y,x) [{},{}] , [{},{}]".format(p[0,0],p[0,1],p[1,0],p[1,1]))

        ax4.plot((p[:,1]), (p[:,0]), '-r')

    return acc_matrix

def process_event(frames, exp_tree, pixels_mask = None):
    print(len(frames))

    event_frames = []

    triggered_pixel_sum_l1_frames = []
    triggered_pixel_thr_l1_frames = []
    triggered_pixel_persist_l1_frames = []

    all_event_triggers = []
    event_triggers_by_frame = [None]*len(frames)

    for frame_num, gtu_pdm_data in enumerate(frames):
        pcd = gtu_pdm_data.photon_count_data
        if len(pcd) > 0 and len(pcd[0]) > 0:
            # TODO warning now only the very first PDM is processed
            event_frames.append(pcd[0][0])
            triggered_pixel_sum_l1 = np.zeros_like(pcd[0][0])
            triggered_pixel_thr_l1 = np.zeros_like(pcd[0][0])
            triggered_pixel_persist_l1 = np.zeros_like(pcd[0][0])

            all_event_triggers += gtu_pdm_data.l1trg_events
            event_triggers_by_frame[frame_num] = gtu_pdm_data.l1trg_events

            for l1trg_ev in gtu_pdm_data.l1trg_events:
                triggered_pixel_sum_l1[l1trg_ev.pix_row, l1trg_ev.pix_col] = l1trg_ev.sum_l1
                triggered_pixel_thr_l1[l1trg_ev.pix_row, l1trg_ev.pix_col] = l1trg_ev.thr_l1
                triggered_pixel_persist_l1[l1trg_ev.pix_row, l1trg_ev.pix_col] = l1trg_ev.persist_l1
            triggered_pixel_sum_l1_frames.append(triggered_pixel_sum_l1)
            triggered_pixel_thr_l1_frames.append(triggered_pixel_thr_l1)
            triggered_pixel_persist_l1_frames.append(triggered_pixel_persist_l1)

    pdm_max_list = [np.max(frame) for frame in event_frames]
    max_value = np.max(pdm_max_list)
    pdm_min_list = [np.max(frame) for frame in event_frames]
    min_value = np.min(pdm_min_list)

    # for frame_num, gtu_pdm_data in enumerate(frames):
    #     pcd = gtu_pdm_data.photon_count_data
    #     if len(pcd) > 0 and len(pcd[0]) > 0:
    #     # TODO warning now only the very
    #         visualize_frame(pcd[0][0], exp_tree, gtu_pdm_data.l1trg_events, "frame: {}, GTU: {}".format(frame_num, gtu_pdm_data.gtu), True, min_value, max_value)

    if len(event_frames) == 0:
        raise Exception("Nothing to visualize")


    # possibly find threshold with average background (another parameter?)

    frame_num_y = []
    frame_num_x = []

    for frame in event_frames:
        frame_num_y.append(np.max(frame, axis=1).reshape(-1,1)) # summing the x axis
        frame_num_x.append(np.max(frame, axis=0).reshape(-1,1)) # summing the y axis
        # frame_num_y.append(np.sum(frame, axis=1).reshape(-1,1)) # summing the x axis
        # frame_num_x.append(np.sum(frame, axis=0).reshape(-1,1)) # summing the y axis

    frame_num_y = np.hstack(frame_num_y)
    frame_num_x = np.hstack(frame_num_x)

    visualize_frame(np.add.reduce(triggered_pixel_sum_l1_frames), exp_tree, all_event_triggers, "summed sum_l1", False)

    visualize_frame_num_relation(frame_num_y, event_triggers_by_frame, "pix_row", "f(frame_num) = \sum_{frame_num} x", False)
    visualize_frame_num_relation(frame_num_x, event_triggers_by_frame, "pix_col", "f(frame_num) = \sum_{frame_num} y", False)

    visualize_frame(np.maximum.reduce(triggered_pixel_thr_l1_frames), exp_tree, all_event_triggers, "maximum thr_l1", False)
    visualize_frame(np.maximum.reduce(triggered_pixel_persist_l1_frames), exp_tree, all_event_triggers, "maximum persist_l1", False)


    # consider pixels mask
    if pixels_mask is None:
        pixels_mask = np.ones_like(event_frames[0])

    weights_mask = np.ones_like(event_frames[0])
    # sum_l1
    # persist_l1
    weights_mask = np.multiply(weights_mask, pixels_mask) # applying mask, should be last

    max_values_arr = np.maximum.reduce(event_frames)
    sum_values_arr = np.add.reduce(event_frames)

    visualize_frame(max_values_arr, exp_tree, all_event_triggers, "max_values_arr", False)
    visualize_frame(sum_values_arr, exp_tree, all_event_triggers, "sum_values_arr", False)

    gray_hough_line(max_values_arr)

    print(len(frames))

    # hough_line()
    # hough_line_peaks()

    plt.show()

    return None


def main(argv):
    parser = argparse.ArgumentParser(description='Find patterns inside triggered pixes')
    # parser.add_argument('files', nargs='+', help='List of files to convert')
    parser.add_argument('-a', '--acquisition-file', help="ACQUISITION root file in \"Lech\" format")
    parser.add_argument('-k', '--kenji-l1trigger-file', help="L1 trigger root file in \"Kenji\" format")
    parser.add_argument('-c', '--corr-map-file', default=None, help="Corrections map .npy file")
    parser.add_argument('--gtu-before', type=int, default=6, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--gtu-after', type=int, default=6, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--persistency-depth', type=int, default=2, help="Number of GTU included in track finding data before the trigger")
    parser.add_argument('--packet-size', type=int, default=128, help="Number of GTU in packet")

    parser.add_argument('--start-gtu', type=int, default=0, help="GTU before will be skipped")

    parser.add_argument('--filter-n-persist-gt', type=int, default=-1, help="Accept only events with at least one GTU with nPersist more than this value.")
    parser.add_argument('--filter-sum-l1-pdm-gt', type=int, default=-1, help="Accept only events with at least one GTU with sumL1PDM more than this value.")
    parser.add_argument('--filter-sum-l1-ec-one-gt', type=int, default=-1, help="Accept only events with at least one GTU with at leas one EC sumL1PDM more than this value.")
    parser.add_argument('--filter-sum-l1-pmt-one-gt', type=int, default=-1, help="Accept only events with at least one GTU with at leas one PMT sumL1PMT more than this value.")

    args = parser.parse_args()

    filter_options = EventFilterOptions()
    filter_options.n_persist = args.filter_n_persist_gt
    filter_options.sum_l1_pdm = args.filter_sum_l1_pdm_gt
    filter_options.sum_l1_ec_one = args.filter_sum_l1_ec_one_gt
    filter_options.sum_l1_pmt_one = args.filter_sum_l1_pmt_one_gt

    ack_l1_reader = AckL1EventReader(args.acquisition_file, args.kenji_l1trigger_file)
    # e = next(ack_l1_reader)

    frame_circ_buffer = collections.deque(maxlen=args.gtu_before+args.persistency_depth+args.gtu_after)

    process_event_down_counter = np.inf
    packet_id = -1

    for gtu_pdm_data in ack_l1_reader.iter_gtu_pdm_data():
        frame_circ_buffer.append(gtu_pdm_data)

        gtu_in_packet = gtu_pdm_data.gtu % args.packet_size
        if gtu_in_packet == 0:
            packet_id += 1 # starts at -1

        print_frame_info(gtu_pdm_data)

        if len(gtu_pdm_data.l1trg_events) > 0:
            process_event_down_counter = args.gtu_after

            for l1trg_ev in gtu_pdm_data.l1trg_events:
                if l1trg_ev.packet_id != packet_id:
                    raise Exception("Unexpected L1 trigger event's packet id (actual: {}, expected: {})".format(l1trg_ev.packet_id, packet_id))
                if l1trg_ev.gtu_in_packet != gtu_in_packet:
                    raise Exception("Unexpected L1 trigger event's gtu in packet (actual: {}, expected: {})".format(l1trg_ev.gtu_in_packet, gtu_in_packet))

            # pcd = gtu_pdm_data.photon_count_data
            # if len(pcd) > 0 and len(pcd[0]) > 0:
            #     visualize_frame(gtu_pdm_data, ack_l1_reader.exp_tree)

        if not np.isinf(process_event_down_counter):    #TODO add packet id check
            if process_event_down_counter == 0 or gtu_in_packet == 127:

                if gtu_pdm_data.gtu >= args.start_gtu:
                    # TODO check event - filetr_options
                    process_event(frame_circ_buffer, ack_l1_reader.exp_tree)

                process_event_down_counter = np.inf
            elif process_event_down_counter > 0:
                if len(gtu_pdm_data.l1trg_events) == 0: # TODO this might require increase size of the circular buffer
                    process_event_down_counter -= 1
            else:
                raise Exception("Unexpected value of process_event_down_counter")



if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)

