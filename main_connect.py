# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:53:20 2023

@author: Congcong
"""
import glob
import pickle
import re
import sys
sys.path.insert(0, 'E:\Congcong\Documents\PythonCodes\MGB-A1-cNE-comparison')
import pandas as pd
import os
import plot_box as plots
import connect_plots as cplot
import connect_toolbox as ct

# ++++++++++++++++++++++++++++++++++++++++++++ get connected pairs +++++++++++++++++++++++++++++++++++++++++++++++
# get connected pairs, saves as xxx-pairs.pkl
ct.batch_save_connection_pairs(datafolder='E:\Congcong\Documents\data\connection\data-pkl', 
                                savefolder='E:\Congcong\Documents\data\connection\data-pkl')
# sanity check, plot ccg (spon, dmr, all & ss)
cplot.batch_plot_ccg()
# get efficacy of connected pairs
ct.batch_get_efficacy()
# get bf of the two neurons in the pairs
ct.batch_get_bf()
# plot neuronal pair waveform, STRF and CCG
cplot.batch_plot_pairs_waveform_strf_ccg()
ct.batch_get_fr()
ct.combine_pair_file()

#get relationship of firing correlation and connectivity
ct.get_corr_common_target()
ct.prob_share_target()


# +++++++++++++++++++++++++++++++++++++++++ MGB cNE and A1 neurons connect +++++++++++++++++++++++++++++++++++++++
# -------------------------------------------- get ccg of NE spikes and A1 neurons ------------------------------
for stim in ('spon', 'spon_ss', 'dmr', 'dmr_ss'):
    ct.batch_get_ne_pair_ccg(stim=stim)
    ct.batch_get_efficacy_ne_ccg(stim=stim)
    ct.batch_label_target_cell_type(stim=stim)
    ct.batch_inclusion(stim=stim)
    ct.combine_df(f'*pairs-ne-{stim}.json', f'ne-pairs-{stim}.json')
    ct.batch_get_effiacay_change_significance(stim=stim)
    ct. batch_get_effiacay_coincident_spk(stim=stim)

# plot ccg on ne-A1 connection
cplot.batch_plot_ne_neuron_connection_ccg()

# correlation of MGB neurons sharing A1 target
ct.get_corr_common_target()
ct.batch_get_effiacay_change_significance()

for stim in ('spon', 'spon_ss'):
    for window in (10, 5, 2):
        ct.batch_get_effiacay_coincident_spk(stim=stim, window=window)
        ct.batch_test_effiacay_coincident_spk(stim=stim, window=window)

