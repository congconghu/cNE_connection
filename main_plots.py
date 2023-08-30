# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:25:02 2023

@author: Congcong
"""
import os
import glob
import pickle

import pandas as pd
import plot_box as plots
import connect_plots as cplot



# ++++++++++++++++++++++++++++++++++++++++ single units properties +++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------------------------- plot strf of all units------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\connection\data-summary'
units = pd.read_json(os.path.join(datafolder, 'single_units.json'))
figfolder = r'E:\Congcong\Documents\data\connection\figure'
plots.plot_strf_df(units, figfolder, order='strf_ri_z', properties=True, smooth=True)
cplot.plot_waveform_ptd(savefolder=r'E:\Congcong\Documents\data\connection\figure\summary')

# ++++++++++++++++++++++++++++++++++++++++ cNE properties +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------ plot 5ms binned strf of cNEs and member neurons-----------------------------------------
datafolder = r'E:\Congcong\Documents\data\comparison\data-pkl'
figpath = r'E:\Congcong\Documents\data\comparison\figure\cNE-5ms-strf'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)

for idx, file in enumerate(files):
    print('({}/{}) plot 5ms STRFs for {}'.format(idx, len(files), file))
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    plots.plot_5ms_strf_ne_and_members(ne, figpath)
    
cplot.batch_plot_icweight()

# ++++++++++++++++++++++++++++++++++++++++ connection +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# neuron
cplot.batch_plot_ccg()
cplot.batch_plot_pairs_waveform_strf_ccg()
# ne
cplot.batch_plot_ne_neuron_connection_ccg()
# connection properties
cplot.plot_corr_common_target(savefolder=r'E:\Congcong\Documents\data\connection\figure\summary')
cplot.plot_corr_ne_members(savefolder=r'E:\Congcong\Documents\data\connection\figure\summary')
cplot.plot_prob_share_target(savefolder=r'E:\Congcong\Documents\data\connection\figure\summary')
# example ccg
cplot.batch_plot_ne_neuron_connection_ccg_ss()


# ++++++++++++++++++++++++++++++++++++++ example plots +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cplot.batch_plot_strf_ccg()