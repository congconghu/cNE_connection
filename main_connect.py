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
for stim in ('spon', 'dmr'):
    ct.batch_get_ne_pair_ccg(stim=stim)
    ct.batch_get_efficacy_ne_ccg(stim=stim)
    ct.batch_label_target_cell_type(stim=stim)
    ct.batch_inclusion(stim=stim)
    ct.combine_df(f'*pairs-ne-{stim}.json', f'ne-pairs-{stim}.json')
    ct.batch_get_effiacay_change_significance(stim=stim)
    ct.batch_get_effiacay_coincident_spk(stim=stim)

# plot ccg on ne-A1 connection
cplot.batch_plot_ne_neuron_connection_ccg()

# correlation of MGB neurons sharing A1 target
ct.get_corr_common_target()
ct.batch_get_effiacay_change_significance()

for stim in ('spon', 'spon_ss'):
    for window in (10, 5, 2):
        ct.batch_get_effiacay_coincident_spk(stim=stim, window=window)
        ct.batch_test_effiacay_coincident_spk(stim=stim, window=window)




# analysis with 100ms cNEs
# ++++++++++++++++++++++++++++++++++++++++++++++ cNE analysis  ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------------------------ get cNEs-------------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\connection\data-pkl'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
for df in [4, 10, 40, 80, 160, 200, 320, 500, 1000, 2000]:
    for idx, file in enumerate(files[1::2]):
        with open(file, 'rb') as f:
            session = pickle.load(f)
        # cNE analysis
        print('({}/{}) Get cNEs for {}'.format(idx+1, len(files), file))
        stim = 'spon'
        savefile_path = re.sub(r'fs20000.pkl', f'fs20000-ne-{df}dft-{stim}.pkl', session.file_path)
        ne = session.get_ne(df=df, stim=stim)
        if ne:
            ne.save_pkl_file(savefile_path)
# ----------------------------------- get cNE activity and membership-------------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\connection\data-pkl'
alpha = 99.5

for df in [4, 10, 40, 80, 160, 200, 320, 500, 1000, 2000]:
    files = glob.glob(datafolder + r'\*{}dft-spon.pkl'.format(df), recursive=False)
    
    for idx, file in enumerate(files):
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        print('({}/{}) get cNE activities and members for {}'.format(idx + 1, len(files), file))
        if not hasattr(ne, 'ne_activity'):
            print('get members and activity')
            ne.get_members()
            ne.get_activity(member_only=True)
            ne.get_activity_thresh()
            ne.save_pkl_file(ne.file_path)
        if not hasattr(ne, 'ne_units'):
            print('get ne spikes')
            ne.get_ne_spikes(alpha=alpha)
            ne.save_pkl_file(ne.file_path)
        
# +++++++++++++++++++++++++++++++++++++++++ MGB cNE and A1 neurons connect +++++++++++++++++++++++++++++++++++++++
# -------------------------------------------- get ccg of NE spikes and A1 neurons ------------------------------
stim = 'spon'
for binsize in [0.5, 1, 2]:
    for df in [4, 10, 20, 40, 80, 160, 200, 320, 500, 1000, 2000]:
       ct.batch_get_ne_pair_ccg(stim=stim, df=df, binsize=binsize)
       ct.batch_get_efficacy_ne_ccg(stim=stim, df=df, binsize=binsize)
       ct.batch_label_target_cell_type(stim=stim, df=df, binsize = binsize)
       ct.batch_inclusion(stim=stim, df=df, binsize=binsize)
       ct.combine_df(f'*pairs-ne-{df}-{stim}-{binsize}ms_bin.json', f'ne-pairs-{df}df-{stim}-{binsize}ms_bin.json')

# plot ne neuron ccg
stim = 'spon'
for binsize in [0.5, 1, 2]:
    for df in [10, 20]:
        file_id = f'{df}-spon-{binsize}ms_bin'
        figfolder = r"E:\Congcong\Documents\data\connection\figure\ccg_{}ms_{}dft".format(binsize, df)
        cplot.batch_plot_ne_neuron_connection_ccg(file_id = file_id, df=df, figfolder=figfolder)
# analysis for 5ms cNEs
ct.batch_get_efficacy_ne_ccg(datafolder, stim='spon', subsample=True, df=10)
# single spike
stim='spon_ss'
df=10
ct.batch_get_ne_pair_ccg(stim=stim, df=df)
ct.batch_get_efficacy_ne_ccg(stim=stim, df=df)
ct.batch_label_target_cell_type(stim=stim, df=df)
ct.combine_df(f'*pairs-ne-{df}-{stim}-0.5ms_bin.json', f'ne-pairs-{df}df-{stim}-0.5ms_bin.json')
# coincident spikes
ct.batch_get_effiacay_coincident_spk(stim='spon', df=10)

ct.batch_get_A1_nspk_after_MGB_spike(df=10)
