# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:05:35 2023

@author: Congcong
"""

# add MGB-A1-cNE-comparison to path
import sys
sys.path.insert(0, 'E:\Congcong\Documents\PythonCodes\MGB-A1-cNE-comparison')

import glob
import pickle
import re
import os
import session_toolbox as st
import connect_toolbox as ct
from helper import get_stim
import ne_toolbox as netools
import connect_plots as cplot


# ++++++++++++++++++++++++++++++++++++++++++++ single unit properties ++++++++++++++++++++++++++++++++++++++++++++++++
# -------------------------------------------- get strf -----------------------------------------------------
ct.batch_get_strf()
ct.batch_get_strf_properties()
# sanity check, plot all strfs
cplot.batch_plot_strf()

# -------------------------------------------- split spike trains to spon and dmr -----------------------------------
datafolder = r'E:\Congcong\Documents\data\connection\data-pkl'
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    
    # load stim file
    stimfile = session.stimfile
    stimfile = re.sub('_stim.mat', '.pkl', stimfile)
    with open(os.path.join(stimfolder, stimfile), 'rb') as f:
        stim = pickle.load(f)
        
    # save spktrains
    print('({}/{}) Save spktrain for {}'.format(idx+1, len(files), file))
    session.save_spktrain_from_stim(stim.stim_mat.shape[1])

ct.batch_get_wavefrom_ptd()

# -------------------------------------------- get response properties-----------------------------------
datafolder = r'E:\Congcong\Documents\data\connection\data-pkl'
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    
    # get stimulus for strf calculation (spectrogram)
    stimfile = session.stimfile
    stimfile = re.sub('_stim.mat', '.pkl', stimfile)
    with open(os.path.join(stimfolder, stimfile), 'rb') as f:
        stim_strf = pickle.load(f)
    stim_strf.down_sample(df=10)

    # get stimulus for crh calculation (mtf)
    stimfile = stimfile[:-4] + '-mtf' + stimfile[-4:]
    with open(os.path.join(stimfolder, stimfile), 'rb') as f:
        stim_crh = pickle.load(f)    
        
    
    print('({}/{}) processing {}'.format(idx + 1, len(files), file))
    print('get unit positions')
    session.get_unit_position()
    
    print('get 5ms binned strf')
    session.get_strf(stim_strf)
    session.save_pkl_file(session.file_path)

st.save_su_df(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
               savefolder=r'E:\Congcong\Documents\data\connection\data-summary')


# ++++++++++++++++++++++++++++++++++++++++++++ cNE analysis +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------------------------------------------get cNEs-------------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\connection\data-pkl'
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
files = files
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        session = pickle.load(f)
    
    # load stimulus
    stimfile = session.stimfile
    stimfile = re.sub('_stim.mat', '.pkl', stimfile)
    with open(os.path.join(stimfolder, stimfile), 'rb') as f:
        stim = pickle.load(f)
        
    # save spktrains
    if not hasattr(session, 'spktrain_dmr'):
        print('({}/{}) Save spktrain for {}'.format(idx+1, len(files), file))
        session.save_spktrain_from_stim(stim.stim_mat.shape[1])
    # cNE analysis
    print('({}/{}) Get cNEs for {}'.format(idx+1, len(files), file))
    for stim in ['dmr', 'spon']:
        savefile_path = re.sub(r'fs20000.pkl', 'fs20000-ne-20dft-{}.pkl'.format(stim), session.file_path)
        ne = session.get_ne(df=20, stim=stim)
        if ne:
            ne.save_pkl_file(savefile_path)
            
# ----------------------------------------------get activities and members-----------------------------------------------
# the ne spikes of members are spikes from individual members in 10ms intervals where activity is above threshold
# ne spike of cNE is the last spike in the 10ms interval where activity is above the threshold
datafolder = r'E:\Congcong\Documents\data\connection\data-pkl'
files = glob.glob(datafolder + r'\*20dft-*.pkl', recursive=False)
alpha = 99.5
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    print('({}/{}) get cNE activities and members for {}'.format(idx + 1, len(files), file))
    if not hasattr(ne, 'ne_activity'):
        print('get members and activity')
        ne.get_members()
        ne.get_activity(member_only=True)
        print('get activity threshold')
        ne.get_activity_thresh()
        ne.save_pkl_file(ne.file_path)
    if not hasattr(ne, 'ne_units'):
        print('get ne spikes')
        ne.get_ne_spikes(alpha=alpha)
        ne.save_pkl_file(ne.file_path)

# ------------------------------------------------ get member xcorr ------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\connection\data-pkl'
files = glob.glob(datafolder + r'\*fs20000.pkl', recursive=False)
xcorr = netools.get_member_nonmember_xcorr(files, df=2, maxlag=200)
xcorr.to_json(r'E:\Congcong\Documents\data\connection\data-summary\member_nonmember_pair_xcorr.json')

# -------------------------------------------- get response properties of cNEs ---------------------------------
datafolder = r'E:\Congcong\Documents\data\connection\data-pkl'
files = glob.glob(datafolder + r'\*20dft-dmr.pkl', recursive=False)
for idx, file in enumerate(files):
    with open(file, 'rb') as f:
        ne = pickle.load(f)
    print('({}/{}) get cNE response properties for {}'.format(idx + 1, len(files), file))
    session = ne.get_session_data()
    stimfile = session.stimfile
    stim, stim_crh = get_stim(stimfile)
    stim.down_sample(df=10)
    
    # get ne strf and strf properties
    print('get 5ms binned strf')
    ne.get_strf(stim)
    ne.save_pkl_file(ne.file_path)




    
