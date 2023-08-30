# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:13:16 2023

@author: Congcong
"""

# add MGB-A1-cNE-comparison to path
import sys
sys.path.insert(0, 'E:\Congcong\Documents\PythonCodes\MGB-A1-cNE-comparison')

import glob
import pickle
import re
import os
from scipy.io import loadmat
import mat73
import pandas as pd
import numpy as np
from session_class import Session, SingleUnit
import session_toolbox as mtp


# ------------------------------------------- pickle single units-------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\connection\data-Matlab'
files = glob.glob(datafolder + r'\*split.mat', recursive=False)

for idx, file in enumerate(files):
    print('({}/{})Save pickle file:'.format(idx+1, len(files)), file)
    session = mtp.Session()
    session.read_mat_file(file)
    file_pkl = re.sub('-spk-curated-split.mat', '.pkl', file)
    file_pkl = re.sub('Matlab', 'pkl', file_pkl)
    session.save_pkl_file(file_pkl)
    
# -------------------------------------------- pickle stimulus file ------------------------------------------------
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
stimfile = r'rn1-500flo-32000fhi-0-4SM-0-40TM-40db-96khz-48DF-10min-seed220711_DFt1_DFf5_stim.mat'
stimfile_pkl = re.sub('_stim.mat', '.pkl', stimfile)

stimfile_path = os.path.join(stimfolder, stimfile)
stimfile_pkl_path = os.path.join(stimfolder, stimfile_pkl)

# save stimulus files from mat to pickle
stim = mtp.Stimulus()
stim.read_mat_file(stimfile_path)
stim.save_pkl_file(stimfile_pkl_path)

# --------------------------------------------get stimulus mtf --------------------------------------------------------
stimfolder = r'E:\Congcong\Documents\stimulus\thalamus'
matfile = r'rn1-500flo-32000fhi-0-4SM-0-40TM-40db-96khz-48DF-10min-seed220711_DFt1_DFf5_mtf'
mtf = loadmat(os.path.join(stimfolder, matfile))
savefile = re.sub('mtf.mat', 'mtf.pkl', matfile)
with open(os.path.join(stimfolder, savefile), 'wb') as outfile:
    pickle.dump(mtf, outfile, pickle.HIGHEST_PROTOCOL)


# ++++++++++++++++++++++++++++++++++++ data from James +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------------------------------------- stim param ------------------------------------------------------------
datafolder = r'E:\Congcong\Documents\data\cne_ftc\data'
stim_time_files = glob.glob(os.path.join(datafolder, 'stim_times*.mat'))

for stim_time_file in stim_time_files:
    
    stim_times = mat73.loadmat(stim_time_file)
    stim_times = pd.DataFrame({'stim_onset': stim_times['stim_onset'][600:],
                               'stim_offset': stim_times['stim_offset'][600:],
                              'freq': stim_times['stim_struct']['stim_val']})
    stim_times.to_csv(re.sub('.mat', '.csv', stim_time_file))

# ---------------------------------------  units to session -------------------------------------------------
unit_folder = r'E:\Congcong\Documents\data\cne_ftc\data\unit data'
savefolder = r'E:\Congcong\Documents\data\cne_ftc\data\units_pkl'
unit_files = glob.glob(os.path.join(unit_folder, '*.mat'))
exps = [re.search('(?<=expt)\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file)[0] for file in unit_files]
exps = np.unique(exps)
for exp in exps:
    unit_files = glob.glob(os.path.join(unit_folder, f'*{exp}*.mat'))
    units = []
    for unit_file in unit_files:
        unit = int(re.search('(?<=unit)\d{3}', unit_file)[0])
        unit_data = mat73.loadmat(unit_file)
        su = SingleUnit(unit, unit_data['spiketimes'])
        su.add_waveform_info(unit_data['wave_mean'], unit_data['wave_sd'])
        units.append(su)
    session_data = Session(exp, units)
    with open(os.path.join(savefolder, f'units_{exp}.pkl'), 'wb') as f:
        pickle.dump(session_data, f)
