# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:56:13 2023

@author: Congcong
"""
import glob
import os
import pickle


# split spike times and add unit info
datafolder =  r'E:\Congcong\Documents\data\cne_ftc\data'
stim_times_folder = datafolder
unit_info_folder = datafolder
unitfolder = os.path.join(datafolder, 'units_pkl')

unit_files = glob.glob(os.path.join(unitfolder, 'units_*.pkl'))
for file in unit_files:
    with open(file, 'rb') as f:
        session = pickle.load(f)
    
    session.split_unit_spiketimes(stim_times_folder)
    session.add_unit_info(unit_info_folder)

    session.save_pkl_file(savefile_path=file)


# get response to ftc
datafolder =  r'E:\Congcong\Documents\data\cne_ftc\data'
stim_times_folder = datafolder
unit_info_folder = datafolder
unitfolder = os.path.join(datafolder, 'units_pkl')
figfolder = os.path.join(datafolder, 'figure', 'ftc')

unit_files = glob.glob(os.path.join(unitfolder, 'units_*.pkl'))
for file in unit_files:
    with open(file, 'rb') as f:
        session = pickle.load(f)
        
    session.calc_response_to_ftc(stim_times_folder)
    session.save_pkl_file()

    session.plot_response_to_ftc(figfolder)
    

# get connected_pairs
