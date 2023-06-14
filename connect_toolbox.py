# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:31:54 2023

@author: Congcong
"""
# add MGB-A1-cNE-comparison to path
import sys
sys.path.insert(0, 'E:\Congcong\Documents\PythonCodes\MGB-A1-cNE-comparison')

import glob
import os
import re
import pickle
import json
import itertools

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy import stats
from plot_box import plot_strf
import ne_toolbox as netools


def load_input_target_files(datafolder, file_id):
    input_target_files = glob.glob(os.path.join(datafolder, f'{file_id}*fs20000.pkl'))
    assert(len(input_target_files) == 2)
    for file in input_target_files:
        with open(file, 'rb') as f:
            session = pickle.load(f)
        if session.depth > 3000: # if the probe reached morthan 3000um down the surface, the region is MGB
            input_units = session.units
        else:
            target_units = session.units
    return input_target_files, input_units, target_units, session.trigger

def load_input_target_session(datafolder, file_id):
    input_target_files = glob.glob(os.path.join(datafolder, f'{file_id}*fs20000.pkl'))
    assert(len(input_target_files) == 2)
    for file in input_target_files:
        with open(file, 'rb') as f:
            session = pickle.load(f)
        if session.depth > 3000: # if the probe reached morthan 3000um down the surface, the region is MGB
            input_session = session
        else:
            target_session = session
    return input_target_files, input_session, target_session

def batch_save_connection_pairs(datafolder:str=r'E:\Congcong\Documents\data\connection\data-pkl', 
                                savefolder:str=r'E:\Congcong\Documents\data\connection\data-pkl',
                                thresh=.999):
    """
    call get_connected_pairs to find connected pairs for all session in the datafolder 

    Parameters
    ----------
    datafolder : str, optional
        The folder where single unit data is saved. 
        The files of single units end with fs20000.pkl and is used to identify single unit files.
        The default is 'E:\Congcong\Documents\data\connection\data-pkl'.
    savefolder : str, optional
        folder to save data of connected pairs.
        files saved with suffix '-pairs'
        The default is 'E:\Congcong\Documents\data\connection\data-pkl'.

    Returns
    -------
    None.

    """
    files = glob.glob(os.path.join(datafolder, '*fs20000.pkl'))
    exp = [re.search('\d{6}_\d{6}', file).group(0) for file in files]
    exp = set(exp)
    for i, file_id in enumerate(exp):
        print('{}/{} get connected pairs in recording {}'.format(i, len(exp), file_id))
        input_target_files, input_units, target_units, triggers = load_input_target_files(datafolder, file_id)
        filename = re.search('\d{6}_\d{6}.*', input_target_files[-1]).group(0)
        filename = re.sub('fs20000.pkl', 'fs20000-pairs.json', filename)
        savename = os.path.join(savefolder, filename)

        pairs = get_connected_pairs(input_units, target_units, triggers=triggers, thresh=thresh)
        if pairs is not None:
            pairs.to_json(savename)
        
    
def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.sosfiltfilt(sos, data)
        return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    cutoff = cutoff / nyq
    b, a = signal.butter(order, cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def get_ccg(spiketimes_input, spiketimes_target, window_size=50, binsize=.5, ss=False, isi=[]):
    """
    get cross correlation of two spike trains

    Parameters
    ----------
    spiketimes_input : TYPE
        DESCRIPTION.
    spiketimes_target : TYPE
        DESCRIPTION.
    window_size : TYPE, optional
        DESCRIPTION. The default is 50.
    binsize : TYPE, optional
        DESCRIPTION. The default is .5.

    Returns
    -------
    ccg : TYPE
        DESCRIPTION.

    """
    ccg = 0
    edges = np.arange(-window_size, window_size+binsize, binsize)
    if ss: # only consider single spikes (spikes with ISI > 20ms)
        if not any(isi):
            isi = get_isi(spiketimes_input)
        spiketimes_input = spiketimes_input[isi > 20]
    
    for spk in spiketimes_input:
        ccg_tmp, _ = np.histogram(spiketimes_target - spk, edges) 
        ccg += ccg_tmp
    return ccg, edges, len(spiketimes_input)


def get_isi(spiketimes:np.array):
    """
    get the interval to the closest spike

    Parameters
    ----------
    spiketimes : np.array
        spike times of the unit, in ms

    Returns
    -------
    isi :  np.array
        distance to the closest spike for each spike in spiketimes.

    """
    isi = np.zeros(len(spiketimes) + 1)
    isi[1:-1] = np.diff(spiketimes)
    isi[0] = isi[1]
    isi[-1] = isi[-2]
    
    isi = [min(isi[i:i+2]) for i in range(len(isi)-1)]
    return np.array(isi)

def get_causal_window_idx(ccg:np.array, binsize=.5, window=[1, 5]):
    ccg = np.array(ccg)
    idx_t0 = ccg.shape[0] // 2
    idx_delay_window = np.arange(idx_t0 + window[0]/binsize, idx_t0 + window[1]/binsize).astype(int)
    return idx_delay_window

def check_connection(ccg, stim, binsize=.5, alpha=.999):
    
    # get 1-5ms delay window index
    idx_delay_window = get_causal_window_idx(ccg, binsize=.5)
    peak_idx = np.argmax(ccg)
    
    # initialize return values
    connect = False
    ccg_filtered = []
    thresh = []
    baseline=[]
    hw = None
    
    if (peak_idx in idx_delay_window): # ccg peak with 1-5ms delay
        baseline =  get_baseline(ccg)
        ccg_filtered = ccg - baseline
        peak_idx_filtered = np.argmax(ccg-baseline)
        if (peak_idx_filtered in idx_delay_window):
            # ccg peak above baseline with 1-5ms delay
            # ccg bove baseline have more than 50 spikes
            thresh = stats.poisson.ppf(alpha, baseline)
            if check_consecutive_above_thresh(ccg, thresh, idx_delay_window, peak_idx_filtered):
                connect = True
                # connect, hw = check_hw(ccg_filtered, binsize)
    return connect, {f'ccg_{stim}': [ccg], f'ccg_filtered_{stim}': [ccg_filtered], f'baseline_{stim}': [baseline],
                     f'thresh_{stim}': [thresh], f'sig_{stim}': connect, f'hw_{stim}': hw}


def get_baseline(ccg, binsize=.5, method='gaussian'):
    
    if method == 'gaussian':
       baseline = gaussian_filter1d(ccg.astype('float'), sigma=7/binsize, mode='nearest')
    elif method == 'bandpass':
        baseline =  butter_lowpass_filter(ccg, 20, fs=1e3/binsize)
    return baseline

def check_consecutive_above_thresh(ccg, thresh, idx_delay_window, idx_peak, method=None):
    if method == 'peak':
        return (ccg[idx_peak - 1] > thresh[idx_peak - 1]) or (ccg[idx_peak + 1] > thresh[idx_peak + 1])
    sig = np.where(ccg[idx_delay_window] > thresh[idx_delay_window])[0]
    return 1 in np.diff(sig)


def check_hw(ccg, binsize):
    idx_t0 = ccg.shape[0] // 2
    idx_window = np.array(range(idx_t0 - 12, idx_t0 + 12))
    peak = ccg.max()
    hh = peak / 2 # half height
    idx_hh = np.where(ccg[idx_window] > hh)[0]
    hw = (idx_hh[-1] - idx_hh[0] + 1) * binsize
    return hw <= 3, hw
    

def get_connected_pairs(input_units, target_units, triggers, window_size=50, binsize=.5, thresh=.999):
    pairs = None
    n_check = len(input_units) * len(target_units)
    checked = 0
    if triggers.ndim == 1:
        triggers = [triggers]
    new_pair = {}
    
    for input_idx, input_unit in enumerate(input_units):
        for target_idx, target_unit in enumerate(target_units):
            checked += 1
            new_pair.update({'input_idx': input_idx, 'target_idx': target_idx, 
                             'input_unit': input_unit.unit, 'target_unit': target_unit.unit})
            print(f'{checked}/{n_check}')
            
            for stim in ('spon', 'spon_ss', 'dmr', 'dmr_ss'):
                if 'spon' in stim:
                    input_spiketimes = input_unit.spiketimes_spon
                    target_spiketimes = target_unit.spiketimes_spon
                elif 'dmr' in stim:
                    input_spiketimes = input_unit.spiketimes_dmr
                    target_spiketimes = target_unit.spiketimes_dmr
                if 'ss' in stim:
                    isi = get_isi(input_spiketimes)
                    input_spiketimes = input_spiketimes[isi > 20]
                    
                ccg, edges, nspk = get_ccg(input_spiketimes, target_spiketimes, 
                                           window_size=window_size, binsize=binsize)
                connect, ccg_dict = check_connection(ccg, stim, alpha=thresh)
                new_pair.update(ccg_dict)
                new_pair.update({f'nspk_{stim}': nspk})
            
                if stim == 'spon':
                    if not new_pair['sig_spon']:
                        new_pair = {}
                        break
                    else:
                        taxis = (edges[1:] + edges[:-1]) / 2
                        new_pair.update({'taxis': [taxis]})
            if new_pair:
                if pairs is None:
                    pairs = pd.DataFrame(new_pair)
                else:
                    pairs = pd.concat([pairs, pd.DataFrame(new_pair)], ignore_index=True)
                     
    return pairs

def refine_connected_pairs(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl\connect-filter-old', 
                           savefolder=r'E:\Congcong\Documents\data\connection\data-pkl\connect-filter'):
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    for file in files:
        print(f'Processing {file}')
        pairs_old = pd.read_json(file)
        pairs = None
        for i in pairs_old.index:
            new_pair = dict(pairs_old.iloc[i])
            new_pair['taxis'] = [new_pair['taxis']]
            for stim in ('spon', 'spon_ss', 'dmr', 'dmr_ss'):
                ccg = np.array(eval(f'new_pair[\'ccg_{stim}\']'))
                connect, ccg_dict = check_connection(ccg, stim)
                new_pair.update(ccg_dict)
            if pairs is None:
                pairs = pd.DataFrame(new_pair)
            else:
                pairs = pd.concat([pairs, pd.DataFrame(new_pair)], ignore_index=True)
        filename = re.search('\d{6}_\d{6}.*', file).group(0)
        pairs.to_json(os.path.join(savefolder, filename))
  

def batch_get_ne_pair_ccg(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
                          savefolder=r'E:\Congcong\Documents\data\connection\data-pkl', stim='spon'):
    files = glob.glob(datafolder + r'\*pairs.json', recursive=False)
    for idx, file in enumerate(files):
        print('{}/{} Processing {}'.format(idx+1, len(files), file))
        nepairs = get_ne_pair_ccg(file, stim, datafolder=datafolder)
        if nepairs is None:
            continue
        filename = re.search('\d{6}_\d{6}.*', file).group(0)
        filename = re.sub('pairs', f'pairs-ne-{stim}', filename)
        nepairs.to_json(os.path.join(savefolder, filename))
        
        
def get_ne_pair_ccg(pairfile, stim='spon', datafolder=r'E:\Congcong\Documents\data\connection\data-pkl'):
    
    # load piars
    pairs = pd.read_json(pairfile)
    pairs = pairs[pairs.sig_spon]
    if len(pairs) < 1:
        return None
    # load single units
    exp = re.search('\d{6}_\d{6}', pairfile).group(0)
    _, input_units, target_units, trigger = load_input_target_files(datafolder, exp)
    #load ne
    s = stim.split("_")[0]
    nefile = glob.glob(os.path.join(datafolder, f'{exp}*-20dft-{s}.pkl'))[0]
    with open(nefile, 'rb') as f:
        ne = pickle.load(f)
    
    nepairs = None
    connected_input = pairs.input_idx.unique()
    for cne, members in ne.ne_members.items():
        ne_connection = [x in connected_input for x in members] # member neurons with A1 target
        if any(ne_connection):
            ne_inputs = members[ne_connection]
            ne_targets = pairs.loc[pairs['input_idx'].isin(ne_inputs), 'target_idx'].unique()
            for target in ne_targets:
                target_unit = target_units[target]
                for member_idx, member in enumerate(members):
                    input_unit = input_units[member]
                    ne_unit = ne.member_ne_spikes[cne][member_idx]
                    new_pair = get_ne_neuron_ccg(input_unit, ne_unit, target_unit, stim=stim)
                    new_pair.update({'target_unit': target_unit.unit, 'target_idx': target, 
                                     'input_unit':input_unit.unit, 'input_idx': member,
                                    'cne': cne})
                    if nepairs is None:
                        nepairs = pd.DataFrame(new_pair)
                    else:
                        nepairs = pd.concat([nepairs, pd.DataFrame(new_pair)], ignore_index=True)
    return nepairs
    
def get_ne_neuron_ccg(input_unit, ne_unit, target_unit, stim='spon'):
    s = stim.split("_")[0]

    # get ccg between neurons
    input_spiketimes = eval(f'input_unit.spiketimes_{s}')
    if 'ss' in stim:
        isi = get_isi(input_spiketimes)
        input_spiketimes = input_spiketimes[isi > 20]
 
    target_spiketimes = eval(f'target_unit.spiketimes_{s}')
    ccg_neuron, edges, nspk_neuron = get_ccg(input_spiketimes, target_spiketimes)
    taxis = [(edges[:-1] + edges[1:]) / 2]
    connect, pair = check_connection(ccg_neuron, f'neuron_{stim}')
    pair.update({f'nspk_neuron_{stim}': nspk_neuron})
    pair.update({'taxis': taxis})

    # get ccg between ne and neuron
    input_spiketimes_ne = ne_unit.spiketimes
    input_spiketimes_ne = np.array(sorted(list(set(input_spiketimes_ne).intersection(set(input_spiketimes)))))
    ccg_ne, edges, nspk_ne = get_ccg(input_spiketimes_ne, target_spiketimes)
    connect, pair_tmp = check_connection(ccg_ne, f'ne_{stim}')
    pair.update(pair_tmp)
    pair.update({f'nspk_ne_{stim}': nspk_ne})
    
    # ccg between nonne and neuron
    ccg_nonne = ccg_neuron - ccg_ne
    connect, pair_tmp = check_connection(ccg_nonne, f'nonne_{stim}')
    pair.update(pair_tmp)
    pair.update({f'nspk_nonne_{stim}': nspk_neuron - nspk_ne})
    
    return pair

def batch_get_baseline_ne_ccg(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', stim='spon'):
    files = glob.glob(os.path.join(datafolder, f'*-pairs-ne-{stim}.json'))
    for file in files:
        pairs = pd.read_json(file)
        for unit_type in ('neuron', 'ne', 'nonne'):
            pairs = update_baseline(pairs, f'ccg_{unit_type}_{stim}')
        pairs.to_json(file)
    
def update_baseline(pairs, field_id):
    field_id_filtered = re.sub('ccg', 'ccg_filtered', field_id)
    field_id_baseline = re.sub('ccg', 'baseline', field_id)
    ccg_filtered = []
    baselines = []
    for i in range(len(pairs)):
        pair = pairs.loc[i]
        ccg = pair[field_id]
        baseline = butter_lowpass_filter(ccg, 20, fs=2e3)
        baselines.append(baseline)
        ccg_filtered.append(ccg - baseline)
    pairs[field_id_filtered] = ccg_filtered
    pairs[field_id_baseline] = baselines
    return pairs


def batch_get_efficacy_ne_ccg(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', stim='spon'):
    files = glob.glob(os.path.join(datafolder, f'*-pairs-ne-{stim}.json'))
    for file in files:
        pairs = pd.read_json(file)
        for unit_type in ('neuron', 'ne', 'nonne'):
            pairs = update_efficacy(pairs, f'_{unit_type}_{stim}')
        pairs.to_json(file)

def batch_get_efficacy(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl'):
    files = glob.glob(os.path.join(datafolder, f'*-pairs.json'))
    for file in files:
        pairs = pd.read_json(file)
        for stim in ('dmr', 'dmr_ss', 'spon', 'spon_ss'):
            pairs = update_efficacy(pairs, f'_{stim}')
        pairs.to_json(file)

def update_efficacy(pairs, field_id):
    efficacy = []
    for i in range(len(pairs)):
        pair = pairs.loc[i]
        ccg = np.array(pair['ccg' + field_id])
        ccg_filtered = np.array(pair['ccg_filtered' + field_id])
        taxis = np.array(pair.taxis)
        nspk = pair['nspk' + field_id]
        efficacy.append(get_efficacy(ccg, ccg_filtered, nspk, taxis))
    pairs['efficacy' + field_id] = efficacy
    return pairs

def get_efficacy(ccg, ccg_filtered, nspk, taxis, method='peak'):
    causal_idx = get_causal_spike_idx(ccg, method)
    baseline = get_causal_spk_baseline(ccg, causal_idx)
    try:
        n_causal_spk = sum(ccg[causal_idx] - baseline)
    except (TypeError, IndexError):
        n_causal_spk = 0
    return n_causal_spk / nspk * 100

def get_causal_spike_idx(ccg, method='peak'):
    if method == 'peak':
        idx_peak = np.argmax(ccg)
        causal_idx = np.array(range(idx_peak - 3, idx_peak + 5))
    elif method == 'window':
        causal_idx = get_causal_window_idx(ccg)
    return causal_idx

def get_causal_spk_baseline(ccg, causal_spk_idx):
    causal_baseline = list(range(causal_spk_idx[0]-4, causal_spk_idx[0])) \
        + list(range(causal_spk_idx[-1]+1, causal_spk_idx[-1]+5))
    try:
        return np.mean(ccg[causal_baseline])
    except IndexError:
        return None

def batch_inclusion(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', stim='spon'):
    files = glob.glob(os.path.join(datafolder, f'*-pairs-ne-{stim}.json'))
    for file in files:
        pairs = pd.read_json(file)
        include = []
        idx = get_causal_window_idx(pairs.loc[0][f'ccg_neuron_{stim}'])
        for i in range(len(pairs)):
            pair = pairs.loc[i]
            include_tmp = True
            # check if there are enough spikes in causal window for a good extimate of efficacy
            for unit_type in ('neuron', 'ne'):
                ccg = np.array(pair[f'ccg_{unit_type}_{stim}'])
                if sum(ccg[idx]) < 20:
                    include_tmp = False
                    include.append(include_tmp)
                    break
            # check if any of the ccg indicate functional connection
            if include_tmp:
                for unit_type in ('neuron', 'ne'):
                    ccg = np.array(pair[f'ccg_{unit_type}_{stim}'])
                    include_tmp, _ = check_connection(ccg, stim)
                    if include_tmp:
                        break
                include.append(include_tmp)
        pairs[f'inclusion_{stim}'] = include
        pairs.to_json(file)

def get_corr_common_target(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
                           savefolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    pairs_corr_all = None
    for file in files:
        pairs = pd.read_json(file)
        multi_input = pairs.groupby('target_idx')['input_idx'].count() > 1
        multi_input = list(multi_input[multi_input].index)
        if any(multi_input):
            su_file = os.path.join(datafolder, re.sub('-pairs.json', '.pkl', file))
            with open(su_file, 'rb') as f:
                session = pickle.load(f)
            spktrain, _ = session.downsample_spktrain(df=20, stim='spon')
            pairs_common_target = []
            target = []
            for target_idx in multi_input:
                inputs_common_target = list(pairs[pairs.target_idx == target_idx]['input_idx'])
                new_pairs = list(itertools.combinations(inputs_common_target, 2))
                pairs_common_target.extend(new_pairs)
                target.extend(list(np.ones(len(new_pairs)) * target_idx))
            target = list(map(int, target))
            input1, input2 = zip(*pairs_common_target)
            pairs_corr = pd.DataFrame({'input1': input1, 'input2': input2, 'target': target})
            all_pairs = set(itertools.combinations(range(spktrain.shape[0]), 2))
            pairs_no_share = all_pairs.difference(set(pairs_common_target))
            input1, input2 = zip(*pairs_no_share)
            pairs_corr = pd.concat([pairs_corr, pd.DataFrame({'input1': input1, 'input2': input2, 'target': -1})])
            pairs_corr.reset_index(inplace=True, drop=True)
            corr = np.corrcoef(spktrain)
            pairs_corr['corr'] = corr[pairs_corr.input1, pairs_corr.input2]
            pairs_corr['exp'] = re.search('\d{6}_\d{6}', file).group(0)
            if  pairs_corr_all is None:
                pairs_corr_all = pairs_corr
            else:
                pairs_corr_all = pd.concat([pairs_corr_all, pairs_corr])
                
    pairs_corr_all.reset_index(inplace=True, drop=True)
    pairs_corr_all.to_json(os.path.join(savefolder, 'pairs_common_target_corr.json'))       
            
            
def prob_share_target(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
                      savefolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    pairs_all = None
    for file in files:
        # get pairs connected to A1
        pairs = pd.read_json(file)
        input_idx = pairs.input_idx.unique()
        input_pairs = set(itertools.combinations(input_idx, 2)) # all pairs of input neurons
        # get pairs in cNEs
        ne_file = re.sub('pairs.json', 'ne-20dft-spon.pkl', file)
        with open(ne_file, 'rb') as f:
            ne = pickle.load(f)
        member_pairs = netools.get_member_pairs(ne) # all member pairs
        # get pairs share target
        common_target_pairs = set() # all pairs of neurons sharing a target
        target_idx =  pairs.target_idx.unique()
        for target in target_idx:
            if len(pairs.target_idx == target):
                common_target_pairs.update(set(itertools.combinations(
                    pairs[pairs.target_idx == target]['input_idx'], 2)))
        is_within_ne = [False] * len(input_pairs)
        is_share_target = [False] * len(input_pairs)
        input1 = []
        input2 = []
        for i, pair in enumerate(input_pairs):
            if pair in common_target_pairs:
                is_within_ne[i] = True
            if pair in member_pairs:
                is_share_target[i] = True
            input1.append(pair[0])
            input2.append(pair[1])
                
        pairs = pd.DataFrame({'input1': input1, 'input2': input2, 
                              'within_ne': is_within_ne, 'share_target': is_share_target})
        pairs['exp'] = re.search('\d{6}_\d{6}', file).group(0)
        if pairs_all is None:
            pairs_all = pairs
        else:
            pairs_all = pd.concat([pairs_all, pairs])
    pairs_all.reset_index(drop=True, inplace=True)
    pairs_all.to_json(os.path.join(savefolder, 'pairs_common_target_ne.json'))       

                
def batch_get_wavefrom_tpd(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl'):
    files = glob.glob(os.path.join(datafolder, '*fs20000.pkl'))
    for file in files:
        
        with open(file, 'rb') as f:
            session = pickle.load(f)
        
        for unit in session.units:
            unit.get_waveform_tpd()
        
        session.save_pkl_file()

def batch_label_target_cell_type(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', stim='spon'):
    files = glob.glob(os.path.join(datafolder, f'*pairs-ne-{stim}.json'))
    for file in files:
        tpd_all = []
        pairs = pd.read_json(file)
        exp = re.search('\d{6}_\d{6}', file).group(0)
        su_file = glob.glob(os.path.join(datafolder, f'{exp}*-fs20000.pkl'))[0]
        with open(su_file, 'rb') as f:
            session = pickle.load(f)
        for i in range(len(pairs)):
            idx = pairs.iloc[i].target_idx
            tpd = session.units[idx].waveform_tpd
            tpd_all.append(tpd)
        pairs['target_waveform_tpd'] = tpd_all
        pairs.to_json(file)
            

def batch_get_strf(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', 
                   stimfolder=r'E:\Congcong\Documents\stimulus\thalamus'):
    spkfiles = glob.glob(os.path.join(datafolder, '*fs20000.pkl'))
    stimfile = ''
    nfiles = len(spkfiles)
    for i, file in enumerate(spkfiles):
        with open(file, 'rb') as f:
            session = pickle.load(f)
        if session.stimfile != stimfile:
            stimfile = session.stimfile
            stimfile_pkl = re.sub('_DFt1_DFf5_stim.mat', '.pkl', stimfile)
            print(f'load stimfile {stimfile_pkl}')
            with open(os.path.join(stimfolder, stimfile_pkl), 'rb') as f:
                stim = pickle.load(f)
        print(f'{i+1}/{nfiles} get strf for {file}')
        session.get_strf(stim=stim, nlead=400, nlag=100)
        session.save_pkl_file(file)
         

def batch_get_strf_properties(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
                        stimfolder=r'E:\Congcong\Documents\stimulus\thalamus'):

    spkfiles = glob.glob(os.path.join(datafolder, '*fs20000.pkl'))
    spkfiles = spkfiles[16:]
    stimfile = ''
    nfiles = len(spkfiles)
    for i, file in enumerate(spkfiles):
        with open(file, 'rb') as f:
            session = pickle.load(f)
        if session.stimfile != stimfile:
            stimfile = session.stimfile
            stimfile_pkl = re.sub('_stim.mat', '.pkl', stimfile)
            print(f'load stimfile {stimfile_pkl}')
            with open(os.path.join(stimfolder, stimfile_pkl), 'rb') as f:
                stim = pickle.load(f)
                stim.down_sample(df=10)
        print(f'{i+1}/{nfiles} get strf properties for {file}')
        session.get_strf_properties()
        print('get strf ri')
        session.get_strf_ri(stim)
        print('get strf sig')
        session.get_strf_significance(criterion='z', thresh=3)

        session.save_pkl_file(file)
        
def batch_get_bf(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl'):
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    for file in files:
        pairs = pd.read_json(file)
        exp = re.search('\d{6}_\d{6}', file).group(0)
        _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
        bf_input = []
        bf_target = []
        for i in pairs.index:
            pair = pairs.loc[i]
            # plot waveform and strfs
            input_unit = input_units[pair.input_idx]
            target_unit = target_units[pair.target_idx]
            if input_unit.strf_sig:
                bf_input.append(input_unit.bf)
            else:
                bf_input.append(None)
            if target_unit.strf_sig:
                bf_target.append(target_unit.bf)
            else:
                bf_target.append(None)
        pairs['input_bf'] = bf_input
        pairs['target_bf'] = bf_target
        pairs.to_json(file)


def combine_pair_file(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
                      savefolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    pairs_all = []
    for file in files:
        pairs = pd.read_json(file)
        pairs['exp'] = re.search('\d{6}_\d{6}', file).group(0)
        pairs_all.append(pairs)
    pairs = pd.concat(pairs_all)
    pairs.reset_index(drop=True, inplace=True)
    pairs.to_json(os.path.join(savefolder, 'pairs.json'))
    print('combined file saved at {}'.format(os.path.join(savefolder, 'pairs.json')))
    
def combine_df(file_id, savename, 
               datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
               savefolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    files = glob.glob(os.path.join(datafolder, file_id))
    pairs_all = []
    for file in files:
        pairs = pd.read_json(file)
        pairs['exp'] = re.search('\d{6}_\d{6}', file).group(0)
        pairs_all.append(pairs)
    pairs = pd.concat(pairs_all)
    pairs.reset_index(drop=True, inplace=True)
    pairs.to_json(os.path.join(savefolder, savename))
    print('combined file saved at {}'.format(os.path.join(savefolder, savename)))


def batch_get_fr(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
                 savefolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    fr_all = {'spon_MGB': [], 'spon_A1': [], 'dmr_MGB': [], 'dmr_A1': []}
    for file in files:
        pairs = pd.read_json(file)
        exp = re.search('\d{6}_\d{6}', file).group(0)
        _, MGB_session, A1_session = load_input_target_session(datafolder, exp)
        for region in ('MGB', 'A1'):
            for stim in ('dmr', 'spon'):
                spktrain = eval(f'{region}_session.spktrain_{stim}')
                spktrain = np.concatenate(spktrain, axis=1)
                fr = np.sum(spktrain, axis=1) / spktrain.shape[1] * 2e3
                fr_all[f'{stim}_{region}'].extend(fr)
                for i, unit in enumerate(eval(f'{region}_session.units')):
                    setattr(unit, f'fr_{stim}', fr[i])
        
        fr = {'spon_MGB': [], 'spon_A1': [], 'dmr_MGB': [], 'dmr_A1': []}
        for i in pairs.index:
            pair = pairs.loc[i]
            # plot waveform and strfs
            MGB_unit = MGB_session.units[pair.input_idx]
            A1_unit = A1_session.units[pair.target_idx]
            for region in ('MGB', 'A1'):
                for stim in ('spon', 'dmr'):
                    fr[f'{stim}_{region}'].append(eval(f'{region}_unit.fr_{stim}'))
        for stim in ('spon', 'dmr'):
            pairs[f'input_fr_{stim}'] = fr[f'{stim}_MGB']
            pairs[f'target_fr_{stim}'] = fr[f'{stim}_A1']
        pairs.to_json(file)
    with open(os.path.join(savefolder, 'fr_all.json'), 'w') as outfile: 
        json.dump(fr_all, outfile)
    
    