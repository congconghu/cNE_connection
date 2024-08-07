# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:07:26 2023

@author: Congcong
"""
import os
import glob
import re
import pickle
import json

import pandas as pd
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from plot_box import (
    plot_strf,
    plot_raster,
    plot_ICweight,
    plot_activity,
    boxplot_scatter, 
    plot_significance_star, 
    set_violin_half,
    plot_corrmat,
    plot_ICweigh_imshow,
    plot_eigen_values,
)
from connect_toolbox import load_input_target_files
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ne_toolbox as netools
import connect_toolbox as ct
import pickle as pkl
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols 
import statsmodels.api as sm 

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'Arial'

mpl.rcParams['axes.linewidth'] = .6
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['axes.labelpad'] = 2
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

mpl.rcParams['xtick.major.width'] = .6
mpl.rcParams['ytick.major.width'] = .6
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.major.pad'] = 1.5
mpl.rcParams['ytick.major.pad'] = .5
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7

mpl.rcParams['lines.linewidth'] = .8
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['patch.linewidth'] = .6

cm = 1/2.54  # centimeters in inches
figure_size = [(17.6*cm, 17*cm), (11.6*cm, 17*cm), (8.5*cm, 17*cm)]
activity_alpha = 99.5
colors = sns.color_palette("Paired")
tpd_color = (colors[5], colors[1], colors[0], colors[4])
A1_color = (colors[1], colors[0])
MGB_color = (colors[5], colors[4])
colors_split = [colors[i] for i in [7, 6, 3, 2, 9, 8]]
colors = sns.color_palette("colorblind")
ne_hiact_color = [colors[i] for i in [2, 6]]
fontsize_figure_axes_label = 8
fontsize_figure_tick_label = 7
fontsize_panel_label = 12
marker_size = 10


# ------------------------------------ccg of single pairs---------------------------------------------------------
def batch_plot_ccg(datafolder:str=r'E:\Congcong\Documents\data\connection\data-pkl', 
                   figfolder:str=r'E:\Congcong\Documents\data\connection\figure\ccg'):
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    for file in files:
        pairs = pd.read_json(file)
        exp = re.search('\d{6}_\d{6}', file).group(0)
        for i in pairs.index:
            if not pairs.loc[i].sig_spon:
                continue
            fig, axes = plt.subplots(2, 2, figsize=[16, 8])
            axes = axes.flatten()
            for j, stim in enumerate(('spon', 'dmr', 'spon_ss', 'dmr_ss')):
                axes_tmp = axes[j]
                axes_tmp.set_title('{} sig-{}'.format(re.sub('_', ' ', stim), pairs.iloc[i]['sig_' + stim]))
                ccg = np.array(pairs.iloc[i]['ccg_' + stim])
                baseline = np.array(pairs.iloc[i]['baseline_' + stim])
                thresh = np.array(pairs.iloc[i]['thresh_' + stim])
                taxis = np.array(pairs.iloc[i]['taxis'])
                plot_ccg(axes_tmp, ccg, baseline, thresh, taxis, nspk=None)
            plt.tight_layout()
            plt.savefig(os.path.join(figfolder, '{}-MGB_{}-A1_{}.jpg'.format(
                exp, pairs.iloc[i]['input_unit'], pairs.iloc[i]['target_unit'])))
            plt.close()
            
            
def plot_ccg_ccg_filtered(axes, ccg, baseline, thresh, taxis=None):
    if taxis is None:
        edges = np.arange(-50, 50.5, .5)
        taxis = (edges[1:] + edges[:-1]) / 2
    
    ccg_filtered = ccg - baseline
    plot_ccg(axes[0], ccg, baseline, thresh, taxis)
    axes[1].plot(taxis, ccg_filtered, 'k')
    axes[1].set_xlim([-50, 50])
    axes[1].plot([-50, 50], 4*ccg_filtered.std() * np.ones(2))
    axes[1].plot([-50, 50], 5*ccg_filtered.std() * np.ones(2))
    axes[1].plot([-50, 50], 6*ccg_filtered.std() * np.ones(2))


def plot_ccg(ax, ccg, baseline, thresh=None, edges=None, nspk=None, causal_method='peak', causal=True, xlim=[-50, 50]):
    if edges is None:
        edges = np.arange(-50, 50.5, .5)
        binsize = .5
    else:
        binsize = edges[1] - edges[0]
    taxis = (edges[1:] + edges[:-1]) / 2
    if nspk: # % of trial
        ccg = ccg / nspk * 100
    
    ax.bar(taxis, ccg, width=binsize, color='k')
    
    # plot causal spikes
    if causal:
        if causal_method == 'peak':
            causal_idx = ct.get_causal_spike_idx(edges, ccg, method=causal_method)
            causal_baseline = ct.get_causal_spk_baseline(ccg, causal_idx)
            ax.plot([-50, 50], [causal_baseline, causal_baseline], 'g', linewidth=.6)
        try:
            ax.bar(taxis[causal_idx], ccg[causal_idx]-causal_baseline, bottom=causal_baseline,
                  width=binsize, color='r')
        except (NameError, TypeError):
            pass
        
    # plot baseline and threshold
    try:
        if nspk:
            baseline = baseline / nspk * 100
            thresh = thresh / nspk * 100
        ax.plot(taxis, baseline, 'b--', linewidth=.6)
        ax.plot(taxis, thresh, 'b', linewidth=.6)
    except (ValueError, TypeError):
        pass
    
    ax.set_xlim(xlim)
    ax.set_xlabel('Time after MGB spike (ms)')
    if nspk:
        ax.set_ylabel('A1 firing rate (spk/s)')
    else:
        ax.set_ylabel('# of A1 spikes')
    return max(ccg)


# paired unit
def batch_plot_pairs_waveform_strf_ccg(datafolder:str=r'E:\Congcong\Documents\data\connection\data-pkl', 
                   figfolder:str=r'E:\Congcong\Documents\data\connection\figure\pairs_strf_ccg'):
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    for file in files:
        pairs = pd.read_json(file)
        exp = re.search('\d{6}_\d{6}', file).group(0)
        _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
        
        for i in pairs.index:
            if not pairs.loc[i].sig_spon:
                continue
            pair = pairs.loc[i]
            # plot waveform and strfs
            input_unit = input_units[pair.input_idx]
            target_unit = target_units[pair.target_idx]
            plot_pair_waveform_strf_ccg(pair, input_unit, target_unit)
            plt.savefig(os.path.join(figfolder, 
                                    '{}-MGB_{}-A1_{}.jpg'.format(exp, pairs.iloc[i]['input_unit'], pairs.iloc[i]['target_unit'])),
                        dpi=300, bbox_inches='tight')
            plt.close()


def plot_pair_waveform_strf_ccg(pair, input_unit, target_unit, tlim = [100, 0]):
    fig = plt.figure(figsize=[13*cm, 12*cm])

    x_start = [.15, .6]
    y_start = .65
    x_waveform = .1
    y_waveform = .15
    x_strf = .2
    y_strf = .25
    
    for i, unit in enumerate((input_unit, target_unit)):
        axes = [fig.add_axes([x_start[i] + .25, y_start + .15, x_waveform, y_waveform]), 
                fig.add_axes([x_start[i], y_start, x_strf, y_strf])]
        # plot waveform
        idx = np.where(unit.adjacent_chan == unit.chan)[0][0]
        waveform_mean = unit.waveforms_mean[idx, :]
        waveform_std = unit.waveforms_std[idx, :]
        plot_waveform(axes[0], waveform_mean, waveform_std)
        
        # plot strf
        strf = np.array(unit.strf)
        taxis = unit.strf_taxis
        idx_start = np.where(taxis == tlim[0])[0][0] + 1
        idx_end = np.where(taxis == tlim[1])[0][0] + 1
        taxis = taxis[idx_start: idx_end]
        strf = strf[:, :, idx_start:idx_end]
        im = plot_strf(axes[1], strf, taxis, faxis=unit.strf_faxis, bf=unit.bf, latency=unit.latency)
        axes[1].set_title(unit.strf_sig)
        # add colorbar for strf
        axins = inset_axes(
            axes[1],
            width="20%",  # width: 5% of parent_bbox width
            height="40%",  # height: 50%
            loc="lower right",
            bbox_to_anchor=(.5, 0, 1, 1),
            bbox_transform=axes[1].transAxes,
            borderpad=0,
        )
        cb = fig.colorbar(im, cax=axins)
        cb.ax.tick_params(axis='y', direction='in')
        axins.tick_params(axis='both', which='major', labelsize=6)
        
        if i == 1:
            axes[1].set_ylabel('')
        else:
            axes[1].set_xlabel('')
        
        ax = fig.add_axes([x_start[i], .42, .4, .18])
        acg, _, _ = ct.get_ccg(unit.spiketimes, unit.spiketimes)
        acg[100] = 0
        plot_ccg(ax, acg, None, causal=False)
        ax.set_xlim([-20, 20])
        
    # plot ccg
    ax = fig.add_axes([.1, .1, .4, .2])
    ccg = np.array(pair.ccg_spon)
    baseline = np.array(pair.baseline_spon)
    thresh = np.array(pair.thresh_spon)
    plot_ccg(ax, ccg, baseline)
    ax.set_title('MGB = {}; A1 = {}'.format(len(input_unit.spiketimes_spon), len(target_unit.spiketimes_spon)))
    ax = fig.add_axes([.6, .1, .4, .2])
    ccg = np.array(pair.ccg_dmr)
    baseline = np.array(pair.baseline_dmr)
    thresh = np.array(pair.thresh_dmr)
    plot_ccg(ax, ccg, baseline)
    ax.set_title('MGB = {}; A1 = {}'.format(len(input_unit.spiketimes_dmr), len(target_unit.spiketimes_dmr)))


  
def plot_waveform(ax, waveform_mean, waveform_std, color='k', color_shade='lightgrey', tpd=None):
    if tpd:
        if tpd < .45:
            color, color_shade = tpd_color[1],  tpd_color[2]
        else:
            color, color_shade = tpd_color[0],  tpd_color[3]
    x = range(waveform_mean.shape[0])
    ax.fill_between(x, waveform_mean + waveform_std, waveform_mean - waveform_std, color=color_shade)
    ax.plot(x, waveform_mean, color=color, linewidth=.8)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim([x[0], x[-1]])
    ax.spines[['left', 'bottom']].set_visible(False)


# ----------------------------------- plot cNE-A1 connection -------------------------------------
def batch_plot_ne_neuron_connection_ccg(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', 
                                        figfolder=r'E:\Congcong\Documents\data\connection\figure',
                                        stim='spon', df=20, file_id=None):
    if file_id is None:
        files = glob.glob(os.path.join(datafolder, f'*-pairs-ne-{df}-{stim}.json'))
    else:
        files = glob.glob(os.path.join(datafolder, f'*-pairs-ne-{file_id}.json'))

    for file in files:
        nepairs = pd.read_json(file)
        exp = re.search('\d{6}_\d{6}', file).group(0)
        cne_target = nepairs[['cne', 'target_idx']].drop_duplicates()
        _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
        nefile = re.sub('-pairs-ne-.*.json', f'-ne-{df}dft-spon.pkl', file)
        with open(nefile, 'rb') as f:
            ne = pkl.load(f)
        patterns = ne.patterns
        for cne, target_idx in cne_target.values:
            fig, ne_neuron_pairs = plot_ne_neuron_connection_ccg(
                nepairs, cne, target_idx, input_units, target_units, patterns, stim=stim)

            # save file
            target_unit = ne_neuron_pairs.iloc[0]['target_unit']
            fig.savefig(os.path.join(figfolder, f'ne_ccg_{stim}', f'{df}dft-{exp}-cne_{cne}-target_{target_unit}.jpg'), dpi=300)
            plt.close()


def plot_ne_neuron_connection_ccg(nepairs, cne, target_idx, input_units, target_units, patterns, stim='spon'):
    ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_idx == target_idx)].copy()
    n_pairs = len(ne_neuron_pairs)
    assert(n_pairs > 1)
    
    bottom_space = .8
    fig = plt.figure(figsize=[8.8*cm, 1.5*n_pairs*cm + bottom_space*cm])
    # probe
    x_start = .1
    x_fig = .08
    # y_start = (1.8*(n_pairs-2) + bottom_space + .5)/ (1.8 * n_pairs + bottom_space)
    # y_fig = 3 / (1.8 * n_pairs + bottom_space)
    y_start = .05
    y_fig = .4
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs, input_units)
    position_idx = np.array(position_idx)
    
    # icweight
    #y_start = 1/ (1.8 * n_pairs + bottom_space)
    #y_fig = 2 / (1.8 * n_pairs + bottom_space)
    y_start = .55
    y_fig = .4
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    weights = patterns[cne]
    weights = weights[position_order]
    member_thresh = 1 / np.sqrt(patterns.shape[1])
    plot_ICweight(ax, weights, member_thresh, direction='v', markersize=2)
    ax.set_xlim([-.4, .8])
    ax.invert_yaxis()
    
    # add axes for ccg plot
    x_start = .3
    y_start = bottom_space / (1.8 * n_pairs + bottom_space)
    x_fig = .2
    x_space = .05
    y_fig = 0.8 * (1.8 / (1.8 * n_pairs + bottom_space))
    y_space =  0.2 * (1.8 / (1.8 * n_pairs + bottom_space))
    axes = add_multiple_axes(fig, n_pairs, 2, x_start, y_start, x_fig, y_fig, x_space, y_space)
    position_idx_ne = [position_idx[unit_idx] + 1 for unit_idx in ne_neuron_pairs.input_idx]
    ne_neuron_pairs['position_idx'] = position_idx_ne
    ne_neuron_pairs = ne_neuron_pairs.sort_values(by='position_idx')
    plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs, stim=stim)
    
    # add axes for waveform plot
    x_start = .75
    y_start = (bottom_space + .4) / (1.8 * n_pairs + bottom_space)
    x_fig = .1
    x_space = 0
    y_fig = 0.4 * (1.8 / (1.8 * n_pairs + bottom_space))
    y_space =  0.6 * (1.8 / (1.8 * n_pairs + bottom_space))
    axes = add_multiple_axes(fig, n_pairs, 1, x_start, y_start, x_fig, y_fig, x_space, y_space)
    plot_all_waveforms(axes, ne_neuron_pairs, input_units, position_idx=position_idx)
    # plot A1 target waveform
    x_start = .87
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_all_waveforms(np.array([ax]), ne_neuron_pairs, target_units, 'target')
    
    return fig, ne_neuron_pairs

def plot_position_on_probe(ax, pairs, units, location='MGB'):
    """
    Plot neurons and ne members along the probe, where ne members are black dots.

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.
    units : TYPE
        DESCRIPTION.

    Returns
    -------
    position_idx : index of ne members based on their positions along the probe
    position_order : order of ne members if you want to retrieve then shallow to deep

    """
    position_idx = []
    for unit in units:
        ax.scatter(unit.position[0], unit.position[1], s=2, color='k')
        position_idx.append(unit.position_idx)
    if location == 'MGB':
        for i in range(len(pairs)):
            pair = pairs.iloc[i]
            unit = units[pair.input_idx]
            ax.scatter(unit.position[0], unit.position[1], s=6, color='r')
            ax.text(unit.position[0], unit.position[1], f'{unit.position_idx+1}', fontsize=6)
    elif location == 'A1':
        for i in pairs.target_idx.unique():
            unit = units[i]
            ax.scatter(unit.position[0], unit.position[1], s=6, color='r')
            ax.text(unit.position[0], unit.position[1], f'{i}:{unit.position_idx+1}', fontsize=6)
        ax.set_xlim([-150, 400])
        ax.yaxis.tick_right()
        ax.spines[['right']].set_visible(True)
        ax.spines[['left']].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.spines[['bottom']].set_visible(False)
    ax.invert_yaxis()
    position_order = np.argsort(position_idx)
    return position_idx, position_order


def get_position_on_probe(units):
    """

    Parameters
    ----------
    pairs : TYPE
        DESCRIPTION.

    Returns
    -------
    position_idx : index of ne members based on their positions along the probe
    position_order : order of ne members if you want to retrieve then shallow to deep

    """
    position_idx = []
    for unit in units:
        position_idx.append(unit.position_idx)
    position_order = np.argsort(position_idx)
    return position_idx, position_order


def plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs, stim='spon', force_causal=False):
    n_pairs = len(ne_neuron_pairs)
    for i in range(n_pairs):
        ax = axes[i]
        pair = ne_neuron_pairs.iloc[i]
        peak_fr = []
        for j, unit_type in enumerate(('neuron','nonne', 'ne')):
            ccg  = np.array(eval(f'pair.ccg_{unit_type}_{stim}'))
            nspk = np.array(eval(f'pair.nspk_{unit_type}_{stim}'))
            if force_causal:
                peak_fr.append(
                    plot_ccg(ax[j], ccg, None, nspk=nspk, causal=True, xlim=[-20, 20])
                    )
            else:
                peak_fr.append(
                    plot_ccg(ax[j], ccg, None, nspk=nspk, causal=pair[f'inclusion_{stim}'], xlim=[-10, 10])
                    )
            # x label and y label
            if i == n_pairs-1 and j == 0:
                ax[j].get_xaxis().set_label_coords(1.2, -.3)
            else:
                ax[j].set_xlabel('')
                ax[j].set_ylabel('')
            # ytick label
            if j == 1:
                ax[j].set_yticklabels([])
            if i < n_pairs - 1:
                ax[j].set_xticklabels([])

        m = max(peak_fr)
        for j, unit_type in enumerate(('nonne', 'ne')):
            ax[j].set_ylim([0, m * 1.1])
            if pair[f'inclusion_{stim}']:
                ax[j].text(-15, m*0.8, 
                       '{:.2f}'.format(eval(f'pair.efficacy_{unit_type}_{stim}')),
                       fontsize=6)


def plot_all_waveforms(axes, pairs, units, unit_type='input', position_idx=None):
    if axes.ndim > 1:
        axes = axes.flatten()
    idx = pairs[f'{unit_type}_idx'].unique()
    for i, unit_idx in enumerate(idx):
        unit = units[unit_idx]
        ax = axes[i]
        idx = np.where(unit.adjacent_chan == unit.chan)[0][0]
        waveform_mean = unit.waveforms_mean[idx, :]
        waveform_std = unit.waveforms_std[idx, :]
        if unit_type == 'target':
            tpd = unit.waveform_tpd
            plot_waveform(ax, waveform_mean, waveform_std, tpd=tpd)
        else:
            plot_waveform(ax, waveform_mean, waveform_std)
        if position_idx is not None:
            ax.set_title('neuron #{}'.format(position_idx[unit_idx]+1), fontsize=6)
        else:
            ax.set_title('neuron #{}'.format(unit_idx+1), fontsize=6)


def add_multiple_axes(fig, nrows, ncols, x_start, y_start, x_fig, y_fig, x_space, y_space):
    axes = []
    for i in range(nrows):
        ax= []
        for j in range(ncols):
            ax.append(fig.add_axes([x_start + j * (x_fig + x_space), 
                                    y_start + i * (y_fig + y_space), 
                                    x_fig, y_fig]))
        axes.append(ax)
    axes = axes[::-1]
    return np.array(axes)


def batch_plot_strf(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', 
                   figfolder=r'E:\Congcong\Documents\data\connection\figure\strf'):
    spkfiles = glob.glob(os.path.join(datafolder, '*fs20000.pkl'))
    for i, file in enumerate(spkfiles):
        with open(file, 'rb') as f:
            session = pickle.load(f)
        session.plot_strf(figfolder=figfolder)
    

def batch_plot_ne_coincidence_spk_ccg(
        datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', 
        summaryfolder=r'E:\Congcong\Documents\data\connection\data-summary', 
        figfolder=r'E:\Congcong\Documents\data\connection\figure\ne_coincidence_ccg',
        stim='spon', coincidence='act-level'):
    pairs = pd.read_json(os.path.join(summaryfolder, f'ne-pairs-{coincidence}-{stim}.json'))
    pairs_sig = pairs[(pairs.efficacy_diff_p < .05) & 
                      (pairs[f'efficacy_ne_{stim}']  > pairs[f'efficacy_nonne_{stim}'])] 
    exp = None
    fields = fields=[f'ccg_neuron_{stim}', 'ccg_hiact', f'ccg_ne_{stim}']
    for i in range(len(pairs_sig)):
        pair = pairs_sig.iloc[i]
        fig = plot_ne_coincidence_spk_ccg(pair, fields)
        fig.savefig(os.path.join(figfolder, '{}-{}-cne{}-MGB{}-A1{}.jpg'.format(
            stim, pair.exp, pair.cne, pair.input_unit, pair.target_unit)), dpi=300)
        plt.close()

def plot_ne_coincidence_spk_ccg(pair, fields):
    fig = plt.figure(figsize=[8.5*cm, 4*cm])
    x_start = .15
    y_start = .2
    x_fig = .25
    x_space = .05
    y_fig = .6
    y_space = 0
    axes = add_multiple_axes(fig, 1, 3, x_start, y_start, x_fig, y_fig, x_space, y_space)
    axes = axes[0]
    peak_fr = []
    for i, field in enumerate(fields):
        ccg = np.array(pair[field])
        nspk_field = re.sub('ccg', 'nspk', field)
        nspk = np.array(pair[nspk_field])
        peak_fr.append(
            plot_ccg(axes[i], ccg, None, nspk=nspk, xlim=[-25, 25])
            )
        axes[i].set_title('{:.2f}'.format(pair[re.sub('ccg', 'efficacy', field)]))
    peak_fr = max(peak_fr) * 1.1
    for i in range(3):
        axes[i].set_ylim([0, peak_fr])
        if i > 0:
            axes[i].set_ylabel('')
            axes[i].set_yticklabels([])
        if i != 1:
            axes[i].set_xlabel('')
    return fig


def batch_plot_common_target_ccg(datafolder=r'E:\Congcong\Documents\data\connection\data-summary',
                                 figfolder=r'E:\Congcong\Documents\data\connection\figure\ccg_common_target'):
    # plot CCGs of MGB neurons that share A1 target or not 
    data = pd.read_json(os.path.join(datafolder, 'pairs_common_target_corr.json'))
    for i in range(len(data)):
        pair = data.iloc[i]
        fig = plt.figure()
        ax = fig.add_axes([.1, .1, .8, .8])
        plot_ccg(ax, np.array(pair.ccg), None, None)
        ax.set_xlim([-40, 40])
        if pair.target > 0:
            fig.savefig(os.path.join(figfolder, f'{pair.exp}_MGB_{pair.input1}_{pair.input2}_A1_{pair.target}.jpg'))
        else:
            fig.savefig(os.path.join(figfolder, f'None_share_{pair.exp}_MGB_{pair.input1}_{pair.input2}.jpg'))
        plt.close()

def batch_plot_ne_membership_ccg(datafolder=r'E:\Congcong\Documents\data\connection\data-summary',
                                 figfolder=r'E:\Congcong\Documents\data\connection\figure\ccg_ne_members'):
    # plot CCGs of MGB neurons that share A1 target or not 
    data = pd.read_json(os.path.join(datafolder, 'member_nonmember_pair_xcorr.json'))
    data = data[(data.stim == 'spon') & (data.region == 'MGB')]
    for i in range(len(data)):
        pair = data.iloc[i]
        fig = plt.figure()
        ax = fig.add_axes([.1, .1, .8, .8])
        plot_ccg(ax, np.array(pair.xcorr)[101: -100], None, None)
        ax.set_xlim([-25, 25])
        if pair.member:
            fig.savefig(os.path.join(figfolder, f'{pair.exp}_MGB_{pair.idx1}_{pair.idx2}.jpg'))
        else:
            fig.savefig(os.path.join(figfolder, f'Nonmember_{pair.exp}_MGB_{pair.idx1}_{pair.idx2}.jpg'))
        plt.close()
        

def batch_plot_strf_ccg_target(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
                        figfolder=r'E:\Congcong\Documents\data\connection\figure\strf_ccg_target'):
    ccg_data = pd.read_json(os.path.join(r'E:\Congcong\Documents\data\connection\data-summary', 
                                        'pairs_common_target_corr.json'))
    files = glob.glob(os.path.join(datafolder, '*-fs20000.pkl'))
    files = [file for file in files if int(re.search('(\d{4})um', file).group(1)) > 2000][6:]
    for file in files:
        with open(file, 'rb') as f:
            session = pickle.load(f)
        exp = int(''.join(session.exp.split('_')))
        n_unit = len(session.units)
        fig, axes = plt.subplots(n_unit, n_unit, figsize=[n_unit, n_unit])
        plt.tight_layout()
        for i, unit in enumerate(session.units):
            for j in range(i, n_unit):
                if i == j:
                    strf = unit.strf
                    plot_strf(axes[i][j], strf, taxis=unit.strf_taxis, faxis=unit.strf_faxis, tlim=[50, 0])
                else:
                    ccg = ccg_data[(ccg_data.exp == exp) 
                                   & (ccg_data.input1 == i) 
                                   & (ccg_data.input2 == j)]
                    if len(ccg) > 0:
                        plot_ccg(axes[i][j], ccg.ccg.values[0], None, None, causal=False)
                        if ccg.target.values[0] >= 0:
                            y = axes[i][j].get_ylim()
                            axes[i][j].text(0, np.mean(y), '*', color='r', fontsize=20)
        fig.savefig(os.path.join(figfolder, f'{exp}.jpg'), dpi=300)
        plt.close()


def batch_plot_strf_ccg_membership(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
                        figfolder=r'E:\Congcong\Documents\data\connection\figure\strf_ccg_membership'):
    ccg_data = pd.read_json(os.path.join(r'E:\Congcong\Documents\data\connection\data-summary', 
                                        'member_nonmember_pair_xcorr.json'))
    ccg_data = ccg_data[ccg_data.stim=='spon']
    files = glob.glob(os.path.join(datafolder, '*-fs20000.pkl'))
    files = [file for file in files if int(re.search('(\d{4})um', file).group(1)) > 2000]
    for file in files:
        with open(file, 'rb') as f:
            session = pickle.load(f)
        exp = int(''.join(session.exp.split('_')))
        n_unit = len(session.units)
        fig, axes = plt.subplots(n_unit, n_unit, figsize=[n_unit, n_unit])
        plt.tight_layout()
        for i, unit in enumerate(session.units):
            for j in range(i, n_unit):
                if i == j:
                    strf = unit.strf
                    plot_strf(axes[i][j], strf, taxis=unit.strf_taxis, faxis=unit.strf_faxis, tlim=[50, 0])
                else:
                    ccg = ccg_data[(ccg_data.exp == exp) 
                                   & (ccg_data.idx1 == i) 
                                   & (ccg_data.idx2 == j)]
                    if len(ccg) > 0:
                        plot_ccg(axes[i][j], ccg.xcorr.values[0][101:-100], None, None, causal=False)
                        if ccg.member.values[0]:
                            y = axes[i][j].get_ylim()
                            axes[i][j].text(0, np.mean(y), '*', color='r', fontsize=20)
        fig.savefig(os.path.join(figfolder, f'{exp}.jpg'), dpi=300)
        plt.close()
    
# ------------------------------------ figure plots ----------------------------------------------
def figure1(datafolder='E:\Congcong\Documents\data\connection\data-pkl', 
            figfolder = r'E:\Congcong\Documents\data\connection\paper',
            example='200820_230604-MGB_61-A1_260'):
    
    exp = re.search('\d{6}_\d{6}', example).group(0)
    input_unit = int(re.search('(?<=MGB_)\d{1,3}', example).group(0))
    target_unit = int(re.search('(?<=A1_)\d{1,3}', example).group(0))
    
    file = glob.glob(os.path.join(datafolder, f'{exp}*pairs.json'))[0]
    pairs = pd.read_json(file)
    pair = pairs[(pairs.input_unit == input_unit) & (pairs.target_unit == target_unit)]
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
    input_unit = input_units[pair.input_idx.values[0]]
    target_unit = target_units[pair.target_idx.values[0]]

    
    fig = plt.figure(figsize=[11.6*cm, 8*cm])
    # plot distribution of efficacy
    print("C")
    x_start = .77
    y_start = .78
    x_fig = .2
    y_fig = .2
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    batch_hist_efficacy(ax, stim='spon', color='k')
    batch_hist_efficacy(ax, stim='dmr', color='grey')
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 15])
    ax.set_yticks(range(0, 16, 5))
    
    # plot fr distribution
    print('D')
    y_start = .44
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    batch_plot_fr_pairs(ax)
    
    # plot best frequency
    print("E")
    y_start = .1
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    batch_scatter_bf(ax, stim='spon', marker='^')
    batch_scatter_bf(ax, stim='dmr', facecolor='grey', edgecolor='w', marker='o', size=15)
    
    # plot ccg
    x_start = .08
    y_start = [.35, .1]
    x_fig = .22
    y_fig = .16
    # spon
    ax = fig.add_axes([x_start, y_start[0], x_fig, y_fig])
    ccg = np.array(pair.ccg_spon.values[0])
    baseline = np.array(pair.baseline_spon.values[0])
    thresh = np.array(pair.thresh_spon.values[0])
    nspk = np.array(pair.nspk_spon.values[0])
    plot_ccg(ax, ccg, baseline, thresh, nspk=nspk)
    efficacy = pair.efficacy_spon.values[0]
    ax.text(10, 4, f'efficacy = {efficacy:.2f}', fontsize=6, color='r')
    ax.set_ylim([0, 8])
    ax.set_yticks(range(0, 9, 4))
    ax.set_xlim([-10, 10])
    ax.set_xlabel('')

    # stim
    ax = fig.add_axes([x_start, y_start[1], x_fig, y_fig])
    ccg = np.array(pair.ccg_dmr.values[0])
    baseline = np.array(pair.baseline_dmr.values[0])
    thresh = np.array(pair.thresh_dmr.values[0])
    nspk = np.array(pair.nspk_dmr.values[0])
    plot_ccg(ax, ccg, baseline, thresh, nspk=nspk)
    efficacy = pair.efficacy_dmr.values[0]
    ax.text(10, 1, f'efficacy = {efficacy:.2f}', fontsize=6, color='r')
    ax.set_ylabel('')
    ax.set_xlim([-10, 10])
    ax.set_ylim([0, 2])
    
    
    # plot example STRFs
    print('B-ii')
    x_start = .45
    y_start =  [.4, .1]
    x_waveform = .04
    y_waveform = .06
    x_strf = .12
    y_strf = .15
    
    for i, unit in enumerate((input_unit, target_unit)):
        axes = [fig.add_axes([x_start + .12, y_start[i] + .1, x_waveform, y_waveform]), 
                fig.add_axes([x_start, y_start[i], x_strf, y_strf])]
        # plot waveform
        idx = np.where(unit.adjacent_chan == unit.chan)[0][0]
        waveform_mean = unit.waveforms_mean[idx, :]
        waveform_std = unit.waveforms_std[idx, :]
        plot_waveform(axes[0], waveform_mean, waveform_std)
        
        # plot strf
        tlim = [50, 0]
        flim = [2, 32]
        strf = np.array(unit.strf)
        
        vmax=np.max(abs(strf))
        im = plot_strf(axes[1], strf, taxis=unit.strf_taxis, faxis=unit.strf_faxis, flabels_arr = np.array([2, 8, 32]),
                       tlim=tlim, flim=flim, vmax=vmax, bf=unit.bf, latency=unit.latency)
        print(unit.bf/1000)
        print(unit.latency)
        # add colorbar for strf
        axins = inset_axes(
            axes[1],
            width="10%",  # width: 5% of parent_bbox width
            height="60%",  # height: 50%
            loc="lower right",
            bbox_to_anchor=(.2, 0, 1, 1),
            bbox_transform=axes[1].transAxes,
            borderpad=0,
        )
        cb = fig.colorbar(im, cax=axins)
        cb.ax.tick_params(axis='y', direction='in')
        cb.ax.set_yticks([-vmax, 0, vmax])
        cb.ax.set_yticklabels(['', '', ''])
        axins.tick_params(axis='both', which='major', labelsize=6, pad=2)
        
        if i == 0:
            axes[1].set_ylabel('')
            axes[1].set_xlabel('')
            cb.ax.set_yticklabels(['-Max', '0', 'Max'])
    
    example_file = os.path.join(
        datafolder,
        '200820_230604-site4-5655um-25db-dmr-31min-H31x64-fs20000-pairs-ne-10-spon.json',
    )
    nepairs = pd.read_json(example_file)
    # load ne info
    exp = re.search('\d{6}_\d{6}', example_file).group(0)
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
   
    ne_neuron_pairs = nepairs.iloc[0:1].copy()
    ne_neuron_pairs.input_idx  = 5
    ne_neuron_pairs.input_unit  = 47
    ne_neuron_pairs.target_idx  = 23
    ne_neuron_pairs.target_unit  = 402
    # 1. probe
    # 1.1 MGB
    x_start = .06
    x_fig = .02
    y_start = .7
    y_fig = .24
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs, input_units)
    ax.set_ylim([5600, 4400])
    ax.set_yticks(range(5600, 4400-1, -200))
    ax.set_ylabel(r'Depth ($\mu$m)', labelpad=0)
    
    x_start = .18
    y_fig = .16
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    _ = plot_position_on_probe(ax, ne_neuron_pairs, target_units, location='A1')
    ax.set_ylim([1000, 200])
    ax.set_yticks(range(1000, 200-1, -200))
    ax.set_title('')

    fig.savefig(os.path.join(figfolder, 'fig1.pdf'), dpi=300)

    
def batch_hist_efficacy(ax, datafolder='E:\Congcong\Documents\data\connection\data-summary',
                           stim='spon', color='k'):
    file = glob.glob(os.path.join(datafolder, 'pairs.json'))
    pairs = pd.read_json(file[0])
    pairs = pairs[pairs[f'sig_{stim}']]
    efficacy = pairs[f'efficacy_{stim}'].values
    ax.hist(efficacy, bins=np.arange(0, 32, 1), color=color)
    ax.scatter(np.mean(efficacy), 12, s=8, marker='v', facecolors='none', edgecolors=color)
    ax.scatter(np.median(efficacy), 13, s=8, marker='v', c=color)
    ax.set_xlim([0, 25])
    ax.set_ylim([0, 15])
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks(range(0, 16, 5))
    ax.set_xlabel('efficacy')
    ax.set_ylabel('# of pairs', labelpad=0)
    print('n(ccg_{}) = {}'.format(stim, len(efficacy)))
    print(stim, np.median(efficacy))
    print(efficacy.mean(), efficacy.std())
    if stim == 'spon':
        _, p = stats.mannwhitneyu(pairs['efficacy_spon'], pairs[pairs.sig_dmr]['efficacy_dmr'])
        print(f'ranksum test: p = {p}')
    elif stim == 'dmr':
        _, p = stats.wilcoxon(pairs['efficacy_spon'], pairs['efficacy_dmr'])
        print(f'signrank test: p = {p}')
        print('spon:', pairs['efficacy_spon'].mean(), pairs['efficacy_spon'].std())
        print('spon:', np.median(pairs['efficacy_spon']))
        print('dmr:', pairs['efficacy_dmr'].mean(), pairs['efficacy_dmr'].std())


def batch_scatter_bf(ax, datafolder='E:\Congcong\Documents\data\connection\data-pkl',
                           stim='spon', facecolor=None, edgecolor='k', marker='x', size=8):
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    bf_input = []
    bf_target = []
    for file in files:
        pairs = pd.read_json(file)
        pairs = pairs[pairs[f'sig_{stim}']]
        if stim == 'spon':
            pairs = pairs[~pairs['sig_dmr']]
        bf_input.extend(pairs.input_bf)
        bf_target.extend(pairs.target_bf)
    bf_input = np.log2(np.array(bf_input) / 500)
    bf_target = np.log2(np.array(bf_target) / 500)
    idx = (bf_input < 6) & (bf_target < 6)
    bf_input, bf_target = bf_input[idx], bf_target[idx]
    ax.plot([0, 6], [0.5, 0.5], 'k--')
    ax.plot([0, 6], [-0.5, -0.5],'k--')
    ax.plot([0, 6], [0, 0], color='k')

    ax.scatter(bf_input, bf_target-bf_input, edgecolor=edgecolor, s=size, marker=marker, facecolor=facecolor)
    ax.set_xticks(range(0, 7, 2))
    ax.set_xticklabels([.5, 2, 8, 32])

    ax.set_xlim([0, 6])
    ax.set_ylim([-2, 2])
    ax.set_yticks(range(-2, 3))
    ax.set_xlabel('MGB neuron BF (kHz)')
    ax.set_ylabel('\u0394BF (oct)')
    print('n(ccg_{}) = {}'.format(stim, sum(idx)))
    diff = bf_target-bf_input
    print(np.abs(diff).mean(), np.abs(diff).std())
    
    
def batch_plot_fr(ax, datafolder='E:\Congcong\Documents\data\connection\data-summary'):
    with open(os.path.join(datafolder, 'fr_all.json'), 'r') as f:
        data = json.load(f)
    
    color = ['k', 'grey']
    for i, region in enumerate(('MGB', 'A1')):
        print(region)
        diff = np.array(data['dmr_'+region]) - np.array(data['spon_'+region])
        ax.hist([diff], bins=np.arange(-10, 11, 1), 
                histtype='step', density=True, linewidth=.8, color=color[i])
        ax.scatter(np.mean(diff), .36, s=8, marker='v', facecolors='none', edgecolors=color[i])
        ax.scatter(np.median(diff), .36, s=8, marker='v', facecolors=color[i], edgecolors=color[i])
        _, p = stats.wilcoxon(diff)
        print('p =', p)
        print('median = ', np.median(diff))
        print('mean = ', np.mean(diff))
        print('std = ', np.std(diff))
   
    ax.set_xlim([-10, 10])
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_ylim([0, .4])
    ax.set_ylabel('Probability density')
    ax.set_xlabel('\u0394Firing rate')

def batch_plot_fr_pairs(ax, datafolder='E:\Congcong\Documents\data\connection\data-summary'):
    data = pd.read_json(os.path.join(datafolder, 'pairs.json'))
    fr_target = data[['exp', 'target_unit', 'target_fr_dmr', 'target_fr_spon']].drop_duplicates()
    fr_input = data[['exp', 'input_unit', 'input_fr_dmr', 'input_fr_spon']].drop_duplicates()
    sns.scatterplot(data=fr_input, x='input_fr_spon', y='input_fr_dmr', 
                    alpha=.8, ax=ax, s=10, color=MGB_color[0])
    sns.scatterplot(data=fr_target, x='target_fr_spon', y='target_fr_dmr', 
                    alpha=.8, ax=ax, s=10, color=A1_color[0])
    _, p = stats.wilcoxon(fr_input['input_fr_spon'], fr_input['input_fr_dmr'])
    fr_diff = fr_input['input_fr_dmr'] - fr_input['input_fr_spon']
    print('MGB: p=', p, fr_diff.mean(), fr_diff.std())
    print('n=', len(fr_input))
    _, p = stats.wilcoxon(fr_target['target_fr_spon'], fr_target['target_fr_dmr'])
    fr_diff = fr_target['target_fr_dmr'] - fr_target['target_fr_spon']
    print('A1: p=', p, fr_diff.mean(), fr_diff.std())
    print('n=', len(fr_target))
    plt.plot([.1, 100], [.1, 100])
    ax.set_xlim([.1, 100])
    ax.set_ylim([.1, 100])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Spon FR (Hz)')
    ax.set_ylabel('Stim FR (Hz)')
    ax.tick_params(axis="both", which="major", labelsize=6)


def figure2a(datafolder: str = r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder: str = r'E:\Congcong\Documents\data\connection\paper'):
    """
    Figure2: groups of neurons with coordinated activities exist in A1 and MGB
    Plot construction procedures for cNEs and  correlated firing around cNE events
    cNE members show significantly higher cross correlation

    Input:
        datafolder: path to *ne-20dft-dmr.pkl files
        figfolder: path to save figure
    Return:
        None
    """
    mpl.rcParams['axes.labelpad'] = 1

    # use example recording to plot construction procedure
    nefile = os.path.join(datafolder, '200820_230604-site4-5655um-25db-dmr-31min-H31x64-fs20000-ne-10dft-spon.pkl')
    with open(nefile, 'rb') as f:
        ne = pickle.load(f)
    session_file = re.sub('-ne-10dft-spon', '', nefile)
    with open(session_file, 'rb') as f:
        session = pickle.load(f)

    figsize = [figure_size[1][0], 13 * cm]
    fig = plt.figure(figsize=figsize)

    # binned spikes trains
    xstart = 0.07
    ystart = 0.76
    figy = 0.16
    figx = 0.22
    window=20
    n_neuron = len(session.units)
    ax = fig.add_axes([xstart, ystart, figx, figy])
    spktrain = ne.spktrain
    nspk = spktrain.sum(axis=0)
    nspk = nspk[:len(nspk) //window * window]
    nspk = nspk.reshape([window, len(nspk)//window]).sum(axis=0)
    idx = np.argmax(nspk)
    spktrain = spktrain[:, idx*window:idx*window + window]
    plt.imshow(spktrain, cmap="gray_r", aspect="auto")
    ax.invert_yaxis()
    ax.set_yticks([5, 10, 15])
    ax.set_ylabel("Neuron #")
    ax.spines[['top']].set_visible(True)
    ax.spines[['right']].set_visible(True)
    for i in range(window-1):
        plt.plot([i+.5, i+.5], [0, n_neuron], 'k', linewidth=.4)
    ax.set_xlim([-.5, window-.5])
    ax.set_ylim([.5, n_neuron-.5])
    ax.set_xticks([])
    
    figy = 0.2
    xstart = 0.4
    xspace = 0.15
    figx = 0.22

    # plot correlation matrix
    ax = fig.add_axes([xstart, ystart, figx, figy])
    corr_mat = np.corrcoef(ne.spktrain)
    im = plot_corrmat(ax, corr_mat)
    im.set_clim([-.12, .12])
    axins = inset_axes(
        ax,
        width="10%",  # width: 5% of parent_bbox width
        height="80%",  # height: 50%
        loc="center left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(im, cax=axins)
    cb.ax.tick_params(axis='y', direction='in')
    axins.set_title(' corr.', fontsize=6, pad=5)
    cb.ax.set_yticks([-0.12, -.06, 0, .06, 0.12,])
    axins.tick_params(axis='both', which='major', labelsize=6)

    # plot eigen values
    ax = fig.add_axes([xstart + figx + xspace, ystart, figx, figy])
    corr_mat = np.corrcoef(ne.spktrain)
    thresh = netools.get_pc_thresh(ne.spktrain)
    plot_eigen_values(ax, corr_mat, thresh)
    ax.set_ylim([.8, 1.4])
    ax.set_yticks([.8, 1, 1.2, 1.4])
    
    # plot ICweights - color coded
    ystart = .48
    xstart = .07
    figx = .18
    ax = fig.add_axes([xstart, ystart, figx, figy])
    patterns = ne.patterns

    patterns[0], patterns[1] = np.array(patterns[1]), np.array(patterns[0])

    members = ne.ne_members
    members[0], members[1] = members[1], members[0]
    im = plot_ICweigh_imshow(ax, ne.patterns, ne.ne_members)
    im.set_clim([-.8, .8])
    axins = inset_axes(
        ax,
        width="10%",  # width: 5% of parent_bbox width
        height="80%",  # height: 50%
        loc="center left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = fig.colorbar(im, cax=axins)
    cb.ax.tick_params(axis='y', direction='in')
    axins.set_title('IC weight', fontsize=6, pad=5)
    cb.ax.set_yticks(np.arange(-.8, .81, .4))
    cb.ax.set_yticklabels([-.8, -.4, 0, .4, .8])
    axins.tick_params(axis='both', which='major', labelsize=6)

    # stem plots for ICweights
    xstart = xstart + figx + xspace
    xspace = 0.002
    figx = 0.07
    n_ne, n_neuron = ne.patterns.shape
    thresh = 1 / np.sqrt(n_neuron)
    c = 0
    for i in range(4):
        ax = fig.add_axes([xstart + i * figx + i * xspace, ystart, figx, figy])
        plot_ICweight(ax, ne.patterns[i], thresh, direction='v', ylim=(-0.3, 0.8), markersize=2)
        if i > 0:
            ax.set_axis_off()

    # second row: activities
    xstart = .77
    ystart = .6
    figx = 0.22
    figy = 0.03
    
    # reorder units
    cne = 2
    edges = ne.edges[0] / 1000
    centers = (edges[:-1] + edges[1:]) / 2
    activity_idx = 5  # 99.5% as threshold
    activity = ne.ne_activity[cne]

    t_start = 597.5
    t_end = t_start + .8

    # plot activity
    ax = fig.add_axes([xstart, ystart, figx, figy])
    activity_thresh = ne.activity_thresh[cne][activity_idx]
    ylim = [-10, 200]
    plot_activity(ax, centers, activity, activity_thresh, [t_start, t_end], ylim)
    ax.set_ylabel('activity (a.u.)', fontsize=6)
    ax.set_title('cNE #3', fontweight='bold')

    # plot raster
    ystart = 0.4
    figy = 0.18
    ax = fig.add_axes([xstart, ystart, figx, figy])
    members = ne.ne_members[cne]
    for member in members:
        p = mpl.patches.Rectangle((t_start, member + 0.6),
                                  t_end - t_start, 0.8, color='gainsboro')
        ax.add_patch(p)
        c += 1

    plot_raster(ax, session.units, linewidth=.6)
    ax.eventplot(ne.ne_units[cne].spiketimes/1000, lineoffsets=n_neuron + 1, linelengths=0.8, colors='r', linewidth=.6)
    plot_raster(ax, ne.member_ne_spikes[cne], offset='unit', color='r', linewidth=.6)
    ax.set_xlim([t_start, t_end])
    ax.spines[['bottom', 'left']].set_visible(False)
    ax.set_xticks([])

    # scale bar
    ax.plot([t_start, t_start + .2], [0.1, 0.1], color='k', linewidth=1)
    ax.set_yticks([n_neuron + 1])
    ax.tick_params(axis='y', length=0)
    ax.set_yticklabels(['cNE'], fontsize=6, color='r')
    ax.set_ylim([0, n_neuron + 1.5])

    fig.savefig(os.path.join(figfolder, 'fig2a.pdf'), dpi=1000)
    
    
def figure2(figfolder=r'E:\Congcong\Documents\data\connection\paper',
            datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            example_file1=r'200820_230604-site4-5655um-25db-dmr-31min-H31x64-fs20000.pkl',
            example_idx1=[0, 15, 10],
            example_file2=r'201005_213847-site5-5105um-20db-dmr-32min-H31x64-fs20000.pkl',
            example_idx2=[5, 6, 4]):
    
    fig = plt.figure(figsize=[11.6*cm, 12*cm])
    # summary plots
    # plot corr of pairs of neurons sharing common target
    x_start = .75
    x_fig = .23
    y_fig = .2
    # plot cNE member corr
    y_start = .4
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_corr_ne_members(ax=ax, df=10)
    # plot probability of cNE members sharing target
    y_start = .1
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_prob_share_target(ax=ax, df=10)
    
    # PART2: correlation of cNE members and nonmembers
    example_file = os.path.join(datafolder, example_file2)
    example_idx = example_idx2
    with open(example_file, 'rb') as f:
        session = pickle.load(f)
    ne_file = re.sub("fs20000", "fs20000-ne-10dft-spon", example_file)
    with open(ne_file, 'rb') as f:
        ne = pickle.load(f)
    cne = 3
    patterns = ne.patterns
    # plot ICweights
    x_start = .08
    y_start = .4
    y_fig = .2
    x_fig = .1
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = get_position_on_probe(session.units)
    weights = patterns[cne]
    weights = weights[position_order]
    member_thresh = 1 / np.sqrt(patterns.shape[1])
    plot_ICweight(ax, weights, member_thresh, direction='v', markersize=2)
    ax.set_xlim([-.4, .8])
    ax.set_xticks([0, .4, .8])
    # plot ccgs
    input1 = session.spktrain_spon[0][example_idx[0]]
    input1 = (input1 - input1.mean()) / input1.std()
    input1 = input1[50:-50]
    # example1
    x_start = .4
    y_start = .55
    x_fig = .1
    y_fig = .08
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])    
    input2 = session.spktrain_spon[0][example_idx[1]]
    input2 = (input2 - input2.mean()) / input2.std()
    corr = np.correlate(input1, input2) / len(input2)
    taxis = np.arange(-25, 25.1, .5)
    ax.bar(taxis, corr, color='k')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-.0025, .03])
    ax.set_yticks([0, .01, .02, .03])
    ax.set_xticklabels([])
    #example2
    y_start = .42
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])    
    input2 = session.spktrain_spon[0][example_idx[2]]
    input2 = (input2 - input2.mean()) / input2.std()
    corr = np.correlate(input1, input2) / len(input2)
    ax.bar(taxis, corr, color='k')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-.0025, .03])
    ax.set_yticks([0, .01, .02, .03])
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Correlation')
    
    #fig.savefig(os.path.join(figfolder, 'fig2.jpg'), dpi=300)
    fig.savefig(os.path.join(figfolder, 'fig2.pdf'), dpi=300)


def plot_corr_common_target(datafolder=r'E:\Congcong\Documents\data\connection\data-summary', 
                            group='exp', ax=None, savefolder=None):
    file = os.path.join(datafolder,'pairs_common_target_corr.json')
    pairs = pd.read_json(file)
    if ax is None:
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_axes([.2, .2, .7, .7])
    pairs.drop_duplicates(subset=['exp', 'input1', 'input2'], inplace=True)
    pairs['share_target'] = pairs.target > -1
    colors=['black', 'grey']
    if group == 'exp':
        corr = pairs.groupby(['exp', 'share_target'])['corr'].mean()
        corr = pd.DataFrame(corr).reset_index()
        corr['n'] = pairs.groupby(['exp', 'share_target'])['corr'].size().values
        corr['std'] = pairs.groupby(['exp', 'share_target'])['corr'].std().values
        for c in range(2):
            corr_tmp = corr[corr.share_target == 1 - c]
            ax.scatter(c * np.ones(len(corr)//2), corr_tmp['corr'], s=corr_tmp['n']/(20 ** c)*5,
                       facecolor=colors[c], edgecolor='w', linewidth=.5)
            ax.errorbar(c * np.ones(len(corr_tmp)), y=corr_tmp['corr'], xerr=corr_tmp['std'] * 10, 
                        fmt='None', color=colors[c], capsize=1, linewidth=1, zorder=1)
            ax.scatter(x=[.7, 1, 1.3], y=np.array([.08]*3) + .015 * c, s=np.array([5, 10, 20])*5,
                       facecolor=colors[c], edgecolor='w', linewidth=.5)
        ax.errorbar(x=1, y=.06, xerr=.01 * 10, capsize=1, linewidth=1, zorder=1, color='k')
        ax.set_ylim([0, .1])
        ax.set_xlim([-.5, 1.5])
        ax.set_ylabel('Mean correlation')
        ax.set_yticks(np.arange(0, .11, .02))
        print('n(recording) = ', len(corr) / 2)
        model = ols('corr ~ C(share_target) + C(share_target):C(exp)', data=pairs).fit() 
        result = sm.stats.anova_lm(model, type=1) 
        print(result)
        plot_significance_star(ax, result.loc['C(share_target)', 'PR(>F)'], [0, 1], .09, .091)
        

    ax.set_xlabel('')
    if savefolder is not None:
        fig.savefig(os.path.join(savefolder, 'pair_corr_share_target.jpg'), dpi=300)
        plt.close()


def plot_corr_ne_members(datafolder=r'E:\Congcong\Documents\data\connection\data-summary', 
                         ax=None, savefolder=None, df=20):
    if ax is None:
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_axes([.2, .2, .7, .7])
    
    try:
        pairs = pd.read_json(os.path.join(datafolder, f'member_nonmember_pair_xcorr_{df}dft.json'))
    except FileNotFoundError:
        pairs = pd.read_json(os.path.join(datafolder,'member_nonmember_pair_xcorr.json'))
    pairs = pairs[pairs.stim == 'spon']
    corr = pairs.groupby(['exp', 'member'])['corr'].mean()
    corr = pd.DataFrame(corr).reset_index()
    corr['n'] = pairs.groupby(['exp', 'member'])['corr'].size().values
    corr['std'] = pairs.groupby(['exp', 'member'])['corr'].std().values
    colors=['black', 'grey']
    for c in range(2):
        corr_tmp = corr[corr.member == 1 - c]
        ax.scatter(c * np.ones(len(corr)//2), corr_tmp['corr'], s=corr_tmp['n']/(2**c),
                   facecolor=colors[c], edgecolor='w', linewidth=.5)
        ax.errorbar(c * np.ones(len(corr_tmp)), y=corr_tmp['corr'], xerr=corr_tmp['std'] * 10, 
                    fmt='None', color='k', capsize=1, linewidth=1, zorder=1)
        ax.scatter(x=[.7, 1, 1.3], y=np.array([.08]*3) + .015 * c, s=np.array([10, 50, 100]),
                   facecolor=colors[c], edgecolor='w', linewidth=.5)
    ax.errorbar(x=1, y=.06, xerr=.01 * 10, capsize=1, linewidth=1, zorder=1, color='k')
    ax.set_ylim([0, .1])
    ax.set_xlim([-.5, 1.5])
    ax.set_ylabel('Mean correlation')
    ax.set_yticks(np.arange(0, .11, .02))
    print('n(recording) = ', len(corr) / 2)
    model = ols('corr ~ C(member) + C(member):C(exp)', data=pairs).fit() 
    result = sm.stats.anova_lm(model, type=1) 
    print(result)
    plot_significance_star(ax, result.loc['C(member)', 'PR(>F)'], [0, 1], .09, .091)
        
    if savefolder is not None:
        fig.savefig(os.path.join(savefolder, 'pair_corr_ne_member.jpg'), dpi=300)
        plt.close()
        

def plot_prob_share_target(datafolder=r'E:\Congcong\Documents\data\connection\data-summary', 
                           ax=None, savefolder=None, df=20):
    if ax is None:
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_axes([.2, .2, .7, .7])
        
    pairs = pd.read_json(os.path.join(datafolder, f'pairs_common_target_ne_{df}dft.json'))

    pairs = pairs.groupby(['exp', 'within_ne']).filter(lambda x: x['within_ne'].count() > 3)
    prob_share =  pairs.groupby(['exp', 'within_ne'])['share_target'].mean()
    prob_share = pd.DataFrame(prob_share).reset_index()
    prob_share['n'] = pairs.groupby(['exp', 'within_ne']).input1.size().values
    for exp in prob_share.exp.unique():
        if len(prob_share[prob_share.exp == exp]) < 2:
            prob_share = prob_share[prob_share.exp != exp]
    m = prob_share.groupby('within_ne')['share_target'].mean()
    sd = prob_share.groupby('within_ne')['share_target'].std()
    m = m.iloc[[1, 0]]
    sd = sd.iloc[[1, 0]]
    m.plot.bar(edgecolor=['k', 'grey'], ax=ax, facecolor='w', linewidth=2)
    ebar_colors=['k', 'grey']
    for i in range(0, len(prob_share), 2):
        ax.plot([1, 0], np.array(prob_share.iloc[i: i+2]['share_target']), 'k', linewidth=.6)
    for c in range(2):
        ax.errorbar(x=c, y=m.iloc[c], yerr=np.array([[0], [sd.iloc[c]]]), fmt='None', color=ebar_colors[c], 
                    capsize=5, linewidth=1, zorder=1)
        prob_share_tmp = prob_share[prob_share.within_ne == 1 - c]
        ax.scatter(c * np.ones(len(prob_share)//2), prob_share_tmp['share_target'], 
                   facecolor=ebar_colors[c], s=prob_share_tmp['n']*2, edgecolor='w', linewidth=.5)
        print('n=', prob_share_tmp['n'].sum())
    ax.scatter([.8, 1, 1.2], [.6, .6, .6], facecolor='k', s=np.array([5, 10, 20])*2, edgecolor='w', linewidth=.5)
    ax.set_ylabel('P(share A1 target)')
    
    _, p = stats.wilcoxon(prob_share[prob_share.within_ne == 1]['share_target'],
                          prob_share[prob_share.within_ne == 0]['share_target'])
    print('n = ', len(prob_share)/2 ,p)
    plot_significance_star(ax, p, [0, 1], .85, .87)
    ax.set_ylim([0, 1])

    ax.set_xlabel('')
    ax.set_xticklabels(['Within cNE', 'Outside cNE'], rotation=0)

    if savefolder is not None:
        fig.savefig(os.path.join(savefolder, 'pair_p_share_target_ne_member.jpg'), dpi=300)
        plt.close()


def figure5(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure_v2'):
    
    fig = plt.figure(figsize=[17.6*cm, 7.5*cm])
    # PART1: plot example cNE and A1 connectin
    # load nepiars
    example_file = os.path.join(datafolder, '200820_230604-site4-5655um-25db-dmr-31min-H31x64-fs20000-pairs-ne-10-spon-0.5ms_bin.json')
    nepairs = pd.read_json(example_file)
    # load ne info
    exp = re.search('\d{6}_\d{6}', example_file).group(0)
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
    nefile = re.sub('pairs-ne-10-spon-0.5ms_bin.json', 'ne-10dft-spon.pkl', example_file)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    patterns = ne.patterns
    # plot example cen
    cne = 3
    target_idx = [3, 38]
    ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_idx.isin(target_idx))].copy()
    n_pairs = len(ne_neuron_pairs.input_idx.unique())
    ne_neuron_pairs1 = ne_neuron_pairs[nepairs.target_idx == target_idx[0]].copy()
    ne_neuron_pairs2 = ne_neuron_pairs[nepairs.target_idx == target_idx[1]].copy()
    # 1. probe
    # 1.1 MGB
    x_start = .06
    x_fig = .02
    y_start = .02
    y_fig = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs1, input_units)
    ax.set_ylim([5800, 4600])
    ax.set_ylabel(r'Depth ($\mu$m)', labelpad=0)
    # add axes for waveform plot
    x_start = .09
    y_start = .2
    x_fig = .03
    y_fig = .05
    y_space = .01
    axes = add_multiple_axes(fig, n_pairs, 1, x_start, y_start, x_fig, y_fig, 0, y_space)
    position_idx_ne = [position_idx[unit_idx] + 1 for unit_idx in ne_neuron_pairs1.input_idx]
    ne_neuron_pairs1['position_idx'] = position_idx_ne
    ne_neuron_pairs1 = ne_neuron_pairs1.sort_values(by='position_idx')
    plot_all_waveforms(axes, ne_neuron_pairs1, input_units, position_idx=position_idx)
    for ax in axes:
        ax[0].set_title('')
    # 1.2 A1
    x_start = .18
    x_fig = .02
    y_start = .15
    y_fig = .4
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    _ = plot_position_on_probe(ax, ne_neuron_pairs, target_units, location='A1')
    ax.set_ylim([1000, 200])
    # add axes for waveform plot
    x_start = .15
    y_start = .35
    x_fig = .03
    y_fig = .05
    axes = add_multiple_axes(fig, 2, 1, x_start, y_start, x_fig, y_fig, 0, y_space)
    plot_all_waveforms(axes, ne_neuron_pairs, target_units, 'target')
    
    # 2. icweight
    #y_start = 1/ (1.8 * n_pairs + bottom_space)
    #y_fig = 2 / (1.8 * n_pairs + bottom_space)
    x_start = .06
    x_fig = .2
    y_start = .8
    y_fig = .12
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    weights = patterns[cne]
    weights = weights[position_order]
    member_thresh = 1 / np.sqrt(patterns.shape[1])
    plot_ICweight(ax, weights, member_thresh, direction='h', markersize=2)
    ax.set_ylim([-.2, .8])
    ax.set_yticks([0, .4, .8])
    
    # cNE3-A1_17
    # add axes for ccg plot
    x_start = .38
    y_start = .08
    x_fig = .12
    x_space = .02
    y_fig = .18
    y_space =  0.03
    ne_neuron_pairs1 = ne_neuron_pairs1[ne_neuron_pairs1.position_idx != 13]
    axes = add_multiple_axes(fig, 3, 2, x_start, y_start, x_fig, y_fig, x_space, y_space)
    plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs1, stim='spon')
    for ax in axes.flatten():
        ax.set_ylim([0, 200])
        ax.set_yticks(range(0, 201, 50))
        ax.set_yticklabels([])
    for ax in axes[:, 0]:
        ax.set_yticklabels([0, '', 100, '', 200]) 
    
    # cNE3-A1_79
    ne_neuron_pairs2['position_idx'] = position_idx_ne
    ne_neuron_pairs2 = ne_neuron_pairs2.sort_values(by='position_idx')
    ne_neuron_pairs2 = ne_neuron_pairs2[ne_neuron_pairs2.position_idx != 13]
    x_start = .72
    axes = add_multiple_axes(fig, 3, 2, x_start, y_start, x_fig, y_fig, x_space, y_space)
    plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs2, stim='spon', force_causal=True)
    for ax in axes.flatten():
        ax.set_ylim([0, 100])
        ax.set_yticks(range(0, 101, 25))
        ax.set_yticklabels([])
    for ax in axes[:, 0]:
        ax.set_yticklabels([0, '', 50, '', 100]) 
        
    fig.savefig(os.path.join(figfolder, 'fig5.pdf'), dpi=300)
    plt.close()

def figure6(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure_v2'):
    
    
    fig = plt.figure(figsize=[17.6*cm, 6.5*cm])
    # PART1: plot example cNE and A1 connectin
    # load nepiars
    example_file = os.path.join(datafolder, '220825_005353-site6-5500um-25db-dmr-61min-H31x64-fs20000-pairs-ne-10-spon-0.5ms_bin.json')
    nepairs = pd.read_json(example_file)
    # load ne info
    exp = re.search('\d{6}_\d{6}', example_file).group(0)
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
    nefile = re.sub('-pairs-ne-10-spon-0.5ms_bin.json', '-ne-10dft-spon.pkl', example_file)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    patterns = ne.patterns
    # plot example cen
    
    target_idx = 10
    cne = [5, 0]
    ne_neuron_pairs1 = nepairs[(nepairs.cne == cne[0]) & (nepairs.target_idx == target_idx)].copy()
    ne_neuron_pairs2 = nepairs[(nepairs.cne == cne[1]) & (nepairs.target_idx == target_idx)].copy()
    x_offset = .3
    # 1. probe
    # 1.1 A1
    x_start = .06 + x_offset
    x_fig = .02
    y_start = .02
    y_fig = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    _ = plot_position_on_probe(ax, ne_neuron_pairs1, target_units, location='A1')
    ax.yaxis.tick_left()
    ax.set_xlim([-100, 100])
    ax.spines[['right']].set_visible(False)
    ax.spines[['left']].set_visible(True)
    ax.set_ylim([1250, 0])
    ax.set_yticks(range(0, 1251, 250))
    ax.set_ylabel(r'Depth ($\mu$m)', labelpad=0)

    # add axes for waveform plot
    x_start = .09 + x_offset
    y_start = .35
    x_fig = .03
    y_fig = .05
    y_space = .01
    axes = add_multiple_axes(fig, 1, 1, x_start, y_start, x_fig, y_fig, 0, y_space)
    plot_all_waveforms(axes, ne_neuron_pairs1, target_units, 'target')
    
    # 1.2.1 MGB
    x_start = .18 + x_offset
    x_fig = .02
    y_start = .02
    y_fig = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs1, input_units)
    ax.set_ylim([5500, 4300])
    ax.set_axis_off()
    # add axes for waveform plot
    x_start = .15 + x_offset
    y_start = .2
    x_fig = .03
    y_fig = .05
    n_pairs = len(ne_neuron_pairs1)
    axes = add_multiple_axes(fig, n_pairs, 1, x_start, y_start, x_fig, y_fig, 0, y_space)
    position_idx_ne = [position_idx[unit_idx] + 1 for unit_idx in ne_neuron_pairs1.input_idx]
    ne_neuron_pairs1['position_idx'] = position_idx_ne
    ne_neuron_pairs1 = ne_neuron_pairs1.sort_values(by='position_idx')
    plot_all_waveforms(axes, ne_neuron_pairs1, input_units, position_idx=position_idx)
    for ax in axes:
        ax[0].set_title('')
    #1.2.2
    x_start = .25 + x_offset
    x_fig = .02
    y_start = .02
    y_fig = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs2, input_units)
    ax.set_ylim([5500, 4300])
    ax.set_yticks(range(4300, 5501, 200))
    ax.yaxis.tick_right()
    ax.spines[['right']].set_visible(True)
    ax.spines[['left']].set_visible(False)
    # add axes for waveform plot
    x_start = .22 + x_offset
    y_start = .2
    x_fig = .03
    y_fig = .05
    n_pairs = len(ne_neuron_pairs2)
    axes = add_multiple_axes(fig, n_pairs, 1, x_start, y_start, x_fig, y_fig, 0, y_space)
    position_idx_ne = [position_idx[unit_idx] + 1 for unit_idx in ne_neuron_pairs2.input_idx]
    ne_neuron_pairs2['position_idx'] = position_idx_ne
    ne_neuron_pairs2 = ne_neuron_pairs2.sort_values(by='position_idx')
    plot_all_waveforms(axes, ne_neuron_pairs2, input_units, position_idx=position_idx)
    for ax in axes:
        ax[0].set_title('')
   
    # 2. icweight
    #y_start = 1/ (1.8 * n_pairs + bottom_space)
    #y_fig = 2 / (1.8 * n_pairs + bottom_space)
    x_start = .06 + x_offset
    x_fig = .12
    y_start = .8
    y_fig = .12
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    weights = patterns[cne[0]]
    weights = weights[position_order]
    member_thresh = 1 / np.sqrt(patterns.shape[1])
    plot_ICweight(ax, weights, member_thresh, direction='h', markersize=2)
    ax.set_ylim([-.2, .8])
    ax.set_yticks([0, .4, .8])
    x_start = .25 + x_offset
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    weights = patterns[cne[1]]
    weights = weights[position_order]
    plot_ICweight(ax, weights, member_thresh, direction='h', markersize=2)
    ax.set_ylim([-.2, .8])
    ax.set_yticks([0, .4, .8])

    # cNE4-A1_149
    # add axes for ccg plot
    x_start = .05
    y_start = .15
    x_fig = .1
    x_space = .02
    y_fig = .22
    y_space =  0.05
    axes = add_multiple_axes(fig, 3, 2, x_start, y_start, x_fig, y_fig, x_space, y_space)
    ne_neuron_pairs1 = ne_neuron_pairs1[ne_neuron_pairs1.position_idx.isin([11, 12, 14])]
    plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs1, stim='spon')
    for ax in axes.flatten():
        ax.set_ylim([0, 100])
        ax.set_yticks(range(0, 101, 50))
        ax.set_yticklabels([])
    for ax in axes[:, 0]:
        ax.set_yticklabels(range(0, 101, 50)) 
    # cNE4-A1_149
    # add axes for ccg plot
    x_start = .75
    axes = add_multiple_axes(fig, 3, 2, x_start, y_start, x_fig, y_fig, x_space, y_space)
    ne_neuron_pairs2 = ne_neuron_pairs2[ne_neuron_pairs2.position_idx.isin([7, 8, 11])]
    plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs2, stim='spon', force_causal=True)
    for ax in axes.flatten():
        ax.set_ylim([0, 50])
        ax.set_yticks(range(0, 51, 25))
        ax.set_yticklabels([])
    for ax in axes[:, 0]:
        ax.set_yticklabels(range(0, 51, 25)) 
    
    fig.savefig(os.path.join(figfolder, 'fig6.pdf'), dpi=300)
    plt.close()


def figure4(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure_v3',
            subsample=False):
    
    fig = plt.figure(figsize=[figure_size[0][0], 12.5 * cm])
    x_fig = .17
    y_fig = .2
    # panel A: BS/NS neurons waveform ptd
    print('A')
    y_start = .7
    x_start = .05
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_waveform_ptd(ax=ax)
    ax.set_ylabel('# of A1 neurons')
    print('B')
    # panel B:NE vs nonNE spike efficacy
    x_start = .3
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_efficacy_ne_vs_nonne(ax, celltype=True, subsample=subsample, file='ne-pairs-10df-spon-0.5ms_bin.json')
    # efficacy gain boxplot
    ax = fig.add_axes([x_start+.1, y_start, x_fig*.3, y_fig*.5])
    plot_efficacy_gain_cell_type(ax=ax, subsample=subsample,  file='ne-pairs-10df-spon-0.5ms_bin.json')

    x_start = .55
    # panel E: percentage of spikes followed by A1 firing
    print('C')
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_percent_spike_with_A1_firing(ax)
    print('D')
    x_start = .8
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_A1_nspk_following(ax)
    fig.savefig(os.path.join(figfolder, 'fig4.pdf'), dpi=300)
    x_start = .05
    y_start = .4
    # panel C: efficacy gain vs fr
    print('E')
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_efficacy_change_vs_target_fr(ax)
    print('F')
    x_start = .3
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_contribution(ax)
    x_start = .55
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_ns_bs_ccg(ax)
    ax = fig.add_axes([x_start+.05, y_start+.08, x_fig *.3, y_fig*.5])
    plot_ns_bs_fr_prior(ax)
    x_start = .8
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_ns_ne_nonne_ccg(ax)
    ax = fig.add_axes([x_start+.05, y_start+.08, x_fig *.3, y_fig*.5])
    plot_ne_nonne_fr_prior2(ax)
    fig.savefig(os.path.join(figfolder, 'fig4_5ms.pdf'), dpi=300)


def plot_efficacy_change_vs_target_fr(ax, stim="spon",
        summary_folder=r'E:\Congcong\Documents\data\connection\data-summary'):
    pair_file = os.path.join(summary_folder, f'ne-pairs-perm-test-{stim}.json')
    pairs = pd.read_json(pair_file)
    pairs["ns"] = pairs.target_waveform_tpd < .45
    pairs["efficacy_change"] = pairs.efficacy_ne_spon - pairs.efficacy_nonne_spon
    
    ax.set_xlim([0, 20])
    sns.regplot(data=pairs, x="target_fr", y="efficacy_change", color="k",
                scatter=False, truncate=False)
    sns.scatterplot(data=pairs, x="target_fr", y="efficacy_change", 
                    hue="ns", hue_order=[True, False], palette=reversed(tpd_color[:2]),
                    s=15, alpha=.8, axes=ax, legend=None)
    #legend_handles, _ = ax.get_legend_handles_labels()
    #ax.legend(handles=legend_handles, labels=["NS", "BS"])
    
    ax.set_xlabel('A1 neuron FR (Hz)')
    ax.set_ylabel('\u0394efficacy (%)')
    ax.set_ylim([-10, 15])
    a, b = np.polyfit(pairs.target_fr, pairs.efficacy_change, 1)
    r2 = r2_score(pairs.efficacy_change, a * pairs.target_fr + b)
    ax.text(13, -8, f"R2={r2:.1e}")


def plot_efficacy_ne_vs_nonne(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary', 
                              file=None,
                              stim='spon', change=None, celltype=False, sig=False, 
                              subsample=False, coincidence=False, df=10):
    if file is None:
        if coincidence:
            file = r"ne-pairs-act-level-{stim}-{}ms.json".format(df/2)
        else:
            file = f'ne-pairs-{df}df-{stim}-0.5ms_bin.json'
        
    pairs = pd.read_json(os.path.join(datafolder, file))
    if 'ss' in stim:
        pairs_tmp = pd.read_json(os.path.join(datafolder,  f'ne-pairs-{df}df-spon-0.5ms_bin.json'))
        pairs[f'inclusion_{stim}'] = pairs_tmp[f'inclusion_spon']
    pairs = pairs[pairs[f'inclusion_{stim}']]
    pairs = pairs[(pairs[f'efficacy_ne_{stim}'] > 0) & (pairs[f'efficacy_nonne_{stim}'] > 0)]
    
    pairs['waveform_ns'] = pairs.target_waveform_tpd < .45
    
    if subsample:
        pairs[f'efficacy_ne_{stim}'] = pairs[f'efficacy_ne_{stim}_subsample']
        pairs[f'efficacy_nonne_{stim}'] = pairs[f'efficacy_nonne_{stim}_subsample']
    elif coincidence:
        pairs[f'efficacy_nonne_{stim}'] = pairs['efficacy_hiact_mean']
        pairs[f'efficacy_ne_{stim}'] = pairs['efficacy_ne_subsample_hiact_mean']
        
    if celltype:
        # color-code cell types
        for i in reversed(range(2)):
            if i == 0:
                pairs_tmp = pairs.query("waveform_ns == False")
            else:
                pairs_tmp = pairs.query("waveform_ns == True")
                             
            ax.scatter(pairs_tmp[f'efficacy_nonne_{stim}'], 
                       pairs_tmp[f'efficacy_ne_{stim}'], 
                       s=15,  color=tpd_color[i], edgecolor='w')
            _, p = stats.wilcoxon(pairs_tmp[f'efficacy_ne_{stim}'], pairs_tmp[f'efficacy_nonne_{stim}'])
            print(f'p = {p} (n={len(pairs_tmp)})')
            if p > .001:
                ax.text(2, 23, f'p = {p:.3f}', fontsize=7)
            else:
                ax.text(2, 23, f'p = {p:.2e}', fontsize=7)
    else:
        ax.scatter(pairs[f'efficacy_nonne_{stim}'], pairs[f'efficacy_ne_{stim}'], 
                   s=15, color='k', edgecolor='w')
        _, p = stats.wilcoxon(pairs[f'efficacy_ne_{stim}'], pairs[f'efficacy_nonne_{stim}'])
        print('p =', p)
        if p > .001:
            ax.text(2, 23, f'p = {p:.3f}', fontsize=7)
        else:
            ax.text(2, 23, f'p = {p:.2e}', fontsize=7)
    
    if change == 'increase':
        pairs = pairs[pairs[f'efficacy_ne_{stim}'] > pairs[f'efficacy_nonne_{stim}']]
    elif change == 'decrease':
        pairs = pairs[pairs[f'efficacy_ne_{stim}'] < pairs[f'efficacy_nonne_{stim}']]
    
    
    if sig:
        # plot data points with significance
        p_thresh = .05 #/ len(pairs)
        pairs_sig = pairs[(pairs.efficacy_diff_p < p_thresh) & (pairs['waveform_ns'])]
        ax.scatter(pairs_sig[f'efficacy_nonne_{stim}'], pairs_sig[f'efficacy_ne_{stim}'], s=15, color=tpd_color[1], edgecolor='w')
        pairs_sig = pairs[(pairs.efficacy_diff_p < p_thresh) & (~pairs['waveform_ns'])]
        ax.scatter(pairs_sig[f'efficacy_nonne_{stim}'], pairs_sig[f'efficacy_ne_{stim}'], s=15, color=tpd_color[0], edgecolor='w')
        print('non-sig: ', sum(pairs.efficacy_diff_p > p_thresh))
        print('sig: ', sum(pairs.efficacy_diff_p < p_thresh))
    
    ax.plot([0, 30], [0, 30], 'k')
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])
    ax.set_xticks(range(0, 31, 10))
    ax.set_yticks(range(0, 31, 10))
    ax.set_xlabel('non-cNE spike efficacy (%)')
    ax.set_ylabel('cNE spike efficacy (%)')
    print(len(pairs))
    
    # add box plot 
    axin = inset_axes(
        ax,
        width="20%",  # width: 5% of parent_bbox width
        height="60%",  # height: 50%
        loc="center left",
        bbox_to_anchor=(0.85, -.1, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    axin.boxplot(pairs[f'efficacy_ne_{stim}']- pairs[f'efficacy_nonne_{stim}'], showfliers=False)
    axin.set_xlim([.8, 1.2])
    axin.set_ylim([-10, 15])
    axin.spines[['bottom']].set_visible(False)
    axin.set_xticks([])
    axin.set_ylabel('\u0394efficacy (%)', fontsize=5)


    
    
def plot_waveform_ptd(datafolder='E:\Congcong\Documents\data\connection\data-pkl', region='A1', 
                      ax=None, savefolder=None):
    files = glob.glob(os.path.join(datafolder, '*fs20000.pkl'))
    tpd = []
    for file in files:
        depth = int(re.search('\d+(?=um)', file).group(0))
        if region == 'MGB' and depth < 3000:
            continue
        elif region == 'A1' and depth > 2000:
            continue
        
        with open(file, 'rb') as f:
            session = pickle.load(f)
        
        for unit in session.units:
            tpd.append(unit.waveform_tpd)
    if ax is  None:
        fig = plt.figure(figsize=[2, 2])
        ax = fig.add_axes([.1, .1, .8, .8])
    ax.hist(tpd, np.arange(0, .5, .05), color=tpd_color[1])
    ax.hist(tpd, np.arange(.45, 1.5, .05), color=tpd_color[0])
    ax.plot([.45, .45], [0, 80], 'k--')
    ax.set_xlabel('Trough-Peak delay (ms)')
    ax.set_ylabel('# of neurons')
    ax.set_xlim([0, 1.5])
    ax.set_ylim([0, 80])

    if savefolder:
        fig.savefig(os.path.join(savefolder, f'tpd-{region}.jpg'), bbox_inches='tight', dpi=300)


def plot_efficacy_gain_cell_type(ax, datafolder='E:\Congcong\Documents\data\connection\data-summary', 
                                 stim='spon', subsample=False, df=10, file = None):
    
    if file is not None:
        file = glob.glob(os.path.join(datafolder, file))[0]
    else:
        file = glob.glob(os.path.join(datafolder,f'ne-pairs-{df}df-{stim}.json'))[0]

    pairs = pd.read_json(file)
    pairs = pairs[pairs[f'inclusion_{stim}']]
    pairs = pairs[(pairs[f'efficacy_ne_{stim}'] > 0) & (pairs[f'efficacy_nonne_{stim}'] > 0)]
    pairs['waveform_ns'] = pairs.target_waveform_tpd < .45
    if subsample:
        pairs[f'efficacy_ne_{stim}'] = pairs[f'efficacy_ne_{stim}_subsample']
        pairs[f'efficacy_nonne_{stim}'] = pairs[f'efficacy_nonne_{stim}_subsample']
    pairs['efficacy_gain'] = (pairs[f'efficacy_ne_{stim}'] - pairs[f'efficacy_nonne_{stim}'])
    boxplot_scatter(ax, x='waveform_ns', y='efficacy_gain', data=pairs, size=3, jitter=.3,
                    order=[True, False], hue='waveform_ns', 
                    palette=tpd_color[1::-1], hue_order=[True, False])
    ax.set_xticklabels(['NS', 'BS'])
    ax.set_xlabel('A1 neuron type')
    ax.set_ylabel('Efficacy gain (%)')
    
    _, p = stats.mannwhitneyu(pairs[pairs.waveform_ns]['efficacy_gain'], 
                              pairs[~pairs.waveform_ns]['efficacy_gain'])
    print('p =', p)
    print('NS: ', pairs['waveform_ns'].sum())
    print('BS: ', len(pairs) - pairs['waveform_ns'].sum())
    plot_significance_star(ax, p, [0, 1], 15, 16)
    ax.plot([-.5, 1.5], [0, 0], 'k--')
    ax.set_ylim([-10, 20])
    ax.set_yticks(range(-10, 21, 10))
    ax.set_xlim([-.5, 1.5])
    

def plot_delta_dfficacy_ne_vs_hiact(axes, datafolder=r'E:\Congcong\Documents\data\connection\data-summary',
                                    stim='spon', coincidence='act-level', mode='raw', window=10, 
                                    savepath=None, method='raw', sig=False):
    # method raw, sig false: stress the difference between ne and hiact spikes efficacy
    # method raw, sig True: show pairs with significant difference
    # method subsample, sig True: show pairs with significant difference
    # method subsample, sig True: show pairs with significant difference

    pairs = pd.read_json(os.path.join(datafolder, f'ne-pairs-{coincidence}-{stim}-{window}ms-zscore.json'))
    zscore_thresh = stats.norm.ppf(1 - 0.05 / len(pairs))
    if method == 'raw':
        pairs[f'efficacy_diff_ne_{stim}'] = pairs[f'efficacy_ne_{stim}'] - pairs[f'efficacy_neuron_{stim}'] 
        pairs[f'efficacy_diff_hiact_{stim}'] = pairs[f'efficacy_hiact'] - pairs[f'efficacy_neuron_{stim}'] 
    elif method == 'subsample':
        pairs[f'efficacy_diff_ne_{stim}'] = pairs[f'efficacy_ne_subsample'].apply(np.mean) - pairs[f'efficacy_neuron_{stim}'] 
        pairs[f'efficacy_diff_hiact_{stim}'] = pairs[f'efficacy_hiact_subsample'].apply(np.mean) - pairs[f'efficacy_neuron_{stim}'] 
    pairs_posi_neg = [pairs[pairs[f'efficacy_diff_ne_{stim}'] > 0], pairs[pairs[f'efficacy_diff_ne_{stim}'] < 0]]

    markersize = 10
    if sig:
        for idx_plot, pairs in enumerate(pairs_posi_neg):
            pairs_sig = pairs[np.abs(pairs.efficacy_ne_hiact_z) > zscore_thresh]
            pairs_nonsig =  pairs[np.abs(pairs.efficacy_ne_hiact_z) < zscore_thresh]
            ax = axes[idx_plot]
            # line plot
            for i in range(len(pairs_sig)):
                pair = pairs_sig.iloc[i]
                c = tpd_color[1] if pair.target_waveform_tpd < .45 else tpd_color[0]
                ax.plot([pair[f'efficacy_diff_ne_{stim}'], pair[f'efficacy_diff_hiact_{stim}']],
                        [pair[f'efficacy_neuron_{stim}'], pair[f'efficacy_neuron_{stim}']], color=c)
            for i in range(len(pairs_nonsig)):
                pair = pairs_nonsig.iloc[i]
                c = 'grey'
                ax.plot([pair[f'efficacy_diff_ne_{stim}'], pair[f'efficacy_diff_hiact_{stim}']],
                        [pair[f'efficacy_neuron_{stim}'], pair[f'efficacy_neuron_{stim}']], color=c)
    
    
            h1 = ax.scatter(pairs_nonsig[f'efficacy_diff_ne_{stim}'], 
                       pairs_nonsig[f'efficacy_neuron_{stim}'], 
                       c='grey', alpha=.8, s=markersize)
            ax.scatter(pairs_sig[pairs_sig.target_waveform_tpd < .45][f'efficacy_diff_ne_{stim}'], 
                       pairs_sig[pairs_sig.target_waveform_tpd < .45][f'efficacy_neuron_{stim}'],
                       color=tpd_color[1], alpha=.8, s=markersize)
            ax.scatter(pairs_sig[pairs_sig.target_waveform_tpd > .45][f'efficacy_diff_ne_{stim}'], 
                       pairs_sig[pairs_sig.target_waveform_tpd > .45][f'efficacy_neuron_{stim}'], 
                       color=tpd_color[0], alpha=.8, s=markersize)
        
        
            h2 = ax.scatter(pairs_nonsig[f'efficacy_diff_hiact_{stim}'], 
                       pairs_nonsig[f'efficacy_neuron_{stim}'], 
                       color='grey', marker='s', alpha=.8, s=markersize)
            ax.scatter(pairs_sig[pairs_sig.target_waveform_tpd < .45][f'efficacy_diff_hiact_{stim}'], 
                       pairs_sig[pairs_sig.target_waveform_tpd < .45][f'efficacy_neuron_{stim}'],
                       color=tpd_color[1], marker='s', alpha=.8, s=markersize)
            ax.scatter(pairs_sig[pairs_sig.target_waveform_tpd > .45][f'efficacy_diff_hiact_{stim}'], 
                       pairs_sig[pairs_sig.target_waveform_tpd > .45][f'efficacy_neuron_{stim}'], 
                       color=tpd_color[0], marker='s', alpha=.8, s=markersize)
            ax.plot([0, 0], [0, 20], 'k')
            ax.set_xlim([-20, 20])
            ax.set_ylim([0, 20])
            if idx_plot == 1:
                ax.set_xlabel('\u0394efficacy (%)')
                ax.set_ylabel('Efficacy of all spikes (%)', labelpad=0)
            else:
                ax.set_xticklabels([])
                ax.legend([h1, h2], ['NE spikes', 'coincident spikes'], bbox_to_anchor=(.5, 1.7), loc='upper center')
    
            _, p = stats.wilcoxon(pairs[f'efficacy_diff_ne_{stim}'], pairs[f'efficacy_diff_hiact_{stim}'])
            print(stim, window, idx_plot, 'p =', p)
            if p < .01:
                ax.text(-18, 3, f'p = {p:.1e}', fontsize=6)
            else:
                ax.text(-18, 3, f'p = {p:.2f}', fontsize=6)

        
    else:
        for idx_plot, pairs in enumerate(pairs_posi_neg):
            ax = axes[idx_plot]
            for i in range(len(pairs)):
                pair = pairs.iloc[i]
                ax.plot([pair[f'efficacy_diff_ne_{stim}'], pair[f'efficacy_diff_hiact_{stim}']],
                    [pair[f'efficacy_neuron_{stim}'], pair[f'efficacy_neuron_{stim}']], color='grey')
            
            h1 = ax.scatter(pairs[f'efficacy_diff_ne_{stim}'], pairs[f'efficacy_neuron_{stim}'], 
                   color=ne_hiact_color[0], s=markersize, alpha=.8)
            h2 = ax.scatter(pairs[f'efficacy_diff_hiact_{stim}'], pairs[f'efficacy_neuron_{stim}'], 
                   color=ne_hiact_color[1], marker='s', s=markersize, alpha=.8)
            ax.plot([0, 0], [0, 20], 'k')
            ax.set_xlim([-20, 20])
            ax.set_ylim([0, 20])
            if idx_plot == 1:
                ax.set_xlabel('\u0394efficacy (%)')
                ax.set_ylabel('Efficacy of all spikes (%)', labelpad=0)
            else:
                ax.set_xticklabels([])
                ax.legend([h1, h2], ['NE spikes', 'coincident spikes'], bbox_to_anchor=(.5, 1.7), loc='upper center')
    
            _, p = stats.wilcoxon(pairs[f'efficacy_diff_ne_{stim}'], pairs[f'efficacy_diff_hiact_{stim}'])
            print(stim, window, idx_plot, 'p =', p)
            if p < .01:
                ax.text(-18, 3, f'p = {p:.1e}', fontsize=6)
            else:
                ax.text(-18, 3, f'p = {p:.2f}', fontsize=6)
    
    if savepath:
        plt.savefig(savepath, dpi=1000)
        plt.close()
    
    
def plot_delta_dfficacy_ne_vs_hiact_hist(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary',
                                    stim='spon', coincidence='act-level', mode='diff'):
    pairs = pd.read_json(os.path.join(datafolder, f'ne-pairs-{coincidence}-{stim}.json'))
    if mode == 'diff':
        pairs['efficacy_diff_ne'] = pairs[f'efficacy_ne_{stim}'] - pairs[f'efficacy_neuron_{stim}'] 
        pairs['efficacy_diff_act'] = pairs[f'efficacy_hiact'] - pairs[f'efficacy_neuron_{stim}'] 
    elif mode == 'raw':
        pairs['efficacy_diff_ne'] = pairs[f'efficacy_ne_{stim}']
        pairs['efficacy_diff_act'] = pairs[f'efficacy_hiact']
    pairs = pairs[(pairs.efficacy_diff_p < .05) & 
                      (pairs[f'efficacy_ne_{stim}']  > pairs[f'efficacy_nonne_{stim}'])] 
    m = [pairs['efficacy_diff_ne'].mean(), pairs['efficacy_diff_act'].mean()]
    sd =  [pairs['efficacy_diff_ne'].std(), pairs['efficacy_diff_act'].std()]
    ax.bar([0, 1], m, edgecolor=['k', 'grey'], facecolor='w', linewidth=1.5)
    ebar_colors=['k', 'grey']
    for i in range(len(pairs)):
        pair = pairs.iloc[i]
        ax.plot([0, 1], [pair.efficacy_diff_ne, pair.efficacy_diff_act], 'k', linewidth=.6)
    for c in range(2):
        ax.errorbar(x=c, y=m[c], yerr=sd[c], fmt='None', color=ebar_colors[c], 
                    capsize=5, linewidth=1.5, zorder=1)
    ax.scatter(0 * np.ones(len(pairs)), pairs.efficacy_diff_ne, 
               facecolor=ebar_colors[0], s=15, edgecolor='w', linewidth=.5)
    ax.scatter(1 * np.ones(len(pairs)), pairs.efficacy_diff_act, 
               facecolor=ebar_colors[1], s=15, edgecolor='w', linewidth=.5)
    if mode == 'diff':
        ax.set_ylabel('\u0394efficacy (%)')
    elif mode == 'raw':
        ax.set_ylabel('efficacy (%)')
    _, p = stats.wilcoxon(pairs.efficacy_diff_ne,
                          pairs.efficacy_diff_act)
    print('Wilcoxon: p =', p)
    if 'ss' in stim:
        plot_significance_star(ax, p, [0, 1], 19, 19.5)
        ax.set_ylim([0, 20])
    else:
        plot_significance_star(ax, p, [0, 1], 14, 14.5)
        ax.set_ylim([0, 15])

    ax.set_xlabel('')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['NE spikes', 'coincident spikes'], rotation=0)


def figure3(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper'):
    
    fig = plt.figure(figsize=[17.6*cm, 8.5*cm])
    
    # summary plot 
    x_start = .6
    y_start = .08
    x_fig = .15
    y_fig = .25
    x_space = .08
    y_space = .25
    axes = add_multiple_axes(fig, 2, 2, x_start, y_start, x_fig, y_fig, x_space, y_space)
    axes= axes.flatten()
    plot_efficacy_ne_vs_nonne(axes[0], stim='spon')
    plot_efficacy_ne_vs_nonne(axes[1], stim='spon', subsample=True)
    plot_efficacy_ne_vs_nonne(axes[2], stim='spon_ss')
    plot_efficacy_ne_vs_nonne(axes[3], stim='spon', coincidence=True, file = "ne-pairs-act-level-spon-5ms.json")
    axes[3].set_xlabel('Coincident spike efficacy (%)')
    fig.savefig(os.path.join(figfolder, 'fig3.pdf'), dpi=300)
    
    # PART1: plot example cNE and A1 connectin
    # load nepiars
    example_file = os.path.join(datafolder, '210401_175257-site2-4900um-20db-dmr-37min-H31x64-fs20000-pairs-ne-10-spon.json')
    nepairs = pd.read_json(example_file)
    # load ne info
    exp = re.search('\d{6}_\d{6}', example_file).group(0)
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
    nefile = re.sub('-pairs-ne-10-spon.json', '-ne-10dft-spon.pkl', example_file)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    patterns = ne.patterns
    # plot example cne
    cne = 0
    target_idx = 32
    ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_idx == target_idx)].copy()
    n_pairs = len(ne_neuron_pairs)
    
    # 1. probe
    # 1.1 MGB
    x_start = .06
    x_fig = .02
    y_start = .02
    y_fig = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs, input_units)
    ax.set_ylim([5800, 4600])
    ax.set_ylabel(r'Depth ($\mu$m)', labelpad=0)
    # add axes for waveform plot
    x_start = .09
    y_start = .2
    x_fig = .03
    y_fig = .05
    y_space = .01
    axes = add_multiple_axes(fig, n_pairs, 1, x_start, y_start, x_fig, y_fig, 0, y_space)
    position_idx_ne = [position_idx[unit_idx] + 1 for unit_idx in ne_neuron_pairs.input_idx]
    ne_neuron_pairs['position_idx'] = position_idx_ne
    ne_neuron_pairs = ne_neuron_pairs.sort_values(by='position_idx')
    plot_all_waveforms(axes, ne_neuron_pairs, input_units, position_idx=position_idx)
    for ax in axes:
        ax[0].set_title('')
    
    x_start = .18
    x_fig = .02
    y_start = .15
    y_fig = .4
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    _ = plot_position_on_probe(ax, ne_neuron_pairs, target_units, location='A1')
    ax.set_ylim([1000, 200])
    # add axes for waveform plot
    x_start = .15
    y_start = .35
    x_fig = .03
    y_fig = .05
    ax = fig.add_axes([x_start, y_start,x_fig, y_fig])
    plot_all_waveforms(np.array([ax]), ne_neuron_pairs, target_units, 'target')
    ax.set_title('')
    
    # 2. icweight
    #y_start = 1/ (1.8 * n_pairs + bottom_space)
    #y_fig = 2 / (1.8 * n_pairs + bottom_space)
    x_start = .06
    x_fig = .15
    y_start = .8
    y_fig = .12
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    weights = patterns[cne]
    weights = weights[position_order]
    member_thresh = 1 / np.sqrt(patterns.shape[1])
    plot_ICweight(ax, weights, member_thresh, direction='h', markersize=2)
    ax.set_ylim([-.2, .8])
    ax.set_yticks([0, .4, .8])
    
    # add axes for ccg plot
    x_start = .3
    y_start = .08
    x_fig = .1
    x_space = .015
    y_fig = .14
    y_space =  0.03
    axes = add_multiple_axes(fig, n_pairs, 3, x_start, y_start, x_fig, y_fig, x_space, y_space)
    plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs, stim='spon')
    for ax in axes.flatten():
        ax.set_ylim([0, 6])
        ax.set_yticks(range(0, 7, 2))
        ax.set_yticklabels([])
    for ax in axes[:, 0]:
        ax.set_yticklabels([0, 2, 4, 6])
    
   
    # plot raw data
    fig.savefig(os.path.join(figfolder, 'fig3.pdf'), dpi=300)


def plot_contribution(ax, summary_folder=r'E:\Congcong\Documents\data\connection\data-summary'):
    pair_file = os.path.join(summary_folder, "pairs.json")
    pairs = pd.read_json(pair_file)
   
    pairs["contribution_spon"] = pairs.nspk_causal_spon / pairs.nspk_target_spon * 100
    pairs["ns"] = pairs.target_waveform_tpd < .45
    
    boxplot_scatter(ax, x='ns', y='contribution_spon', data=pairs, size=3, jitter=.3,
                   order=[True, False], hue='ns', 
                   palette=tpd_color[1::-1], hue_order=[True, False])
    
    ax.set_xticklabels(['NS', 'BS'])
    ax.set_xlabel('A1 cell type')
    ax.set_ylabel('MGB neuron contribution (%)')
    ax.set_ylim([0, 15])
    _, p = stats.mannwhitneyu(pairs[pairs.ns]["contribution_spon"], pairs[pairs.ns == False]["contribution_spon"])


def plot_percent_spike_with_A1_firing(ax, summary_folder=r'E:\Congcong\Documents\data\connection\data-summary'):
    pairs = pd.read_json(os.path.join(summary_folder, 'ne-pairs-nspk_following_ne_nonne_10dft_5ms.json'))
    ne_nspk_following = np.zeros([len(pairs), 5])
    nonne_nspk_following = np.zeros([len(pairs), 5])
    for i in range(len(pairs)):
        for input_type in ('ne', 'nonne'):
            nspk = pairs.iloc[i][f'{input_type}_nspk_following']
            for n, c in nspk.items():
                n = int(n)
                if input_type == 'ne':
                    ne_nspk_following[i][n] = c
                else:
                    nonne_nspk_following[i][n] = c
    # percent of spikes followed by A1 spike
    ne_prc = np.sum(ne_nspk_following[:, 1:], axis=1)/np.sum(ne_nspk_following, axis=1)
    prc1 = pd.DataFrame({'target_waveform_tpd': pairs['target_waveform_tpd'], 'prc_w_spk': ne_prc})
    prc1['ne'] = True
    nonne_prc = np.sum(nonne_nspk_following[:, 1:], axis=1)/np.sum(nonne_nspk_following, axis=1)
    prc2 = pd.DataFrame({'target_waveform_tpd': pairs['target_waveform_tpd'], 'prc_w_spk': nonne_prc})
    prc2['ne'] = False
    prc = pd.concat([prc1, prc2])
    prc['ns'] = prc['target_waveform_tpd'] < .45
    prc['ne_ns'] = prc['ne'].astype(str) + '_' + prc['ns'].astype(str)
    order=['True_True', 'False_True', 'True_False', 'False_False']
    colors = [tpd_color[x] for x in [1, 2, 0, 3]]
    boxplot_scatter(ax, x='ne_ns', y='prc_w_spk', data=prc, 
                    order=order, hue='ne_ns', palette=colors, 
                    hue_order=order, size=1.5, alpha=1, jitter=.3)
    model = ols( 'prc_w_spk ~ C(ne) + C(ns) +C(ne):C(ns)', data=prc).fit() 
    result = sm.stats.anova_lm(model, typ=2)
    print(result)
    # test for ne, nonne
    for i, ns in enumerate((True, False)):
        prc_tmp = prc[prc.ns == ns]
        _, p = stats.wilcoxon(prc_tmp[prc_tmp['ne']].prc_w_spk, prc_tmp[~prc_tmp['ne']].prc_w_spk)
        p *= 4
        print(p)
        plot_significance_star(ax, p, [i*2, i*2+1], .48, .49)
    # test for cell type
    for i, ne in enumerate((True, False)):
        prc_tmp = prc[prc['ne'] == ne]
        _, p = stats.mannwhitneyu(prc_tmp[prc_tmp['ns']].prc_w_spk, prc_tmp[~prc_tmp['ns']].prc_w_spk)
        p *= 4
        print(p)
        plot_significance_star(ax, p, [i, i+2], .55 + .05*i, .56 + .05*i)
    
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('Proportion of spikes\nfollowed by A1 activity (%)')
    ax.set_ylim([0, .6])
                 
    
def plot_A1_nspk_following(ax, summary_folder=r'E:\Congcong\Documents\data\connection\data-summary'):
    pairs = pd.read_json(os.path.join(summary_folder, 'ne-pairs-nspk_following_ne_nonne_10dft_20ms.json'))
    ne_nspk_following = np.zeros([len(pairs), 10])
    nonne_nspk_following = np.zeros([len(pairs), 10])
    for i in range(len(pairs)):
        for input_type in ('ne', 'nonne'):
            nspk = pairs.iloc[i][f'{input_type}_nspk_following']
            for n, c in nspk.items():
                n = int(n)
                if input_type == 'ne':
                    ne_nspk_following[i][n] = c
                else:
                    nonne_nspk_following[i][n] = c
    # percent of spikes followed by A1 spike
    ne_nspk = np.sum(ne_nspk_following[:, 1:] * range(1,10), axis=1)/np.sum(ne_nspk_following[:, 1:], axis=1)
    nspk1 = pd.DataFrame({'target_waveform_tpd': pairs['target_waveform_tpd'], 'nspk': ne_nspk})
    nspk1['ne'] = True
    nonne_nspk = np.sum(nonne_nspk_following[:, 1:] * range(1,10), axis=1)/np.sum(nonne_nspk_following[:, 1:], axis=1)
    nspk2 = pd.DataFrame({'target_waveform_tpd': pairs['target_waveform_tpd'], 'nspk': nonne_nspk})
    nspk2['ne'] = False
    nspk = pd.concat([nspk1, nspk2])
    nspk['ns'] = nspk['target_waveform_tpd'] < .45
    nspk['ne_ns'] = nspk['ne'].astype(str) + '_' + nspk['ns'].astype(str)
    order=['True_True', 'False_True', 'True_False', 'False_False']
    colors = [tpd_color[x] for x in [1, 2, 0, 3]]
    boxplot_scatter(ax, x='ne_ns', y='nspk', data=nspk, 
                    order=order, hue='ne_ns', palette=colors, 
                    hue_order=order, size=1.5, alpha=1, jitter=.3)
    model = ols( 'nspk ~ C(ne) + C(ns) +C(ne):C(ns)', data=nspk).fit() 
    result = sm.stats.anova_lm(model, typ=2)
    print(result)
    # test for cell type
    for i, ne in enumerate((True, False)):
        nspk_tmp = nspk[nspk['ne'] == ne]
        _, p = stats.mannwhitneyu(nspk_tmp[nspk_tmp['ns']].nspk, nspk_tmp[~nspk_tmp['ns']].nspk)
        p *= 2
        print(p)
        plot_significance_star(ax, p, [i, i+2], 1.84 + .05 * i, 1.245 + .05*i)
    
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('Mean # of A1 spikes follwing')
    ax.set_ylim([.99, 2])
    ax.set_yticks(np.arange(1, 2.1, .2))

def plot_ns_bs_ccg(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    pairs = pd.read_json(os.path.join(datafolder, 'pairs.json'))
    ccg_ns = np.stack(pairs[pairs.target_waveform_tpd < .45].ccg_10ms_norm)
    ccg_ns_mean = np.mean(ccg_ns, axis=0)
    ccg_ns_std = np.std([ccg_ns[:,:20], ccg_ns[:,-20:]])
    ccg_bs = np.stack(pairs[pairs.target_waveform_tpd >= .45].ccg_10ms_norm)
    ccg_bs_mean = np.mean(ccg_bs, axis=0)
    ccg_bs_std = np.std([ccg_bs[:, :20], ccg_bs[:, -20:]])
    edges = np.arange(-500, 500, 10)
    ax.step(edges, ccg_ns_mean, color=tpd_color[1], where='post')
    ax.step(edges, ccg_bs_mean, color=tpd_color[0], where='post')
    ax.step([-500, 500], (1 + 2 * ccg_ns_std) * np.array([1, 1]), color=tpd_color[1], ls='--', linewidth=.6)
    ax.step([-500, 500], (1 + 2 * ccg_bs_std) * np.array([1, 1]), color=tpd_color[0], ls='--', linewidth=.6)
    ax.step([-500, 500], (1 - 2 * ccg_ns_std) * np.array([1, 1]), color=tpd_color[1], ls='--', linewidth=.6)
    ax.step([-500, 500], (1 - 2 * ccg_bs_std) * np.array([1, 1]), color=tpd_color[0], ls='--', linewidth=.6)
    plt.legend({'NS', 'BS'})
    # significance test
    ccg_sig = []
    ccg_sig_ns = []
    ccg_sig_bs = []
    for i in range(len(ccg_ns_mean)-1):
        _, p = stats.mannwhitneyu(ccg_ns[:, i], ccg_bs[:,i])
        if p < .05 / len(ccg_ns_mean) / 3:
            ax.plot([edges[i], edges[i+1]], [.5, .5], 'k', linewidth=.8)
            ccg_sig.append(edges[i])
        if ccg_ns_mean[i] > 1 + 2 * ccg_ns_std:
            ax.plot([edges[i], edges[i+1]], [.6, .6], color=tpd_color[1], linewidth=.8)
            ccg_sig_ns.append(edges[i])
        if ccg_bs_mean[i] > 1 + 2 * ccg_bs_std:
            ax.plot([edges[i], edges[i+1]], [.7, .7], color=tpd_color[0], linewidth=.8)
            ccg_sig_bs.append(edges[i])
    print(ccg_sig)
    print(ccg_sig_ns)
    print(ccg_sig_bs)
    
    ax.set_xlim([-500, 500])
    ax.set_ylim([0, 5])
    ax.set_ylabel('Average normalized\nnumber of spikes')
    ax.set_xlabel('Time after MGB spikes (ms)')
    
def plot_ns_ne_nonne_ccg(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    pairs = pd.read_json(os.path.join(datafolder, 'ne-pairs-spon_ccg_5ms.json'))
    pairs = pairs[pairs.target_waveform_tpd < .45]
    ccg_ne = np.stack(pairs.ccg_5ms_ne)
    ccg_ne_mean = np.mean(ccg_ne, axis=0)
    ccg_ne_std = np.std([ccg_ne[:,:20], ccg_ne[:,-20:]])
    ccg_nonne = np.stack(pairs.ccg_5ms_nonne)
    ccg_nonne_mean = np.mean(ccg_nonne, axis=0)
    ccg_nonne_std = np.std([ccg_nonne[:,:20], ccg_nonne[:,-20:]])
    edges = np.arange(-500, 500, 10)

    ax.step(edges, ccg_ne_mean, color=tpd_color[1], where='post')
    ax.step(edges, ccg_nonne_mean, color=tpd_color[2], where='post')
    ax.step([-500, 500], (1 + 2 * ccg_ne_std) * np.array([1, 1]), color=tpd_color[1], ls='--', linewidth=.6)
    ax.step([-500, 500], (1 + 2 * ccg_nonne_std) * np.array([1, 1]), color=tpd_color[2], ls='--', linewidth=.6)
    plt.legend({'cNE spikes', 'non-cNE spikes'})
    # significance test
    ccg_sig = []
    ccg_sig_ne = []
    ccg_sig_nonne = []
    for i in range(len(ccg_ne_mean)):
        _, p = stats.wilcoxon(ccg_ne[:, i], ccg_nonne[:,i])
        if p < .01 / len(ccg_ne_mean):
            ax.plot([edges[i], edges[i+1]], [.5, .5], 'k', linewidth=.8)
            ccg_sig.append(edges[i])
        if ccg_ne_mean[i] > 1 + 2 * ccg_ne_std:
            ax.plot([edges[i], edges[i+1]], [.6, .6], color=tpd_color[1], linewidth=.8)
            ccg_sig_ne.append(edges[i])
        if ccg_nonne_mean[i] > 1 + 2 * ccg_nonne_std:
            ax.plot([edges[i], edges[i+1]], [.7, .7], color=tpd_color[2], linewidth=.8)
            ccg_sig_nonne.append(edges[i])
    print(ccg_sig)
    print(ccg_sig_ne)
    print(ccg_sig_nonne)
    
    ax.set_xlim([-500, 500])
    ax.set_ylim([0, 8])
    ax.set_ylabel('Average normalized\nnumber of spikes')
    ax.set_xlabel('Time after MGB spikes (ms)')

def plot_ns_ne_nonne_isi(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    pairs = pd.read_json(os.path.join(datafolder, 'ne-pairs-spon_ccg_10ms.json'))
    pairs = pairs[pairs.target_waveform_tpd < .45]
    isi_ne = np.concatenate(list(pairs.isi_input_ne))
    isi_nonne = np.concatenate(list(pairs.isi_input_nonne))
    isi_ne = isi_ne[isi_ne > 10]
    isi_nonne = isi_nonne[isi_nonne > 10]
    
    ax.hist(isi_nonne, bins=range(10, 501, 5), histtype='step', color=tpd_color[2], density=True)
    ax.hist(isi_ne, bins=range(10, 501, 5), histtype='step', color=tpd_color[1], density=True)
    
    ax.set_ylim([0, .014])
    ax.set_yticks(np.arange(0, .021, .002))
    ax.set_xlim([10, 500])
    ax.set_ylabel('Probability density')
    ax.set_xlabel('ISI (ms)')
    plt.legend({'non-cNE spikes', 'cNE spikes'})


def plot_ne_nonne_fr_prior(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    window = 100
    pairs = pd.read_json(os.path.join(datafolder, f'ne-pairs-spon_fr_prior_{window}.json'))
    pairs = pairs[pairs.target_waveform_tpd < .45]
    fr_target_ne = np.stack(pairs.fr_prior_target_ne)
    fr_target_ne_mean = np.mean(fr_target_ne, axis=0)
    fr_target_ne_std = np.std(fr_target_ne, axis=0)
    fr_target_nonne = np.stack(pairs.fr_prior_target_nonne)
    fr_target_nonne_mean = np.mean(fr_target_nonne, axis=0)
    fr_target_nonne_std = np.std(fr_target_nonne, axis=0)
    edges = range(0, 10)
    h1 = ax.errorbar(edges, fr_target_nonne_mean, yerr=fr_target_nonne_std, color=tpd_color[3], capsize=3)
    h2 = ax.errorbar(edges, fr_target_ne_mean, yerr=fr_target_ne_std, color=tpd_color[0], capsize=3)
    
    # significance test
    p_all = []
    for i in range(len(fr_target_ne_mean)):
        _, p = stats.mannwhitneyu(fr_target_ne[:, i], fr_target_nonne[:,i])
        p *= len(fr_target_ne_mean)
        if p < 1e-3:
            ax.text(i, .95, '***')
        elif p < 1e-2:
            ax.text(i, .95, '**')
        elif p < .05:
            ax.text(i, .95, '*')
        p_all.append(p)
    print(p_all)
    
    fr_input_ne = np.stack(pairs.fr_prior_input_ne)
    fr_input_ne_mean = np.mean(fr_input_ne, axis=0)
    fr_input_ne_std = np.std(fr_input_ne, axis=0)
    fr_input_nonne = np.stack(pairs.fr_prior_input_nonne)
    fr_input_nonne_mean = np.mean(fr_input_nonne, axis=0)
    fr_input_nonne_std = np.std(fr_input_nonne, axis=0)
    h3 = ax.errorbar(edges, fr_input_nonne_mean, yerr=fr_input_nonne_std, color=tpd_color[2], capsize=3)
    h4 = ax.errorbar(edges, fr_input_ne_mean, yerr=fr_input_ne_std, color=tpd_color[1], capsize=3)
    # significance test
    p_all = []
    for i in range(len(fr_input_ne_mean)):
        _, p = stats.mannwhitneyu(fr_input_ne[:, i], fr_input_nonne[:,i])
        p *= len(fr_target_ne_mean)
        if p < 1e-3:
            ax.text(i, .95, '***')
        elif p < 1e-2:
            ax.text(i, .95, '**')
        elif p < .05:
            ax.text(i, .95, '*')
        p_all.append(p)
    print(p_all)
    
    ax.legend(handles=[h1, h2, h3, h4], labels=['non-cNE (A1)', 'cNE (A1)', 'non-cNE (MGB)', 'cNE (MGB)'])
    ax.set_xlim([-.5, 9.5])
    ax.set_ylim([0, 1])
    ax.set_xticks(range(0,10))
    ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Firing rate prior to MGB spikes')


def plot_ns_bs_fr_prior(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    pairs = pd.read_json(os.path.join(datafolder, 'pairs.json'))
    
    fr_prior_ns = np.array(pairs[pairs.target_waveform_tpd < .45].fr_prior)
    fr_ns = np.array(pairs[pairs.target_waveform_tpd < .45].target_fr_spon)

    m = [np.mean(fr_ns), np.mean(fr_prior_ns)]
    sd = [np.std(fr_ns), np.std(fr_prior_ns)]
    for i in range(2):
        ax.bar(i, m[i], color=tpd_color[2-i])
    for i in range(len(fr_prior_ns)):
        ax.plot([0, 1], [fr_ns[i], fr_prior_ns[i]], color='grey')
    ax.errorbar(range(2), m, yerr=sd, fmt='None', color="k", capsize=2, linewidth=.6)
    _, p = stats.wilcoxon(fr_prior_ns, fr_ns)
    plot_significance_star(ax, p, [0, 1], 28, 29, linewidth=.6, fontsize=10)
   
    fr_prior_bs = np.array(pairs[pairs.target_waveform_tpd >= .45].fr_prior)
    fr_bs = np.array(pairs[pairs.target_waveform_tpd >= .45].target_fr_spon)
   
    m = [np.mean(fr_bs), np.mean(fr_prior_bs)]
    sd = [np.std(fr_bs), np.std(fr_prior_bs)]
    for i in range(2):
        ax.bar(i+3, m[i], color=tpd_color[3-i*3])
    for i in range(len(fr_prior_bs)):
        ax.plot([3, 4], [fr_bs[i], fr_prior_bs[i]], color='grey')
    ax.errorbar(range(3, 5), m, yerr=sd, fmt='None', color="k", capsize=2, linewidth=.6)
    _, p = stats.wilcoxon(fr_prior_bs, fr_bs)
    plot_significance_star(ax, p, [3, 4], 28, 29, linewidth=.6, fontsize=10)
    
    ax.set_xlim([-.5, 4.5])
    ax.set_ylim([0, 30])
    ax.set_xticks([0, 1, 3, 4])
    ax.set_yticks([0, 15, 30])
    ax.set_ylabel('Firirng rate (Hz)')
    ax.set_xticklabels(['all', 'prior to MGB spike', 'all', 'prior to MGB spike'])
    
def plot_ne_nonne_fr_prior2(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary'):
    pairs = pd.read_json(os.path.join(datafolder, 'ne-pairs-spon_fr_prior.json'))
    pairs = pairs[pairs.target_waveform_tpd < .45]
    
    fr_target_ne = np.array(pairs.fr_prior_target_ne)
    fr_target_nonne = np.array(pairs.fr_prior_target_nonne)

    m = [np.mean(fr_target_nonne), np.mean(fr_target_ne)]
    sd = [np.std(fr_target_nonne), np.std(fr_target_ne)]
    for i in range(2):
        ax.bar(i, m[i], color=tpd_color[2-i])
    for i in range(len(fr_target_ne)):
        ax.plot([0, 1], [fr_target_nonne[i], fr_target_ne[i]], color='grey')
    ax.errorbar(range(2), m, yerr=sd, fmt='None', color="k", capsize=2, linewidth=.6)
    _, p = stats.wilcoxon(fr_target_ne, fr_target_nonne)
    print(p)
    plot_significance_star(ax, p, [0, 1], 38, 39, linewidth=.6, fontsize=10)
   
    fr_input_ne = np.unique(np.array(pairs.fr_prior_input_ne))
    fr_input_nonne = np.unique(np.array(pairs.fr_prior_input_nonne))

    m = [np.mean(fr_input_nonne), np.mean(fr_input_ne)]
    sd = [np.std(fr_input_nonne), np.std(fr_input_ne)]
    for i in range(2):
        ax.bar(i+3, m[i], color=tpd_color[2-i])
    for i in range(len(fr_input_ne)):
        ax.plot([3, 4], [fr_input_nonne[i], fr_input_ne[i]], color='grey')
    ax.errorbar(range(3, 5), m, yerr=sd, fmt='None', color="k", capsize=2, linewidth=.6)
    _, p = stats.wilcoxon(fr_input_ne, fr_input_nonne)
    print(p)
    plot_significance_star(ax, p, [3, 4], 38, 39, linewidth=.6, fontsize=10)
    
    ax.set_xlim([-.5, 4.5])
    ax.set_ylim([0, 30])
    ax.set_xticks([0, 1, 3, 4])
    ax.set_yticks([0, 15, 30])
    ax.set_ylabel('Firirng rate (Hz)')
    ax.set_xticklabels(['non-cNE', 'cNE', 'non-cNE', 'cNE'])

def plot_ACG(summary_folder=r'E:\Congcong\Documents\data\connection\data-summary',
             datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
             figfolder=r'E:\Congcong\Documents\data\connection\figure'):
    
    def get_ccg(spktrain1, spktrain2, pre=-50, post=50):
        spktrain1 = np.concatenate([np.zeros(-pre), spktrain1, np.zeros(post)])
        spktrain2 = np.concatenate([np.zeros(-pre), spktrain2, np.zeros(post)])
        lags = range(pre, post+1)
        ccg = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            spktrain1_tmp = np.roll(spktrain1, lag)
            ccg[i] = int(sum(spktrain1_tmp * spktrain2))
        return ccg, np.array(lags)
        
    pairs = pd.read_json(os.path.join(summary_folder, 'pairs.json'))
    exp_loaded = None
    binsize=10
    # plot all MGB neurons
    pairs_tmp = pairs.groupby(['exp', 'input_idx']).size().reset_index()
    fig, axes = plt.subplots(7, 7, figsize=[15, 10])
    axes = axes.flatten()
    for i in range(len(pairs_tmp)):
        print(i)
        exp = pairs_tmp.iloc[i].exp
        if exp != exp_loaded:
            _, input_units, target_units, trigger = load_input_target_files(datafolder, exp)
        input_idx = pairs_tmp.iloc[i].input_idx
        input_unit = input_units[input_idx]
        spiketimes = input_unit.spiketimes_spon
        spktrain, _ = np.histogram(spiketimes, bins=np.arange(0, spiketimes[-1]+binsize, binsize))
        ccg, lags = get_ccg(spktrain, spktrain)
        ccg[50] = 0
        axes[i].bar(lags * 10, ccg, width=10, color='k')
        axes[i].set_title(f'{exp} unit{input_idx}')
        axes[i].set_xlim([-500, 500])
    axes[42].set_xlabel('Lag(ms)')
    axes[42].set_ylabel('# of spikes') 
    fig.suptitle('MGB neurons ACG', fontsize=20)     
    fig.tight_layout()
    fig.savefig(os.path.join(figfolder, 'MGB_ACG.pdf'))
    plt.close()
    
    exp_loaded = None
    # plot all A1 BS neurons
    pairs_tmp = pairs[pairs.target_waveform_tpd >= .45].groupby(['exp', 'target_idx']).size().reset_index()
    fig, axes = plt.subplots(4, 4, figsize=[9, 6])
    axes = axes.flatten()
    for i in range(len(pairs_tmp)):
        print(i)
        exp = pairs_tmp.iloc[i].exp
        if exp != exp_loaded:
            _, _, target_units, trigger = load_input_target_files(datafolder, exp)
        target_idx = pairs_tmp.iloc[i].target_idx
        target_unit = target_units[target_idx]
        spiketimes = target_unit.spiketimes_spon
        spktrain, _ = np.histogram(spiketimes, bins=np.arange(0, spiketimes[-1]+binsize, binsize))
        ccg, lags = get_ccg(spktrain, spktrain)
        ccg[50] = 0
        axes[i].bar(lags * 10, ccg, width=10, color='k')
        axes[i].set_title(f'{exp} unit{target_idx}')
        axes[i].set_xlim([-500, 500])
    axes[12].set_xlabel('Lag(ms)')
    axes[12].set_ylabel('# of spikes') 
    fig.suptitle('A1 neurons ACG (BS)', fontsize=20)        
    fig.tight_layout()
    fig.savefig(os.path.join(figfolder, 'A1_ACG_BS.pdf'))
    plt.close()
    
    exp_loaded = None
    # plotall A1 NS neurons
    pairs_tmp = pairs[pairs.target_waveform_tpd < .45].groupby(['exp', 'target_idx']).size().reset_index()
    fig, axes = plt.subplots(4, 4, figsize=[9, 6])
    axes = axes.flatten()
    for i in range(len(pairs_tmp)):
        print(i)
        exp = pairs_tmp.iloc[i].exp
        if exp != exp_loaded:
            _, _, target_units, trigger = load_input_target_files(datafolder, exp)
        target_idx = pairs_tmp.iloc[i].target_idx
        target_unit = target_units[target_idx]
        spiketimes = target_unit.spiketimes_spon
        spktrain, _ = np.histogram(spiketimes, bins=np.arange(0, spiketimes[-1]+binsize, binsize))
        ccg, lags = get_ccg(spktrain, spktrain)
        ccg[50] = 0
        axes[i].bar(lags * 10, ccg, width=10, color='k')
        axes[i].set_title(f'{exp} unit{target_idx}')
        axes[i].set_xlim([-500, 500])
    axes[12].set_xlabel('Lag(ms)')
    axes[12].set_ylabel('# of spikes') 
    fig.suptitle('A1 neurons ACG (NS)', fontsize=20)        
    fig.tight_layout()
    fig.savefig(os.path.join(figfolder, 'A1_ACG_NS.pdf'))
    plt.close()
    

def plot_CCG_NS_BS(summary_folder=r'E:\Congcong\Documents\data\connection\data-summary',
             datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
             figfolder=r'E:\Congcong\Documents\data\connection\figure'):
        
    pairs = pd.read_json(os.path.join(summary_folder, 'pairs.json'))
    exp_loaded = None
    binsize=10

    # plot all A1 BS neurons
    for j in range(2):
        if j == 0:
            pairs_tmp = pairs[pairs.target_waveform_tpd >= .45]
            fig, axes = plt.subplots(6, 6, figsize=[12, 8])
        else:
            pairs_tmp = pairs[pairs.target_waveform_tpd < .45]
            fig, axes = plt.subplots(6, 7, figsize=[12, 8])
        axes = axes.flatten()
        for i in range(len(pairs_tmp)):
            print(i)
            exp = pairs_tmp.iloc[i].exp
            if exp != exp_loaded:
                _, input_units, target_units, trigger = load_input_target_files(datafolder, exp)
            input_idx = pairs_tmp.iloc[i].input_idx
            target_idx = pairs_tmp.iloc[i].target_idx
            target_unit = target_units[pairs_tmp.iloc[i].target_idx]
            input_unit = input_units[pairs_tmp.iloc[i].input_idx]
            spiketimes_target = target_unit.spiketimes_spon
            spiketimes_target = np.array(sorted(spiketimes_target))
            spiketimes_input = input_unit.spiketimes_spon
            spiketimes_input = np.array(sorted(spiketimes_input))
            ccg, edges, _ = ct.get_ccg(spiketimes_input, spiketimes_target, window_size=500, binsize=10)
            centers = (edges[1:] + edges[:-1]) / 2
            axes[i].bar(centers, ccg, width=10, color='k')
            axes[i].set_xlim([-500, 500])
            axes[i].set_title(f'{exp} unit{input_idx}->unit{target_idx}')
        if j == 0:
            idx = 30
            cell_type = 'BS'
        else:
            idx = 35
            cell_type = 'NS'
        axes[idx].set_xlabel('Lag (ms)')
        axes[idx].set_ylabel('# of spikes') 
        fig.suptitle(f'A1 neurons ACG ({cell_type})', fontsize=20)        
        fig.tight_layout()
        fig.savefig(os.path.join(figfolder, f'CCG_{cell_type}.pdf'))
    

def plot_CCG_ne_nonne(summary_folder=r'E:\Congcong\Documents\data\connection\data-summary',
             datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
             figfolder=r'E:\Congcong\Documents\data\connection\figure'):
        
    pairs = pd.read_json(os.path.join(summary_folder, 'ne-pairs-spon.json'))
    pairs = pairs[pairs.inclusion_spon]
    pairs = pairs[pairs.target_waveform_tpd < .45]
    exp_loaded = None
    binsize=10
    
    for j in range(2):
        fig, axes = plt.subplots(6, 6, figsize=[12, 8])
        axes = axes.flatten()
        for i in range(len(pairs)):
            print(i)
            exp = pairs.iloc[i].exp
            cne = pairs.iloc[i].cne
            if exp != exp_loaded:
                _, input_units, target_units, trigger = load_input_target_files(datafolder, exp)
                exp = str(exp)
                exp = exp[:6] + '_' + exp[6:] 
                nefile = glob.glob(os.path.join(datafolder, f'{exp}*-ne-20dft-spon.pkl'))[0]
                with open(nefile, 'rb') as f:
                    ne = pickle.load(f)
            exp = pairs.iloc[i].exp
            
            input_idx = pairs.iloc[i].input_idx
            target_idx = pairs.iloc[i].target_idx
            target_unit = target_units[target_idx]
            input_unit = input_units[input_idx]
            spiketimes_target = target_unit.spiketimes_spon
            spiketimes_target = np.array(sorted(spiketimes_target))
            spiketimes_input = input_unit.spiketimes_spon
            spiketimes_input = np.array(sorted(spiketimes_input))
    
            member_idx = np.where(ne.ne_members[cne] == input_idx)[0][0]
            ne_unit = ne.member_ne_spikes[cne][member_idx]
            ne_spiketimes = ne_unit.spiketimes
            if j == 0:
                spiketimes_input = ne_spiketimes
            else:
                spiketimes_input = np.array(list(set(spiketimes_input).difference(set(ne_spiketimes))))
            ccg, edges, _ = ct.get_ccg(spiketimes_input, spiketimes_target, window_size=500, binsize=10)
            centers = (edges[1:] + edges[:-1]) / 2
            axes[i].bar(centers, ccg, width=10, color='k')
            axes[i].set_xlim([-500, 500])
            axes[i].set_title(f'{exp} unit{input_idx}->unit{target_idx} (cNE{cne+1})')
        if j == 0:
            cell_type = 'cNE'
        else:
            cell_type = 'non-cNE'
        idx = 30
        axes[idx].set_xlabel('Lag (ms)')
        axes[idx].set_ylabel('# of spikes') 
        fig.suptitle(f'A1 neurons ACG ({cell_type})', fontsize=20)        
        fig.tight_layout()
        fig.savefig(os.path.join(figfolder, f'CCG_{cell_type}.pdf'))


def figure8(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure_v2'):
    
    fig = plt.figure(figsize=[8.5*cm, 4*cm])
    # summary plot
    ax = fig.add_axes([.1, .2, .3, .5])
    file = r"ne-pairs-200df-spon.json"
    plot_efficacy_ne_vs_nonne(ax, stim='spon', file=file, celltype=True)
    ax.set_xlim([0, 20])
    ax.set_xticks(range(0, 21, 5))
    ax.set_ylim([0, 15])
    ax.set_yticks(range(0, 16, 5))
    # violin plot
    ax = fig.add_axes([.57, .2, .45, .7])
    plot_efficacy_change_vs_binsize(ax)
    fig.savefig(os.path.join(figfolder, 'fig8c.pdf'), dpi=300)
    plt.close()
    
    fig = plt.figure(figsize=[8.5*cm, 8.5*cm])
    # PART1: plot example cNE and A1 connectin
    # load nepiars
    example_file = os.path.join(datafolder, '200821_015617-site6-5655um-25db-dmr-31min-H31x64-fs20000-pairs-ne-200-spon.json')
    nepairs = pd.read_json(example_file)
    exp = re.search('\d{6}_\d{6}', example_file).group(0)
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
    nefile = re.sub('-pairs-ne-200-spon.json', '-ne-200dft-spon.pkl', example_file)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    patterns = ne.patterns
    cne = 1
    target_unit = 21
    activity_100 = ne.ne_activity[cne]
    member_ne_spikes_100 = ne.member_ne_spikes[cne]
    # plot example cen
    ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_unit == target_unit)].copy()
    # 1. probe
    # 1.1 MGB
    x_start = .12
    x_fig = .04
    y_start = .02
    y_fig = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs, input_units)
    ax.set_ylim([5800, 4600])
    ax.set_ylabel(r'Depth ($\mu$m)', labelpad=0)
    # add axes for waveform plot
    x_start = .18
    y_start = .2
    x_fig = .06
    y_fig = .05
    y_space = .01
    n_pairs = len(ne_neuron_pairs)
    axes = add_multiple_axes(fig, n_pairs, 1, x_start, y_start, x_fig, y_fig, 0, y_space)
    position_idx_ne = [position_idx[unit_idx] + 1 for unit_idx in ne_neuron_pairs.input_idx]
    ne_neuron_pairs['position_idx'] = position_idx_ne
    ne_neuron_pairs = ne_neuron_pairs.sort_values(by='position_idx')
    plot_all_waveforms(axes, ne_neuron_pairs, input_units, position_idx=position_idx)
    for ax in axes:
        ax[0].set_title('')
    ne_neuron_pairs = ne_neuron_pairs.iloc[2:]
    n_pairs = len(ne_neuron_pairs)
    
    # 2. icweight
    x_start = .1
    x_fig = .3
    y_start = .8
    y_fig = .12
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    weights = patterns[cne]
    weights = weights[position_order]
    position_idx = np.array(position_idx)
    members_100 = sorted(position_idx[ne.ne_members[cne]] + 1)
    member_thresh = 1 / np.sqrt(patterns.shape[1])
    plot_ICweight(ax, weights, member_thresh, direction='h', markersize=2)
    ax.set_ylim([-.2, .8])
    ax.set_yticks([0, .4, .8])
    
    # add axes for ccg plot
    x_start = .55
    y_start = .08
    x_fig = .2
    x_space = .03
    y_fig = .14
    y_space =  0.03
    axes = add_multiple_axes(fig, n_pairs, 2, x_start, y_start, x_fig, y_fig, x_space, y_space)
    plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs, stim='spon')
    for ax in axes.flatten():
        ax.set_ylim([0, 150])
        ax.set_yticks(range(0, 151, 50))
        ax.set_yticklabels([])
    for ax in axes[:, 0]:
        ax.set_yticklabels(range(0, 151, 50))
    fig.savefig(os.path.join(figfolder, 'fig8a.pdf'), dpi=300)
    plt.close()
    
    # plot activity and raster
    fig = plt.figure(figsize=[9*cm, 6*cm])
    # get 10ms cNE
    nefile = re.sub('-pairs-ne-200-spon.json', '-ne-10dft-spon.pkl', example_file)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    cne = 0
    activity_10 = ne.ne_activity[cne]
    members_10 = sorted(position_idx[ne.ne_members[cne]] + 1)
    member_ne_spikes_10 = ne.member_ne_spikes[cne]    
    # plot activity of 10s and 100ms cNEs
    xstart = .1
    xfig = .85
    ystart = .05
    yfig = .1
    ax = fig.add_axes([xstart, ystart, xfig, yfig])
    [t_start, t_end] = [511.5, 516.5]
    # plot 100ms activity
    activity_100 = activity_100 / max(activity_100[int(t_start * 10): int(t_end * 10)])
    centers = np.array(range(0, len(activity_100))) * 100 + 50
    centers = centers / 1000
    ylim = [-.1, 1]
    plot_activity(ax, centers, activity_100, None, [t_start, t_end], ylim, 'darkturquoise')
    # plot 10ms activity
    activity_10 = activity_10 / max(activity_10[int(t_start * 200): int(t_end * 200)])
    centers = np.array(range(0, len(activity_10))) * 5 + 2.5
    centers = centers / 1000
    plot_activity(ax, centers, activity_10, None, [t_start, t_end], ylim, 'blue')
    ax.set_ylabel('Normalized\nactivity', fontsize=6)
    ax.legend({'100ms', '5ms'})
    ax.set_yticks([0, .5, 1])
    # plot raster of A1 and MGB neurons
    ystart = 0.24
    yfig = .75
    ax = fig.add_axes([xstart, ystart, xfig, yfig])
    common_members = set(members_10).intersection(set(members_100))
    for member in common_members:
        p = mpl.patches.Rectangle((t_start, member -.4),
                                  t_end - t_start, 0.8, color='orange')
        ax.add_patch(p)
    members_10_only = set(members_10) - common_members
    for member in members_10_only:
         p = mpl.patches.Rectangle((t_start, member -.4),
                                   t_end - t_start, 0.8, color='cornflowerblue')
         ax.add_patch(p)
    members_100_only = set(members_100) - common_members
    for member in members_100_only:
         p = mpl.patches.Rectangle((t_start, member -.4),
                                   t_end - t_start, 0.8, color='paleturquoise')
         ax.add_patch(p)
    # A1
    p = mpl.patches.Rectangle((t_start, len(input_units) + 1 -.4),
                              t_end - t_start, 0.8, color='grey')
    ax.add_patch(p)
    plot_raster(ax, [target_units[32]], linewidth=.6, new_order=[len(input_units)])
    spktimes_target = target_units[32].spiketimes
    spktimes_target  = spktimes_target[(spktimes_target > t_start) & (spktimes_target < t_end)]
    spktimes_input = np.concatenate([input_units[6].spiketimes, input_units[-2].spiketimes]) / 1e3
    spktimes_input  = spktimes_input[(spktimes_input > t_start) & (spktimes_input < t_end)]
    spktimes_induced = [spiketime for spiketime in spktimes_target 
                          if any((1/1e3 <= spiketime - spktimes_input) & (5/1e3 >= spiketime - spktimes_input))]
    target_units[32].spiketimes = spktimes_induced
    plot_raster(ax, [target_units[32]], linewidth=.6, new_order=[len(input_units)], color='r')
    # MGB
    plot_raster(ax, input_units, linewidth=.6, new_order=position_idx)
    plot_raster(ax, member_ne_spikes_100, offset='unit', color='lightcoral', linewidth=.6, new_order=position_idx)
    plot_raster(ax, member_ne_spikes_10, offset='unit', color='r', linewidth=.6, new_order=position_idx)
    ax.set_xlim([t_start, t_end])
    ax.set_ylim([0, 19])
    ax.set_yticks([5, 10, 15])
    plt.gca().invert_yaxis()
    plt.plot([512.8, 513.3], [0, 0])
    plt.plot([t_end-1, t_end], [0, 0])
    ax.spines[['bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    fig.savefig(os.path.join(figfolder, 'fig8b.pdf'), dpi=300)
    


        
def plot_efficacy_change_vs_binsize(ax, file_id='spon-0.5ms_bin',
    datafolder=r"E:\Congcong\Documents\data\connection\data-summary"):
    dfs = [4, 10, 20, 40, 80, 200, 500, 1000]
    data_all = []
    for df in dfs:
        file = os.path.join(datafolder, f'ne-pairs-{df}df-{file_id}.json')
        data = pd.read_json(file)
        data = data[(data.inclusion_spon) & (data.efficacy_ne_spon > 0) & (data.efficacy_nonne_spon > 0)]
        data['df'] = df
        data_all.append(data)
    data = pd.concat(data_all)
    data['efficacy_change'] = data['efficacy_ne_spon'] - data['efficacy_nonne_spon']
    
    positions = np.log2(dfs)

    data_ns = data[data.target_waveform_tpd < .45].groupby('df')['efficacy_change'].apply(list)
    v1 = ax.violinplot(data_ns, points=100, positions=positions, showextrema=False, widths=.8)
    set_violin_half(v1, half='l', color=tpd_color[1])
    data_bs = data[data.target_waveform_tpd >= .45].groupby('df')['efficacy_change'].apply(list)
    v2 = ax.violinplot(data_bs, points=100, positions=positions, showextrema=False, widths=.8)
    set_violin_half(v2, half='r', color=tpd_color[0])
    ax.set_xticks(positions)
    binsize = (np.array(dfs)/2).astype(int)
    ax.plot([0, 12], [0, 0], 'k--')
    ax.set_xlim([1, 11])
    ax.set_xticklabels(binsize)
    ax.set_xlabel('Bin size (ms)')
    ax.set_ylabel('\u0394Efficacy (%)')
    ax.set_ylim([-20, 30])
    
    for i, df in enumerate(dfs):
        print(df/2)
        _, p = stats.wilcoxon(data_ns[df])
        print(p * 16)
        _, p = stats.wilcoxon(data_bs[df])
        print(p * 16)


def figure9(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure_v3'):
    
    fig = plt.figure(figsize=[8.5*cm, 8.5*cm])
    # PART1: plot example cNE and A1 connectin
    # load nepiars
    #example_file = os.path.join(datafolder, '200820_230604-site4-5655um-25db-dmr-31min-H31x64-fs20000-pairs-ne-10-spon-0.5ms_bin.json')
    #nefile = re.sub('-pairs-ne-10-spon-0.5ms_bin.json', '-ne-10dft-spon.pkl', example_file)
    # cne = 2
    example_file = os.path.join(datafolder, '200820_230604-site4-5655um-25db-dmr-31min-H31x64-fs20000-pairs-ne-20-spon-0.5ms_bin.json')
    nefile = re.sub('-pairs-ne-20-spon-0.5ms_bin.json', '-ne-20dft-spon.pkl', example_file)
    # cne = 1
    # target_unit = 43
    cne = 2
    target_unit = 33
    # plot example cen
    nepairs = pd.read_json(example_file)
    exp = re.search('\d{6}_\d{6}', example_file).group(0)
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    patterns = ne.patterns
    
    ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_unit == target_unit)].copy()
    # 1. probe
    # 1.1 MGB
    x_start = .12
    x_fig = .04
    y_start = .02
    y_fig = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs, input_units)
    ax.set_ylim([5800, 4600])
    ax.set_ylabel(r'Depth ($\mu$m)', labelpad=0)
    # add axes for waveform plot
    x_start = .18
    y_start = .2
    x_fig = .06
    y_fig = .05
    y_space = .01
    n_pairs = len(ne_neuron_pairs)
    axes = add_multiple_axes(fig, n_pairs, 1, x_start, y_start, x_fig, y_fig, 0, y_space)
    position_idx_ne = [position_idx[unit_idx] + 1 for unit_idx in ne_neuron_pairs.input_idx]
    ne_neuron_pairs['position_idx'] = position_idx_ne
    ne_neuron_pairs = ne_neuron_pairs.sort_values(by='position_idx')
    plot_all_waveforms(axes, ne_neuron_pairs, input_units, position_idx=position_idx)
    for ax in axes:
        ax[0].set_title('')
    n_pairs = len(ne_neuron_pairs)
    # 1.2 A1
    x_start = .24
    x_fig = .04
    y_start = .02
    y_fig = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    _ = plot_position_on_probe(ax, ne_neuron_pairs, target_units, location='A1')
    ax.set_ylim([1000, 200])
    # add axes for waveform plot
    x_start = .22
    y_start = .2
    x_fig = .06
    y_fig = .05
    ax = fig.add_axes([x_start, y_start,x_fig, y_fig])
    plot_all_waveforms(np.array([ax]), ne_neuron_pairs, target_units, 'target')
    ax.set_title('')
    
    # 2. icweight
    x_start = .1
    x_fig = .3
    y_start = .8
    y_fig = .12
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    weights = patterns[cne]
    weights = weights[position_order]
    position_idx = np.array(position_idx)
    member_thresh = 1 / np.sqrt(patterns.shape[1])
    plot_ICweight(ax, weights, member_thresh, direction='h', markersize=2)
    ax.set_ylim([-.2, .8])
    ax.set_yticks([0, .4, .8])
    
    # add axes for ccg plot
    x_start = .55
    y_start = .08
    x_fig = .16
    x_space = .03
    y_fig = .12
    y_space =  0.03
    axes = add_multiple_axes(fig, n_pairs, 2, x_start, y_start, x_fig, y_fig, x_space, y_space)
    plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs, stim='spon')
    for ax in axes.flatten():
        ax.set_ylim([0, 150])
        ax.set_yticks(range(0, 151, 50))
        ax.set_yticklabels([])
    for ax in axes[:, 0]:
        ax.set_yticklabels(range(0, 151, 50))
    fig.savefig(os.path.join(figfolder, 'fig9a.pdf'), dpi=300)
    plt.close()
  
    
def xcorr_normalize_imshow(ax, xcorr):
    # normalize by maximum value of each row
    row_max = xcorr.max(axis=1)
    xcorr_norm = xcorr / row_max[:, np.newaxis]
    latency = np.argmax(xcorr_norm, axis=1)
    latency_order = np.argsort(latency)
    xcorr_norm = xcorr_norm[latency_order, :]
    im = ax.imshow(xcorr_norm, aspect='auto',  cmap='viridis', vmax=1, vmin=0)
    ax.set_xticks(np.arange(-.5, 40, 10))
    ax.set_xticklabels(range(-10, 11, 5))
    ax.set_xlabel('Time from MGB spike (ms)')
    ax.set_ylabel('MGB-A1 neuronal pair #')
    y = ax.get_ylim()
    ax.plot([19.5, 19.5], y, 'w-')
    plt.colorbar(im)
    


def plot_NS_BS_xcorr():
    fig = plt.figure(figsize=[8*cm, 6*cm])
    datafolder = r'E:\Congcong\Documents\data\connection\data-summary'
    figfolder = r'E:\Congcong\Documents\data\connection\paper\figure_v3'
    pairs = pd.read_json(os.path.join(datafolder, 'pairs.json'))
    xstart = .1
    xfig = .4
    yfig = .35
    ax = fig.add_axes([xstart, .55, xfig, yfig])
    pairs_ns = pairs[pairs.target_waveform_tpd < .45]
    xcorr_ns = np.stack(pairs_ns.ccg_spon)[:, 80:120]
    xcorr_normalize_imshow(ax, xcorr_ns)
    ax.set_xlim([20-.5, 40-.5])
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax = fig.add_axes([xstart, .1, xfig, yfig])
    pairs_bs = pairs[pairs.target_waveform_tpd >= .45]
    xcorr_bs = np.stack(pairs_bs.ccg_spon)[:, 80:120]
    xcorr_normalize_imshow(ax, xcorr_bs)
    ax.set_xlim([20-.5, 40-.5])
    
    
    xstart = .64
    xfig=.35
    ax = fig.add_axes([xstart, .55, xfig, yfig])
    pairs['latency'] = pairs['ccg_spon'].apply(np.argmax)
    pairs['latency']  = (pairs['latency'] - 100) / 2
    pairs['is_ns'] = pairs['target_waveform_tpd'] < .45
    
    boxplot_scatter(ax, x='is_ns', y='latency', data=pairs, size=3, jitter=.3,
                    order=[True, False], hue='is_ns', 
                    palette=tpd_color[1::-1], hue_order=[True, False])
    _, p = stats.mannwhitneyu(pairs[pairs['target_waveform_tpd'] < .45].latency,
                              pairs[pairs['target_waveform_tpd'] >= .45].latency)
    plot_significance_star(ax, p, [0, 1], 4.85, 4.9)
    ax.set_ylim([0, 5])
    ax.set_yticks([0, 2.5, 5])
    ax.set_ylabel('Peak latency (ms)')
    
    ax = fig.add_axes([xstart, .1, xfig, yfig])
    xcorr_ns = np.stack(pairs[pairs.target_waveform_tpd < .45].ccg_spon)
    peak_shift = list(map(int, pairs[pairs.target_waveform_tpd < .45].latency.values * 2))
    for i in range(len(peak_shift)):
        xcorr_ns[i] = np.roll(xcorr_ns[i], -peak_shift[i])
    row_max = xcorr_ns.max(axis=1)
    xcorr_ns = xcorr_ns[:,80:121] / row_max[:, np.newaxis]
    baseline = np.mean(xcorr_ns[:, :5] + xcorr_ns[:, -5:], axis=1) / 2
    hh_ns = baseline + (1 - baseline) / 2
    hw_ns = []
    for i in range(len(hh_ns)):
        hw_ns.append(len(np.where(xcorr_ns[i] >= hh_ns[i])[0]))
    hw_ns = np.array(hw_ns) / 2
    corr_avg = xcorr_ns.mean(axis=0)
    corr_std = xcorr_ns.std(axis=0)
    # shade for SD
    ax.fill_between(np.arange(-10, 10.1, .5), corr_avg - corr_std, corr_avg + corr_std,
                    alpha=0.5, edgecolor=None, facecolor=tpd_color[1])
    ax.plot(np.arange(-10, 10.1, .5), corr_avg, color=tpd_color[1])
    xcorr_bs = np.stack(pairs[pairs.target_waveform_tpd >= .45].ccg_spon)
    peak_shift = list(map(int, pairs[pairs.target_waveform_tpd >= .45].latency.values * 2))
    for i in range(len(peak_shift)):
        xcorr_bs[i] = np.roll(xcorr_bs[i], -peak_shift[i])
    row_max = xcorr_bs.max(axis=1)
    xcorr_bs = xcorr_bs[:,80:121] / row_max[:, np.newaxis]
    corr_avg = xcorr_bs.mean(axis=0)
    corr_std = xcorr_bs.std(axis=0)
    baseline = np.mean(xcorr_bs[:, :5] + xcorr_bs[:, -5:], axis=1) / 2
    hh_bs = baseline + (1 - baseline) / 2
    hw_bs = []
    for i in range(len(hh_bs)):
        hw_bs.append(len(np.where(xcorr_bs[i] >= hh_bs[i])[0]))
    hw_bs = np.array(hw_bs) / 2
    # shade for SD
    ax.fill_between(np.arange(-10, 10.1, .5), corr_avg - corr_std, corr_avg + corr_std,
                    alpha=0.5, edgecolor=None, facecolor=tpd_color[0])
    ax.plot(np.arange(-10, 10.1, .5), corr_avg, color=tpd_color[0])
    for i in range(41):
        _, p = stats.mannwhitneyu(xcorr_bs[:, i], xcorr_ns[:, i])
        if p * 40 < .05:
            print(i, p)
    ax.set_xlabel('Time from peak (ms)')
    ax.set_ylabel('Normalized FR')
    ax.set_yticks([0, .5, 1])
    ax.set_xticks(range(-10, 11, 5))
    ax.set_xlim([-10, 10])
    ax.set_ylim([0, 1])
    
    
    ax = fig.add_axes([.8, .2, .1, .1])
    boxplot_scatter(ax, x='waveform_ns', y='efficacy_gain', data=pairs, size=3, jitter=.3,
                    order=[True, False], hue='waveform_ns', 
                    palette=tpd_color[1::-1], hue_order=[True, False])
    ax.set_xticklabels(['NS', 'BS'])
    ax.set_xlabel('A1 neuron type')
    ax.set_ylabel('Efficacy gain (%)')
    fig.savefig(os.path.join(figfolder, 'fig5.pdf'), dpi=300)



def plot_causal_spikes_raster(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure_v3\raster',
            t_start=0, t_end=10):
  
    fig = plt.figure(figsize=[27*cm, 18*cm])
    # PART1: plot example cNE and A1 connectin
    # load nepiars
    example_file = os.path.join(datafolder, '200821_015617-site6-5655um-25db-dmr-31min-H31x64-fs20000-pairs-ne-200-spon.json')
    nepairs = pd.read_json(example_file)
    exp = re.search('\d{6}_\d{6}', example_file).group(0)
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
    nefile = re.sub('-pairs-ne-200-spon.json', '-ne-200dft-spon.pkl', example_file)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    cne = 1
    target_unit = 21
    activity_100 = ne.ne_activity[cne]
    member_ne_spikes_100 = ne.member_ne_spikes[cne]
    # get position index
    ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_unit == target_unit)].copy()
    ax = fig.add_axes([.1, .1, .5, .5])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs, input_units)
    position_idx = np.array(position_idx)
    ax.remove()
    members_100 = sorted(position_idx[ne.ne_members[cne]] + 1)
    
    # plot activity and raster
    # get 10ms cNE
    nefile = re.sub('-pairs-ne-200-spon.json', '-ne-10dft-spon.pkl', example_file)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    cne = 0
    activity_5 = ne.ne_activity[cne]
    members_5 = sorted(position_idx[ne.ne_members[cne]] + 1)
    member_ne_spikes_5 = ne.member_ne_spikes[cne]
        
    # plot activity of 10s and 100ms cNEs
    xstart = .1
    xfig = .8
    ystart = .7
    yfig = .2
    ax = fig.add_axes([xstart, ystart, xfig, yfig])
    # plot 100ms activity
    activity_100 = activity_100 / max(activity_100)
    centers = np.array(range(0, len(activity_100))) * 100 + 50
    centers = centers / 1000
    ylim = [-.1, .4]
    plot_activity(ax, centers, activity_100, None, [t_start, t_end], ylim, 'darkturquoise')
    # plot 10ms activity
    activity_5 = activity_5 / max(activity_5)
    centers = np.array(range(0, len(activity_5))) * 5 + 2.5
    centers = centers / 1000
    plot_activity(ax, centers, activity_5, None, [t_start, t_end], ylim, 'blue')
    ax.set_ylabel('Normalized\nactivity', fontsize=6)
    ax.legend({'100ms', '5ms'})
    ax.set_yticks([0, .2, .4])
    ax.set_xlim([t_start, t_end])
    # plot raster of A1 and MGB neurons
    ystart = 0.2
    yfig = .45
    ax = fig.add_axes([xstart, ystart, xfig, yfig])
    common_members = set(members_5).intersection(set(members_100))
    for member in common_members:
        p = mpl.patches.Rectangle((t_start, member -.4),
                                  t_end - t_start, 0.8, color='orange')
        ax.add_patch(p)
    members_5_only = set(members_5) - common_members
    for member in members_5_only:
         p = mpl.patches.Rectangle((t_start, member -.4),
                                   t_end - t_start, 0.8, color='cornflowerblue')
         ax.add_patch(p)
    members_100_only = set(members_100) - common_members
    for member in members_100_only:
         p = mpl.patches.Rectangle((t_start, member -.4),
                                   t_end - t_start, 0.8, color='paleturquoise')
         ax.add_patch(p)
    # A1
    p = mpl.patches.Rectangle((t_start, 0.6 - 1),
                              t_end - t_start, 0.8, color='grey')
    ax.add_patch(p)
    plot_raster(ax, [target_units[32]], linewidth=.6, new_order=[-1])
    spktimes_target = target_units[32].spiketimes
    spktimes_target  = spktimes_target[(spktimes_target > t_start) & (spktimes_target < t_end)]
    spktimes_input = np.concatenate([input_units[6].spiketimes, input_units[-2].spiketimes]) / 1e3
    spktimes_input  = spktimes_input[(spktimes_input > t_start) & (spktimes_input < t_end)]
    spktimes_induced = [spiketime for spiketime in spktimes_target 
                          if any((1/1e3 <= spiketime - spktimes_input) & (5/1e3 >= spiketime - spktimes_input))]
    target_units[32].spiketimes = spktimes_induced
    plot_raster(ax, [target_units[32]], linewidth=.6, new_order=[-1], color='r') 
    # MGB
    plot_raster(ax, input_units, linewidth=.6, new_order=position_idx)
    plot_raster(ax, member_ne_spikes_5, offset='unit', color='r', linewidth=.6, new_order=position_idx)
    plot_raster(ax, member_ne_spikes_100, offset='unit', color='lightcoral', linewidth=.6, new_order=position_idx)
    ax.set_xlim([t_start, t_end])
    ax.set_ylim([-1, 18])
    plt.gca().invert_yaxis()
    fig.savefig(os.path.join(figfolder, f'raster-{t_start}.jpg'), dpi=300)
    
    ax.spines[['bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    plt.close()