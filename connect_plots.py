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
from plot_box import plot_strf, boxplot_scatter, plot_significance_star, plot_ICweight, set_violin_half
from connect_toolbox import load_input_target_files
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import connect_toolbox as ct
import pickle as pkl


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


def plot_ccg(ax, ccg, baseline, thresh=None, taxis=None, nspk=None, causal_method='peak', causal=True, xlim=[-50, 50]):
    if taxis is None:
        edges = np.arange(-50, 50.5, .5)
        taxis = (edges[1:] + edges[:-1]) / 2
    if nspk: # normalize to firing rate
        ccg = ccg / (nspk*.5 / 1e3)
    
    ax.bar(taxis, ccg, width=.5, color='k')
    
    # plot causal spikes
    if causal:
        if causal_method == 'peak':
            causal_idx = ct.get_causal_spike_idx(ccg, method=causal_method)
            causal_baseline = ct.get_causal_spk_baseline(ccg, causal_idx)
    
        try:
            ax.bar(taxis[causal_idx], ccg[causal_idx]-causal_baseline, bottom=causal_baseline,
                  width=.5, color='r')
        except (NameError, TypeError):
            pass
        
    # plot baseline and threshold
    try:
        ax.plot(taxis, baseline, 'b')
        ax.plot(taxis, thresh, 'b--')
    except ValueError:
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
                                        stim='spon'):
    files = glob.glob(os.path.join(datafolder, f'*-pairs-ne-{stim}.json'))
    for file in files:
        nepairs = pd.read_json(file)
        exp = re.search('\d{6}_\d{6}', file).group(0)
        cne_target = nepairs[['cne', 'target_idx']].drop_duplicates()
        _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
        nefile = re.sub(f'-pairs-ne-{stim}.json', '-ne-20dft-spon.pkl', file)
        with open(nefile, 'rb') as f:
            ne = pkl.load(f)
        patterns = ne.patterns
        for cne, target_idx in cne_target.values:
            fig, ne_neuron_pairs = plot_ne_neuron_connection_ccg(
                nepairs, cne, target_idx, input_units, target_units, patterns, stim=stim)

            # save file
            target_unit = ne_neuron_pairs.iloc[0]['target_unit']
            fig.savefig(os.path.join(figfolder, f'ne_ccg_{stim}', f'{exp}-cne_{cne}-target_{target_unit}.jpg'), dpi=300)
            plt.close()


def plot_ne_neuron_connection_ccg(nepairs, cne, target_idx, input_units, target_units, patterns, stim='spon'):
    ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_idx == target_idx)]
    n_pairs = len(ne_neuron_pairs)
    assert(n_pairs > 1)
    
    bottom_space = .8
    fig = plt.figure(figsize=[8.8*cm, 1.8*n_pairs*cm + bottom_space*cm])
    # probe
    x_start = .08
    y_start = (1.8*(n_pairs-2) + bottom_space + .5)/ (1.8 * n_pairs + bottom_space)
    x_fig = .05
    y_fig = 3 / (1.8 * n_pairs + bottom_space)
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    position_idx, position_order = plot_position_on_probe(ax, ne_neuron_pairs, input_units)
    
    # icweight
    y_start = 1/ (1.8 * n_pairs + bottom_space)
    y_fig = 2 / (1.8 * n_pairs + bottom_space)
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

def plot_position_on_probe(ax, pairs, units):
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
    for i in range(len(pairs)):
        pair = pairs.iloc[i]
        unit = units[pair.input_idx]
        ax.scatter(unit.position[0], unit.position[1], s=4, color='r')
        ax.text(unit.position[0], unit.position[1], f'{unit.position_idx+1}', fontsize=6)
    ax.get_xaxis().set_visible(False)
    ax.spines[['bottom']].set_visible(False)
    ax.invert_yaxis()
    position_order = np.argsort(position_idx)
    return position_idx, position_order



def plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs, stim='spon'):
    n_pairs = len(ne_neuron_pairs)
    for i in range(n_pairs):
        ax = axes[i]
        pair = ne_neuron_pairs.iloc[i]
        peak_fr = []
        for j, unit_type in enumerate(('nonne', 'ne')):
            ccg = np.array(eval(f'pair.ccg_{unit_type}_{stim}'))
            nspk = np.array(eval(f'pair.nspk_{unit_type}_{stim}'))
            peak_fr.append(
                plot_ccg(ax[j], ccg, None, nspk=nspk, causal=pair[f'inclusion_{stim}'], xlim=[-25, 25])
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
    n_pairs = len(pairs) if unit_type == 'input' else 1
    if axes.ndim > 1:
        axes = axes.flatten()
    for i in range(n_pairs):
        unit_idx = pairs.iloc[i][f'{unit_type}_idx']
        unit = units[unit_idx]
        ax = axes[i]
        idx = np.where(unit.adjacent_chan == unit.chan)[0][0]
        waveform_mean = unit.waveforms_mean[idx, :]
        waveform_std = unit.waveforms_std[idx, :]
        if unit_type == 'input':
            plot_waveform(ax, waveform_mean, waveform_std)
            if position_idx is not None:
                ax.set_title('neuron #{}'.format(position_idx[unit_idx]+1), fontsize=6)
            else:
                ax.set_title('neuron #{}'.format(unit_idx+1), fontsize=6)
        else:
            tpd = unit.waveform_tpd
            plot_waveform(ax, waveform_mean, waveform_std, tpd=tpd)
            ax.set_title('unit{}'.format(pairs.iloc[i][f'{unit_type}_unit']), fontsize=6)


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

def batch_plot_common_target_ccg(datafolder=r'E:\Congcong\Documents\data\connection\data-summary',
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
        
    
# ------------------------------------ figure plots ----------------------------------------------
def figure1(datafolder='E:\Congcong\Documents\data\connection\data-pkl', 
            figfolder = r'E:\Congcong\Documents\data\connection\paper\figure',
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

    
    fig = plt.figure(figsize=[8.5*cm, 12*cm])
    
    # plot example STRFs
    x_start = [.45, .75]
    y_start = .8
    x_waveform = .07
    y_waveform = .05
    x_strf = .15
    y_strf = .12
    
    for i, unit in enumerate((input_unit, target_unit)):
        axes = [fig.add_axes([x_start[i] + .16, y_start + .08, x_waveform, y_waveform]), 
                fig.add_axes([x_start[i], y_start, x_strf, y_strf])]
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
        
        if i == 1:
            axes[1].set_ylabel('')
            axes[1].set_xlabel('')
            cb.ax.set_yticklabels(['-Max', '0', 'Max'])

    # plot ccg
    x_start = .1
    y_start = .5
    x_fig = .3
    y_fig = .12
    # spon
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    ccg = np.array(pair.ccg_spon.values[0])
    baseline = np.array(pair.baseline_spon.values[0])
    thresh = np.array(pair.thresh_spon.values[0])
    plot_ccg(ax, ccg, baseline, thresh)
    efficacy = pair.efficacy_spon.values[0]
    ax.text(20, 40, f'efficacy = {efficacy:.2f}', fontsize=6, color='r')
    ax.set_ylim([0, 100])

    # stim
    ax = fig.add_axes([.5, y_start, x_fig, y_fig])
    ccg = np.array(pair.ccg_dmr.values[0])
    baseline = np.array(pair.baseline_dmr.values[0])
    thresh = np.array(pair.thresh_dmr.values[0])
    plot_ccg(ax, ccg, baseline, thresh)
    efficacy = pair.efficacy_dmr.values[0]
    ax.text(20, 20, f'efficacy = {efficacy:.2f}', fontsize=6, color='r')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_ylim([0, 50])
    
    # plot distribution of efficacy
    x_start = .08
    y_start = .1
    x_fig = .2
    y_fig = .15
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    batch_hist_efficacy(ax, stim='spon', color='k')
    batch_hist_efficacy(ax, stim='dmr', color='grey')
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 15])
    ax.set_yticks(range(0, 16, 5))
    
    
    # plot best frequency
    x_start = .43
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    batch_scatter_bf(ax, stim='spon', color='k')
    batch_scatter_bf(ax, stim='dmr', color='grey')
    
    # plot fr distribution
    x_start = .78
    x_fig = .2
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    batch_plot_fr(ax)
    #fig.savefig(os.path.join(figfolder, 'fig1.jpg'), dpi=300)
    #fig.savefig(os.path.join(figfolder, 'fig1.pdf'), dpi=300)

    
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
    print(stim, efficacy.mean(), efficacy.std())
    if stim == 'spon':
        _, p = stats.mannwhitneyu(pairs['efficacy_spon'], pairs[pairs.sig_dmr]['efficacy_dmr'])
        print(f'ranksum test: p = {p}')
    elif stim == 'dmr':
        _, p = stats.wilcoxon(pairs['efficacy_spon'], pairs['efficacy_dmr'])
        print(f'signrank test: p = {p}')
        print('spon:', pairs['efficacy_spon'].mean(), pairs['efficacy_spon'].std())
        print('dmr:', pairs['efficacy_dmr'].mean(), pairs['efficacy_dmr'].std())


def batch_scatter_bf(ax, datafolder='E:\Congcong\Documents\data\connection\data-pkl',
                           stim='spon', color='k'):
    files = glob.glob(os.path.join(datafolder, '*pairs.json'))
    bf_input = []
    bf_target = []
    for file in files:
        pairs = pd.read_json(file)
        pairs = pairs[pairs[f'sig_{stim}']]
        bf_input.extend(pairs.input_bf)
        bf_target.extend(pairs.target_bf)
    bf_input = np.log2(np.array(bf_input) / 500)
    bf_target = np.log2(np.array(bf_target) / 500)
    idx = (bf_input < 6) & (bf_target < 6)
    bf_input, bf_target = bf_input[idx], bf_target[idx]
    ax.plot([0, 6], [0, 6], color='k')

    ax.scatter(bf_input, bf_target, color=color, s=3)
    ax.set_xticks(range(0, 7, 2))
    ax.set_yticks(range(0, 7, 2))
    ax.set_xticklabels([.5, 2, 8, 32])
    ax.set_yticklabels([.5, 2, 8, 32])

    ax.set_xlim([0, 6])
    ax.set_ylim([0, 6])
    ax.set_xlabel('MGB neuron BF (kHz)')
    ax.set_ylabel('A1 neuron BF (kHz)')
    print('n(ccg_{}) = {}'.format(stim, sum(idx)))
    
    
def batch_plot_fr(ax, datafolder='E:\Congcong\Documents\data\connection\data-summary'):
    with open(os.path.join(datafolder, 'fr_all.json'), 'r') as f:
        data = json.load(f)
    
    color = ['k', 'grey']
    for i, region in enumerate(('MGB', 'A1')):
        print('region')
        diff = np.array(data['dmr_'+region]) - np.array(data['spon_'+region])
        ax.hist([diff], bins=np.arange(-10, 11, 1), 
                histtype='step', density=True, linewidth=.8, color=color[i])
        ax.scatter(np.mean(diff), .36, s=8, marker='v', facecolors='none', edgecolors=color[i])
        _, p = stats.wilcoxon(diff)
        print('p =', p)
        print('mean = ', np.mean(diff))
        print('std = ', np.std(diff))
   
    ax.set_xlim([-10, 10])
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_ylim([0, .4])
    ax.set_ylabel('Probability density')
    ax.set_xlabel('\u0394Firing rate')


def figure2(figfolder = r'E:\Congcong\Documents\data\connection\paper\figure'):
    
    fig = plt.figure(figsize=[11.6*cm, 12*cm])
    
    # PART1: correlation of MGB neuros sharing A1 targets
    # example CCG MGB neuronal pairs
    example_file = os.path.join(r'E:\Congcong\Documents\data\connection\data-pkl',
                                r'200820_230604-site4-5655um-25db-dmr-31min-H31x64-fs20000.pkl')
    with open(example_file, 'rb') as f:
        session = pickle.load(f)
    input1 = session.spktrain_spon[0][0]
    input1 = (input1 - input1.mean()) / input1.std()
    input1 = input1[50:-50]
    # example1
    x_start = .1
    y_start = .85
    x_fig = .15
    y_fig = .1
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])    
    input2 = session.spktrain_spon[0][15]
    input2 = (input2 - input2.mean()) / input2.std()
    corr = np.correlate(input1, input2) / len(input2)
    taxis = np.arange(-25, 25.1, .5)
    ax.bar(taxis, corr, color='k')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-.005, .04])
    ax.set_yticks([0, .02, .04])
    ax.set_xticklabels([])
    #example2
    y_start = .72
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])    
    input2 = session.spktrain_spon[0][7]
    input2 = (input2 - input2.mean()) / input2.std()
    corr = np.correlate(input1, input2) / len(input2)
    ax.bar(taxis, corr, color='k')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-.005, .04])
    ax.set_yticks([0, .02, .04])
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Correlation')
    # plot corr of pairs of neurons sharing common target
    x_start = .7
    y_start = .7
    x_fig = .29
    y_fig = .2
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_corr_common_target(ax=ax)
    
    # PART2: correlation of cNE members and nonmembers
    input1 = session.spktrain_spon[0][8]
    input1 = (input1 - input1.mean()) / input1.std()
    input1 = input1[50:-50]
    # example1
    x_start = .1
    y_start = .55
    x_fig = .15
    y_fig = .1
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])    
    input2 = session.spktrain_spon[0][15]
    input2 = (input2 - input2.mean()) / input2.std()
    corr = np.correlate(input1, input2) / len(input2)
    taxis = np.arange(-25, 25.1, .5)
    ax.bar(taxis, corr, color='k')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-.0025, .02])
    ax.set_yticks([0, .01, .02])
    ax.set_xticklabels([])
    #example2
    y_start = .42
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])    
    input2 = session.spktrain_spon[0][11]
    input2 = (input2 - input2.mean()) / input2.std()
    corr = np.correlate(input1, input2) / len(input2)
    ax.bar(taxis, corr, color='k')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-.0025, .02])
    ax.set_yticks([0, .01, .02])
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Correlation')
    # plot cNE member corr
    y_start = .4
    x_start = .7
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_corr_ne_members(ax=ax)
    
    # plot probability of cNE members sharing target
    y_start = .1
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_prob_share_target(ax=ax)
    fig.savefig(os.path.join(figfolder, 'fig2.jpg'), dpi=300)
    fig.savefig(os.path.join(figfolder, 'fig2.pdf'), dpi=300)


def plot_corr_common_target(datafolder=r'E:\Congcong\Documents\data\connection\data-summary', 
                            group='exp', ax=None, savefolder=None):
    file = os.path.join(datafolder,'pairs_common_target_corr.json')
    pairs = pd.read_json(file)
    if ax is None:
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_axes([.2, .2, .7, .7])
    if group == 'exp':
        pairs.drop_duplicates(subset=['exp', 'input1', 'input2'], inplace=True)
        pairs['share_target'] = pairs.target > -1
        corr = pairs.groupby(['exp', 'share_target'])['corr'].median()
        corr = pd.DataFrame(corr).reset_index()
        m = corr.groupby('share_target')['corr'].mean()
        sd = corr.groupby('share_target')['corr'].std()
        m = m.iloc[[1, 0]]
        sd = sd.iloc[[1, 0]]
        m.plot.bar(edgecolor=['k', 'grey'], ax=ax, facecolor='w', linewidth=2)
        ebar_colors=['k', 'grey']
        for i in range(0, len(corr), 2):
            ax.plot([1, 0], corr.iloc[i: i+2]['corr'], 'k', linewidth=.6)
        for c in range(2):
            ax.errorbar(x=c, y=m.iloc[c], yerr=sd.iloc[c], fmt='None', color=ebar_colors[c], 
                        capsize=5, linewidth=1, zorder=1)
            ax.scatter(c * np.ones(len(corr)//2), corr[corr.share_target == 1 - c]['corr'], 
                       facecolor=ebar_colors[c], s=15, edgecolor='w', linewidth=.5)
      
        ax.set_ylabel('Median correlation')
        
        _, p = stats.wilcoxon(corr[corr.share_target]['corr'],
                              corr[~corr.share_target]['corr'])
        print('n(recording) = ', len(corr) / 2)
        print('Wilcoxon: p =', p)
        plot_significance_star(ax, p, [0, 1], .079, .08)
        ax.set_ylim([0, .08])

    ax.set_xlabel('')
    ax.set_xticklabels(['share common target', 'no common terget'], rotation=0)
    if savefolder is not None:
        fig.savefig(os.path.join(savefolder, 'pair_corr_share_target.jpg'), dpi=300)
        plt.close()


def plot_corr_ne_members(datafolder=r'E:\Congcong\Documents\data\connection\data-summary', 
                         ax=None, savefolder=None):
    if ax is None:
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_axes([.2, .2, .7, .7])
        
    pairs = pd.read_json(os.path.join(datafolder,'member_nonmember_pair_xcorr.json'))
    pairs = pairs[pairs.stim == 'spon']
    v = ax.violinplot([pairs[pairs.member]['corr']], points=100, positions=[0], 
                       showextrema=False, widths=.8)
    set_violin_half(v, half=None, color='k')
    v = ax.violinplot([pairs[~pairs.member]['corr']], points=100, positions=[1], 
                       showextrema=False, widths=.8)
    set_violin_half(v, half=None, color='grey')
    ax.set_ylabel('Pairwise correlation')
        
    _, p = stats.mannwhitneyu(pairs[pairs.member]['corr'],
                          pairs[~pairs.member]['corr'])
    print(p)
    plot_significance_star(ax, p, [0, 1], .28, .29)
    ax.set_ylim([-.1, .3])
    ax.set_yticks(np.arange(-.1, .31, .1))

    ax.set_xlabel('')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Within cNE\n(n={})'.format(sum(pairs.member)), 
                        'Outside cNE\n(n={})'.format(len(pairs) - sum(pairs.member))])
    if savefolder is not None:
        fig.savefig(os.path.join(savefolder, 'pair_corr_ne_member.jpg'), dpi=300)
        plt.close()
        

def plot_prob_share_target(datafolder=r'E:\Congcong\Documents\data\connection\data-summary', 
                           ax=None, savefolder=None):
    if ax is None:
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_axes([.2, .2, .7, .7])
        
    pairs = pd.read_json(os.path.join(datafolder,'pairs_common_target_ne.json'))

    pairs = pairs.groupby(['exp', 'within_ne']).filter(lambda x: x['within_ne'].count() > 3)
    prob_share =  pairs.groupby(['exp', 'within_ne'])['share_target'].mean()
    prob_share = pd.DataFrame(prob_share).reset_index()
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
        ax.plot([1, 0], prob_share.iloc[i: i+2]['share_target'], 'k', linewidth=.6)
    for c in range(2):
        ax.errorbar(x=c, y=m.iloc[c], yerr=np.array([[0], [sd.iloc[c]]]), fmt='None', color=ebar_colors[c], 
                    capsize=5, linewidth=1, zorder=1)
        ax.scatter(c * np.ones(len(prob_share)//2), prob_share[prob_share.within_ne == 1 - c]['share_target'], 
                   facecolor=ebar_colors[c], s=15, edgecolor='w', linewidth=.5)
  
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


def figure3(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure'):
    
    example_file = os.path.join(
        datafolder, '200820_230604-site4-5655um-25db-dmr-31min-H31x64-fs20000-pairs-ne-spon.json')
    nepairs = pd.read_json(example_file)
    exp = re.search('\d{6}_\d{6}', example_file).group(0)
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
    nefile = re.sub('-pairs-ne-spon.json', '-ne-20dft-spon.pkl', example_file)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    patterns = ne.patterns
    cne = 2
    # cNE2-A1_33
    target_idx = 3
    fig, ne_neuron_pairs = plot_ne_neuron_connection_ccg(nepairs, cne, target_idx, input_units, target_units, patterns)
    axes = fig.get_axes()
    axes[0].set_ylim([4600, 5800])
    axes[0].invert_yaxis()
    axes_ccg = axes[2:8]
    for ax in axes_ccg:
        ax.set_ylim([0, 200])
        ax.set_yticks(range(0, 201, 50))
        ax.set_yticklabels([0, '', 100, '', 200])
    fig.savefig(os.path.join(figfolder, 'fig3-1.jpg'), dpi=300)
    fig.savefig(os.path.join(figfolder, 'fig3-1.pdf'), dpi=300)
    # cNE2-A1_43
    target_idx = 38
    fig, ne_neuron_pairs = plot_ne_neuron_connection_ccg(nepairs, cne, target_idx, input_units, target_units, patterns)
    axes = fig.get_axes()
    axes[0].set_ylim([4600, 5800])
    axes[0].invert_yaxis()
    axes_ccg = axes[2:8]
    for ax in axes_ccg:
        ax.set_ylim([0, 150])
        ax.set_yticks(range(0, 151, 50))
        ax.set_yticklabels(range(0, 151, 50))
    fig.savefig(os.path.join(figfolder, 'fig3-2.jpg'), dpi=300)
    fig.savefig(os.path.join(figfolder, 'fig3-2.pdf'), dpi=300)
    plt.close()


def figure4(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure'):
    example_file = os.path.join(
        datafolder, '220825_005353-site6-5500um-25db-dmr-61min-H31x64-fs20000-pairs-ne-spon.json')
    nepairs = pd.read_json(example_file)
    exp = re.search('\d{6}_\d{6}', example_file).group(0)
    _, input_units, target_units, _ = load_input_target_files(datafolder, exp)
    nefile = re.sub('-pairs-ne-spon.json', '-ne-20dft-spon.pkl', example_file)
    with open(nefile, 'rb') as f:
        ne = pkl.load(f)
    patterns = ne.patterns
    target_idx = 10
    plt.close()
    # cNE3-A1_149
    cne = 3
    fig, ne_neuron_pairs = plot_ne_neuron_connection_ccg(nepairs, cne, target_idx, input_units, target_units, patterns)
    axes = fig.get_axes()
    axes[0].set_ylim([4400, 5400])
    axes[1].set_xlim([-.3, .7])
    axes[0].invert_yaxis()
    axes_ccg = axes[2:12]
    for ax in axes_ccg:
        ax.set_ylim([0, 15])
        ax.set_yticks(range(0, 16, 5))
        ax.set_yticklabels(range(0, 16, 5))
    fig.savefig(os.path.join(figfolder, 'fig4-2.jpg'), dpi=300)
    fig.savefig(os.path.join(figfolder, 'fig4-2.pdf'), dpi=300)
    plt.close()
    # cNE4-A1_149
    cne = 4
    fig, ne_neuron_pairs = plot_ne_neuron_connection_ccg(nepairs, cne, target_idx, input_units, target_units, patterns)
    axes = fig.get_axes()
    axes[0].set_ylim([4400, 5400])
    axes[1].set_xlim([-.3, .7])
    axes[0].invert_yaxis()
    axes_ccg = axes[2:14]
    for ax in axes_ccg:
        ax.set_ylim([0, 50])
        ax.set_yticks(range(0, 51, 25))
        ax.set_yticklabels(range(0, 51, 25))
    fig.savefig(os.path.join(figfolder, 'fig4-1.jpg'), dpi=300)
    fig.savefig(os.path.join(figfolder, 'fig4-1.pdf'), dpi=300)
    plt.close()


def figure5(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure'):
    
    fig = plt.figure(figsize=[figure_size[2][0], figure_size[2][0]])
    x_fig = .35
    y_fig = .35
    # panel A: BS/NS neurons
    print('A')
    y_start = .6
    x_start = .1
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_waveform_ptd(ax=ax)
    ax.set_ylabel('# of A1 neurons')
    print('B')
    # panel B: BS/NS neurons
    x_start = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_efficacy_gain_cell_type(ax=ax)
    
    x_start = .1
    y_start = .1
    # panel C: NE vs nonNE spike efficacy
    print('C')
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_efficacy_ne_vs_nonne(ax, change='increase')
    print('D')
    x_start = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_efficacy_ne_vs_nonne(ax, change='decrease')
    
    fig.savefig(os.path.join(figfolder, 'fig5.jpg'), dpi=300)
    fig.savefig(os.path.join(figfolder, 'fig5.pdf'), dpi=300)
    
   
def plot_efficacy_ne_vs_nonne(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary', stim='spon', change='increase'):
    pairs = pd.read_json(os.path.join(datafolder, f'ne-pairs-perm-test-{stim}.json'))
    if 'ss' not in stim:
        pairs = pairs[pairs[f'efficacy_ne_{stim}'] > 0]
        pairs = pairs[pairs[f'efficacy_nonne_{stim}'] > 0]
    pairs['waveform_ns'] = pairs.target_waveform_tpd < .45
    ax.scatter(pairs[f'efficacy_nonne_{stim}'], pairs[f'efficacy_ne_{stim}'], s=15, color='grey', edgecolor='w')
    if change == 'increase':
        pairs = pairs[pairs[f'efficacy_ne_{stim}'] > pairs[f'efficacy_nonne_{stim}']]
    else:
        pairs = pairs[pairs[f'efficacy_ne_{stim}'] > pairs[f'efficacy_nonne_{stim}']]
    pairs_sig = pairs[(pairs.efficacy_diff_p < .05) & (pairs['waveform_ns'])]
    ax.scatter(pairs_sig[f'efficacy_nonne_{stim}'], pairs_sig[f'efficacy_ne_{stim}'], s=15, color=tpd_color[1], edgecolor='w')
    pairs_sig = pairs[(pairs.efficacy_diff_p < .05) & (~pairs['waveform_ns'])]
    ax.scatter(pairs_sig[f'efficacy_nonne_{stim}'], pairs_sig[f'efficacy_ne_{stim}'], s=15, color=tpd_color[0], edgecolor='w')
    
    ax.plot([0, 30], [0, 30], 'k')
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])
    ax.set_xticks(range(0, 31, 10))
    ax.set_yticks(range(0, 31, 10))
    ax.set_xlabel('nonNE spike efficacy (%)')
    ax.set_ylabel('NE spike efficacy (%)')

    _, p = stats.wilcoxon(pairs[f'efficacy_ne_{stim}'], pairs[f'efficacy_nonne_{stim}'])
    print('p =', p)
    if p > .001:
        ax.text(2, 23, f'p = {p:.3f}', fontsize=7)
    else:
        ax.text(2, 23, f'p = {p:.2e}', fontsize=7)

    
    
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


def plot_efficacy_gain_cell_type(ax, datafolder='E:\Congcong\Documents\data\connection\data-summary', stim='spon'):
    file = glob.glob(os.path.join(datafolder, f'ne-pairs-perm-test-{stim}.json'))[0]
    pairs = pd.read_json(file)
    pairs['waveform_ns'] = pairs.target_waveform_tpd < .45
    pairs['efficacy_gain'] = (pairs[f'efficacy_ne_{stim}'] - pairs[f'efficacy_nonne_{stim}'])
    boxplot_scatter(ax, x='waveform_ns', y='efficacy_gain', data=pairs, size=3, jitter=.3,
                    order=[False, True], hue='waveform_ns', palette=tpd_color[:2], hue_order=[False, True])
    ax.set_xticklabels(['BS', 'NS'])
    ax.set_xlabel('A1 neuron type')
    ax.set_ylabel('Efficacy gain (%)')
    
    _, p = stats.mannwhitneyu(pairs[pairs.waveform_ns][f'efficacy_gain'], 
                              pairs[~pairs.waveform_ns][f'efficacy_gain'])
    print('p =', p)
    plot_significance_star(ax, p, [0, 1], 15, 16)
    ax.plot([-.5, 1.5], [0, 0], 'k--')
    ax.set_ylim([-10, 15])
    ax.set_yticks(range(-10, 16, 5))
    ax.set_xlim([-.5, 1.5])


def figure6(datafolder=r'E:\Congcong\Documents\data\connection\data-summary',
            figfolder=r'E:\Congcong\Documents\data\connection\paper\figure'):
    fig = plt.figure(figsize=[figure_size[2][0], 4*cm])
    x_start = .1
    y_start = .2
    x_fig = .3
    y_fig = .6
    # panel C: NE vs nonNE spike efficacy
    print('A-i')
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_efficacy_ne_vs_nonne(ax, stim='spon_ss')
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 30])
    ax.set_xticks(range(0, 31, 10))
    ax.set_yticks(range(0, 31, 10))

    # panel D-i: BS/NS neurons
    print('B-ii')
    x_start = .6
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_efficacy_gain_cell_type(ax=ax, stim='spon_ss')
    ax.set_ylim([-10, 30])
    ax.set_yticks(range(-10,31,10))
    fig.savefig(os.path.join(figfolder, 'fig4-A.jpg'), dpi=300)
    fig.savefig(os.path.join(figfolder, 'fig4-A.pdf'), dpi=300)
    
    
def plot_delta_dfficacy_ne_vs_hiact(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary',
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
    