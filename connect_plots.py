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
tpd_color = (colors[1], colors[5], colors[0], colors[4])
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


def plot_ccg(ax, ccg, baseline, thresh=None, taxis=None, nspk=None, causal_method='peak'):
    if taxis is None:
        edges = np.arange(-50, 50.5, .5)
        taxis = (edges[1:] + edges[:-1]) / 2
    if nspk: # normalize to firing rate
        ccg = ccg / (nspk*.5 / 1e3)
    
    ax.bar(taxis, ccg, width=.5, color='k')
    
    # plot causal spikes
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
    ax.set_xlim([-50, 50])
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
    # plot ccg
    ax = fig.add_axes([.1, .1, .6, .3])
    ccg = np.array(pair.ccg_spon)
    baseline = np.array(pair.baseline_spon)
    thresh = np.array(pair.thresh_spon)
    plot_ccg(ax, ccg, baseline)
    ccg = np.array(pair.ccg_dmr)
    
  
def plot_waveform(ax, waveform_mean, waveform_std, color='k', color_shade='lightgrey', tpd=None):
    if tpd:
        if tpd < .45:
            color, color_shade = tpd_color[0],  tpd_color[2]
        else:
            color, color_shade = tpd_color[1],  tpd_color[3]
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
        cne_target = nepairs[['cne', 'target_idx']].drop_duplicates()
        for cne, target_idx in cne_target.values:
            ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_idx == target_idx)]
            n_pairs = len(ne_neuron_pairs)
            assert(n_pairs > 1)
            fig, axes = plt.subplots(n_pairs, 3, figsize=[figure_size[2][0], 2*n_pairs*cm])
            plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs)
            exp = re.search('\d{6}_\d{6}', file).group(0)
            target_unit = ne_neuron_pairs.iloc[0]['target_unit']
            plt.tight_layout()
            axes[len(ne_neuron_pairs) - 1][0].get_xaxis().set_label_coords(1.5,-.5)
            fig.savefig(os.path.join(figfolder, f'ne_ccg_{stim}', f'{exp}-cne_{cne}-target_{target_unit}.jpg'), dpi=300)
            plt.close()


def plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs, stim='spon'):
    for i in range(len(ne_neuron_pairs)):
        ax = axes[i]
        pair = ne_neuron_pairs.iloc[i]
        peak_fr = []
        for j, unit_type in enumerate(('neuron', 'ne', 'nonne')):
            ccg = np.array(eval(f'pair.ccg_{unit_type}_{stim}'))
            nspk = np.array(eval(f'pair.nspk_{unit_type}_{stim}'))
            peak_fr.append(plot_ccg(ax[j], ccg, None, nspk=nspk))
            if i == 0 and j == 0:
                ax[j].get_xaxis().set_label_coords(1.2, -.3)
            else:
                ax[j].set_xlabel('')
                ax[j].set_ylabel('')
        m = max(peak_fr)
        for j, unit_type in enumerate(('neuron', 'ne', 'nonne')):
            ax[j].set_ylim([0, m * 1.1])
            ax[j].text(20, m*0.8, 
                       '{:.2f}'.format(eval(f'pair.efficacy_{unit_type}_{stim}')),
                       fontsize=6)
            if not pair[f'inclusion_{stim}']:
                rect = mpl.patches.Rectangle([-50, 0], 100, 300, facecolor='w', edgecolor='w', alpha=.8)
                ax[j].add_patch(rect)


def batch_plot_ne_neuron_connection_ccg_ss(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', 
                                        figfolder=r'E:\Congcong\Documents\data\connection\figure\ne_ccg_spon_all'):
    files = glob.glob(os.path.join(datafolder, '*-pairs-ne-spon.json'))
    for file in files:
        ss_file = re.sub('spon', 'spon_ss', file)
        nepairs = pd.read_json(file)
        nepairs_ss = pd.read_json(ss_file)
        cne_target = nepairs[['cne', 'target_idx']].drop_duplicates()
        session_file = re.sub('-pairs-ne-spon.json', '.pkl', file)
        with open(session_file, 'rb') as f:
            session = pickle.load(f)
        units = session.units
        for cne, target_idx in cne_target.values:
            ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_idx == target_idx)]
            ne_neuron_pairs_ss = nepairs_ss[(nepairs_ss.cne == cne) & (nepairs_ss.target_idx == target_idx)]
            n_pairs = len(ne_neuron_pairs)
            assert(n_pairs > 1)
            fig = plt.figure(figsize=[figure_size[0][0], (2*n_pairs + .2)*cm])
            # plot waveform of all units
            x_fig = .1
            y_fig = 1.5 /  (2*n_pairs + .2)
            x_space = .03
            y_space = .5 /  (2*n_pairs + .2)
            x_start = .01
            y_start = .2 / (2*n_pairs + .2) + y_space
            for i in range(n_pairs):
                unit_idx = ne_neuron_pairs.iloc[i].input_idx
                unit = units[unit_idx]
                ax = fig.add_axes([x_start, y_start + i * (y_space + y_fig) - y_space, x_fig, y_fig])
                ax.set_title('neuron #{}'.format(unit_idx+1))
                idx = np.where(unit.adjacent_chan == unit.chan)[0][0]
                waveform_mean = unit.waveforms_mean[idx, :]
                waveform_std = unit.waveforms_std[idx, :]
                plot_waveform(ax, waveform_mean, waveform_std)
                
            # plot ccg of all spikes
            nrows = n_pairs
            ncols = 3
            x_start = .18
            axes = add_multiple_axes(fig, nrows, ncols, x_start, y_start, x_fig, y_fig, x_space, y_space)
            axes = axes[::-1]
            plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs, stim='spon')
            x_start = .6
            axes = add_multiple_axes(fig, nrows, ncols, x_start, y_start, x_fig, y_fig, x_space, y_space)
            axes = axes[::-1]
            plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs_ss, stim='spon_ss')
            
            exp = re.search('\d{6}_\d{6}', file).group(0)
            target_unit = ne_neuron_pairs.iloc[0]['target_unit']
            fig.savefig(os.path.join(figfolder, f'{exp}-cne_{cne}-target_{target_unit}-all_ss.jpg'), dpi=300)
            plt.close()


def batch_plot_ne_neuron_connection_strf_ccg_ss(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', 
                                        figfolder=r'E:\Congcong\Documents\data\connection\figure\ne_ccg_dmr'):
    files = glob.glob(os.path.join(datafolder, '*-pairs-ne-dmr.json'))
    for file in files:
        ss_file = re.sub('ne-dmr', 'ne-dmr_ss', file)
        nepairs = pd.read_json(file)
        nepairs_ss = pd.read_json(ss_file)
        cne_target = nepairs[['cne', 'target_idx']].drop_duplicates()
        exp = re.search('\d{6}_\d{6}', file).group(0)
        session_files = glob.glob(os.path.join(datafolder, f'{exp}*fs20000.pkl'))
        assert(len(session_files) == 2)
        target_file, input_file = session_files
        with open(target_file, 'rb') as f:
            session = pickle.load(f)
            target_units = session.units
        with open(input_file, 'rb') as f:
            session = pickle.load(f)
            input_units = session.units
        ne_file = glob.glob(os.path.join(datafolder, f'{exp}*fs20000-ne-20dft-dmr.pkl'))[0]
        with open(ne_file, 'rb') as f:
            ne = pickle.load(f)
            
        for cne, target_idx in cne_target.values:
            ne_neuron_pairs = nepairs[(nepairs.cne == cne) & (nepairs.target_idx == target_idx)]
            ne_neuron_pairs_ss = nepairs_ss[(nepairs_ss.cne == cne) & (nepairs_ss.target_idx == target_idx)]
            n_pairs = len(ne_neuron_pairs)
            assert(n_pairs > 1)
            fig = plt.figure(figsize=[figure_size[0][0], (2*(n_pairs+1) + .2)*cm])
            # plot waveform of all units
            x_fig = .1
            y_fig = 1.5 /  (2*(n_pairs+1) + .2)
            x_space = .03
            y_space = .46 /  (2*(n_pairs+1) + .2)
            x_start = .005
            y_start = .2 / (2*n_pairs + .2) + y_space
            for i in range(n_pairs):
                unit_idx = ne_neuron_pairs.iloc[i].input_idx
                unit = input_units[unit_idx]
                ax = fig.add_axes([x_start, y_start + i * (y_space + y_fig) - y_space, x_fig, y_fig])
                ax.set_title('neuron #{}'.format(unit_idx+1))
                idx = np.where(unit.adjacent_chan == unit.chan)[0][0]
                waveform_mean = unit.waveforms_mean[idx, :]
                waveform_std = unit.waveforms_std[idx, :]
                plot_waveform(ax, waveform_mean, waveform_std)
            
            # plot A1 unit waveform
            ax = fig.add_axes([x_start ,
                               y_start + n_pairs * (y_space + y_fig) - y_space, x_fig, y_fig])
            target_idx = ne_neuron_pairs.iloc[i].target_idx
            unit = target_units[target_idx]
            idx = np.where(unit.adjacent_chan == unit.chan)[0][0]
            waveform_mean = unit.waveforms_mean[idx, :]
            waveform_std = unit.waveforms_std[idx, :]
            plot_waveform(ax, waveform_mean, waveform_std, tpd=unit.waveform_tpd)
            
            # plot strf of units
            x_start = .15
            nrows = n_pairs + 1
            ncols = 1
            axes = add_multiple_axes(fig, nrows, ncols, x_start, y_start, x_fig, y_fig, x_space, y_space)
            axes = axes[::-1]
            for i, input_idx in enumerate(list( ne_neuron_pairs.input_idx)):
                strf = np.array(input_units[input_idx].strf)
                taxis = input_units[input_idx].strf_taxis
                faxis = input_units[input_idx].strf_faxis
                plot_strf(ax=axes[i][0], strf=strf, taxis=taxis, faxis=faxis)
                if i > 0:
                    axes[i][0].set_xlabel('')
                    axes[i][0].set_ylabel('')
            # A1
            strf = np.array(target_units[target_idx].strf)
            plot_strf(ax=axes[-1][0], strf=strf, taxis=taxis, faxis=faxis)
            axes[-1][0].set_xlabel('')
            axes[-1][0].set_ylabel('')
            
            # plot strf of ne units
            x_start = x_start + x_fig + x_space
            nrows = n_pairs + 1
            ncols = 1
            axes = add_multiple_axes(fig, nrows, ncols, x_start, y_start, x_fig, y_fig, x_space, y_space)
            axes = axes[::-1]
            ne_units = ne.member_ne_spikes[cne]
            for i, unit in enumerate(ne_units):
                strf = np.array(unit.strf)
                plot_strf(ax=axes[i][0], strf=strf, taxis=taxis, faxis=faxis)
                axes[i][0].set_xlabel('')
                axes[i][0].set_ylabel('')
                axes[i][0].set_yticklabels([])
            # cne
            strf = np.array(ne.ne_units[cne].strf)
            plot_strf(ax=axes[-1][0], strf=strf, taxis=taxis, faxis=faxis)
            axes[-1][0].set_xlabel('')
            axes[-1][0].set_ylabel('')
            axes[i][0].set_yticklabels([])

            # plot ccg of all spikes
            nrows = n_pairs
            ncols = 3
            x_start = .45
            axes = add_multiple_axes(fig, nrows, ncols, x_start, y_start, x_fig, y_fig, x_space, y_space)
            axes = axes[::-1]
            plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs, stim='dmr')
            for ax in axes:
                ax[-1].remove()
            x_start = .75
            axes = add_multiple_axes(fig, nrows, ncols, x_start, y_start, x_fig, y_fig, x_space, y_space)
            axes = axes[::-1]
            plot_ne_neuron_pairs_connection_ccg(axes, ne_neuron_pairs_ss, stim='dmr_ss')
            for ax in axes:
                ax[-1].remove()
            exp = re.search('\d{6}_\d{6}', file).group(0)
            target_unit = ne_neuron_pairs.iloc[0]['target_unit']
            fig.savefig(os.path.join(figfolder, f'{exp}-cne_{cne}-target_{target_unit}.jpg'), dpi=300)
            plt.close()


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
    return axes


def batch_plot_icweight(datafolder='E:\Congcong\Documents\data\connection\data-pkl',
                        savefolder=r'E:\Congcong\Documents\data\connection\figure\icweight'):
    files = glob.glob(os.path.join(datafolder, '*ne-20dft-spon.pkl'))
    for file in files:
        with open(file, 'rb') as f:
            ne = pickle.load(f)
        n_neuron = ne.patterns.shape[-1]
        member_thresh = 1 / np.sqrt(n_neuron)
        exp = ne.exp
        for cne, members in ne.ne_members.items():
            fig = plt.figure(figsize=[1, 3])
            ax = fig.add_axes([.3, .1, .5, .8])
            weights = ne.patterns[cne]
            plot_ICweight(ax, weights, member_thresh, direction='v')
            fig.savefig(os.path.join(savefolder, f'{exp}-cne{cne}.jpg'), dpi=300)
            plt.close()


def batch_plot_strf(datafolder=r'E:\Congcong\Documents\data\connection\data-pkl', 
                   figfolder=r'E:\Congcong\Documents\data\connection\figure\strf'):
    spkfiles = glob.glob(os.path.join(datafolder, '*fs20000.pkl'))
    for i, file in enumerate(spkfiles):
        with open(file, 'rb') as f:
            session = pickle.load(f)
        session.plot_strf(figfolder=figfolder)

# ---------------------------summary plots -----------------------------------------
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
    ax.hist(tpd, np.arange(0, .5, .05), color=tpd_color[0])
    ax.hist(tpd, np.arange(.45, 1.5, .05), color=tpd_color[1])
    ax.set_xlabel('Trough-Peak delay (ms)')
    ax.set_ylabel('# of neurons')
    if savefolder:
        fig.savefig(os.path.join(savefolder, f'tpd-{region}.jpg'), bbox_inches='tight', dpi=300)


def plot_efficacy_ne_vs_nonne(ax, datafolder=r'E:\Congcong\Documents\data\connection\data-summary', stim='spon'):
    pairs = pd.read_json(os.path.join(datafolder, f'ne-pairs-{stim}.json'))
    pairs = pairs[pairs[f'inclusion_{stim}']]
    pairs = pairs[pairs[f'efficacy_ne_{stim}'] > 0]
    pairs = pairs[pairs[f'efficacy_nonne_{stim}'] > 0]
    ax.scatter(pairs.efficacy_nonne_spon, pairs.efficacy_ne_spon, s=20, color='k')
    ax.plot([0, 40], [0, 40], 'k')
    ax.set_xlim([0, 25])
    ax.set_ylim([0, 25])
    
    _, p = stats.wilcoxon(pairs[f'efficacy_ne_{stim}'], pairs[f'efficacy_nonne_{stim}'])
    print(p)
    
    
        
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
        im = plot_strf(axes[1], strf, taxis=unit.strf_taxis, faxis=unit.strf_faxis, 
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
    y_start = .5
    x_fig = .5
    y_fig = .12
    ax = fig.add_axes([x_start[0], y_start, x_fig, y_fig])
    ccg = np.array(pair.ccg_spon.values[0])
    baseline = np.array(pair.baseline_spon.values[0])
    thresh = np.array(pair.thresh_spon.values[0])
    plot_ccg(ax, ccg, baseline, thresh)
    ccg = np.array(pair.ccg_dmr.values[0])
    taxis = np.array(pair.taxis.values[0])
    ax.plot(taxis, ccg, color='grey', linewidth=.8)
    efficacy = pair.efficacy_spon.values[0]
    ax.text(20, 40, f'efficacy = {efficacy:.2f}', fontsize=6, color='r')
    efficacy = pair.efficacy_dmr.values[0]
    ax.text(20, 20, f'efficacy = {efficacy:.2f}', fontsize=6, color='grey')

    ax.set_ylim([0, 100])
    
    # plot distribution of efficacy
    x_start = .5
    y_start = .1
    x_fig = .25
    y_fig = .15
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    batch_hist_efficacy(ax, stim='spon', color='k')
    batch_hist_efficacy(ax, stim='dmr', color='grey')
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 15])
    ax.set_yticks(range(0, 16, 5))
    
    
    # plot best frequency
    x_start = .1
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    batch_scatter_bf(ax, stim='spon', color='k')
    batch_scatter_bf(ax, stim='dmr', color='grey')
    
    
    fig.savefig(os.path.join(figfolder, 'fig1.jpg'), dpi=300)
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
    
def batch_plot_fr(ax, datafolder='E:\Congcong\Documents\data\connection\data-summary',
                    method='all'):
    if method == 'all':
        with open(os.path.join(datafolder, 'fr_all.json'), 'r') as f:
            data = json.load(f)
        positions = [1, 3]

        v1 = ax.violinplot([data['spon_MGB'], data['dmr_MGB']], points=100, positions=positions, 
                           showextrema=False, widths=.8)
        set_violin_half(v1, half='l', color=MGB_color[0])

        v2 = ax.violinplot([data['spon_A1'], data['dmr_A1']], points=100, positions=positions, 
                           showextrema=False, widths=.8)
        set_violin_half(v2, half='r', color=A1_color[0])
    elif method == 'pairs':
        data = pd.read_json(os.path.join(datafolder, 'pairs.json'))
        v1 = ax.violinplot([np.unique(data['input_fr_spon']), np.unique(data['input_fr_dmr'])], 
                           points=100, positions=positions, showextrema=False, widths=.8)
        set_violin_half(v1, half='l', color=MGB_color[0])

        v2 = ax.violinplot([data['spon_A1'], data['dmr_A1']], points=100, positions=positions, 
                           showextrema=False, widths=.8)


def figure2(figfolder = r'E:\Congcong\Documents\data\connection\paper\figure'):
    
    fig = plt.figure(figsize=[8.5*cm, 12*cm])
    
    # plot corr of pairs of neurons sharing common target
    x_start = .12
    y_start = .7
    x_fig = .4
    y_fig = .2
    ax = fig.add_axes([x_start, y_start, x_fig, y_fig])
    plot_corr_common_target(ax=ax)
    
    # plot cNE member corr
    y_start = .4
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

    