# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:34:42 2023

@author: Congcong
"""
# define classes of neuron, recording_session and subclasses of neurons
import pandas as pd
import numpy as np
import scipy.stats as stats
import pickle
import os
import matplotlib.pyplot as plt
import connect_toolbox as ct


class Session:

    def __init__(self, exp, units):
        """Initiate session object for a single recording

        INPUT:
        exp: time of recording in the format of "yyyy-mm-dd_hh-mm-ss"
        units: single unit data, see SingleUnit class
        """
        self.exp = exp
        self.units = units
    
    
    def save_pkl_file(self, savefile_path=None):
        """
        save session data as pickle files

        Parameters
        ----------
        savefile_path : TYPE, optional
            file path to save the pickle file. If None, get file_path from session data

        """
        if savefile_path is None:
            savefile_path = self.file_path
        else:
            self.file_path = savefile_path
        with open(savefile_path, 'wb') as output:
            pickle.dump(self, output)
    
    
    def split_unit_spiketimes(self, stim_times_folder):
        """
        split spike times based on the stimulus conditions, deal with 'tone_ftc' and 'spon'
        saved under each single unitas spiketimes['tone_ftc'] and spiketimes['spon']

        Parameters
        ----------
        stim_times_folder: 
            folder where stim_times_*.csv is saved
            
        """
        stim_times = pd.read_csv(os.path.join(stim_times_folder, f'stim_times_{self.exp}.csv'), index_col=0)
        unique_stims = np.unique(stim_times.stim_label)
        for stim in unique_stims:
            s_start = stim_times[stim_times.stim_label == stim]['stim_onset'].min()
            s_end = stim_times[stim_times.stim_label == stim]['stim_offset'].max()
            for unit in self.units:
                unit.add_spiketimes(stim, [s_start, s_end])
        
        # add spike times of spon activity
        s_start = 0
        s_end = stim_times.stim_onset.min()
        for unit in self.units:
            unit.add_spiketimes('spon', [s_start, s_end])
                
    
    def add_unit_info(self, unit_info_folder:str):
        """
        add unit info to each unit in the recording

        Parameters
        ----------
        unit_info_folder : str
            where the unit_info_*.csv is saved
        """
        unit_info =  pd.read_csv(os.path.join(unit_info_folder, f'unit_info_{self.exp}.csv'))
        for unit in self.units:
            unit_id = unit.unit
            info = unit_info.loc[unit_id]
            unit.add_unit_info(info)
    
    
    def calc_response_to_ftc(self, stim_times_folder):
        stim_times = pd.read_csv(os.path.join(stim_times_folder, f'stim_times_{self.exp}.csv'), index_col=0)
        stim_times = stim_times[stim_times.stim_label == 'tone_ftc']
        self.ftc_freqs = stim_times.freq
        for unit in self.units:
            unit.calc_response_to_ftc(stim_times.stim_onset)
            
            
    def plot_response_to_ftc(self, figfolder):
        fig = plt.figure(figsize=[30, 8])
        
        # plot location of neurons
        ax = fig.add_axes([.03, .1, .1, .8])
        for unit in self.units:
            fr_ratio = unit.ftc['fr_ratio']
            c = 'red' if unit.ftc['p'] * len(self.units) < .01 else 'gray' # red for significant fr change, grey for nonsig
            ax.scatter(fr_ratio, unit.ch, s=unit.firing_rate*5, color=c, alpha=.3)
        ax.set_xscale('log')
        ax.set_xlim([.1, 100])
        ax.set_ylim(0, 350)
        ax.set_ylabel('Channel #')
        ax.set_xlabel('fr_ratio')
        
        x_start = .16
        x_fig = .035
        x_space = .005
        y_start = .1
        y_fig = .4
        y_space = .01
        axes_raster = []
        axes_psth = []
        unique_freqs = np.unique(self.ftc_freqs)
        for i, req in enumerate(unique_freqs):
            axes_psth.append(fig.add_axes([x_start + i *(x_fig + x_space), y_start, x_fig, y_fig]))
            axes_raster.append(fig.add_axes([x_start + i *(x_fig + x_space), y_start + y_fig + y_space, x_fig, y_fig]))
        axes = np.array([axes_raster, axes_psth])
        # raster of su responses to pure tones
        for unit in self.units:
            fr_ratio = unit.ftc['fr_ratio']
            c = 'purple' if unit.ftc['p'] * len(self.units) < .01 else 'dimgray'
            sc = ax.scatter(fr_ratio, unit.ch, s=unit.firing_rate*5, color=c, alpha=1)
            unit.plot_response_to_ftc(axes, self.ftc_freqs)
            fig.savefig(os.path.join(figfolder, f'{self.exp}_uinit{unit.unit:03d}.jpg'), dpi=300)
            sc.remove()
        plt.close()
            
    
    def get_connected_pairs(self, input_chs, target_chs, window_size=50, binsize=.5, thresh=.999):
        # get input and target units
        chs = []
        for unit in self.units:
            chs.append(unit.ch)
        input_units = [self.units[i] for i, ch in enumerate(chs) if ch in input_chs]
        target_units = [self.units[i] for i, ch in enumerate(chs) if ch in target_chs]
        
        pairs = None
        n_check = len(input_units) * len(target_units)
        checked = 0
        new_pair = {}
        
        for input_unit in input_units:
            for target_unit in target_units:
                checked += 1
                new_pair.update({'input_unit': input_unit.unit, 'target_unit': target_unit.unit})
                print(f'{checked}/{n_check}')
                
                input_spiketimes = input_unit.spiketimes['spon'] * 1e3 # spiketimes in ms
                target_spiketimes = target_unit.spiketimes['spon'] * 1e3 # spiketimes in ms
                        
                ccg, edges, nspk = ct.get_ccg(input_spiketimes, target_spiketimes, 
                                           window_size=window_size, binsize=binsize)
                connect, ccg_dict = ct.check_connection(ccg, 'spon', alpha=thresh)
                new_pair.update(ccg_dict)
                new_pair.update({'nspk_spon': nspk})
                
                taxis = (edges[1:] + edges[:-1]) / 2
                new_pair.update({'taxis': [taxis]})
                    
                if pairs is None:
                    pairs = pd.DataFrame(new_pair)
                else:
                    pairs = pd.concat([pairs, pd.DataFrame(new_pair)], ignore_index=True)
                         
        return pairs
    
    
class SingleUnit:

    def __init__(self, unit, spiketimes):
        """
        Initiate SingleUnit object for a unit

        INPUT:
        unit: unit number
        spiketimes: all spiketimes of the unit, to be saved under spiketimes['all']
        """
        self.unit = unit
        self.spiketimes = {'all': spiketimes} # in seconds
    
    
    def add_waveform_info(self, waveform_mean, waveform_std):
        self.waveform_mean = np.array(waveform_mean)
        self.waveform_std = np.array(waveform_std)
        
        
    def add_spiketimes(self, stim, block):
        """
        get spike times under the stimulus block

        INPUT:
        stim: stimulus label
        block: [t_start, t_end] of the stimulus block
        """
        spiketimes = self.spiketimes['all']
        self.spiketimes[stim] = spiketimes[(spiketimes > block[0]) & (spiketimes < block[1])]
        
    
    def add_unit_info(self, info):
        assert(info.cluster_id == self.unit)
        self.ch = int(info.unit_ch)
        self.ch_idx = int(info.unit_ch_idx)
        self.nch = int(info.unit_n_ch)
        self.n_spike = int(info.n_spike)
        assert(len(self.spiketimes['all']) == self.n_spike)
        self.frac_under_rp = info.frac_under_rp
        self.contam_pct = info.contam_pct
        self.firing_rate = info.firing_rate
        self.fr_stable_start = info.fr_stable_start
        self.fr_stable_stop = info.fr_stable_stop
        self.fr_stable_pct = info.fr_stable_pct
        self.wave_amplitude = info.wave_amplitude
        self.wave_ptd_sd_ratio = info.wave_ptd_sd_ratio
        self.wave_ptd_pre_post_ratio = info.wave_ptd_pre_post_ratio

        
    def calc_response_to_ftc(self, t_stim, t_start=-.1, t_end=.2):
        window = .1 # window for significance test: 100ms before and after stim onset
        t_spk = self.spiketimes['tone_ftc']
        raster = SingleUnit.get_raster(t_spk, t_stim, t_start, t_end)
        # significance test for evoked and baseline activity
        n_baseline = [sum((spktimes < 0) & (spktimes > -window)) for spktimes in raster]
        n_stim = [sum((spktimes > 0) & (spktimes < window)) for spktimes in raster]
        _, p = stats.wilcoxon(n_baseline, n_stim)
        self.ftc = {'p': p,
                    'fr_ratio': np.mean(n_stim) / np.mean(n_baseline),
                    'raster': raster,
                    't_start': t_start,
                    't_end': t_end}
    
    
    def plot_response_to_ftc(self, axes, freqs):
        unique_freq = np.unique(freqs)
        t_start = self.ftc['t_start']
        t_end= self.ftc['t_end']
        fr_max = 0
        for i, freq in enumerate(unique_freq):
            raster = [self.ftc['raster'][x] for x in range(len(freqs)) if freqs[x] == freq]
            axes[0, i].clear()
            raster_mat, binsize = SingleUnit.plot_ftc_raster(axes[0, i], raster, t_start, t_end)
            axes[0, i].set_title('{:.1f}kHz'.format(freq/1e3))
            axes[1, i].clear()
            fr_max_new = SingleUnit.plot_ftc_psth(axes[1, i], raster_mat, t_start, t_end, binsize)
            fr_max = max(fr_max, fr_max_new)
        
        fr_max = max(100, fr_max)
        fr_max = np.ceil(fr_max / 20) * 20
        for i in range(len(unique_freq)):
            axes[1, i].plot([0, 0], [0, fr_max], 'grey', linewidth=.5)
            axes[1, i].set_ylim([0, fr_max])
            if i > 0:
                axes[0, i].set_yticklabels([])
                axes[1, i].set_yticklabels([])
        axes[0, 0].set_ylabel('Trial')
        axes[1, 0].set_ylabel('Firirng Rate (Hz)')
        
    @classmethod 
    def plot_ftc_raster(cls, ax, raster, t_start, t_end, binsize=.01):
        n_trials = len(raster)
        bins = np.arange(t_start, t_end, binsize)
        raster_mat = np.zeros([n_trials, len(bins)-1])
        for i, tspk in enumerate(raster):
            nspk, _ = np.histogram(tspk, bins)
            raster_mat[i] = nspk 
        ax.imshow(raster_mat, extent=[t_start, t_end, 1, n_trials], 
                  aspect='auto', cmap='gray_r')
        ax.set_xlim([t_start, t_end])
        ax.set_ylim([0, n_trials])
        ax.set_xticks(np.arange(t_start, t_end, .1))
        ax.set_xticklabels([])
        ax.set_yticks(range(20, n_trials+1, 20))
        return raster_mat, binsize
    
    @classmethod 
    def plot_ftc_psth(cls, ax, raster_mat, t_start, t_end, binsize):
        n_trials = raster_mat.shape[0]
        raster_mat = raster_mat / binsize
        fr = np.mean(raster_mat, axis=0)
        fr_sem = np.std(raster_mat, axis=0) / np.sqrt(n_trials)
        x = np.arange(t_start, t_end, binsize)
        x = (x[1:] + x[:-1]) / 2
        ax.fill_between(x, fr-fr_sem, fr+fr_sem, color='lightgrey')
        ax.plot(x, fr, 'k')
        
        ax.set_xlim([t_start, t_end])
        ax.set_xticks(np.arange(t_start, t_end-.01, .1))
        ax.set_xlabel('Time (s)')
        ax.spines[['right', 'top']].set_visible(False)
        return max(fr)
        
    @classmethod
    def get_raster(cls, tspk, tstim, tstart:float, tend:float):
        """
        get raster aligned to tstim

        Parameters
        ----------
        tspk :   time of spikes
        tstim :  time of stimulus
        tstart : time before stimulus onset
        tend :   time after stimulus onset

        Returns
        -------
        raster: liast of spike times aligned to stimulus onset
        """
        raster = []
        for tx in tstim:
            raster.append(tspk[(tspk > tx + tstart) & (tspk < tx + tend)] - tx)
        return raster
