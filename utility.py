"""
All functions in this file were copied from previous assignment files.
each file described in each sections below

"""


# -*- coding: utf-8 -*-
"""
import_ssvep_data.py

Load data and plot frequency specturm of steady-state visual evoked potentials (SSVEPs).
BME6710 BCI Spring 2024 Lab #3

The SSVEP dataset is derived from a tutorial in the MNE-Python package.The dataset
includes electropencephalograpy (EEG) data from 32 channels collected during
a visual checkerboard experiment, where the checkerboard flickered at 12 Hz or
15 Hz. 

The functions in this module can be used to load the dataset into variables, 
plot the raw data, epoch the data, calculate the Fourier Transform (FT), 
and plot the power spectra for each subject. 

Created on Feb 24 2024

@author: 
    Ardyn Olszko
    Yoonki Hong
"""

#%% Part 1: Load the Data

# import packages

import numpy as np
from matplotlib import pyplot as plt
import scipy.fft 
import os

# function to load data
def load_ssvep_data(subject, data_directory):
    '''
    Load the SSVEP EEG data for a given subject.
    The format of the data is described in the README.md file with the data.

    Parameters
    ----------
    subject : int
        Subject number, 1 or 2.
    data_directory : str
        Path to the folder where the data files exist.

    Returns
    -------
    data_dict : dict
        Dictionary of data for a subject.

    '''
    
    # Load dictionary
    data_dict = np.load(data_directory + f'/SSVEP_S{subject}.npz',allow_pickle=True)    

    return data_dict

#%% Part 2: Plot the Data

# function to plot the raw data
def plot_raw_data(data,subject,channels_to_plot):
    '''
    Plot events and raw data from specified electrodes.
    Creates a figure with two subplots, where the first is the events and the
    second is the EEG voltage for specified channels. The figure is saved to
    the current directory.

    Parameters
    ----------
    data : dict
        Dictionary of data for a subject.
    subject : int
        Subject number, 1 or 2 (used to annotate plot)
    channels_to_plot : list or array of size n where n is the number of channels
        Channel names of data to plot. Channel name must be in "data['channels']".

    Returns
    -------
    None.

    '''
    
    # extract variables from dictionary
    eeg = data['eeg'] # eeg data in Volts. Each row is a channel and each column is a sample.
    channels = data['channels'] # name of each channel, in the same order as the eeg matrix.
    fs = data['fs'] # sampling frequency in Hz.
    event_samples = data['event_samples'] # sample when each event occurred.
    event_durations = data['event_durations'] # durations of each event in samples.
    event_types = data['event_types'] # frequency of flickering checkerboard for each event.
    
    # calculate time array
    time = np.arange(0,1/fs*eeg.shape[1],1/fs)
    
    # set up figure
    plt.figure(f'raw subject{subject}',clear=True)
    plt.suptitle(f'SSVEP subject {subject} raw data')
    
    # plot the event start and end times and types
    ax1 = plt.subplot(2,1,1)
    start_times = time[event_samples]
    end_times = time[event_samples+event_durations.astype(int)]
    for event_type in np.unique(event_types):
        is_event = event_types == event_type
        plt.plot()
        event_data = np.array([start_times[is_event],end_times[is_event]])
        plt.plot(event_data,
                 np.full_like(event_data,float(event_type[:-2])),
                 marker='o',linestyle='-',color='b',
                 label=event_type)
    plt.xlabel('time (s)')
    plt.ylabel('flash frequency (Hz)')
    plt.grid()
    
    # plot the raw data from the channels spcified
    plt.subplot(2,1,2, sharex=ax1)
    for channel in channels_to_plot:
        is_channel = channels == channel
        plt.plot(time, 10e5*eeg[is_channel,:].transpose(),label=channel) # multiply by 10e5 to convert to uV (confirmed that this matches the original dataset from mne)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (uV)')
    plt.grid()
    plt.legend()
    
    # save the figure
    plt.tight_layout()
    
    if not os.path.exists('plots/SSVEP'):
        os.makedirs('plots/SSVEP')
    plt.savefig(f'plots/SSVEP/SSVEP_S{subject}_rawdata.png')
    
    return


#%% Part 3: Extract the Epochs

# function to epoch the data
def epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20):
    '''
    Epoch the EEG data.

    Parameters
    ----------
    data_dict : dict
        Dictionary of data for a subject.
    epoch_start_time : int or float, optional
        Start time of the epoch relative the the event time. Units in seconds. The default is 0.
    epoch_end_time : int or float, optional
        End time of the epoch relative the the event time. Units in seconds. The default is 20.

    Returns
    -------
    eeg_epochs : array of float, size M x N x T where M is the number of trials,
    N is the number of channels, and T is the number of samples in the epoch
        EEG data after each epoch. Units in uV.
    epoch_times : array of float, size T where T is the number of samples
        Time (relative to the event) of each sample. Units in seconds.
    is_trial_15Hz : array of bool, size M where M is the number of trials
        Event label, where True is 15 Hz event and False is 12 Hz event.

    '''
    
    # extract relevant variables from dictionary
    eeg_raw = data_dict['eeg'] # eeg data in Volts. Each row is a channel and each column is a sample. (not sure data are actually in Volts)
    fs = data_dict['fs'] # sampling frequency in Hz.
    event_samples = data_dict['event_samples'] # sample when each event occurred.
    event_types = data_dict['event_types'] # frequency of flickering checkerboard for each event. 

    # convert eeg data to uV
    eeg = 10e5*eeg_raw # multiply by 10e5 to convert to uV (confirmed that this matches the original dataset from mne)
    
    # define boolean for event types
    is_trial_15Hz = event_types == '15hz'
    
    # create time array for each epoch
    epoch_times = np.arange(epoch_start_time,epoch_end_time,1/fs)
    
    # define size of 3d array of epochs
    epoch_count = len(event_samples) 
    sample_count = len(epoch_times)
    channel_count = eeg.shape[0]
       
    # calculate end samples for all epochs
    start_samples = event_samples + int(epoch_start_time*fs)
    end_samples = event_samples + int(epoch_end_time*fs)
    
    # adjust any start and end samples that occur outside the times of the available data
    start_samples[start_samples<0] = 0
    end_samples[end_samples>eeg.shape[1]]=eeg.shape[1]
    
    # define 3d array for epoch data
    eeg_epochs = np.full((epoch_count,channel_count,sample_count),np.nan,dtype='float32')
    
    # fill in the epoch data
    for epoch_index in np.arange(epoch_count):
        eeg_epochs[epoch_index] = eeg[:, start_samples[epoch_index]:end_samples[epoch_index]]
    
    return eeg_epochs, epoch_times, is_trial_15Hz

#%% Part 4: Take the Fourier Transform

# function to calculate frequency spectrum
def get_frequency_spectrum(eeg_epochs, fs):
    '''
    Calcluate the FT each channel in each epoch.

    Parameters
    ----------
    eeg_epochs : array of float, size M x N x T where M is the number of trials,
    N is the number of channels, and T is the number of samples in the epoch
        EEG data after each epoch. Units in uV.
    fs : int or float
        Sampling frequency of the EEG data.

    Returns
    -------
    eeg_epochs_fft : array of float, size M x N X F where M is the number of trials,
    N is the number of channels, and F is number of frequencies measured in the epoch
        FT frequency content of each channel in each epoch .
    fft_frequencies : array of float, size F
        Frequencies measured, where the maximum frequency measured is 1/2*fs.

    '''
    
    # calculate FT on each channel
    eeg_epochs_fft = scipy.fft.rfft(eeg_epochs)
    
    # calculate frequencies
    sample_count = eeg_epochs.shape[2]
    total_duration = sample_count/fs
    fft_frequencies = np.arange(0,eeg_epochs_fft.shape[2])/total_duration
    
    return eeg_epochs_fft, fft_frequencies

#%% Part 5: Plot the Power Spectra

# function to plot the mean power spectra for specified channesl
def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject):
    '''
    Calculate and plot the mean power spectra for specified channels.
    Each channel is plotted on a separate subplot. Event types, 12 Hz and 15 Hz,
    are plotted separately for each channel.

    Parameters
    ----------
    eeg_epochs_fft : array of float, size M x N X F where M is the number of trials,
    N is the number of channels, and F is number of frequencies measured in the epoch
        FT frequency content of each channel in each epoch.
    fft_frequencies : array of float, size F
        Frequencies measured, where the maximum frequency measured is 1/2*fs.
    is_trial_15Hz : array of bool, size M where M is the number of trials
        Event label, where True is 15 Hz event and False is 12 Hz event.
    channels : list of size N where N is the number of channels
        Channel names available in  the original dataset.
    channels_to_plot : list or array of size n where n is the number of channels
        Channel names of data to plot. Channel name must be in "data['channels']".
    subject : int
        Subject number, 1 or 2 (used to annotate plot)
        
    Returns
    -------
    spectrum_db_12Hz : array of float, size n x F where n is the number of channels
    and F is the number of frequencies
        Mean power spectrum of 12 Hz trials. Units in dB.
    spectrum_db_15Hz : array of float, size n x F where n is the number of channels
    and F is the number of frequencies
        Mean power spectrum of 15 Hz trials. Units in dB.

    '''
    
    # calculate mean power spectra for each channel
    signal_power = abs(eeg_epochs_fft)**2 # calculate power by squaring absolute value
    # calculate mean across trials
    power_12Hz = np.mean(signal_power[~is_trial_15Hz],axis=0)
    power_15Hz = np.mean(signal_power[is_trial_15Hz],axis=0)
    
    # normalize (divide by max value)
    norm_power_12Hz = power_12Hz/np.reshape(np.max(power_12Hz, axis=1), (power_12Hz.shape[0],1))
    norm_power_15Hz = power_15Hz/np.reshape(np.max(power_15Hz, axis=1), (power_12Hz.shape[0],1))
    
    # convert to decibel units
    power_db_12Hz = 10*np.log10(norm_power_12Hz)
    power_db_15Hz = 10*np.log10(norm_power_15Hz)
    
    # set up figure and arrays for mean power spectra
    channel_count = len(channels_to_plot)
    freq_count = len(fft_frequencies)
    spectrum_db_12Hz = np.full([channel_count,freq_count],np.nan,dtype=float) # set up arrays to store power spectrum
    spectrum_db_15Hz = np.full_like(spectrum_db_12Hz,np.nan)
    row_count = int(np.ceil(np.sqrt(channel_count))) # calculate number of rows of subplots
    if (row_count**2 - channel_count) >= row_count: # calculate number of columns of subplots
        col_count = row_count-1 
    else:
        col_count = row_count

    fig = plt.figure(f'spectrum subject{subject}',clear=True,figsize=(6+0.5*channel_count,6+0.5*channel_count))
    plt.suptitle(f'Frequency content for SSVEP subject {subject}')
    axs=[] # set up empty list for subplot axes
    
    # plot and extract data for specified channels
    for channel_index, channel in enumerate(channels_to_plot):
        is_channel = channels == channel
        spectrum_db_12Hz[channel_index,:] = power_db_12Hz[is_channel,:]
        spectrum_db_15Hz[channel_index,:] = power_db_15Hz[is_channel,:]
        
        if channel_index == 0: 
            axs.append(fig.add_subplot(row_count,col_count,channel_index+1))
        else:
            axs.append(fig.add_subplot(row_count,col_count,channel_index+1,
                                       sharex=axs[0],
                                       sharey=axs[0]))
        # plot the mean power spectra
        axs[channel_index].plot(fft_frequencies,spectrum_db_12Hz[channel_index,:],label='12Hz',color='r')
        axs[channel_index].plot(fft_frequencies,spectrum_db_15Hz[channel_index,:],label='15Hz',color='g')
        # plot corresponding frequencies
        axs[channel_index].axvline(12,color='r',linestyle=':')
        axs[channel_index].axvline(15,color='g',linestyle=':')
        # annotate
        axs[channel_index].set_title(channel)
        axs[channel_index].set(xlabel='frequency (Hz)',ylabel='power (db)')
        axs[channel_index].grid()
        axs[channel_index].legend()
           
    plt.tight_layout()
    
    return spectrum_db_12Hz, spectrum_db_15Hz





"""
Lab 4: Filtering

filter_ssvep_data.py

Functions in this module are used to create filters and apply them to raw EEG data, and
plot poewr spectra of raw, filtered, and enveloped data.

@ author:
    Yoonki Hong
    Lexi Reinsborough
"""


# %% Part 2: Design a Filter

import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
import scipy.fft

# function to create a filter
firwin=scipy.signal.firwin
# function to get frequency response
freqz=scipy.signal.freqz

#upfirdn= scipy.signal.upfirdn

def make_bandpass_filter(low_cutoff, high_cutoff, filter_type="hann", filter_order=10, fs=1000):
    """
    Parameters
    ----------
    low_cutoff : (float)
        the lower cutoff frequency (in Hz)
    high_cutoff : (float)
        the higher cutoff frequency (in Hz)
    filter_type : (str), optional
        filter type. The default is "hann".
    filter_order : (int), optional
        the filter order. The default is 10.
    fs : (float), optional
        the sampling frequency. The default is 1000.

    Returns
    -------
    filter_coefficients : (ndarray), size - (filter_order+1)
        1d array that contains coefficients of the filter
        
    This function creates a filter, an array of coefficients, with given arguments
    """
    # create a filter (coefficients) using firwin function
    filter_coefficients= firwin(filter_order+1, [low_cutoff, high_cutoff], window=filter_type, fs=fs, pass_zero='bandpass')
    
    # plot impulse response and frequency response
    fig, ax = plt.subplots(2,1)  
    
    # plot impulse response (subplot)
    impulse_response_plot = ax[0]
    
    x= np.arange(filter_coefficients.shape[0])/fs        # get x range for impulse response plot
    impulse_response_plot.plot(x,filter_coefficients)    # plot impulse response
    impulse_response_plot.grid()                         # create grid
    impulse_response_plot.set_xlabel('time (s)')         # x label
    impulse_response_plot.set_ylabel('gain')             # y label
    impulse_response_plot.set_title('impulse response')  # title

    # plot frequency response (subplot)
    frequency_response_plot = ax[1]
    
    h=filter_coefficients
    # get freqeuncy in rad and freqeuncy response
    w, H = freqz(h)
    
    # convert rad to frequency and decibel
    frequencies = w * fs / (2 * np.pi)
    decibel = 20 * np.log10(np.abs(H))
    
       
    frequency_response_plot.plot(frequencies, decibel)       # plot frequency response
    frequency_response_plot.grid()                           # create grid
    frequency_response_plot.set_title('frequency response')  # title
    frequency_response_plot.set_xlabel('frequency (Hz)')     # x label 
    frequency_response_plot.set_ylabel('amplitude (dB)')     # y label 
    

    # set sup title and save the plot to a file
    plt.suptitle(f'bandpass {filter_type} with fc=[{low_cutoff}, {high_cutoff}], order={filter_order}')
    plt.tight_layout()
    
    if not os.path.exists('plots/filter'):
        os.makedirs('plots/filter')
    plt.savefig(f'plots/filter/{filter_type}_filter_{low_cutoff}_{high_cutoff}Hz_order{filter_order}.png')
    #plt.show()
    plt.close()
    
    
    return filter_coefficients # return the filter
    
    


# %% Part 3: Filter the EEG Signals

# filtfilt function applys the given filter to the given data twice in forward and backward
filtfilt = scipy.signal.filtfilt

def filter_data(data, b):
    """
    c - number of channels
    s - number of samples of an experiment
    o - filter order
    
    Parameters
    ----------
    data : (ndarray), dimension - c x s
        raw EEG data
    b : (ndarray), size - o+1
        filter (coefficients)

    Returns
    -------
    filtered_data : (ndarray), dimension - c x s
        filtered data
    
    apply the given filter to the data
    """
    
    filtered_data = filtfilt(b, 1 , data)
    return filtered_data


# %% Part 4: Calculate the Envelope

hilbert=scipy.signal.hilbert

def get_envelope(data, filtered_data, channel_to_plot=None, ssvep_frequency=None):
    """
    c - number of channels
    s - number of samples of an experiment

    Parameters
    ----------
    data : (dictionary or NpzFile)
        a dictionary or NpzFilie object that contains raw EEG data and info about the data.
    filtered_data : (ndarray), dimension - c x s
        filtered EEG data
    channel_to_plot : (str), optional
        channel name to plot. The default is None.
    ssvep_frequency : flot, optional
        the SSVEP frequency being isolated.
        used to title the plot. The default is None.

    Returns
    -------
    envelope : (ndarray), dimension - c x s
        Envelope of filtered data
        
    Apply Hilbert transformation to the the filtered data
    """
    
    # get envelope
    envelope = np.abs(hilbert(filtered_data))

    
    # plot if channel_to_plot is not None
    if channel_to_plot is not None:
        # get index of the channel to plot
        channel_names= data['channels']
        channel_index=np.where(channel_names==channel_to_plot)[0][0]
        
        # sampling frequncy and filtered_data to use to get time array
        fs = data['fs']   
        eeg=filtered_data 

        # calculate time array
        time = np.arange(0,1/fs*eeg.shape[1],1/fs)
        
        # plot the envelope
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # grid
        ax.grid()
        # labels
        ax.set_ylabel('Voltage (uV)')            
        ax.set_xlabel('time (s)')
        # plot original signal and the envelope
        ax.plot(time, filtered_data[channel_index], label='Filtered Signal')
        ax.plot(time, envelope[channel_index], label='Envelope')
        # set title
        title = (f'{ssvep_frequency}Hz' if ssvep_frequency is not None else 'Unknown frequency')
        title += f' BPF Data for channel {channel_to_plot}'
        fig.suptitle(title)
        fig.legend()

        plt.show()
    return envelope
    

# %% Part 5: Plot the Amplitudes

def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot, ssvep_freq_a, ssvep_freq_b, subject):
    """
    c - number of channels
    s - number of samples of an experiment    
    
    Parameters
    ----------
    data : (dictionary or NpzFile)
        a dictionary or NpzFilie object that contains raw EEG data and info about the data.
    envelope_a : (ndarray), dimension - c x s
        envelope data of EEG data that was filtered with a ssvep_freq_a bandpass filter
    envelope_b : (ndarray), dimension - c x s
        envelope data of EEG data that was filtered with a ssvep_freq_a bandpass filter
    channel_to_plot : (str)
        channel name to plot.
    ssvep_freq_a : (float)
        the SSVEP frequency being isolated in the first envelope
    ssvep_freq_b : (float)
        the SSVEP frequency being isolated in the second envelope
    subject : (int)
        the subject number.

    Returns
    -------
    None.

    plot events and given envelope data
    """
    
    # get event start time and end time
    event_samples= data['event_samples']
    event_durations= data['event_durations']
    event_types= data['event_types'] # whether 12Hz or 15Hz
    
    # sampling frequency and raw EEG data
    fs = data['fs']
    eeg=data['eeg']
    
    # calculate time array
    time = np.arange(0,1/fs*eeg.shape[1],1/fs)
    
    
    # get indices of event start and end times
    event_start_idxs = event_samples
    event_end_idxs   = event_samples+event_durations.astype(int)
    
    event_start_times = time[event_start_idxs]
    event_end_times   = time[event_end_idxs]
    
    
    
    # plot 2 x 1 subplots
    fig, ax = plt.subplots(2,1, sharex=True)
    
    # plot events in the first subplot
    event_plot= ax[0]
    
    # set the labels
    event_plot.set_yticks([ssvep_freq_a,ssvep_freq_b])
    event_plot.set_yticklabels([f'{ssvep_freq_a}hz',f'{ssvep_freq_b}hz'])
    event_plot.set_xlabel('times (s)')
    event_plot.set_ylabel('Flash frequency')
    event_plot.set_title(f'Subject {subject} SSVEP Amplitudes')
    event_plot.grid()
    
    # get the events for the first frequency and plot
    freq_a_idxs= np.where(event_types==f'{ssvep_freq_a}hz')[0]
    freq_a_start_times = event_start_times[freq_a_idxs]
    freq_a_end_times = event_end_times[freq_a_idxs]
    freq_a_event_count= freq_a_idxs.shape[0]
    # plot the events
    for event_idx in range(freq_a_event_count):
        event_plot.plot( [freq_a_start_times[event_idx], freq_a_end_times[event_idx]], [ssvep_freq_a,ssvep_freq_a], color='tab:blue', markersize=2, marker='o' )
    
    # get the events for the second frequency and plot
    freq_b_idxs= np.where(event_types==f'{ssvep_freq_b}hz')[0]
    freq_b_start_times = event_start_times[freq_b_idxs]
    freq_b_end_times = event_end_times[freq_b_idxs]
    freq_b_event_count= freq_b_idxs.shape[0]
    # plot the events
    for event_idx in range(freq_b_event_count):
        event_plot.plot( [freq_b_start_times[event_idx], freq_b_end_times[event_idx]], [ssvep_freq_b,ssvep_freq_b], color='tab:orange', markersize=2, marker='o' )
    
    
    
    # plot envelope in the second subplot
    envelope_plot = ax[1]
    
    # index of the channel to plot
    channel_idx = np.where(data['channels']==channel_to_plot)[0][0]
    
    # get the evenlope of the channel
    envelope_a_ch=envelope_a[channel_idx]
    envelope_b_ch=envelope_b[channel_idx]
    
    # set the labels
    envelope_plot.set_xlabel('times (s)')
    envelope_plot.set_ylabel('Voltage (uV)')
    envelope_plot.set_title('Envelope Comparison')
    envelope_plot.grid()
    
    # plot envelope
    envelope_plot.plot(time, envelope_a_ch, label=f'{ssvep_freq_a}Hz Envelope')
    envelope_plot.plot(time, envelope_b_ch, label=f'{ssvep_freq_b}Hz Envelope')
    envelope_plot.legend()
    
    plt.tight_layout()
    
    
    
# %% Part 6: Examine the Spectra

#from import_ssvep_data import epoch_ssvep_data, get_frequency_spectrum
import matplotlib.gridspec as gridspec



def get_power_spectra(data_dict, channel_idxs_to_plot, fs):
    """
    cp - number of channels to plot
    s - number of samples of an experiment
    Parameters
    ----------
    data : (dictionary or NpzFile)
        a dictionary or NpzFilie object that contains raw EEG data and info about the data.
        this raw EEG data only contains channels to plot.
    channel_idxs_to_plot : (ndarray), size - cp
        indices of channels to plot.
    fs : (float)
        sampling frequency

    Returns
    -------
    fft_frequencies : (ndarray), size - s/2+1
        Frequencies. Maximum frequency is fs/2
    power_db_12Hz : (ndarray), dimensino - (number of channels to plot) x s
        power spectra of 12 Hz trials
    power_db_15Hz : (ndarray), dimensino - (number of channels to plot) x s
        power spectra of 15 Hz trials
        
    Helper function to get power spectra of given data.
    This function is modied and combined functions from Lab 3.
    This function is not instructed to include in this file.
    """
    
    # get epochs and frequencies
    eeg_epochs, epoch_times, is_trial_15Hz = epoch_ssvep_data(data_dict)
    eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)
    eeg_epochs_fft_to_plot = eeg_epochs_fft[:, channel_idxs_to_plot, :]
    
    # calculate mean power spectra for each channel
    signal_power = abs(eeg_epochs_fft_to_plot)**2 # calculate power by squaring absolute value
    # calculate mean across trials
    power_12Hz = np.mean(signal_power[~is_trial_15Hz],axis=0)
    power_15Hz = np.mean(signal_power[is_trial_15Hz],axis=0)
    
    # normalize (divide by max value)
    norm_power_12Hz = power_12Hz/np.reshape(np.max(power_12Hz, axis=1), (power_12Hz.shape[0],1))
    norm_power_15Hz = power_15Hz/np.reshape(np.max(power_15Hz, axis=1), (power_12Hz.shape[0],1))
    
    # convert to decibel units
    power_db_12Hz = 10*np.log10(norm_power_12Hz)
    power_db_15Hz = 10*np.log10(norm_power_15Hz)
    
    return fft_frequencies, power_db_12Hz, power_db_15Hz




def plot_filtered_spectra(data, filtered_data, envelope, channels):
    """
    c - nubmer of channels to plot
    s - number of samples of an experiment

    Parameters
    ----------
    data : (dictionary or NpzFile)
        a dictionary or NpzFilie object that contains raw EEG data and info about the data.
    filtered_data : (ndarray), dimension - c x s
        envelope data of EEG data that was filtered with a ssvep_freq_a bandpass filter
    envelope : (ndarray), dimension - c x s
        envelope data
    channels : (list), size - number of channels to plot
        channel names to plot

    Returns
    -------
    None.
    
    get power spectra of raw data, filtered data, and envelope, and plot them
    """
    
    # retrive data from the dictionary
    #eeg = data['eeg']
    fs = data['fs']
    eeg_channels = data['channels']
    
    # create dictionaries for filtered and enveloped data to be passed into the helper function
    data_filtered={}
    data_envelope={}
    
    for key in data:
        if key=='eeg':
            data_filtered[key] = filtered_data
            data_envelope[key] = envelope
        else:
            data_filtered[key] = data[key]
            data_envelope[key] = data[key]
            
    
    
    
     
    # get channel indices to plot
    channel_idxs_to_plot = np.where(eeg_channels == np.array(channels)[..., None])[1]

    # get power spectra of raw data for 12Hz events and 15Hz events    
    fft_frequencies, power_db_12Hz_raw, power_db_15Hz_raw = get_power_spectra(data, channel_idxs_to_plot, fs)
    
    # get power spectra of filtered data for 12Hz events and 15Hz events
    _, power_db_12Hz_filtered, power_db_15Hz_filtered = get_power_spectra(data_filtered,  channel_idxs_to_plot, fs)
    
    # get power spectra of envelope for 12Hz events and 15Hz events
    _, power_db_12Hz_envelope, power_db_15Hz_envelope = get_power_spectra(data_envelope,  channel_idxs_to_plot, fs)
    
        
    # plot
    channel_count= channel_idxs_to_plot.shape[0]    # get the number of channels to plot
    
    fig = plt.figure(figsize=(15, channel_count*3)) # create a fig
    
    
    # create a number of channels x 1 subplots
    outer_frame= gridspec.GridSpec(channel_count, 1)# outer frame to subplots
    
    # store the first subplot. Other subplots share axes with this axis.
    axis_first=[] 
    
    # plot power spectra for each channel
    for channel in range(channel_count):
        # in each subplot, create 1 x 3 sub-subplots
        inner_frame = gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec=outer_frame[channel])
        
        # get name of the channel to plot and set the title for this channel
        channel_name = channels[channel]
        channel_subplot = fig.add_subplot(inner_frame[:])
        channel_subplot.set_title(f'\n{channel_name} - power spectra\n\n')
        channel_subplot.axis('off')
        
        # power spectra of 12Hz events of this channel
        raw_12Hz_events = power_db_12Hz_raw[channel]
        filtered_12Hz_events = power_db_12Hz_filtered[channel]
        envelope_12Hz_events = power_db_12Hz_envelope[channel]
        
        # power spectra of 15Hz events of this channel
        raw_15Hz_events = power_db_15Hz_raw[channel]
        filtered_15Hz_events = power_db_15Hz_filtered[channel]
        envelope_15Hz_events = power_db_15Hz_envelope[channel]
        
        # raw sub-subplot
        # keep the first raw subplot to share its axes
        if channel==0:
            raw_subplot = plt.Subplot(fig, inner_frame[0])
            axis_first.append(raw_subplot)
        else:
            raw_subplot = plt.Subplot(fig, inner_frame[0], sharex= axis_first[0], sharey= axis_first[0])
        
        raw_subplot.set_title('Frequency content for raw data')
        raw_subplot.set_xlabel('frequency (Hz)')
        raw_subplot.set_ylabel('power (dB)')
        raw_subplot.grid()
        raw_subplot.axvline(12,color='r',linestyle=':')
        raw_subplot.axvline(15,color='g',linestyle=':')
        raw_subplot.plot(fft_frequencies, raw_12Hz_events, color='red', label='12Hz')
        raw_subplot.plot(fft_frequencies, raw_15Hz_events, color = 'green', label='15Hz')
        raw_subplot.legend()
        fig.add_subplot(raw_subplot)
        
        # filtered sub-subplot
        filtered_subplot = plt.Subplot(fig, inner_frame[1], sharex=raw_subplot, sharey=raw_subplot)
        filtered_subplot.set_title('Frequency content for filtered data')
        filtered_subplot.set_xlabel('frequency (Hz)')
        filtered_subplot.grid()
        filtered_subplot.axvline(12,color='r',linestyle=':')
        filtered_subplot.axvline(15,color='g',linestyle=':')
        filtered_subplot.plot(fft_frequencies, filtered_12Hz_events, color='red', label='12Hz')
        filtered_subplot.plot(fft_frequencies, filtered_15Hz_events, color='green', label='15Hz')
        filtered_subplot.legend()
        fig.add_subplot(filtered_subplot)
        
        # envelope sub-subplot
        envelope_subplot = plt.Subplot(fig, inner_frame[2], sharex=raw_subplot, sharey=raw_subplot)
        envelope_subplot.set_title('Frequency content for enveloped data')
        envelope_subplot.set_xlabel('frequency (Hz)')
        envelope_subplot.grid()
        envelope_subplot.axvline(12,color='r',linestyle=':')
        envelope_subplot.axvline(15,color='g',linestyle=':')
        envelope_subplot.plot(fft_frequencies, envelope_12Hz_events, color='red', label='12Hz')
        envelope_subplot.plot(fft_frequencies, envelope_15Hz_events, color='green', label='15Hz')
        envelope_subplot.legend()
        fig.add_subplot(envelope_subplot)
    
    
        
    plt.tight_layout()
    
    channels_str=''
    for ch in channels:
        channels_str+= ch
        channels_str+= '_'
    channels_str=channels_str[:-1]
    plt.savefig(f'power_spectra_{channels_str}.png')
    fig.show()
    
        
        
        
        
        
        
# %%
      
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_topo.py

Created on Mon Sep  6 13:06:10 2021

@author: djangraw
"""

# Import packages
#import numpy as np
#from matplotlib import pyplot as plt
import mne


# Declare main function
# axes argument added in Lab 5
def plot_topo(axes=None, channel_names=[], channel_data=[],title='',cbar_label='Voltage (uV)',montage_name='biosemi64'):  
    """
    Plots a topomap (colored brain) of the specified channels and values.

    Parameters
    ----------
    axes : added in Lab 5
    channel_names : list/arr of strings, optional
        Channels to plot (must be . The default is [].
    channel_data : Arr of shape [len(channel_names),1], optional
        Voltages to plot on each channel. The default is [].
    title : str, optional
        Title to place above the plot. The default is ''.
    cbar_label : str, optional
        Label to go on the colorbar. The default is 'Voltage (uV)'.
    montage_name : str, optional
        Name of the channel montage to use (must be valid input to 
        mne.channels.make_standard_montage). The default is 'biosemi64'.
    
    Returns
    -------
    im : image
        Topoplot image object.
    cbar : colorbar
        Colorbar object associated with the image.

    """

    # create montage according to montage_name specs
    montage = mne.channels.make_standard_montage(montage_name)
    if len(channel_names)==0: # if no channel names were given
        channel_names = montage.ch_names # plot all by default
    n_channels = len(channel_names)
    # Create MNE info struct
    fake_info = mne.create_info(ch_names=channel_names, sfreq=250.,
                                ch_types='eeg')
    
    # Prepare data
    if len(channel_data)==0: # if no input was given
        channel_data = np.random.normal(size=(n_channels, 1)) # plot random data by default
    if channel_data.ndim==1: # if it's a 1D array
        channel_data = channel_data.reshape([-1,1]) # make it 2D
    
    # Create MNE evoked array object with our data & channel info
    fake_evoked = mne.EvokedArray(channel_data, fake_info)
    fake_evoked.set_montage(montage) # set montage (channel locations)

    # Clear current axes
    #plt.cla()    # commented in Lab 5
    # Plot topomap on current axes    
    im,_ = mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info,show=False, axes=axes)
    # Annotate plot

    plt.title(title) # commented in Lab 5
    
    #### added in Lab 5 #######
    if axes==None:
        plt.title(title) 
    else:
        axes.set_title(title) # added in Lab 5
    ###########################

    cbar = plt.colorbar(im,label=cbar_label)
    
    # return image and colorbar objects
    return im


# Helper and QA functions
def get_channel_names(montage_name='biosemi64'):
    """
    Returns all the channels contained in a given montage. Useful for checking 
    capitalization conventions and subsets of channels found in a given montage.

    Parameters
    ----------
    montage_name : str, optional
        Name of the channel montage to use (must be valid input to 
        mne.channels.make_standard_montage). The default is 'biosemi64'.

    Returns
    -------
    arr of strings
        names of channels in the given montage.

    """
    # create montage
    montage = mne.channels.make_standard_montage(montage_name)
    # return channel names in that montage
    return montage.ch_names



# %% Part 1: Load the Data


#import numpy as np
#from matplotlib import pyplot as plt
#from plot_topo import plot_topo 


def load_data(data_directory, channels_to_plot):
    '''
    Load subject data from the folder specified and plot if any channel name is provided.
    
    Parameters
    ----------
    data_directory (str) : Path to the folder where the data files exist.
    channels_to_plot (list or array) : list of channels to plot
        

    Returns
    -------
    data (dictionary) : python dictionary that contains data of a subject

    '''
    
    
    # load data
    filename='AudVisData.npy'    
    filepath=data_directory+filename
    data=np.load(filepath, allow_pickle=True).item()
    
    # channels
    all_channels=data['channels']
    
    
    
    # number of channels in chanels_to_plot
    channels_to_plot=np.array(channels_to_plot)
    ch_to_plot_count=channels_to_plot.shape[0]
    
    if ch_to_plot_count>0:
        
        
        
        # channel indices
        ch_idxs_to_plot= np.where(all_channels==channels_to_plot[...,None])[1]
        ch_to_plot_count=ch_idxs_to_plot.shape[0]
        
        if ch_to_plot_count==0:
            return data
        
        
        # eeg
        eeg = data['eeg']
        
        # plot
        fig, axes = plt.subplots(ch_to_plot_count, 1, sharex=True, sharey=True, figsize=(8, 2.5*ch_to_plot_count))
        
        # prepare time (x axis)
        sample_count = eeg.shape[1]
        fs = data['fs']
        times = np.arange(sample_count)
        times = times/fs
        
        # when there is only one channel to plot
        if ch_to_plot_count==1:
            axes.plot(times, eeg[ch_idxs_to_plot[0]])
            axes.set_xlabel('times (s)')
            axes.set_ylabel(f'Voltage on {all_channels[ch_idxs_to_plot[0]]} (uV)')
        # when there are more than just one
        else:
            for idx in range(ch_to_plot_count):
                axes[idx].plot(times, eeg[ch_idxs_to_plot[idx]])
                axes[idx].set_xlabel('times (s)')
                axes[idx].set_ylabel(f'Voltage on {all_channels[ch_idxs_to_plot[idx]]} (uV)')
                        
        fig.suptitle('Raw AudVis EEG Data')
        plt.tight_layout()
        plt.savefig("plots/raw_audio_vis")
        plt.close()            
    
    return data
        

# %% Part 2: Plot the Components

def plot_components(mixing_matrix, channels, subject, series, close=False ,components_to_plot=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    '''
    Plot topological maps of components specified by components_to_plot list
    
    Parameters
    ----------
    mixing_matrix (array) c x c : mixing matrix, where c is the number of all channels
    channels (array of string) : all channel names
    components_to_plot (list of int) : a list that contains indices of components to plot
        

    Returns
    -------
    None.

    '''

    # component count
    component_count = len(components_to_plot)
    
    # when the number of component to plot is 1
    if component_count==1:
        idx=0
        plot_topo(axes=None, channel_names=list(channels), channel_data=mixing_matrix[:,components_to_plot[idx]], title='ICA component'+' '+str(components_to_plot[idx]), cbar_label='', montage_name='standard_1005')
        return
    
    # when the number of component to plot is less than 5
    if component_count<5:
        # plot subplots
        fig, axes= plt.subplots(1, component_count, figsize=(15,5))
        
        for idx in range(component_count):
            
            # call plot_topo
            plot_topo(axes=axes[idx], channel_names=list(channels), channel_data=mixing_matrix[:,components_to_plot[idx]], title='ICA component'+' '+str(components_to_plot[idx]), cbar_label='', montage_name='standard_1005')
        
        plt.tight_layout()
        
        
        if not os.path.exists(f'plots/components/subject_{subject:02}/series_{series}'):
            os.makedirs(f'plots/components/subject_{subject:02}/series_{series}')
            
        #channels_to_plot = '_'.join(list(channels[components_to_plot]))
        #channels_to_plot = '_'.join([components_to_plot])
        channels_to_plot = '_'.join([str(c) for c in components_to_plot])
        plt.savefig(f"plots/components/subject_{subject:02}/series_{series}/components_topo_subject_{subject}_series_{series}_{channels_to_plot}.png")
        #plt.close()
        return
            
        
    # row and col for subplots
    row_count = component_count // 5
    col_count = component_count % 5
    
    if col_count==0:
        col_count=5
    else:
        row_count+=1
        col_count=5
    
    # plot subplots
    fig, axes= plt.subplots(row_count, col_count, figsize=(15,5))
    
    for idx in range(component_count):
        # get row and column for axes index
        row= idx // 5
        col = idx % 5
        # call plot_topo
        plot_topo(axes=axes[row][col], channel_names=list(channels), channel_data=mixing_matrix[:,components_to_plot[idx]], title='ICA component'+' '+str(components_to_plot[idx]), cbar_label='', montage_name='standard_1005')
    

    # remove empty axes
    for idx in range(component_count, row_count*col_count):
        row= idx // 5
        col = idx % 5
        fig.delaxes(axes[row][col])
    
    plt.tight_layout()
    
    
    if not os.path.exists(f'plots/components/subject_{subject:02}/series_{series}'):
        os.makedirs(f'plots/components/subject_{subject:02}/series_{series}')
    #channels_to_plot = '_'.join(list(channels[components_to_plot]))
    #channels_to_plot = '_'.join([components_to_plot])
    channels_to_plot = '_'.join([str(c) for c in components_to_plot])
    plt.savefig(f"plots/components/subject_{subject:02}/series_{series}/components_topo_subject_{subject}_series_{series}_{channels_to_plot}.png")
    #plt.savefig("plots/components")
    if close:
        plt.close()
    


def get_sources(eeg, unmixing_matrix, fs, sources_to_plot):
    '''
    Transform EEG data into source space using an unmixing matrix.

    Parameters:
    - eeg (numpy.ndarray) (c, t): EEG data.
    - unmixing_matrix (numpy.ndarray) (c, c): Unmixing matrix obtained from ICA.
    - fs (float): Sampling rate.
    - sources_to_plot (list): List of indices of sources to plot.

    where c is the number of channels, and s is number of data points. 

    Returns:
    - source_activations (numpy.ndarray): Source activation timecourses.
    '''
    source_activations = np.dot(unmixing_matrix, eeg)
    
    
    # Plot the activity of specified sources
    if sources_to_plot != []:
        fig, axes = plt.subplots(len(sources_to_plot), 1, sharex=True, figsize=(7,7))
        if len(sources_to_plot) !=1:
            axes_flat = axes.flatten()
        sample_count = eeg.shape[1]
        times = np.arange(sample_count)
        times = times/fs

        for plot_index, source_index in enumerate(sources_to_plot):
            if len(sources_to_plot) !=1:
                axes = axes_flat[plot_index]
            axes.plot(times, source_activations[source_index,:], label="reconstructed")
            axes.set_xlabel("time (s)") 
            axes.set_ylabel(f"Source {source_index} (uV)")

        plt.xlim([55, 60])
        #formatting 
        fig.suptitle("AudVis EEG Data in ICA source space")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/source_activations")   
        plt.close()

    return source_activations


def remove_sources(source_activations, mixing_matrix, sources_to_remove):
    '''
    Remove specified artifact sources from EEG data and reconstruct cleaned data.

    Parameters:
    - source_activations (numpy.ndarray) (c, t): Matrix of source activation timecourses obtained from ICA.
    - mixing_matrix (numpy.ndarray) (c, c): Mixing matrix used to transform source data back into electrode space.
    - sources_to_remove (list): List of indices corresponding to the artifact sources to be removed.

    where c is the number of channels, and s is number of data points. 

    Returns:
    - cleaned_eeg (numpy.ndarray): Cleaned EEG data after removing specified artifact sources.
    '''
    # Zero out the specified sources
    cleaned_source_activations = np.copy(source_activations)
    cleaned_source_activations[sources_to_remove] = 0
    
    # Transform the cleaned source activations back into electrode space
    cleaned_eeg = np.dot(mixing_matrix, cleaned_source_activations)
    
    return cleaned_eeg


def compare_reconstructions(eeg, reconstructed_eeg, cleaned_eeg, fs, channels, channels_to_plot):
    '''    
    Plot the original, reconstructed, and cleaned EEG data.

    Parameters:
    - eeg (numpy.ndarray (c, s)  Original EEG data.
    - reconstructed_eeg (numpy.ndarray) (c, s) : Reconstructed EEG data.
    - cleaned_eeg (numpy.ndarray) (c, s) : Cleaned EEG data.
    - fs (float): Sampling rate.
    - channels (list): List of all channel names.
    - channels_to_plot (list): List of channel names to plot.

    where c is the number of channels, and s is number of data points. 

    Returns:
    - None
    '''
    fig, axes = plt.subplots(len(channels_to_plot), 1, sharex=True, figsize=(7,7))
    if len(channels_to_plot) != 1:
        axes_flat =axes.flatten()
    sample_count = eeg.shape[1]
    times = np.arange(sample_count)
    times = times/fs

    for plot_index, channel_name in enumerate(channels_to_plot): 
        #plot relevant channels 
        if len(channels_to_plot) != 1:
            axes = axes_flat[plot_index]
        channel_index = np.where(channels == channel_name)[0][0]
        axes.plot(times, eeg[channel_index,:], label="source")
        axes.plot(times, reconstructed_eeg[channel_index,:], label="reconstructed", linestyle='dotted')
        axes.plot(times, cleaned_eeg[channel_index,:], label="cleaned",linestyle='dashed')
        axes.set_xlabel("time (s)") 
        axes.set_ylabel(f"Voltage on {channels[channel_index]} (uV)") 
        axes.legend()
    #formatting 
    plt.xlim([55, 60])
    fig.suptitle("AudVis EEG Data reconstructed & cleaned after ICA")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/reconstructed_cleaned_original")   
    
# %% project 1 bootstrapping

def bootstrap_data(target_erp, nontarget_erp, bootstrap_iteration=3000, ntrials=-10):
    """
    (c : number of channels)
    (n' : number of samples in an epoch)
    (nt : number of target events)
    (nn : number of nontarget events)
    (nb : number of iterations of bootstrapping)
    (nbt : number of trials at each iteration)

    Parameters
    ----------
    target_erp : (numpy array, float), size - c x n'
        each element contains erp data of each sample in an epoch
        
    nontarget_erp : (numpy array, float), size - c x n'
        each element contains erp data of each sample in an epoch
        
    bootstrap_iteration : (int), optional
        Number of iterations of Bootstrapping. The default is 3000.
        
    ntrials : (int), optional
        DESCRIPTION. The default is -10, which is defined as number of samples in an epoch if not given by user.

    Returns
    -------
    erp_diff : (numpy array, float), size c x n'
        each element contains difference of target and nontarget ERP data
    
    erp_sampled : (numpy array, float), size - c x nb
        sampled data by bootstrapping
        
    This function takes ERP data and returns bootstrapped data and ERP difference data

    """
    
    
    # number of channel
    channel_count =target_erp.shape[0]
    
    # absolute difference between target and nontarget ERPS, c x n' (number of channels x number of samples in an epoch)
    erp_diff = np.abs(target_erp - nontarget_erp)

    # set the number of trials at each iteration if not given
    if ntrials<0:
        ntrials=erp_diff.shape[1]

    # get permutation of indices to pick from real data, c x nb x nbt
    idx_for_bootstrap = np.random.randint(0, erp_diff.shape[1], (channel_count, bootstrap_iteration, ntrials ))
    #i=index_for_bootstrap
    
    
    # bootstrapped ERP diff data, c x nb x nbt
    erp_diff_bootstrap=np.array([erp_diff[channel][idx_for_bootstrap[channel]] for channel in np.arange(channel_count)])
    
    # compute the mean at each iteration, c x nb
    erp_sampled= np.mean(erp_diff_bootstrap, axis=2)
    
    return erp_diff, erp_sampled


def get_pv(erp_diff, erp_sampled):
    """
    (c : number of channels)
    (n' : number of samples in an epoch)
    (nb : number of iterations of bootstrapping)


    Parameters
    ----------
    erp_diff : (numpy array, float), size - c x n'
        each element contains difference of target and nontarget ERP data
    
    erp_sampled : (numpy array, float), size - c x nb
        sampled data by bootstrapping

    Returns
    -------
    p_values : (numpy array, float), size - c x n'
        p values of each time step of all channels
        
    This function takes sampled data and difference data, then returns p values

    """
    # number of channels
    channel_count=erp_diff.shape[0]
    
    # get the samples greater than ERP difference at each step of all channels, c x n'
    values_greater_than_erp_diff=np.array([  erp_diff[channel][...,None]<erp_sampled[channel] for channel in np.arange(channel_count)])
    
    # get the P values, cdf.
    bootstrap_iteration=erp_sampled.shape[1]
    p_values=np.count_nonzero(values_greater_than_erp_diff, axis=-1)/bootstrap_iteration
    
    return p_values


# import fdr_correction function
from mne.stats import fdr_correction

def fdr_pv_and_significance(p_values, alpha=0.05):
    """
    (c : number of channels)
    (n' : number of samples in an epoch)
    
    Parameters
    ----------
    p_values : (numpy array, float), size - c x n'
        p values of each time step of all channels
        
    alpha : (float), optional
        Threshold to check p-values. The default is 0.05.

    Returns
    -------
    p_values_corrected : (numpy array, float), size - c x n'
        corrected p values of each time step of all channels
        
    significance_bool : (numpy array, bool), size - c x n'
        boolean which indicates whether current p-value should reject the null hypothesis or not.

    """
    
    # number of channels
    channel_count= p_values.shape[0]
    
    # fdr correction
    fdr_corrected=[fdr_correction(p_values[channel],alpha=alpha)  for channel in np.arange(channel_count)  ] 
    
    # corrected p_values
    p_values_corrected= [ fdr_corrected[channel][1] for channel in np.arange(channel_count)]
    significance_bool= [ fdr_corrected[channel][0] for channel in np.arange(channel_count)]
    
    return np.array(p_values_corrected), np.array(significance_bool)


def get_true_blocks(arr):
    result=[]
    
    length= len(arr)
    
    idx =0
    current_block_start=-1
    
    while idx<length:
        if arr[idx]==True:
            # start a new block
            if current_block_start==-1:
                current_block_start=idx
        
        else:
            if current_block_start!=-1:
                result.append([current_block_start,idx-1])
                current_block_start=-1
        idx+=1
    
    if current_block_start!=-1:
        result.append([current_block_start, idx-1])
    
    return np.array(result)


    
    
    