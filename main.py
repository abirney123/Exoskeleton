#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 07:50:20 2024

@author: Alaina Birney, Yoonki Hong, Ashley Heath

A python module to load original .mat data files for the WAY-EEG-GAL dataset,
combine those files into a single file for each subject,load data for
each subject from the combined .mat file, epoch data for each event type, create 
ERPs for each event type, plot ERPs for subjects, channels, and events of interest, 
create and apply a bandpass filter to EEG data, generate mixing and unmixing 
matrices for EEG data, remove sources, get the frequency and power spectra, 
plot the power spectra, calculate when there are statistically significant 
differences between ERPs related to the events HandStart and FirstDigitTouch, 
and generate plots to visualize when these statistically significant differences 
occur.
"""
import os
import loadmat
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind


#%% Load original .mat data files
def load_data(subjects, data_directory = "./WAY-EEG-GAL/"):
    '''
    Data files of each subjec must be in 'PX' folder (X is subjec id) within the data_directory.
    Example)  .../WAY-EEG-GAL/P1/HS_P1_S1.mat ....
                             /P2/ ...
    
    A function for loading in .mat data files for the WAY-EEG-GAL dataset.
    Note: ChatGPT (3.5) was used to aid in writing the following docstring.

    Parameters
    ----------
    subjects : List of int, size can range from 1 to 12 (the number of subjects
    for which we have data)
        The ID numbers of subjects to load data for. Values can be numbers within
        the range 1-12.
    data_directory : str, optional
        Relative path to the data files. The default is "./WAY-EEG-GAL/".

    Returns
    -------
    result_data_set: A dictionary containing loaded data for each series for 
    each subject specified. The keys follow a pattern where each subject's data
    is identified by "subject_<subject_number>". Within a subject's data, eeg
    data is organized into series. Each series is identified by the key 
    "series_<series_number>". Within each series key, the following data is present:
        - eeg: array of float, size (S, 32) where S is the number of samples 
        present.
            EEG data for the subject. Columns represent channels.
        - hs_time_first_eeg: float.
            The relative time of the first eeg data in "eeg"
        - trial_end_time: list of length T where T is the number of trials.
            The end times of each trial in seconds.
        - trial_start_time: list of length T where T is the number of trials.
            The start times of each trial in seconds.
        - channel_names: Array of object, size (32,)
            The name of each channel of eeg data. Indices align with columns 
            of eeg such that EEG[:,i] corresponds to the channel name at 
            channel_names[i]
    
    Additionally, the keys "all_lifts" and "all_lifts_column_name" are present 
    within a subject's data. "all_lifts" contains an array representing lift
    data for all series. Each row represents a single lifting trial, and each
    column represents a specific aspect of the trial, such as participant number,
    series number, weight, surface, and time stamps for various events. 
    "all_lifts_column_names" contains the column names for "all_lifts", and 
    more detailed descriptions of what each of those column names mean can be found
    in the readme for this dataset.
    '''
    # initialize list to store results
    result_data_set={}
    
    for subject in subjects: # loop through subjects to load data for each
        subject_data_directory = f"{data_directory}P{subject}" 
        result_data_set[f'subject_{subject}']={}
        #result_data_set[subject]={}
        for series in range(1,10):
            series_dict={}
            series_dict['hs_time_first_eeg']=2.0000 # relative time of first eeg data in 'eeg'
            
            hs_file_path = os.path.join(subject_data_directory, f"HS_P{subject}_S{series}.mat")
            series_dict['eeg'] = loadmat.loadmat(hs_file_path)['hs']['eeg']['sig'] # eeg of this series
            
            
            # ws 
            ws_file_path = os.path.join(subject_data_directory, f"WS_P{subject}_S{series}.mat")
            ws_dict_one_series = loadmat.loadmat(ws_file_path)['ws'] # load file
            
            # extract channel names
            series_dict["channel_names"] = ws_dict_one_series["names"]["eeg"]
            
            # start and end time of trials relative to the start of the series
            series_dict['trial_start_time']=[]
            series_dict['trial_end_time']=[]
    
    
            trials_data_one_series=ws_dict_one_series['win']
            for i in range(len(trials_data_one_series)):
                trialss = loadmat._todict(trials_data_one_series[i])
                trial_start_time = trialss['trial_start_time']
                trial_end_time = trialss['trial_end_time']
                
                series_dict['trial_start_time'].append(trial_start_time)
                series_dict['trial_end_time'].append(trial_end_time)
            
            result_data_set[f'subject_{subject}'][f'series_{series}']=series_dict
            #result_data_set[subject][series]=series_dict
            
        
        lifts_file_path = os.path.join(subject_data_directory, f"P{subject}_AllLifts.mat")
        # load file
        lifts_dict_one_subject = loadmat.loadmat(lifts_file_path)['P']
        
        # indices to extract from all lifts
        all_lifts_indices= np.array([0,1,2, 8,9, 11,12,13,14,15,16,17,18,19,20,21,22])
        
        result_data_set[f'subject_{subject}']['all_lifts']=lifts_dict_one_subject['AllLifts'][:,all_lifts_indices]
        result_data_set[f'subject_{subject}']['all_lifts_column_name']= lifts_dict_one_subject['ColNames'][all_lifts_indices]            
            
    return result_data_set



#%% Create combined files
def combine_files(subjects):
    """
    
    A function to create a combined .mat file for each subject. For each subject
    specified in the subjects list, a separate .mat file combining lifts, eeg data,
    and trial metadata will be created. load_data is called in order to accomplish
    this.

    Parameters
    ----------
    subjects : List. Length can range from 1-12, the total number of subjects for
    which we have data.
        The subjects to create combined .mat files for.

    Returns
    -------
    None.

    """
    for subject in subjects:
        ss=load_data([subject],"WAY-EEG-GAL/")
        scipy.io.savemat(f"s{subject}.mat",ss)
        
# %% Load subject data from combined .mat file

def load_subject_data(subject, data_directory=""):
    """
    A function for loading in data for one subject from the combined .mat file.
    Calls loadmat.py from (https://github.com/djangraw/BCIs-S24/blob/main/loadmat.py)
    in order to do this.
    
    Parameters
    ----------
    subject: Int
        The number of the subject/ participant for whom the data is to be loaded.
    data_directory: Str, optional
        The relative path to the combined .mat file containing the data for the
        subject. The default is an empty string, assuming the current directory.
        
    Returns
    -------
    subject_data: A dictionary containing loaded data for each series for 
    the specified subject. Within this dictionary, eeg data is organized into 
    series. Each series is identified by the key "series_<series_number>". Within 
    each series key, the following data is present:
        - eeg: array of float, size (S, 32) where S is the number of samples 
        present.
            EEG data for the subject. Columns represent channels.
        - hs_time_first_eeg: float.
            The relative time of the first eeg data in "eeg"
        - trial_end_time: list of length T where T is the number of trials.
            The end times of each trial in seconds.
        - trial_start_time: list of length T where T is the number of trials.
            The start times of each trial in seconds.
        - channel_names: Array of object, size (32,)
            The name of each channel of eeg data. Indices align with columns 
            of eeg such that EEG[:,i] corresponds to the channel name at 
            channel_names[i]
    
    Additionally, the keys "all_lifts" and "all_lifts_column_name" are present 
    within a subject's data. "all_lifts" contains an array representing lift
    data for all series. Each row represents a single lifting trial, and each
    column represents a specific aspect of the trial, such as participant number,
    series number, weight, surface, and time stamps for various events. 
    "all_lifts_column_names" contains the column names for "all_lifts", and 
    more detailed descriptions of what each of those column names mean can be found
    in the readme for this dataset.
    
    """
    
    data_directory+=f"s{subject}.mat"
    subject_data = loadmat.loadmat(data_directory)[f'subject_{subject}']
    
    return subject_data


def epoch_data(subjects_data, epoch_start_time = -1, epoch_end_time = 0.85, fs=500):
    """
    A function to separate EEG data into epochs. Epochs are kept separate for
    each subject and series as well as for each event.
    Note: This function is a heavily modified version of epoch_data within the
    plot_p300_erps script written by Nick Bosley and Alaina Birney for lab 2.

    Parameters
    ----------
    subjects_data : A dictionary containing loaded data for each series for 
        all subjects for whom data was loaded. Within this dictionary, eeg data 
        is organized by subject and series. Each subject is identified by the key
        "subject_<subject_number>". Within each subject dictionary, each series is 
        identified by the key "series_<series_number>". Within each series key, 
        the following data is present:
            - eeg: array of float, size (S, 32) where S is the number of samples 
            present.
                EEG data for the subject. Columns represent channels.
            - hs_time_first_eeg: float.
                The relative time of the first eeg data in "eeg"
            - trial_end_time: list of length T where T is the number of trials.
                The end times of each trial in seconds.
            - trial_start_time: list of length T where T is the number of trials.
                The start times of each trial in seconds.
            - channel_names: Array of object, size (32,)
                The name of each channel of eeg data. Indices align with columns 
                of eeg such that EEG[:,i] corresponds to the channel name at 
                channel_names[i]
        
        Additionally, the keys "all_lifts" and "all_lifts_column_name" are present 
        within a subject's data. "all_lifts" contains an array representing lift
        data for all series. Each row represents a single lifting trial, and each
        column represents a specific aspect of the trial, such as participant number,
        series number, weight, surface, and time stamps for various events. 
        "all_lifts_column_names" contains the column names for "all_lifts", and 
        more detailed descriptions of what each of those column names mean can be found
        in the readme for this dataset.
    epoch_start_time : Int, optional
        The start time in seconds for each epoch relative to an event. The default 
        is -1, representing that the epoch should begin 1 second prior to the 
        event start.
    epoch_end_time : Float, optional
        The end time in seconds for each epoch relative to a trial. The default 
        is .85, representing that the epoch should end .85 seconds after the
        event ends.
    fs : Int, optional
        The sampling frequency in seconds. The default is 500.

    Returns
    -------
    all_subject_epochs: Dictionary of size S where S is the number of subjects
    for which data was loaded.
    EEG epoch data organized by subject, series, and event type. 
    - First level keys represent distinct subjects. 
    - Within each subject's key, there is a key for each series, there are 9 
    series per subject. 
    - Within each series key, there is a key for each event type (There are 6 
    event types: "HandStart", "FirstDigitTouch", BothStartLoadPhase", "LiftOff", 
    "Replace", and "BothReleased".)
    - Within each event type key, there is a tuple. The first value in the tuple
    is the epoch data, which is a 3 dimensional array of shape (number of epochs,
    number of samples per epoch, number of channels of EEG data). The second
    value in the tuple is a list of times representing the time of each
    sample in the epoch. Times correspond to sample indices such that the
    first entry in the list of times corresponds to the first sample in the 
    epoch and so on. For each event type, series, and subject, there should be 
    approximately 270 epochs (approximately 30 trials were performed per
    series and each trial included each event type).
    epoch_times: List of float. Size (Se,) where Se is the number of samples
    in an epoch.
        A list of times representing the time of each sample in the epoch. 
        Times correspond to sample indices such that the first entry in the list 
        of times corresponds to the first sample in the epoch and so on. These 
        values are included in all_subject_epochs as well as this list so that 
        epoch times can be easily accessed when working on epoch data or easily 
        accessed when later working on erps.
        
    """
    # define mapping to time column for each of 6 distinct event types
    event_mapping = {
        "HandStart": "LEDOn",
        "FirstDigitTouch": "tFirstDigitTouch",
        "BothStartLoadPhase": "tBothStartLoadPhase",
        "LiftOff": "tLiftOff",
        "Replace": "tReplace",
        "BothReleased": "tBothReleased"}
    
    # separate data into epochs
    # epochs should begin 1000 ms before event so that readiness potential
    # can be detected, end 850ms after event end to capture post movement beta rebound
    
    # initialize dictionaries to store epochs for different event types and times
    all_subject_epochs = {}

    # get samples per epoch
    seconds_per_epoch = epoch_end_time - epoch_start_time
    samples_per_epoch = int(fs * seconds_per_epoch)
    #print(f"samples per epoch: {samples_per_epoch}")
    
    # get number of channels- will be same for everyone so we can just do it once here
    for subject_key, subject_data in subjects_data.items(): 
        for series_key, series_data in subject_data.items():
            # skip over lifts keys
            if series_key.startswith("series_"):
                # get number of channels
                num_channels = series_data["eeg"].shape[1]
                # exit the inner loop
                break
            # exit the outer loop
            break
    # create times 
    epoch_times = np.arange(epoch_start_time, epoch_end_time,1/fs)
    
    
    
    
    # iterate over subject data
    for subject_key, subject_data in subjects_data.items():
        # get lift data
        lifts = subject_data["all_lifts"]
        lift_column_names = subject_data["all_lifts_column_name"]
        
        
        # get event data by type using mapping dictionary
        event_data_by_type = {event: lifts[:, np.where(lift_column_names == time_col_name)[0][0]] 
                              for event, time_col_name in event_mapping.items() if time_col_name in lift_column_names}
        
        # initialize dictionary for this subject's series
        all_series_epochs = {}
        
        
        event_index_offset=0
        # iterate over series for this subject
        for series_key, series_data in subject_data.items():
            # skip over lift data- only look at keys that start with series
            if series_key.startswith("series_"):
                # initialize dictionary for this series
                series_epochs = {}
                
                # trial start time
                trial_start_times = series_data['trial_start_time']
                first_eeg_time = series_data['hs_time_first_eeg'] # time of first eeg relative to the onset of the series
                
                eeg_data=series_data['eeg']
                
                
                
                # iterate over event types and times
                for event_type, event_times in event_data_by_type.items():

                    num_epochs = len(trial_start_times) # create one epoch for each event
                    # initialize array to hold epoch data
                    epochs = np.zeros((num_epochs, samples_per_epoch, num_channels))
                    
                    for trial_index, trial_start_time in enumerate(trial_start_times):
                        
                        # event_time_index in all the events across all the series
                        event_time_index = trial_index + event_index_offset
                        # start time of the event relative to the onset of the trial
                        event_time = event_times[event_time_index]
                        
                        # start time of the event relative to the onset of series
                        # actual start is event time plus offset
                        epoch_actual_start_time = event_time + trial_start_time + epoch_start_time-first_eeg_time
                        # get start and end indices
                        epoch_start_index = int(round(epoch_actual_start_time * fs))
                        epoch_end_index = epoch_start_index + samples_per_epoch 
                        
                        if (0 <= epoch_start_index < len(eeg_data)) and (epoch_end_index <= len(eeg_data)):
                            epoch = eeg_data[epoch_start_index: epoch_end_index, :]
                            #epochs[event_time_index] = epoch
                            epochs[trial_index] = epoch
                        else:
                            print(trial_index, event_time_index, epoch_start_index, epoch_end_index)
                            print(f"Epoch out of bounds for event {event_type} at time {event_time}")
                    # store this series epochs
                    series_epochs[event_type] = (epochs, epoch_times)
                event_index_offset+= len(trial_start_times)    
                
            # store all series epochs 
            all_series_epochs[series_key] = series_epochs
        # store all epochs for this subject
        all_subject_epochs[subject_key]= all_series_epochs
       
    return all_subject_epochs, epoch_times

def get_erps(all_subject_epochs):
    """
    A function to extract event-related potentials (ERPs) for each subject, 
    series, and event type from epoched EEG data. Please note that this function 
    is a heavily modified version of get_erps within the plot_p300_erps script 
    written by Nick Bosley and Alaina Birney for lab 2.

    Parameters
    ----------
    all_subject_epochs: Dictionary of size S where S is the number of subjects
    for which data was loaded.
        EEG epoch data organized by subject, series, and event type. 
        - First level keys represent distinct subjects. 
        - Within each subject's key, there is a key for each series, there are 9 
        series per subject. 
        - Within each series key, there is a key for each event type (There are 6 
        event types: "HandStart", "FirstDigitTouch", BothStartLoadPhase", "LiftOff", 
        "Replace", and "BothReleased".)
        - Within each event type key, there is a tuple. The first value in the tuple
        is the epoch data, which is a 3 dimensional array of shape (number of epochs,
        number of samples per epoch, number of channels of EEG data). The second
        value in the tuple is a array of times representing the time of each
        sample in the epoch. Times correspond to sample indices such that the
        first entry in the array of times corresponds to the first sample in the 
        epoch and so on. For each event type, series, and subject, there should be 
        approximately 270 epochs (approximately 30 trials were performed per
        series and each trial included each event type).

    Returns
    -------
    all_subject_erps: Dictionary of size S where S is the number of subjects
    for which data was loaded.
        Event-related potentials (ERPs) organized by subject, series, and event type.
        - First level keys represent distinct subjects. 
        - Within each subject's key, there is a key for each series, there are 9 
        series per subject. 
        - Within each series key, there is a key for each event type (There are 6 
        event types: "HandStart", "FirstDigitTouch", BothStartLoadPhase", "LiftOff", 
        "Replace", and "BothReleased".)
        - Within each event type key, there is an array of shape (Se, 32) where Se
        is the number of ERPs. There is one ERP for each value within the original
        epoch, because EEG amplitudes are averaged across epochs to form ERPs.
    """
    # create dictionary with level 1 keys as subjects, level 2 keys as series,
    # level 3 keys as event types, level 3 values erps for each event type
    all_subject_erps = {}
    
    # define event order in all_subjects_epochs
    event_order = ["HandStart", "FirstDigitTouch", "BothStartLoadPhase", "LiftOff",
                   "Replace", "BothReleased"]
    # loop through all subject epochs to get epochs for each event type
    for subject_key, subject_data in all_subject_epochs.items():
        all_series_erps = {}
        for series_key, series_data in subject_data.items():
            series_erps = {}
            # reset event type index
            event_type_index = 0
            for event_type_key, event_data in series_data.items():
                # tuple val 1 is epoch, val 2 is time
                # add erp to series_erps at key of current event
                series_erps[event_order[event_type_index]] = np.mean(event_data[0],axis = 0)
                # increment event type index
                event_type_index += 1
            # store all series erps
            all_series_erps[series_key] = series_erps
        # store all erps for this subject
        all_subject_erps[subject_key] = all_series_erps
    
    return all_subject_erps
            
def plot_erps(all_subject_erps, all_subject_epochs, subjects_data, channels_to_plot, subjects_to_plot, events_to_plot):
    '''
    A function to plot ERPs for specified EEG channels. For each subject listed
    in subjects_to_plot, 6 figures containing C subplots each will be created, 
    where C is the number of channels to plot data for. Each figure corresponds 
    to an event so that the plots within that figure represent ERPs corresponding 
    to that event. Each subplot contains the event's ERP for a given channel 
    with the x-axis representing the time from the event onset in seconds and 
    the y-axis representing the EEG voltage in uV. Additionally, each subplot 
    contains a vertical line at the time that the event began.
    
    Parameters
    ----------
    all_subject_erps: Dictionary of size S where S is the number of subjects
    for which data was loaded.
        Event-related potentials (ERPs) organized by subject, series, and event type.
        - First level keys represent distinct subjects. 
        - Within each subject's key, there is a key for each series, there are 9 
        series per subject. 
        - Within each series key, there is a key for each event type (There are 6 
        event types: "HandStart", "FirstDigitTouch", BothStartLoadPhase", "LiftOff", 
        "Replace", and "BothReleased".)
        - Within each event type key, there is an array of shape (Se, 32) where Se
        is the number of ERPs. There is one ERP for each value within the original
        epoch, because EEG amplitudes are averaged across epochs to form ERPs.
    all_subject_epochs: Dictionary of size S where S is the number of subjects
    for which data was loaded.
        EEG epoch data organized by subject, series, and event type. 
        - First level keys represent distinct subjects. 
        - Within each subject's key, there is a key for each series, there are 9 
        series per subject. 
        - Within each series key, there is a key for each event type (There are 6 
        event types: "HandStart", "FirstDigitTouch", BothStartLoadPhase", "LiftOff", 
        "Replace", and "BothReleased".)
        - Within each event type key, there is a tuple. The first value in the tuple
        is the epoch data, which is a 3 dimensional array of shape (number of epochs,
        number of samples per epoch, number of channels of EEG data). The second
        value in the tuple is an array of times representing the time of each
        sample in the epoch. Times correspond to sample indices such that the
        first entry in the array of times corresponds to the first sample in the 
        epoch and so on. For each event type, series, and subject, there should be 
        approximately 270 epochs (approximately 30 trials were performed per
        series and each trial included each event type).
    subjects_data : A dictionary containing loaded data for each series for 
        all subjects for whom data was loaded. Within this dictionary, eeg data 
        is organized by subject and series. Each subject is identified by the key
        "subject_<subject_number>". Within each subject dictionary, each series is 
        identified by the key "series_<series_number>". Within each series key, 
        the following data is present:
            - eeg: array of float, size (S, 32) where S is the number of samples 
            present.
                EEG data for the subject. Columns represent channels.
            - hs_time_first_eeg: float.
                The relative time of the first eeg data in "eeg"
            - trial_end_time: list of length T where T is the number of trials.
                The end times of each trial in seconds.
            - trial_start_time: list of length T where T is the number of trials.
                The start times of each trial in seconds.
            - channel_names: Array of object, size (32,)
                The name of each channel of eeg data. Indices align with columns 
                of eeg such that EEG[:,i] corresponds to the channel name at 
                channel_names[i]
        
        Additionally, the keys "all_lifts" and "all_lifts_column_name" are present 
        within a subject's data. "all_lifts" contains an array representing lift
        data for all series. Each row represents a single lifting trial, and each
        column represents a specific aspect of the trial, such as participant number,
        series number, weight, surface, and time stamps for various events. 
        "all_lifts_column_names" contains the column names for "all_lifts", and 
        more detailed descriptions of what each of those column names mean can be found
        in the readme for this dataset.
    channels_to_plot: List of str. Size can range from 1 to 32 (the number of 
    EEG channels for which we have data)
        The channels to create plots of ERPs for.
    subjects_to_plot: List of int. Size can range from 1 to S where S is the 
    number of subjects for which data was loaded.
        The subjects to plot ERPs for.
    events_to_plot: List of str. Size can range from 1 to 6 because the data
    contains 6 distinct event types.
        The names of the events to create plots for.
    Returns
    -------
    None.

    '''
    event_types = ["BothReleased", "BothStartLoadPhase", "FirstDigitTouch",
                   "HandStart", "LiftOff", "Replace"]
    
    # get channel names from subjects data
    for subject_key, subject_data in subjects_data.items():
        for series_key, series_data in subject_data.items():
            # skip series data, looking to lift data only
            if series_key.startswith("series_"):
                channel_names = series_data["channel_names"]
                break # exit loops once channel names found
        break
    # loop through specified subjects
    for subject_number in subjects_to_plot:
        # set key for erp dictionary based on current subject number
        subject_key = f"subject_{subject_number}" 
        if subject_key in all_subject_erps: 
            # loop through event types
            for event_type_index, event_type in enumerate(events_to_plot): # was event_types
                plt.figure(figsize=(15,12)) # initialize figure for event type plots
                plt.suptitle(f"ERPs for Subject {subject_number}, Event: {event_type}")
                # loop through channels to plot
                for channel_index, channel in enumerate(channels_to_plot):
                    # create subplot for each channel
                    plt.subplot(len(channels_to_plot),1,channel_index + 1)
                    plt.title(f"Channel {channel}")
                    # plot ERPs for each series for current subject
                    for series_key in all_subject_erps[subject_key]:
                        if event_type in all_subject_erps[subject_key][series_key]:
                            erp = all_subject_erps[subject_key][series_key][event_type]
                            epoch_tuple = all_subject_epochs[subject_key][series_key][event_type]
                            times = epoch_tuple[1] # times are second entry in epoch tuple
                            # get index for current channel
                            channel_to_plot_index = None
                            channel_to_plot_index = np.where(channel_names == channel)[0][0]
                            if channel_to_plot_index == None:
                                print(f"Channel {channel} not found. Please ensure the specified channels are included in the data and spelled correctly.")
                            # plot erp for current channel, add label to represent series
                            #print(f"ERP size for one channel: {erp[:,channel_index].shape}")
                            # only plot with labels on first legend to avoid redundant legends (series are consistent across subjects)
                            if channel_index == 0:
                                plt.plot(times, erp[:,channel_to_plot_index], label = f"Series {series_key[-1]}")
                                handles, labels = plt.gca().get_legend_handles_labels()  # store handles and labels 
                            else:
                                plt.plot(times, erp[:,channel_to_plot_index])
                        
                    # add labels after all series data has been plotted
                    plt.xlabel('Time from event onset (s)')
                    plt.ylabel('Voltage (uV)')
                    plt.axvline(x=0, linestyle=':', color='black') # event onset
                    plt.figlegend(handles, labels, loc="upper left", ncol = 3)
                # apply tight layout and show after all aspects of figure have been generated
                plt.tight_layout() # add padding to accomodate legend
                plt.savefig(f"ERPs_Subject_{subject_number}_Event_{event_type}_Channels_{channels_to_plot}.png")
                #plt.show()
                plt.close()
                
        else:
            # print a message if no data for this subject exists
            print(f"No data available for Subject {subject_number}")
    

    

        
    

# %%

import utility as util
import scipy.signal





# get making a filter function from filter_ssvep_data
make_band_pass_filter = util.make_bandpass_filter


# get applying a fliter function from filter_ssvep_data
filter_data = util.filter_data



def filter_all_series(subjects_data, filter_to_apply):
    """
    
    Parameters
    ----------
    subjects_data : A dictionary containing loaded data for each series for 
    each subject specified. The keys follow a pattern where each subject's data
    is identified by "subject_<subject_number>". Within a subject's data, eeg
    data is organized into series. Each series is identified by the key 
    "series_<series_number>". Within each series key, the following data is present:
        - eeg: array of float, size (S, 32) where S is the number of samples 
        present.
            EEG data for the subject. Columns represent channels.
        - hs_time_first_eeg: float.
            The relative time of the first eeg data in "eeg"
        - trial_end_time: list of length T where T is the number of trials.
            The end times of each trial in seconds.
        - trial_start_time: list of length T where T is the number of trials.
            The start times of each trial in seconds.
        - channel_names: Array of object, size (32,)
            The name of each channel of eeg data. Indices align with columns 
            of eeg such that EEG[:,i] corresponds to the channel name at 
            channel_names[i]
    filter_to_apply : array of floats. size - (filter_order+1)
        coefficients of the filter
    

    Returns
    -------
    None.
    
    This function replaces all the EEG data in the dictionary with filtered EEG data.

    """
    
    # iterate subjects
    for subject in subjects_data:
        subject_data=subjects_data[subject]
        
        # iterate series
        for series_idx in range(1,10):
            
            # get the EEG data of the current series            
            series_data= subject_data[f'series_{series_idx}']['eeg']
            
            # filter the EEG data
            series_filtered = filter_data(np.transpose(series_data), filter_to_apply)
            
            # replace the EEG data in the dictionary with the filtered data
            subjects_data[subject][f'series_{series_idx}']['eeg'] = np.transpose( series_filtered)
           

# %% ICA components


from sklearn.decomposition import FastICA
#import warnings
from tqdm import tqdm

def get_mixing_matrices(eeg_data, n_components=32 ):
    """
    
    Parameters
    ----------
    eeg_data : Array of float, size (C, S) where C is the number of channels and 
                                                 S is the number of samples per epoch.
    
        EEG data
    n_components : int, optional
        number of components. The default is 32.

    Returns
    -------
    mixing_matrix : Array of floats, size (F, P) where F is the number of features and
                                                       P is the number of components.
        mixing matrix to get the observed values. F is same as the number of channels
    unmixing_matrix : Array of floats, size (F, P) where F is the number of features and
                                                       P is the number of components.
        unmixing matrix to get the source values. F is same as the number of channels
        
    get the mixing and unmixing matrices of the eeg_data.

    """
    
    
    # create FastICA object
    ica = FastICA(n_components=n_components, random_state=0, whiten='unit-variance', max_iter=2000)
    
    
    ica.fit_transform(eeg_data.T).T  # Fit and transform the data

    # get the mixing and unmixing matrix
    mixing_matrix = ica.mixing_
    unmixing_matrix = np.linalg.inv(mixing_matrix)
    
    return mixing_matrix, unmixing_matrix


def add_mixing_matrices_to_dict(subjects_data, n_components):
    """
    Parameters
    ----------
    subjects_data : A dictionary containing loaded data for each series for 
                 each subject specified. The keys follow a pattern where each subject's data
                 is identified by "subject_<subject_number>". Within a subject's data, eeg
                 data is organized into series. Each series is identified by the key 
                 "series_<series_number>". Within each series key, the following data is present:
        - eeg: array of float, size (S, 32) where S is the number of samples 
        present.
            EEG data for the subject. Columns represent channels.
        - hs_time_first_eeg: float.
            The relative time of the first eeg data in "eeg"
        - trial_end_time: list of length T where T is the number of trials.
            The end times of each trial in seconds.
        - trial_start_time: list of length T where T is the number of trials.
            The start times of each trial in seconds.
        - channel_names: Array of object, size (32,)
            The name of each channel of eeg data. Indices align with columns 
            of eeg such that EEG[:,i] corresponds to the channel name at 
            channel_names[i]
    n_components : int, optional
        number of components. The default is 32.

    Returns
    -------
    non_convergent : list of strings.
        list of strings in which an element shows the subject and series numbers that didn't converge.
        
    Add mixing matrices to the data dictionary
    """
    
    # create an empty list
    non_convergent=[]
    
    
    # iterate subjects data
    for subject in subjects_data:
        print(subject+' - FastICA')
        # iterate series
        for i in tqdm(range(1,10)):
            series = f'series_{i}'
            
            # get the EEG data of this series
            eeg = subjects_data[subject][series]['eeg'].T
            
            # catch if did not converge
            try:
                # get the mixing and unmixing matrices 
                mix, unmix = get_mixing_matrices(eeg, n_components)
                
                # and store them in new fields of this series dictionary
                subjects_data[subject][series]['mixing']= mix
                subjects_data[subject][series]['unmixing']= unmix
                subjects_data[subject][series]['components_to_remove']=[]
            except:
                warning_print =f'{subject}_{series} did not converge'
                print(warning_print)
                non_convergent.append(warning_print)
                subjects_data[subject][series]['mixing']= []
                subjects_data[subject][series]['unmixing']= []
                subjects_data[subject][series]['components_to_remove']=[]
            
    return non_convergent

def remove_sources(eeg_data, mixing_matrix, unmixing_matrix, components_to_remove):
    """

    Parameters
    ----------
    eeg_data : Array of float, size (C, S) where C is the number of channels and 
                                                 S is the number of samples per epoch.
    
        EEG data
    mixing_matrix : Array of floats, size (F, P) where F is the number of features and
                                                       P is the number of components.
        mixing matrix to get the observed values. F is same as the number of channels
    unmixing_matrix : Array of floats, size (F, P) where F is the number of features and
                                                       P is the number of components.
        unmixing matrix to get the source values. F is same as the number of channels
    components_to_remove : List of ints, size (R) where R is the number of channel indices to remove
        list of channel indices to remove

    Returns
    -------
    recovered_eeg : Array of float, size (C, S) where C is the number of channels and 
                                                 S is the number of samples per epoch.
        EEG data after removing sources that are in the components_ro_remove
    
    remove resources
    """
    
    # get source data
    source_data=np.dot(unmixing_matrix, eeg_data)
    # replace values of channels in components_to_remove with zeros
    source_data[components_to_remove]=0
    # recover data
    recovered_eeg= np.dot(mixing_matrix, source_data)
    
    return recovered_eeg

def remove_sources_all_subjects(subjects_data):
    """
    Parameters
    ----------
    subjects_data :subjects_data : A dictionary containing loaded data for each series for 
                 each subject specified. The keys follow a pattern where each subject's data
                 is identified by "subject_<subject_number>". Within a subject's data, eeg
                 data is organized into series. Each series is identified by the key 
                 "series_<series_number>". Within each series key, the following data is present:
        - eeg: array of float, size (S, 32) where S is the number of samples 
        present.
            EEG data for the subject. Columns represent channels.
        - hs_time_first_eeg: float.
            The relative time of the first eeg data in "eeg"
        - trial_end_time: list of length T where T is the number of trials.
            The end times of each trial in seconds.
        - trial_start_time: list of length T where T is the number of trials.
            The start times of each trial in seconds.
        - channel_names: Array of object, size (32,)
            The name of each channel of eeg data. Indices align with columns 
            of eeg such that EEG[:,i] corresponds to the channel name at 
            channel_names[i]

    Returns
    -------
    None.

    """
    
    # iterate subjects data
    for subject in subjects_data:
        print(subject + " remove sources")
        # iterate series
        for series_idx in tqdm(range(1,10)):
            series = f'series_{series_idx}'
            
            # get the componenets to remove
            components_to_remove = subjects_data[subject][series]['components_to_remove']
            
            # if length of the list is gerater than 0, then set the values of channels in the list to 0
            if len(components_to_remove)>0:
                eeg = subjects_data[subject][series]['eeg'].T
                mixing_matrix=subjects_data[subject][series]['mixing']
                unmixing_matrix=subjects_data[subject][series]['unmixing']
                modified_eeg = remove_sources(eeg, mixing_matrix, unmixing_matrix, components_to_remove)
                subjects_data[subject][series]['eeg'] = modified_eeg.T

            
            
            
def most_variance_idx(mixing_matrix, number_to_get=10):
    # compute variances and get 10 most variance components
    mat_var= np.var(mixing_matrix, axis=0)
    most_var_idxs= np.argsort(mat_var)
    most_var_idxs=most_var_idxs[-number_to_get:][::-1]
    
    return most_var_idxs
    

    
plot_components= util.plot_components

def statistical_significance(eeg_epochs):
    """
    Function to calculate whether the differences between HandStart and FirstDigitTouch ERPs are statistically significant
    Parameters:
        - eeg_epochs, a dictionary with string keys in the format "subject_n", where n is an integer. Each value contains an additional dictionary
        with string keys in the format "series_n", where in is an integer. Within each value for a series key is another dictionary with string keys for each event type
        (There are 6 event types: "HandStart", "FirstDigitTouch", BothStartLoadPhase", "LiftOff", "Replace", and "BothReleased"). Each value for an event type key contains a tuple with
        three entries. The first value in the tuple is the epoch data, which is a 3 dimensional array of floats shape (number of epochs, number of samples per epoch, number of channels 
        of EEG data). The second value in the tuple is a list of times representing the time of each sample in the epoch. Times correspond to sample indices such that the first entry 
        in the list of times corresponds to the first sample in the epoch and so on. For each event type, series, and subject, there should be approximately 270 epochs (approximately 
        30 trials were performed per series and each trial included each event type).
    Returns:
        - statistic_dictionary, a dictionary with string keys in the format "subject_n", where n is an integer. Each value contains an additional dictionary with string keys in the
        format "series_n", where in is an integer. Each value contains an array of floats in the shape n x m, where n is the number of samples per epoch and m is the number of eeg channels.
        each individual entry is the z-statistic produced by a two-sample welch's t-test.
        - p_dictionary, a dictionary with string keys in the format "subject_n", where n is an integer. Each value contains an additional dictionary with string keys in the
        format "series_n", where in is an integer. Each value contains an array of floats in the shape n x m, where n is the number of samples per epoch and m is the number of eeg channels.
        each individual entry is the p-value produced by a two-sample welch's t-test.
        - result_dictionary, a dictionary with string keys in the format "subject_n", where n is an integer. Each value contains an additional dictionary with string keys in the
        format "series_n", where in is an integer. Each value contains an array of booleans in the shape n x m, where n is the number of samples per epoch and m is the number of eeg channels.
        each individual entry is True if the p-value produced by a two-sample welch's t-test is less than or equal to 0.025 (dividing the significance level of 0.05 by 2 for a two-sample t-test).
    """
    #create dictionaries
    statistic_dictionary = {}
    p_dictionary = {}
    result_dictionary = {}
    
    #for each subject:
    for subject_key, subject_data in eeg_epochs.items():
        #create an empty sub-dictionary
        statistic_dictionary[subject_key] = {}
        p_dictionary[subject_key] = {}
        result_dictionary[subject_key] = {}
        
        #for each series in the subject:
        for series_key, series_data in subject_data.items():
            #initialize arrays 
            statistic_dictionary[subject_key][series_key] = np.full((np.shape(series_data["HandStart"][0])[1], np.shape(series_data["HandStart"][0])[2]), 0.0)
            p_dictionary[subject_key][series_key] = np.full((np.shape(series_data["HandStart"][0])[1], np.shape(series_data["HandStart"][0])[2]), 0.0)
            result_dictionary[subject_key][series_key] = np.full((np.shape(series_data["HandStart"][0])[1], np.shape(series_data["HandStart"][0])[2]), False)
            
            #for each sample in the series:
            for sample_index in range(np.shape(series_data["HandStart"][0])[1]):
                #for each channel in the sample:
                for channel_index in range(np.shape(series_data["HandStart"][0])[2]):
                    #perform the two-sample t test (welch's t-test for unequal variances)
                    sample_a = series_data["HandStart"][0][:, sample_index, channel_index]
                    sample_b = series_data["FirstDigitTouch"][0][:, sample_index, channel_index]
                    result = ttest_ind(sample_a, sample_b, equal_var=False)
                    
                    #extract the test statistic and p-value, and calculate a pass or fail at the alpha=0.05 significance level
                    statistic_dictionary[subject_key][series_key][sample_index][channel_index] = result.statistic
                    p_dictionary[subject_key][series_key][sample_index][channel_index] = result.pvalue
                    if result.pvalue < 0.025:
                        result_dictionary[subject_key][series_key][sample_index][channel_index] = True   
    #return the results
    return statistic_dictionary, p_dictionary, result_dictionary

    
def plot_eeg_significance(eeg_epochs, epoch_times, result_dictionary, subjects_to_plot, channels_to_plot, channel_names):
    """
    Function to plot the ERP values for each series of each subject by channel for the Hand Start and First Digit Touch events.
    Parameters:
        - eeg_epochs, a dictionary with string keys in the format "subject_n", where n is an integer. Each value contains an additional dictionary
        with string keys in the format "series_n", where in is an integer. Within each value for a series key is another dictionary with string keys for each event type
        (There are 6 event types: "HandStart", "FirstDigitTouch", BothStartLoadPhase", "LiftOff", "Replace", and "BothReleased"). Each value for an event type key contains a tuple with
        three entries. The first value in the tuple is the epoch data, which is a 3 dimensional array of floats shape (number of epochs, number of samples per epoch, number of channels 
        of EEG data). The second value in the tuple is a list of times representing the time of each sample in the epoch. Times correspond to sample indices such that the first entry 
        in the list of times corresponds to the first sample in the epoch and so on. For each event type, series, and subject, there should be approximately 270 epochs (approximately 
        30 trials were performed per series and each trial included each event type).
        - epoch_times, n x 1 array of floats, where n is the number of samples per epoch. Each entry represents a time at which a sample was taken. 
        - result_dictionary, a dictionary with string keys in the format "subject_n", where n is an integer. Each value contains an additional dictionary with string keys in the
        format "series_n", where in is an integer. Each value contains an array of booleans in the shape n x m, where n is the number of samples per epoch and m is the number of eeg channels.
        each individual entry is True if the p-value produced by a two-sample welch's t-test is less than or equal to 0.025 (dividing the significance level of 0.05 by 2 for a two-sample t-test).
        - subjects_to_plot, n x 1 array of ints, where each integer represents one of n subjects of the experiment that will plotted
        - channels_to_plot, n x 1 array of ints, where each integer represents one of n eeg channels in the experiment that will be plotted
        - channel_names, n x 1 array of strings, where each string represents the name of one of the n total eeg channels in the experiment
    Returns:
        None. All plots presented via matplotlib and saved as png images.
    """
    #for each subject being plotted
    for subject_index in subjects_to_plot:
        subject_key = f"subject_{subject_index}"
        #for each series in that subject 
        for series_key in eeg_epochs[subject_key].keys():
            plt.figure(figsize=(15, 12))
            plt.suptitle(f"Statistical Significance for Subject {subject_index}, {series_key}")
            #for each channel being plotted
            for channel_index in range(len(channels_to_plot)):
                channel_key = channels_to_plot[channel_index]
                
                #calculate the data to plot
                handstart_data = np.mean(eeg_epochs[subject_key][series_key]["HandStart"][0], axis=0)[:, channel_key]
                firstdigittouch_data = np.mean(eeg_epochs[subject_key][series_key]["FirstDigitTouch"][0], axis=0)[:, channel_key]
                results_indices = np.asarray(result_dictionary[subject_key][series_key][:, channel_key])
                significant_x_positions = epoch_times[results_indices]
                
                #perform the plotting
                plt.subplot(len(channels_to_plot),1,channel_index + 1)
                plt.title(f"ERPs for Channel {channel_names[channel_key]}")
                plt.plot(epoch_times, handstart_data, label="Hand Start Event")
                plt.plot(epoch_times, firstdigittouch_data, label="First Digit Touch Event")
                plt.scatter(significant_x_positions, handstart_data[results_indices])
                plt.xlabel("Time from event onset (s)")
                plt.ylabel("ERPs")
                if channel_index == 0:
                    plt.legend()
            plt.tight_layout() # add padding to accomodate legend
            plt.savefig(f"significance_plot_subject{subject_index}_{series_key}.png")
            #plt.show()
            plt.close()

# %% plot power significance

def get_event_eeg(eeg_epochs, event_types=['HandStart', 'FirstDigitTouch']):
    """

    Parameters
    ----------
    eeg_epochs : Dictionary of size S where S is the number of subjects
        for which data was loaded.
        EEG epoch data organized by subject, series, and event type. 
        - First level keys represent distinct subjects. 
        - Within each subject's key, there is a key for each series, there are 9 
        series per subject. 
        - Within each series key, there is a key for each event type (There are 6 
        event types: "HandStart", "FirstDigitTouch", BothStartLoadPhase", "LiftOff", 
        "Replace", and "BothReleased".)
        - Within each event type key, there is a tuple. The first value in the tuple
        is the epoch data, which is a 3 dimensional array of shape (number of epochs,
        number of samples per epoch, number of channels of EEG data). The second
        value in the tuple is a list of times representing the time of each
        sample in the epoch. Times correspond to sample indices such that the
        first entry in the list of times corresponds to the first sample in the 
        epoch and so on. For each event type, series, and subject, there should be 
        approximately 270 epochs (approximately 30 trials were performed per
        series and each trial included each event type).
        epoch_times: List of float. Size (Se,) where Se is the number of samples
        in an epoch.
            A list of times representing the time of each sample in the epoch. 
            Times correspond to sample indices such that the first entry in the list 
            of times corresponds to the first sample in the epoch and so on. These 
            values are included in all_subject_epochs as well as this list so that 
            epoch times can be easily accessed when working on epoch data or easily 
            accessed when later working on erps. 
    event_types : (list of strings), optional
        Event names to grab from eeg_epochs. The default is ['HandStart', 'FirstDigitTouch'].

    Returns
    -------
    eeg_epochs_list : (list of floats), size (E, R, V, T, C) where E is the number of subjects, R is the number of series, V is the number of events,
                                                              T is the number of trials, and C is the number of channels.
        EEG epochs by event.
        
    From the EEG epochs dictionary, extract events in event_types

    """
    
    #result list
    eeg_epochs_list=[]
    

    # iterate subject
    for subject_idx in range(1,len(eeg_epochs)+1):
        subject = f'subject_{subject_idx}'
        eeg_epochs_subject=[]
        # iterate series
        for series_idx in range(1, 10):
            series= f'series_{series_idx}'
            
            eeg_epochs_events =[]
            
            # get the event epochos only specified in event_types
            for event_type in event_types:
                eeg_epochs_events.append(np.transpose(np.array(eeg_epochs[subject][series][event_type][0]), axes=(0,2,1)))
            eeg_epochs_subject.append(eeg_epochs_events)
        eeg_epochs_list.append(eeg_epochs_subject)
    
    return eeg_epochs_list


def get_frequency_spectrum(eeg_epochs, fs=500):
    """
    Parameters
    ----------
    eeg_epochs : (array of floats), size (V, T, C, S) where  V is the number of events, T is the number of trials, C is the number of channels, and S is the number of samples.
        Epoch data
    fs : (int or floats), optional
        sampling rate. The default is 500.

    Returns
    -------
    eeg_epochs_fft : (array of floats), size (V, T, C, F) where  V is the number of events, T is the number of trials, C is the number of channels, 
                                                                    and F is the number of frequencies.
    fft_frequencies : (array of floats), size F where F is the number of frequencies
        frequencies

    """
    
    # calculate FT on each channel
    eeg_epochs_fft = scipy.fft.rfft(eeg_epochs)
    
    # calculate frequencies
    sample_count = eeg_epochs.shape[-1]
    total_duration = sample_count/fs
    fft_frequencies = np.arange(0,eeg_epochs_fft.shape[-1])/total_duration 
    
    return eeg_epochs_fft, fft_frequencies

channels_str_zz = np.array(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
       'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
       'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
       'O2', 'PO10'])




def get_power_spectrum(eeg_epochs_fft):
    '''
    Parameters
    ----------
    eeg_epochs_fft :(array of floats), size (V, T, C, F) where  V is the number of events, T is the number of trials, C is the number of channels, 
                                                                    and F is the number of frequencies.

        
    Returns
    -------
    poewr_db : (array of floats), size (V, T, C, F) where  V is the number of events, T is the number of trials, C is the number of channels, 
                                                                    and F is the number of frequencies.
        Calculate power of FFT

    '''
    
    event_count = eeg_epochs_fft.shape[0]
    channel_count = eeg_epochs_fft.shape[-2]

    
    # calculate mean power spectra for each channel
    signal_power = abs(eeg_epochs_fft)**2 # calculate power by squaring absolute value    
    power= np.mean(signal_power, axis=-3)
    
    # normalize
    norm_power = power/np.reshape(np.max(power, axis=-1), (event_count, channel_count, 1))
    
    # power decibel
    power_db = 10*np.log10(norm_power)

    return power_db




def plot_power_spectrum_series(power_db, fft_frequencies, subject, series, min_frequency=0, max_frequency=30, event_types=['HandStart', 'FirstDigitTouch'], channels_to_plot=['Fz', 'Pz', 'Oz'], channels=channels_str_zz):
    """
    Parameters
    ----------
    power_db : (array of floats), size (E, C, F) where  E is the number of events, C is the number of channels, 
                                                                    and F is the number of frequencies.
        Power spectrum
    fft_frequencies : (array of floats), size F where F is the number of frequencies
        frequencies
    subject : (int)
        subject number used for title
    series : (int)
        series number used for title
    min_frequency : (int or float), optional
        minimum frequency to zoom in. The default is 0.
    max_frequency : (int or float), optional
        maximum frequency to zoom in. The default is 30.
    event_types : (list of strings), optional
        Event names to grab from eeg_epochs. The default is ['HandStart', 'FirstDigitTouch'].
    channels_to_plot : (list of strings), optional
        Channel names to plot. The default is ['Cz', 'C3', 'C4', 'Pz', 'Fz'].
    channels : (array of strings), optional
       names of all channels. The default is channels_str.

    Returns
    -------
    None.
    
    Plot power spectrum of a series
    """
    
    # number of channels
    channels_to_plot_count = len(channels_to_plot)
    # channels indices
    channels_to_plot_idxs = np.where(channels==np.array(channels_to_plot)[...,None])[1]
    # number of events
    event_count = len(event_types)
    
    
    
    # plot
    plt.figure(figsize=(7,2+2*channels_to_plot_count))
    

    x= fft_frequencies
    
    # subplot list
    subplots=[]
    
    for subplot_idx in range(channels_to_plot_count):
        
        # x axis only defined at subplot 0 and all the other subplots share this x axis
        if subplot_idx==0:
            channel_subplot = plt.subplot(channels_to_plot_count, 1, subplot_idx+1)            
        else:
            channel_subplot = plt.subplot(channels_to_plot_count, 1, subplot_idx+1, sharex= subplots[0])
                
        subplots.append(channel_subplot)
        
        # get the index of the channel
        channel_idx = channels_to_plot_idxs[subplot_idx]
        
        # plot all the events in this channel
        for event_idx in range(event_count):
            channel_subplot.plot(x,power_db[event_idx][channel_idx], label  = event_types[event_idx])

        
        # labels
        channel_subplot.set_xlabel('frequency (Hz)')
        channel_subplot.set_ylabel('power (db)')
        channel_subplot.grid()
        channel_subplot.legend()
        channel_subplot.set_title(f'Channel {channels_to_plot[subplot_idx]}')
        
    
    # titles and save
    plt.suptitle(f'Subject {subject} series {series} Frequency content')

    plt.tight_layout()
    
    path= f'plots/spectrum/{subject:02}'
    
    if not os.path.exists(path):
        os.makedirs(path)
    event_str='_'.join(event_types)
    channel_str = '_'.join(channels_to_plot)
    file= path+f'/{subject:02}_series_{series}_{event_str}_{channel_str}.png'
    
    plt.xlim(min_frequency, max_frequency)
    plt.savefig(file)
    
    plt.close()


def plot_power_spectrum(eeg_epochs,  min_frequency=0, max_frequency=30, event_types=['HandStart', 'FirstDigitTouch'], channels_to_plot=['Cz', 'C3', 'C4', 'Pz', 'Fz'], channels=channels_str_zz):
    """
    Parameters
    ----------
    eeg_epochs : (list of floats), size (E, R, V, T, C) where E is the number of subjects, R is the number of series, V is the number of events,
                                                              T is the number of trials, and C is the number of channels.
        EEG epochs by event.
    min_frequency : (int or float), optional
        minimum frequency to zoom in. The default is 0.
    max_frequency : (int or float), optional
        maximum frequency to zoom in. The default is 30.
    event_types : (list of strings), optional
        Event names to grab from eeg_epochs. The default is ['HandStart', 'FirstDigitTouch'].
    channels_to_plot : (list of strings), optional
        Channel names to plot. The default is ['Cz', 'C3', 'C4', 'Pz', 'Fz'].
    channels : (array of strings), optional
       names of all channels. The default is channels_str.

    Returns
    -------
    power : (list of floats), size (E, R, V, C, F) where E is the number of subjects, R is the number of series, V is the number of events,
                                                              C is the number of channels, and F is the number of frequencies.
        power spectrum of events.
    fft_frequencies : (array of floats), size F where F is the number of frequencies
        Array of frequencies
        
    Plot power spectrum of all subjects

    """
    
    # number of subjects and series
    subject_count = len(eeg_epochs)
    series_count = len(eeg_epochs[0])
    
    # result list
    power =[]
    
    # iterate subject
    for subject_idx in range(subject_count):
        power_subject=[]
        # iterate series
        for series_idx in range(series_count):
            # get the all the trials of series
            eeg_epochs_series = eeg_epochs[subject_idx][series_idx]
            eeg_epochs_series= np.array(eeg_epochs_series)
            # get the FFT of trials
            eeg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs_series)
            
            #print(eeg_epochs_fft.shape)
            # get the power spectrum
            power_db_epochs = get_power_spectrum(eeg_epochs_fft)
            power_subject.append(power_db_epochs)
            # plot power spectrum of this series
            plot_power_spectrum_series(power_db_epochs, fft_frequencies, subject_idx+1, series_idx+1,min_frequency=min_frequency, max_frequency=max_frequency, \
                                       event_types=event_types, channels_to_plot=channels_to_plot, channels=channels )
        power.append(power_subject)
    return np.array(power), fft_frequencies
    
        

    
    

