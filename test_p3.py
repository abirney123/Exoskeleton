#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:39:04 2024

@author: Alaina Birney, Yoonki Hong, Ashley Heath

A python module closely related to Project_3.py. Calls functions in Project_3.py
and combines data to result in a single dictionary containing data for all
subjects. Additionally, separates EEG data into epochs for each event, gets
the time of each sample in each epoch relative to the event, calculates ERPs
for each event type, plots ERPs with indications of the event occurance 
for specified channels for events HandStart and FirstDigitTouch, creates and 
applies a bandpass filter to EEG data, generates mixing and unmixing matrices 
for EEG data, removes sources related to artifacts, gets the frequency and power 
spectra, plots the power spectra, calculates when there are statistically 
significant differences between ERPs related to the events HandStart and 
FirstDigitTouch, and generates plots to visualize when these statistically 
significant differences occur.

Please download participant data here https://figshare.com/collections/WAY_EEG_GAL_Multi_channel_EEG_Recordings_During_3_936_Grasp_and_Lift_Trials_with_Varying_Weight_and_Friction/988376
and place all participant data folders in a folder titled "WAY-EEG-GAL" to run this code. The files 
are too large to upload via gradescope.
"""
import Project_3 as P3
import numpy as np

#%% Create combined .mat file for each subject
subjects = [1,2,3,4,5,6,7,8,9,10,11,12]
P3.combine_files(subjects)
#%% Load data for each subject with a combined .mat file, store in a single dictionary
subjects_data={}
for subject in subjects:  
    subjects_data[f"subject_{subject}"]= P3.load_subject_data(subject) 
print("data loaded")


# %% create a filter and apply to the entire eeg data in the subjects_data

"""
Create a bandpass filter that passes only beta & mu rythms (8 - 30 hz) and apply it to all the series in the subjects_data dictionary.

After this cell, filtered EEG data will be replace the original EEG data of subjects_data dictionary.
"""

filter_beta = P3.make_band_pass_filter(8, 30, filter_order=100)

P3.filter_all_series(subjects_data, filter_beta)


    
# %% create mixing, unmixing matrices of EEG data of each series
import Project_3 as P3


"""
Performe FastICA on all series.
Create mixing and unmixing matrices and add them to the subjects_data dictionary with key 'mixing' and 'unmixing', and also add 'components_to_remove' key with an empty list.
In this list, indices of components to be removed will be stored.
If the algorithm doens't converge all empty list will be stroed in 'mixing' and 'unmixing' fields.
"""
# create mixing and unmixing matrices of all subjects
non_converge=P3.add_mixing_matrices_to_dict(subjects_data, 32)
    

#%% plot topological map of mixing matrix
"""
Plot topographical map of ICA components.
Plot all the components of all the series. Plots are saved in 'plots' folder
"""


channels = np.array(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
       'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
       'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
       'O2', 'PO10'])
#print(channels)
fs=500

for subject_idx in range(1,len(subjects)+1):
    subject = f'subject_{subject_idx}'
    for series_idx in range(1, 10):
        series= f'series_{series_idx}'
        mix = subjects_data[subject][series]['mixing']
        
        most_var_10 = P3.most_variance_idx(mix, number_to_get = 10)
        P3.plot_components(mix, channels, subject_idx, series_idx, True, most_var_10)
        
        for components_idxs in range(4):
            components_indices = range(components_idxs*8, (components_idxs+1)*8 )    
            P3.plot_components(mix, channels, subject_idx, series_idx, True, components_indices)
# %%
"""
components to remove
We decided the indices of components to remove for each series after analyzing the topo maps
and manually implemented them

For a component, if the range of values is huge, i.e, the difference of the greatest and least values of features is huge, 
and the changes of values in near areas are abrupt, then we chose that component as one to be removed

If the selected component is corresponding to one of Cz, C3, C4, Pz, and Fz, we didn't remove the component.
"""

# example 5-1, 5-6
# use 9-1, 12-8
subject_1_components_to_remove=[[0,1, 8, 19, 25], [25], [25], [30], [25], [25], [6, 22], [12], [12]]
subject_2_components_to_remove=[[12], [8,12], [17], [9, 17, 18], [0], [17], [3,11,17,23,31], [19], [12,13]]
subject_3_components_to_remove=[[29 ], [], [], [25 ], [17,24,25], [14], [17,29], [22], []]
subject_4_components_to_remove=[[12,28], [10, 16], [14], [0,6,11], [], [13,19,23], [0, 30], [0, 17, 21, 27], [1, 9, 19, 25]]
subject_5_components_to_remove=[[0, 19, 21, 24], [12,13], [11], [12],[22] , [12], [12], [12], [11]]
subject_6_components_to_remove=[[12], [12], [12], [12], [12], [12,28], [5], [12], [12]]
subject_7_components_to_remove=[[12], [12], [12], [3, 12], [12], [12], [12], [12], [12]]
subject_8_components_to_remove=[[6], [0], [22], [], [9], [8,16], [3,19,24], [8,9], [4]]
subject_9_components_to_remove=[[24,29 ], [12], [19], [12], [12], [16], [12], [5], [30]]
subject_10_components_to_remove=[[], [11], [], [12], [12], [12], [], [], [12]]
subject_11_components_to_remove=[[6,12], [19], [17], [13], [13, 22,30], [17], [], [24, 26], [24]]
subject_12_components_to_remove=[[19], [13,27], [8, 9, 14, 30], [6], [8], [18,21,22,30], [2,4, 22], [3, 11, 16,20, 21, 26], [4,8,11]]

all_subjects_components_to_remove=[subject_1_components_to_remove, subject_2_components_to_remove, subject_3_components_to_remove,
                                    subject_4_components_to_remove, subject_5_components_to_remove, subject_6_components_to_remove,
                                    subject_7_components_to_remove, subject_8_components_to_remove, subject_9_components_to_remove,
                                    subject_10_components_to_remove, subject_11_components_to_remove, subject_12_components_to_remove]


# set components_to_remove in the subjects_data dictionary
for subject_idx in range(1,len(subjects)+1):
    subject = f'subject_{subject_idx}'
    for series_idx in range(1, 10):
        series= f'series_{series_idx}'
        #print(subject+"_"+series)
        subjects_data[subject][series]['components_to_remove']= all_subjects_components_to_remove[subject_idx-1][series_idx-1]
        
P3.remove_sources_all_subjects(subjects_data)




#%% epoch data and get times of samples relative to epoch start after filtering

eeg_epochs, epoch_times = P3.epoch_data(subjects_data)

"""
# get epoch dictionary details
for subject, subject_data in eeg_epochs.items():
    print(f"level one key: {subject}")
    print(f"size: {len(subject_data)}") # should be 9 because there are 9 series per subject
    for series, series_data in subject_data.items():
        print(len(series_data)) 
        print(f"level two key: {series}")
        print(f"size: {len(series_data)}, type: {type(series_data)}") # should be 6 because there are 6 event types
        for event, event_data in series_data.items():
            print(f"level three key: {event}")
            # level three data is tuples, get size and type of each entry
            print(f"first element size: {event_data[0].shape}, type: {type(event_data[0])}")
            print(f"second element size: {event_data[1].shape}, type: {type(event_data[1])}")
            # tuple values represent epoch and epoch times, respectively. There should be approximately
            # 270 epochs (varies between subjects, array dimension 0) for each event type and 
            # as many times as samples per epoch (925, epoch array dimension 1)
"""

print("epochs formed")


# %% plot power spectra
"""
Plot power spectra of 'HandStart' and 'FirstDigitTouch' events epochs on Cz, C3, C4, Pz, and Fz for all series.
Plots are saved to plots folder
"""


event_types = ['HandStart', 'FirstDigitTouch']
event_epochs= P3.get_event_eeg(eeg_epochs, event_types)
power = P3.plot_power_spectrum(event_epochs)


    

#%% Get ERP for each event type
all_subject_erps = P3.get_erps(eeg_epochs)
print("ERPs calculated")

#%% Plot ERPs with event time indications for most important channels 

"""
plotting for Cz, C3, C4, Pz, and Fz because the readiness potential should
reflect activity in the supplementary motor area and in primary motor and sensory cortices
Cz is over the primary motor cortex while C3 is on the left 
hemisphere over the primary motor cortex. So, because subjects moved their right 
hands we may see more pronounced activity in C3. Pz and Fz are also of interest 
because they are located over the parietal region 
and frontal lobe, respectively. The parietal lobe is involved in sensory processing and
spatial orientation while the frontal lobe is involved in decision making and 
behavior control, which are also both related to voluntary movement.

Sources: Dr. J lectures (mainly EEG Rhythms and Bands) & Wolpaw text (Chapter 2, Neuronal
Activity in Motor Cortex and Related Areas).
"""
channels_to_plot = ["Cz", "C3", "Pz", "Fz"]
subjects_to_plot = subjects
events_to_plot = ["HandStart", "FirstDigitTouch"]
P3.plot_erps(all_subject_erps, eeg_epochs, subjects_data, channels_to_plot, subjects_to_plot, events_to_plot)    


#%% Statistical Analysis

"""
Determine if there is a statistically significant difference in 
ERPs for event types HandStart and FirstDigitTouch, create plots to show when
statistically significant differences occur.
"""
        
statistic_dictionary, p_dictionary, result_dictionary = P3.statistical_significance(eeg_epochs)   

P3.plot_eeg_significance(eeg_epochs, epoch_times, result_dictionary, [1,2,3,4,5], [4, 12, 13, 24], channels)



