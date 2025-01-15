# Theraputic Robotic Exoskeleton

In this project, we analyze EEG data with the intention of improving brain-computer interfaces (BCIs) aimed at restoring hand function in individuals with motor impairments, such as those caused by stroke. Specifically, we examine EEG signals related to events such as initiating hand movement in order to identify features that could inform BCIs for movement rehabilitation. 

This project was implemented by Alaina Birney, Yoonki Hong, and Ashley Heath of the University of Vermont for the course Brain Computer Interfaces. A full report on this project can be found within the file "BCI_Project_3_Report.pdf": https://github.com/abirney123/Exoskeleton/blob/b60fe66f92959574c19d7306a7c4dc82aaac78b8/BCI_Project_3_Report.pdf

## README for WAY-EEG-GAL dataset
Multi-channel EEG Recordings During 3,936 Grasp and Lift Trials with Varying Weight and Friction

#### *** the name of data directory is set to 'WAY-EEG-GAL' at default. 
Data of each subject must be in the sub folder of this folder its must be namped 'PX' where X is the number of subject. ###
ex -  WAY-EEG-GAL/P1/ ...
                   /P2/ ...


Twelve participants performed lifiting an object. Each participant performed 9 series of lifting and total 3936 trials were done.
In each trial, with a cue, the participant tried to grasp, lift and hold the object for seconds with the thumb and index finger. 
During a series EEG (32 channels), EMG (five arm and hand muscles), the 3D position of both the hand and object, and force/torque at both contact plates were recorded.

### Data Records

For each of the 12 participants, a single P structure is provided, and one HS structure and one WS structure are provided for each series. However, for a single weight series per participant, the non-EEG information was excluded, and is kept secret for a later competition. The total size of all MATLAB data structures, for all participants, stands at ~15 GB.

#### HS_P1_S1.mat—HS_P12_S9.mat (108 files)
The HS_PX_SY.mat file (where X is participant number and Y is series number), contains a structure
with all data in a single lifting series. For example, HS_P3_S2.mat contains the data for the Series 2 of Participant #3.

structure
-hs
    -hs.name        : participant's initial
    -hs.participant : particiapnt number
    -hs.series      : series number
    -hs.emg
        -hs.emg.names        : name of columns in hs.emg.sig matrix (5 channel names)
        -hs.emg.sig          : EMG signals. samples x 5 channels. (channel names found in hs.emg.names)
        -hs.emg.samplingrate : sampling rate
    -hs.eeg 
        -hs.eeg.names        : name of columns in hs.eeg.sig matrix (32 channel names)
        -hs.eeg.sig          : EEG signals. samples x 32 channels. (channel names found in hs.eeg.names)
        -hs.eeg.samplingrate : sampling rate.
    -hs.kin
        -hs.kin.names         : names of 24 sensors and 12 force plates.
        -hs.kin.sig           : sensor and force signals. samples x 36 channels. (channel names found in hs.kin.names) 
        -hs.kin.samplingrate  : sampling rate
    -hs.env
        -hs.env.names         : surface and weight
        -hs.env.sig           : surface and weight data. samples x 2 channels. (channel names found in hs.env.names) 
        -hs.env.samplingrate  : sampling rate
    -hs.misc
        -hs.misc.names        : names of miscellaneous data (Button, Magnet, SurfaceLED, ParticipantLED, Temperature1, Temperature2)
        -hs.misc.sig          : miscellaneous signals. samples x 6 channels. (channel names found in hs.misc.names) 
        -hs.misc.samplingrate : sampling rate


#### HS_P1_ST.mat—HS_P12_ST.mat (12 files)
Each of these files contains the eeg matrix, but not emg, kin, env, or misc.

structure
-hs
    -hs.name        : participant's initial
    -hs.participant : particiapnt number
    -hs.series      : series number
    -hs.eeg 
        -hs.eeg.names        : name of columns in hs.eeg.sig matrix (32 channel names)
        -hs.eeg.sig          : samples x 32 channels
        -hs.eeg.samplingrate : sampling rate



#### WS_P1_S1.mat—WS_P12_S9.mat (108 files)
The WS_PX_SY.mat files contains a structure with the data organized in windows around every single lift, to allow easy extraction of single trials. 
For example, WS_P2_S3.mat contains the data from the 3rd series of Participant 2.

structure
-ws
    -ws.name : participant's initial
    -ws.participant : particiapnt number
    -ws.series      : series number
    -ws.names 
        -ws.names.eeg : name of columns in the ws.win(n).eeg matrix. (32 channels)
        -ws.names.kin : name of columns in the ws.win(n).kin matrix. (45 channels)
        -ws.names.emg : name of columns in the ws.win(n).emg matrix. (5 channels)
    -ws.win (A structure for each single lifting trial containing the following fields)
        -ws.win.eeg              : samples × 32 channels (channel names found in ws.names.eeg)
        -ws.win.kin              : samples × 45 channels (channel names found in ws.names.kin)
        -ws.win.emg              : samples × 5 channels (channel names found in ws.names.emg)
        -ws.win.eeg_t            : samples × 1 giving time of each row in ws.win.eeg and ws.win.kin
        -ws.win.emg_t            : samples × 1 giving time of each row in ws.win.emg
        -ws.win.trial_start_time : absolute starting time ( =StartTime in AllLifts,)
        -ws.win.LEDon            : time of LED onset (=LEDOn in AllLifts)
        -ws.win.LEDoff           : time of LED offset (=LEDOff in AllLifts)
        -ws.win.weight           : integer corresponding to weight (CurW in AllLifts)
        -ws.win.weight_id        : text representing the weight (e.g., ‘330 g’)
        -ws.win.surf             : integer corresponding to surface (CurS in AllLifts)
        -ws.win.surf_id          : text representing the surface (e.g., ‘silk’)
        -ws.win.weight_prev      : integer corresponding to weight in the previous trial
        -ws.win.weight_prev_id   : text representing the weight in previous trial
        -ws.win.surf_prev        : integer corresponding to the surface in the previous trial
        -ws.win.surf_prev_id     : text representing the surface in previous trial

Nine derived signals are included in ws.win.kin,
Column 37 index finger load force
Column 38 thumb load force
Column 39 total load force
Column 40 index finger grip force
Column 41 thumb grip force
Column 42 averaged grip force
Column 43 index finger grip force/load force ratio
Column 44 thumb grip force/load force ratio
Column 45 total grip force/load force ratio


#### P1_AllLifts.mat—P12_AllLifts.mat (12 files)
The PX_AllLifts.mat file contains a structure P with information about every lift performed by each participant X, such as
the times at which specific events occurred. 
The matrix P.AllLifts contains one row for each recorded lifting trial and 43 columns that each represents a variable pertaining to single trials. The names of the columns in P.AllLifts can be found in P.ColNames.

structure
-P
    -P.AllLifts : trials x 43 variables
    -P.ColNames : names of variables

variables table

column    - variable            - unit       - description
1         - Part                - integer    - Participant number   
2         - Run                 - integer    - Series number
3         - Lift                - integer    - Sequential trial within series
4         - CurW                - integer    - Current weight—[1=165 g, 2=330 g, 4=660 g]
5         - CurS                - integer    - Current surface—[1 =sandpaper, 2 =suede, 3=silk]
6         - PrevW               - integer    - Weight in previous Lift—[1=165 g, 2=330 g, 4=660 g]
7         - PrevS               - integer    - Surface in previous Lift—[1 =sandpaper, 2 =suede, 3=silk]
8         - StartTime           - seconds    - Start time relative to start of series.
9         - LEDOn               - seconds    - Time when the LED in the Perspex plate was turned on; this the signal to the participant to commence a Lift (always 2)
10        - LEDOff              - seconds    - Time when the LED in the Perspex plate was turned off; this was the signal to the participant to replace the object
11        - BlockType           - integer    - Type of Series—[1=Weight series; 2=Friction series; 3 =Mixed weight and friction series]
12        - tIndTouch           - seconds    - Time when the index finger touched the object
13        - tThumbTouch         - seconds    - Time when the thumb touched the object
14        - tFirstDigitTouch    - seconds    - Time when the first digit touched the object
15        - tBothDigitTouch     - seconds    - Time when both digits have touched the object
16        - tIndStartLoadPhase  - seconds    - Time when the index finger start to apply load force
17        - tThuStartLoadPhase  - seconds    - Time when the thumb finger start to apply load force
18        - tBothStartLoadPhase - seconds    - Time when both digits have started to apply load force
19        - tLiftOff            - seconds    - Time when the object lifted off from the support
20        - tReplace            - seconds    - Time when the object was replaced on the support
21        - tIndRelease         - seconds    - Time when the index finger released the object
22        - tThuRelease         - seconds    - Time when the thumb released the object
23        - tBothReleased       - seconds    - Time when both digits have released the object
24        - GF_Max              - N          - Maximum grip force (mean of the maximum GF applied by the index finger and the thumb)
25        - LF_Max              - N          - Maximum load force (sum of the maximum LF applied by the index finger and the thumb)
26        - dGF_Max             - N/s        - Maximum GF rate
27        - dLF_Max             - N/s        - Maximum LF rate
28        - tGF_Max             - seconds    - Time when the maximum GF occurred
29        - tLF_Max             - seconds    - Time when the maximum LF occurred
30        - tdGF_Max            - seconds    - Time when the maximum GF rate occurred
31        - tdLF_Max            - seconds    - Time when the maximum LF rate occurred
32        - GF_hold             - N          - Mean GF in a 200 ms time window starting 300 ms before LEDOff
33        - LF_hold             - N          - Mean LF in a 200 ms time window starting 300 ms before LEDOff
34        - tHandStart          - seconds    - Time when the hand starts to move (after LEDOn)
35        - tHandStop           - seconds    - Time when the hand stops (returned to blue area)
36        - tPeakVelHandReach   - seconds    - Time when the tangential hand velocity reaches its maximum during the reaching phase
37        - tPeakVelHandRetract - seconds    - Time when the tangential hand velocity reaches its maximum during the retraction phase
38        - GripAparture_Max    - cm         - Maximum grip aperture (MGA) during the reaching movement
39        - tGripAparture_Max   - seconds    - Time of MGA
40        - Dur_Reach           - seconds    - Duration of the reaching phase (from start of hand movement to initial object touch)
41        - Dur_Preload         - seconds    - Duration of the preload phase, i.e., from digit contact until LF application commenced
42        - Dur_LoadPhase       - seconds    - Duration of the load phase, i.e., from LF was applied until object lift-off
43        - Dur_Release         - seconds    - Duration of the release phase, i.e., from moment the object was replaced on the table (tReplace) until both digits had released the object




## References

#### Dataset, description
Multi-channel EEG Recordings During 3,936 Grasp and Lift Trials with Varying Weight and Friction
https://www.nature.com/articles/sdata201447#Sec11

#### Dataset, download
https://figshare.com/collections/_/988376





## Project 3

With this dataset, we will clean, analyze, and process to build a machine learning model to classify which type of event the participant is experiencing. We will mainly use EEG and EEG related data. Therefore, we reconstructed the original dataset by excluding certain types of data including EMG values, object surface types, angle and 3d position of sensors, and so on. More info about the original dataset are found in the link above. 

Our reconstructed data can be loaded using the code in test_project_3.py

```python
import Project_3 as P3

# Create combined .mat file for each subject
# create files for five subjects to get a sufficiently large sample
#subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # in case we want to load all 12 subjects data
subjects = [1, 3, 5]
P3.combine_files(subjects)
# Load data for each subject with a combined .mat file, store in a single dictionary
subjects_data={}
for subject in subjects:  
    subjects_data[f"subject_{subject}"]= P3.load_subject_data(subject) 
```


Details about the fields of our reconstructed data are as follows:
**Notes:**
number of fields can be different depending on how many subjects data we want to load.
Here, we assume we want to load subject 1,3 and 5.
The data is a nested dictionary, so an inner dictionary is represented with an indentation.
When there are multiple same structure dictionaries on the same level, only one dictionary is expanded to show inner dictionaries.

Times are recorded in seconds.
EEG values are recorded in uV (microvolts).

- subject_1 
    - series_1 
        - hs_time_first_eeg : relative time to the onset of the series that the first EEG data 
                               in 'eeg' was recorded
        - eeg               : EEG data. Each row is a channel and each column is a sample
        - channel_names     : the name of each channel, in the same order as the eeg matrix.
        - trial_start_time  : relative start time of trials to the onset of the series.
        - trial_end_time    : relative end time of trials to the onset of the series.
    - series_2
    - series_3
    - series_4
    - series_5
    - series_6
    - series_7
    - series_8
    - series_9
    - all_lifts : each row contains one recorded lifting trial and each column represents a variable
                  pertaining to single trials
    - all_lifts_column_name* : each element contains a variable name corresponding to a column in 
                               'all_lifts'.
- subject_3
    - ... same as subject_1
- subject_5
    - ... same as subject_1


* all_lifts_column_name
Part                : subject number
Run                 : series number
Lift                : sequential trial within series
LEDOn               : time when the LED in the Perspex plate was turned on; this the signal to the participant to 
                      commence a Lift (always 2)
LEDOff              : time when the LED in the Perspex plate was turned off; this was the signal to the participant 
                      to replace the object
tIndTouch           : time when the index finger touched the object
tThumbTouch         : time when the thumb touched the object
tFirstDigitTouch    : time when the first digit touched the object
tBothDigitTouch     : time when both digits have touched the object
tIndStartLoadPhase  : time when the index finger start to apply load force
tThuStartLoadPhase  : time when the thumb finger start to apply load force
tBothStartLoadPhase : time when both digits have started to apply load force
tLiftOff            : time when the object lifted off from the support
tReplace            : time when the object was replaced on the support
tIndRelease         : time when the index finger released the object
tThuRelease         : time when the thumb released the object
tBothReleased       : time when both digits have released the object
