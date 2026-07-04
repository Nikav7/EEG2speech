# CODE FOR ALIGNMENT WITH MATLAB LOGS
import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
from scipy.signal import butter
from preprocess.loadXDF import loadXDF
from mne.preprocessing import regress_artifact, ICA
from mne_icalabel import label_components
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
import spkit as sp
from matplotlib.widgets import Button
import re
import scipy.io as sio

folder = r'C:\Users\Thesis\data\TriParadigm\experimentDATA'

all_logs = {}

# Process all subjects
import os

for subj_num in range(15, 17):
    
    subj = f"sub-P{subj_num:03d}"
    subj_path = os.path.join(folder, subj)
    if not os.path.isdir(subj_path):
        print(f"Subject folder not found: {subj_path}")
        continue
    
    logs_path = os.path.join(subj_path, f"logs{subj_num}.mat")
    
    # Check if log file exists
    if not os.path.exists(logs_path):
        print(f"Log file not found: {logs_path}")
        continue
    
    try:
        print(f"Processing subject {subj_num}...")
        logs_data = sio.loadmat(logs_path)
        
        logs = np.array(logs_data['experimentLog'])
        
        # Extract values from nested arrays in 4th column and flatten into single array
        times = []
        for item in logs[:, 3]:
            if isinstance(item, np.ndarray):
                times.extend(item.flatten())
            else:
                times.append(item)
        
        blocks = []
        for item in logs[:, 0]:
            if isinstance(item, np.ndarray):
                blocks.extend(item.flatten())
            else:
                blocks.append(item)
        
        markers = []
        for item in logs[:, 2]:
            if isinstance(item, np.ndarray):
                markers.extend(item.flatten())
            else:
                markers.append(item)
        
        times = np.array(times)
        blocks = np.array(blocks)
        markers = np.array(markers)
        samples = (times * 1000).astype(int)
        
        logs_ = np.column_stack((samples, blocks, markers))
        # adjust markers
        logs_[blocks == 2] += 200
        logs_[blocks == 3] += 100
        logs_[blocks == 4] += 300
        
        durations = np.zeros(len(markers), dtype=int)
        # events as mne format (sample, duration, marker id)
        marker_events = np.column_stack((samples, durations, logs_[:, 2]))
        
        # Store in dictionary
        all_logs[subj_num] = marker_events
        
        # --- Save events as .npy ---
        np.save(f'data/eventsLogs_subj{subj_num}.npy', marker_events)
        print(f"Saved events for subject {subj_num}")
        
    except Exception as e:
        print(f"Failed to process subject {subj_num}: {e}")
