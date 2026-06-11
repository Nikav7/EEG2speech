import os
import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd

# load XDF data, create mne raw and events
def loadXDF(file_path):
    streams, header = pyxdf.load_xdf(file_path)
    print(f"Successfully loaded {len(streams)} streams from {file_path}")

    # Collect stream metadata for inspection
    stream_metadata = []
    for i, stream in enumerate(streams):
        info = stream.get('info', {})
        stream_info = {
            'Stream Index': i,
            'Name': info.get('name', ['N/A'])[0],
            'Type': info.get('type', ['N/A'])[0],
            'Sampling Rate (Hz)': float(info.get('nominal_srate', [0])[0]),
            'Channel Count': int(info.get('channel_count', [0])[0]),
            'Data Shape': stream['time_series'].shape if 'time_series' in stream and isinstance(stream['time_series'], np.ndarray) else (len(stream['time_series']) if 'time_series' in stream else 'N/A'),
            'First Timestamp': stream['time_stamps'][0] if 'time_stamps' in stream and len(stream['time_stamps']) > 0 else 'N/A',
            'Last Timestamp': stream['time_stamps'][-1] if 'time_stamps' in stream and len(stream['time_stamps']) > 0 else 'N/A',
            'Duration (s)': (stream['time_stamps'][-1] - stream['time_stamps'][0]) if 'time_stamps' in stream and len(stream['time_stamps']) > 0 else 'N/A'
        }
        stream_metadata.append(stream_info)
    metadata_df = pd.DataFrame(stream_metadata)
    print(metadata_df.to_string())
    print("-" * 30)

    # Find EEG and marker streams
    eeg_stream = None
    marker_stream = None
    for stream in streams:
        info = stream.get('info', {})
        stype = info.get('type', [''])[0]
        name = info.get('name', [''])[0]
        if stype == 'EEG' or name == 'EEG':
            eeg_stream = stream
        if stype == 'Markers' and name in ['VeroExperimentMarkers', 'VeronicaExperimentMarkers']:
            marker_stream = stream

    # Extract EEG data
    data = np.array(eeg_stream['time_series'])
    fs = float(eeg_stream['info']['nominal_srate'][0])
    ch_names = ['Fp1','Fz','F3','F7','EOG1','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz','P3','P7','O1','EOG3','O2','P4','P8','TP10','CP6','CP2','Cz','C4','T8','EOG2','FC6','FC2','F4','F8','Fp2']
    eog_channels = ['EOG1', 'EOG3', 'EOG2']
    ch_types = ['eog' if ch in eog_channels else 'eeg' for ch in ch_names]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    
    # streams alignment
    eeg_first_timestamp = eeg_stream['time_stamps'][0]
    marker_first_timestamp = marker_stream['time_stamps'][0]
    time_offset = marker_first_timestamp - eeg_first_timestamp
    start_sample_offset = int(time_offset * fs)
    print("SAMPLE OFFSET: ", start_sample_offset, start_sample_offset/fs, " seconds" )
    # crop eeg and timestamps to align with the first marker, new 't=0'
    data = np.array(eeg_stream['time_series'][start_sample_offset:])
    eeg_times = eeg_stream['time_stamps'][start_sample_offset:]
    marker_times = np.array(marker_stream['time_stamps'])
    adjusted_marker_times = marker_times  - eeg_times[0]
   
    event_samples = (adjusted_marker_times * fs).astype(int)
    # marker labels
    marker_values = np.array([int(float(m[0])) for m in marker_stream['time_series']], dtype=int)
    # duration - zeros
    durations = np.zeros(len(marker_values), dtype=int)
    # events as mne format (sample, duration, marker id)
    marker_events = np.column_stack((event_samples, durations, marker_values))
    data = data / 1e6  # convert microvolts to volts
    eeg_duration_minutes = data.shape[0] / (fs * 60) 
    print(f"EEG shape: {data.shape}, Duration: {eeg_duration_minutes:.2f} minutes")
    
    raw = mne.io.RawArray(data.T, info)

    return data, raw, info, marker_events