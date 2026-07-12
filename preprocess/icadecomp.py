import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd
from scipy.signal import butter
from loadXDF import loadXDF
from mne.preprocessing import regress_artifact, ICA
from mne_icalabel import label_components
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
import spkit as sp
from matplotlib.widgets import Button
import re
import scipy.io as sio
import pickle
import time
import os
from pathlib import Path


def autodrop_badchansptp(raw_obj, z_thresh=6.0):
    """Drop EEG channels with extreme peak-to-peak amplitude (robust z-score)."""
    eeg_picks = mne.pick_types(raw_obj.info, eeg=True, eog=False)
    if len(eeg_picks) == 0:
        return []

    data = raw_obj.get_data(picks=eeg_picks)
    ptp = np.ptp(data, axis=1)
    med = np.median(ptp)
    mad = np.median(np.abs(ptp - med))
    if mad == 0:
        return []

    robust_z = 0.6745 * (ptp - med) / mad
    eeg_names = [raw_obj.ch_names[i] for i in eeg_picks]
    bads = [name for name, z in zip(eeg_names, robust_z) if np.abs(z) > z_thresh]
    if bads:
        raw_obj.info["bads"] = sorted(set(raw_obj.info["bads"] + bads))
        raw_obj.interpolate_bads(reset_bads=True)
    return bads

# load saved ICA components
def load_ica_components(subject_num):
    """Load pre-computed ICA components and mixing matrix for a subject."""
    ica_path = f'data/ica_components_subj{subject_num}.pkl'
    mixing_matrix_path = f'data/ica_mixing_matrix_subj{subject_num}.npy'
    sources_path = f'data/ica_sources_subj{subject_num}.npy'
    
    with open(ica_path, 'rb') as f:
        ica = pickle.load(f)
    mixing_matrix = np.load(mixing_matrix_path)
    sources = np.load(sources_path)
    
    print(f"Loaded ICA components for subject {subject_num}")
    print(f"  - Mixing matrix shape: {mixing_matrix.shape}")
    print(f"  - Sources shape: {sources.shape}")
    
    return ica, mixing_matrix, sources

# reconstruct original signal from sources and mixing matrix
def reconstruct_signal_from_sources(mixing_matrix, sources):
    """Reconstruct the original EEG signal from sources and mixing matrix."""
    reconstructed = mixing_matrix @ sources
    print(f"Reconstructed signal shape: {reconstructed.shape}")
    return reconstructed

# ============================================================================
# Set subject and session numbers
# ============================================================================
#SUBJECT_NUM = 15  # Subject (e.g., 15 for sub-P015)
#SESSION_NUM = 3   # Session (e.g., 3 for ses-S003)

SUBJECT_NUM = [15,16,17,18,19]
SESSION_NUM = [3,1,1,1,1]

BASE_DATA_PATH = Path(r"C:\Users\hssn_\Desktop\RAWEEG\Veronica_DataThesis")


def make_xdf_file_path(subject_num, session_num):
    return (
        BASE_DATA_PATH
        / f"sub-P{subject_num:03d}"
        / f"ses-S{session_num:03d}"
        / "eeg"
        / f"sub-P{subject_num:03d}_ses-S{session_num:03d}_task-Default_run-001_eeg.xdf"
    )


def process_subject_session(subject_num, session_num):
    xdf_file_path = make_xdf_file_path(subject_num, session_num)

    if not xdf_file_path.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_file_path}")

    output_dir = Path('extended_infomaxcomps025-120Hz') / f'subj{subject_num}'
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path('clean_data025-120Hz')
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================")
    print(f"Processing Subject {subject_num}, Session {session_num}")
    print(f"XDF file: {xdf_file_path}")
    print(f"Output directory: {output_dir}")
    print(f"============================================\n")

    data, raw, info, markers = loadXDF(str(xdf_file_path))
    #markers = marker_events
    raw.info = info

    print(markers)

    ch_names = info['ch_names']

    print(ch_names)

    # # # Just for subjects < 10, e.g. subject 4! but not for subject 5
    # info['ch_names'][20] = 'EOG2'
    # info['chs'][20]['ch_name'] = 'EOG2'
    # info['chs'][20]['kind'] = mne.io.constants.FIFF.FIFFV_EOG_CH

    # # # just subjects < 10!!
    # info['ch_names'][26] = 'FT10'
    # info['chs'][26]['ch_name'] = 'FT10'
    # info['chs'][26]['kind'] = mne.io.constants.FIFF.FIFFV_EEG_CH
    # raw.drop_channels(['FT10'])

    # # # Just for subject 5!
    # info['ch_names'][16] = 'Oz'
    # info['chs'][16]['ch_name'] = 'Oz'
    # info['chs'][16]['kind'] = mne.io.constants.FIFF.FIFFV_EEG_CH
    # raw.drop_channels(['Oz'])

    # reject chanels TP9 and TP10
    print(info['ch_names'])
    raw.drop_channels(['TP9', 'TP10'])
    print("channel names:")
    print(info['ch_names'])

    #pd.DataFrame(raw.ch_names, columns=['Channel']).to_csv(f'stand_data/channel_names.csv', index=False)

    # FILTERING
    # first IIR Butterworth bandpass filter 5th order between 1-120 Hz
    # to capture speech information
    l_freq=0.25
    h_freq=120.0
    order=4
    iir_params = dict(order=order, ftype='butter')
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params)
    #raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=40.0, method='fir', fir_design='firwin')
    #raw_filtered.plot(duration=10, events=markers, title='filtered EEG with Events')

    # secondly a Notch filter at 50 Hz and 100 Hz simultaneously
    raw_filtered = raw_filtered.copy().notch_filter(freqs=[50, 100], filter_length='auto', notch_widths=None, trans_bandwidth=1, method='fir', phase='zero', fir_window='hamming', fir_design='firwin', pad='reflect_limited', verbose=None)

    # downsampling to 250 Hz -> AFTER FILTERING, to keep control on what spectral content is kept before sample-rate reduction
    orig_sfreq = raw_filtered.info['sfreq']
    print(f"Original sampling frequency: {orig_sfreq} Hz")
    #raw_filtered = raw_filtered.copy().resample(sfreq=250, npad="auto")
    markers[:, 0] = np.rint(markers[:, 0] * (raw_filtered.info['sfreq'] / orig_sfreq)).astype(int)
    raw_filtered.set_montage('standard_1020')
    #raw_filtered.plot(duration=10, events=markers, title='eeg with Events')
    #raw_filtered.plot_psd(fmin=0.5, fmax=120, average=True, spatial_colors=False, picks='eeg', show=True)


    #DROP BAD CHENNELS
    bad_channels = autodrop_badchansptp(raw_filtered, z_thresh=6.0)
    print(f"Interpolated bad channels: {bad_channels}")
    # Print max and mean frequencies across EEG picks
    eeg_data = raw_filtered.get_data(picks='eeg')
    # fs = raw_filtered.info['sfreq']
    # N = eeg_data.shape[1]
    # yf = np.abs(fft(eeg_data, axis=1))
    # xf = fftfreq(N, 1/fs)
    # pos_mask = xf > 0
    # max_freq = np.max(xf[pos_mask])
    # mean_power_freq_per_channel = np.array([np.average(xf[pos_mask], weights=yf[i, pos_mask]) for i in range(yf.shape[0])])
    # mean_power_freq = np.mean(mean_power_freq_per_channel)  # average across channels
    # print(f"Nyquist frequency (fs/2): {fs/2}")
    # print(f"FFT max freq (EEG picks): {max_freq}")
    # print(f"Mean power-weighted freq per channel: {mean_power_freq_per_channel}")
    # print(f"Mean power-weighted freq (EEG picks, avg across channels): {mean_power_freq}")

    ##################################################
    ## artefacts removal ##
    # ICA MNE #
    #standardized data without TP9 and TP10 for all subjects
    #ica = ICA(n_components=27, random_state=23, max_iter='auto', method='fastica')

    ica = ICA(
        n_components=27,
        max_iter="auto",
        method="infomax",
        random_state=37, #42 0.5-40 Hz, 37 1-120 Hz
        fit_params=dict(extended=True),
    )

    # ICA fit (excluding EOG chans)
    ica_start_time = time.time()
    ica.fit(raw_filtered)
    ica_end_time = time.time()
    ica_fit_time = ica_end_time - ica_start_time
    print(f"\nICA fit took {ica_fit_time:.2f} seconds ({ica_fit_time/60:.2f} minutes)")
    ica

    mixing_matrix_save_path = data_dir / f'ica_mixing_matrix_subj{subject_num}.npy'
    sources_save_path = data_dir / f'ica_sources_subj{subject_num}.npy'
    np.save(mixing_matrix_save_path, ica.mixing_matrix_)
    ica_sources = ica.get_sources(raw_filtered).get_data() # shape:(n_components, n_times)
    np.save(sources_save_path, ica_sources)
    print(f"Saved ICA data to: {mixing_matrix_save_path.parent}")

    #Plot components
    #ica.plot_components(show=True)

    #ica.plot_sources(raw_filtered, show_scrollbars=False, show=True)

    # blinks
    #ica.plot_overlay(raw, exclude=[0], picks="eeg")

    # We can also plot some diagnostics of IC using
    # `~mne.preprocessing.ICA.plot_properties`:

    #ica.plot_properties(raw, picks=[0])


    # Selecting ICA components automatically
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Now that we've explored what components need to be removed, we can
    # apply the automatic ICA component labeling algorithm, which will
    # assign a probability value for each component being one of:
    #
    # - brain
    # - muscle artifact
    # - eye blink
    # - heart beat
    # - line noise
    # - channel noise
    # - other
    #
    # The output of the ICLabel ``label_components`` function produces
    # predicted probability values for each of these classes in that order.
    # See :footcite:`PionTonachini2019` for full details.

    ic_labels = label_components(raw_filtered, ica, method="iclabel")
    print(ic_labels["labels"])


    # Extract Labels and Reconstruct Raw Data
    # ---------------------------------------
    #
    # We can extract the labels of each component and exclude
    # non-brain classified components, keeping 'brain' and 'other'.
    # "Other" is a catch-all that for non-classifiable components.
    # We will stay on the side of caution and assume we cannot blindly remove these.

    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    print(f"Excluding these ICA components: {exclude_idx}")

    # Plot all components with labels for verification
    #from mne
    # fig = ica.plot_components(picks='eeg', 
    #                     ch_type='eeg',  
    #                     inst=None,  
    #                     reject=None, 
    #                     sensors=True, 
    #                     show_names=False, 
    #                     contours=6, 
    #                     outlines='head', 
    #                     sphere=None, 
    #                     image_interp='cubic', 
    #                     extrapolate='auto', 
    #                     border='mean', 
    #                     res=64, 
    #                     size=1, 
    #                     cmap='RdBu_r', 
    #                     vlim=(None, None), 
    #                     cnorm=None, 
    #                     colorbar=False, 
    #                     cbar_fmt='%3.2f', 
    #                     axes=None, 
    #                     title=None, 
    #                     nrows='auto', 
    #                     ncols='auto', 
    #                     show=False, 
    #                     image_args=None, 
    #                     psd_args=None, 
    #                     verbose=None)

    n_components = ica_sources.shape[0]
    # Calculate explained variance ratio using get_explained_variance_ratio from MNE
    explained_var_ratio = ica.get_explained_variance_ratio(raw_filtered, ch_type='eeg')
    # Get per-component variance explained
    ica_var_ratio = np.array([ica.get_explained_variance_ratio(raw_filtered, components=i, ch_type='eeg')['eeg'] 
                            for i in range(n_components)])

    print("\n--- ICA Component Explained Variance ---")
    for i in range(n_components):
        print(f"Component {i:2d} ({labels[i]:8s}): {ica_var_ratio[i]*100:6.2f}% variance explained")


    fig = ica.plot_components(picks=list(range(len(labels))), show=False, size=3, nrows=4, ncols=8)
    fig.set_size_inches(20, 12)
    fig.suptitle(f'ICA Components labeled with IClabel for MNE - Subject {subject_num}', fontsize=14, fontweight='bold')

    # Add explained variance text under each component
    axes = fig.get_axes()
    for idx, ax in enumerate(axes):
        if idx < len(ica_var_ratio):
            ax.text(0.5, -0.25, f'{ica_var_ratio[idx]*100:.2f}%', 
                    transform=ax.transAxes, ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    rejected_str = f"Rejected components: {[(idx, labels[idx]) for idx in exclude_idx]}"
    kept_str = f"Kept components: {[(idx, labels[idx]) for idx in range(len(labels)) if idx not in exclude_idx]}"
    fig.text(0.5, 0.02, f"{rejected_str}\n{kept_str}", ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.5))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    components_plot_path = output_dir / f'ica_components_labeled{subject_num}_stand.png'
    plt.savefig(components_plot_path, dpi=600)
    print(f"Saved components plot to: {components_plot_path}")
    #plt.show()

    # After exclusions, we reconstruct the signals
    # with artifacts removed using the mne.preprocessing.ICA.apply method
    # ica.apply() changes the Raw object in-place, make a copy.
    reconst_raw = raw_filtered.copy()
    clean = ica.apply(reconst_raw, exclude=exclude_idx)

    # Plot original and reconstructed signals for comparison
    # raw_filtered.plot(title="Original Raw Data")
    #clean.plot(duration=5, title="Reconstructed filtered Data (ICA Cleaned)")

    #########################################################

    # --- Save cleaned EEG data as .npy ---
    clea_eeg_data = clean.get_data()  # shape: (n_channels, n_times)
    clean_eeg_path = data_dir / f'clean_eeg_subj{subject_num}.npy'
    np.save(clean_eeg_path, clea_eeg_data)
    print(f"Saved cleaned EEG data to: {clean_eeg_path}")

    # --- Save events as .npy ---
    events_path = data_dir / f'events_subj{subject_num}.npy'
    np.save(events_path, markers)
    print(f"Saved events to: {events_path}")

    #########################################################

    # --- Plot component properties inspired by MNE's plot_properties and EEGLAB ---
    # Plot each component using plot_properties (MNE built-in, inspired by EEGLAB)
    # This outputs comprehensive plots with topography, spectrum, ERP, etc.
    # print("\nGenerating component property plots..")
    # for i in range(n_components):
    #     try:
    #         fig = ica.plot_properties(raw_filtered, picks=i, show=False, 
    #                                    dB=True, plot_std=True, figsize=(10, 8))
    #         # Save with variance info in title
    #         title = f"ICA{i:02d}_{labels[i]}_Var{ica_var_ratio[i]*100:.1f}pct"
    #         properties_plot_path = output_dir / f'ica{i:02d}_{labels[i]}_properties_{subject_num}stand.png'
    #         plt.savefig(properties_plot_path, dpi=600, bbox_inches='tight')
    #         # fig is a list of figures, so close each one
    #         if isinstance(fig, list):
    #             for f in fig:
    #                 plt.close(f)
    #         else:
    #             plt.close(fig)
    #     except Exception as e:
    #         print(f"  Error plotting component {i}: {e}")


    # #find EOG-related components automatically using the eog idxs
    # eog_inds, eog_scores = ica.find_bads_eog(raw_filtered)
    # print(f"Detected EOG-related ICA components: {eog_inds}")
    # #mark these components for exclusion
    # ica.exclude = eog_inds
    # #remove the marked components
    # raw_clean = ica.apply(raw_filtered.copy())


if len(SUBJECT_NUM) != len(SESSION_NUM):
    raise ValueError("SUBJECT_NUM and SESSION_NUM must have the same length")

for subject_num, session_num in zip(SUBJECT_NUM, SESSION_NUM):
    process_subject_session(subject_num, session_num)