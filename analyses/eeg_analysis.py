import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import os
from collections import Counter

# Load EEG data and events for subjects:
subjects = [15, 16, 17, 18, 19]
data_dir = 'clean_data120Hz'
raw_sfreq = 250.0
event_sfreq = 1000.0  # Original event sampling basis

eeg_data_all = {}
markers_all = {}
info_all = {}
raw_all = {}


def build_event_id(event_codes):
    return {f"event{int(code)}": int(code) for code in event_codes}


def add_evoked_if_nonempty(epochs_dict, evoked_dict, subject, skip_message=None):
    ep = epochs_dict[subject]
    if ep is not None and len(ep) > 0:
        evoked_dict[subject] = ep.average()
    elif skip_message:
        print(skip_message.format(subject=subject))

for subject in subjects:
    # Load cleaned EEG data
    eeg_file = os.path.join(data_dir, f'clean_eeg_subj{subject}.npy')
    if os.path.exists(eeg_file):
        eeg_data_all[subject] = np.load(eeg_file)
        print(f"Loaded EEG data for subject {subject}")
    else:
        print(f"Warning: EEG file not found for subject {subject}")
        continue
    
    # Load events
    events_file = os.path.join(data_dir, f'events_subj{subject}.npy')
    if os.path.exists(events_file):
        markers = np.load(events_file)
        if event_sfreq != raw_sfreq:
            scale = raw_sfreq / event_sfreq
            markers = markers.copy()
            markers[:, 0] = np.round(markers[:, 0] * scale).astype(markers.dtype)
            # Keep markers in-bounds for this subject's loaded EEG length.
            markers[:, 0] = np.clip(markers[:, 0], 0, eeg_data_all[subject].shape[1] - 1)
            print(f"Rescaled event samples for subject {subject}: {event_sfreq}Hz -> {raw_sfreq}Hz")
        markers_all[subject] = markers
        print(f"Loaded events for subject {subject}")
    else:
        print(f"Warning: Events file not found for subject {subject}")
    
    # Load channel names and create info
    ch_names_filename = 'channel_names5.csv' if subject == 5 else 'channel_names.csv'
    ch_names_file = os.path.join(data_dir, ch_names_filename)
    if os.path.exists(ch_names_file):
        ch_names_df = pd.read_csv(ch_names_file)
        ch_names = ch_names_df['Channel'].tolist()
        eog_channels = ['EOG1', 'EOG2', 'EOG3']
        available_eog = [ch for ch in eog_channels if ch in ch_names]
        if len(available_eog) == 0:
            print(f"Warning: No EOG channels found for subject {subject}. Proceeding with EEG-only types.")
        ch_types = ['eog' if ch in available_eog else 'eeg' for ch in ch_names]
        info_all[subject] = mne.create_info(ch_names=ch_names, sfreq=raw_sfreq, ch_types=ch_types)
        raw_all[subject] = mne.io.RawArray(eeg_data_all[subject], info_all[subject])
        raw_all[subject].set_montage('standard_1020')

print(f"\nSuccessfully loaded data for {len(eeg_data_all)} subjects")

# EPOCH ALL SUBJECTS ON THE SAME EVENT TYPES
# Store epochs for all subjects by condition
epochs_200_all = {}  # 200 series - LISTENING
epochs_100_all = {}  # 100 + 300 series - IMAGINED SPEECH
epochs_300_all = {}  # after event 50 (click) - ATTEMPTED SPEECH
epochs_yesterday_239 = {}  # 239 series - YESTERDAY listening
epochs_yesterday_139 = {}  # 139 and 339 series - YESTERDAY imagined
epochs_yesterday_439 = {}  # 439 series - YESTERDAY attempted

for subject in subjects:
    markers = markers_all[subject].copy()
    raw = raw_all[subject]

    # Adjust markers for subject 5: convert 100-series events in block 3 to 300-series
    if subject == 5:
        block_start_idxs = np.where(markers[:, 2] == 99)[0]
        block_end_idxs = np.where(markers[:, 2] == 98)[0]

        if len(block_start_idxs) >= 4:
            block_start_samps = markers[block_start_idxs, 0]
            block_end_samps = markers[block_end_idxs, 0] if len(block_end_idxs) > 0 else np.array([])

            block3_start = block_start_samps[3]
            block3_end = block_end_samps[3] if len(block_end_samps) > 3 else markers[-1, 0]

            block3_mask = (
                (markers[:, 0] >= block3_start) &
                (markers[:, 0] <= block3_end) &
                (markers[:, 2] >= 100) &
                (markers[:, 2] < 200)
            )
            adjusted_count = int(block3_mask.sum())
            markers[block3_mask, 2] += 200
            if adjusted_count > 0:
                print(f"Adjusted {adjusted_count} 100-series markers to 300-series in block 3 for subject {subject}")

    attempted_markers = []
    marker50_idxs = np.where(markers[:, 2] == 50)[0]
    for idx in marker50_idxs:
        prev_imagined = np.where(((markers[:idx, 2] >= 100) & (markers[:idx, 2] < 200)) |
                                 ((markers[:idx, 2] >= 300) & (markers[:idx, 2] < 400)))[0]
        if prev_imagined.size == 0:
            continue
        prev_code = int(markers[prev_imagined[-1], 2])
        word_code = prev_code - 100 if prev_code < 200 else prev_code - 300
        attempted_markers.append([markers[idx, 0], 0, 400 + word_code])

    attempted_markers = np.array(attempted_markers, dtype=markers.dtype) if attempted_markers else np.empty((0, 3), dtype=markers.dtype)

    unique_codes = np.unique(markers[:, 2])
    event_ids_listening = [code for code in unique_codes if 200 <= code < 300]
    event_ids_imagined = [code for code in unique_codes if 100 <= code < 200 or 300 <= code < 400]
    event_ids_attempted = [code for code in np.unique(attempted_markers[:, 2]) if 400 <= code < 500]
    n_listening_events = int(np.sum((markers[:, 2] >= 200) & (markers[:, 2] < 300)))
    n_imagined_events = int(np.sum(((markers[:, 2] >= 100) & (markers[:, 2] < 200)) | ((markers[:, 2] >= 300) & (markers[:, 2] < 400))))
    n_attempted_events = int(attempted_markers.shape[0])
    n_unpaired_clicks = int(marker50_idxs.size - n_attempted_events)

    print(f"\nSubject {subject}:")
    print(f"  Found {n_listening_events} LISTENING event occurrences ({len(event_ids_listening)} unique labels)")
    print(f"  Found {n_imagined_events} IMAGINED SPEECH event occurrences ({len(event_ids_imagined)} unique labels)")
    print(f"  Found {n_attempted_events} ATTEMPTED SPEECH event occurrences (paired from marker 50)")
    # if n_unpaired_clicks > 0:
    #     print(f"  Warning: {n_unpaired_clicks} marker-50 events had no preceding imagined-speech event")

    
    epochs_200_all[subject] = mne.Epochs(
        raw,
        markers,
        event_id=build_event_id(event_ids_listening),
        tmin=0.1,
        tmax=1.5,
        picks='eeg',
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
    )
    print(f"  Created {len(epochs_200_all[subject])} epochs for LISTENING")

    
    epochs_100_all[subject] = mne.Epochs(
        raw,
        markers,
        event_id=build_event_id(event_ids_imagined),
        tmin=0.1,
        tmax=1.5,
        picks='eeg',
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
    )
    print(f"  Created {len(epochs_100_all[subject])} epochs for IMAGINED SPEECH")


    epochs_300_all[subject] = mne.Epochs(
        raw,
        attempted_markers,
        event_id=build_event_id(event_ids_attempted),
        tmin=0.1,
        tmax=1.5,
        picks='eeg',
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
    )

    print(f"  Created {len(epochs_300_all[subject])} epochs for ATTEMPTED SPEECH")
    if len(epochs_300_all[subject]) == 0:
        dropped = [reason for reasons in epochs_300_all[subject].drop_log for reason in reasons if reason]
        if dropped:
            print(f"  Drop reasons for ATTEMPTED SPEECH: {dict(Counter(dropped))}")

    epochs_yesterday_239[subject] = mne.Epochs(
        raw,
        markers,
        event_id=239,
        tmin=0.1,
        tmax=1.5,
        picks='eeg',
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
    )
    
    epochs_yesterday_139[subject] = mne.Epochs(
        raw,
        markers,
        event_id=[139, 339],
        tmin=0.1,
        tmax=1.5,
        picks='eeg',
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
    )

    epochs_yesterday_439[subject] = mne.Epochs(
        raw,
        attempted_markers,
        event_id=439,
        tmin=0.2,
        tmax=1.5,
        picks='eeg',
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
    )

    # Inspect rare-event epoch loss explicitly; TOO_SHORT indicates boundary truncation.
    for label, ep in [
        ('YESTERDAY 239', epochs_yesterday_239[subject]),
        ('YESTERDAY 139/339', epochs_yesterday_139[subject]),
        ('YESTERDAY 439', epochs_yesterday_439[subject]),
    ]:
        dropped = [reason for reasons in ep.drop_log for reason in reasons if reason]
        if dropped:
            print(f"  Drop reasons for {label}: {dict(Counter(dropped))}")
    

selected_channels = ['FC5', 'T7', 'C3', 'CP5', 'FC6', 'T8', 'C4', 'CP6']
dx_channels = ['FC6', 'T8', 'C4', 'CP6']
motor_channels = ['C3', 'Cz', 'C4', 'CP1', 'CP2']
frontal_channels = ['F3', 'Fz', 'F4', 'AF3', 'AF4']

# PLOT EVOKED RESPONSES AND TOPOPLOTS
# Time points for the topoplots
time_points = [0.20, 0.25, 0.35, 0.45,0.50, 0.55, 0.65, 0.8, 1.0]

# Create evoked responses (averaged across epochs) for each subject
evoked_200_all = {}
evoked_100_all = {}
evoked_300_all = {}
evoked_yesterday_239 = {}
evoked_yesterday_139 = {}
evoked_yesterday_439 = {}

for subject in epochs_200_all.keys():
    add_evoked_if_nonempty(epochs_200_all, evoked_200_all, subject)

for subject in epochs_100_all.keys():
    add_evoked_if_nonempty(epochs_100_all, evoked_100_all, subject)

for subject in epochs_300_all.keys():
    add_evoked_if_nonempty(
        epochs_300_all,
        evoked_300_all,
        subject,
        skip_message="Skipping attempted evoked average for subject {subject}: no valid attempted epochs",
    )

for subject in epochs_yesterday_239.keys():
    add_evoked_if_nonempty(epochs_yesterday_239, evoked_yesterday_239, subject)

for subject in epochs_yesterday_139.keys():
    add_evoked_if_nonempty(epochs_yesterday_139, evoked_yesterday_139, subject)

for subject in epochs_yesterday_439.keys():
    add_evoked_if_nonempty(epochs_yesterday_439, evoked_yesterday_439, subject)

def n_epochs_before_drop(ep):
    dropped_target = sum(
        1 for reasons in ep.drop_log if reasons and not all(r == 'IGNORED' for r in reasons)
    )
    return len(ep) + dropped_target


def print_evoked_summary(label, epochs_dict, evoked_dict):
    all_subjects = sorted(epochs_dict.keys())
    contributing_subjects = sorted(evoked_dict.keys())
    total_original = sum(n_epochs_before_drop(epochs_dict[s]) for s in all_subjects)
    total_kept = sum(len(epochs_dict[s]) for s in all_subjects)

    print(
        f"{label}: original={total_original}, used={total_kept} "
        f"across {len(contributing_subjects)}/{len(all_subjects)} subjects"
    )
    if all_subjects:
        per_subject = ", ".join([
            f"S{s}={n_epochs_before_drop(epochs_dict[s])}->{len(epochs_dict[s])}"
            for s in all_subjects
        ])
        print(f"  per-subject (original->used): {per_subject}")
    else:
        print("  per-subject (original->used): none")

print("\n" + "=" * 60)
print("EVOKED AVERAGING SUMMARY")
print("=" * 60)
print_evoked_summary("LISTENING (200)", epochs_200_all, evoked_200_all)
print_evoked_summary("IMAGINED SPEECH (100/300)", epochs_100_all, evoked_100_all)
print_evoked_summary("ATTEMPTED SPEECH (400)", epochs_300_all, evoked_300_all)
print_evoked_summary("YESTERDAY LISTENING (239)", epochs_yesterday_239, evoked_yesterday_239)
print_evoked_summary("YESTERDAY IMAGINED (139/339)", epochs_yesterday_139, evoked_yesterday_139)
print_evoked_summary("YESTERDAY ATTEMPTED (439)", epochs_yesterday_439, evoked_yesterday_439)
print("=" * 60 + "\n")

# Grand average across all subjects for each condition
if len(evoked_200_all) > 0:
    total_segments_200 = sum(len(epochs_200_all[subject]) for subject in epochs_200_all.keys())
    print(f"Grand Average LISTENING uses {total_segments_200} total segments across {len(evoked_200_all)} subjects")
    grand_avg_200 = mne.grand_average(list(evoked_200_all.values()), interpolate_bads=False, drop_bads=True)
    grand_avg_200.plot_joint(
        times=time_points,
        title=f'Grand Average - LISTENING - N={len(evoked_200_all)} subjects',
        show=False
    )
    plt.show()

if len(evoked_100_all) > 0:
    total_segments_100 = sum(len(epochs_100_all[subject]) for subject in epochs_100_all.keys())
    print(f"Grand Average IMAGINED SPEECH uses {total_segments_100} total segments across {len(evoked_100_all)} subjects")
    grand_avg_100 = mne.grand_average(list(evoked_100_all.values()), interpolate_bads=False, drop_bads=True)
    grand_avg_100.plot_joint(
        times=time_points,
        title=f'Grand Average - IMAGINED SPEECH - N={len(evoked_100_all)} subjects',
        show=False
    )
    plt.show()

if len(evoked_300_all) > 0:
    total_segments_300 = sum(len(epochs_300_all[subject]) for subject in epochs_300_all.keys())
    print(f"Grand Average ATTEMPTED SPEECH uses {total_segments_300} total segments across {len(evoked_300_all)} subjects")
    grand_avg_300 = mne.grand_average(list(evoked_300_all.values()), interpolate_bads=False, drop_bads=True)
    grand_avg_300.plot_joint(
        times=time_points,
        title=f'Grand Average - ATTEMPTED SPEECH - N={len(evoked_300_all)} subjects',
        show=False
    )
    plt.show()

if len(evoked_yesterday_239) > 0:
    total_segments_239 = sum(len(epochs_yesterday_239[subject]) for subject in evoked_yesterday_239.keys())
    print(f"Grand Average Yesterday listening uses {total_segments_239} total segments across {len(evoked_yesterday_239)} subjects")
    grand_avg_239 = mne.grand_average(list(evoked_yesterday_239.values()), interpolate_bads=False, drop_bads=True)
    grand_avg_239.plot_joint(
    times=time_points,
    title=f'Grand Average - Yesterday listening - N={len(evoked_yesterday_239)} subjects',
    show=False
    )
    plt.show()

if len(evoked_yesterday_139) > 0:
    total_segments_139 = sum(len(epochs_yesterday_139[subject]) for subject in evoked_yesterday_139.keys())
    print(f"Grand Average Yesterday imagined speech uses {total_segments_139} total segments across {len(evoked_yesterday_139)} subjects")
    grand_avg_139 = mne.grand_average(list(evoked_yesterday_139.values()), interpolate_bads=False, drop_bads=True)
    grand_avg_139.plot_joint(
    times=time_points,
    title=f'Grand Average - Yesterday imagined speech - N={len(evoked_yesterday_139)} subjects',
    show=False
    )
    plt.show()

if len(evoked_yesterday_439) > 0:
    total_segments_439 = sum(len(epochs_yesterday_439[subject]) for subject in evoked_yesterday_439.keys())
    print(f"Grand Average Yesterday attempted speech uses {total_segments_439} total segments across {len(evoked_yesterday_439)} subjects")
    grand_avg_439 = mne.grand_average(list(evoked_yesterday_439.values()), interpolate_bads=True)
    grand_avg_439.plot_joint(
    times=time_points,
    title=f'Grand Average - Yesterday attempted speech - N={len(evoked_yesterday_439)} subjects',
    show=False
    )
    plt.show()

