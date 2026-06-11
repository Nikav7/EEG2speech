import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import os
from mne.decoding import CSP
from scipy.signal import stft
from scipy.fftpack import dct

# Load EEG data and events for subjects:
subjects = [15, 16, 17, 18, 19]
data_dir = 'clean_data'
n_mels = 80
n_mfcc = 40

eeg_data_all = {}
markers_all = {}
info_all = {}
raw_all = {}

for subject in subjects:
    # Load cleaned EEG data
    eeg_file = os.path.join(data_dir, f'clean_eeg_subj{subject}s.npy')
    if os.path.exists(eeg_file):
        eeg_data_all[subject] = np.load(eeg_file)
        print(f"Loaded EEG data for subject {subject}")
    else:
        print(f"Warning: EEG file not found for subject {subject}")
        continue
    
    # Load events
    events_file = os.path.join(data_dir, f'events_subj{subject}.npy')
    if os.path.exists(events_file):
        markers_all[subject] = np.load(events_file)
        markers_all[subject] = markers_all[subject][:-1]
        #np.save(events_file, markers_all[subject])
        print(f"Loaded events for subject {subject}")
    else:
        print(f"Warning: Events file not found for subject {subject}")
    
    # Load channel names and create info
    ch_names_file = os.path.join(data_dir, 'channel_names.csv')
    if os.path.exists(ch_names_file):
        ch_names_df = pd.read_csv(ch_names_file)
        ch_names = ch_names_df['Channel'].tolist()
        eog_channels = ['EOG1', 'EOG2', 'EOG3']
        ch_types = ['eog' if ch in eog_channels else 'eeg' for ch in ch_names]
        info_all[subject] = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
        raw_all[subject] = mne.io.RawArray(eeg_data_all[subject], info_all[subject])
        raw_all[subject].set_montage('standard_1020')
        print(f"Created info and raw object for subject {subject}")
    else:
        print(f"Warning: Channel names file not found")

print(f"\nSuccessfully loaded data for {len(eeg_data_all)} subjects")

epochs_200_all = {}  # 200 series - AUDIOS
epochs_100_all = {}  # 100 series - SILENT READING
epochs_300_all = {}  # 300 series - ATTEMPTED SPEECH

for subject in subjects:
    markers = markers_all[subject]
    raw = raw_all[subject]
    
    # 200 series - AUDIOS
    event_ids_200 = [code for code in np.unique(markers[:, 2]) if 200 <= code < 300]
    # 100 series - SILENT READING
    event_ids_100 = [code for code in np.unique(markers[:, 2]) if 100 <= code < 200]
    # 300 series - ATTEMPTED SPEECH
    event_ids_300 = [code for code in np.unique(markers[:, 2]) if 300 <= code < 400]

    all_event_ids = [code for code in np.unique(markers[:, 2]) if 100 <= code < 400]
    if len(all_event_ids) == 0:
        print(f"Subject {subject}: no 100/200/300 series events")
        continue

    epochs_2s = mne.Epochs(
        raw,
        markers,
        event_id=dict([('event' + str(e), int(e)) for e in all_event_ids]),
        tmin=0.0,
        tmax=2.0,
        picks='eeg',
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
    )

    event_names_200 = [f'event{code}' for code in event_ids_200]
    event_names_100 = [f'event{code}' for code in event_ids_100]
    event_names_300 = [f'event{code}' for code in event_ids_300]

    epochs_200_all[subject] = epochs_2s[event_names_200]
    epochs_100_all[subject] = epochs_2s[event_names_100]
    epochs_300_all[subject] = epochs_2s[event_names_300]


def hz_to_mel(freq_hz):
    return 2595.0 * np.log10(1.0 + (freq_hz / 700.0))


def mel_to_hz(freq_mel):
    return 700.0 * (10 ** (freq_mel / 2595.0) - 1.0)


def create_mel_filterbank(freqs_hz, n_mels=80, fmin=1.0, fmax=120.0):
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mels)
    mel_filters = np.zeros((n_mels, len(freqs_hz)), dtype=np.float32)

    for index in range(n_mels):
        left = hz_points[index]
        center = hz_points[index + 1]
        right = hz_points[index + 2]

        if center <= left or right <= center:
            continue

        left_mask = (freqs_hz >= left) & (freqs_hz < center)
        right_mask = (freqs_hz >= center) & (freqs_hz <= right)

        mel_filters[index, left_mask] = (freqs_hz[left_mask] - left) / (center - left)
        mel_filters[index, right_mask] = (right - freqs_hz[right_mask]) / (right - center)

    mel_sums = mel_filters.sum(axis=1, keepdims=True)
    mel_filters = mel_filters / np.maximum(mel_sums, 1e-12)
    return mel_filters


def compute_stft_power_and_mel(
    epochs_obj,
    win_length_ms=200.0,
    hop_length_ms=40.0,
    fmin=2.0,
    fmax=120.0,
    n_mels=80,
):
    epoch_data = epochs_obj.get_data(copy=True)
    sfreq = float(epochs_obj.info['sfreq'])
    win_samples = int(round((win_length_ms / 1000.0) * sfreq))
    hop_samples = int(round((hop_length_ms / 1000.0) * sfreq))
    noverlap = win_samples - hop_samples
    nfft = int(2 ** np.ceil(np.log2(win_samples)))

    if win_samples <= 0 or hop_samples <= 0:
        raise ValueError('win_length_ms and hop_length_ms must produce positive sample counts.')
    if noverlap < 0:
        raise ValueError('hop_length_ms cannot be larger than win_length_ms.')

    freqs, times, stft_complex = stft(
        epoch_data,
        fs=sfreq,
        window='hann',
        nperseg=win_samples,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
        axis=-1,
    )

    power = np.abs(stft_complex) ** 2
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_band = freqs[freq_mask]
    power_band = power[:, :, freq_mask, :]

    mel_filterbank = create_mel_filterbank(freqs_band, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_power = np.tensordot(power_band, mel_filterbank, axes=([2], [1]))
    mel_power = np.transpose(mel_power, (0, 1, 3, 2))

    return {
        'power_band': power_band,
        'freqs_band': freqs_band,
        'times': times,
        'mel_filterbank': mel_filterbank,
        'mel_power': mel_power,
    }


def log_and_normalize_mel_per_subject_condition(mel_power, eps=1e-8):
    mel_log = np.log1p(mel_power).astype(np.float32)
    mean = mel_log.mean(axis=(0, 2, 3), keepdims=True)
    std = mel_log.std(axis=(0, 2, 3), keepdims=True)
    mel_norm = (mel_log - mean) / np.maximum(std, eps)
    return mel_log, mel_norm.astype(np.float32)


def normalize_feature_per_subject_condition(feature_tensor, eps=1e-8):
    mean = feature_tensor.mean(axis=(0, 2, 3), keepdims=True)
    std = feature_tensor.std(axis=(0, 2, 3), keepdims=True)
    normalized = (feature_tensor - mean) / np.maximum(std, eps)
    return normalized.astype(np.float32)


def mel_log_to_cepstra(mel_log, n_ceps=40):
    cepstra = dct(mel_log, type=2, axis=2, norm='ortho')
    return cepstra[:, :, :n_ceps, :].astype(np.float32)


mel_output_dir = os.path.join('clean_data', 'eeg_features')
os.makedirs(mel_output_dir, exist_ok=True)

N_CEPS = 40

conditions = {
    '100': epochs_100_all,
    '200': epochs_200_all,
    '300': epochs_300_all,
}

shared_freqs = None
shared_times = None
shared_mel_filterbank = None

training_x = []
training_y = []
training_subject_ids = []
training_condition_names = []
training_event_codes = []
training_mfcc_blocks = []
training_mfcc_norm_blocks = []

for condition_name, epochs_by_subject in conditions.items():
    if len(epochs_by_subject) == 0:
        print(f'Condition {condition_name}: no epochs available, skipping mel computation.')
        continue

    cond_power_blocks = []
    cond_mel_blocks = []
    cond_mel_log_blocks = []
    cond_mel_norm_blocks = []
    cond_cepstra_blocks = []
    cond_cepstra_norm_blocks = []

    for subject in sorted(epochs_by_subject.keys()):
        result = compute_stft_power_and_mel(
            epochs_by_subject[subject],
            win_length_ms=200.0,
            hop_length_ms=40.0,
            fmin=1.0,
            fmax=120.0,
            n_mels=80,
        )

        mel_log, mel_norm = log_and_normalize_mel_per_subject_condition(result['mel_power'])

        cond_power_blocks.append(result['power_band'].astype(np.float32))
        cond_mel_blocks.append(result['mel_power'].astype(np.float32))
        cond_mel_log_blocks.append(mel_log)
        cond_mel_norm_blocks.append(mel_norm)

        mfcc = mel_log_to_cepstra(mel_log, n_ceps=N_CEPS)
        mfcc_norm = normalize_feature_per_subject_condition(mfcc)
        cond_cepstra_blocks.append(mfcc)
        cond_cepstra_norm_blocks.append(mfcc_norm)
        training_mfcc_blocks.append(mfcc)
        training_mfcc_norm_blocks.append(mfcc_norm)

        np.save(
            os.path.join(mel_output_dir, f'mel_power_{n_mels}_cond{condition_name}_subj{subject}.npy'),
            result['mel_power'].astype(np.float32),
        )
        np.save(
            os.path.join(mel_output_dir, f'mel_log1p_{n_mels}_cond{condition_name}_subj{subject}.npy'),
            mel_log,
        )
        np.save(
            os.path.join(mel_output_dir, f'mel_log1p_zscore_{n_mels}_cond{condition_name}_subj{subject}.npy'),
            mel_norm,
        )
        np.save(
            os.path.join(mel_output_dir, f'mfcc_{N_CEPS}_cond{condition_name}_subj{subject}.npy'),
            mfcc,
        )
        np.save(
            os.path.join(mel_output_dir, f'mfcc_{N_CEPS}_zscore_cond{condition_name}_subj{subject}.npy'),
            mfcc_norm,
        )
        np.save(
            os.path.join(mel_output_dir, f'mel_log1p_zscore_{n_mels}_cond{condition_name}_subj{subject}.npy'),
            mel_norm,
        )

        mfcc = mel_log_to_cepstra(mel_log, n_ceps=N_CEPS)
        mfcc_norm = normalize_feature_per_subject_condition(mfcc)
        cond_cepstra_blocks.append(mfcc)
        cond_cepstra_norm_blocks.append(mfcc_norm)

        np.save(
            os.path.join(mel_output_dir, f'mfcc_{N_CEPS}_cond{condition_name}_subj{subject}.npy'),
            mfcc,
        )
        np.save(
            os.path.join(mel_output_dir, f'mfcc_{N_CEPS}_zscore_cond{condition_name}_subj{subject}.npy'),
            mfcc_norm,
        )

        n_epochs_subject = mel_norm.shape[0]
        event_codes_subject = epochs_by_subject[subject].events[:, 2].astype(np.int32)
        if len(event_codes_subject) != n_epochs_subject:
            raise ValueError(
                f"Mismatch between epochs and event codes for subject {subject}, condition {condition_name}: "
                f"epochs={n_epochs_subject}, events={len(event_codes_subject)}"
            )

        training_x.append(mel_norm)
        training_y.append(np.full((n_epochs_subject,), int(condition_name), dtype=np.int32))
        training_subject_ids.append(np.full((n_epochs_subject,), int(subject), dtype=np.int32))
        training_condition_names.extend([condition_name] * n_epochs_subject)
        training_event_codes.append(event_codes_subject)

        if shared_freqs is None:
            shared_freqs = result['freqs_band']
            shared_times = result['times']
            shared_mel_filterbank = result['mel_filterbank']

        print(
            f"Condition {condition_name}, subject {subject}: "
            f"mel {result['mel_power'].shape} -> log/z {mel_norm.shape}"
        )

    cond_power = np.concatenate(cond_power_blocks, axis=0)
    cond_mel = np.concatenate(cond_mel_blocks, axis=0)
    cond_mel_log = np.concatenate(cond_mel_log_blocks, axis=0)
    cond_mel_norm = np.concatenate(cond_mel_norm_blocks, axis=0)

    np.save(os.path.join(mel_output_dir, f'stft_power_1_120_cond{condition_name}.npy'), cond_power)
    np.save(os.path.join(mel_output_dir, f'mel_power_{n_mels}_cond{condition_name}.npy'), cond_mel)
    np.save(os.path.join(mel_output_dir, f'mel_log1p_{n_mels}_cond{condition_name}.npy'), cond_mel_log)
    np.save(os.path.join(mel_output_dir, f'mel_log1p_zscore_{n_mels}_cond{condition_name}.npy'), cond_mel_norm)

    if len(cond_cepstra_blocks) > 0:
        cond_cepstra = np.concatenate(cond_cepstra_blocks, axis=0)
        cond_cepstra_norm = np.concatenate(cond_cepstra_norm_blocks, axis=0)
        np.save(os.path.join(mel_output_dir, f'mfcc_{N_CEPS}_cond{condition_name}.npy'), cond_cepstra)
        np.save(
            os.path.join(mel_output_dir, f'mfcc_{N_CEPS}_zscore_cond{condition_name}.npy'),
            cond_cepstra_norm,
        )

    print(
        f"Condition {condition_name}: "
        f"power {cond_power.shape}, mel {cond_mel.shape}, mel_log {cond_mel_log.shape}, mel_norm {cond_mel_norm.shape}"
    )


if shared_freqs is not None:
    np.save(os.path.join(mel_output_dir, 'stft_freqs_1_120.npy'), shared_freqs)
    np.save(os.path.join(mel_output_dir, 'stft_times.npy'), shared_times)
    np.save(os.path.join(mel_output_dir, f'mel_filterbank_{n_mels}.npy'), shared_mel_filterbank)

if len(training_x) > 0:
    x_train_ready = np.concatenate(training_x, axis=0).astype(np.float32)
    y_train_ready = np.concatenate(training_y, axis=0).astype(np.int32)
    subject_train_ready = np.concatenate(training_subject_ids, axis=0).astype(np.int32)
    condition_train_ready = np.array(training_condition_names)
    event_code_train_ready = np.concatenate(training_event_codes, axis=0).astype(np.int32)
    x_train_ready_mfcc = np.concatenate(training_mfcc_blocks, axis=0).astype(np.float32) if len(training_mfcc_blocks) > 0 else None
    x_train_ready_mfcc_z = np.concatenate(training_mfcc_norm_blocks, axis=0).astype(np.float32) if len(training_mfcc_norm_blocks) > 0 else None
    feature_name = f'mel_log1p_zscore_{n_mels}'

    save_kwargs = {
        'X': x_train_ready,
        'y': y_train_ready,
        'subject_ids': subject_train_ready,
        'condition_names': condition_train_ready,
        'event_codes': event_code_train_ready,
        'freqs_hz': shared_freqs,
        'times_s': shared_times,
        'n_ceps': np.array([int(N_CEPS)], dtype=np.int32),
    }
    if x_train_ready_mfcc is not None:
        save_kwargs['X_mfcc'] = x_train_ready_mfcc
    if x_train_ready_mfcc_z is not None:
        save_kwargs['X_mfcc_z'] = x_train_ready_mfcc_z

    np.savez_compressed(os.path.join(mel_output_dir, f'{feature_name}_dataset.npz'), **save_kwargs)


    pd.DataFrame(
        {
            'epoch_index': np.arange(len(y_train_ready), dtype=np.int32),
            'condition_code': y_train_ready,
            'subject_id': subject_train_ready,
            'condition_name': condition_train_ready,
            'event_code': event_code_train_ready,
        }
    ).to_csv(os.path.join(mel_output_dir, f'{feature_name}_dataset_index.csv'), index=False)

    print(
        f"Training-ready dataset: X {x_train_ready.shape}, y {y_train_ready.shape}, "
        f"subject_ids {subject_train_ready.shape}"
    )

print(f"Saved outputs in {mel_output_dir}")

    