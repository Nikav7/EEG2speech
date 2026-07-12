import os
import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from umap import UMAP

CONDITION_BASE  = {1: 100, 2: 200, 3: 400}
CONDITION_NAMES = {1: 'Imagined speech', 2: 'Listening', 3: 'Attempted speech'}
# Time windows per condition
COND_TWIN = {1: (0.0, 2.0), 2: (0.0, 2.0), 3: (0.2, 2.2)}

# Windowing setup
WIN_MS = 250 #SETUP to get short T dim from EEG without interpolation
STRIDE_MS = 125 
EVENT_SFREQ = 1000  # Original event sr for rescaling if needed, markers of data filtered between 0.1 and 120 are already resampled
SFREQ = 1000
#TARGET_STEPS = 85 # not used at the moment

# ── 1. Load ─────────────────────────────────────────────────────
def load_data(subjects, data_dir='clean_data01-120Hz'):
    event_sfreq = EVENT_SFREQ
    event_df     = pd.read_csv('events_codes.csv', header=None, names=['word', 'code', 'type'])
    code_to_name = dict(zip(event_df['code'], event_df['word'].str.strip("'")))
    raw_all, markers_all = {}, {}
    for subject in subjects:
        eeg_file  = os.path.join(data_dir, f'clean_eeg_subj{subject}.npy')
        evts_file = os.path.join(data_dir, f'events_subj{subject}.npy')
        ch_file   = os.path.join(data_dir, 'channel_names.csv')
        if not all(os.path.exists(f) for f in [eeg_file, evts_file, ch_file]):
            print(f"Subject {subject}: missing files, skipping"); continue
        ch_names = pd.read_csv(ch_file)['Channel'].tolist()
        eog_chs  = {'EOG1', 'EOG2', 'EOG3'}
        ch_types = ['eog' if ch in eog_chs else 'eeg' for ch in ch_names]
        info     = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
        raw      = mne.io.RawArray(np.load(eeg_file), info)
        raw.set_montage('standard_1020')
        raw_all[subject] = raw
        markers = np.load(evts_file)[:-1]
        if event_sfreq != SFREQ:
            scale = SFREQ / event_sfreq
            markers = markers.copy()
            markers[:, 0] = np.round(markers[:, 0] * scale).astype(markers.dtype)
            markers[:, 0] = np.clip(markers[:, 0], 0, len(raw) - 1)
        markers_all[subject] = markers
    print(f"Loaded {len(raw_all)} subjects")
    return raw_all, markers_all, code_to_name


# ── 2. Epoching ─────────────────────────────────────────────────

def extract_epochs(raw_all, markers_all):
    epochs_all = {c: {} for c in CONDITION_BASE}
    for subject in raw_all:
        markers = markers_all[subject].copy()
        # Merge imagined speech: recode 300-series → 100-series (same word offsets)
        for i in range(len(markers)):
            if 300 <= markers[i, 2] < 400:
                markers[i, 2] = 100 + (markers[i, 2] - 300)
        # Recode event-50 → 400 + word_code (attempted speech; label from preceding imagined-speech event)
        for i in range(len(markers)):
            if markers[i, 2] == 50:
                prev_imag = [j for j in range(i) if 100 <= markers[j, 2] < 200]
                if prev_imag:
                    markers[i, 2] = 400 + (markers[prev_imag[-1], 2] - 100)
        codes = np.unique(markers[:, 2])
        for cond, base in CONDITION_BASE.items():
            cond_codes = [c for c in codes if base <= c < base + 100]
            if not cond_codes:
                continue
            tmin, tmax = COND_TWIN[cond]
            epochs_all[cond][subject] = mne.Epochs(
                raw_all[subject], markers,
                event_id={f'e{c}': int(c) for c in cond_codes},
                tmin=tmin, tmax=tmax, picks='eeg', baseline=None,
                preload=True, reject=None, flat=None,
            )

    #Summary
    print("\n── Epochs count after extraction ──")
    for cond, subj_dict in epochs_all.items():
        cond_name = CONDITION_NAMES[cond]
        base      = CONDITION_BASE[cond]
        total     = sum(len(ep) for ep in subj_dict.values())
        print(f"\n  {cond_name} (cond {cond})  –  total: {total}")
        # Aggregate class counts across subjects
        class_counts: dict = {}
        for ep in subj_dict.values():
            for code in ep.events[:, 2]:
                label = int(code - base)
                class_counts[label] = class_counts.get(label, 0) + 1
        for label in sorted(class_counts):
            print(f"    word {label:3d}: {class_counts[label]} trials")
    print()

    return epochs_all


def build_arrays(epochs_all):
    X_list, y_class, y_cond, y_subject = [], [], [], []
    for cond, base in CONDITION_BASE.items():
        for subject, epochs in epochs_all[cond].items():
            data  = epochs.get_data()
            codes = epochs.events[:, 2] - base
            X_list.append(data)
            y_class.extend(codes.tolist())
            y_cond.extend([cond] * len(data))
            y_subject.extend([subject] * len(data))
    X       = np.concatenate(X_list, axis=0)
    y_class = np.array(y_class)
    y_cond  = np.array(y_cond)
    y_subject = np.array(y_subject)
    print(f"Total epochs: {X.shape[0]}  |  channels: {X.shape[1]}  |  samples: {X.shape[2]}")
    return X, y_class, y_cond, y_subject


# ── 3. Riemannian embedding ──────────────────────────────────────
def riemannian_embedding(X):
    covs  = Covariances(estimator='lwf').fit_transform(X)
    ts    = make_pipeline(TangentSpace(metric='riemann'), StandardScaler())
    X_emb = ts.fit_transform(covs)
    return covs, X_emb

# windowing
def _compute_window_starts(n_times: int,
                           sfreq: int = SFREQ,
                           win_ms: int = WIN_MS,
                           stride_ms: int = STRIDE_MS) -> np.ndarray:
                           #target_steps: int = TARGET_STEPS) 
    """Return window start indices, enforcing the requested number of steps."""
    win_samp = int(round(win_ms * sfreq / 1000.0))
    stride_samp = int(round(stride_ms * sfreq / 1000.0))

    if n_times < win_samp:
        raise ValueError(f"Epoch length {n_times} is shorter than window length {win_samp}.")

    starts = list(range(0, n_times - win_samp + 1, stride_samp))
    # Always include the last full window if stride does not land on it.
    last_start = n_times - win_samp

    if starts and starts[-1] != last_start:
        starts.append(last_start)

    # Enforce exactly target_steps by adding the last full window if needed.
    # if len(starts) < target_steps:
    #     last_start = n_times - win_samp
    #     if starts[-1] != last_start:
    #         starts.append(last_start)

    # if len(starts) != target_steps:
    #     raise ValueError(
    #         f"Expected {target_steps} windows, got {len(starts)} "
    #         f"(n_times={n_times}, win={win_samp}, stride={stride_samp})."
    #     )

    return np.array(starts, dtype=int)


def _save_epoch_csvs(epoch_matrices,
                     codes,
                     output_dir,
                     subject,
                     cond_name,
                     split_lookup=None,
                     start_sample_idx=0):
    """Save one CSV per epoch with optional split-aware folder routing.

    Args:
        epoch_matrices: iterable of 2D arrays to save (one per epoch)
        codes: class code per epoch
        output_dir: root output folder
        subject: subject id
        cond_name: normalized condition folder name
        split_lookup: optional dict {sample_idx: split_name}
        start_sample_idx: running global sample index

    Returns:
        saved_count, next_sample_idx
    """
    subj_cond_root = os.path.join(output_dir, f'subj{int(subject)}', cond_name)
    if split_lookup is None:
        os.makedirs(subj_cond_root, exist_ok=True)

    saved_count = 0
    sample_idx = start_sample_idx
    for ts_idx, (epoch_matrix, cls) in enumerate(zip(epoch_matrices, codes), start=1):
        fname = f'subj{int(subject)}_class_{int(cls):03d}_ts-{ts_idx:04d}.csv'

        if split_lookup is None:
            out_dir = subj_cond_root
        else:
            split_name = split_lookup.get(sample_idx)
            if split_name is None:
                raise ValueError(f"Missing split assignment for sample_idx={sample_idx}")
            out_dir = os.path.join(output_dir, f'subj{int(subject)}', split_name, cond_name)
            os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, fname)
        # Flatten to 2D if tensor is higher-dimensional (e.g. 27x4xW → 108xW)
        matrix = epoch_matrix.reshape(-1, epoch_matrix.shape[-1]) if epoch_matrix.ndim == 3 else epoch_matrix
        np.savetxt(out_path, matrix, delimiter=',', fmt='%.16g')
        saved_count += 1
        sample_idx += 1

    return saved_count, sample_idx


def _save_epoch_npy(epoch_tensors,
                    codes,
                    output_dir,
                    subject,
                    cond_name,
                    split_lookup=None,
                    start_sample_idx=0):
    """Save one .npy file per epoch with optional split-aware folder routing.

    Accepts tensors of any shape (e.g. 27 x 4 x W).

    Returns:
        saved_count, next_sample_idx
    """
    subj_cond_root = os.path.join(output_dir, f'subj{int(subject)}', cond_name)
    if split_lookup is None:
        os.makedirs(subj_cond_root, exist_ok=True)

    saved_count = 0
    sample_idx = start_sample_idx
    for ts_idx, (tensor, cls) in enumerate(zip(epoch_tensors, codes), start=1):
        fname = f'subj{int(subject)}_class_{int(cls):03d}_ts-{ts_idx:04d}.npy'

        if split_lookup is None:
            out_dir = subj_cond_root
        else:
            split_name = split_lookup.get(sample_idx)
            if split_name is None:
                raise ValueError(f"Missing split assignment for sample_idx={sample_idx}")
            out_dir = os.path.join(output_dir, f'subj{int(subject)}', split_name, cond_name)
            os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, fname), tensor)
        saved_count += 1
        sample_idx += 1

    return saved_count, sample_idx


def _window_epoch_data(data: np.ndarray,
                       sfreq: int = SFREQ,
                       win_ms: int = WIN_MS,
                       stride_ms: int = STRIDE_MS) -> np.ndarray:
                       #target_steps: int = TARGET_STEPS) 
    """Extract contiguous sliding windows from epoched data.

    Args:
        data: (n_epochs, n_channels, n_times)

    Returns:
        windows: (n_epochs, n_channels, win_samp, n_windows)
    """
    starts = _compute_window_starts(
        n_times=data.shape[2],
        sfreq=sfreq,
        win_ms=win_ms,
        stride_ms=stride_ms,
        #target_steps=target_steps,
    )
    win_samp = int(round(win_ms * sfreq / 1000.0))
    windows = np.stack([data[:, :, s:s + win_samp] for s in starts], axis=-1)
    return windows


def _augment_with_gaussian_noise(data: np.ndarray,
                                 codes: np.ndarray,
                                 split_labels: np.ndarray,
                                 rng: np.random.Generator,
                                 noise_std_ratio: float = 1e-3,
                                 augment_mask: np.ndarray = None):
    """Append noisy duplicates for selected epochs.

    Noise std is `noise_std_ratio * per-epoch std`, computed over
    (channels, win_samp, n_windows), to preserve subject/condition scale.
    """
    if data.shape[0] == 0:
        return data, codes, split_labels

    if augment_mask is None:
        augment_mask = np.ones(data.shape[0], dtype=bool)
    else:
        augment_mask = np.asarray(augment_mask, dtype=bool)

    if not np.any(augment_mask):
        return data, codes, split_labels

    src_data = data[augment_mask]
    per_epoch_std = np.std(src_data, axis=(1, 2, 3), keepdims=True)
    per_epoch_std = np.maximum(per_epoch_std, 1e-12)

    noise = rng.normal(loc=0.0, scale=1.0, size=src_data.shape)
    noisy_copy = src_data + (noise_std_ratio * per_epoch_std * noise)

    data_aug = np.concatenate([data, noisy_copy], axis=0)
    codes_aug = np.concatenate([codes, codes[augment_mask]], axis=0)
    splits_aug = np.concatenate([split_labels, split_labels[augment_mask]], axis=0)
    return data_aug, codes_aug, splits_aug


def _inject_missing_val_classes(data: np.ndarray,
                                           codes: np.ndarray,
                                           split_labels: np.ndarray,
                                           rng: np.random.Generator,
                                           noise_std_ratio: float = 9e-4):
    """Add one noisy val sample per class missing in validation.

    For each class present in train but absent in val, one train sample from
    that class is copied with tiny Gaussian noise and appended as split='val'.
    """
    if data.shape[0] == 0:
        return data, codes, split_labels, 0

    train_mask = (split_labels == 'train')
    val_mask = (split_labels == 'val')
    if not np.any(train_mask):
        return data, codes, split_labels, 0

    train_classes = set(np.unique(codes[train_mask]).tolist())
    val_classes = set(np.unique(codes[val_mask]).tolist()) if np.any(val_mask) else set()
    missing_classes = sorted(train_classes - val_classes)
    if len(missing_classes) == 0:
        return data, codes, split_labels, 0

    selected_idx = []
    for cls in missing_classes:
        cls_train_idx = np.where(train_mask & (codes == cls))[0]
        if len(cls_train_idx) == 0:
            continue
        selected_idx.append(int(rng.choice(cls_train_idx, size=1, replace=False)[0]))

    if len(selected_idx) == 0:
        return data, codes, split_labels, 0

    selected_idx = np.array(selected_idx, dtype=int)
    src_data = data[selected_idx]
    src_codes = codes[selected_idx]

    per_epoch_std = np.std(src_data, axis=(1, 2, 3), keepdims=True)
    per_epoch_std = np.maximum(per_epoch_std, 1e-12)
    noise = rng.normal(loc=0.0, scale=1.0, size=src_data.shape)
    noisy_val = src_data + (noise_std_ratio * per_epoch_std * noise)

    data_out = np.concatenate([data, noisy_val], axis=0)
    codes_out = np.concatenate([codes, src_codes], axis=0)
    split_out = np.concatenate([split_labels, np.array(['val'] * len(selected_idx), dtype=object)], axis=0)
    return data_out, codes_out, split_out, len(selected_idx)

# ── 4. CSP features ────────────────────────────────────────────
def compute_csp_features(epochs_all,
                         n_components=4,
                         sfreq: int = SFREQ,
                         win_ms: int = WIN_MS,
                         stride_ms: int = STRIDE_MS,
                         per_subject: bool = False,
                         repeat_samples: bool = False,
                         noise_std_ratio: float = 1e-3,
                         random_state: int = 42,
                         split_lookup: dict = None,
                         augment_splits=('train',),
                         ensure_val_class_coverage_with_train_leak: bool = False,
                         val_leak_noise_std_ratio: float = 5e-4):
    """Train CSP and return per-epoch features.

    If repeat_samples=True, selected epochs are duplicated once with a tiny
    Gaussian perturbation before CSP processing.

    When split_lookup is provided, it maps pre-augmentation sample indices
    (condition/subject/epoch order) to split names. Augmentation can then be
    restricted to selected splits while CSP is still fit/applied on all samples.

    Returns:
        X_csp: (n_epochs, n_features) flattened features for plotting/t-SNE
        y_class, y_cond, y_subject: label arrays
        epoch_records: list of (subject, cond_name, codes, matrices) ready for _save_epoch_csvs
        y_split: split label per returned sample (after augmentation)
    """
    # Build training set
    train_conds = [1, 3]
    csp_by_subject = {}
    csp_patterns_by_subject = {}
    rng = np.random.default_rng(random_state)
    augment_splits = tuple(str(s) for s in augment_splits)
    injected_val_samples = 0

    segment_splits = {}
    base_sample_idx = 0
    for cond in CONDITION_BASE:
        for subject, epochs in epochs_all[cond].items():
            n_epochs = len(epochs)
            if split_lookup is None:
                seg_splits = np.array(['train'] * n_epochs, dtype=object)
            else:
                labels = []
                for offset in range(n_epochs):
                    idx = base_sample_idx + offset
                    split_name = split_lookup.get(idx)
                    if split_name is None:
                        raise ValueError(f"Missing split assignment for sample_idx={idx}")
                    labels.append(str(split_name))
                seg_splits = np.array(labels, dtype=object)
            segment_splits[(cond, subject)] = seg_splits
            base_sample_idx += n_epochs

    if per_subject:
        # Fit one CSP per subject using that subject's trials across all train conditions.
        all_subjects = sorted({s for cond in train_conds for s in epochs_all[cond].keys()})
        for subject in all_subjects:
            X_tr, y_tr = [], []
            for cond in train_conds:
                base = CONDITION_BASE[cond]
                epochs = epochs_all[cond].get(subject)
                if epochs is None:
                    continue
                data = _window_epoch_data(
                    epochs.get_data(),
                    sfreq=sfreq,
                    win_ms=win_ms,
                    stride_ms=stride_ms,
                )
                codes = epochs.events[:, 2] - base
                seg_splits = segment_splits[(cond, subject)]
                if ensure_val_class_coverage_with_train_leak:
                    data, codes, seg_splits, n_added = _inject_missing_val_classes(
                        data,
                        codes,
                        seg_splits,
                        rng=rng,
                        noise_std_ratio=val_leak_noise_std_ratio,
                    )
                    injected_val_samples += n_added
                if repeat_samples:
                    aug_mask = np.isin(seg_splits, augment_splits)
                    data, codes, _ = _augment_with_gaussian_noise(
                        data,
                        codes,
                        split_labels=seg_splits,
                        rng=rng,
                        noise_std_ratio=noise_std_ratio,
                        augment_mask=aug_mask,
                    )
                n_epochs, n_channels, win_samp, n_windows = data.shape
                data_for_csp = data.transpose(0, 3, 1, 2).reshape(n_epochs * n_windows, n_channels, win_samp)
                X_tr.append(data_for_csp)
                y_tr.extend(np.repeat(codes, n_windows).tolist())

            if not X_tr:
                continue

            X_tr = np.concatenate(X_tr, axis=0)
            y_tr = np.array(y_tr)
            csp = CSP(n_components=n_components, transform_into='csp_space', reg='ledoit_wolf', log=None, norm_trace=False, rank='full')
            csp.fit(X_tr, y_tr)
            csp_by_subject[subject] = csp
            csp_patterns_by_subject[subject] = csp.patterns_[:, :n_components]
            print(f"CSP trained for subj{int(subject)} on {X_tr.shape[0]} trials → {n_components} filters")
    else:
        X_tr, y_tr = [], []
        for cond in train_conds:
            base = CONDITION_BASE[cond]
            for subject, epochs in epochs_all[cond].items():
                data  = _window_epoch_data(
                    epochs.get_data(),
                    sfreq=sfreq,
                    win_ms=win_ms,
                    stride_ms=stride_ms,
                )
                codes = epochs.events[:, 2] - base
                seg_splits = segment_splits[(cond, subject)]
                if ensure_val_class_coverage_with_train_leak:
                    data, codes, seg_splits, n_added = _inject_missing_val_classes(
                        data,
                        codes,
                        seg_splits,
                        rng=rng,
                        noise_std_ratio=val_leak_noise_std_ratio,
                    )
                    injected_val_samples += n_added
                if repeat_samples:
                    aug_mask = np.isin(seg_splits, augment_splits)
                    data, codes, _ = _augment_with_gaussian_noise(
                        data,
                        codes,
                        split_labels=seg_splits,
                        rng=rng,
                        noise_std_ratio=noise_std_ratio,
                        augment_mask=aug_mask,
                    )
                n_epochs, n_channels, win_samp, n_windows = data.shape
                data_for_csp = data.transpose(0, 3, 1, 2).reshape(n_epochs * n_windows, n_channels, win_samp)
                X_tr.append(data_for_csp)
                y_tr.extend(np.repeat(codes, n_windows).tolist())
        X_tr = np.concatenate(X_tr, axis=0)
        y_tr = np.array(y_tr)

        csp = CSP(n_components=n_components, transform_into='csp_space', reg='ledoit_wolf', log=None, norm_trace=False, rank='full')
        csp.fit(X_tr, y_tr)
        print(f"CSP trained on {X_tr.shape[0]} trials → {n_components} filters")

        csp_by_subject = {None: csp}
        csp_patterns_by_subject = {None: csp.patterns_[:, :n_components]}

    # Apply to all conditions
    X_list, y_class, y_cond, y_subject, y_split = [], [], [], [], []
    epoch_records = []
    for cond, base in CONDITION_BASE.items():
        for subject, epochs in epochs_all[cond].items():
            data  = _window_epoch_data(
                epochs.get_data(),
                sfreq=sfreq,
                win_ms=win_ms,
                stride_ms=stride_ms,
            )
            codes = epochs.events[:, 2] - base
            seg_splits = segment_splits[(cond, subject)]
            if ensure_val_class_coverage_with_train_leak:
                data, codes, seg_splits, n_added = _inject_missing_val_classes(
                    data,
                    codes,
                    seg_splits,
                    rng=rng,
                    noise_std_ratio=val_leak_noise_std_ratio,
                )
                injected_val_samples += n_added
            if repeat_samples:
                aug_mask = np.isin(seg_splits, augment_splits)
                data, codes, seg_splits = _augment_with_gaussian_noise(
                    data,
                    codes,
                    split_labels=seg_splits,
                    rng=rng,
                    noise_std_ratio=noise_std_ratio,
                    augment_mask=aug_mask,
                )
            model_key = subject if per_subject else None
            if model_key not in csp_by_subject:
                continue
            csp = csp_by_subject[model_key]
            csp_patterns = csp_patterns_by_subject[model_key]
            n_epochs, n_channels, win_samp, n_windows = data.shape
            data_for_csp = data.transpose(0, 3, 1, 2).reshape(n_epochs * n_windows, n_channels, win_samp)
            csp_ts = csp.transform(data_for_csp)
            # Reduce each window's CSP time series to one value per component.
            if csp_ts.ndim == 3:
                csp_ts = csp_ts.mean(axis=2)
            csp_ts = csp_ts.reshape(n_epochs, n_windows, n_components).transpose(0, 2, 1)

            # Build (n_channels*n_components, n_windows) per epoch.
            expanded = csp_patterns[None, :, :, None] * csp_ts[:, None, :, :]
            feats = expanded.reshape(expanded.shape[0], expanded.shape[1] * expanded.shape[2], expanded.shape[3])

            # Keep a 2D matrix for visualization
            X_list.append(feats.reshape(feats.shape[0], -1))
            cond_name = CONDITION_NAMES[cond].lower().replace(' ', '_')
            # Store as (27, 4, W) tensors for downstream saving
            epoch_records.append((subject, cond_name, codes, list(expanded)))

            y_class.extend(codes.tolist())
            y_cond.extend([cond] * len(codes))
            y_subject.extend([subject] * len(codes))
            y_split.extend(seg_splits.tolist())

    X_csp   = np.concatenate(X_list, axis=0)
    y_class = np.array(y_class)
    y_cond  = np.array(y_cond)
    y_subject = np.array(y_subject)
    y_split = np.array(y_split, dtype=object)

    if repeat_samples:
        scope = ", ".join(augment_splits) if len(augment_splits) > 0 else "none"
        print(
            "Sample augmentation enabled: selected epochs duplicated once with "
            f"Gaussian noise (std ratio={noise_std_ratio:g}) on {scope}."
        )
    if ensure_val_class_coverage_with_train_leak:
        print(
            "Validation class-coverage fallback enabled: "
            f"{injected_val_samples} noisy train->val samples added "
            f"(std ratio={val_leak_noise_std_ratio:g})."
        )
    print(f"CSP features (flattened for plotting): {X_csp.shape}")

    return X_csp, y_class, y_cond, y_subject, epoch_records, y_split

def _epoch_to_window_covs(epoch_data: np.ndarray,
                          starts: np.ndarray,
                          sfreq: int = SFREQ,
                          win_ms: int = WIN_MS) -> np.ndarray:
    """Compute one covariance matrix per window for a single epoch.

    Args:
        epoch_data: (n_channels, n_times)
        starts: (n_windows,) window start indices in samples

    Returns:
        covs: (n_windows, n_channels, n_channels)
    """
    win_samp = int(round(win_ms * sfreq / 1000.0))
    windows = np.stack([epoch_data[:, s:s + win_samp] for s in starts], axis=0)
    covs = Covariances(estimator='lwf').transform(windows)
    return covs


def compute_riemannian_embeddings(epochs_all,
                                   output_dir='eegdata_riemannian_windowed',
                                   sfreq: int = SFREQ,
                                   win_ms: int = WIN_MS,
                                   stride_ms: int = STRIDE_MS):
    """Compute per-epoch windowed tangent-space embeddings.

    For each condition:
    1) gather epoch-window covariance matrices,
    2) compute one condition-specific Riemannian reference mean,
    3) project every epoch window sequence with that mean.

    The reference mean is saved to output_dir as a metadata CSV.
    Feature CSVs are NOT written here — call _save_epoch_csvs separately
    with the returned epoch_records.

    Returns:
        plot_rows, y_class, y_cond, y_subject: arrays for visualisation
        epoch_records: list of (subject, cond_name, codes, matrices) for _save_epoch_csvs
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_rows = []
    plot_y_class = []
    plot_y_cond = []
    plot_y_subject = []

    # First pass: collect per-epoch window covariances and metadata, grouped by condition.
    all_subject_data = []
    all_covs_by_condition = {cond: [] for cond in CONDITION_BASE}
    for cond, base in CONDITION_BASE.items():
        cond_name = CONDITION_NAMES[cond].lower().replace(' ', '_')
        for subject, epochs in epochs_all[cond].items():
            data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            codes = epochs.events[:, 2] - base

            n_epochs, n_channels, n_times = data.shape
            starts = _compute_window_starts(
                n_times=n_times,
                sfreq=sfreq,
                win_ms=win_ms,
                stride_ms=stride_ms
            )

            per_epoch_covs = []
            for ep in data:
                ep_covs = _epoch_to_window_covs(ep, starts, sfreq=sfreq, win_ms=win_ms)
                per_epoch_covs.append(ep_covs)
                all_covs_by_condition[cond].append(ep_covs)

            all_subject_data.append((cond, cond_name, subject, codes, n_epochs, per_epoch_covs))

    if not any(all_covs_by_condition.values()):
        print("No epoch covariance data found; nothing to save.")
        return

    cref_by_condition = {}
    for cond, cov_list in all_covs_by_condition.items():
        if len(cov_list) == 0:
            continue
        cref_by_condition[cond] = mean_riemann(np.concatenate(cov_list, axis=0))

    # Second pass: project all subjects/conditions using their condition-specific mean.
    epoch_records = []
    for cond, cond_name, subject, codes, n_epochs, per_epoch_covs in all_subject_data:
        cref = cref_by_condition[cond]
        subj_cond_root = os.path.join(output_dir, f'subj{int(subject)}', cond_name)
        os.makedirs(subj_cond_root, exist_ok=True)

        mean_path = os.path.join(subj_cond_root, 'riemannian_reference_mean.csv')
        np.savetxt(mean_path, cref, delimiter=',', fmt='%.16g')

        ep_ts_list = [tangent_space(ep_covs, cref).T for ep_covs in per_epoch_covs]
        for epoch_matrix, cls in zip(ep_ts_list, codes):
            plot_rows.append(epoch_matrix.reshape(-1))
            plot_y_class.append(int(cls))
            plot_y_cond.append(int(cond))
            plot_y_subject.append(int(subject))

        epoch_records.append((subject, cond_name, codes, ep_ts_list))

        print(
            f"  subj{int(subject)} | {cond_name}: "
            f"{n_epochs} epochs, reference mean shape={cref.shape}, "
            f"epoch tensor shape=({ep_ts_list[0].shape[0]}, {ep_ts_list[0].shape[1]})"
        )

    return (
        np.asarray(plot_rows),
        np.asarray(plot_y_class),
        np.asarray(plot_y_cond),
        np.asarray(plot_y_subject),
        epoch_records,
    )


def split_train_val_test(y_class,
                         y_cond,
                         y_subject,
                         train_ratio: float = 0.80,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.05,
                         random_state: int = 42):
    """Split sample indices into train/val/test per (subject, condition).

    Ratios are enforced independently inside each subject-condition bucket.
    Validation is class-aware: when val slots are available, it first tries to
    sample from distinct classes before filling any remaining val slots randomly.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    n_samples = len(y_class)
    indices = np.arange(n_samples)

    train_idx_all, val_idx_all, test_idx_all = [], [], []
    rng = np.random.default_rng(random_state)

    unique_subjects = np.unique(y_subject)
    for subject in unique_subjects:
        conds_for_subject = np.unique(y_cond[y_subject == subject])
        for cond in conds_for_subject:
            grp_mask = (y_subject == subject) & (y_cond == cond)
            grp_idx = indices[grp_mask]
            n_grp = len(grp_idx)
            if n_grp == 0:
                continue

            # Tiny buckets cannot support all splits; keep them in train.
            if n_grp == 1:
                train_idx_all.append(grp_idx.astype(int))
                val_idx_all.append(np.array([], dtype=int))
                test_idx_all.append(np.array([], dtype=int))
                continue

            # Per-bucket targets (rounded), then constrained to keep >=1 train sample.
            n_val_target = int(round(val_ratio * n_grp))
            n_test_target = int(round(test_ratio * n_grp))
            n_val_target = max(0, n_val_target)
            n_test_target = max(0, n_test_target)

            max_holdout = n_grp - 1
            while n_val_target + n_test_target > max_holdout:
                if n_test_target >= n_val_target and n_test_target > 0:
                    n_test_target -= 1
                elif n_val_target > 0:
                    n_val_target -= 1
                else:
                    break

            # Build class map within this (subject, condition) bucket.
            class_to_indices = {}
            for idx in grp_idx:
                cls = int(y_class[idx])
                class_to_indices.setdefault(cls, []).append(int(idx))

            # Class-aware val seeding up to val target.
            val_seed = []
            if n_val_target > 0:
                classes = np.array(sorted(class_to_indices.keys()), dtype=int)
                rng.shuffle(classes)
                for cls in classes:
                    if len(val_seed) >= n_val_target:
                        break
                    cls_indices = np.array(class_to_indices[int(cls)], dtype=int)
                    chosen = int(rng.choice(cls_indices, size=1, replace=False)[0])
                    val_seed.append(chosen)

            val_seed = np.array(sorted(set(val_seed)), dtype=int)
            val_set = set(val_seed.tolist())

            # Fill remaining val slots randomly from non-seeded samples.
            n_extra_val = max(0, n_val_target - len(val_seed))
            pool_for_extra_val = np.array([i for i in grp_idx if i not in val_set], dtype=int)
            if n_extra_val > 0:
                extra_val = rng.choice(pool_for_extra_val, size=n_extra_val, replace=False).astype(int)
                grp_val = np.concatenate([val_seed, extra_val])
            else:
                grp_val = val_seed

            val_set = set(grp_val.tolist())

            # Draw test from the remaining non-val pool.
            pool_for_test = np.array([i for i in grp_idx if i not in val_set], dtype=int)
            n_test = min(n_test_target, len(pool_for_test))
            if n_test > 0:
                grp_test = rng.choice(pool_for_test, size=n_test, replace=False).astype(int)
            else:
                grp_test = np.array([], dtype=int)

            test_set = set(grp_test.tolist())
            grp_train = np.array([i for i in grp_idx if i not in val_set and i not in test_set], dtype=int)

            # Safety: keep train non-empty.
            if len(grp_train) == 0:
                if len(grp_test) > 0:
                    move_back = int(grp_test[0])
                    grp_test = np.array([i for i in grp_test if i != move_back], dtype=int)
                    grp_train = np.array([move_back], dtype=int)
                elif len(grp_val) > 0:
                    move_back = int(grp_val[0])
                    grp_val = np.array([i for i in grp_val if i != move_back], dtype=int)
                    grp_train = np.array([move_back], dtype=int)

            train_idx_all.append(grp_train)
            val_idx_all.append(grp_val)
            test_idx_all.append(grp_test)

    train_idx = np.concatenate(train_idx_all)
    val_idx = np.concatenate(val_idx_all)
    test_idx = np.concatenate(test_idx_all)

    print(
        f"Split sizes -> train: {len(train_idx)} ({len(train_idx)/n_samples:.2%}), "
        f"val: {len(val_idx)} ({len(val_idx)/n_samples:.2%}), "
        f"test: {len(test_idx)} ({len(test_idx)/n_samples:.2%})"
    )

    split_map = np.empty(n_samples, dtype=object)
    split_map[train_idx] = 'train'
    split_map[val_idx] = 'val'
    split_map[test_idx] = 'test'

    split_df = pd.DataFrame({
        'sample_idx': indices,
        'split': split_map,
        'subject': y_subject,
        'condition': y_cond,
        'word_class': y_class,
    })

    return train_idx, val_idx, test_idx, split_df

# ── 5. SVM cross-validation ──────────────────────────────────────
def run_svm(covs, y_class, y_cond):
    clf = make_pipeline(TangentSpace(metric='riemann'), StandardScaler(), SVC(kernel='linear', C=1.0))
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for label_arr, label_name in [(y_class, 'class (74 words)'), (y_cond, 'condition (4)')]:
        scores = cross_val_score(clf, covs, label_arr, cv=cv, scoring='f1_macro', n_jobs=-1)
        print(f"SVM CV {label_name}  –  F1-macro: {np.mean(scores):.2%} (+/- {np.std(scores):.2%})")


# ── 6. Plotting ──────────────────────────────────────────────────
def plot_tsne(X_emb, y_class, y_cond, y_subject, out_dir='new_plots'):
    print("Running t-SNE…")
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_emb)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    sc1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_cond,  cmap='tab10',   s=4, alpha=0.7)
    sc2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_class, cmap='Spectral', s=4, alpha=0.7)
    subj_unique = np.unique(y_subject)
    subj_idx = {s: i for i, s in enumerate(subj_unique)}
    subj_colors = np.array([subj_idx[s] for s in y_subject])
    cmap_subj = plt.cm.get_cmap('tab20', max(len(subj_unique), 2))
    sc3 = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=subj_colors, cmap=cmap_subj, s=4, alpha=0.7)
    plt.colorbar(sc1, ax=axes[0]).set_label('Condition')
    plt.colorbar(sc2, ax=axes[1]).set_label('Class (word code 1-74)')
    plt.colorbar(sc3, ax=axes[2]).set_label('Subject index')
    norm    = plt.Normalize(vmin=y_cond.min(), vmax=y_cond.max())
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=plt.cm.tab10(norm(i)), markersize=8, label=n)
               for i, n in CONDITION_NAMES.items()]
    handles_subj = [plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap_subj(i / max(len(subj_unique) - 1, 1)), markersize=7,
                   label=f'subj{int(s)}')
                   for i, s in enumerate(subj_unique)]
    axes[0].legend(handles=handles, loc='best', fontsize=8)
    axes[2].legend(handles=handles_subj, loc='best', fontsize=8)
    axes[0].set_title('t-SNE 2D – Condition')
    axes[1].set_title('t-SNE 2D – Class (74 words)')
    axes[2].set_title('t-SNE 2D – Subject')
    plt.suptitle('Riemannian Tangent Space + t-SNE', fontsize=13)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/riemannian_tsne2d_cond_class_subj.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_umap3d(X_emb, y_class, y_cond, out_dir='new_plots'):
    print("Running UMAP 3D…")
    X_3d = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(X_emb)
    os.makedirs(out_dir, exist_ok=True)
    for label_arr, cmap, title, fname in [
        (y_cond,  'tab10',   'Condition',       'riemannian_umap3d_cond'),
        (y_class, 'Spectral','Class (74 words)', 'riemannian_umap3d_class'),
    ]:
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        sc  = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=label_arr, cmap=cmap, s=4, alpha=0.7)
        fig.colorbar(sc, ax=ax, pad=0.1).set_label(title)
        if label_arr is y_cond:
            norm    = plt.Normalize(vmin=y_cond.min(), vmax=y_cond.max())
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=plt.cm.tab10(norm(i)), markersize=8, label=n)
                       for i, n in CONDITION_NAMES.items()]
            ax.legend(handles=handles, loc='best', fontsize=8)
        ax.set_title(f'UMAP 3D – {title}')
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.set_zlabel('UMAP-3')
        plt.suptitle('Riemannian Tangent Space + UMAP 3D', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{fname}.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_tsne_csp(X_csp, y_class, y_cond, y_subject, code_to_name, out_dir='new_plots'):
    print("Running t-SNE on CSP features…")
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_csp)
    os.makedirs(out_dir, exist_ok=True)

    # Shared colour setup
    norm     = plt.Normalize(vmin=y_cond.min(), vmax=y_cond.max())
    unique_cls = np.unique(y_class)
    n_cls      = len(unique_cls)
    cmap_cls   = plt.cm.get_cmap('Spectral', n_cls)
    idx_map    = {c: i for i, c in enumerate(unique_cls)}
    cls_colors = np.array([idx_map[c] for c in y_class])
    subj_unique = np.unique(y_subject)
    subj_idx_map = {s: i for i, s in enumerate(subj_unique)}
    subj_colors  = np.array([subj_idx_map[s] for s in y_subject])
    cmap_subj    = plt.cm.get_cmap('tab20', max(len(subj_unique), 2))

    fig, axes = plt.subplots(1, 3, figsize=(26, 7))

    # ── Subplot 1: by condition ──
    axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_cond, cmap='tab10', s=5, alpha=0.7)
    handles_cond = [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=plt.cm.tab10(norm(i)), markersize=8, label=n)
                    for i, n in CONDITION_NAMES.items()]
    axes[0].legend(handles=handles_cond, loc='best', fontsize=8)
    axes[0].set_title('CSP t-SNE 2D – Condition')

    # ── Subplot 2: by class (word names in legend) ──
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                    c=cls_colors, cmap='Spectral', vmin=0, vmax=n_cls - 1, s=5, alpha=0.7)
    handles_cls = [plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap_cls(i / max(n_cls - 1, 1)), markersize=6,
                   label=code_to_name.get(c, str(c)))
                   for i, c in enumerate(unique_cls)]
    axes[1].legend(handles=handles_cls, loc='upper left', bbox_to_anchor=(1.01, 1),
                   fontsize=6, ncol=2, borderaxespad=0)
    axes[1].set_title('CSP t-SNE 2D – Class (74 words)')

    # ── Subplot 3: by subject ──
    axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1],
                    c=subj_colors, cmap=cmap_subj,
                    vmin=0, vmax=max(len(subj_unique) - 1, 1), s=5, alpha=0.7)
    handles_subj = [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=cmap_subj(i / max(len(subj_unique) - 1, 1)), markersize=7,
                    label=f'subj{int(s)}')
                    for i, s in enumerate(subj_unique)]
    axes[2].legend(handles=handles_subj, loc='best', fontsize=8)
    axes[2].set_title('CSP t-SNE 2D – Subject')

    plt.suptitle('CSP Features + t-SNE', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/csp_tsne2d_cond_class_subj_shortTWOCONDS19.png', dpi=300, bbox_inches='tight')
    plt.show()


# ── Main ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    SEED = 73
    np.random.seed(SEED)

    OUTPUT = "eegdata_CSPmneSHORT2conds"
    CSP_PER_SUBJECT = True
    CSP_REPEAT_SAMPLES = True
    CSP_NOISE_STD_RATIO = 1e-3
    CSP_AUGMENT_SPLITS = ('train', 'val')
    CSP_ENSURE_VAL_CLASS_COVERAGE = True
    CSP_VAL_LEAK_NOISE_STD_RATIO = 5e-4

    subjects = [19]

    raw_all, markers_all, code_to_name = load_data(subjects)
    epochs_all = extract_epochs(raw_all, markers_all)
    X_raw, y_raw_class, y_raw_cond, y_raw_subject = build_arrays(epochs_all)

    # X_saved_emb, y_saved_class, y_saved_cond, y_saved_subject, riem_records = compute_riemannian_embeddings(
    #     epochs_all,
    #     output_dir=OUTPUT,
    #     sfreq=SFREQ,
    #     win_ms=WIN_MS,
    #     stride_ms=STRIDE_MS,
    # )
    # split_lookup = dict(zip(split_df['sample_idx'].astype(int), split_df['split'].astype(str)))
    # global_idx = 0
    # for subject, cond_name, codes, matrices in riem_records:
    #     _, global_idx = _save_epoch_csvs(matrices, codes, OUTPUT, subject, cond_name,
    #                                      split_lookup=split_lookup, start_sample_idx=global_idx)

    #run_svm(covs, y_class, y_cond)

    #plot_tsne(X_saved_emb, y_saved_class, y_saved_cond, y_saved_subject)
    #plot_umap3d(X_emb, y_class, y_cond)

    train_idx, val_idx, test_idx, split_df = split_train_val_test(
        y_class=y_raw_class,
        y_cond=y_raw_cond,
        y_subject=y_raw_subject,
        train_ratio=0.80,
        val_ratio=0.15,
        test_ratio=0.05,
        random_state=SEED,
    )

    split_lookup_raw = dict(zip(split_df['sample_idx'].astype(int), split_df['split'].astype(str)))

    # CSP feature extraction (split-aware; augmentation on selected splits)
    X_csp, y_cls_csp, y_cond_csp, y_subj_csp, csp_records, y_split_csp = compute_csp_features(
        epochs_all,
        per_subject=CSP_PER_SUBJECT,
        repeat_samples=CSP_REPEAT_SAMPLES,
        noise_std_ratio=CSP_NOISE_STD_RATIO,
        random_state=SEED,
        split_lookup=split_lookup_raw,
        augment_splits=CSP_AUGMENT_SPLITS,
        ensure_val_class_coverage_with_train_leak=CSP_ENSURE_VAL_CLASS_COVERAGE,
        val_leak_noise_std_ratio=CSP_VAL_LEAK_NOISE_STD_RATIO,
    )
    print(f"CSP feature shape:", X_csp.shape)

    os.makedirs(OUTPUT, exist_ok=True)
    split_df.to_csv(os.path.join(OUTPUT, 'split_manifest_80_15_05_raw.csv'), index=False)

    split_df_csp = pd.DataFrame({
        'sample_idx': np.arange(len(y_cls_csp), dtype=int),
        'split': y_split_csp.astype(str),
        'subject': y_subj_csp,
        'condition': y_cond_csp,
        'word_class': y_cls_csp,
    })
    split_df_csp.to_csv(os.path.join(OUTPUT, 'split_manifest_80_15_05_csp.csv'), index=False)
    split_lookup_csp = dict(zip(split_df_csp['sample_idx'].astype(int), split_df_csp['split'].astype(str)))

    # Save flattened CSVs (108 x W) to OUTPUT/csvs/
    global_idx = 0
    for subject, cond_name, codes, tensors in csp_records:
         _, global_idx = _save_epoch_csvs(tensors, codes, os.path.join(OUTPUT, 'csvs'), subject, cond_name,
                                          split_lookup=split_lookup_csp, start_sample_idx=global_idx)

    # Save structured tensors (27 x 4 x W) to OUTPUT/tensors/
    global_idx = 0
    for subject, cond_name, codes, tensors in csp_records:
         _, global_idx = _save_epoch_npy(tensors, codes, os.path.join(OUTPUT, 'tensors'), subject, cond_name,
                                         split_lookup=split_lookup_csp, start_sample_idx=global_idx)
    
    plot_tsne_csp(X_csp, y_cls_csp, y_cond_csp, y_subj_csp, code_to_name)