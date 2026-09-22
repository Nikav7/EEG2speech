from __future__ import annotations

import os
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

import numpy as np
import mne
from mne.decoding import CSP
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random

# Data Loading Setup
CONDITION_BASE  = {1: 100, 2: 200, 3: 400}
CONDITION_NAMES = {1: 'Imagined speech', 2: 'Listening', 3: 'Attempted speech'}
COND_TWIN = {1: (-0.1, 2.0), 2: (-0.1, 2.0), 3: (0.2, 2.2)}
EVENT_SFREQ = 250
SFREQ = 250


def load_data(subjects, data_dir=None):
    event_sfreq = EVENT_SFREQ
    event_df = pd.read_csv('events_codes.csv', header=None, names=['word', 'code', 'type'])
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


def extract_epochs(raw_all, markers_all):
    epochs_all = {c: {} for c in CONDITION_BASE}
    for subject in raw_all:
        markers = markers_all[subject].copy()
        for i in range(len(markers)):
            if 300 <= markers[i, 2] < 400:
                markers[i, 2] = 100 + (markers[i, 2] - 300)
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
                tmin=tmin, tmax=tmax, picks='eeg', baseline=(None if cond== 3 else (-0.1, tmin)),
                preload=True, reject=None, flat=None,
            )
    return epochs_all


@dataclass
class SplitIndices:
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray


def make_simple_split_indices(
    n_total: int,
    *,
    seed: int = 0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> SplitIndices:
    if n_total < 3:
        raise ValueError("Need at least 3 samples total to split into train/val/test.")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)

    n_val = max(1, int(round(n_total * val_ratio)))
    n_test = max(1, int(round(n_total * test_ratio)))
    if n_val + n_test >= n_total:
        n_val = min(n_val, n_total - 2)
        n_test = min(n_test, n_total - n_val - 1)

    val_idx = np.sort(perm[:n_val])
    test_idx = np.sort(perm[n_val:n_val + n_test])
    train_idx = np.sort(perm[n_val + n_test:])
    return SplitIndices(train=train_idx, test=test_idx, val=val_idx)


def to_decoded_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int32) + 1
    if y.ndim != 1:
        raise ValueError(f"Labels must be 1D or 2D, got shape={y.shape}")

    y = y.astype(np.int32)
    if y.min() == 0:
        y = y + 1
    return y


def make_split_indices(
    y_dec: np.ndarray,
    seed: int = 0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    enforce_val_class_coverage: bool = True,
) -> SplitIndices:
    y_dec = np.asarray(y_dec).astype(np.int32)
    rng = np.random.RandomState(seed)

    n_total = y_dec.shape[0]
    required_classes = np.unique(y_dec)
    class_seed_val = []
    if enforce_val_class_coverage:
        for cls in required_classes:
            cls_idx = np.flatnonzero(y_dec == cls)
            if cls_idx.size > 0:
                class_seed_val.append(rng.choice(cls_idx))

    class_seed_val = np.array(sorted(set(class_seed_val)), dtype=np.int64)
    n_required_val = class_seed_val.size

    n_val_target = max(int(round(n_total * val_ratio)), n_required_val)
    n_test_target = max(1, int(round(n_total * test_ratio)))

    remaining_after_seed = np.setdiff1d(np.arange(n_total, dtype=np.int64), class_seed_val)
    n_extra_val = n_val_target - n_required_val

    extra_val = np.array([], dtype=np.int64)
    if n_extra_val > 0 and remaining_after_seed.size >= n_extra_val:
        extra_val = rng.choice(remaining_after_seed, size=n_extra_val, replace=False)

    val_idx = np.sort(np.concatenate([class_seed_val, extra_val]))
    remaining_after_val = np.setdiff1d(np.arange(n_total, dtype=np.int64), val_idx)

    test_idx = np.sort(rng.choice(remaining_after_val, size=min(n_test_target, remaining_after_val.size - 1), replace=False))
    train_idx = np.sort(np.setdiff1d(remaining_after_val, test_idx))

    return SplitIndices(train=train_idx, test=test_idx, val=val_idx)

def _augment_split_(
    x_data: np.ndarray,
    y_dec: np.ndarray,
    *,
    num_class: int,
    target_per_class: int,
    noise_std: float,
    rng: np.random.RandomState,
    extra_arrays: Dict[str, np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Augments a single condition dataset independently.

    extra_arrays lets parallel per-sample arrays (e.g. subject id, global index)
    get duplicated using the same `pick` indices as the augmented x/y rows.
    """
    extra_arrays = extra_arrays or {}
    x_aug = [x_data]
    y_aug = [y_dec]
    extra_aug = {name: [arr] for name, arr in extra_arrays.items()}

    for cls in range(1, num_class + 1):
        cls_idx = np.flatnonzero(y_dec == cls)
        n_have = cls_idx.size
        if n_have == 0:
            continue

        n_add = min(target_per_class - n_have, 2 * n_have)
        if n_add > 0:
            pick = cls_idx[np.arange(n_add) % n_have]
            scale = rng.uniform(0.05, 0.2, size=(n_add, 1, 1))
            noise = rng.normal(0.0, noise_std, size=x_data[pick].shape) * scale
            
            x_aug.append((x_data[pick] + noise).astype(x_data.dtype, copy=False))
            y_aug.append(np.full(n_add, cls, dtype=np.int32))
            for name, arr in extra_arrays.items():
                extra_aug[name].append(arr[pick])

    x_out = np.concatenate(x_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)
    extra_out = {name: np.concatenate(parts, axis=0) for name, parts in extra_aug.items()}
    return x_out, y_out, extra_out

def load_split_data(
    data_dir: str = "eegdata_rawsplits/raw_pre_augmentation_6_sets_subtog15",
    subject: int | None = None,
    condition: str = "imagined_speech",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one condition's train, validation, and test trial arrays."""
    root = Path(data_dir)
    subject_dirs = [root / f"subj{subject}"] if subject is not None else sorted(root.glob("subj*"))
    valid_conditions = {"imagined_speech", "attempted_speech", "listening"}
    if condition not in valid_conditions:
        raise ValueError(f"Unsupported condition={condition!r}; choose from {sorted(valid_conditions)}.")

    split_arrays = []
    split_indices = []
    trial_shape = None
    for split in ("train", "val", "test"):
        features = []
        labels = []
        indices = []
        expected_shape = trial_shape
        for subject_dir in subject_dirs:
            split_dir = subject_dir / condition / split
            for csv_path in sorted(split_dir.glob("*.csv")):
                match = re.search(r"label(\d+)", csv_path.name)
                if match is None:
                    continue
                index_match = re.search(r"samplegidx(\d+)", csv_path.name)
                if index_match is None:
                    raise ValueError(f"Missing global sample index in filename: {csv_path.name}")

                trial = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float32)
                if trial.size == 0:
                    continue
                if expected_shape is None:
                    expected_shape = trial.shape
                    trial_shape = trial.shape
                if trial.shape != expected_shape:
                    raise ValueError(
                        f"Inconsistent trial shape in {csv_path}: "
                        f"expected {expected_shape}, got {trial.shape}."
                    )
                features.append(trial)
                labels.append(int(match.group(1)))
                indices.append(int(index_match.group(1)))

        x_split = (
            np.stack(features)
            if features
            else np.empty((0, *trial_shape), dtype=np.float32)
            if trial_shape is not None
            else np.empty((0, 0, 0), dtype=np.float32)
        )
        y_split = np.asarray(labels, dtype=np.int32)
        split_arrays.extend((x_split, y_split))
        split_indices.append(np.asarray(indices, dtype=np.int64))

    x_train, y_train, x_val, y_val, x_test, y_test = split_arrays
    idx_train, idx_val, idx_test = split_indices

    return x_train, y_train, x_val, y_val, x_test, y_test, idx_train, idx_val, idx_test


def save_splits_to_csv(out: Dict[str, np.ndarray], output_dir: str, condition_name: str, condition_prefix: str, original_labels: np.ndarray, label_prefix: str | None = None, raw: bool = False) -> None:
    label_prefix = condition_prefix if label_prefix is None else label_prefix
    key_prefix = "raw_" if raw else ""
    split_map = {
        "train": (f"{key_prefix}{condition_prefix}_train", f"y_{label_prefix}_train_dec", f"subj_{condition_prefix}_train", f"idx_{condition_prefix}_train"),
        "val": (f"{key_prefix}{condition_prefix}_val", f"y_{label_prefix}_val_dec", f"subj_{condition_prefix}_val", f"idx_{condition_prefix}_val"),
        "test": (f"{key_prefix}{condition_prefix}_test", f"y_{label_prefix}_test_dec", f"subj_{condition_prefix}_test", f"idx_{condition_prefix}_test"),
    }

    for split_name, (x_key, y_key, subj_key, idx_key) in split_map.items():
        if x_key not in out or y_key not in out:
            continue
        x_split, y_split = out[x_key], out[y_key]
        subj_split, idx_split = out[subj_key], out[idx_key]

        for i in range(x_split.shape[0]):
            remapped_label = int(y_split[i])
            label = int(original_labels[remapped_label - 1])
            split_dir = os.path.join(output_dir, f"subj{int(subj_split[i])}", condition_name, split_name)
            os.makedirs(split_dir, exist_ok=True)
            csv_path = os.path.join(
                split_dir,
                f"label{label:03d}_samplegidx{int(idx_split[i]):05d}_row{i:06d}.csv",
            )
            pd.DataFrame(x_split[i]).to_csv(csv_path, index=False, header=False)


def save_csp_metadata(csv_name: str, metadata: List[Dict[str, Any]], output_dir: str) -> None:
    """Save aggregated metadata to a named CSV in the main output folder."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{csv_name}.csv")
    
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv(csv_path, index=False)

def rndm_nonov_splits(items, chunk_size=13):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
   
def return_rnd_splits(data):

    sorted_items = sorted(data, key=lambda x: len(x[0]), reverse=True)
    #print(sorted_items)
    rnd_items = sorted_items.copy()
    random.shuffle(rnd_items)
    #print(rnd_items)

    sets = rndm_nonov_splits(rnd_items, chunk_size=13)
    last_set = sets[-1]
    needed = 13 - len(last_set)
        
    previous_items = [item for s in sets[:-1] for item in s]
    overlap_samples = random.sample(previous_items, needed)
    sets[-1] = last_set + overlap_samples

    return sets


##########################   CSP   ######################

def get_way_matrix(n_classes: int, way: str) -> np.ndarray:
    if way == "one-vs-all":
        return 2 * np.eye(n_classes, dtype=np.int32) - np.ones((n_classes, n_classes), dtype=np.int32)
    raise ValueError(f"Unsupported way='{way}'.")


def proc_multicsp_train(
    x: np.ndarray,
    y_one_hot: np.ndarray,
    n_comps: int = 4,
    centered: bool = True,
    method: str = "all",
    way: str = "one-vs-all",
):
    dat = np.transpose(x, (1, 2, 0))
    n_chan = dat.shape[0]
    n_classes = y_one_hot.shape[0]

    sig = np.zeros((n_chan, n_chan, n_classes), dtype=np.float64)
    for i in range(n_classes):
        tr_idx = np.where(y_one_hot[i, :] > 0)[0]
        if tr_idx.size == 0:
            continue
        da = dat[:, :, tr_idx].reshape(n_chan, -1)
        if centered:
            da = da - da.mean(axis=1, keepdims=True)
        sig[:, :, i] = (da @ da.T) / max(1, da.shape[1])

    way_mat = get_way_matrix(n_classes, way)
    all_w, all_lam = [], []
    eps = 1e-12

    for i in range(way_mat.shape[0]):
        ind1 = np.where(way_mat[i, :] == 1)[0]
        ind2 = np.where(way_mat[i, :] == -1)[0]

        sig1 = np.mean(sig[:, :, ind1], axis=2)
        sig2 = np.mean(sig[:, :, ind2], axis=2)

        d, p = np.linalg.eigh(sig1 + sig2)
        d = np.maximum(d, eps)
        p = p @ np.diag(np.sqrt(1.0 / d))

        sig1_w = p.T @ sig1 @ p
        sig1_w = 0.5 * (sig1_w + sig1_w.T)

        d2, r = np.linalg.eigh(sig1_w)
        order = np.argsort(d2)
        pick = np.concatenate([order[:n_comps], order[-n_comps:]])

        all_lam.append(d2[pick])
        all_w.append(p @ r[:, pick])

    w = np.concatenate(all_w, axis=1)
    la = np.concatenate(all_lam, axis=0)
    return w, la


def apply_linear_derivation(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.einsum("ck,nct->nkt", w, x)


def _segment_variance_log(csp_ts: np.ndarray, n_sess: int, eps: float = 1e-12) -> np.ndarray:
    segments = np.array_split(csp_ts, n_sess, axis=2)
    var_segments = np.stack([np.var(seg, axis=2) for seg in segments], axis=2)
    return np.log(np.maximum(var_segments, eps))


def svm_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    classifier = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
    )
    classifier.fit(x_train.reshape(x_train.shape[0], -1), y_train)
    return float(classifier.score(x_eval.reshape(x_eval.shape[0], -1), y_eval))

def rndm_nonov_splits(items, chunk_size=13):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
   
def return_rnd_splits(data):

    sorted_items = sorted(data, key=lambda x: len(x[0]), reverse=True)
    #print(sorted_items)
    rnd_items = sorted_items.copy()
    random.shuffle(rnd_items)
    #print(rnd_items)

    sets = rndm_nonov_splits(rnd_items, chunk_size=13)
    last_set = sets[-1]
    needed = 13 - len(last_set)
        
    previous_items = [item for s in sets[:-1] for item in s]
    overlap_samples = random.sample(previous_items, needed)
    sets[-1] = last_set + overlap_samples

    return sets