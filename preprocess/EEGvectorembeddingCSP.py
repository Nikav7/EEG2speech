from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

import numpy as np
import mne
from mne.decoding import CSP
import pandas as pd


# Data Loading Setup
CONDITION_BASE  = {1: 100, 2: 200, 3: 400}
CONDITION_NAMES = {1: 'Imagined speech', 2: 'Listening', 3: 'Attempted speech'}
COND_TWIN = {1: (0.0, 2.0), 2: (0.0, 2.0), 3: (0.2, 2.2)}
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
                tmin=tmin, tmax=tmax, picks='eeg', baseline=None,
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
    num_class: int = 13,
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


def get_way_matrix(n_classes: int, way: str) -> np.ndarray:
    if way == "one-vs-all":
        return 2 * np.eye(n_classes, dtype=np.int32) - np.ones((n_classes, n_classes), dtype=np.int32)
    raise ValueError(f"Unsupported way='{way}'.")


def proc_multicsp_train(
    x: np.ndarray,
    y_one_hot: np.ndarray,
    n_comps: int = 2,
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


def _augment_split_independently(
    x_data: np.ndarray,
    y_dec: np.ndarray,
    *,
    num_class: int,
    target_per_class: int,
    noise_std: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Augments a single condition dataset independently."""
    x_aug = [x_data]
    y_aug = [y_dec]

    for cls in range(1, num_class + 1):
        cls_idx = np.flatnonzero(y_dec == cls)
        n_have = cls_idx.size
        if n_have == 0:
            continue

        n_add = min(target_per_class - n_have, 2 * n_have)
        if n_add > 0:
            pick = cls_idx[np.arange(n_add) % n_have]
            scale = rng.uniform(0.95, 1.05, size=(n_add, 1, 1))
            noise = rng.normal(0.0, noise_std, size=x_data[pick].shape) * scale
            
            x_aug.append((x_data[pick] + noise).astype(x_data.dtype, copy=False))
            y_aug.append(np.full(n_add, cls, dtype=np.int32))

    return np.concatenate(x_aug, axis=0), np.concatenate(y_aug, axis=0)


def run_vector_embedding_pipeline(
    x_imagined: np.ndarray,
    y_imagined: np.ndarray,
    x_attempted: np.ndarray,
    y_attempted: np.ndarray,
    x_listening: np.ndarray,
    y_listening: np.ndarray,
    *,
    numcsp: int = 4,
    n_sess: int = 16,
    num_class: int = 13,
    label_num_class: int = 13,
    seed: int = 0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    augment_target_per_class: int = 9,
    augment_noise_std: float = 1e-4,
    use_augmentation: bool = True,
    enforce_val_class_coverage: bool = True,
    csp_class_ids: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:

    y_im_dec = to_decoded_labels(y_imagined)
    y_at_dec = to_decoded_labels(y_attempted)
    y_li_dec = to_decoded_labels(y_listening)

# Class-coverage splits for Imagined and Attempted, split Each Condition Independently
    split_im = make_split_indices(
        y_im_dec, num_class=label_num_class, seed=seed, 
        val_ratio=val_ratio, test_ratio=test_ratio, 
        enforce_val_class_coverage=enforce_val_class_coverage
    )
    split_at = make_split_indices(
        y_at_dec, num_class=label_num_class, seed=seed, 
        val_ratio=val_ratio, test_ratio=test_ratio, 
        enforce_val_class_coverage=enforce_val_class_coverage
    )

    # Simple random split (70/20/10) for Listening
    # split_li = make_simple_split_indices(
    #     n_total=len(y_li_dec),
    #     seed=seed,
    #     val_ratio=0.1,  # 0.2
    #     test_ratio=test_ratio # 0.1
    # )

    split_li = make_split_indices(
            y_li_dec, num_class=label_num_class, seed=seed, 
            val_ratio=val_ratio, test_ratio=test_ratio, 
            enforce_val_class_coverage=enforce_val_class_coverage
        )
    

    x_tr_im_pre, y_tr_im_pre = x_imagined[split_im.train], y_im_dec[split_im.train]
    x_val_im_pre, y_val_im_pre = x_imagined[split_im.val], y_im_dec[split_im.val]
    x_ts_im_pre, y_ts_im_pre = x_imagined[split_im.test], y_im_dec[split_im.test]

    x_tr_at_pre, y_tr_at_pre = x_attempted[split_at.train], y_at_dec[split_at.train]
    x_val_at_pre, y_val_at_pre = x_attempted[split_at.val], y_at_dec[split_at.val]
    x_ts_at_pre, y_ts_at_pre = x_attempted[split_at.test], y_at_dec[split_at.test]

    x_tr_li_pre, y_tr_li_pre = x_listening[split_li.train], y_li_dec[split_li.train]
    x_val_li_pre, y_val_li_pre = x_listening[split_li.val], y_li_dec[split_li.val]
    x_ts_li_pre, y_ts_li_pre = x_listening[split_li.test], y_li_dec[split_li.test]

    # Independent Augmentation for Training Sets (Listening is NOT augmented or used in CSP training)
    rng = np.random.RandomState(seed)
    if use_augmentation:
        x_tr_im, y_tr_im = _augment_split_independently(x_tr_im_pre, y_tr_im_pre, num_class=label_num_class, target_per_class=augment_target_per_class, noise_std=augment_noise_std, rng=rng)
        x_tr_at, y_tr_at = _augment_split_independently(x_tr_at_pre, y_tr_at_pre, num_class=label_num_class, target_per_class=augment_target_per_class, noise_std=augment_noise_std, rng=rng)
    else:
        x_tr_im, y_tr_im = x_tr_im_pre, y_tr_im_pre
        x_tr_at, y_tr_at = x_tr_at_pre, y_tr_at_pre

    x_val_im, y_val_im = x_val_im_pre, y_val_im_pre
    x_ts_im, y_ts_im = x_ts_im_pre, y_ts_im_pre
    x_val_at, y_val_at = x_val_at_pre, y_val_at_pre
    x_ts_at, y_ts_at = x_ts_at_pre, y_ts_at_pre

    # Fit CSP on Combined Imagined + Attempted Training Data ONLY
    x_tr_both = np.concatenate([x_tr_im, x_tr_at], axis=0)
    y_tr_both = np.concatenate([y_tr_im, y_tr_at], axis=0)

    if csp_class_ids is None:
        csp_class_ids = np.arange(1, num_class + 1, dtype=np.int32)

    y_tr_one_hot = np.zeros((len(csp_class_ids), y_tr_both.shape[0]), dtype=np.int32)
    for i, cls in enumerate(csp_class_ids):
        y_tr_one_hot[i, y_tr_both == cls] = 1

    w, la = proc_multicsp_train(x_tr_both, y_tr_one_hot, n_comps=numcsp, centered=True, method="all", way="one-vs-all")

    # Feature Extraction (Applying fitted W on Imagined, Attempted, and Listening)
    return {
        "csp_w": w, "csp_eigvals": la,
        
        # Labels
        "y_train_dec": y_tr_im, "y_val_dec": y_val_im, "y_test_dec": y_ts_im,
        "y_post_attempted_train_dec": y_tr_at, "y_post_attempted_val_dec": y_val_at, "y_post_attempted_test_dec": y_ts_at,
        "y_listening_train_dec": y_tr_li_pre, "y_listening_val_dec": y_val_li_pre, "y_listening_test_dec": y_ts_li_pre,
        
        "y_pre_imagined_train_dec": y_tr_im_pre, "y_pre_imagined_val_dec": y_val_im_pre, "y_pre_imagined_test_dec": y_ts_im_pre,
        "y_pre_attempted_train_dec": y_tr_at_pre, "y_pre_attempted_val_dec": y_val_at_pre, "y_pre_attempted_test_dec": y_ts_at_pre,

        # Raw Data
        "raw_pre_imagined_train": x_tr_im_pre, "raw_pre_imagined_val": x_val_im_pre, "raw_pre_imagined_test": x_ts_im_pre,
        "raw_pre_attempted_train": x_tr_at_pre, "raw_pre_attempted_val": x_val_at_pre, "raw_pre_attempted_test": x_ts_at_pre,
        "raw_pre_listening_train": x_tr_li_pre, "raw_pre_listening_val": x_val_li_pre, "raw_pre_listening_test": x_ts_li_pre,

        "raw_post_imagined_train": x_tr_im, "raw_post_imagined_val": x_val_im, "raw_post_imagined_test": x_ts_im,
        "raw_post_attempted_train": x_tr_at, "raw_post_attempted_val": x_val_at, "raw_post_attempted_test": x_ts_at,
        "raw_post_listening_train": x_tr_li_pre, "raw_post_listening_val": x_val_li_pre, "raw_post_listening_test": x_ts_li_pre,

        # CSP Filtered Features
        "post_imagined_train": _segment_variance_log(apply_linear_derivation(x_tr_im, w), n_sess),
        "post_imagined_val": _segment_variance_log(apply_linear_derivation(x_val_im, w), n_sess),
        "post_imagined_test": _segment_variance_log(apply_linear_derivation(x_ts_im, w), n_sess),
        
        "post_attempted_train": _segment_variance_log(apply_linear_derivation(x_tr_at, w), n_sess),
        "post_attempted_val": _segment_variance_log(apply_linear_derivation(x_val_at, w), n_sess),
        "post_attempted_test": _segment_variance_log(apply_linear_derivation(x_ts_at, w), n_sess),
        
        "post_listening_train": _segment_variance_log(apply_linear_derivation(x_tr_li_pre, w), n_sess),
        "post_listening_val": _segment_variance_log(apply_linear_derivation(x_val_li_pre, w), n_sess),
        "post_listening_test": _segment_variance_log(apply_linear_derivation(x_ts_li_pre, w), n_sess),
    }


def prepare_vector_embedding_inputs(epochs_all: Dict[int, Dict[int, mne.Epochs]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    imagined_by_subject = epochs_all.get(1, {})
    listening_by_subject = epochs_all.get(2, {})
    attempted_by_subject = epochs_all.get(3, {})
    common_subjects = sorted(set(imagined_by_subject.keys()) & set(listening_by_subject.keys()) & set(attempted_by_subject.keys()))

    x_im_list, y_im_list = [], []
    x_li_list, y_li_list = [], []
    x_at_list, y_at_list = [], []

    for subject in common_subjects:
        ep_im, ep_li, ep_at = imagined_by_subject[subject], listening_by_subject[subject], attempted_by_subject[subject]
        x_im_list.append(ep_im.get_data())
        y_im_list.append(ep_im.events[:, 2] - CONDITION_BASE[1])
        x_li_list.append(ep_li.get_data())
        y_li_list.append(ep_li.events[:, 2] - CONDITION_BASE[2])
        x_at_list.append(ep_at.get_data())
        y_at_list.append(ep_at.events[:, 2] - CONDITION_BASE[3])

    x_imagined, y_imagined = np.concatenate(x_im_list, axis=0), np.concatenate(y_im_list, axis=0)
    x_listening, y_listening = np.concatenate(x_li_list, axis=0), np.concatenate(y_li_list, axis=0)
    x_attempted, y_attempted = np.concatenate(x_at_list, axis=0), np.concatenate(y_at_list, axis=0)

    common_classes = np.intersect1d(np.intersect1d(np.unique(y_imagined), np.unique(y_attempted)), np.unique(y_listening))
    
    im_mask = np.isin(y_imagined, common_classes)
    li_mask = np.isin(y_listening, common_classes)
    at_mask = np.isin(y_attempted, common_classes)

    x_imagined, y_imagined = x_imagined[im_mask], y_imagined[im_mask]
    x_listening, y_listening = x_listening[li_mask], y_listening[li_mask]
    x_attempted, y_attempted = x_attempted[at_mask], y_attempted[at_mask]

    class_map = {int(cls): idx + 1 for idx, cls in enumerate(np.sort(common_classes))}
    y_imagined = np.asarray([class_map[int(v)] for v in y_imagined], dtype=np.int32)
    y_listening = np.asarray([class_map[int(v)] for v in y_listening], dtype=np.int32)
    y_attempted = np.asarray([class_map[int(v)] for v in y_attempted], dtype=np.int32)

    return x_imagined, y_imagined, x_attempted, y_attempted, x_listening, y_listening, common_classes.astype(np.int32), class_map


def save_raw_splits_to_csv(out: Dict[str, np.ndarray], output_dir: str, subject_id: int, condition_name: str, condition_prefix: str, original_labels: np.ndarray, label_prefix: str | None = None) -> None:
    subj_dir = os.path.join(output_dir, f"subj{subject_id}")
    cond_dir = os.path.join(subj_dir, condition_name)
    label_prefix = condition_prefix if label_prefix is None else label_prefix
    
    split_map = {
        "train": (f"raw_{condition_prefix}_train", f"y_{label_prefix}_train_dec" if label_prefix not in ["imagined", "attempted"] else "y_train_dec"),
        "val": (f"raw_{condition_prefix}_val", f"y_{label_prefix}_val_dec" if label_prefix not in ["imagined", "attempted"] else "y_val_dec"),
        "test": (f"raw_{condition_prefix}_test", f"y_{label_prefix}_test_dec" if label_prefix not in ["imagined", "attempted"] else "y_test_dec"),
    }

    for split_name, (x_key, y_key) in split_map.items():
        if x_key not in out or y_key not in out:
            continue
        x_split, y_split = out[x_key], out[y_key]
        split_dir = os.path.join(cond_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i in range(x_split.shape[0]):
            remapped_label = int(y_split[i])
            label = int(original_labels[remapped_label - 1])
            csv_path = os.path.join(split_dir, f"label{label:03d}_epoch{i:04d}.csv")
            pd.DataFrame(x_split[i]).to_csv(csv_path, index=False, header=False)


def save_splits_to_csv(out: Dict[str, np.ndarray], output_dir: str, subject_id: int, condition_name: str, condition_prefix: str, original_labels: np.ndarray, label_prefix: str | None = None) -> None:
    subj_dir = os.path.join(output_dir, f"subj{subject_id}")
    cond_dir = os.path.join(subj_dir, condition_name)
    label_prefix = condition_prefix if label_prefix is None else label_prefix

    split_map = {
        "train": (f"{condition_prefix}_train", f"y_{label_prefix}_train_dec" if label_prefix not in ["imagined", "attempted"] else "y_train_dec"),
        "val": (f"{condition_prefix}_val", f"y_{label_prefix}_val_dec" if label_prefix not in ["imagined", "attempted"] else "y_val_dec"),
        "test": (f"{condition_prefix}_test", f"y_{label_prefix}_test_dec" if label_prefix not in ["imagined", "attempted"] else "y_test_dec"),
    }

    for split_name, (x_key, y_key) in split_map.items():
        if x_key not in out or y_key not in out:
            continue
        x_split, y_split = out[x_key], out[y_key]
        split_dir = os.path.join(cond_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i in range(x_split.shape[0]):
            remapped_label = int(y_split[i])
            label = int(original_labels[remapped_label - 1])
            csv_path = os.path.join(split_dir, f"label{label:03d}_epoch{i:04d}.csv")
            pd.DataFrame(x_split[i]).to_csv(csv_path, index=False, header=False)


def save_csp_metadata(all_subject_metadata: List[Dict[str, Any]], output_dir: str) -> None:
    """Saves a single aggregated csp_metadata.csv for all subjects in the main output folder."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "csp_metadata.csv")
    
    df_meta = pd.DataFrame(all_subject_metadata)
    df_meta.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[16, 17, 18, 19])
    parser.add_argument("--eeg-data-dir", default="clean_data01-120Hz")
    parser.add_argument("--output-dir", default="eegdata_250sr_new")
    parser.add_argument("--non-augmented-output-dir", default="non_augmented")
    args = parser.parse_args()

    raw_pre_aug_dir = os.path.join(args.non_augmented_output_dir, "raw_pre_augmentation")
    raw_post_aug_dir = os.path.join(args.output_dir, "raw_post_augmentation_no_csp")
    csp_post_aug_dir = os.path.join(args.output_dir, "csp_post_augmentation")

    all_subject_metadata = []

    for subject_id in args.subjects:
        raw_all, markers_all, _ = load_data([subject_id], data_dir=args.eeg_data_dir)
        if not raw_all:
            continue
        epochs_all = extract_epochs(raw_all, markers_all)

        x_im, y_im, x_at, y_at, x_li, y_li, common_classes, _ = prepare_vector_embedding_inputs(epochs_all)

        out = run_vector_embedding_pipeline(
            x_imagined=x_im, y_imagined=y_im,
            x_attempted=x_at, y_attempted=y_at,
            x_listening=x_li, y_listening=y_li,
            num_class=len(common_classes), label_num_class=len(common_classes)
        )

        # Save Raw Pre-Augmentation Splits
        save_raw_splits_to_csv({"raw_imagined_train": out["raw_pre_imagined_train"], "raw_imagined_val": out["raw_pre_imagined_val"], "raw_imagined_test": out["raw_pre_imagined_test"], "y_train_dec": out["y_pre_imagined_train_dec"], "y_val_dec": out["y_pre_imagined_val_dec"], "y_test_dec": out["y_pre_imagined_test_dec"]}, raw_pre_aug_dir, subject_id, "imagined_speech", "imagined", common_classes)
        save_raw_splits_to_csv({"raw_attempted_train": out["raw_pre_attempted_train"], "raw_attempted_val": out["raw_pre_attempted_val"], "raw_attempted_test": out["raw_pre_attempted_test"], "y_train_dec": out["y_pre_attempted_train_dec"], "y_val_dec": out["y_pre_attempted_val_dec"], "y_test_dec": out["y_pre_attempted_test_dec"]}, raw_pre_aug_dir, subject_id, "attempted_speech", "attempted", common_classes)
        save_raw_splits_to_csv({"raw_listening_train": out["raw_pre_listening_train"], "raw_listening_val": out["raw_pre_listening_val"], "raw_listening_test": out["raw_pre_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"]}, raw_pre_aug_dir, subject_id, "listening", "listening", common_classes, label_prefix="listening")

        # Save Raw Post-Augmentation Splits
        save_raw_splits_to_csv({"raw_imagined_train": out["raw_post_imagined_train"], "raw_imagined_val": out["raw_post_imagined_val"], "raw_imagined_test": out["raw_post_imagined_test"], "y_train_dec": out["y_train_dec"], "y_val_dec": out["y_val_dec"], "y_test_dec": out["y_test_dec"]}, raw_post_aug_dir, subject_id, "imagined_speech", "imagined", common_classes)
        save_raw_splits_to_csv({"raw_attempted_train": out["raw_post_attempted_train"], "raw_attempted_val": out["raw_post_attempted_val"], "raw_attempted_test": out["raw_post_attempted_test"], "y_train_dec": out["y_post_attempted_train_dec"], "y_val_dec": out["y_post_attempted_val_dec"], "y_test_dec": out["y_post_attempted_test_dec"]}, raw_post_aug_dir, subject_id, "attempted_speech", "attempted", common_classes)
        save_raw_splits_to_csv({"raw_listening_train": out["raw_post_listening_train"], "raw_listening_val": out["raw_post_listening_val"], "raw_listening_test": out["raw_post_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"]}, raw_post_aug_dir, subject_id, "listening", "listening", common_classes, label_prefix="listening")

        # Save CSP Features
        save_splits_to_csv({"imagined_train": out["post_imagined_train"], "imagined_val": out["post_imagined_val"], "imagined_test": out["post_imagined_test"], "y_train_dec": out["y_train_dec"], "y_val_dec": out["y_val_dec"], "y_test_dec": out["y_test_dec"]}, csp_post_aug_dir, subject_id, "imagined_speech", "imagined", common_classes)
        save_splits_to_csv({"attempted_train": out["post_attempted_train"], "attempted_val": out["post_attempted_val"], "attempted_test": out["post_attempted_test"], "y_train_dec": out["y_post_attempted_train_dec"], "y_val_dec": out["y_post_attempted_val_dec"], "y_test_dec": out["y_post_attempted_test_dec"]}, csp_post_aug_dir, subject_id, "attempted_speech", "attempted", common_classes)
        save_splits_to_csv({"listening_train": out["post_listening_train"], "listening_val": out["post_listening_val"], "listening_test": out["post_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"]}, csp_post_aug_dir, subject_id, "listening", "listening", common_classes, label_prefix="listening")

        # Save CSP Metadata
        csp_w = out["csp_w"]
        csp_eigvals = out["csp_eigvals"]
        
        all_subject_metadata.append({
            "subject_id": subject_id,
            "n_channels": csp_w.shape[0],
            "n_csp_filters": csp_w.shape[1],
            "n_classes": len(common_classes),
            "classes_included": [int(c) for c in common_classes],
            "n_train_imagined": out["raw_post_imagined_train"].shape[0],
            "n_train_attempted": out["raw_post_attempted_train"].shape[0],
            "n_listening_total": x_li.shape[0],
            "min_eigenvalue": float(np.min(csp_eigvals)),
            "max_eigenvalue": float(np.max(csp_eigvals)),
            "mean_eigenvalue": float(np.mean(csp_eigvals)),
        })

    # CSP aggregated metadata in the root output folder
    save_csp_metadata(all_subject_metadata, csp_post_aug_dir)

if __name__ == "__main__":
    main()