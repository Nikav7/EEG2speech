"""
1) Build class-balanced train/test/val indices using random-index logic.
    split_li = make_split_indices(
        y_dec=y_li_dec,
        num_class=num_class,
        n_fold=n_fold,
        seed=seed,
        trials_per_class=trials_per_class,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
2) Fit CSP on imagined and attempted epochs.
3) Apply fitted CSP to imagined and attempted train/test/val.
4) Convert CSP time series to per-segment variance and log-transform.

Expected epoch tensor shape: (n_epochs, n_channels, n_times).
Labels can be integer class IDs or one-hot arrays.
"""
from __future__ import annotations

import os

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import mne
from mne.decoding import CSP
import pandas as pd


#Data Loading functions

CONDITION_BASE  = {1: 100, 2: 200, 3: 400}
CONDITION_NAMES = {1: 'Imagined speech', 2: 'Listening', 3: 'Attempted speech'}
# Time windows per condition
COND_TWIN = {1: (0.0, 2.0), 2: (0.0, 2.0), 3: (0.2, 2.2)}

# Windowing setup
#WIN_MS = 250 #SETUP USED BY LAST PAPER DEC 2025, DISCOVERED EMPIRICALLY!
#STRIDE_MS = 125 # SETUP DISCOVERED EMPIRICALLY
EVENT_SFREQ = 250  # Original event sr for rescaling if needed, markers of data filtered between 0.1 and 120 are already resampled
SFREQ = 250
#TARGET_STEPS = 85 # not used at the moment

#Load
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


#Epoching
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
    """Create a simple random split by position, without class balancing."""
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
    """Convert one-hot or integer labels to 1-based decoded labels."""
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int32) + 1
    if y.ndim != 1:
        raise ValueError(f"Labels must be 1D or 2D, got shape={y.shape}")

    y = y.astype(np.int32)
    if y.min() == 0:
        # Keep parity with MATLAB 1..K label style.
        y = y + 1
    return y


def make_split_indices(
    y_dec: np.ndarray,
    num_class: int =74,
    n_fold: int = 3,
    seed: int = 0,
    trials_per_class: int | None = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> SplitIndices:
    """Create global train/val/test splits using all imagined samples.

    Split targets follow 70/20/10 globally (rounded to integers), while
    validation is constrained to include at least one sample per class.
    """
    y_dec = np.asarray(y_dec).astype(np.int32)
    rng = np.random.RandomState(seed)

    if not (0.0 < val_ratio < 1.0 and 0.0 < test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be between 0 and 1")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    n_total = y_dec.shape[0]
    if n_total < 3:
        raise ValueError("Need at least 3 samples total to split into train/val/test.")

    required_classes = np.arange(1, num_class + 1, dtype=np.int32)
    class_seed_val = []
    for cls in required_classes:
        cls_idx = np.flatnonzero(y_dec == cls)
        if cls_idx.size == 0:
            raise ValueError(f"Class {cls} has 0 samples; cannot enforce validation coverage.")
        class_seed_val.append(rng.choice(cls_idx))

    class_seed_val = np.array(sorted(set(class_seed_val)), dtype=np.int64)
    n_required_val = class_seed_val.size

    n_val_target = max(int(round(n_total * val_ratio)), n_required_val)
    n_test_target = max(1, int(round(n_total * test_ratio)))

    # Preserve at least one train sample after allocating val/test.
    if n_val_target + n_test_target >= n_total:
        n_val_target = min(n_val_target, n_total - 2)
        n_test_target = min(n_test_target, n_total - n_val_target - 1)
    if n_test_target < 1:
        n_test_target = 1
    if n_val_target < n_required_val:
        raise ValueError("Unable to keep validation coverage for all classes.")

    remaining_after_seed = np.setdiff1d(np.arange(n_total, dtype=np.int64), class_seed_val, assume_unique=False)
    n_extra_val = n_val_target - n_required_val
    if n_extra_val > remaining_after_seed.size:
        raise ValueError("Not enough remaining samples to complete validation split.")

    extra_val = np.array([], dtype=np.int64)
    if n_extra_val > 0:
        extra_val = rng.choice(remaining_after_seed, size=n_extra_val, replace=False)

    val_idx = np.sort(np.concatenate([class_seed_val, extra_val]))
    remaining_after_val = np.setdiff1d(np.arange(n_total, dtype=np.int64), val_idx, assume_unique=False)

    if n_test_target > remaining_after_val.size:
        n_test_target = max(1, remaining_after_val.size - 1)
    if n_test_target < 1:
        raise ValueError("Unable to allocate non-empty test split.")

    test_idx = np.sort(rng.choice(remaining_after_val, size=n_test_target, replace=False))
    train_idx = np.sort(np.setdiff1d(remaining_after_val, test_idx, assume_unique=False))

    if train_idx.size == 0:
        raise ValueError("Train split is empty after allocation.")

    return SplitIndices(train=train_idx, test=test_idx, val=val_idx)


def get_way_matrix(n_classes: int, way: str) -> np.ndarray:
    if way == "one-vs-all":
        return 2 * np.eye(n_classes, dtype=np.int32) - np.ones((n_classes, n_classes), dtype=np.int32)

    if way == "pairwise":
        rows = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                vec = np.zeros(n_classes, dtype=np.int32)
                vec[i] = 1
                vec[j] = -1
                rows.append(vec)
        return np.asarray(rows, dtype=np.int32)

    raise ValueError(f"Unsupported way='{way}'. Use 'one-vs-all' or 'pairwise'.")


def proc_multicsp_train(
    x: np.ndarray,
    y_one_hot: np.ndarray,
    n_comps: int = 2,
    centered: bool = True,
    method: str = "all",
    way: str = "one-vs-all",
):
    dat = np.transpose(x, (1, 2, 0))

    if method == "all":
        pass
    elif method == "mean":
        dat = dat.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported method='{method}'. Use 'all' or 'mean'.")

    n_chan = dat.shape[0]
    n_classes = y_one_hot.shape[0]
    if n_comps * n_classes >= n_chan:
        print("Warning: requested multiclass CSP filter count exceeds channel count; continuing with redundant projections.")

    sig = np.zeros((n_chan, n_chan, n_classes), dtype=np.float64)
    for i in range(n_classes):
        tr_idx = np.where(y_one_hot[i, :] > 0)[0]
        if tr_idx.size == 0:
            continue
        da = dat[:, :, tr_idx].reshape(n_chan, -1)
        if centered:
            da = da - da.mean(axis=1, keepdims=True)
        sig[:, :, i] = (da @ da.T) / da.shape[1]

    way_mat = get_way_matrix(n_classes, way)

    all_w = []
    all_lam = []
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

        lam = d2[pick]
        v = p @ r[:, pick]

        all_lam.append(lam)
        all_w.append(v)

    w = np.concatenate(all_w, axis=1)
    la = np.concatenate(all_lam, axis=0)
    return w, la


def apply_linear_derivation(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.einsum("ck,nct->nkt", w, x)


def _segment_variance_log(csp_ts: np.ndarray, n_sess: int, eps: float = 1e-12) -> np.ndarray:
    """Compute variance per time segment and apply natural log.

    Input:
        csp_ts: (n_epochs, n_components, n_times)
    Output:
        (n_epochs, n_components, n_sess)
    """
    if csp_ts.ndim != 3:
        raise ValueError(f"Expected CSP time series with 3 dims, got {csp_ts.shape}")
    if n_sess < 1:
        raise ValueError("n_sess must be >= 1")

    segments = np.array_split(csp_ts, n_sess, axis=2)
    var_segments = np.stack([np.var(seg, axis=2) for seg in segments], axis=2)
    return np.log(np.maximum(var_segments, eps))


def _fit_mne_csp(x_train: np.ndarray, y_train_dec: np.ndarray, numcsp: int) -> CSP:
    """Fit MNE CSP on imagined-train data.

    MATLAB multicsp keeps both low/high eigenvalue ends per pairwise setting.
    In MNE, we map this to n_components=2*numcsp for similar feature count.
    """
    n_components = 2 * int(numcsp)
    y0 = y_train_dec.astype(np.int32) - 1
    csp = CSP(
        n_components=n_components,
        reg=None,
        log=None,
        cov_est="concat",
        transform_into="csp_space",
        norm_trace=False,
        rank="full",
    )
    csp.fit(x_train, y0)
    return csp


def _transform_with_csp(csp: CSP, x: np.ndarray, n_sess: int) -> np.ndarray:
    csp_ts = csp.transform(x)
    if csp_ts.ndim == 2:
        # Fallback when transform returns already-reduced features.
        csp_ts = csp_ts[:, :, None]
    return _segment_variance_log(csp_ts, n_sess=n_sess)


def _match_attempted_to_labels(
    x_attempted: np.ndarray,
    y_attempted_dec: np.ndarray,
    target_labels_dec: np.ndarray,
    num_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match attempted samples to target labels by cycling class samples.

    If attempted has fewer samples for a class than the target split needs,
    samples are repeated in-order for that class.
    """
    x_attempted = np.asarray(x_attempted)
    y_attempted_dec = np.asarray(y_attempted_dec).astype(np.int32)
    target_labels_dec = np.asarray(target_labels_dec).astype(np.int32)

    x_matched = np.empty(
        (target_labels_dec.shape[0], x_attempted.shape[1], x_attempted.shape[2]),
        dtype=x_attempted.dtype,
    )

    for cls in range(1, num_class + 1):
        tgt_idx = np.flatnonzero(target_labels_dec == cls)
        if tgt_idx.size == 0:
            continue

        src_idx = np.flatnonzero(y_attempted_dec == cls)
        if src_idx.size == 0:
            raise ValueError(f"Attempted condition has 0 samples for class {cls}.")

        cyc = src_idx[np.arange(tgt_idx.size) % src_idx.size]
        x_matched[tgt_idx] = x_attempted[cyc]

    return x_matched, target_labels_dec.copy()


def _augment_split_to_target(
    x_im: np.ndarray,
    x_sp: np.ndarray,
    y_dec: np.ndarray,
    *,
    num_class: int,
    target_per_class: int,
    noise_std: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augment a split so each class has at least target_per_class samples."""
    x_im_aug = [x_im]
    x_sp_aug = [x_sp]
    y_aug = [y_dec]

    for cls in range(1, num_class + 1):
        cls_idx = np.flatnonzero(y_dec == cls)
        n_have = cls_idx.size

        if n_have == 0:
            # Allow split augmentation to proceed even when some classes are absent.
            continue
        if n_have >= target_per_class:
            continue

        # Do not over-augment: each original sample can create at most two
        # augmented copies.
        n_add = min(target_per_class - n_have, 2 * n_have)
        if n_add <= 0:
            continue
        pick = cls_idx[np.arange(n_add) % n_have]

        # Per-copy scale keeps noise minimal while making repeated copies non-identical.
        im_scale = rng.uniform(0.95, 1.05, size=(n_add, 1, 1))
        sp_scale = rng.uniform(0.95, 1.05, size=(n_add, 1, 1))
        im_noise = rng.normal(0.0, noise_std, size=x_im[pick].shape) * im_scale
        sp_noise = rng.normal(0.0, noise_std, size=x_sp[pick].shape) * sp_scale
        im_add = x_im[pick] + im_noise
        sp_add = x_sp[pick] + sp_noise
        y_add = np.full(n_add, cls, dtype=np.int32)

        x_im_aug.append(im_add.astype(x_im.dtype, copy=False))
        x_sp_aug.append(sp_add.astype(x_sp.dtype, copy=False))
        y_aug.append(y_add)
        
    return (
        np.concatenate(x_im_aug, axis=0),
        np.concatenate(x_sp_aug, axis=0),
        np.concatenate(y_aug, axis=0),
    )


def _augment_split_once(
    x_im: np.ndarray,
    x_sp: np.ndarray,
    y_dec: np.ndarray,
    *,
    noise_std: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Light augmentation: add one noisy copy per sample in the split."""
    if x_im.shape[0] == 0:
        return x_im, x_sp, y_dec

    im_noise = rng.normal(0.0, noise_std, size=x_im.shape)
    sp_noise = rng.normal(0.0, noise_std, size=x_sp.shape)

    x_im_aug = np.concatenate([x_im, (x_im + im_noise).astype(x_im.dtype, copy=False)], axis=0)
    x_sp_aug = np.concatenate([x_sp, (x_sp + sp_noise).astype(x_sp.dtype, copy=False)], axis=0)
    y_aug = np.concatenate([y_dec, y_dec], axis=0)
    return x_im_aug, x_sp_aug, y_aug


def _infer_trials_per_class(y_dec: np.ndarray, num_class: int) -> int:
    """Infer class-balanced split depth from imagined labels."""
    counts = [int(np.sum(y_dec == cls)) for cls in range(1, num_class + 1)]
    if min(counts) <= 0:
        missing = [str(i + 1) for i, c in enumerate(counts) if c <= 0]
        raise ValueError(f"Missing imagined samples for classes: {', '.join(missing)}")
    return min(counts)


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
    num_class: int = 74,
    n_fold: int = 5,
    seed: int = 0,
    trials_per_class: int | None = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    augment_target_per_class: int = 15,
    augment_noise_std: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """Run MATLAB-like vector embedding CSP flow using MNE CSP.

    Returns keys:
      imagined_train, imagined_test, imagined_val,
      attempted_train, attempted_test, attempted_val,
      listening_train, listening_test, listening_val,
      y_train_dec, y_test_dec, y_val_dec
    """
    x_imagined = np.asarray(x_imagined)
    x_attempted = np.asarray(x_attempted)
    x_listening = np.asarray(x_listening)

    if x_imagined.ndim != 3 or x_attempted.ndim != 3 or x_listening.ndim != 3:
        raise ValueError("All condition tensors must be shape (n_epochs, n_channels, n_times)")

    y_im_dec = to_decoded_labels(y_imagined)
    y_at_dec = to_decoded_labels(y_attempted)
    y_li_dec = to_decoded_labels(y_listening)

    split = make_split_indices(
        y_dec=y_im_dec,
        num_class=num_class,
        n_fold=n_fold,
        seed=seed,
        trials_per_class=trials_per_class,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    listening_split = make_simple_split_indices(
        x_listening.shape[0],
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    x_tr_im = x_imagined[split.train]
    x_ts_im = x_imagined[split.test]
    x_val_im = x_imagined[split.val]

    y_tr_im = y_im_dec[split.train]
    y_ts_im = y_im_dec[split.test]
    y_val_im = y_im_dec[split.val]

    x_tr_li = x_listening[listening_split.train]
    x_ts_li = x_listening[listening_split.test]
    x_val_li = x_listening[listening_split.val]

    y_tr_li = y_li_dec[listening_split.train]
    y_ts_li = y_li_dec[listening_split.test]
    y_val_li = y_li_dec[listening_split.val]

    # Attempted splits are built by class-wise matching to imagined split labels for CSP training.
    x_tr_sp, y_tr_sp = _match_attempted_to_labels(x_attempted, y_at_dec, y_tr_im, num_class)
    x_ts_sp, y_ts_sp = _match_attempted_to_labels(x_attempted, y_at_dec, y_ts_im, num_class)
    x_val_sp, y_val_sp = _match_attempted_to_labels(x_attempted, y_at_dec, y_val_im, num_class)
    
    
    print("\nSplit sizes before augmentation:")
    print(f"  train: imagined={x_tr_im.shape[0]}, attempted={x_tr_sp.shape[0]}, listening={x_tr_li.shape[0]}")
    print(f"  val:   imagined={x_val_im.shape[0]}, attempted={x_val_sp.shape[0]}, listening={x_val_li.shape[0]}")
    print(f"  test:  imagined={x_ts_im.shape[0]}, attempted={x_ts_sp.shape[0]}, listening={x_ts_li.shape[0]}")

    rng = np.random.RandomState(seed)
    x_tr_im, x_tr_sp, y_train_dec = _augment_split_to_target(
        x_tr_im,
        x_tr_sp,
        y_tr_im,
        num_class=num_class,
        target_per_class=augment_target_per_class,
        noise_std=augment_noise_std,
        rng=rng,
    )
    

    y_test_dec = y_ts_im
    x_val_im, x_val_sp, y_val_dec = _augment_split_once(
        x_val_im,
        x_val_sp,
        y_val_im,
        noise_std=augment_noise_std,
        rng=rng,
    )
    

    print("Split sizes after augmentation:")
    print(f"  train: imagined={x_tr_im.shape[0]}, attempted={x_tr_sp.shape[0]}, listening={x_tr_li.shape[0]}")
    print(f"  val:   imagined={x_val_im.shape[0]}, attempted={x_val_sp.shape[0]}, listening={x_val_li.shape[0]}")
    print(f"  test:  imagined={x_ts_im.shape[0]}, attempted={x_ts_sp.shape[0]}, listening={x_ts_li.shape[0]}")

    # Fit shared filters on TRAIN only, using both imagined and attempted conditions.
    x_tr_both = np.concatenate([x_tr_im, x_tr_sp], axis=0)
    y_tr_both = np.concatenate([y_train_dec, y_tr_sp], axis=0)

    y_tr_one_hot = np.zeros((num_class, y_tr_both.shape[0]), dtype=np.int32)
    for cls in range(1, num_class + 1):
        y_tr_one_hot[cls - 1, y_tr_both == cls] = 1

    w, la = proc_multicsp_train(
        x_tr_both,
        y_tr_one_hot,
        n_comps=numcsp,
        centered=True,
        method="all",
        way="one-vs-all",
    )

    tr_im_ts = apply_linear_derivation(x_tr_im, w)
    ts_im_ts = apply_linear_derivation(x_ts_im, w)
    val_im_ts = apply_linear_derivation(x_val_im, w)
    tr_sp_ts = apply_linear_derivation(x_tr_sp, w)
    ts_sp_ts = apply_linear_derivation(x_ts_sp, w)
    val_sp_ts = apply_linear_derivation(x_val_sp, w)
    tr_li_ts = apply_linear_derivation(x_tr_li, w)
    ts_li_ts = apply_linear_derivation(x_ts_li, w)
    val_li_ts = apply_linear_derivation(x_val_li, w)

    out = {
        "imagined_train": _segment_variance_log(tr_im_ts, n_sess=n_sess),
        "imagined_test": _segment_variance_log(ts_im_ts, n_sess=n_sess),
        "imagined_val": _segment_variance_log(val_im_ts, n_sess=n_sess),
        "attempted_train": _segment_variance_log(tr_sp_ts, n_sess=n_sess),
        "attempted_test": _segment_variance_log(ts_sp_ts, n_sess=n_sess),
        "attempted_val": _segment_variance_log(val_sp_ts, n_sess=n_sess),
        "listening_train": _segment_variance_log(tr_li_ts, n_sess=n_sess),
        "listening_test": _segment_variance_log(ts_li_ts, n_sess=n_sess),
        "listening_val": _segment_variance_log(val_li_ts, n_sess=n_sess),
        "y_train_dec": y_train_dec,
        "y_test_dec": y_test_dec,
        "y_val_dec": y_val_dec,
        "y_listening_train_dec": y_tr_li,
        "y_listening_test_dec": y_ts_li,
        "y_listening_val_dec": y_val_li,
        "csp_w": w,
        "csp_eigvals": la,
    }
    return out


def prepare_vector_embedding_inputs(
    epochs_all: Dict[int, Dict[int, mne.Epochs]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build imagined/attempted/listening arrays for run_vector_embedding_pipeline.

    Matching strategy:
    - per subject
    - keep all trials from each condition
    - keep only classes present in both conditions globally

    Returns
    -------
    x_imagined, y_imagined, x_attempted, y_attempted, x_listening, y_listening
    """
    imagined_cond = 1
    listening_cond = 2
    attempted_cond = 3
    imagined_base = CONDITION_BASE[imagined_cond]
    listening_base = CONDITION_BASE[listening_cond]
    attempted_base = CONDITION_BASE[attempted_cond]

    imagined_by_subject = epochs_all.get(imagined_cond, {})
    listening_by_subject = epochs_all.get(listening_cond, {})
    attempted_by_subject = epochs_all.get(attempted_cond, {})
    common_subjects = sorted(
        set(imagined_by_subject.keys())
        & set(listening_by_subject.keys())
        & set(attempted_by_subject.keys())
    )

    if not common_subjects:
        raise ValueError("No subjects with imagined, attempted, and listening epochs were found.")

    x_im_list = []
    y_im_list = []
    x_li_list = []
    y_li_list = []
    x_at_list = []
    y_at_list = []

    for subject in common_subjects:
        ep_im = imagined_by_subject[subject]
        ep_li = listening_by_subject[subject]
        ep_at = attempted_by_subject[subject]

        x_im = ep_im.get_data()
        y_im = (ep_im.events[:, 2] - imagined_base).astype(np.int32)
        x_li = ep_li.get_data()
        y_li = (ep_li.events[:, 2] - listening_base).astype(np.int32)
        x_at = ep_at.get_data()
        y_at = (ep_at.events[:, 2] - attempted_base).astype(np.int32)

        x_im_list.append(x_im)
        y_im_list.append(y_im)
        x_li_list.append(x_li)
        y_li_list.append(y_li)
        x_at_list.append(x_at)
        y_at_list.append(y_at)

    if not x_im_list or not x_at_list or not x_li_list:
        raise ValueError("No imagined/attempted/listening samples were produced from common subjects.")
    
    x_imagined = np.concatenate(x_im_list, axis=0)
    y_imagined = np.concatenate(y_im_list, axis=0)
    x_listening = np.concatenate(x_li_list, axis=0)
    y_listening = np.concatenate(y_li_list, axis=0)
    x_attempted = np.concatenate(x_at_list, axis=0)
    y_attempted = np.concatenate(y_at_list, axis=0)

    common_classes = np.intersect1d(
        np.intersect1d(np.unique(y_imagined), np.unique(y_attempted)),
        np.unique(y_listening),
    )
    if common_classes.size == 0:
        raise ValueError("No common class labels across imagined, attempted, and listening conditions.")

    im_mask = np.isin(y_imagined, common_classes)
    li_mask = np.isin(y_listening, common_classes)
    at_mask = np.isin(y_attempted, common_classes)
    x_imagined = x_imagined[im_mask]
    y_imagined = y_imagined[im_mask]
    x_listening = x_listening[li_mask]
    y_listening = y_listening[li_mask]
    x_attempted = x_attempted[at_mask]
    y_attempted = y_attempted[at_mask]

    return x_imagined, y_imagined, x_attempted, y_attempted, x_listening, y_listening


def summarize_class_counts(epochs_all: Dict[int, Dict[int, mne.Epochs]]) -> None:
    """Print per-class counts for imagined/attempted and matched minima.

    This helps determine how much augmentation is needed per class before
    fold-wise balancing.
    """
    imagined_cond = 1
    attempted_cond = 3
    imagined_base = CONDITION_BASE[imagined_cond]
    attempted_base = CONDITION_BASE[attempted_cond]

    imagined_by_subject = epochs_all.get(imagined_cond, {})
    attempted_by_subject = epochs_all.get(attempted_cond, {})
    common_subjects = sorted(set(imagined_by_subject.keys()) & set(attempted_by_subject.keys()))

    if not common_subjects:
        print("No common subjects found between imagined and attempted conditions.")
        return

    print("\nPer-class counts (before pairing/augmentation):")
    print("class | imagined_total | attempted_total | matched_min_total")

    classes = list(range(1, 100))
    for cls in classes:
        im_total = 0
        at_total = 0
        matched_total = 0
        for subject in common_subjects:
            ep_im = imagined_by_subject[subject]
            ep_at = attempted_by_subject[subject]

            y_im = (ep_im.events[:, 2] - imagined_base).astype(np.int32)
            y_at = (ep_at.events[:, 2] - attempted_base).astype(np.int32)

            n_im = int(np.sum(y_im == cls))
            n_at = int(np.sum(y_at == cls))
            im_total += n_im
            at_total += n_at
            matched_total += min(n_im, n_at)

        if im_total > 0 or at_total > 0:
            print(f"{cls:5d} | {im_total:14d} | {at_total:14d} | {matched_total:17d}")


def save_splits_to_csv(
    out: Dict[str, np.ndarray],
    output_dir: str,
    subject_id: int,
    condition_name: str,
    condition_prefix: str,
    label_prefix: str | None = None,
) -> None:
    """Save one CSV per epoch under output_dir/subj#/condition/train|val|test.

    Each CSV contains only feature values (no header/index). The class label is
    embedded in the file name.
    """
    subj_dir = os.path.join(output_dir, f"subj{subject_id}")
    cond_dir = os.path.join(subj_dir, condition_name)
    label_prefix = condition_prefix if label_prefix is None else label_prefix
    split_map = {
        "train": (f"{condition_prefix}_train", f"y_{label_prefix}_train_dec" if label_prefix != "imagined" and label_prefix != "attempted" else "y_train_dec"),
        "val": (f"{condition_prefix}_val", f"y_{label_prefix}_val_dec" if label_prefix != "imagined" and label_prefix != "attempted" else "y_val_dec"),
        "test": (f"{condition_prefix}_test", f"y_{label_prefix}_test_dec" if label_prefix != "imagined" and label_prefix != "attempted" else "y_test_dec"),
    }

    for split_name, (x_key, y_key) in split_map.items():
        x_split = out[x_key]
        y_split = out[y_key]

        if x_split.ndim != 3:
            raise ValueError(f"Expected 3D split for {condition_prefix}/{split_name}, got {x_split.shape}")
        if x_split.shape[0] != y_split.shape[0]:
            raise ValueError(
                f"Mismatch between features and labels for {split_name}: "
                f"{x_split.shape[0]} vs {y_split.shape[0]}"
            )

        split_dir = os.path.join(cond_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i in range(x_split.shape[0]):
            label = int(y_split[i])
            epoch_mat = x_split[i]
            csv_name = f"label{label:03d}_epoch{i:04d}.csv"
            csv_path = os.path.join(split_dir, csv_name)
            pd.DataFrame(epoch_mat).to_csv(csv_path, index=False, header=False)

        print(
            f"Saved {condition_name} {split_name} epoch CSVs: "
            f"{x_split.shape[0]} files in {split_dir}"
        )



def main() -> None:
    print("MATLAB-like vector_embedding pipeline custom csp multi-class one-vs-all extraction")
    eeg_data_dir = "clean_data01-120Hz"
    output_dir = "eegdata"
    #p.add_argument("--input", type=Path, required=True, help="NPZ with x_imagined, y_imagined, x_attempted, y_attempted")
    #p.add_argument("--output", type=Path, required=True, help="Output NPZ path for CSP feature tensors")
    #return p.parse_args()
    #args = _parse_args()
    numcsp = 4
    n_sess = 16
    num_class = 13
    n_fold = 3
    seed = 0
    trials_per_class = None  # inferred from imagined data; split-wise augmentation reaches 15/class
    
    subjects = [18]

    raw_all, markers_all, code_to_name = load_data(subjects, data_dir=eeg_data_dir)
    epochs_all = extract_epochs(raw_all, markers_all)
    #X_raw, y_raw_class, y_raw_cond, y_raw_subject = build_arrays(epochs_all)

    summarize_class_counts(epochs_all)

    x_imagined, y_imagined, x_attempted, y_attempted, x_listening, y_listening = prepare_vector_embedding_inputs(epochs_all)
    print(
        "Prepared inputs | "
        f"imagined: {x_imagined.shape}, attempted: {x_attempted.shape}, listening: {x_listening.shape}"
    )

    out = run_vector_embedding_pipeline(
        x_imagined=x_imagined,
        y_imagined=y_imagined,
        x_attempted=x_attempted,
        y_attempted=y_attempted,
        x_listening=x_listening,
        y_listening=y_listening,
        numcsp=numcsp,
        n_sess=n_sess,
        num_class=num_class,
        n_fold=n_fold,
        seed=seed,
        trials_per_class=trials_per_class,
        val_ratio=0.2,
        test_ratio=0.1,
    )

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "csp_features_sub18.npz"), **out)
    save_splits_to_csv(out, output_dir, subjects[0], "imagined_speech", "imagined")
    save_splits_to_csv(out, output_dir, subjects[0], "attempted_speech", "attempted")
    save_splits_to_csv(out, output_dir, subjects[0], "listening", "listening", label_prefix="listening")
    print(f"Saved CSP features to: {os.path.join(output_dir, 'csp_features_sub18.npz')}")
    for key, value in out.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
            if value.ndim >= 1 and value.shape[0] > 0:
                print(f"  single epoch shape: {value[0].shape}")


if __name__ == "__main__":
    main()
