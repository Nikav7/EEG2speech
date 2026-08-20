"""
1) Build class-balanced train/test/val indices using class-aware logic.
2) Fit CSP on imagined and attempted epochs.
3) Apply fitted CSP to all conditions; imagined, attempted and listening - train/test/val.
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
#WIN_MS = 250 #not used
#STRIDE_MS = 125 #not used
EVENT_SFREQ = 250  # Original event sr for rescaling if needed, markers of data filtered between 0.1 and 120 are already resampled
SFREQ = 250
#TARGET_STEPS = 85 # not used at the moment

#Load
def load_data(subjects, data_dir=None):
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
    seed: int = 0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    enforce_val_class_coverage: bool = True,
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
    if enforce_val_class_coverage:
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

    # Safety: enforce strict disjointness so imagined samples cannot leak across splits.
    if np.intersect1d(train_idx, val_idx).size > 0:
        raise ValueError("Leakage detected: train and val imagined indices overlap.")
    if np.intersect1d(train_idx, test_idx).size > 0:
        raise ValueError("Leakage detected: train and test imagined indices overlap.")
    if np.intersect1d(val_idx, test_idx).size > 0:
        raise ValueError("Leakage detected: val and test imagined indices overlap.")

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
    y_dec_im: np.ndarray,
    y_dec_at: np.ndarray,
    *,
    num_class: int,
    target_per_class: int,
    noise_std: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augment a split so each class has at least target_per_class samples."""
    x_im_aug = [x_im]
    x_at_aug = [x_sp]
    y_aug_im = [y_dec_im]
    y_aug_at = [y_dec_at]

    for cls in range(1, num_class + 1):
        cls_idx_im = np.flatnonzero(y_dec_im == cls)
        n_have_im = cls_idx_im.size

        cls_idx_at = np.flatnonzero(y_dec_at == cls)
        n_have_at = cls_idx_at.size


        # Do not over-augment: each original sample can create at most two
        # augmented copies.
        n_add_im = min(target_per_class - n_have_im, 2 * n_have_im)
        if n_add_im > 0:
            pick_im = cls_idx_im[np.arange(n_add_im) % n_have_im]

        n_add_at = min(target_per_class - n_have_at, 2 * n_have_at)
        if n_add_at > 0:
            pick_at = cls_idx_at[np.arange(n_add_at) % n_have_at]

        # Per-copy scale keeps noise minimal while making repeated copies non-identical.
        if n_add_im > 0:
            im_scale = rng.uniform(0.95, 1.05, size=(n_add_im, 1, 1))
            im_noise = rng.normal(0.0, noise_std, size=x_im[pick_im].shape) * im_scale
            im_add = x_im[pick_im] + im_noise
            y_add_im = np.full(n_add_im, cls, dtype=np.int32)

            x_im_aug.append(im_add.astype(x_im.dtype, copy=False))
            y_aug_im.append(y_add_im)

        if n_add_at > 0:
            at_scale = rng.uniform(0.95, 1.05, size=(n_add_at, 1, 1))
            at_noise = rng.normal(0.0, noise_std, size=x_sp[pick_at].shape) * at_scale
            at_add = x_sp[pick_at] + at_noise
            y_add_at = np.full(n_add_at, cls, dtype=np.int32)

            x_at_aug.append(at_add.astype(x_sp.dtype, copy=False))
            y_aug_at.append(y_add_at)
        
    return (
        np.concatenate(x_im_aug, axis=0),
        np.concatenate(x_at_aug, axis=0),
        np.concatenate(y_aug_im, axis=0),
        np.concatenate(y_aug_at, axis=0),
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


# def _infer_trials_per_class(y_dec: np.ndarray, num_class: int) -> int:
#     """Infer class-balanced split depth from imagined labels."""
#     counts = [int(np.sum(y_dec == cls)) for cls in range(1, num_class + 1)]
#     if min(counts) <= 0:
#         missing = [str(i + 1) for i, c in enumerate(counts) if c <= 0]
#         raise ValueError(f"Missing imagined samples for classes: {', '.join(missing)}")
#     return min(counts)


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
    label_num_class: int = 74,
    n_fold: int = 5,
    seed: int = 0,
    trials_per_class: int | None = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    augment_target_per_class: int = 9, #changed to from 15 to 9 when training on more subjects
    augment_noise_std: float = 1e-4,
    use_augmentation: bool = True,
    include_listening_in_csp_train: bool = False,
    enforce_val_class_coverage: bool = True,
    csp_class_ids: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Run MATLAB-like vector embedding CSP flow using manual CSP algo, loading with mne.

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

    # Keep class cardinality fixed (label_num_class), label IDs to contiguous 1..label_num_class for counting
    im_unique = np.unique(y_im_dec)
    at_unique = np.unique(y_at_dec)
    li_unique = np.unique(y_li_dec)

    if im_unique.size != label_num_class:
        # Subject-specific exception path to allow fewer classes in validation
        # class coverage enforcement is disabled for subject 15.
        if (not enforce_val_class_coverage) and (im_unique.size < label_num_class):
            print(
                "Info: reduced label_num_class from "
                f"{label_num_class} to {im_unique.size} for exception subject"
            )
            label_num_class = int(im_unique.size)
        else:
            raise ValueError(
                f"Imagined labels contain {im_unique.size} classes, expected {label_num_class}."
            )
    if not np.array_equal(im_unique, at_unique) or not np.array_equal(im_unique, li_unique):
        raise ValueError(
            "Imagined/attempted/listening label sets differ; cannot build a consistent 74-class mapping."
        )

    target = np.arange(1, label_num_class + 1, dtype=np.int32)


    if use_augmentation:
        split = make_split_indices(
            y_dec=y_im_dec,
            num_class=label_num_class,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            enforce_val_class_coverage=enforce_val_class_coverage,
        )
    else:
        # When augmentation is disabled, allow a plain random 70/20/10 split.
        split = make_simple_split_indices(
            x_imagined.shape[0],
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    # Listening is not augmented; split with class coverage to keep ≥1 sample per class in val.
    listening_split = make_split_indices(
        y_dec=y_li_dec,
        num_class=label_num_class,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        enforce_val_class_coverage=enforce_val_class_coverage,
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

    # Train: keep matching so imagined and attempted are class-aligned for CSP and augmentation.
    x_tr_at, y_tr_at = _match_attempted_to_labels(x_attempted, y_at_dec, y_tr_im, label_num_class)

    # Val/Test: independent split so no attempted sample leaks across splits via cycling.
    attempted_split = make_split_indices(
        y_dec=y_at_dec,
        num_class=label_num_class,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        enforce_val_class_coverage=enforce_val_class_coverage,
    )
    x_ts_at = x_attempted[attempted_split.test]
    y_ts_at = y_at_dec[attempted_split.test]
    x_val_at = x_attempted[attempted_split.val]
    y_val_at = y_at_dec[attempted_split.val]

    # Save pre-augmentation labels for per-subject stats export.
    y_pre_im_train = y_tr_im.copy()
    y_pre_im_val = y_val_im.copy()
    y_pre_im_test = y_ts_im.copy()
    y_pre_at_train = y_tr_at.copy()
    y_pre_at_val = y_val_at.copy()
    y_pre_at_test = y_ts_at.copy()
    x_pre_im_train = x_tr_im.copy()
    x_pre_im_val = x_val_im.copy()
    x_pre_im_test = x_ts_im.copy()
    x_pre_at_train = x_tr_at.copy()
    x_pre_at_val = x_val_at.copy()
    x_pre_at_test = x_ts_at.copy()
    x_pre_li_train = x_tr_li.copy()
    x_pre_li_val = x_val_li.copy()
    x_pre_li_test = x_ts_li.copy()


    print("\nSplit sizes before augmentation:")
    print(f"  train: imagined={x_tr_im.shape[0]}, attempted={x_tr_at.shape[0]}, listening={x_tr_li.shape[0]}")
    print(f"  val:   imagined={x_val_im.shape[0]}, attempted={x_val_at.shape[0]}, listening={x_val_li.shape[0]}")
    print(f"  test:  imagined={x_ts_im.shape[0]}, attempted={x_ts_at.shape[0]}, listening={x_ts_li.shape[0]}")

    if use_augmentation:
        rng = np.random.RandomState(seed)
        x_tr_im, x_tr_at, y_train_dec, y_tr_at = _augment_split_to_target(
            x_tr_im,
            x_tr_at,
            y_tr_im,
            y_tr_at,
            num_class=label_num_class,
            target_per_class=augment_target_per_class,
            noise_std=augment_noise_std,
            rng=rng,
        )
        
        # Keep attempted-train labels aligned with augmented attempted-train epochs.
        #y_tr_at = y_train_dec.copy()

        y_test_dec = y_ts_im

        # double imagined val only for subsequent gan training 
        # #attempted val uses its own independent split for csp training.
        im_noise = rng.normal(0.0, augment_noise_std, size=x_val_im.shape)
        x_val_im = np.concatenate([x_val_im, (x_val_im + im_noise).astype(x_val_im.dtype, copy=False)], axis=0)
        y_val_dec = np.concatenate([y_val_im, y_val_im], axis=0)
    else:
        y_train_dec = y_tr_im
        y_test_dec = y_ts_im
        y_val_dec = y_val_im
        # Keep attempted labels explicit and aligned in non-augmented mode.
        y_tr_at = y_train_dec.copy()
        y_val_at = y_val_dec.copy()

    print("Split sizes after augmentation:")
    print(f"  train: imagined={x_tr_im.shape[0]}, attempted={x_tr_at.shape[0]}, listening={x_tr_li.shape[0]}")
    print(f"  val:   imagined={x_val_im.shape[0]}, attempted={x_val_at.shape[0]}, listening={x_val_li.shape[0]}")
    print(f"  test:  imagined={x_ts_im.shape[0]}, attempted={x_ts_at.shape[0]}, listening={x_ts_li.shape[0]}")

    # Fit shared filters on TRAIN only, using imagined + attempted, with optional listening.
    x_tr_sources = [x_tr_im, x_tr_at]
    y_tr_sources = [y_train_dec, y_tr_at]
    if include_listening_in_csp_train:
        x_tr_sources.append(x_tr_li)
        y_tr_sources.append(y_tr_li)
        print("Info: including listening_train in CSP filter fitting")

    x_tr_both = np.concatenate(x_tr_sources, axis=0)
    y_tr_both = np.concatenate(y_tr_sources, axis=0)

    # Build one-hot using the num_class of common classes across subjs for CSP reference cl IDs.
    if csp_class_ids is None:
        csp_class_ids = np.arange(1, num_class + 1, dtype=np.int32)
    if len(csp_class_ids) != num_class:
        raise ValueError(f"csp_class_ids must have exactly {num_class} entries, got {len(csp_class_ids)}")
    y_tr_one_hot = np.zeros((len(csp_class_ids), y_tr_both.shape[0]), dtype=np.int32)
    for i, cls in enumerate(csp_class_ids):
        y_tr_one_hot[i, y_tr_both == cls] = 1

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
    tr_at_ts = apply_linear_derivation(x_tr_at, w)
    ts_at_ts = apply_linear_derivation(x_ts_at, w)
    val_at_ts = apply_linear_derivation(x_val_at, w)
    tr_li_ts = apply_linear_derivation(x_tr_li, w)
    ts_li_ts = apply_linear_derivation(x_ts_li, w)
    val_li_ts = apply_linear_derivation(x_val_li, w)
    tr_im_pre_ts = apply_linear_derivation(x_pre_im_train, w)
    ts_im_pre_ts = apply_linear_derivation(x_pre_im_test, w)
    val_im_pre_ts = apply_linear_derivation(x_pre_im_val, w)
    tr_at_pre_ts = apply_linear_derivation(x_pre_at_train, w)
    ts_at_pre_ts = apply_linear_derivation(x_pre_at_test, w)
    val_at_pre_ts = apply_linear_derivation(x_pre_at_val, w)
    tr_li_pre_ts = apply_linear_derivation(x_pre_li_train, w)
    ts_li_pre_ts = apply_linear_derivation(x_pre_li_test, w)
    val_li_pre_ts = apply_linear_derivation(x_pre_li_val, w)

    out = {

        # Label arrays for augmented data
        "y_train_dec": y_train_dec,
        "y_test_dec": y_test_dec,
        "y_val_dec": y_val_dec,
        "y_listening_train_dec": y_tr_li,
        "y_listening_test_dec": y_ts_li,
        "y_listening_val_dec": y_val_li,
        "y_pre_imagined_train_dec": y_pre_im_train,
        "y_pre_imagined_test_dec": y_pre_im_test,
        "y_pre_imagined_val_dec": y_pre_im_val,
        "y_pre_attempted_train_dec": y_pre_at_train,
        "y_pre_attempted_test_dec": y_pre_at_test,
        "y_pre_attempted_val_dec": y_pre_at_val,
        "y_post_attempted_train_dec": y_tr_at,
        "y_post_attempted_test_dec": y_ts_at,
        "y_post_attempted_val_dec": y_val_at,

        # CSP-transformed post-augmentation (log-variance features)
        "post_imagined_train": _segment_variance_log(tr_im_ts, n_sess=n_sess),
        "post_imagined_test": _segment_variance_log(ts_im_ts, n_sess=n_sess),
        "post_imagined_val": _segment_variance_log(val_im_ts, n_sess=n_sess),
        "post_attempted_train": _segment_variance_log(tr_at_ts, n_sess=n_sess),
        "post_attempted_test": _segment_variance_log(ts_at_ts, n_sess=n_sess),
        "post_attempted_val": _segment_variance_log(val_at_ts, n_sess=n_sess),
        "post_listening_train": _segment_variance_log(tr_li_ts, n_sess=n_sess),
        "post_listening_test": _segment_variance_log(ts_li_ts, n_sess=n_sess),
        "post_listening_val": _segment_variance_log(val_li_ts, n_sess=n_sess),

        # CSP-transformed non-augmented (log-variance features)
        "pre_imagined_train": _segment_variance_log(tr_im_pre_ts, n_sess=n_sess),
        "pre_imagined_test": _segment_variance_log(ts_im_pre_ts, n_sess=n_sess),
        "pre_imagined_val": _segment_variance_log(val_im_pre_ts, n_sess=n_sess),
        "pre_attempted_train": _segment_variance_log(tr_at_pre_ts, n_sess=n_sess),
        "pre_attempted_test": _segment_variance_log(ts_at_pre_ts, n_sess=n_sess),
        "pre_attempted_val": _segment_variance_log(val_at_pre_ts, n_sess=n_sess),
        "pre_listening_train": _segment_variance_log(tr_li_pre_ts, n_sess=n_sess),
        "pre_listening_test": _segment_variance_log(ts_li_pre_ts, n_sess=n_sess),
        "pre_listening_val": _segment_variance_log(val_li_pre_ts, n_sess=n_sess),
        
        #Raw pre-augmentation (full time-series)
        "raw_pre_imagined_train": x_pre_im_train,
        "raw_pre_imagined_test": x_pre_im_test,
        "raw_pre_imagined_val": x_pre_im_val,
        "raw_pre_attempted_train": x_pre_at_train,
        "raw_pre_attempted_test": x_pre_at_test,
        "raw_pre_attempted_val": x_pre_at_val,
        "raw_pre_listening_train": x_pre_li_train,
        "raw_pre_listening_test": x_pre_li_test,
        "raw_pre_listening_val": x_pre_li_val,
        
        # Raw post-augmentation (full time-series, before CSP)
        "raw_post_imagined_train": x_tr_im,
        "raw_post_imagined_test": x_ts_im,
        "raw_post_imagined_val": x_val_im,
        "raw_post_attempted_train": x_tr_at,
        "raw_post_attempted_test": x_ts_at,
        "raw_post_attempted_val": x_val_at,
        "raw_post_listening_train": x_tr_li,
        "raw_post_listening_test": x_ts_li,
        "raw_post_listening_val": x_val_li,

        
        # CSP filters and eigenvalues
        "csp_w": w,
        "csp_eigvals": la,
    }
    return out


def prepare_vector_embedding_inputs(
    epochs_all: Dict[int, Dict[int, mne.Epochs]],
    global_class_map: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build imagined/attempted/listening arrays for run_vector_embedding_pipeline.

    Matching strategy:
    - per subject
    - keep all trials from each condition
    - keep only classes present in both conditions globally

    Returns
    -------
    x_imagined, y_imagined, x_attempted, y_attempted, x_listening, y_listening, common_classes, class_map
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

    # Global common classes first (consistent IDs 1..N), then per-subject extras sequentially.
    if global_class_map is not None:
        class_map = dict(global_class_map)
        extra_id = len(class_map) + 1
        for cls in np.sort(common_classes):
            if int(cls) not in class_map:
                class_map[int(cls)] = extra_id
                extra_id += 1
    else:
        class_map = {int(cls): idx + 1 for idx, cls in enumerate(np.sort(common_classes))}
    y_imagined = np.asarray([class_map[int(v)] for v in y_imagined], dtype=np.int32)
    y_listening = np.asarray([class_map[int(v)] for v in y_listening], dtype=np.int32)
    y_attempted = np.asarray([class_map[int(v)] for v in y_attempted], dtype=np.int32)

    return x_imagined, y_imagined, x_attempted, y_attempted, x_listening, y_listening, common_classes.astype(np.int32, copy=True), class_map


def save_raw_splits_to_csv(
    out: Dict[str, np.ndarray],
    output_dir: str,
    subject_id: int,
    condition_name: str,
    condition_prefix: str,
    original_labels: np.ndarray,
    label_prefix: str | None = None,
) -> None:
    """Save one CSV per epoch (raw time-series) under output_dir/subj#/condition/train|val|test.

    Each CSV contains raw epoch data with shape (n_channels, n_times).
    The class label is embedded in the file name.
    """
    subj_dir = os.path.join(output_dir, f"subj{subject_id}")
    cond_dir = os.path.join(subj_dir, condition_name)
    label_prefix = condition_prefix if label_prefix is None else label_prefix
    original_labels = np.asarray(original_labels).astype(np.int32)
    split_map = {
        "train": (f"raw_{condition_prefix}_train", f"y_{label_prefix}_train_dec" if label_prefix not in ["imagined", "attempted"] else "y_train_dec"),
        "val": (f"raw_{condition_prefix}_val", f"y_{label_prefix}_val_dec" if label_prefix not in ["imagined", "attempted"] else "y_val_dec"),
        "test": (f"raw_{condition_prefix}_test", f"y_{label_prefix}_test_dec" if label_prefix not in ["imagined", "attempted"] else "y_test_dec"),
    }

    for split_name, (x_key, y_key) in split_map.items():
        if x_key not in out or y_key not in out:
            print(f"Warning: skipping {condition_prefix}/{split_name} - missing keys {x_key}, {y_key}")
            continue
        
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
            remapped_label = int(y_split[i])
            if remapped_label < 1 or remapped_label > original_labels.size:
                raise ValueError(
                    f"Remapped label {remapped_label} is out of range for {condition_name}/{split_name}."
                )
            label = int(original_labels[remapped_label - 1])
            epoch_mat = x_split[i]  # shape: (n_channels, n_times)
            csv_name = f"label{label:03d}_epoch{i:04d}.csv"
            csv_path = os.path.join(split_dir, csv_name)
            pd.DataFrame(epoch_mat).to_csv(csv_path, index=False, header=False)

        print(
            f"Saved {condition_name} {split_name} raw epoch CSVs: "
            f"{x_split.shape[0]} files in {split_dir}"
        )


def save_label_metadata_csv(
    output_dir: str,
    subject_id: int,
    common_classes: np.ndarray,
) -> None:
    """Save original-to-remapped label metadata for one subject."""
    subj_dir = os.path.join(output_dir, f"subj{subject_id}")
    os.makedirs(subj_dir, exist_ok=True)

    remapped_labels = np.arange(1, common_classes.size + 1, dtype=np.int32)
    rows = []
    for condition_name in ("imagined_speech", "attempted_speech", "listening"):
        for original_label, remapped_label in zip(common_classes.tolist(), remapped_labels.tolist()):
            rows.append(
                {
                    "subject_id": int(subject_id),
                    "condition": condition_name,
                    "original_label": int(original_label),
                    "remapped_label": int(remapped_label),
                }
            )

    metadata_path = os.path.join(subj_dir, "label_metadata.csv")
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    print(f"Saved label metadata CSV: {metadata_path}")


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


def accumulate_preaugm_imagined_split_counts(
    aggregate: Dict[str, Dict[int, int]],
    y_imagined: np.ndarray,
    original_labels: np.ndarray,
    *,
    use_augmentation: bool,
    label_num_class: int,
    n_fold: int,
    seed: int,
    trials_per_class: int | None,
    val_ratio: float,
    test_ratio: float,
    enforce_val_class_coverage: bool,
) -> None:
    """Accumulate preaugmentation imagined samples counts per split and original label."""
    y_im_dec = to_decoded_labels(y_imagined)
    if use_augmentation:
        split = make_split_indices(
            y_dec=y_im_dec,
            num_class=label_num_class,
            n_fold=n_fold,
            seed=seed,
            trials_per_class=trials_per_class,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            enforce_val_class_coverage=enforce_val_class_coverage,
        )
    else:
        split = make_simple_split_indices(
            y_im_dec.shape[0],
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    split_labels = {
        "train": y_im_dec[split.train],
        "val": y_im_dec[split.val],
        "test": y_im_dec[split.test],
    }
    original_labels = np.asarray(original_labels).astype(np.int32)
    for split_name, y_split in split_labels.items():
        uniq, counts = np.unique(y_split, return_counts=True)
        for remapped_label, count in zip(uniq.tolist(), counts.tolist()):
            original_label = int(original_labels[int(remapped_label) - 1])
            aggregate[split_name][original_label] = aggregate[split_name].get(original_label, 0) + int(count)


def save_splits_to_csv(
    out: Dict[str, np.ndarray],
    output_dir: str,
    subject_id: int,
    condition_name: str,
    condition_prefix: str,
    original_labels: np.ndarray,
    label_prefix: str | None = None,
) -> None:
    """Save one CSV per epoch under output_dir/subj#/condition/train|val|test.

    Each CSV contains only feature values (no header/index). The class label is
    embedded in the file name.
    """
    subj_dir = os.path.join(output_dir, f"subj{subject_id}")
    cond_dir = os.path.join(subj_dir, condition_name)
    label_prefix = condition_prefix if label_prefix is None else label_prefix
    original_labels = np.asarray(original_labels).astype(np.int32)
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
            remapped_label = int(y_split[i])
            if remapped_label < 1 or remapped_label > original_labels.size:
                raise ValueError(
                    f"Remapped label {remapped_label} is out of range for {condition_name}/{split_name}."
                )
            label = int(original_labels[remapped_label - 1])
            epoch_mat = x_split[i]
            csv_name = f"label{label:03d}_epoch{i:04d}.csv"
            csv_path = os.path.join(split_dir, csv_name)
            pd.DataFrame(epoch_mat).to_csv(csv_path, index=False, header=False)

        print(
            f"Saved {condition_name} {split_name} epoch CSVs: "
            f"{x_split.shape[0]} files in {split_dir}"
        )


def save_subject_augmentation_stats_csv(
    output_dir: str,
    subject_id: int,
    out: Dict[str, np.ndarray],
    original_labels: np.ndarray,
) -> None:
    """Save per-class before/after augmentation counts per split for one subject."""
    subj_dir = os.path.join(output_dir, f"subj{subject_id}")
    os.makedirs(subj_dir, exist_ok=True)

    original_labels = np.asarray(original_labels).astype(np.int32)
    remapped_labels = np.arange(1, original_labels.size + 1, dtype=np.int32)

    split_keys = {
        "train": {
            "imagined_before": "y_pre_imagined_train_dec",
            "imagined_after": "y_train_dec",
            "attempted_before": "y_pre_attempted_train_dec",
            "attempted_after": "y_post_attempted_train_dec",
        },
        "val": {
            "imagined_before": "y_pre_imagined_val_dec",
            "imagined_after": "y_val_dec",
            "attempted_before": "y_pre_attempted_val_dec",
            "attempted_after": "y_post_attempted_val_dec",
        },
        "test": {
            "imagined_before": "y_pre_imagined_test_dec",
            "imagined_after": "y_test_dec",
            "attempted_before": "y_pre_attempted_test_dec",
            "attempted_after": "y_post_attempted_test_dec",
        },
    }

    rows = []
    for split_name, key_map in split_keys.items():
        for remapped_label, original_label in zip(remapped_labels.tolist(), original_labels.tolist()):
            im_before = int(np.sum(out[key_map["imagined_before"]] == remapped_label))
            im_after = int(np.sum(out[key_map["imagined_after"]] == remapped_label))
            sp_before = int(np.sum(out[key_map["attempted_before"]] == remapped_label))
            sp_after = int(np.sum(out[key_map["attempted_after"]] == remapped_label))

            rows.append(
                {
                    "subject_id": int(subject_id),
                    "split": split_name,
                    "condition": "imagined_speech",
                    "original_label": int(original_label),
                    "remapped_label": int(remapped_label),
                    "before_augmentation": im_before,
                    "after_augmentation": im_after,
                }
            )
            rows.append(
                {
                    "subject_id": int(subject_id),
                    "split": split_name,
                    "condition": "attempted_speech",
                    "original_label": int(original_label),
                    "remapped_label": int(remapped_label),
                    "before_augmentation": sp_before,
                    "after_augmentation": sp_after,
                }
            )

    csv_path = os.path.join(subj_dir, "augmentation_class_stats.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved augmentation class stats CSV: {csv_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSP vector-embedding preprocessing")
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=[16, 17, 18, 19],
        help="Subject IDs to process, e.g. --subjects 16 17 18 19",
    )
    parser.add_argument(
        "--eeg-data-dir",
        default="clean_data01-120Hz",
        help="Directory containing clean_eeg_subjXX.npy and events_subjXX.npy files.",
    )
    parser.add_argument(
        "--output-dir",
        default="eegdata_250sr_new",
        help="Output root directory for per-subject split CSVs.",
    )
    parser.add_argument(
        "--non-augmented-output-dir",
        default="non_augmented",
        help="Output root directory for pre-augmentation split CSVs.",
    )
    parser.add_argument(
        "--include-listening-in-csp-train",
        type=bool,
        default=False,
        help="Include listening train samples when fitting shared CSP filters.",
    )
    parser.add_argument(
        "--csp-class-seed",
        type=int,
        default=42, #clsrnd1, clsrnd2 is seed 24
        help="Seed for randomly selecting the 13 CSP reference classes from global common classes.",
    )
    return parser.parse_args()


def compute_global_common_classes(subjects: list, data_dir: str) -> np.ndarray:
    """Return original class codes present across all three conditions for every subject."""
    global_common: np.ndarray | None = None
    for subject_id in subjects:
        raw_all, markers_all, _ = load_data([subject_id], data_dir=data_dir)
        if not raw_all:
            print(f"  pre-pass: subject {subject_id} missing data, skipping")
            continue
        epochs_all = extract_epochs(raw_all, markers_all)
        try:
            _, _, _, _, _, _, subject_common, _ = prepare_vector_embedding_inputs(epochs_all)
        except ValueError as exc:
            print(f"  pre-pass: subject {subject_id} skipped ({exc})")
            continue
        global_common = subject_common.copy() if global_common is None else np.intersect1d(global_common, subject_common)
    if global_common is None or global_common.size == 0:
        raise ValueError("No common classes found across all subjects.")
    return global_common


def main() -> None:
    args = parse_args()
    print("MATLAB-like vector_embedding pipeline custom csp multi-class one-vs-all extraction")
    eeg_data_dir = args.eeg_data_dir
    output_dir = args.output_dir
    non_augmented_output_dir = args.non_augmented_output_dir
    
    numcsp = 4
    n_sess = 16
    num_class = 13
    label_num_class = 74
    n_fold = 5
    seed = 0
    trials_per_class = None  # inferred from imagined data; split-wise augmentation reaches 9/class
    use_augmentation = True
    enforce_val_class_coverage = True
    include_listening_in_csp_train = args.include_listening_in_csp_train
    csp_class_seed = args.csp_class_seed
    
    subjects = sorted(set(args.subjects))
    print(f"Subjects to process: {subjects}")

    # Pre-pass: find classes present in all subjects and randomly pick num_class for CSP.
    print("\nPre-pass: computing global common classes across all subjects...")
    global_common = compute_global_common_classes(subjects, eeg_data_dir)
    print(f"Global common classes ({global_common.size}): {global_common}")
    if global_common.size < num_class:
        raise ValueError(f"Only {global_common.size} global common classes available, need {num_class}.")
    rng_ref = np.random.RandomState(csp_class_seed)
    csp_reference_original_classes = np.sort(
        rng_ref.choice(global_common, size=num_class, replace=False)
    )
    print(f"CSP reference original classes (seed={csp_class_seed}, randomly chosen {num_class}): {csp_reference_original_classes}")

    # Save global CSP metadata: parameters and chosen reference classes.
    os.makedirs(output_dir, exist_ok=True)
    csp_meta_params = {
        "parameter": ["numcsp", "n_sess", "num_class", "csp_class_seed", "augment_seed",
                       "val_ratio", "test_ratio", "augment_target_per_class", "n_fold",
                       "global_common_classes_count", "csp_reference_original_classes"],
        "value": [numcsp, n_sess, num_class, csp_class_seed, seed,
                  0.2, 0.1, 9, n_fold,
                  int(global_common.size), str(csp_reference_original_classes.tolist())],
    }
    pd.DataFrame(csp_meta_params).to_csv(os.path.join(output_dir, "csp_metadata.csv"), index=False)
    # Per-subject remapped IDs accumulated during main loop.
    csp_subject_rows: list = []

    # Build consistent cross-subject label map
    global_class_map = {int(c): i + 1 for i, c in enumerate(np.sort(global_common))}

    # Accumulators for cross-subject summary CSVs (post-augmentation counts).
    imagined_summary_rows: list = []
    attempted_summary_rows: list = []
    listening_summary_rows: list = []

    # Create output directories for three stages
    os.makedirs(output_dir, exist_ok=True)
    
    # save 1, raw pre-augmentation (no augmentation, no CSP)
    raw_pre_aug_dir = os.path.join(non_augmented_output_dir, "raw_pre_augmentation")
    os.makedirs(raw_pre_aug_dir, exist_ok=True)
    
    # save 2, raw post-augmentation (with augmentation, but no CSP)
    raw_post_aug_dir = os.path.join(output_dir, "raw_post_augmentation_no_csp")
    os.makedirs(raw_post_aug_dir, exist_ok=True)
    
    # save 3, CSP-transformed post-augmentation
    csp_post_aug_dir = os.path.join(output_dir, "csp_post_augmentation")
    os.makedirs(csp_post_aug_dir, exist_ok=True)
    
    for subject_id in subjects:
        print(f"\n=== Processing subject {subject_id} ===")
        # Coverage is enforced for all subjects because subject_label_num_class
        # reflects the actual class count.
        enforce_val_class_coverage = True
        raw_all, markers_all, code_to_name = load_data([subject_id], data_dir=eeg_data_dir)
        if not raw_all:
            print(f"Subject {subject_id}: no data loaded, skipping")
            continue

        epochs_all = extract_epochs(raw_all, markers_all)
        summarize_class_counts(epochs_all)

        x_imagined, y_imagined, x_attempted, y_attempted, x_listening, y_listening, common_classes, subject_class_map = prepare_vector_embedding_inputs(epochs_all, global_class_map=global_class_map)
        print(
            "Prepared inputs | "
            f"imagined: {x_imagined.shape}, attempted: {x_attempted.shape}, listening: {x_listening.shape}"
        )
        save_label_metadata_csv(output_dir, subject_id, common_classes)

        print(f"  common_classes ({common_classes.size}): {common_classes}")
        print(f"  y_imagined unique ({np.unique(y_imagined).size}): {np.unique(y_imagined)}")
        print(f"  y_attempted unique ({np.unique(y_attempted).size}): {np.unique(y_attempted)}")
        print(f"  y_listening unique ({np.unique(y_listening).size}): {np.unique(y_listening)}")
        print(f"  x_imagined: {x_imagined.shape}, x_attempted: {x_attempted.shape}, x_listening: {x_listening.shape}")

        # Map global CSP reference classes to remapped IDs using the shared global class map.
        csp_class_ids = np.array(
            [global_class_map[int(c)] for c in csp_reference_original_classes], dtype=np.int32
        )
        print(f"  CSP class IDs (remapped) for this subject: {csp_class_ids}")

        # Record per-subject CSP class mapping.
        for orig_cls, remapped_id in zip(csp_reference_original_classes.tolist(), csp_class_ids.tolist()):
            csp_subject_rows.append({
                "subject_id": subject_id,
                "original_label": int(orig_cls),
                "remapped_label": int(remapped_id),
            })
        subject_label_num_class = int(common_classes.size)
        if subject_label_num_class != label_num_class:
            print(f"  label_num_class adjusted to {subject_label_num_class} (expected {label_num_class})")
        print(f"  class_map (first 5): { {k: subject_class_map[k] for k in list(subject_class_map)[:5]} }")

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
            label_num_class=subject_label_num_class,
            n_fold=n_fold,
            seed=seed,
            trials_per_class=trials_per_class,
            use_augmentation=use_augmentation,
            include_listening_in_csp_train=include_listening_in_csp_train,
            enforce_val_class_coverage=enforce_val_class_coverage,
            csp_class_ids=csp_class_ids,
        )
        # raw pre-augmentation data (no augmentation, no CSP)
        print(f"\n--- pre-augmentation ---")
        pre_aug_imagined_out = {
            "raw_imagined_train": out["raw_pre_imagined_train"],
            "raw_imagined_test": out["raw_pre_imagined_test"],
            "raw_imagined_val": out["raw_pre_imagined_val"],
            "y_train_dec": out["y_pre_imagined_train_dec"],
            "y_test_dec": out["y_pre_imagined_test_dec"],
            "y_val_dec": out["y_pre_imagined_val_dec"],
        }
        pre_aug_attempted_out = {
            "raw_attempted_train": out["raw_pre_attempted_train"],
            "raw_attempted_test": out["raw_pre_attempted_test"],
            "raw_attempted_val": out["raw_pre_attempted_val"],
            "y_train_dec": out["y_pre_attempted_train_dec"],
            "y_test_dec": out["y_pre_attempted_test_dec"],
            "y_val_dec": out["y_pre_attempted_val_dec"],
        }
        pre_aug_listening_out = {
            "raw_listening_train": out["raw_pre_listening_train"],
            "raw_listening_test": out["raw_pre_listening_test"],
            "raw_listening_val": out["raw_pre_listening_val"],
            "y_listening_train_dec": out["y_listening_train_dec"],
            "y_listening_test_dec": out["y_listening_test_dec"],
            "y_listening_val_dec": out["y_listening_val_dec"],
        }
        
        save_raw_splits_to_csv(pre_aug_imagined_out, raw_pre_aug_dir, subject_id, "imagined_speech", "imagined", common_classes)
        save_raw_splits_to_csv(pre_aug_attempted_out, raw_pre_aug_dir, subject_id, "attempted_speech", "attempted", common_classes)
        save_raw_splits_to_csv(pre_aug_listening_out, raw_pre_aug_dir, subject_id, "listening", "listening", common_classes, label_prefix="listening")

        # raw post-augmentation data (no CSP)
        post_aug_imagined_out = {
            "raw_imagined_train": out["raw_post_imagined_train"],
            "raw_imagined_test": out["raw_post_imagined_test"],
            "raw_imagined_val": out["raw_post_imagined_val"],
            "y_train_dec": out["y_train_dec"],
            "y_test_dec": out["y_test_dec"],
            "y_val_dec": out["y_val_dec"],
        }
        post_aug_attempted_out = {
            "raw_attempted_train": out["raw_post_attempted_train"],
            "raw_attempted_test": out["raw_post_attempted_test"],
            "raw_attempted_val": out["raw_post_attempted_val"],
            "y_train_dec": out["y_post_attempted_train_dec"],
            "y_test_dec": out["y_post_attempted_test_dec"],
            "y_val_dec": out["y_post_attempted_val_dec"],
        }
        post_aug_listening_out = {
            "raw_listening_train": out["raw_post_listening_train"],
            "raw_listening_test": out["raw_post_listening_test"],
            "raw_listening_val": out["raw_post_listening_val"],
            "y_listening_train_dec": out["y_listening_train_dec"],
            "y_listening_test_dec": out["y_listening_test_dec"],
            "y_listening_val_dec": out["y_listening_val_dec"],
        }
        
        save_raw_splits_to_csv(post_aug_imagined_out, raw_post_aug_dir, subject_id, "imagined_speech", "imagined", common_classes)
        save_raw_splits_to_csv(post_aug_attempted_out, raw_post_aug_dir, subject_id, "attempted_speech", "attempted", common_classes)
        save_raw_splits_to_csv(post_aug_listening_out, raw_post_aug_dir, subject_id, "listening", "listening", common_classes, label_prefix="listening")

    
        # CSP-transformed post-augmentation data
        print(f"\n--- Saving STAGE 3: CSP-transformed post-augmentation ---")
        csp_imagined_out = {
            "imagined_train": out["post_imagined_train"],
            "imagined_test": out["post_imagined_test"],
            "imagined_val": out["post_imagined_val"],
            "y_train_dec": out["y_train_dec"],
            "y_test_dec": out["y_test_dec"],
            "y_val_dec": out["y_val_dec"],
        }
        csp_attempted_out = {
            "attempted_train": out["post_attempted_train"],
            "attempted_test": out["post_attempted_test"],
            "attempted_val": out["post_attempted_val"],
            "y_train_dec": out["y_post_attempted_train_dec"],
            "y_test_dec": out["y_post_attempted_test_dec"],
            "y_val_dec": out["y_post_attempted_val_dec"],
        }
        csp_listening_out = {
            "listening_train": out["post_listening_train"],
            "listening_test": out["post_listening_test"],
            "listening_val": out["post_listening_val"],
            "y_listening_train_dec": out["y_listening_train_dec"],
            "y_listening_test_dec": out["y_listening_test_dec"],
            "y_listening_val_dec": out["y_listening_val_dec"],
        }
        
        save_splits_to_csv(csp_imagined_out, csp_post_aug_dir, subject_id, "imagined_speech", "imagined", common_classes)
        save_splits_to_csv(csp_attempted_out, csp_post_aug_dir, subject_id, "attempted_speech", "attempted", common_classes)
        save_splits_to_csv(csp_listening_out, csp_post_aug_dir, subject_id, "listening", "listening", common_classes, label_prefix="listening")
        
        # Save augmentation statistics
        save_subject_augmentation_stats_csv(output_dir, subject_id, out, common_classes)

        # Accumulate cross-subject summary rows.
        for split_name, y_key in [("train", "y_train_dec"), ("val", "y_val_dec"), ("test", "y_test_dec")]:
            for remapped_label, count in zip(*np.unique(out[y_key], return_counts=True)):
                imagined_summary_rows.append({
                    "subject_id": subject_id, "split": split_name,
                    "original_label": int(common_classes[int(remapped_label) - 1]),
                    "remapped_label": int(remapped_label), "count": int(count),
                })
        for split_name, y_key in [("train", "y_post_attempted_train_dec"), ("val", "y_post_attempted_val_dec"), ("test", "y_post_attempted_test_dec")]:
            for remapped_label, count in zip(*np.unique(out[y_key], return_counts=True)):
                attempted_summary_rows.append({
                    "subject_id": subject_id, "split": split_name,
                    "original_label": int(common_classes[int(remapped_label) - 1]),
                    "remapped_label": int(remapped_label), "count": int(count),
                })
        for split_name, y_key in [("train", "y_listening_train_dec"), ("val", "y_listening_val_dec"), ("test", "y_listening_test_dec")]:
            for remapped_label, count in zip(*np.unique(out[y_key], return_counts=True)):
                listening_summary_rows.append({
                    "subject_id": subject_id, "split": split_name,
                    "original_label": int(common_classes[int(remapped_label) - 1]),
                    "remapped_label": int(remapped_label), "count": int(count),
                })

        # Print data shapes for verification
        print(f"\nData shapes saved for subject {subject_id}:")
        for key in ["raw_pre_imagined_train", "raw_post_imagined_train", "imagined_train"]:
            if key in out:
                print(f"  {key}: {out[key].shape}")

    # Save cross-subject summary CSVs.
    pd.DataFrame(imagined_summary_rows).to_csv(
        os.path.join(output_dir, "summary_imagined_speech.csv"), index=False
    )
    pd.DataFrame(attempted_summary_rows).to_csv(
        os.path.join(output_dir, "summary_attempted_speech.csv"), index=False
    )
    pd.DataFrame(listening_summary_rows).to_csv(
        os.path.join(output_dir, "summary_listening.csv"), index=False
    )
    pd.DataFrame(csp_subject_rows).to_csv(
        os.path.join(output_dir, "csp_subject_class_mapping.csv"), index=False
    )
    print(f"\nSaved cross-subject summary CSVs and CSP metadata to {output_dir}/")


if __name__ == "__main__":
    main()
