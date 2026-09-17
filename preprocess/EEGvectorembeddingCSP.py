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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Data Loading Setup
CONDITION_BASE  = {1: 100, 2: 200, 3: 400}
CONDITION_NAMES = {1: 'Imagined speech', 2: 'Listening', 3: 'Attempted speech'}
COND_TWIN = {1: (0.0, 2.0), 2: (0.0, 2.0), 3: (0.3, 2.3)}
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


def _augment_split_(
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
    label_num_class: int = 74,
    seed: int = 0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    augment_target_per_class: int = 9,
    augment_noise_std: float = 1e-4,
    use_augmentation: bool = True,
    enforce_val_class_coverage: bool = True,
    debug_csp: bool = False,
    csp_class_ids: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:

    y_im_dec = to_decoded_labels(y_imagined)
    y_at_dec = to_decoded_labels(y_attempted)
    y_li_dec = to_decoded_labels(y_listening)

# Class-coverage splits for Imagined and Attempted, split Each Condition Independently
    split_im = make_split_indices(
        y_im_dec, seed=seed, 
        val_ratio=val_ratio, test_ratio=test_ratio, 
        enforce_val_class_coverage=enforce_val_class_coverage
    )
    split_at = make_split_indices(
        y_at_dec, seed=seed, 
        val_ratio=val_ratio, test_ratio=test_ratio, 
        enforce_val_class_coverage=enforce_val_class_coverage
    )


    split_li = make_split_indices(
            y_li_dec, seed=seed, 
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
        x_tr_im, y_tr_im = _augment_split_(x_tr_im_pre, y_tr_im_pre, num_class=label_num_class, target_per_class=augment_target_per_class, noise_std=augment_noise_std, rng=rng)
        x_val_im, y_val_im = _augment_split_(x_val_im_pre, y_val_im_pre, num_class=label_num_class, target_per_class=2, noise_std=augment_noise_std, rng=rng)
        x_tr_at, y_tr_at = _augment_split_(x_tr_at_pre, y_tr_at_pre, num_class=label_num_class, target_per_class=augment_target_per_class, noise_std=augment_noise_std, rng=rng)
    else:
        x_tr_im, y_tr_im = x_tr_im_pre, y_tr_im_pre
        x_tr_at, y_tr_at = x_tr_at_pre, y_tr_at_pre

    #x_val_im, y_val_im = x_val_im_pre, y_val_im_pre
    x_ts_im, y_ts_im = x_ts_im_pre, y_ts_im_pre
    x_val_at, y_val_at = x_val_at_pre, y_val_at_pre
    x_ts_at, y_ts_at = x_ts_at_pre, y_ts_at_pre

    # Fit CSP on Combined Imagined + Attempted Training Data ONLY
    x_tr_both = np.concatenate([x_tr_im, x_tr_at], axis=0)
    y_tr_both = np.concatenate([y_tr_im, y_tr_at], axis=0)
    x_val_both = np.concatenate([x_val_im, x_val_at], axis=0)
    y_val_both = np.concatenate([y_val_im, y_val_at], axis=0)

    if csp_class_ids is None: #take the first 13 classes
        csp_class_ids = np.arange(1, num_class + 1, dtype=np.int32)


    if debug_csp:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_fold_eigvals = []
        cv_fold_accuracy = []

        for train_idx, val_idx in cv.split(x_tr_both, y_tr_both):
            x_fold_train = x_tr_both[train_idx]
            y_fold_train = y_tr_both[train_idx]
            x_fold_val = x_tr_both[val_idx]
            y_fold_val = y_tr_both[val_idx]

            y_fold_one_hot = np.zeros((len(csp_class_ids), x_fold_train.shape[0]), dtype=np.int32)
            for i, cls in enumerate(csp_class_ids):
                y_fold_one_hot[i, y_fold_train == cls] = 1

            w_fold, fold_eigvals = proc_multicsp_train(
                x_fold_train, y_fold_one_hot, n_comps=numcsp,
                centered=True, method="all", way="one-vs-all",
            )
            cv_fold_eigvals.append(fold_eigvals)
            cv_fold_accuracy.append(svm_score(
                _segment_variance_log(apply_linear_derivation(x_fold_train, w_fold), n_sess),
                y_fold_train,
                _segment_variance_log(apply_linear_derivation(x_fold_val, w_fold), n_sess),
                y_fold_val,
            ))

    y_tr_one_hot = np.zeros((len(csp_class_ids), x_tr_both.shape[0]), dtype=np.int32)
    for i, cls in enumerate(csp_class_ids):
        y_tr_one_hot[i, y_tr_both == cls] = 1

    w, la = proc_multicsp_train(
        x_tr_both, y_tr_one_hot, n_comps=numcsp,
        centered=True, method="all", way="one-vs-all",
    )

    train_accuracies = svm_score(
                    _segment_variance_log(apply_linear_derivation(x_tr_both, w), n_sess),
                    y_tr_both,
                    _segment_variance_log(apply_linear_derivation(x_val_both, w), n_sess),
                    y_val_both,)

    # Feature Extraction (Applying fitted W on Imagined, Attempted, and Listening)
    return {
        "csp_w": w, "csp_eigvals": la, "csp_accuracy": train_accuracies,
        **({
            "csp_cv_eigvals": np.stack(cv_fold_eigvals),
            "csp_cv_fold_accuracy": np.asarray(cv_fold_accuracy),
            "csp_cv_mean_accuracy": float(np.mean(cv_fold_accuracy)),
        } if debug_csp else {}),
        
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

    common_classes = np.intersect1d(np.unique(y_imagined), np.unique(y_attempted))
    #np.unique(y_listening) - Include listening (to do if later included in training)
    #li_mask = np.isin(y_listening, common_classes)
    
    im_mask = np.isin(y_imagined, common_classes)
    at_mask = np.isin(y_attempted, common_classes)

    x_imagined, y_imagined = x_imagined[im_mask], y_imagined[im_mask]
    #x_listening, y_listening = x_listening[li_mask], y_listening[li_mask]
    x_attempted, y_attempted = x_attempted[at_mask], y_attempted[at_mask]

    class_map = {int(cls): idx + 1 for idx, cls in enumerate(np.sort(common_classes))}
    y_imagined = np.asarray([class_map[int(v)] for v in y_imagined], dtype=np.int32)
    #y_listening = np.asarray([class_map[int(v)] for v in y_listening], dtype=np.int32)
    y_attempted = np.asarray([class_map[int(v)] for v in y_attempted], dtype=np.int32)

    return x_imagined, y_imagined, x_attempted, y_attempted, x_listening, y_listening, common_classes.astype(np.int32), class_map



def save_splits_to_csv(out: Dict[str, np.ndarray], output_dir: str, subject_id: int, condition_name: str, condition_prefix: str, original_labels: np.ndarray, label_prefix: str | None = None, raw: bool = False) -> None:
    subj_dir = os.path.join(output_dir, f"subj{subject_id}")
    cond_dir = os.path.join(subj_dir, condition_name)
    label_prefix = condition_prefix if label_prefix is None else label_prefix
    key_prefix = "raw_" if raw else ""
    split_map = {
        "train": (f"{key_prefix}{condition_prefix}_train", f"y_{label_prefix}_train_dec"),
        "val": (f"{key_prefix}{condition_prefix}_val", f"y_{label_prefix}_val_dec"),
        "test": (f"{key_prefix}{condition_prefix}_test", f"y_{label_prefix}_test_dec"),
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


def save_csp_metadata(csv_name: str, metadata: List[Dict[str, Any]], output_dir: str) -> None:
    """Save aggregated metadata to a named CSV in the main output folder."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{csv_name}.csv")
    
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[16, 17, 18, 19])
    parser.add_argument("--eeg-data-dir", default="clean_data01-120Hz")
    parser.add_argument("--output-dir", default="eegdata2")
    parser.add_argument("--rawdata-output-dir", default="eegdata2_rawsplits")
    args = parser.parse_args()

    raw_pre_aug_dir = os.path.join(args.rawdata_output_dir, "raw_pre_augmentation")
    raw_post_aug_dir = os.path.join(args.rawdata_output_dir, "raw_post_augmentation_no_csp")
    csp_post_aug_dir = os.path.join(args.output_dir, "csp_post_augmentation")

    #CSP params and others
    numcsp = 4
    n_sess = 16
    num_class_csp = 13
    label_num_class = 74
    seed = 1
    val_ratio = 0.2
    test_ratio = 0.1

    csp_class_seed = 3

    all_subject_metadata = []
    folds_metadata = []

    for subject_id in args.subjects:
        raw_all, markers_all, _ = load_data([subject_id], data_dir=args.eeg_data_dir)
        if not raw_all:
            continue
        epochs_all = extract_epochs(raw_all, markers_all)

        x_im, y_im, x_at, y_at, x_li, y_li, common_classes, class_map = prepare_vector_embedding_inputs(epochs_all)

        rng_ref = np.random.RandomState(csp_class_seed)
        csp_reference_original_classes = np.sort(
                rng_ref.choice(common_classes, size=num_class_csp, replace=False)
            )

        #csp_reference_original_classes = np.array([1, 5, 10, 11, 13, 19, 29, 32, 35, 56, 64, 66, 74]) #rnd1classes
        print(f"CSP reference original classes (seed={csp_class_seed}, randomly selected {num_class_csp}): {csp_reference_original_classes}")

        out = run_vector_embedding_pipeline(
            x_imagined=x_im, y_imagined=y_im,
            x_attempted=x_at, y_attempted=y_at,
            x_listening=x_li, y_listening=y_li,
            num_class=num_class_csp, label_num_class=len(common_classes), seed=seed, csp_class_ids=csp_reference_original_classes,
            debug_csp=True)

        # Save Raw Pre-Augmentation Splits
        # args: output_dir: str, subject_id: int, condition_name: str, condition_prefix: str, original_labels: np.ndarray,
        save_splits_to_csv({"raw_imagined_train": out["raw_pre_imagined_train"], "raw_imagined_val": out["raw_pre_imagined_val"], "raw_imagined_test": out["raw_pre_imagined_test"], "y_imagined_train_dec": out["y_pre_imagined_train_dec"], "y_imagined_val_dec": out["y_pre_imagined_val_dec"], "y_imagined_test_dec": out["y_pre_imagined_test_dec"]}, raw_pre_aug_dir, subject_id, "imagined_speech", "imagined", common_classes, raw=True)
        save_splits_to_csv({"raw_attempted_train": out["raw_pre_attempted_train"], "raw_attempted_val": out["raw_pre_attempted_val"], "raw_attempted_test": out["raw_pre_attempted_test"], "y_attempted_train_dec": out["y_pre_attempted_train_dec"], "y_attempted_val_dec": out["y_pre_attempted_val_dec"], "y_attempted_test_dec": out["y_pre_attempted_test_dec"]}, raw_pre_aug_dir, subject_id, "attempted_speech", "attempted", original_labels=common_classes, raw=True)
        save_splits_to_csv({"raw_listening_train": out["raw_pre_listening_train"], "raw_listening_val": out["raw_pre_listening_val"], "raw_listening_test": out["raw_pre_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"]}, raw_pre_aug_dir, subject_id, "listening", "listening", common_classes, raw=True)

        # Save Raw Post-Augmentation Splits
        save_splits_to_csv({"raw_imagined_train": out["raw_post_imagined_train"], "raw_imagined_val": out["raw_post_imagined_val"], "raw_imagined_test": out["raw_post_imagined_test"], "y_imagined_train_dec": out["y_train_dec"], "y_imagined_val_dec": out["y_val_dec"], "y_imagined_test_dec": out["y_test_dec"]}, raw_post_aug_dir, subject_id, "imagined_speech", "imagined", common_classes, raw=True)
        save_splits_to_csv({"raw_attempted_train": out["raw_post_attempted_train"], "raw_attempted_val": out["raw_post_attempted_val"], "raw_attempted_test": out["raw_post_attempted_test"], "y_attempted_train_dec": out["y_post_attempted_train_dec"], "y_attempted_val_dec": out["y_post_attempted_val_dec"], "y_attempted_test_dec": out["y_post_attempted_test_dec"]}, raw_post_aug_dir, subject_id, "attempted_speech","attempted", common_classes, raw=True)
        save_splits_to_csv({"raw_listening_train": out["raw_post_listening_train"], "raw_listening_val": out["raw_post_listening_val"], "raw_listening_test": out["raw_post_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"]}, raw_post_aug_dir, subject_id, "listening", "listening", common_classes, raw=True)

        # Save CSP Features
        save_splits_to_csv({"imagined_train": out["post_imagined_train"], "imagined_val": out["post_imagined_val"], "imagined_test": out["post_imagined_test"], "y_imagined_train_dec": out["y_train_dec"], "y_imagined_val_dec": out["y_val_dec"], "y_imagined_test_dec": out["y_test_dec"]}, csp_post_aug_dir, subject_id, "imagined_speech", "imagined", common_classes, raw = False)
        save_splits_to_csv({"attempted_train": out["post_attempted_train"], "attempted_val": out["post_attempted_val"], "attempted_test": out["post_attempted_test"], "y_attempted_train_dec": out["y_post_attempted_train_dec"], "y_attempted_val_dec": out["y_post_attempted_val_dec"], "y_attempted_test_dec": out["y_post_attempted_test_dec"]}, csp_post_aug_dir, subject_id, "attempted_speech", "attempted", common_classes, raw=False)
        save_splits_to_csv({"listening_train": out["post_listening_train"], "listening_val": out["post_listening_val"], "listening_test": out["post_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"]}, csp_post_aug_dir, subject_id, "listening", "listening", common_classes, raw=False)

        # Save CSP Metadata
        csp_w = out["csp_w"]
        csp_eigvals = out["csp_eigvals"]
        svm_accuracy = out["csp_accuracy"]
        
        all_subject_metadata.append({
            "subject_id": subject_id,
            "n_channels": csp_w.shape[0],
            "n_csp_filters": csp_w.shape[1],
            "n_classes": len(common_classes),
            "classes_included": [int(c) for c in csp_reference_original_classes],
            "n_train_imagined": out["post_imagined_train"].shape[0],
            "n_train_attempted": out["post_attempted_train"].shape[0],
            "n_listening_total": out["post_listening_train"].shape[0] + out["post_listening_val"].shape[0] + out["post_listening_test"].shape[0],
            "min_eigenvalue": float(np.min(csp_eigvals)),
            "max_eigenvalue": float(np.max(csp_eigvals)),
            "mean_eigenvalue": float(np.mean(csp_eigvals)),
            "svm_accuracy": float(svm_accuracy),
            "mean_folds_accuracy": float(np.mean(out["csp_cv_fold_accuracy"])),

            })
        folds_metadata.append({
            "subject_id": subject_id,
            "fold_accuracies": str(out["csp_cv_fold_accuracy"].tolist()),
            "mean_folds_accuracy": float(np.mean(out["csp_cv_fold_accuracy"])),
            "min_eigenvalue": float(np.min(out["csp_cv_eigvals"])),
            "max_eigenvalue": float(np.max(out["csp_cv_eigvals"])),
            "mean_eigenvalue": float(np.mean(out["csp_cv_eigvals"]))
        })
    # CSP aggregated metadata in the root output folder
    csp_metameta_params = {
            "parameter": ["numcsp", "n_sess", "num_class", "csp_class_seed", "augment_seed",
                           "global_common_classes_count", "csp_reference_original_classes"],
            "value": [numcsp, n_sess, num_class_csp, csp_class_seed, seed,
                      int(common_classes.size), str(csp_reference_original_classes.tolist())],
        }
    pd.DataFrame(csp_metameta_params).to_csv(os.path.join(csp_post_aug_dir, "csp_metametadata.csv"), index=False)

    save_csp_metadata("csp_metadata", all_subject_metadata, csp_post_aug_dir)
    save_csp_metadata("csp_fold_metadata", folds_metadata, csp_post_aug_dir)

if __name__ == "__main__":
    main()