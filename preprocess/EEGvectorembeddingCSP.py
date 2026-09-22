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
import random
from preprocess_utils import load_data, extract_epochs, to_decoded_labels, make_split_indices, save_splits_to_csv, save_csp_metadata

# Data Loading Setup
CONDITION_BASE  = {1: 100, 2: 200, 3: 400}
# CONDITION_NAMES = {1: 'Imagined speech', 2: 'Listening', 3: 'Attempted speech'}
# COND_TWIN = {1: (-0.1, 2.0), 2: (-0.1, 2.0), 3: (0.2, 2.3)}
# EVENT_SFREQ = 250
# SFREQ = 250


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


def run_vector_embedding_pipeline(
    x_imagined: np.ndarray,
    y_imagined: np.ndarray,
    x_attempted: np.ndarray,
    y_attempted: np.ndarray,
    x_listening: np.ndarray,
    y_listening: np.ndarray,
    subj_imagined: np.ndarray,
    subj_attempted: np.ndarray,
    subj_listening: np.ndarray,
    *,
    numcsp: int = 4,
    n_sess: int = 16,
    num_class: int = 13,
    label_num_class: int = 74,
    seed: int = 0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    augment_target_per_class: int = 20,
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
    subj_tr_im_pre, subj_val_im_pre, subj_ts_im_pre = subj_imagined[split_im.train], subj_imagined[split_im.val], subj_imagined[split_im.test]
    idx_tr_im_pre, idx_val_im_pre, idx_ts_im_pre = split_im.train, split_im.val, split_im.test

    x_tr_at_pre, y_tr_at_pre = x_attempted[split_at.train], y_at_dec[split_at.train]
    x_val_at_pre, y_val_at_pre = x_attempted[split_at.val], y_at_dec[split_at.val]
    x_ts_at_pre, y_ts_at_pre = x_attempted[split_at.test], y_at_dec[split_at.test]
    subj_tr_at_pre, subj_val_at_pre, subj_ts_at_pre = subj_attempted[split_at.train], subj_attempted[split_at.val], subj_attempted[split_at.test]
    idx_tr_at_pre, idx_val_at_pre, idx_ts_at_pre = split_at.train, split_at.val, split_at.test

    x_tr_li_pre, y_tr_li_pre = x_listening[split_li.train], y_li_dec[split_li.train]
    x_val_li_pre, y_val_li_pre = x_listening[split_li.val], y_li_dec[split_li.val]
    x_ts_li_pre, y_ts_li_pre = x_listening[split_li.test], y_li_dec[split_li.test]
    subj_tr_li_pre, subj_val_li_pre, subj_ts_li_pre = subj_listening[split_li.train], subj_listening[split_li.val], subj_listening[split_li.test]
    idx_tr_li_pre, idx_val_li_pre, idx_ts_li_pre = split_li.train, split_li.val, split_li.test

    # Independent Augmentation for Training Sets (Listening is NOT augmented or used in CSP training)
    rng = np.random.RandomState(seed)
    if use_augmentation:
        x_tr_im, y_tr_im, extra_im = _augment_split_(x_tr_im_pre, y_tr_im_pre, num_class=label_num_class, target_per_class=augment_target_per_class, noise_std=augment_noise_std, rng=rng, extra_arrays={"subj": subj_tr_im_pre, "idx": idx_tr_im_pre})
        subj_tr_im, idx_tr_im = extra_im["subj"], extra_im["idx"]
        x_tr_at, y_tr_at, extra_at = _augment_split_(x_tr_at_pre, y_tr_at_pre, num_class=label_num_class, target_per_class=augment_target_per_class, noise_std=augment_noise_std, rng=rng, extra_arrays={"subj": subj_tr_at_pre, "idx": idx_tr_at_pre})
        subj_tr_at, idx_tr_at = extra_at["subj"], extra_at["idx"]
    else:
        x_tr_im, y_tr_im = x_tr_im_pre, y_tr_im_pre
        subj_tr_im, idx_tr_im = subj_tr_im_pre, idx_tr_im_pre
        x_tr_at, y_tr_at = x_tr_at_pre, y_tr_at_pre
        subj_tr_at, idx_tr_at = subj_tr_at_pre, idx_tr_at_pre

    x_val_im, y_val_im = x_val_im_pre, y_val_im_pre
    x_ts_im, y_ts_im = x_ts_im_pre, y_ts_im_pre
    x_val_at, y_val_at = x_val_at_pre, y_val_at_pre
    x_ts_at, y_ts_at = x_ts_at_pre, y_ts_at_pre
    subj_val_im, subj_ts_im = subj_val_im_pre, subj_ts_im_pre
    subj_val_at, subj_ts_at = subj_val_at_pre, subj_ts_at_pre
    idx_val_im, idx_ts_im = idx_val_im_pre, idx_ts_im_pre
    idx_val_at, idx_ts_at = idx_val_at_pre, idx_ts_at_pre

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

        # Subject ids (per-sample, aligned with the pre and post augmentation arrays above)
        "subj_pre_imagined_train": subj_tr_im_pre, "subj_pre_imagined_val": subj_val_im_pre, "subj_pre_imagined_test": subj_ts_im_pre,
        "subj_pre_attempted_train": subj_tr_at_pre, "subj_pre_attempted_val": subj_val_at_pre, "subj_pre_attempted_test": subj_ts_at_pre,
        "subj_pre_listening_train": subj_tr_li_pre, "subj_pre_listening_val": subj_val_li_pre, "subj_pre_listening_test": subj_ts_li_pre,
        "subj_post_imagined_train": subj_tr_im, "subj_post_imagined_val": subj_val_im, "subj_post_imagined_test": subj_ts_im,
        "subj_post_attempted_train": subj_tr_at, "subj_post_attempted_val": subj_val_at, "subj_post_attempted_test": subj_ts_at,
        "subj_post_listening_train": subj_tr_li_pre, "subj_post_listening_val": subj_val_li_pre, "subj_post_listening_test": subj_ts_li_pre,

        # Global array index (for traceability back to the pre-split condition array)
        "idx_pre_imagined_train": idx_tr_im_pre, "idx_pre_imagined_val": idx_val_im_pre, "idx_pre_imagined_test": idx_ts_im_pre,
        "idx_pre_attempted_train": idx_tr_at_pre, "idx_pre_attempted_val": idx_val_at_pre, "idx_pre_attempted_test": idx_ts_at_pre,
        "idx_pre_listening_train": idx_tr_li_pre, "idx_pre_listening_val": idx_val_li_pre, "idx_pre_listening_test": idx_ts_li_pre,
        "idx_post_imagined_train": idx_tr_im, "idx_post_imagined_val": idx_val_im, "idx_post_imagined_test": idx_ts_im,
        "idx_post_attempted_train": idx_tr_at, "idx_post_attempted_val": idx_val_at, "idx_post_attempted_test": idx_ts_at,
        "idx_post_listening_train": idx_tr_li_pre, "idx_post_listening_val": idx_val_li_pre, "idx_post_listening_test": idx_ts_li_pre,

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


def prepare_vector_embedding_inputs(epochs_all: Dict[int, Dict[int, mne.Epochs]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    imagined_by_subject = epochs_all.get(1, {})
    listening_by_subject = epochs_all.get(2, {})
    attempted_by_subject = epochs_all.get(3, {})
    common_subjects = sorted(set(imagined_by_subject.keys()) & set(listening_by_subject.keys()) & set(attempted_by_subject.keys()))

    x_im_list, y_im_list = [], []
    x_li_list, y_li_list = [], []
    x_at_list, y_at_list = [], []

    subj_im_list, subj_li_list, subj_at_list = [], [], []

    for subject in common_subjects:
        ep_im, ep_li, ep_at = imagined_by_subject[subject], listening_by_subject[subject], attempted_by_subject[subject]
        x_im_list.append(ep_im.get_data())
        y_im_list.append(ep_im.events[:, 2] - CONDITION_BASE[1])
        x_li_list.append(ep_li.get_data())
        y_li_list.append(ep_li.events[:, 2] - CONDITION_BASE[2])
        x_at_list.append(ep_at.get_data())
        y_at_list.append(ep_at.events[:, 2] - CONDITION_BASE[3])
        subj_im_list.extend([subject] * ep_im.get_data().shape[0])
        subj_li_list.extend([subject] * ep_li.get_data().shape[0])
        subj_at_list.extend([subject] * ep_at.get_data().shape[0])

    x_imagined, y_imagined = np.concatenate(x_im_list, axis=0), np.concatenate(y_im_list, axis=0)
    x_listening, y_listening = np.concatenate(x_li_list, axis=0), np.concatenate(y_li_list, axis=0)
    x_attempted, y_attempted = np.concatenate(x_at_list, axis=0), np.concatenate(y_at_list, axis=0)

    subj_imagined = np.array(subj_im_list, dtype=np.int32)
    subj_listening = np.array(subj_li_list, dtype=np.int32)
    subj_attempted = np.array(subj_at_list, dtype=np.int32)

    # common_classes = np.intersect1d(np.unique(y_imagined), np.unique(y_attempted))
    # #np.unique(y_listening) - Include listening (to do if later included in training)
    # #li_mask = np.isin(y_listening, common_classes)
    
    # #im_mask = np.isin(y_imagined, common_classes)
    # #at_mask = np.isin(y_attempted, common_classes)

    # #x_imagined, y_imagined = x_imagined[im_mask], y_imagined[im_mask]
    # #x_listening, y_listening = x_listening[li_mask], y_listening[li_mask]
    # #x_attempted, y_attempted = x_attempted[at_mask], y_attempted[at_mask]

    # class_map = {int(cls): idx + 1 for idx, cls in enumerate(np.sort(common_classes))}
    # y_imagined = np.asarray([class_map[int(v)] for v in y_imagined], dtype=np.int32)
    # #y_listening = np.asarray([class_map[int(v)] for v in y_listening], dtype=np.int32)
    # y_attempted = np.asarray([class_map[int(v)] for v in y_attempted], dtype=np.int32)

    return x_imagined, y_imagined, x_attempted, y_attempted, x_listening, y_listening, subj_imagined, subj_attempted, subj_listening #common_classes.astype(np.int32), class_map



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[15, 16, 17, 18, 19])
    parser.add_argument("--eeg-data-dir", default="clean_data01-120Hz")
    parser.add_argument("--output-dir", default="eegdata1")
    parser.add_argument("--rawdata-output-dir", default="eegdata_rawsplits")
    args = parser.parse_args()

    raw_pre_aug_dir = os.path.join(args.rawdata_output_dir, "raw_pre_augmentation2")
    raw_post_aug_dir = os.path.join(args.rawdata_output_dir, "raw_post_augmentation_no_csp1")
    csp_post_aug_dir = os.path.join(args.output_dir, "csp_post_augmentation20_6_sets_newnoise")

    #CSP params and others
    numcsp = 4
    n_sess = 16
    num_class_csp = 13
    label_num_class = 74
    seed = 1
    val_ratio = 0.2
    test_ratio = 0.1

    #csp_class_seed = 3

    csp_sets = [
        np.array([17, 66, 21, 33, 46, 73, 61, 53, 36, 45, 70, 47, 30]),
        np.array([12, 67, 11, 26, 48, 41, 51, 23, 9, 35, 10, 59, 68]),
        np.array([20, 50, 42, 1, 32, 5, 37, 58, 57, 14, 25, 56, 69]),
        np.array([71, 64, 13, 16, 43, 3, 19, 65, 2, 62, 22, 8, 40]),
        np.array([18, 39, 28, 24, 7, 38, 31, 44, 29, 49, 52, 54, 34]),
        np.array([27, 74, 60, 15, 63, 55, 6, 72, 4, 46, 49, 5, 40]),
    ]
    metadata_by_set = {idx: [] for idx in range(len(csp_sets))}
    folds_by_set = {idx: [] for idx in range(len(csp_sets))}

    raw_all, markers_all, _ = load_data(args.subjects, data_dir=args.eeg_data_dir)
    epochs_all = extract_epochs(raw_all, markers_all)

    subjects_label = "-".join(str(s) for s in sorted(raw_all.keys()))
    x_im, y_im, x_at, y_at, x_li, y_li, subj_im, subj_at, subj_li = prepare_vector_embedding_inputs(epochs_all)

    common_classes = np.intersect1d(np.unique(y_im), np.unique(y_at))

        # rng_ref = np.random.RandomState(csp_class_seed)
        # csp_reference_original_classes = np.sort(
        #         rng_ref.choice(common_classes, size=num_class_csp, replace=False)
        #     )

        #rnd1classes_oldexp = np.array([1, 5, 10, 11, 13, 19, 29, 32, 35, 56, 64, 66, 74])

        # I do this to get the raw splits for saving first, using the first set, redundant, to be adjusted
    csp_reference_original_classes = csp_sets[0]

    out = run_vector_embedding_pipeline(
            x_imagined=x_im, y_imagined=y_im,
            x_attempted=x_at, y_attempted=y_at,
            x_listening=x_li, y_listening=y_li,
            subj_imagined=subj_im, subj_attempted=subj_at, subj_listening=subj_li,
            num_class=num_class_csp, label_num_class=len(common_classes), seed=seed, csp_class_ids=csp_reference_original_classes,
            debug_csp=True)

    # Save Raw Pre-Augmentation Splits
    save_splits_to_csv({"raw_imagined_train": out["raw_pre_imagined_train"], "raw_imagined_val": out["raw_pre_imagined_val"], "raw_imagined_test": out["raw_pre_imagined_test"], "y_imagined_train_dec": out["y_pre_imagined_train_dec"], "y_imagined_val_dec": out["y_pre_imagined_val_dec"], "y_imagined_test_dec": out["y_pre_imagined_test_dec"], "subj_imagined_train": out["subj_pre_imagined_train"], "subj_imagined_val": out["subj_pre_imagined_val"], "subj_imagined_test": out["subj_pre_imagined_test"], "idx_imagined_train": out["idx_pre_imagined_train"], "idx_imagined_val": out["idx_pre_imagined_val"], "idx_imagined_test": out["idx_pre_imagined_test"]}, raw_pre_aug_dir, "imagined_speech", "imagined", common_classes, raw=True)
    save_splits_to_csv({"raw_attempted_train": out["raw_pre_attempted_train"], "raw_attempted_val": out["raw_pre_attempted_val"], "raw_attempted_test": out["raw_pre_attempted_test"], "y_attempted_train_dec": out["y_pre_attempted_train_dec"], "y_attempted_val_dec": out["y_pre_attempted_val_dec"], "y_attempted_test_dec": out["y_pre_attempted_test_dec"], "subj_attempted_train": out["subj_pre_attempted_train"], "subj_attempted_val": out["subj_pre_attempted_val"], "subj_attempted_test": out["subj_pre_attempted_test"], "idx_attempted_train": out["idx_pre_attempted_train"], "idx_attempted_val": out["idx_pre_attempted_val"], "idx_attempted_test": out["idx_pre_attempted_test"]}, raw_pre_aug_dir, "attempted_speech", "attempted", original_labels=common_classes, raw=True)
    save_splits_to_csv({"raw_listening_train": out["raw_pre_listening_train"], "raw_listening_val": out["raw_pre_listening_val"], "raw_listening_test": out["raw_pre_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"], "subj_listening_train": out["subj_pre_listening_train"], "subj_listening_val": out["subj_pre_listening_val"], "subj_listening_test": out["subj_pre_listening_test"], "idx_listening_train": out["idx_pre_listening_train"], "idx_listening_val": out["idx_pre_listening_val"], "idx_listening_test": out["idx_pre_listening_test"]}, raw_pre_aug_dir, "listening", "listening", common_classes, raw=True)

    # Save Raw Post-Augmentation Splits
    save_splits_to_csv({"raw_imagined_train": out["raw_post_imagined_train"], "raw_imagined_val": out["raw_post_imagined_val"], "raw_imagined_test": out["raw_post_imagined_test"], "y_imagined_train_dec": out["y_train_dec"], "y_imagined_val_dec": out["y_val_dec"], "y_imagined_test_dec": out["y_test_dec"], "subj_imagined_train": out["subj_post_imagined_train"], "subj_imagined_val": out["subj_post_imagined_val"], "subj_imagined_test": out["subj_post_imagined_test"], "idx_imagined_train": out["idx_post_imagined_train"], "idx_imagined_val": out["idx_post_imagined_val"], "idx_imagined_test": out["idx_post_imagined_test"]}, raw_post_aug_dir, "imagined_speech", "imagined", common_classes, raw=True)
    save_splits_to_csv({"raw_attempted_train": out["raw_post_attempted_train"], "raw_attempted_val": out["raw_post_attempted_val"], "raw_attempted_test": out["raw_post_attempted_test"], "y_attempted_train_dec": out["y_post_attempted_train_dec"], "y_attempted_val_dec": out["y_post_attempted_val_dec"], "y_attempted_test_dec": out["y_post_attempted_test_dec"], "subj_attempted_train": out["subj_post_attempted_train"], "subj_attempted_val": out["subj_post_attempted_val"], "subj_attempted_test": out["subj_post_attempted_test"], "idx_attempted_train": out["idx_post_attempted_train"], "idx_attempted_val": out["idx_post_attempted_val"], "idx_attempted_test": out["idx_post_attempted_test"]}, raw_post_aug_dir, "attempted_speech","attempted", common_classes, raw=True)
    save_splits_to_csv({"raw_listening_train": out["raw_post_listening_train"], "raw_listening_val": out["raw_post_listening_val"], "raw_listening_test": out["raw_post_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"], "subj_listening_train": out["subj_post_listening_train"], "subj_listening_val": out["subj_post_listening_val"], "subj_listening_test": out["subj_post_listening_test"], "idx_listening_train": out["idx_post_listening_train"], "idx_listening_val": out["idx_post_listening_val"], "idx_listening_test": out["idx_post_listening_test"]}, raw_post_aug_dir, "listening", "listening", common_classes, raw=True)

    for set_idx, csp_reference_original_classes in enumerate(csp_sets):
            set_dir = os.path.join(csp_post_aug_dir, f"set{set_idx + 1}")
            print(f"CSP set {set_idx + 1}: {csp_reference_original_classes}")

            out = run_vector_embedding_pipeline(
                x_imagined=x_im, y_imagined=y_im,
                x_attempted=x_at, y_attempted=y_at,
                x_listening=x_li, y_listening=y_li,
                subj_imagined=subj_im, subj_attempted=subj_at, subj_listening=subj_li,
                num_class=num_class_csp, label_num_class=len(common_classes), seed=seed,
                csp_class_ids=csp_reference_original_classes, debug_csp=True)

            save_splits_to_csv({"imagined_train": out["post_imagined_train"], "imagined_val": out["post_imagined_val"], "imagined_test": out["post_imagined_test"], "y_imagined_train_dec": out["y_train_dec"], "y_imagined_val_dec": out["y_val_dec"], "y_imagined_test_dec": out["y_test_dec"], "subj_imagined_train": out["subj_post_imagined_train"], "subj_imagined_val": out["subj_post_imagined_val"], "subj_imagined_test": out["subj_post_imagined_test"], "idx_imagined_train": out["idx_post_imagined_train"], "idx_imagined_val": out["idx_post_imagined_val"], "idx_imagined_test": out["idx_post_imagined_test"]}, set_dir, "imagined_speech", "imagined", common_classes)
            save_splits_to_csv({"attempted_train": out["post_attempted_train"], "attempted_val": out["post_attempted_val"], "attempted_test": out["post_attempted_test"], "y_attempted_train_dec": out["y_post_attempted_train_dec"], "y_attempted_val_dec": out["y_post_attempted_val_dec"], "y_attempted_test_dec": out["y_post_attempted_test_dec"], "subj_attempted_train": out["subj_post_attempted_train"], "subj_attempted_val": out["subj_post_attempted_val"], "subj_attempted_test": out["subj_post_attempted_test"], "idx_attempted_train": out["idx_post_attempted_train"], "idx_attempted_val": out["idx_post_attempted_val"], "idx_attempted_test": out["idx_post_attempted_test"]}, set_dir, "attempted_speech", "attempted", common_classes)
            save_splits_to_csv({"listening_train": out["post_listening_train"], "listening_val": out["post_listening_val"], "listening_test": out["post_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"], "subj_listening_train": out["subj_post_listening_train"], "subj_listening_val": out["subj_post_listening_val"], "subj_listening_test": out["subj_post_listening_test"], "idx_listening_train": out["idx_post_listening_train"], "idx_listening_val": out["idx_post_listening_val"], "idx_listening_test": out["idx_post_listening_test"]}, set_dir, "listening", "listening", common_classes)

            csp_eigvals = out["csp_eigvals"]
            metadata_by_set[set_idx].append({
                "subjects": subjects_label,
                "n_channels": out["csp_w"].shape[0],
                "n_csp_filters": out["csp_w"].shape[1],
                "n_classes": len(common_classes),
                "classes_included": [int(c) for c in csp_reference_original_classes],
                "svm_accuracy": float(out["csp_accuracy"]),
                "mean_folds_accuracy": float(np.mean(out["csp_cv_fold_accuracy"])),
            })
            folds_by_set[set_idx].append({
                "subjects": subjects_label,
                "fold_accuracies": str(out["csp_cv_fold_accuracy"].tolist()),
                "mean_folds_accuracy": float(np.mean(out["csp_cv_fold_accuracy"])),
                "min_eigenvalue": float(np.min(out["csp_cv_eigvals"])),
                "max_eigenvalue": float(np.max(out["csp_cv_eigvals"])),
                "mean_eigenvalue": float(np.mean(out["csp_cv_eigvals"])),
            })

    for set_idx, csp_reference_original_classes in enumerate(csp_sets):
        set_dir = os.path.join(csp_post_aug_dir, f"set{set_idx + 1}")
        save_csp_metadata("csp_metadata", metadata_by_set[set_idx], set_dir)
        save_csp_metadata("csp_fold_metadata", folds_by_set[set_idx], set_dir)
        pd.DataFrame({
            "parameter": ["numcsp", "n_sess", "num_class", "augment_seed", "global_common_classes_count", "csp_reference_original_classes"],
            "value": [numcsp, n_sess, num_class_csp, seed, int(common_classes.size), str(csp_reference_original_classes.tolist())],
        }).to_csv(os.path.join(set_dir, "csp_metametadata.csv"), index=False)

if __name__ == "__main__":
    main()