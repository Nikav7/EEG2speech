from __future__ import annotations

import os
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
import random
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
from EEGvectorembeddingCSP import save_splits_to_csv
from preprocess_utils import load_split_data, proc_multicsp_train, svm_score, _segment_variance_log, apply_linear_derivation


# Data
# CONDITION_BASE  = {1: 100, 2: 200, 3: 400}
# CONDITION_NAMES = {1: 'Imagined speech', 2: 'Listening', 3: 'Attempted speech'}


def CSP_training(
    x_imagined: np.ndarray,
    y_imagined: np.ndarray,
    x_attempted: np.ndarray,
    y_attempted: np.ndarray,
    x_listening: np.ndarray,
    y_listening: np.ndarray,
    x_val_im: np.ndarray, y_val_im: np.ndarray,
    x_val_at: np.ndarray, y_val_at: np.ndarray,
    x_val_li: np.ndarray, y_val_li: np.ndarray,
    x_test_im: np.ndarray, y_test_im: np.ndarray,
    x_test_at: np.ndarray, y_test_at: np.ndarray,
    x_test_li: np.ndarray, y_test_li: np.ndarray,
    idx_imagined: np.ndarray, idx_attempted: np.ndarray, idx_listening: np.ndarray,
    idx_val_im: np.ndarray, idx_val_at: np.ndarray, idx_val_li: np.ndarray,
    idx_test_im: np.ndarray, idx_test_at: np.ndarray, idx_test_li: np.ndarray,
    *,
    
    numcsp: int = 4,
    n_sess: int = 16,
    num_class: int = 13,
    label_num_class: int = 74,
    seed: int = 0,
    debug_csp: bool = False,
    csp_class_ids: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:


   

    # # Independent Augmentation for Training Sets (Listening is NOT augmented or used in CSP training)
    # rng = np.random.RandomState(seed)
    # if use_augmentation:
    #     x_tr_im, y_tr_im = _augment_split_(x_imagined, y_imagined, num_class=label_num_class, target_per_class=augment_target_per_class, noise_std=augment_noise_std, rng=rng)
    #     #x_val_im, y_val_im = _augment_split_(x_val_im, y_val_im, num_class=label_num_class, target_per_class=2, noise_std=augment_noise_std, rng=rng)
    #     x_tr_at, y_tr_at = _augment_split_(x_attempted, y_attempted, num_class=label_num_class, target_per_class=augment_target_per_class, noise_std=augment_noise_std, rng=rng)
    # else:
    #     x_tr_im, y_tr_im = x_imagined, y_imagined
    #     x_tr_at, y_tr_at = x_attempted, y_attempted

    x_tr_im, y_tr_im = x_imagined, y_imagined
    x_tr_at, y_tr_at = x_attempted, y_attempted
    x_val_im, y_val_im = x_val_im, y_val_im
    x_ts_im, y_ts_im = x_test_im, y_test_im
    x_val_at, y_val_at = x_val_at, y_val_at
    x_ts_at, y_ts_at = x_test_at, y_test_at
    x_tr_li, y_tr_li = x_listening, y_listening
    x_val_li, y_val_li = x_val_li, y_val_li
    x_ts_li, y_ts_li = x_test_li, y_test_li
    idx_tr_im, idx_tr_at, idx_tr_li = idx_imagined, idx_attempted, idx_listening
    idx_ts_im, idx_ts_at, idx_ts_li = idx_test_im, idx_test_at, idx_test_li

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
        "y_listening_train_dec": y_tr_li, "y_listening_val_dec": y_val_li, "y_listening_test_dec": y_ts_li,

         # Global array index (for traceability back to the pre-split condition array)
        # "idx_pre_imagined_train": idx_tr_im_pre, "idx_pre_imagined_val": idx_val_im_pre, "idx_pre_imagined_test": idx_ts_im_pre,
        # "idx_pre_attempted_train": idx_tr_at_pre, "idx_pre_attempted_val": idx_val_at_pre, "idx_pre_attempted_test": idx_ts_at_pre,
        # "idx_pre_listening_train": idx_tr_li_pre, "idx_pre_listening_val": idx_val_li_pre, "idx_pre_listening_test": idx_ts_li_pre,
        "idx_post_imagined_train": idx_tr_im, "idx_post_imagined_val": idx_val_im, "idx_post_imagined_test": idx_ts_im,
        "idx_post_attempted_train": idx_tr_at, "idx_post_attempted_val": idx_val_at, "idx_post_attempted_test": idx_ts_at,
        "idx_post_listening_train": idx_tr_li, "idx_post_listening_val": idx_val_li, "idx_post_listening_test": idx_ts_li,

        
        # "y_pre_imagined_train_dec": y_tr_im_pre, "y_pre_imagined_val_dec": y_val_im_pre, "y_pre_imagined_test_dec": y_ts_im_pre,
        # "y_pre_attempted_train_dec": y_tr_at_pre, "y_pre_attempted_val_dec": y_val_at_pre, "y_pre_attempted_test_dec": y_ts_at_pre,

        # Raw Data
        # "raw_pre_imagined_train": x_tr_im_pre, "raw_pre_imagined_val": x_val_im_pre, "raw_pre_imagined_test": x_ts_im_pre,
        # "raw_pre_attempted_train": x_tr_at_pre, "raw_pre_attempted_val": x_val_at_pre, "raw_pre_attempted_test": x_ts_at_pre,
        # "raw_pre_listening_train": x_tr_li_pre, "raw_pre_listening_val": x_val_li_pre, "raw_pre_listening_test": x_ts_li_pre,

        # "raw_post_imagined_train": x_tr_im, "raw_post_imagined_val": x_val_im, "raw_post_imagined_test": x_ts_im,
        # "raw_post_attempted_train": x_tr_at, "raw_post_attempted_val": x_val_at, "raw_post_attempted_test": x_ts_at,
        # "raw_post_listening_train": x_tr_li_pre, "raw_post_listening_val": x_val_li_pre, "raw_post_listening_test": x_ts_li_pre,

        # CSP Filtered Features
        "post_imagined_train": _segment_variance_log(apply_linear_derivation(x_tr_im, w), n_sess),
        "post_imagined_val": _segment_variance_log(apply_linear_derivation(x_val_im, w), n_sess),
        "post_imagined_test": _segment_variance_log(apply_linear_derivation(x_ts_im, w), n_sess),
        
        "post_attempted_train": _segment_variance_log(apply_linear_derivation(x_tr_at, w), n_sess),
        "post_attempted_val": _segment_variance_log(apply_linear_derivation(x_val_at, w), n_sess),
        "post_attempted_test": _segment_variance_log(apply_linear_derivation(x_ts_at, w), n_sess),
        
        "post_listening_train": _segment_variance_log(apply_linear_derivation(x_tr_li, w), n_sess),
        "post_listening_val": _segment_variance_log(apply_linear_derivation(x_val_li, w), n_sess),
        "post_listening_test": _segment_variance_log(apply_linear_derivation(x_ts_li, w), n_sess),
    }


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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[15, 16, 17, 18, 19])
    parser.add_argument("--eeg-data-dir", default="clean_data01-120Hz")
    parser.add_argument("--output-dir", default="eegdata3")
    parser.add_argument("--rawdata-output-dir", default="eegdata3_rawsplits")
    args = parser.parse_args()

    raw_pre_aug_dir = os.path.join(args.rawdata_output_dir, "raw_pre_augmentation")
    raw_post_aug_dir = os.path.join(args.rawdata_output_dir, "raw_post_augmentation_no_csp")
    csp_post_aug_dir = os.path.join(args.output_dir, "csp_post_augmentation_cls1-13")

    #CSP params and others
    numcsp = 4
    n_sess = 16
    num_class_csp = 13
    label_num_class = 74
    seed = 1
    val_ratio = 0.2
    test_ratio = 0.1

    eeg_splits = "eegdata_rawsplits/raw_pre_augmentation_6_sets_subtog15"

    csp_class_seed = 3

    all_subject_metadata = []
    folds_metadata = []
    

    for subject_id in args.subjects:
        # raw_all, markers_all, _ = load_data([subject_id], data_dir=args.eeg_data_dir)
        # if not raw_all:
        #     continue
        
        x_tr_im, y_tr_im, idx_tr_im, x_val_im, y_val_im, idx_val_im, x_ts_im, y_ts_im, idx_ts_im = load_split_data(eeg_splits, subject_id, condition="imagined_speech", return_indices=True)
        x_tr_at, y_tr_at, idx_tr_at, x_val_at, y_val_at, idx_val_at, x_ts_at, y_ts_at, idx_ts_at = load_split_data(eeg_splits, subject_id, condition="attempted_speech", return_indices=True)
        x_tr_li, y_tr_li, idx_tr_li, x_val_li, y_val_li, idx_val_li, x_ts_li, y_ts_li, idx_ts_li = load_split_data(eeg_splits, subject_id, condition="listening", return_indices=True)
        

        common_classes = np.arange(1, label_num_class + 1)

        # rng_ref = np.random.RandomState(csp_class_seed)
        # csp_reference_original_classes = np.sort(
        #         rng_ref.choice(common_classes, size=num_class_csp, replace=False)
        #     )
        csp_reference_original_classes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        #csp_reference_original_classes = np.array([1, 5, 10, 11, 13, 19, 29, 32, 35, 56, 64, 66, 74]) #rnd1classes
        print(f"CSP reference original classes (seed={csp_class_seed}, randomly selected {num_class_csp}): {csp_reference_original_classes}")

        out = CSP_training(
            x_imagined=x_tr_im, y_imagined=y_tr_im,
            x_attempted=x_tr_at, y_attempted=y_tr_at,
            x_listening=x_tr_li, y_listening=y_tr_li,
            x_val_im=x_val_im, y_val_im=y_val_im,
            x_val_at=x_val_at, y_val_at=y_val_at,
            x_val_li=x_val_li, y_val_li=y_val_li,
            x_test_im=x_ts_im, y_test_im=y_ts_im,
            x_test_at=x_ts_at, y_test_at=y_ts_at,
            x_test_li=x_ts_li, y_test_li=y_ts_li,
            idx_imagined=idx_tr_im, idx_attempted=idx_tr_at, idx_listening=idx_tr_li,
            idx_val_im=idx_val_im, idx_val_at=idx_val_at, idx_val_li=idx_val_li,
            idx_test_im=idx_ts_im, idx_test_at=idx_ts_at, idx_test_li=idx_ts_li,
            num_class=num_class_csp, label_num_class=len(common_classes), seed=seed, csp_class_ids=csp_reference_original_classes,
            debug_csp=True)

        # Save Raw Pre-Augmentation Splits
        # args: output_dir: str, subject_id: int, condition_name: str, condition_prefix: str, original_labels: np.ndarray,
        # save_splits_to_csv({"raw_imagined_train": out["raw_pre_imagined_train"], "raw_imagined_val": out["raw_pre_imagined_val"], "raw_imagined_test": out["raw_pre_imagined_test"], "y_imagined_train_dec": out["y_pre_imagined_train_dec"], "y_imagined_val_dec": out["y_pre_imagined_val_dec"], "y_imagined_test_dec": out["y_pre_imagined_test_dec"]}, raw_pre_aug_dir, subject_id, "imagined_speech", "imagined", common_classes, raw=True)
        # save_splits_to_csv({"raw_attempted_train": out["raw_pre_attempted_train"], "raw_attempted_val": out["raw_pre_attempted_val"], "raw_attempted_test": out["raw_pre_attempted_test"], "y_attempted_train_dec": out["y_pre_attempted_train_dec"], "y_attempted_val_dec": out["y_pre_attempted_val_dec"], "y_attempted_test_dec": out["y_pre_attempted_test_dec"]}, raw_pre_aug_dir, subject_id, "attempted_speech", "attempted", original_labels=common_classes, raw=True)
        # save_splits_to_csv({"raw_listening_train": out["raw_pre_listening_train"], "raw_listening_val": out["raw_pre_listening_val"], "raw_listening_test": out["raw_pre_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"]}, raw_pre_aug_dir, subject_id, "listening", "listening", common_classes, raw=True)

        # # Save Raw Post-Augmentation Splits
        # save_splits_to_csv({"raw_imagined_train": out["raw_post_imagined_train"], "raw_imagined_val": out["raw_post_imagined_val"], "raw_imagined_test": out["raw_post_imagined_test"], "y_imagined_train_dec": out["y_train_dec"], "y_imagined_val_dec": out["y_val_dec"], "y_imagined_test_dec": out["y_test_dec"]}, raw_post_aug_dir, subject_id, "imagined_speech", "imagined", common_classes, raw=True)
        # save_splits_to_csv({"raw_attempted_train": out["raw_post_attempted_train"], "raw_attempted_val": out["raw_post_attempted_val"], "raw_attempted_test": out["raw_post_attempted_test"], "y_attempted_train_dec": out["y_post_attempted_train_dec"], "y_attempted_val_dec": out["y_post_attempted_val_dec"], "y_attempted_test_dec": out["y_post_attempted_test_dec"]}, raw_post_aug_dir, subject_id, "attempted_speech","attempted", common_classes, raw=True)
        # save_splits_to_csv({"raw_listening_train": out["raw_post_listening_train"], "raw_listening_val": out["raw_post_listening_val"], "raw_listening_test": out["raw_post_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"]}, raw_post_aug_dir, subject_id, "listening", "listening", common_classes, raw=True)

        # # Save CSP Features
        save_splits_to_csv({"imagined_train": out["post_imagined_train"], "imagined_val": out["post_imagined_val"], "imagined_test": out["post_imagined_test"], "y_imagined_train_dec": out["y_train_dec"], "y_imagined_val_dec": out["y_val_dec"], "y_imagined_test_dec": out["y_test_dec"], "subj_imagined_train": np.full(len(out["post_imagined_train"]), subject_id), "subj_imagined_val": np.full(len(out["post_imagined_val"]), subject_id), "subj_imagined_test": np.full(len(out["post_imagined_test"]), subject_id), "idx_imagined_train": out["idx_post_imagined_train"], "idx_imagined_val": out["idx_post_imagined_val"], "idx_imagined_test": out["idx_post_imagined_test"]}, csp_post_aug_dir, "imagined_speech", "imagined", common_classes)
        save_splits_to_csv({"attempted_train": out["post_attempted_train"], "attempted_val": out["post_attempted_val"], "attempted_test": out["post_attempted_test"], "y_attempted_train_dec": out["y_post_attempted_train_dec"], "y_attempted_val_dec": out["y_post_attempted_val_dec"], "y_attempted_test_dec": out["y_post_attempted_test_dec"], "subj_attempted_train": np.full(len(out["post_attempted_train"]), subject_id), "subj_attempted_val": np.full(len(out["post_attempted_val"]), subject_id), "subj_attempted_test": np.full(len(out["post_attempted_test"]), subject_id), "idx_attempted_train": out["idx_post_attempted_train"], "idx_attempted_val": out["idx_post_attempted_val"], "idx_attempted_test": out["idx_post_attempted_test"]}, csp_post_aug_dir, "attempted_speech", "attempted", common_classes)
        save_splits_to_csv({"listening_train": out["post_listening_train"], "listening_val": out["post_listening_val"], "listening_test": out["post_listening_test"], "y_listening_train_dec": out["y_listening_train_dec"], "y_listening_val_dec": out["y_listening_val_dec"], "y_listening_test_dec": out["y_listening_test_dec"], "subj_listening_train": np.full(len(out["post_listening_train"]), subject_id), "subj_listening_val": np.full(len(out["post_listening_val"]), subject_id), "subj_listening_test": np.full(len(out["post_listening_test"]), subject_id), "idx_listening_train": out["idx_post_listening_train"], "idx_listening_val": out["idx_post_listening_val"], "idx_listening_test": out["idx_post_listening_test"]}, csp_post_aug_dir, "listening", "listening", common_classes)

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