import os
import sys
import numpy as np
import pandas as pd
import mne

# Parameters
subjects = [19]
data_dir = "clean_data01-120Hz"
numcsp = 1
n_sess = 32
seed = 18
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


def one_hot_from_labels(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    y = np.zeros((len(classes), labels.shape[0]), dtype=np.float64)
    class_to_row = {c: i for i, c in enumerate(classes)}
    for idx, lbl in enumerate(labels):
        y[class_to_row[int(lbl)], idx] = 1.0
    return y


def decode_labels_from_event_codes(event_codes: np.ndarray, task: str) -> np.ndarray:
    labels = np.zeros(event_codes.shape[0], dtype=np.int32)

    if task == "listening":
        labels = event_codes.astype(int) - 200 + 1
    elif task == "imaginedspeech":
        codes = event_codes.astype(int)
        mask_100 = (codes >= 100) & (codes < 200)
        mask_300 = (codes >= 300) & (codes < 400)
        labels[mask_100] = codes[mask_100] - 100 + 1
        labels[mask_300] = codes[mask_300] - 300 + 1

        if np.any(labels == 0):
            raise ValueError("Imaginedspeech task received event codes outside 100/300 ranges.")
    else:
        raise ValueError("task must be 'listening' or 'imaginedspeech'")

    return labels


def decode_attempted_labels_from_event50(markers: np.ndarray, onset_samples: np.ndarray) -> np.ndarray:
    marker_samples = markers[:, 0].astype(np.int64)
    marker_codes = markers[:, 2].astype(np.int32)
    labels = np.full(onset_samples.shape[0], -1, dtype=np.int32)

    for i, onset in enumerate(onset_samples.astype(np.int64)):
        start = np.searchsorted(marker_samples, onset + 1, side="left")
        next_codes = marker_codes[start:]
        valid = np.where((next_codes >= 300) & (next_codes < 400))[0]
        if valid.size == 0:
            continue
        labels[i] = next_codes[valid[0]] - 300 + 1

    return labels


def split_seen_unseen_by_class(x: np.ndarray, y: np.ndarray, unseen_ids: np.ndarray):
    unseen_mask = np.isin(y, unseen_ids)
    seen_mask = ~unseen_mask
    return x[seen_mask], y[seen_mask], x[unseen_mask], y[unseen_mask]


def split_train_val_by_class(labels: np.ndarray, seed_value: int):
    rng = np.random.default_rng(seed_value)
    train_idx = []
    val_idx = []
    test_idx = []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]

        perm = rng.permutation(idx.size)
        # 15% validation, 15% test, 70% training (ensures balanced split per class)
        n_val = max(1, int(np.ceil(idx.size * 0.15)))
        n_test = max(1, int(np.ceil(idx.size * 0.15)))
        n_train = idx.size - n_val - n_test

        if n_train > 0:
            train_idx.append(idx[perm[:n_train]])
        if n_val > 0:
            val_idx.append(idx[perm[n_train:n_train + n_val]])
        if n_test > 0:
            test_idx.append(idx[perm[n_train + n_val:]])

    if train_idx:
        train_idx = np.concatenate(train_idx).astype(np.int32)
    else:
        train_idx = np.array([], dtype=np.int32)

    if val_idx:
        val_idx = np.concatenate(val_idx).astype(np.int32)
    else:
        val_idx = np.array([], dtype=np.int32)

    if test_idx:
        test_idx = np.concatenate(test_idx).astype(np.int32)
    else:
        test_idx = np.array([], dtype=np.int32)

    return train_idx, val_idx, test_idx


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


def segmented_log_variance(projected: np.ndarray, n_segments: int) -> np.ndarray:
    n_trials, n_components, n_times = projected.shape
    n_segments = max(1, min(n_segments, n_times))
    time_splits = np.array_split(np.arange(n_times), n_segments)
    feats = np.zeros((n_trials, n_components * n_segments), dtype=np.float64)

    for seg_i, time_idx in enumerate(time_splits):
        seg_var = np.var(projected[:, :, time_idx], axis=2)
        feats[:, seg_i * n_components:(seg_i + 1) * n_components] = np.log(np.maximum(seg_var, 1e-12))

    return feats


def transform_to_logvar_features(x: np.ndarray, w: np.ndarray, n_segments: int) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0, w.shape[1] * n_segments), dtype=np.float64)

    projected = apply_linear_derivation(x, w)
    return segmented_log_variance(projected, n_segments)


def transform_to_csp_fft_features(x: np.ndarray, w: np.ndarray, target_time_bins: int = 85) -> np.ndarray:
    """Apply CSP and FFT-resample time axis to target bins; output shape [N, K, T]."""
    if x.shape[0] == 0:
        return np.zeros((0, w.shape[1], target_time_bins), dtype=np.float64)

    if target_time_bins <= 0:
        raise ValueError("target_time_bins must be a positive integer.")

    projected = apply_linear_derivation(x, w)  # [n_trials, n_components, n_times]

    if projected.shape[2] == target_time_bins:
        return projected

    from scipy.signal import resample

    return resample(projected, num=target_time_bins, axis=2)


def reshape_trial_to_matrix(feature_row: np.ndarray, n_segments: int) -> np.ndarray:
    """Reshape one flattened feature row to NeuroTalk layout: (n_components, n_segments)."""
    n_components = feature_row.shape[0] // n_segments
    return feature_row.reshape(n_segments, n_components).T


def export_trial_csvs(
    features_train: np.ndarray,
    y_train: np.ndarray,
    features_val: np.ndarray,
    y_val: np.ndarray,
    features_unseen: np.ndarray,
    y_unseen: np.ndarray,
    subject: int,
    output_dir: str = "eegdata",
):
    """Export one headerless CSV per trial under train/val/unseentest folders.

    Expected feature shape per split: [n_trials, n_components, n_time_bins].
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name, features_split, y_split in [
        ("train", features_train, y_train),
        ("val", features_val, y_val),
        ("unseentest", features_unseen, y_unseen),
    ]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        if features_split.shape[0] == 0:
            continue

        for trial_idx, (feature_trial, class_id) in enumerate(zip(features_split, y_split), start=1):
            if feature_trial.ndim == 1:
                # Backward compatibility if flat log-variance features are passed.
                trial_matrix = reshape_trial_to_matrix(feature_trial, n_sess)
            else:
                trial_matrix = feature_trial
            out_csv = os.path.join(
                split_dir,
                f"subj{subject}_class_{int(class_id):03d}_ts-{trial_idx:04d}.csv",
            )
            pd.DataFrame(trial_matrix).to_csv(out_csv, index=False, header=False)


def main():

    eeg_data_all = {}
    markers_all = {}
    raw_all = {}

    for subject in subjects:
        eeg_file = os.path.join(data_dir, f"clean_eeg_subj{subject}s_band30120.npy")
        if not os.path.exists(eeg_file):
            print(f"Warning: EEG file not found for subject {subject}")
            continue
        eeg_data_all[subject] = np.load(eeg_file)

        events_file = os.path.join(data_dir, f"events_subj{subject}.npy")
        if not os.path.exists(events_file):
            print(f"Warning: Events file not found for subject {subject}")
            continue
        markers_all[subject] = np.load(events_file)[:-1]

        ch_names_file = os.path.join(data_dir, "channel_names.csv")
        if not os.path.exists(ch_names_file):
            print("Warning: Channel names file not found")
            continue

        ch_names_df = pd.read_csv(ch_names_file)
        eog_channels = ["EOG1", "EOG2", "EOG3"]
        ch_names_all = ch_names_df["Channel"].tolist()
        eeg_indices = [idx for idx, ch in enumerate(ch_names_all) if ch not in eog_channels]
        ch_names = [ch_names_all[idx] for idx in eeg_indices]
        eeg_data = eeg_data_all[subject][eeg_indices, :]

        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=["eeg"] * len(ch_names))
        raw = mne.io.RawArray(eeg_data, info)
        raw.set_montage("standard_1020")
        raw_all[subject] = raw
        print(f"Loaded subject {subject}: {len(ch_names)} EEG channels")

    for subject in subjects:
        if subject not in raw_all or subject not in markers_all:
            continue

        markers = markers_all[subject]
        raw = raw_all[subject]

        event_ids_100 = [code for code in np.unique(markers[:, 2]) if 100 <= code < 200]
        event_ids_300 = [code for code in np.unique(markers[:, 2]) if 300 <= code < 400]
        imagined_event_ids = event_ids_100 + event_ids_300

        if len(imagined_event_ids) == 0:
            print(f"Subject {subject}: no imagined speech events (100/300)")
            continue

        epochs_2s = mne.Epochs(
            raw,
            markers,
            event_id={"event" + str(e): int(e) for e in imagined_event_ids},
            tmin=0.0,
            tmax=2.0,
            picks="eeg",
            baseline=None,
            preload=True,
            reject=None,
            flat=None,
        )

        epochs_imagined = epochs_2s

        x_imagined = epochs_imagined.get_data(copy=True)
        y_imagined = decode_labels_from_event_codes(epochs_imagined.events[:, 2], task="imaginedspeech")

        # --- attempted speech: epochs at event_id=50 ---
        # Labels come from the 300-series event that follows each event-50 onset.
        unique_codes = np.unique(markers[:, 2])
        if 50 in unique_codes:
            epochs_attempted = mne.Epochs(
                raw,
                markers,
                event_id={"event50": 50},
                tmin=0.0,
                tmax=2.0,
                picks="eeg",
                baseline=None,
                preload=True,
                reject=None,
                flat=None,
            )
            x_attempted = epochs_attempted.get_data(copy=True)
            y_attempted = decode_attempted_labels_from_event50(markers, epochs_attempted.events[:, 0])
         
        print(
            f"Subject {subject}: loaded {x_imagined.shape[0]} imagined and {x_attempted.shape[0]} attempted trials"
        )

    
        if y_imagined.shape[0] == 0 and y_attempted.shape[0] == 0:
            print(f"Subject {subject}: no seen samples in imagined or attempted after metadata filtering")
            continue


        x_csp_train = np.concatenate([x_imagined, x_attempted], axis=0)
        y_csp_train = np.concatenate([y_imagined, y_attempted], axis=0)

        if x_csp_train.shape[0] == 0:
            print(f"Subject {subject}: no CSP training samples across imagined+attempted")
            continue

        csp_classes = np.array(sorted(np.unique(y_csp_train)), dtype=np.int32)
        y_train_one_hot = one_hot_from_labels(y_csp_train, csp_classes)
        csp_w_tr, csp_eigvals = proc_multicsp_train(
            x_csp_train,
            y_train_one_hot,
            n_comps=numcsp,
            centered=True,
            method="all",
            way="one-vs-all",
        )

        features_imagined_train = transform_to_csp_fft_features(x_imagined, csp_w_tr, target_time_bins=n_sess)
        features_attempted_train = transform_to_csp_fft_features(x_attempted, csp_w_tr, target_time_bins=n_sess)

        imagined_train_idx, imagined_val_idx, imagined_test_idx = split_train_val_by_class(y_imagined, seed)
        attempted_train_idx, attempted_val_idx, attempted_test_idx = split_train_val_by_class(y_attempted, seed)

        x_imagined_tr = features_imagined_train[imagined_train_idx]
        x_imagined_val = features_imagined_train[imagined_val_idx]
        x_imagined_test = features_imagined_train[imagined_test_idx]
        y_imagined_tr = y_imagined[imagined_train_idx]
        y_imagined_val = y_imagined[imagined_val_idx]
        y_imagined_test = y_imagined[imagined_test_idx]

        x_attempted_tr = features_attempted_train[attempted_train_idx]
        x_attempted_val = features_attempted_train[attempted_val_idx]
        x_attempted_test = features_attempted_train[attempted_test_idx]
        y_attempted_tr = y_attempted[attempted_train_idx]
        y_attempted_val = y_attempted[attempted_val_idx]
        y_attempted_test = y_attempted[attempted_test_idx]

        # Export one CSV per trial as in NeuroTalk layout.
        csv_dir = "eegdata"

        export_trial_csvs(
            x_imagined_tr,
            y_imagined_tr,
            x_imagined_val,
            y_imagined_val,
            x_imagined_test,
            y_imagined_test,
            subject,
            os.path.join(csv_dir, "imagined_speech"),
        )

        export_trial_csvs(
            x_attempted_tr,
            y_attempted_tr,
            x_attempted_val,
            y_attempted_val,
            x_attempted_test,
            y_attempted_test,
            subject,
            os.path.join(csv_dir, "attempted_speech"),
        )

        out_path = os.path.join(data_dir, f"csp_features_subj{subject}_imagined_attempted.npz")
        np.savez_compressed(
            out_path,
            classes_used=csp_classes,
            csp_w_tr=csp_w_tr,
            csp_eigvals=csp_eigvals,
            x_train_imagined=x_imagined_tr,
            x_val_imagined=x_imagined_val,
            x_test_imagined=x_imagined_test,
            y_train_imagined=y_imagined_tr,
            y_val_imagined=y_imagined_val,
            y_test_imagined=y_imagined_test,
            x_train_attempted=x_attempted_tr,
            x_val_attempted=x_attempted_val,
            x_test_attempted=x_attempted_test,
            y_train_attempted=y_attempted_tr,
            y_val_attempted=y_attempted_val,
            y_test_attempted=y_attempted_test,
        )
        print(f"Subject {subject}: saved imagined+attempted CSP features to {out_path}")


if __name__ == "__main__":
    main()
