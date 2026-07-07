import os
import re
import importlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#from skimage.metrics import structural_similarity
from preprocess.audio_prep import compute_mel_spectrograms

import json
import sys
from types import SimpleNamespace


STIMULI_DIR = r"C:\Users\lalli\Desktop\Thesis\data\TriParadigm\stimuli"
OUTPUT_DIR = os.path.join(STIMULI_DIR, "all_80002s")
NEWSTIM_DIR = os.path.join(STIMULI_DIR, "twos_22050")
EVENTS_CSV = os.path.join(os.path.dirname(__file__), "..", "events_codes.csv")
AUDIODATA_DIR = os.path.join(os.path.dirname(__file__), "..", "audiodata/twos_16000")

SR = 22050
TARGET_DURATION_S = 2.0
PREPEND_SILENCE_S = 0.1

#WIN_MS = 200.0
#HOP_MS = 20.0
N_MELS = 80
FMIN = 20.0
FMAX = SR / 2.0
N_MFCC = 40
W2V_MODEL_NAME = "facebook/wav2vec2-base-960h"
RUN_W2V_TSNE = True
DIST_CLUSTER_THRESHOLD = 0.4  # for macro clusters
MEL_KMEANS_CLUSTERS = 13
ELBOW_K_MIN = 13
ELBOW_K_MAX = 74


def load_word_labels(csv_path: str) -> dict:
    """Return {audio_number: word_label} from events_codes.csv."""
    import csv
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                word = row[0].strip().strip("'")
                try:
                    idx = int(row[1].strip())
                    labels[idx] = word
                except ValueError:
                    pass
    return labels


def natural_key(name):
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)), name) if m else (10**9, name)


def process_and_save(stimuli_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    wav_names = sorted(
        [f for f in os.listdir(stimuli_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not wav_names:
        raise RuntimeError(f"No WAV files found in {stimuli_dir}")

    expected_samples = int(round(TARGET_DURATION_S * SR))
    prepend_samples = int(round(PREPEND_SILENCE_S * SR))
    content_samples = expected_samples - prepend_samples

    for wav_name in wav_names:
        waveform, orig_sr = librosa.load(os.path.join(stimuli_dir, wav_name), sr=None, mono=True)

        if int(orig_sr) != SR:
            waveform = librosa.resample(waveform, orig_sr=int(orig_sr), target_sr=SR)

        if len(waveform) == 0:
            stretched = np.zeros(content_samples, dtype=np.float32)
        else:
            rate = len(waveform) / float(content_samples)
            stretched = librosa.effects.time_stretch(waveform.astype(np.float32), rate=rate)
            if len(stretched) < content_samples:
                stretched = np.pad(stretched, (0, content_samples - len(stretched)))
            else:
                stretched = stretched[:content_samples]

        out = np.concatenate([np.zeros(prepend_samples, dtype=np.float32), stretched])
        sf.write(os.path.join(output_dir, wav_name), out, SR)
        print(f"Saved: {wav_name} ({len(out)/SR:.2f}s)")

    print(f"\nDone. {len(wav_names)} files saved to: {output_dir}")


def load_processed_wavs(wav_dir: str):
    wav_names = sorted(
        [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not wav_names:
        raise RuntimeError(f"No WAV files found in {wav_dir}")
    waveforms = []
    for wav_name in wav_names:
        waveform, _ = librosa.load(os.path.join(wav_dir, wav_name), sr=SR, mono=True)
        waveforms.append(waveform)
        print(f"Loaded: {wav_name} ({len(waveform)/SR:.2f}s)")
    print(f"\nLoaded {len(waveforms)} files from: {wav_dir}")
    return wav_names, waveforms


def load_log_mel_csvs(csv_dir: str) -> tuple:
    """Load existing log-mel CSV files and stack them into [N, N_MELS, T]."""
    csv_names = sorted(
        [f for f in os.listdir(csv_dir) if f.lower().endswith("_logmel.csv")],
        key=natural_key,
    )
    if not csv_names:
        raise RuntimeError(f"No log-mel CSV files found in {csv_dir}")

    mel_list = []
    expected_shape = None
    for csv_name in csv_names:
        csv_path = os.path.join(csv_dir, csv_name)
        log_mel = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        log_mel = np.atleast_2d(log_mel)
        if log_mel.shape[0] > 1:
            expected_index = np.arange(log_mel.shape[1], dtype=np.float32)
            if np.allclose(log_mel[0], expected_index, rtol=0.0, atol=1e-6):
                log_mel = log_mel[1:]

        if expected_shape is None:
            expected_shape = log_mel.shape
        elif log_mel.shape != expected_shape:
            raise ValueError(
                f"Inconsistent log-mel shape for {csv_name}: {log_mel.shape}, expected {expected_shape}"
            )

        mel_list.append(log_mel)
        print(f"Loaded: {csv_name} (shape={log_mel.shape})")

    mel_arr = np.stack(mel_list, axis=0)
    print(f"\nLoaded {len(mel_arr)} log-mel CSV files from: {csv_dir}")
    print(f"Stacked log-mel array shape: {mel_arr.shape}")
    return csv_names, mel_arr


def compute_kmeans_wcss_curve(
    representations: np.ndarray,
    k_min: int = ELBOW_K_MIN,
    k_max: int = ELBOW_K_MAX,
    random_state: int = 42,
) -> tuple:
    """Run KMeans for each K in [k_min, k_max] and return K values with WCSS."""

    n_samples = representations.shape[0]
    upper_k = min(k_max, n_samples)
    if k_min > upper_k:
        raise ValueError(
            f"Invalid K range [{k_min}, {k_max}] for {n_samples} samples; maximum valid K is {upper_k}"
        )

    flat_representations = representations.reshape(n_samples, -1)
    k_values = list(range(k_min, upper_k + 1))
    wcss_values = []

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=40, random_state=random_state)
        model.fit(flat_representations)
        wcss = float(model.inertia_)
        wcss = np.sqrt(wcss)
        wcss_values.append(wcss)
        print(f"K={k:2d} -> WCSS={wcss:.4f}")

    return np.array(k_values, dtype=np.int32), np.array(wcss_values, dtype=np.float32)


def plot_wcss_elbow(k_values: np.ndarray, wcss_values: np.ndarray, out_path: str):
    """Plot WCSS against K for elbow-method inspection."""
    if len(k_values) != len(wcss_values):
        raise ValueError("k_values and wcss_values must have the same length")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(k_values, wcss_values, marker="o", linewidth=2)
    ax.set_title("K-means Elbow Plot on wav2vec embeddings")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved WCSS elbow plot: {out_path}")


def run_k_elbow_analysis(
    csv_dir: str = None,
    out_plot_path = str,
    out_csv_path = str,
    k_min: int = ELBOW_K_MIN,
    k_max: int = ELBOW_K_MAX,
    random_state: int = 42,
    other_embeddings: bool = True,
    arr: np.ndarray = None,
) -> tuple:
    """Load log-mel CSVs, compute WCSS across K, and save the elbow plot."""
    if other_embeddings:
        mel_arr = arr
    else:
        csv_names, mel_arr = load_log_mel_csvs(csv_dir)
    
    k_values, wcss_values = compute_kmeans_wcss_curve(
        mel_arr,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )

   
    out_plot_path = os.path.join(out_plot_path, f"wav2vec_kmeans_elbow_k{k_min}_to_{k_values[-1]}.png")
    plot_wcss_elbow(k_values, wcss_values, out_plot_path)

    
    out_csv_path = os.path.join(out_csv_path, f"wav2vec_kmeans_wcss_k{k_min}_to_{k_values[-1]}.csv")
    np.savetxt(
        out_csv_path,
        np.column_stack((k_values, wcss_values)),
        delimiter=",",
        header="k,wcss",
        comments="",
        fmt=["%d", "%.8f"],
    )
    print(f"Saved WCSS values: {out_csv_path}")
    return k_values, wcss_values



def compute_mfccs(log_mel_arr: np.ndarray, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Compute MFCCs from a log-mel spectrogram array [N, N_MELS, T].
    Returns mfcc_arr of shape [N, n_mfcc, T].
    """
    mfcc_list = []
    for log_mel in log_mel_arr:  # [N_MELS, T]
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=n_mfcc).astype(np.float32)
        mfcc_list.append(mfcc)
    mfcc_arr = np.stack(mfcc_list, axis=0)  # [N, n_mfcc, T]
    print(f"MFCCs shape: {mfcc_arr.shape}")
    return mfcc_arr


def mel_cepstral_distortion(mfcc_ref: np.ndarray, mfcc_deg: np.ndarray) -> float:
    """Compute the Mean Mel-Cepstral Distortion (MCD) between two MFCC arrays.

    mfcc_ref, mfcc_deg: [n_mfcc, T] or [N, n_mfcc, T].
    MCD (dB) = (10 / ln(10)) * mean_over_frames( sqrt(2 * sum_c((c_ref - c_deg)^2)) )
    Coefficient c0 (index 0) is excluded per convention.
    """
    K = 10.0 / np.log(10.0)

    def _mcd_single(ref, deg):  # [n_mfcc, T]
        min_t = min(ref.shape[1], deg.shape[1])
        diff = ref[1:, :min_t] - deg[1:, :min_t]  # exclude c0
        return K * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=0)))

    if mfcc_ref.ndim == 2:
        return float(_mcd_single(mfcc_ref, mfcc_deg))

    # [N, n_mfcc, T] — return per-sample MCD array
    scores = np.array([_mcd_single(mfcc_ref[i], mfcc_deg[i]) for i in range(len(mfcc_ref))])
    print(f"MCD per sample: mean={scores.mean():.4f} dB, std={scores.std():.4f} dB")
    return scores


def mcd_pairwise_matrix(mfcc_arr: np.ndarray, wav_names: list) -> np.ndarray:
    """Compute the full NxN pairwise MCD distance matrix.

    mfcc_arr: [N, n_mfcc, T]
    Returns dist_matrix [N, N] and prints the closest neighbour for each audio.
    """
    K = 10.0 / np.log(10.0)
    N = len(mfcc_arr)
    dist = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i + 1, N):
            min_t = min(mfcc_arr[i].shape[1], mfcc_arr[j].shape[1])
            diff = mfcc_arr[i][1:, :min_t] - mfcc_arr[j][1:, :min_t]
            d = K * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=0)))
            dist[i, j] = d
            dist[j, i] = d

    print(f"\nPairwise MCD matrix ({N}x{N}):")
    for i in range(N):
        row = dist[i].copy()
        row[i] = np.inf  # exclude self
        nearest_idx = int(np.argmin(row))
        print(f"  {wav_names[i]:20s}  closest: {wav_names[nearest_idx]:20s}  MCD={dist[i, nearest_idx]:.4f} dB")

    return dist


def ssim_pairwise_matrix(mel_arr: np.ndarray, wav_names: list, win_size: int = 7) -> np.ndarray:
    """Compute the full NxN pairwise SSIM similarity matrix on log-mel spectrograms.

    mel_arr: [N, N_MELS, T] — treated as 2D grayscale images.
    SSIM in [-1, 1]: higher = more similar.
    """

    N = len(mel_arr)
    sim = np.zeros((N, N), dtype=np.float32)
    data_range = float(mel_arr.max() - mel_arr.min())

    for i in range(N):
        sim[i, i] = 1.0
        for j in range(i + 1, N):
            s = structural_similarity(
                mel_arr[i], mel_arr[j],
                win_size=win_size,
                data_range=data_range,
            )
            sim[i, j] = s
            sim[j, i] = s

    print(f"\nPairwise SSIM matrix ({N}x{N}):")
    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self
        nearest_idx = int(np.argmax(row))
        print(f"  {wav_names[i]:20s}  most similar: {wav_names[nearest_idx]:20s}  SSIM={sim[i, nearest_idx]:.4f}")

    return sim


def ms_ssim_pairwise_matrix(
    mel_arr: np.ndarray,
    wav_names: list,
    win_size: int = 7,
    scales: int = 3,
    downsample_factor: int = 2,
    weights: tuple = (0.25, 0.25, 0.50),
) -> np.ndarray:
    """Compute the full NxN pairwise MS-SSIM similarity matrix on log-mel spectrograms.

    Computes SSIM at `scales` resolution levels, each downsampled by `downsample_factor`,
    then combines them as a weighted sum with `weights` (must sum to 1).
    mel_arr: [N, N_MELS, T]
    """
    #from skimage.transform import rescale

    assert len(weights) == scales and abs(sum(weights) - 1.0) < 1e-6

    N = len(mel_arr)
    sim = np.zeros((N, N), dtype=np.float32)

    # Pre-build downsampled pyramids for each sample
    pyramids = []
    for i in range(N):
        pyramid = []
        img = mel_arr[i].astype(np.float64)
        for s in range(scales):
            pyramid.append(img)
            if s < scales - 1:
                img = rescale(img, 1.0 / downsample_factor, anti_aliasing=True, channel_axis=None)
        pyramids.append(pyramid)

    for i in range(N):
        sim[i, i] = 1.0
        for j in range(i + 1, N):
            score = 0.0
            for s in range(scales):
                a, b = pyramids[i][s], pyramids[j][s]
                data_range = float(mel_arr.max() - mel_arr.min())
                # Clamp win_size if image got too small after downsampling
                effective_win = min(win_size, a.shape[0], a.shape[1])
                if effective_win % 2 == 0:
                    effective_win -= 1
                score += weights[s] * structural_similarity(
                    a, b, win_size=effective_win, data_range=data_range
                )
            sim[i, j] = score
            sim[j, i] = score

    print(f"\nPairwise MS-SSIM matrix ({N}x{N}):")
    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf
        nearest_idx = int(np.argmax(row))
        print(f"  {wav_names[i]:20s}  most similar: {wav_names[nearest_idx]:20s}  MS-SSIM={sim[i, nearest_idx]:.4f}")

    return sim


def plot_msssim_matrix(sim_matrix: np.ndarray, wav_names: list, out_path: str, word_labels: dict = None):
    N = len(wav_names)
    if word_labels:
        labels = [word_labels.get(natural_key(n)[0], os.path.splitext(n)[0]) for n in wav_names]
    else:
        labels = [os.path.splitext(n)[0] for n in wav_names]

    fig, ax = plt.subplots(figsize=(max(10, N * 0.25), max(8, N * 0.25)))
    im = ax.imshow(sim_matrix, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="MS-SSIM")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Pairwise MS-SSIM Matrix (log-mel, 7×7 Gaussian, 3 scales, weights=[0.25,0.25,0.50])")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved MS-SSIM matrix plot: {out_path}")


def plot_ssim_matrix(sim_matrix: np.ndarray, wav_names: list, out_path: str, word_labels: dict = None):
    N = len(wav_names)
    if word_labels:
        labels = [word_labels.get(natural_key(n)[0], os.path.splitext(n)[0]) for n in wav_names]
    else:
        labels = [os.path.splitext(n)[0] for n in wav_names]

    fig, ax = plt.subplots(figsize=(max(10, N * 0.25), max(8, N * 0.25)))
    im = ax.imshow(sim_matrix, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="SSIM")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Pairwise SSIM Matrix (log-mel, 7×7 kernel)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved SSIM matrix plot: {out_path}")


def plot_mcd_matrix(dist_matrix: np.ndarray, wav_names: list, out_path: str, word_labels: dict = None):
    N = len(wav_names)
    if word_labels:
        labels = [word_labels.get(natural_key(n)[0], os.path.splitext(n)[0]) for n in wav_names]
    else:
        labels = [os.path.splitext(n)[0] for n in wav_names]

    fig, ax = plt.subplots(figsize=(max(10, N * 0.25), max(8, N * 0.25)))
    im = ax.imshow(dist_matrix, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="MCD (dB)")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Pairwise Mel-Cepstral Distortion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved MCD matrix plot: {out_path}")


def plot_mel_tsne(
    mel_arr: np.ndarray,
    wav_names: list,
    out_path: str,
    word_labels: dict = None,
    perplexity: float = 10.0,
    random_state: int = 42,
):
    """Run 2D t-SNE on flattened log-mel spectrograms and save a labeled scatter plot."""
    if mel_arr.ndim != 3:
        raise ValueError(f"Expected mel_arr shape [N, N_MELS, T], got {mel_arr.shape}")
    if len(wav_names) != len(mel_arr):
        raise ValueError("wav_names length must match number of mel spectrograms")

    n_samples = mel_arr.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to run t-SNE reliably")

    effective_perplexity = min(perplexity, float(n_samples - 1))
    flat_mels = mel_arr.reshape(n_samples, -1)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    points = tsne.fit_transform(flat_mels)

    if word_labels:
        labels = [word_labels.get(natural_key(n)[0], os.path.splitext(n)[0]) for n in wav_names]
    else:
        labels = [os.path.splitext(n)[0] for n in wav_names]

    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_idx = {label: index for index, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for label in unique_labels:
        indices = [index for index, value in enumerate(labels) if value == label]
        xy = points[indices]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=30,
            alpha=0.85,
            color=colors(label_to_idx[label]),
            label=label,
        )

    ax.set_title("t-SNE of Log-Mel Spectrograms")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)

    legend_cols = 1 if len(unique_labels) <= 20 else 2
    ax.legend(
        title="Words",
        fontsize=7,
        title_fontsize=8,
        ncol=legend_cols,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mel t-SNE plot: {out_path}")


def plot_mel_kmeans_clusters(
    mel_arr: np.ndarray,
    cluster_ids: np.ndarray,
    wav_names: list,
    out_path: str,
    perplexity: float = 10.0,
    random_state: int = 42,
):

    n_samples = mel_arr.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to run t-SNE reliably")

    flat_mels = mel_arr.reshape(n_samples, -1)
    effective_perplexity = min(perplexity, float(n_samples - 1))
    points = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(flat_mels)

    unique_clusters = sorted(np.unique(cluster_ids).tolist())
    colors = plt.cm.get_cmap("tab10", max(len(unique_clusters), 1))

    fig, ax = plt.subplots(figsize=(11, 8))
    for index, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        xy = points[mask]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=35,
            alpha=0.85,
            color=colors(index),
            label=f"Cluster {int(cluster_id)}",
        )

    ax.set_title("K-means Clusters of Log-Mel Spectrograms")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Clusters", fontsize=8, title_fontsize=9, loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mel K-means cluster plot: {out_path}")


def compute_wav2vec_embeddings(
    wav_dir: str,
    model_name: str = W2V_MODEL_NAME,
    target_sr: int = 16000,
    device: str = None,
) -> tuple:
    """Compute one wav2vec embedding per audio using mean pooled hidden states.

    Returns:
        wav_names: sorted WAV file names
        embeddings: np.ndarray [N, D]
    """
  
    transformers_mod = importlib.import_module("transformers")
    Wav2Vec2Model = getattr(transformers_mod, "Wav2Vec2Model")
    Wav2Vec2Processor = getattr(transformers_mod, "Wav2Vec2Processor")
  
    wav_names = sorted(
        [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not wav_names:
        raise RuntimeError(f"No WAV files found in {wav_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for wav_name in wav_names:
            wav_path = os.path.join(wav_dir, wav_name)
            waveform, wav_sr = librosa.load(wav_path, sr=None, mono=True)
            if int(wav_sr) != int(target_sr):
                waveform = librosa.resample(waveform, orig_sr=int(wav_sr), target_sr=int(target_sr))

            inputs = processor(
                waveform,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

            outputs = model(input_values=input_values, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state[0]
            emb = hidden.mean(dim=0).cpu().numpy().astype(np.float32)
            embeddings.append(emb)
            print(f"{wav_name}: wav2vec embedding shape {emb.shape}")

    emb_arr = np.stack(embeddings, axis=0)
    out_path = os.path.join(wav_dir, "wav2vec_embeddings.npy")
    np.save(out_path, emb_arr)
    print(f"\nSaved wav2vec embeddings: {emb_arr.shape} -> {out_path}")
    return wav_names, emb_arr


def plot_wav2vec_tsne(
    embeddings: np.ndarray,
    wav_names: list,
    out_path: str,
    word_labels: dict = None,
    perplexity: float = 10.0,
    random_state: int = 42,
):
    """Run 2D t-SNE on wav2vec embeddings and save a labeled scatter plot."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {embeddings.shape}")
    if len(wav_names) != len(embeddings):
        raise ValueError("wav_names length must match number of embeddings")

    n_samples = embeddings.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to run t-SNE reliably")
    # t-SNE requires perplexity < n_samples
    effective_perplexity = min(perplexity, float(n_samples - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    points = tsne.fit_transform(embeddings)

    if word_labels:
        labels = [word_labels.get(natural_key(n)[0], os.path.splitext(n)[0]) for n in wav_names]
    else:
        labels = [os.path.splitext(n)[0] for n in wav_names]

    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for lab in unique_labels:
        idxs = [i for i, value in enumerate(labels) if value == lab]
        xy = points[idxs]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=30,
            alpha=0.85,
            color=colors(label_to_idx[lab]),
            label=lab,
        )

    ax.set_title("t-SNE of wav2vec Audio Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)

    legend_cols = 1 if len(unique_labels) <= 20 else 2
    ax.legend(
        title="Words",
        fontsize=7,
        title_fontsize=8,
        ncol=legend_cols,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved wav2vec t-SNE plot: {out_path}")


def cluster_by_distance_threshold(embeddings: np.ndarray, threshold: float = DIST_CLUSTER_THRESHOLD) -> np.ndarray:
    """Cluster embeddings by graph connectivity where Euclidean distance <= threshold."""
    n = embeddings.shape[0]
    dmat = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)
    adjacency = dmat <= threshold

    cluster_ids = -np.ones(n, dtype=np.int32)
    cid = 0
    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        stack = [i]
        cluster_ids[i] = cid
        while stack:
            u = stack.pop()
            neighbors = np.where(adjacency[u])[0]
            for v in neighbors:
                if cluster_ids[v] == -1:
                    cluster_ids[v] = cid
                    stack.append(v)
        cid += 1
    return cluster_ids


def cluster_by_kmeans(embeddings: np.ndarray, n_clusters: int = MEL_KMEANS_CLUSTERS, random_state: int = 42) -> np.ndarray:
    """Optional baseline: KMeans cluster IDs."""
    return KMeans(n_clusters=n_clusters, n_init=40, random_state=random_state).fit_predict(embeddings)


def cluster_log_mels_kmeans(mel_arr: np.ndarray, n_clusters: int = MEL_KMEANS_CLUSTERS, random_state: int = 42) -> np.ndarray:
    """Apply K-means to flattened log-mel spectrograms."""
    if mel_arr.ndim != 3:
        raise ValueError(f"Expected mel_arr shape [N, N_MELS, T], got {mel_arr.shape}")

    n_samples = mel_arr.shape[0]

    effective_clusters = min(max(1, n_clusters), n_samples)
    flat_mels = mel_arr.reshape(n_samples, -1)
    return cluster_by_kmeans(flat_mels, n_clusters=effective_clusters, random_state=random_state)


def save_cluster_report(
    cluster_ids: np.ndarray,
    wav_names: list,
    out_path: str,
    word_labels: dict = None,
    method_name: str = "clusters",
):
    labels = [word_labels.get(natural_key(n)[0], os.path.splitext(n)[0]) for n in wav_names] if word_labels else wav_names
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("wav_name,word_label,cluster_id\n")
        for wav_name, label, cid in zip(wav_names, labels, cluster_ids):
            f.write(f"{wav_name},{label},{int(cid)}\n")

    unique, counts = np.unique(cluster_ids, return_counts=True)
    print(f"\n{method_name}:")
    for c, k in zip(unique, counts):
        print(f"  cluster {int(c)}: {int(k)} samples")
    print(f"Saved cluster report: {out_path}")


if __name__ == "__main__":
    # process_and_save(STIMULI_DIR, OUTPUT_DIR)
    wav_names, _ = load_processed_wavs(NEWSTIM_DIR)
    #mels = compute_mel_spectrograms(OUTPUT_DIR)
    #run_log_mel_csv_elbow_analysis()
    #mfccs = compute_mfccs(mels)
    #dist_matrix = mcd_pairwise_matrix(mfccs, wav_names)
    #np.save(os.path.join(OUTPUT_DIR, "mcd_pairwise.npy"), dist_matrix)
    word_labels = load_word_labels(EVENTS_CSV)
    #plot_mel_tsne(mels, wav_names, os.path.join(OUTPUT_DIR, "mel_tsne_2d.png"), word_labels=word_labels)
    # mel_clusters = cluster_log_mels_kmeans(mels)
    # plot_mel_kmeans_clusters(mels, mel_clusters, wav_names, os.path.join(OUTPUT_DIR, f"mel_kmeans_clusters{MEL_KMEANS_CLUSTERS}.png"))
    # save_cluster_report(
    #     mel_clusters,
    #     wav_names,
    #     os.path.join(OUTPUT_DIR, f"mel_kmeans_clusters{MEL_KMEANS_CLUSTERS}.csv"),
    #     word_labels=word_labels,
    #     method_name="Log-mel K-means clusters",
    # )
    #ssim_matrix = ssim_pairwise_matrix(mels, wav_names)
    #plot_ssim_matrix(ssim_matrix, wav_names, os.path.join(OUTPUT_DIR, "ssim_pairwise.png"), word_labels=word_labels)
    #msssim_matrix = ms_ssim_pairwise_matrix(mels, wav_names)
    #plot_msssim_matrix(msssim_matrix, wav_names, os.path.join(OUTPUT_DIR, "msssim_pairwise.png"), word_labels=word_labels)
    if RUN_W2V_TSNE:
        w2v_names, w2v_embeddings = compute_wav2vec_embeddings(NEWSTIM_DIR)
        run_k_elbow_analysis(out_plot_path=AUDIODATA_DIR, out_csv_path=AUDIODATA_DIR, other_embeddings=True, arr=w2v_embeddings)
        plot_wav2vec_tsne(
            w2v_embeddings,
            w2v_names,
            os.path.join(OUTPUT_DIR, "wav2vec_tsne_2d_22050hz.png"),
            word_labels=word_labels,
        )
        # macro_clusters = cluster_by_distance_threshold(w2v_embeddings, threshold=DIST_CLUSTER_THRESHOLD)
        # save_cluster_report(
        #     macro_clusters,
        #     w2v_names,
        #     os.path.join(OUTPUT_DIR, "wav2vec_macro_clusters.csv"),
        #     word_labels=word_labels,
        # )

    
    #plot_mcd_matrix(dist_matrix, wav_names, os.path.join(OUTPUT_DIR, "mcd_pairwise_names.png"), word_labels=word_labels)
    


    # IMPORTANT!!! THIS CHECK ON 25/04/26 CONFIRMS
    # THAT YOU NEED TO USE 22050 Hz mel spectrograms to keep everything aligned with hi-fi GAN
    # AND keep GOOD RESOLUTION
    # wav0, _ = librosa.load(os.path.join(NEWSTIM_DIR, wav_names[0]), sr=SR, mono=True)
    # mel8k = librosa.power_to_db(librosa.feature.melspectrogram(y=wav0, sr=SR, n_fft=1024, hop_length=256, win_length=400, n_mels=N_MELS, fmin=FMIN, fmax=FMAX), ref=np.max)
    # target_frames = int(round(len(wav0) / SR * 22050 / 256))  # expected HiFi-GAN time frames
    # mel22k = torch.nn.functional.interpolate(torch.from_numpy(mel8k).unsqueeze(0).unsqueeze(0), size=(N_MELS, target_frames), mode="bilinear", align_corners=False).squeeze(0).squeeze(0).numpy()
    
    # _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # if _root not in sys.path: sys.path.insert(0, _root)
    # from models.models_HiFi import Generator as _HiFiG
    
    # _hcfg = SimpleNamespace(**json.load(open(os.path.join(_root, "UNIVERSAL_V1", "config.json"))))
    # _hg = _HiFiG(_hcfg).eval()
    # _ckpt = torch.load(os.path.join(_root, "UNIVERSAL_V1", "g_02500000"), map_location="cpu")
    # _hg.load_state_dict(_ckpt["generator"] if isinstance(_ckpt, dict) and "generator" in _ckpt else _ckpt)
    # with torch.no_grad():
    #     _wav_out = _hg(torch.from_numpy(mel22k).float().unsqueeze(0)).squeeze().numpy()
    # sf.write(os.path.join(NEWSTIM_DIR, "check_hifi.wav"), _wav_out / (np.abs(_wav_out).max() + 1e-9), 22050)
