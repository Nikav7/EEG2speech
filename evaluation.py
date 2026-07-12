import os
import re
import importlib
import csv
from typing import Dict, List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#from skimage.metrics import structural_similarity

import json
import sys
from types import SimpleNamespace


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

GENERATED_DIR = os.path.join(PROJECT_ROOT, "inference22kHz_4subs")
EVENTS_CSV = os.path.join(PROJECT_ROOT, "events_codes.csv")
AUDIODATA_DIR = os.path.join(PROJECT_ROOT, "audiodata", "twos_22050")
ORIGINAL_MELS = os.path.join(PROJECT_ROOT, "audiodata", "logmel22")

SR = 22050

N_MELS = 80
FMIN = 20.0
FMAX = SR / 2.0
N_MFCC = 40
W2V_MODEL_NAME = "facebook/wav2vec2-base-960h"
HUBERT_MODEL_NAME = "facebook/hubert-large-ls960-ft"
W2V_FT_PATH = os.path.join(PROJECT_ROOT, "wav2vec2_finetuned")
RUN_W2V_TSNE = True
DIST_CLUSTER_THRESHOLD = 0.4  # for macro clusters
MEL_KMEANS_CLUSTERS = 13
ELBOW_K_MIN = 13
ELBOW_K_MAX = 74
RUN_PESQ = True
CER_TARGET_SR = 16000


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


def _resolve_generated_subject_dirs(generated_root: str) -> List[dict]:
    """Return subject-specific generated wav/mel/output dirs.

    Supports:
    - New layout: <generated_root>/subjXX/wav and <generated_root>/subjXX/mel_csv
    - Legacy layout: <generated_root>/ and <generated_root>/mel_csv
    """
    if not os.path.isdir(generated_root):
        raise FileNotFoundError(f"Generated inference directory not found: {generated_root}")

    resolved: List[dict] = []
    for name in sorted(os.listdir(generated_root)):
        subj_dir = os.path.join(generated_root, name)
        if not os.path.isdir(subj_dir):
            continue
        wav_dir = os.path.join(subj_dir, "wav")
        mel_dir = os.path.join(subj_dir, "mel_csv")
        if not os.path.isdir(wav_dir) or not os.path.isdir(mel_dir):
            continue
        if not any(f.lower().endswith(".wav") for f in os.listdir(wav_dir)):
            continue
        if not any(f.lower().endswith(".csv") for f in os.listdir(mel_dir)):
            continue
        resolved.append(
            {
                "subject_id": name,
                "generated_wav_dir": wav_dir,
                "generated_mel_dir": mel_dir,
                "output_dir": os.path.join(subj_dir, "evaluation_output"),
            }
        )

    if resolved:
        return resolved

    # Backward-compatible single folder layout.
    legacy_wavs = [f for f in os.listdir(generated_root) if f.lower().endswith(".wav")]
    legacy_mel_dir = os.path.join(generated_root, "mel_csv")
    legacy_mels = []
    if os.path.isdir(legacy_mel_dir):
        legacy_mels = [f for f in os.listdir(legacy_mel_dir) if f.lower().endswith(".csv")]

    if legacy_wavs and legacy_mels:
        return [
            {
                "subject_id": "all",
                "generated_wav_dir": generated_root,
                "generated_mel_dir": legacy_mel_dir,
                "output_dir": os.path.join(generated_root, "evaluation_output"),
            }
        ]

    raise RuntimeError(
        "No generated subject outputs found. Expected subject folders at "
        "<generated_root>/subjXX/{wav,mel_csv}"
    )


def natural_key(name):
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)), name) if m else (10**9, name)



def load_wavs(wav_dir: str):
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


def _extract_class_code(name: str) -> int:
    """Extract class code from file names like label61, audio61, etc."""
    match = re.search(r"(?:label|audio)(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    fallback = re.search(r"(\d+)", name)
    if not fallback:
        raise ValueError(f"Could not extract class code from file name: {name}")
    return int(fallback.group(1))


def _to_mono_float32(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32)


def _normalize_text_for_cer(text: str) -> str:
    text = str(text).upper()
    text = "".join(ch if ("A" <= ch <= "Z") else " " for ch in text)
    return "".join(text.split())


def _character_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(_normalize_text_for_cer(reference))
    hyp = list(_normalize_text_for_cer(hypothesis))

    if not ref:
        return 0.0 if not hyp else 1.0

    previous_row = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current_row = [i]
        for j, hyp_char in enumerate(hyp, start=1):
            substitution_cost = 0 if ref_char == hyp_char else 1
            current_row.append(
                min(
                    previous_row[j] + 1,
                    current_row[j - 1] + 1,
                    previous_row[j - 1] + substitution_cost,
                )
            )
        previous_row = current_row

    return float(previous_row[-1]) / float(max(1, len(ref)))


def _load_mel_csv_clean(mel_csv_path: str) -> np.ndarray:
    mel = np.loadtxt(mel_csv_path, delimiter=",", dtype=np.float32)
    mel = np.atleast_2d(mel)

    if mel.shape[0] > 1:
        expected_index = np.arange(mel.shape[1], dtype=np.float32)
        if np.allclose(mel[0], expected_index, rtol=0.0, atol=1e-6):
            mel = mel[1:]
    return mel


def _build_paired_sample_rows(
    generated_wav_dir: str,
    original_wav_dir: str,
    generated_mel_dir: str,
    original_mel_dir: str,
    word_labels: Dict[int, str],
) -> List[dict]:
    generated_wavs = sorted(
        [f for f in os.listdir(generated_wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not generated_wavs:
        raise RuntimeError(f"No generated WAV files found in {generated_wav_dir}")

    original_wavs = sorted(
        [f for f in os.listdir(original_wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    generated_mels = sorted(
        [f for f in os.listdir(generated_mel_dir) if f.lower().endswith(".csv")],
        key=natural_key,
    )
    original_mels = sorted(
        [f for f in os.listdir(original_mel_dir) if f.lower().endswith("_logmel.csv")],
        key=natural_key,
    )

    original_wav_by_code = {_extract_class_code(name): name for name in original_wavs}
    generated_mel_by_code = {_extract_class_code(name): name for name in generated_mels}
    original_mel_by_code = {_extract_class_code(name): name for name in original_mels}

    pairs: List[dict] = []
    for generated_wav in generated_wavs:
        class_code = _extract_class_code(generated_wav)

        ref_wav = original_wav_by_code.get(class_code)
        gen_mel = generated_mel_by_code.get(class_code)
        ref_mel = original_mel_by_code.get(class_code)

        if ref_wav is None:
            print(f"[PAIR] Skip {generated_wav}: missing reference WAV for class {class_code}")
            continue
        if gen_mel is None:
            print(f"[PAIR] Skip {generated_wav}: missing generated MEL for class {class_code}")
            continue
        if ref_mel is None:
            print(f"[PAIR] Skip {generated_wav}: missing reference MEL for class {class_code}")
            continue

        pairs.append(
            {
                "class_label": int(class_code),
                "word": word_labels.get(class_code, f"class{class_code}"),
                "generated_wav": generated_wav,
                "reference_wav": ref_wav,
                "generated_mel": gen_mel,
                "reference_mel": ref_mel,
            }
        )

    if not pairs:
        raise RuntimeError("No valid paired samples were found for evaluation")

    return pairs


def _write_metric_summary_csv(rows: List[dict], metric_key: str, out_csv_path: str) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write for {metric_key}")

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    fieldnames = [
        "class_label",
        "word",
        "generated_wav",
        "reference_wav",
        "generated_mel",
        "reference_mel",
        metric_key,
    ]

    # Keep any additional per-row metrics (e.g., CER ground truth) in the output CSV.
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    metric_values = [float(row[metric_key]) for row in rows]
    avg_value = float(np.mean(metric_values))

    extra_avg_values = {}
    for key in fieldnames:
        if key in ("class_label", "word", "generated_wav", "reference_wav", "generated_mel", "reference_mel", metric_key):
            continue
        numeric_vals = []
        for row in rows:
            try:
                numeric_vals.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        if numeric_vals:
            extra_avg_values[key] = float(np.mean(numeric_vals))

    with open(out_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        avg_row = {
            "class_label": "AVG",
            "word": "AVG",
            "generated_wav": "",
            "reference_wav": "",
            "generated_mel": "",
            "reference_mel": "",
            metric_key: f"{avg_value:.6f}",
        }
        for key, value in extra_avg_values.items():
            avg_row[key] = f"{value:.6f}"
        writer.writerow(avg_row)

    print(f"Saved {metric_key.upper()} summary: {out_csv_path}")


def compute_mcd_paired_rows(
    paired_rows: List[dict],
    generated_mel_dir: str,
    original_mel_dir: str,
    n_mfcc: int = N_MFCC,
) -> List[dict]:
    out_rows: List[dict] = []
    for row in paired_rows:
        ref_mel = _load_mel_csv_clean(os.path.join(original_mel_dir, row["reference_mel"]))
        gen_mel = _load_mel_csv_clean(os.path.join(generated_mel_dir, row["generated_mel"]))

        ref_mfcc = librosa.feature.mfcc(S=ref_mel, n_mfcc=n_mfcc).astype(np.float32)
        gen_mfcc = librosa.feature.mfcc(S=gen_mel, n_mfcc=n_mfcc).astype(np.float32)
        mcd_score = float(mel_cepstral_distortion(ref_mfcc, gen_mfcc))

        enriched = dict(row)
        enriched["mcd"] = mcd_score
        out_rows.append(enriched)
        print(
            f"[MCD] class={row['class_label']:02d} {row['generated_mel']} vs {row['reference_mel']}: {mcd_score:.4f}"
        )
    return out_rows


def compute_and_save_mcd_matrix_from_pairs(
    paired_rows: List[dict],
    generated_mel_dir: str,
    out_dir: str,
    n_mfcc: int = N_MFCC,
    word_labels: Dict[int, str] = None,
) -> np.ndarray:
    ordered_rows = sorted(paired_rows, key=lambda r: int(r["class_label"]))

    mel_list = []
    wav_names = []
    for row in ordered_rows:
        mel = _load_mel_csv_clean(os.path.join(generated_mel_dir, row["generated_mel"]))
        mel_list.append(mel)
        wav_names.append(row["generated_wav"])

    mel_arr = np.stack(mel_list, axis=0)
    mfcc_arr = compute_mfccs(mel_arr, n_mfcc=n_mfcc)
    mcd_matrix = mcd_pairwise_matrix(mfcc_arr, wav_names)

    os.makedirs(out_dir, exist_ok=True)
    matrix_npy_path = os.path.join(out_dir, "mcd_pairwise.npy")
    matrix_csv_path = os.path.join(out_dir, "mcd_pairwise.csv")
    matrix_png_path = os.path.join(out_dir, "mcd_pairwise_names.png")

    np.save(matrix_npy_path, mcd_matrix)
    np.savetxt(matrix_csv_path, mcd_matrix, delimiter=",", fmt="%.6f")
    plot_mcd_matrix(mcd_matrix, wav_names, matrix_png_path, word_labels=word_labels)

    print(f"Saved MCD matrix NPY: {matrix_npy_path}")
    print(f"Saved MCD matrix CSV: {matrix_csv_path}")
    return mcd_matrix


def compute_pesq_paired_rows(
    paired_rows: List[dict],
    generated_wav_dir: str,
    original_wav_dir: str,
    target_sr: int = 16000,
    mode: str = "wb",
) -> List[dict]:
    try:
        pesq_mod = importlib.import_module("pesq")
        pesq_fn = getattr(pesq_mod, "pesq")
    except Exception as exc:
        raise ImportError("PESQ package is required. Install with: pip install pesq") from exc

    out_rows: List[dict] = []
    for row in paired_rows:
        ref_wav, ref_sr = sf.read(os.path.join(original_wav_dir, row["reference_wav"]))
        gen_wav, gen_sr = sf.read(os.path.join(generated_wav_dir, row["generated_wav"]))

        ref_wav = _to_mono_float32(ref_wav)
        gen_wav = _to_mono_float32(gen_wav)

        if int(ref_sr) != int(target_sr):
            ref_wav = librosa.resample(ref_wav, orig_sr=int(ref_sr), target_sr=int(target_sr))
        if int(gen_sr) != int(target_sr):
            gen_wav = librosa.resample(gen_wav, orig_sr=int(gen_sr), target_sr=int(target_sr))

        min_len = min(len(ref_wav), len(gen_wav))
        if min_len < int(0.25 * target_sr):
            print(f"[PESQ] Skip class={row['class_label']:02d}: too short after alignment")
            continue

        score = float(pesq_fn(target_sr, ref_wav[:min_len], gen_wav[:min_len], mode))
        enriched = dict(row)
        enriched["pesq"] = score
        out_rows.append(enriched)
        print(
            f"[PESQ] class={row['class_label']:02d} {row['generated_wav']} vs {row['reference_wav']}: {score:.4f}"
        )

    if not out_rows:
        raise RuntimeError("No valid PESQ scores were computed")
    return out_rows


def compute_cer_paired_rows(
    paired_rows: List[dict],
    generated_wav_dir: str,
    original_wav_dir: str,
    model_name: str = W2V_MODEL_NAME,
    finetuned_path: str = W2V_FT_PATH,
    target_sr: int = CER_TARGET_SR,
) -> List[dict]:
    transformers_mod = importlib.import_module("transformers")
    AutoModelForCTC = getattr(transformers_mod, "AutoModelForCTC")
    AutoProcessor = getattr(transformers_mod, "AutoProcessor")

    source = finetuned_path if finetuned_path and os.path.isdir(finetuned_path) else model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(source)
    model = AutoModelForCTC.from_pretrained(source).to(device)
    model.eval()

    out_rows: List[dict] = []
    with torch.inference_mode():
        for row in paired_rows:
            generated_wav_path = os.path.join(generated_wav_dir, row["generated_wav"])
            reference_wav_path = os.path.join(original_wav_dir, row["reference_wav"])

            generated_waveform, generated_sr = librosa.load(generated_wav_path, sr=None, mono=True)
            reference_waveform, reference_sr = librosa.load(reference_wav_path, sr=None, mono=True)
            if int(generated_sr) != int(target_sr):
                generated_waveform = librosa.resample(generated_waveform, orig_sr=int(generated_sr), target_sr=int(target_sr))
            if int(reference_sr) != int(target_sr):
                reference_waveform = librosa.resample(reference_waveform, orig_sr=int(reference_sr), target_sr=int(target_sr))

            generated_inputs = processor(
                generated_waveform,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
            )
            generated_values = generated_inputs.input_values.to(device)
            generated_mask = generated_inputs.attention_mask.to(device) if "attention_mask" in generated_inputs else None
            generated_logits = model(input_values=generated_values, attention_mask=generated_mask).logits
            generated_pred_ids = torch.argmax(generated_logits, dim=-1)
            generated_pred_text = processor.batch_decode(generated_pred_ids, skip_special_tokens=True)[0]

            reference_inputs = processor(
                reference_waveform,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
            )
            reference_values = reference_inputs.input_values.to(device)
            reference_mask = reference_inputs.attention_mask.to(device) if "attention_mask" in reference_inputs else None
            reference_logits = model(input_values=reference_values, attention_mask=reference_mask).logits
            reference_pred_ids = torch.argmax(reference_logits, dim=-1)
            reference_pred_text = processor.batch_decode(reference_pred_ids, skip_special_tokens=True)[0]

            cer_score = _character_error_rate(row["word"], generated_pred_text)
            cer_gt_score = _character_error_rate(row["word"], reference_pred_text)
            enriched = dict(row)
            enriched["cer"] = float(cer_score)
            enriched["cer_gt"] = float(cer_gt_score)
            enriched["predicted_word_generated"] = generated_pred_text
            enriched["predicted_word_groundtruth"] = reference_pred_text
            out_rows.append(enriched)
            print(
                f"[CER] class={row['class_label']:02d} {row['generated_wav']} word='{row['word']}' "
                f"pred_gen='{generated_pred_text}' cer={cer_score:.4f} pred_gt='{reference_pred_text}' cer_gt={cer_gt_score:.4f}"
            )

    return out_rows


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


# def ssim_pairwise_matrix(mel_arr: np.ndarray, wav_names: list, win_size: int = 7) -> np.ndarray:
#     """Compute the full NxN pairwise SSIM similarity matrix on log-mel spectrograms.

#     mel_arr: [N, N_MELS, T] — treated as 2D grayscale images.
#     SSIM in [-1, 1]: higher = more similar.
#     """

#     N = len(mel_arr)
#     sim = np.zeros((N, N), dtype=np.float32)
#     data_range = float(mel_arr.max() - mel_arr.min())

#     for i in range(N):
#         sim[i, i] = 1.0
#         for j in range(i + 1, N):
#             s = structural_similarity(
#                 mel_arr[i], mel_arr[j],
#                 win_size=win_size,
#                 data_range=data_range,
#             )
#             sim[i, j] = s
#             sim[j, i] = s

#     print(f"\nPairwise SSIM matrix ({N}x{N}):")
#     for i in range(N):
#         row = sim[i].copy()
#         row[i] = -np.inf  # exclude self
#         nearest_idx = int(np.argmax(row))
#         print(f"  {wav_names[i]:20s}  most similar: {wav_names[nearest_idx]:20s}  SSIM={sim[i, nearest_idx]:.4f}")

#     return sim


# def ms_ssim_pairwise_matrix(
#     mel_arr: np.ndarray,
#     wav_names: list,
#     win_size: int = 7,
#     scales: int = 3,
#     downsample_factor: int = 2,
#     weights: tuple = (0.25, 0.25, 0.50),
# ) -> np.ndarray:
#     """Compute the full NxN pairwise MS-SSIM similarity matrix on log-mel spectrograms.

#     Computes SSIM at `scales` resolution levels, each downsampled by `downsample_factor`,
#     then combines them as a weighted sum with `weights` (must sum to 1).
#     mel_arr: [N, N_MELS, T]
#     """
#     #from skimage.transform import rescale

#     assert len(weights) == scales and abs(sum(weights) - 1.0) < 1e-6

#     N = len(mel_arr)
#     sim = np.zeros((N, N), dtype=np.float32)

#     # Pre-build downsampled pyramids for each sample
#     pyramids = []
#     for i in range(N):
#         pyramid = []
#         img = mel_arr[i].astype(np.float64)
#         for s in range(scales):
#             pyramid.append(img)
#             if s < scales - 1:
#                 img = rescale(img, 1.0 / downsample_factor, anti_aliasing=True, channel_axis=None)
#         pyramids.append(pyramid)

#     for i in range(N):
#         sim[i, i] = 1.0
#         for j in range(i + 1, N):
#             score = 0.0
#             for s in range(scales):
#                 a, b = pyramids[i][s], pyramids[j][s]
#                 data_range = float(mel_arr.max() - mel_arr.min())
#                 # Clamp win_size if image got too small after downsampling
#                 effective_win = min(win_size, a.shape[0], a.shape[1])
#                 if effective_win % 2 == 0:
#                     effective_win -= 1
#                 score += weights[s] * structural_similarity(
#                     a, b, win_size=effective_win, data_range=data_range
#                 )
#             sim[i, j] = score
#             sim[j, i] = score

#     print(f"\nPairwise MS-SSIM matrix ({N}x{N}):")
#     for i in range(N):
#         row = sim[i].copy()
#         row[i] = -np.inf
#         nearest_idx = int(np.argmax(row))
#         print(f"  {wav_names[i]:20s}  most similar: {wav_names[nearest_idx]:20s}  MS-SSIM={sim[i, nearest_idx]:.4f}")

#     return sim



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
    finetuned_path: str = W2V_FT_PATH,
    target_sr: int = 16000,
    device: str = None,
) -> tuple:
    """Compute one wav2vec embedding per audio using mean pooled hidden states.

    Returns:
        wav_names: sorted WAV file names
        embeddings: np.ndarray [N, D]
    """

    transformers_mod = importlib.import_module("transformers")
    AutoModel = getattr(transformers_mod, "AutoModel")
    AutoModelForCTC = getattr(transformers_mod, "AutoModelForCTC")
    AutoProcessor = getattr(transformers_mod, "AutoProcessor")

    wav_names = sorted(
        [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not wav_names:
        raise RuntimeError(f"No WAV files found in {wav_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prefer a local fine-tuned checkpoint when available, otherwise use model_name.
    pretrained_source = finetuned_path if finetuned_path and os.path.isdir(finetuned_path) else model_name
    if pretrained_source == finetuned_path:
        print(f"Loading wav2vec from fine-tuned path: {pretrained_source}")
    else:
        print(f"Fine-tuned path not found ({finetuned_path}); loading: {pretrained_source}")

    processor = AutoProcessor.from_pretrained(pretrained_source)
    try:
        model = AutoModel.from_pretrained(pretrained_source).to(device)
    except Exception as exc:
        # Some fine-tuned checkpoints are saved as CTC heads; use their backbone for embeddings.
        print(f"AutoModel load failed ({exc}); falling back to AutoModelForCTC backbone.")
        ctc_model = AutoModelForCTC.from_pretrained(pretrained_source).to(device)
        model = ctc_model.wav2vec2

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
    word_labels = load_word_labels(EVENTS_CSV)
    subject_runs = _resolve_generated_subject_dirs(GENERATED_DIR)

    print(
        "Found generated outputs for subjects: "
        + ", ".join(run["subject_id"] for run in subject_runs)
    )

    for run in subject_runs:
        subject_id = run["subject_id"]
        generated_wav_dir = run["generated_wav_dir"]
        generated_mel_dir = run["generated_mel_dir"]
        output_dir = run["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n===== Evaluating subject: {subject_id} =====")
        paired_rows = _build_paired_sample_rows(
            generated_wav_dir=generated_wav_dir,
            original_wav_dir=AUDIODATA_DIR,
            generated_mel_dir=generated_mel_dir,
            original_mel_dir=ORIGINAL_MELS,
            word_labels=word_labels,
        )

        mcd_rows = compute_mcd_paired_rows(
            paired_rows=paired_rows,
            generated_mel_dir=generated_mel_dir,
            original_mel_dir=ORIGINAL_MELS,
            n_mfcc=N_MFCC,
        )
        _write_metric_summary_csv(mcd_rows, "mcd", os.path.join(output_dir, "mcd_summary.csv"))
        compute_and_save_mcd_matrix_from_pairs(
            paired_rows=paired_rows,
            generated_mel_dir=generated_mel_dir,
            out_dir=output_dir,
            n_mfcc=N_MFCC,
            word_labels=word_labels,
        )

        pesq_summary_path = os.path.join(output_dir, "pesq_summary.csv")
        try:
            pesq_rows = compute_pesq_paired_rows(
                paired_rows=paired_rows,
                generated_wav_dir=generated_wav_dir,
                original_wav_dir=AUDIODATA_DIR,
                target_sr=16000,
                mode="wb",
            )
            _write_metric_summary_csv(pesq_rows, "pesq", pesq_summary_path)
        except ImportError as exc:
            print(f"[WARN] PESQ unavailable: {exc}")
            print("[WARN] Skipping PESQ. Continuing with MCD and CER.")
            with open(pesq_summary_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "class_label",
                        "word",
                        "generated_wav",
                        "reference_wav",
                        "generated_mel",
                        "reference_mel",
                        "pesq",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "class_label": "UNAVAILABLE",
                        "word": "UNAVAILABLE",
                        "generated_wav": "",
                        "reference_wav": "",
                        "generated_mel": "",
                        "reference_mel": "",
                        "pesq": "nan",
                    }
                )
            print(f"Saved placeholder PESQ summary: {pesq_summary_path}")
        except Exception as exc:
            print(f"[WARN] PESQ computation failed: {exc}")
            print("[WARN] Skipping PESQ. Continuing with MCD and CER.")

        # 1) CER with wav2vec base model
        cer_w2v_base_rows = compute_cer_paired_rows(
            paired_rows=paired_rows,
            generated_wav_dir=generated_wav_dir,
            original_wav_dir=AUDIODATA_DIR,
            model_name=W2V_MODEL_NAME,
            finetuned_path=None,
            target_sr=CER_TARGET_SR,
        )
        _write_metric_summary_csv(
            cer_w2v_base_rows,
            "cer",
            os.path.join(output_dir, "cer_wav2vec_base_summary.csv"),
        )

        # 2) CER with wav2vec fine-tuned checkpoint
        cer_w2v_ft_summary_path = os.path.join(output_dir, "cer_wav2vec_finetuned_summary.csv")
        try:
            if not os.path.isdir(W2V_FT_PATH):
                raise FileNotFoundError(f"Fine-tuned wav2vec checkpoint folder not found: {W2V_FT_PATH}")
            cer_w2v_ft_rows = compute_cer_paired_rows(
                paired_rows=paired_rows,
                generated_wav_dir=generated_wav_dir,
                original_wav_dir=AUDIODATA_DIR,
                model_name=W2V_MODEL_NAME,
                finetuned_path=W2V_FT_PATH,
                target_sr=CER_TARGET_SR,
            )
            _write_metric_summary_csv(cer_w2v_ft_rows, "cer", cer_w2v_ft_summary_path)
        except Exception as exc:
            print(f"[WARN] wav2vec fine-tuned CER unavailable: {exc}")
            with open(cer_w2v_ft_summary_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "class_label",
                        "word",
                        "generated_wav",
                        "reference_wav",
                        "generated_mel",
                        "reference_mel",
                        "cer",
                        "cer_gt",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "class_label": "UNAVAILABLE",
                        "word": "UNAVAILABLE",
                        "generated_wav": "",
                        "reference_wav": "",
                        "generated_mel": "",
                        "reference_mel": "",
                        "cer": "nan",
                        "cer_gt": "nan",
                    }
                )
            print(f"Saved placeholder CER summary: {cer_w2v_ft_summary_path}")

        # 3) CER with HuBERT
        cer_hubert_rows = compute_cer_paired_rows(
            paired_rows=paired_rows,
            generated_wav_dir=generated_wav_dir,
            original_wav_dir=AUDIODATA_DIR,
            model_name=HUBERT_MODEL_NAME,
            finetuned_path=None,
            target_sr=CER_TARGET_SR,
        )
        _write_metric_summary_csv(
            cer_hubert_rows,
            "cer",
            os.path.join(output_dir, "cer_hubert_summary.csv"),
        )
