import os
import re
import importlib
import csv
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from transformers import AutoModelForSpeechSeq2Seq

from utils import _extract_class_code, natural_key

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

GENERATED_DIR = os.path.join(PROJECT_ROOT, "inference22kHz_3subs1618_ep475")
EVENTS_CSV = os.path.join(PROJECT_ROOT, "events_codes.csv")
AUDIODATA_DIR = os.path.join(PROJECT_ROOT, "audiodata", "twos_22050")
ORIGINAL_MELS = os.path.join(PROJECT_ROOT, "audiodata", "logmel22")

SR = 22050
N_MELS = 80
FMIN = 20.0
FMAX = SR / 2.0
N_MFCC = 40

W2V_MODEL_NAME = "facebook/wav2vec2-base-960h"
WHISPER_MODEL_NAME = "openai/whisper-base"
HUBERT_MODEL_NAME = "facebook/hubert-large-ls960-ft"
W2V_FT_PATH = os.path.join(PROJECT_ROOT, "wav2vec2_finetuned")

CER_TARGET_SR = 16000


def load_word_labels(csv_path: str) -> dict:
    """Return {audio_number: word_label} from events_codes.csv."""
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
    """Return subject-specific generated wav/mel/output dirs."""
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

    # single folder layout
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

    raise RuntimeError("No generated subject outputs found.")


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


def dtw_align_spectrograms(ref_mel: np.ndarray, gen_mel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align gen_mel to ref_mel using fastdtw on un-normalized mel-spectrograms.
    Returns:
        ref_mel: [N_MELS, T_ref]
        warped_gen_mel: [N_MELS, T_ref] (mapped onto ref_mel time frames)
    """
    # fastdtw aligns along dimension 0, so transpose to [T, N_MELS]
    ref_frames = ref_mel.T
    gen_frames = gen_mel.T

    _, path = fastdtw(ref_frames, gen_frames, dist=euclidean)

    # Construct frame map: map each frame of ref_mel to mean frame of aligned gen_mel
    ref_to_gen_map = {}
    for r_idx, g_idx in path:
        ref_to_gen_map.setdefault(r_idx, []).append(g_idx)

    warped_gen_frames = []
    num_ref_frames = ref_mel.shape[1]

    for r_idx in range(num_ref_frames):
        if r_idx in ref_to_gen_map:
            g_indices = ref_to_gen_map[r_idx]
            mean_frame = np.mean(gen_frames[g_indices], axis=0)
        else:
            mean_frame = gen_frames[min(r_idx, gen_frames.shape[0] - 1)]
        warped_gen_frames.append(mean_frame)

    warped_gen_mel = np.array(warped_gen_frames).T  # back to [N_MELS, T_ref]
    return ref_mel, warped_gen_mel


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

    # Map class codes to original reference files
    original_wav_by_code = {_extract_class_code(name): name for name in original_wavs}
    original_mel_by_code = {_extract_class_code(name): name for name in original_mels}
    
    # Map exact trial prefix (e.g. "label008_trial0032") to generated mel files
    generated_mel_by_prefix = {
        os.path.splitext(f)[0].replace("_pred_mel", "").replace("_mel", ""): f 
        for f in generated_mels
    }

    pairs: List[dict] = []
    for gen_wav in generated_wavs:
        class_code = _extract_class_code(gen_wav)
        wav_prefix = os.path.splitext(gen_wav)[0].replace("_pred", "").replace("_wav", "")

        ref_wav = original_wav_by_code.get(class_code)
        ref_mel_name = original_mel_by_code.get(class_code)
        
        # Try matching by prefix first, fallback to class code match
        gen_mel_name = generated_mel_by_prefix.get(wav_prefix)
        if gen_mel_name is None:
            gen_mel_matches = [m for m in generated_mels if _extract_class_code(m) == class_code]
            if gen_mel_matches:
                gen_mel_name = gen_mel_matches[0]

        if ref_wav is None or gen_mel_name is None or ref_mel_name is None:
            print(f"[PAIR SKIP] Could not pair {gen_wav} (class {class_code}). Missing ref/mel.")
            continue

        ref_mel_path = os.path.join(original_mel_dir, ref_mel_name)
        gen_mel_path = os.path.join(generated_mel_dir, gen_mel_name)

        if not os.path.exists(ref_mel_path) or not os.path.exists(gen_mel_path):
            print(f"[FILE MISSING] Check path: {ref_mel_path} or {gen_mel_path}")
            continue

        ref_mel = _load_mel_csv_clean(ref_mel_path)
        raw_gen_mel = _load_mel_csv_clean(gen_mel_path)

        # DTW alignment before evaluation
        ref_mel_aligned, warped_gen_mel = dtw_align_spectrograms(ref_mel, raw_gen_mel)

        pairs.append(
            {
                "class_label": int(class_code),
                "word": word_labels.get(class_code, f"class{class_code}"),
                "generated_wav": gen_wav,
                "reference_wav": ref_wav,
                "generated_mel": gen_mel_name,
                "reference_mel": ref_mel_name,
                "ref_mel_arr": ref_mel_aligned,
                "gen_mel_arr": warped_gen_mel,
            }
        )

    print(f"Successfully paired {len(pairs)} out of {len(generated_wavs)} samples.")
    if not pairs:
        raise RuntimeError("No valid paired samples were found for evaluation.")

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

    for row in rows:
        for key in row.keys():
            if key not in fieldnames and key not in ("ref_mel_arr", "gen_mel_arr"):
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
            clean_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(clean_row)
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


def _format_mean_plus_std(rows: List[dict], metric_key: str) -> str:
    if not rows:
        return "N/A"
    values = np.asarray([float(row[metric_key]) for row in rows], dtype=np.float64)
    return f"{values.mean():.6f} +/- {values.std():.6f}"


def _write_cumulative_subject_summary(rows: List[dict], out_csv_path: str) -> None:
    metric_keys = [
        "mcd",
        "rmse",
        "pesq",
        #"cer_whisper",
        #"cer_gt_whisper",
        "cer_wav2vec_finetuned",
        "cer_gt_wav2vec_finetuned",
        "cer_hubert",
        "cer_gt_hubert",
    ]
    fieldnames = ["metric"] + [row["subject_id"] for row in rows]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric_key in metric_keys:
            writer.writerow(
                {
                    "metric": metric_key,
                    **{row["subject_id"]: row[metric_key] for row in rows},
                }
            )
    print(f"Saved cumulative subject summary: {out_csv_path}")


def _write_subject_metric_statistics(
    metric_rows: List[tuple],
    out_csv_path: str,
) -> None:
    fieldnames = ["metric", "mean", "std", "min", "min_sample", "max", "max_sample"]
    report_rows = []
    for metric_name, rows, metric_key, include_extrema in metric_rows:
        if not rows:
            report_rows.append(
                {
                    "metric": metric_name,
                    "mean": "N/A",
                    "std": "N/A",
                    "min": "N/A",
                    "min_sample": "N/A",
                    "max": "N/A",
                    "max_sample": "N/A",
                }
            )
            continue

        values = np.asarray([float(row[metric_key]) for row in rows], dtype=np.float64)
        report_row = {
            "metric": metric_name,
            "mean": f"{values.mean():.6f}",
            "std": f"{values.std():.6f}",
            "min": "N/A",
            "min_sample": "N/A",
            "max": "N/A",
            "max_sample": "N/A",
        }
        if include_extrema:
            min_index = int(np.argmin(values))
            max_index = int(np.argmax(values))
            report_row.update(
                {
                    "min": f"{values[min_index]:.6f}",
                    "min_sample": rows[min_index]["generated_wav"],
                    "max": f"{values[max_index]:.6f}",
                    "max_sample": rows[max_index]["generated_wav"],
                }
            )
        report_rows.append(report_row)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Saved subject metric statistics: {out_csv_path}")


def mel_cepstral_distortion(mfcc_ref: np.ndarray, mfcc_deg: np.ndarray) -> float:
    """Compute MCD (dB) on pre-aligned MFCC features."""
    K = 10.0 / np.log(10.0)
    min_t = min(mfcc_ref.shape[1], mfcc_deg.shape[1])
    diff = mfcc_ref[1:, :min_t] - mfcc_deg[1:, :min_t]  # exclude c0
    mcd = K * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=0)))
    return float(mcd)


import scipy.fftpack

def logmel_to_mfcc(log_mel: np.ndarray, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Compute MFCCs directly from log-mel spectrogram via Discrete Cosine Transform (DCT-II)."""
    # Type-II DCT along frequency bins (axis 0), taking first n_mfcc coefficients
    mfcc = scipy.fftpack.dct(log_mel, axis=0, type=2, norm="ortho")[:n_mfcc]
    return mfcc.astype(np.float32)


def compute_mcd_paired_rows(
    paired_rows: List[dict],
    n_mfcc: int = N_MFCC,
) -> List[dict]:
    out_rows: List[dict] = []
    for row in paired_rows:
        ref_mel = row["ref_mel_arr"]
        gen_mel = row["gen_mel_arr"]

        # Directly compute MFCCs from log-mels via DCT-II
        ref_mfcc = logmel_to_mfcc(ref_mel, n_mfcc=n_mfcc)
        gen_mfcc = logmel_to_mfcc(gen_mel, n_mfcc=n_mfcc)

        mcd_score = mel_cepstral_distortion(ref_mfcc, gen_mfcc)

        enriched = dict(row)
        enriched["mcd"] = mcd_score
        out_rows.append(enriched)
        print(
            f"[MCD] class={row['class_label']:02d} {row['generated_mel']} vs {row['reference_mel']}: {mcd_score:.4f} dB"
        )
    return out_rows


def compute_and_save_mcd_matrix_from_pairs(
    paired_rows: List[dict],
    out_dir: str,
    n_mfcc: int = N_MFCC,
    word_labels: Dict[int, str] = None,
) -> np.ndarray:
    ordered_rows = sorted(paired_rows, key=lambda r: int(r["class_label"]))
    wav_names = [row["generated_wav"] for row in ordered_rows]
    N = len(ordered_rows)

    mfccs = [
        logmel_to_mfcc(row["gen_mel_arr"], n_mfcc=n_mfcc)
        for row in ordered_rows
    ]

    mcd_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            score = mel_cepstral_distortion(mfccs[i], mfccs[j])
            mcd_matrix[i, j] = score
            mcd_matrix[j, i] = score

    os.makedirs(out_dir, exist_ok=True)
    matrix_npy_path = os.path.join(out_dir, "mcd_pairwise.npy")
    matrix_csv_path = os.path.join(out_dir, "mcd_pairwise.csv")
    matrix_png_path = os.path.join(out_dir, "mcd_pairwise_names.png")

    np.save(matrix_npy_path, mcd_matrix)
    np.savetxt(matrix_csv_path, mcd_matrix, delimiter=",", fmt="%.6f")
    plot_mcd_matrix(mcd_matrix, wav_names, matrix_png_path, word_labels=word_labels)

    return mcd_matrix


def plot_mcd_matrix(dist_matrix: np.ndarray, wav_names: list, out_path: str, word_labels: dict = None):
    N = len(wav_names)
    labels = [word_labels.get(natural_key(n)[0], os.path.splitext(n)[0]) for n in wav_names] if word_labels else [os.path.splitext(n)[0] for n in wav_names]

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


def paired_spectrograms_plot(
    paired_rows: List[dict],
    out_dir: str,
    sr: int = SR,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> None:
    plots_dir = os.path.join(out_dir, "spectrogram_pairs")
    os.makedirs(plots_dir, exist_ok=True)

    for row in paired_rows:
        ref_mel = row["ref_mel_arr"]
        gen_mel = row["gen_mel_arr"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        img0 = librosa.display.specshow(
            ref_mel, sr=sr, x_axis="time", y_axis="mel", fmin=fmin, fmax=fmax, ax=axes[0]
        )
        axes[0].set_title(f"Reference: {row['word']} (Class {row['class_label']})")
        fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

        img1 = librosa.display.specshow(
            gen_mel, sr=sr, x_axis="time", y_axis="mel", fmin=fmin, fmax=fmax, ax=axes[1]
        )
        axes[1].set_title(f"DTW-Warped Gen: {row['word']} (Class {row['class_label']})")
        fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

        plt.tight_layout()
        
        #wav name in output name
        wav_stem = os.path.splitext(row["generated_wav"])[0]
        safe_word = re.sub(r'[\\/*?:"<>|]', "", row["word"])
        out_path = os.path.join(plots_dir, f"{wav_stem}_class{row['class_label']:02d}_{safe_word}.png")
        
        plt.savefig(out_path, dpi=300)
        plt.close(fig)


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

        ref_wav = ref_wav.astype(np.float32)
        gen_wav = gen_wav.astype(np.float32)

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


def compute_rmse_paired_rows(
    paired_rows: List[dict],
    generated_wav_dir: str,
    original_wav_dir: str,
    target_sr: int = 16000,
) -> List[dict]:
    """Waveform RMSE between generated and reference wavs (resampled, truncated to common length)."""
    out_rows: List[dict] = []
    for row in paired_rows:
        gen_wav, gen_sr = librosa.load(os.path.join(generated_wav_dir, row["generated_wav"]), sr=None, mono=True)
        ref_wav, ref_sr = librosa.load(os.path.join(original_wav_dir, row["reference_wav"]), sr=None, mono=True)

        if int(gen_sr) != int(target_sr):
            gen_wav = librosa.resample(gen_wav, orig_sr=int(gen_sr), target_sr=int(target_sr))
        if int(ref_sr) != int(target_sr):
            ref_wav = librosa.resample(ref_wav, orig_sr=int(ref_sr), target_sr=int(target_sr))

        min_len = min(len(ref_wav), len(gen_wav))
        if min_len == 0:
            print(f"[RMSE] Skip class={row['class_label']:02d}: empty audio")
            continue

        diff = ref_wav[:min_len].astype(np.float64) - gen_wav[:min_len].astype(np.float64)
        score = float(np.sqrt(np.mean(diff ** 2)))
        enriched = dict(row)
        enriched["rmse"] = score
        out_rows.append(enriched)
        print(f"[RMSE] class={row['class_label']:02d} {row['generated_wav']} vs {row['reference_wav']}: {score:.6f}")

    if not out_rows:
        raise RuntimeError("No valid RMSE scores were computed")
    return out_rows


def t_whisper(waveform, processor, model, device, sr):
    feats = processor(waveform, sampling_rate=sr, return_tensors="pt").input_features.to(device=device, dtype=model.dtype)
    ids = model.generate(feats, language="en", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def compute_cer_paired_rows(
    paired_rows: List[dict],
    generated_wav_dir: str,
    original_wav_dir: str,
    model_name: str = W2V_MODEL_NAME,
    finetuned_path: str = W2V_FT_PATH,
    target_sr: int = CER_TARGET_SR,
) -> List[dict]:

    source = finetuned_path if finetuned_path and os.path.isdir(finetuned_path) else model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformers_mod = importlib.import_module("transformers")
    AutoModelForCTC = getattr(transformers_mod, "AutoModelForCTC")
    AutoProcessor = getattr(transformers_mod, "AutoProcessor")

    processor = AutoProcessor.from_pretrained(source)

    if model_name == WHISPER_MODEL_NAME:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(source).to(device)
        model.eval()
    else:
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

            if model_name == WHISPER_MODEL_NAME:
                generated_pred_text = t_whisper(generated_waveform, processor, model, device, target_sr)
                reference_pred_text = t_whisper(reference_waveform, processor, model, device, target_sr)
            else:
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


if __name__ == "__main__":
    word_labels = load_word_labels(EVENTS_CSV)
    subject_runs = _resolve_generated_subject_dirs(GENERATED_DIR)
    cumulative_subject_rows: List[dict] = []

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

        # Builds paired rows and aligns log-mel spectrograms using fastdtw before evaluation
        paired_rows = _build_paired_sample_rows(
            generated_wav_dir=generated_wav_dir,
            original_wav_dir=AUDIODATA_DIR,
            generated_mel_dir=generated_mel_dir,
            original_mel_dir=ORIGINAL_MELS,
            word_labels=word_labels,
        )

        paired_spectrograms_plot(
            paired_rows=paired_rows,
            out_dir=output_dir,
            sr=SR,
            fmin=FMIN,
            fmax=FMAX,
        )

        mcd_rows = compute_mcd_paired_rows(
            paired_rows=paired_rows,
            n_mfcc=N_MFCC,
        )
        _write_metric_summary_csv(mcd_rows, "mcd", os.path.join(output_dir, "mcd_summary.csv"))
        
        compute_and_save_mcd_matrix_from_pairs(
            paired_rows=paired_rows,
            out_dir=output_dir,
            n_mfcc=N_MFCC,
            word_labels=word_labels,
        )

        rmse_rows = []
        try:
            rmse_rows = compute_rmse_paired_rows(
                paired_rows=paired_rows,
                generated_wav_dir=generated_wav_dir,
                original_wav_dir=AUDIODATA_DIR,
                target_sr=16000,
            )
            _write_metric_summary_csv(rmse_rows, "rmse", os.path.join(output_dir, "rmse_summary.csv"))
        except Exception as exc:
            print(f"[WARN] RMSE skipped or failed: {exc}")

        pesq_summary_path = os.path.join(output_dir, "pesq_summary.csv")
        pesq_rows = []
        try:
            pesq_rows = compute_pesq_paired_rows(
                paired_rows=paired_rows,
                generated_wav_dir=generated_wav_dir,
                original_wav_dir=AUDIODATA_DIR,
                target_sr=16000,
                mode="wb",
            )
            _write_metric_summary_csv(pesq_rows, "pesq", pesq_summary_path)
        except Exception as exc:
            print(f"[WARN] PESQ skipped or failed: {exc}")

        # CER with Whisper
        # cer_w2v_base_rows = compute_cer_paired_rows(
        #     paired_rows=paired_rows,
        #     generated_wav_dir=generated_wav_dir,
        #     original_wav_dir=AUDIODATA_DIR,
        #     model_name=WHISPER_MODEL_NAME,
        #     finetuned_path=None,
        #     target_sr=CER_TARGET_SR,
        # )
        # _write_metric_summary_csv(
        #     cer_w2v_base_rows,
        #     "cer",
        #     os.path.join(output_dir, "cer_whisper_summary.csv"),
        # )

        # CER with fine-tuned wav2vec
        cer_w2v_ft_summary_path = os.path.join(output_dir, "cer_wav2vec_finetuned_summary.csv")
        cer_w2v_ft_rows = []
        try:
            if os.path.isdir(W2V_FT_PATH):
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
            print(f"[WARN] Fine-tuned wav2vec CER failed: {exc}")

        # CER with HuBERT
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

        _write_subject_metric_statistics(
            [
                ("mcd", mcd_rows, "mcd", True),
                ("rmse", rmse_rows, "rmse", True),
                ("pesq", pesq_rows, "pesq", True),
                #("cer_whisper", cer_w2v_base_rows, "cer", True),
                #("cer_gt_whisper", cer_w2v_base_rows, "cer_gt", False),
                ("cer_wav2vec_finetuned", cer_w2v_ft_rows, "cer", True),
                ("cer_gt_wav2vec_finetuned", cer_w2v_ft_rows, "cer_gt", False),
                ("cer_hubert", cer_hubert_rows, "cer", True),
                ("cer_gt_hubert", cer_hubert_rows, "cer_gt", False),
            ],
            os.path.join(output_dir, "cumulative_metric_statistics.csv"),
        )

        cumulative_subject_rows.append(
            {
                "subject_id": subject_id,
                "mcd": _format_mean_plus_std(mcd_rows, "mcd"),
                "rmse": _format_mean_plus_std(rmse_rows, "rmse"),
                "pesq": _format_mean_plus_std(pesq_rows, "pesq"),
                #"cer_whisper": _format_mean_plus_std(cer_w2v_base_rows, "cer"),
                #"cer_gt_whisper": _format_mean_plus_std(cer_w2v_base_rows, "cer_gt"),
                "cer_wav2vec_finetuned": _format_mean_plus_std(cer_w2v_ft_rows, "cer"),
                "cer_gt_wav2vec_finetuned": _format_mean_plus_std(cer_w2v_ft_rows, "cer_gt"),
                "cer_hubert": _format_mean_plus_std(cer_hubert_rows, "cer"),
                "cer_gt_hubert": _format_mean_plus_std(cer_hubert_rows, "cer_gt"),
            }
        )

    _write_cumulative_subject_summary(
        cumulative_subject_rows,
        os.path.join(GENERATED_DIR, "cumulative_subject_metric_summary.csv"),
    )