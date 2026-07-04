import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import json
import sys
from types import SimpleNamespace


STIMULI_DIR = r"C:\Users\lalli\Desktop\Thesis\data\TriParadigm\stimuli"
EVENTS_CSV = os.path.join(os.path.dirname(__file__), "..", "events_codes.csv")
OUTPUT_DIR = os.path.join(STIMULI_DIR, "twos_16000")
AUDIODATA_DIR = os.path.join(os.path.dirname(__file__), "..", "audiodata")


SR = 24000
TARGET_SR = 16000
TARGET_DURATION_S = 2.0
PREPEND_SILENCE_S = 0.1

#WIN_MS = 250.0 not used, windowing done as in hi-fiGAN train data, 1024 for the fft, see def compute_mel_spectrograms
#HOP_MS = 25.0 not used, windowing done as in hi-fiGAN train data, 256 hop for the fft, see def compute_mel_spectrograms
N_MELS = 80
FMIN = 20.0
FMAX = 8000.0
N_MFCC = 40
DEFAULT_ASR_MODELS = {
    "wav2vec": "facebook/wav2vec2-base-960h",
    "hubert": "facebook/hubert-large-ls960-ft",
}



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


def _normalize_event_text_to_ctc_label(text: str) -> str:
    text = text.strip().strip("\"'").upper()
    cleaned = "".join(ch if ("A" <= ch <= "Z") or ch == " " else " " for ch in text)
    cleaned = " ".join(cleaned.split())
    return cleaned.replace(" ", "|") + "|" if cleaned else ""


def _character_error_rate(reference: str, hypothesis: str) -> float:
    reference_chars = list(str(reference).replace("|", "").strip().upper())
    hypothesis_chars = list(str(hypothesis).replace("|", "").strip().upper())

    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0

    previous_row = list(range(len(hypothesis_chars) + 1))
    for i, ref_char in enumerate(reference_chars, start=1):
        current_row = [i]
        for j, hyp_char in enumerate(hypothesis_chars, start=1):
            substitution_cost = 0 if ref_char == hyp_char else 1
            current_row.append(
                min(
                    previous_row[j] + 1,
                    current_row[j - 1] + 1,
                    previous_row[j - 1] + substitution_cost,
                )
            )
        previous_row = current_row

    return float(previous_row[-1]) / float(max(1, len(reference_chars)))


def _collapse_consecutive_repetitions(text: str) -> str:
    if not text:
        return text

    collapsed = [text[0]]
    for char in text[1:]:
        if char != collapsed[-1]:
            collapsed.append(char)
    return "".join(collapsed)


def _normalize_for_lexicon_match(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if "A" <= ch <= "Z")


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            substitution_cost = 0 if char_a == char_b else 1
            current_row.append(
                min(
                    previous_row[j] + 1,
                    current_row[j - 1] + 1,
                    previous_row[j - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def _lexicon_correct_prediction(
    prediction: str,
    candidate_words: List[str],
    max_normalized_distance: float = 0.35,
) -> Tuple[str, Optional[str], float]:
    normalized_prediction = _normalize_for_lexicon_match(prediction)
    if not normalized_prediction or not candidate_words:
        return prediction, None, float("inf")

    best_word = None
    best_distance = float("inf")
    best_score = float("inf")

    for candidate in candidate_words:
        normalized_candidate = _normalize_for_lexicon_match(candidate)
        if not normalized_candidate:
            continue
        distance = _edit_distance(normalized_prediction, normalized_candidate)
        score = float(distance) / float(max(1, max(len(normalized_prediction), len(normalized_candidate))))
        if score < best_score:
            best_score = score
            best_distance = float(distance)
            best_word = candidate

    if best_word is None or best_score > max_normalized_distance:
        return prediction, None, best_score

    return best_word, best_word, best_score


def compute_wav2vec_ctc_metrics(
    wav_dir: str,
    events_csv: str = EVENTS_CSV,
    model_name: str = "facebook/hubert-large-ls960-ft",
    target_sr: int = TARGET_SR,
    device: Optional[str] = None,
    output_csv: Optional[str] = None,
    enable_lexicon_correction: bool = True,
    lexicon_max_normalized_distance: float = 0.35,
) -> Dict[str, object]:
    """Compute wav2vec CTC loss and CER for processed WAV files in ``wav_dir``.

    The function pairs each ``audioN.wav`` file with the corresponding label from
    ``events_codes.csv``, encodes the audio with a Hugging Face CTC ASR model,
    returns aggregated metrics plus per-file rows.
    """
    try:
        from transformers import AutoModelForCTC, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers is required to compute wav2vec CTC metrics"
        ) from exc

    wav_names = sorted(
        [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not wav_names:
        raise RuntimeError(f"No WAV files found in {wav_dir}")

    labels_by_index = load_word_labels(events_csv)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name).to(device)
    model.eval()
    criterion_ctc = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)

    total_loss = 0.0
    total_cer = 0.0
    evaluated = 0
    rows = []
    candidate_words = list(dict.fromkeys(labels_by_index.values()))

    with torch.inference_mode():
        for wav_name in wav_names:
            match = re.search(r"(\d+)", wav_name)
            if match is None:
                continue

            audio_index = int(match.group(1))
            if audio_index not in labels_by_index:
                continue

            label_text = _normalize_event_text_to_ctc_label(labels_by_index[audio_index])
            if not label_text:
                continue

            waveform, wav_sr = librosa.load(os.path.join(wav_dir, wav_name), sr=None, mono=True)
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
            logits = outputs.logits
            log_probs = logits.log_softmax(dim=-1)

            target_ids = processor.tokenizer(label_text, add_special_tokens=False).input_ids
            if not target_ids:
                continue

            targets = torch.tensor(target_ids, dtype=torch.long, device=device)
            input_lengths = torch.full((1,), log_probs.shape[1], dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(target_ids)], dtype=torch.long, device=device)

            ctc_loss = criterion_ctc(
                log_probs.transpose(0, 1),
                targets,
                input_lengths,
                target_lengths,
            )

            predicted_ids = torch.argmax(logits, dim=-1)
            raw_predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            # collapsed_text = _collapse_consecutive_repetitions(raw_predicted_text)
            # corrected_text = raw_predicted_text
            # corrected_by_lexicon = None
            # lexicon_score = float("inf")
            # if enable_lexicon_correction:
            #     corrected_text, corrected_by_lexicon, lexicon_score = _lexicon_correct_prediction(
            #         corrected_text,
            #         candidate_words,
            #         max_normalized_distance=lexicon_max_normalized_distance,
            #     )
            cer = _character_error_rate(label_text, raw_predicted_text)

            total_loss += float(ctc_loss.item())
            total_cer += float(cer)
            evaluated += 1

            rows.append(
                {
                    "wav_name": wav_name,
                    "audio_index": audio_index,
                    "label": label_text,
                    "raw_prediction": raw_predicted_text,
                    "ctc_loss": float(ctc_loss.item()),
                    "cer": float(cer),
                }
            )
            print(f"{wav_name}: ctc_loss={float(ctc_loss.item()):.4f} cer={float(cer):.4f}")

    if evaluated == 0:
        raise RuntimeError(f"No WAV files in {wav_dir} matched labels from {events_csv}")

    result = {
        "ctc_loss": total_loss / evaluated,
        "cer": total_cer / evaluated,
        "evaluated": evaluated,
        "rows": rows,
    }

    if output_csv:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"Saved CTC metrics to: {output_csv}")

    print(f"\nCTC loss: {result['ctc_loss']:.4f} | CER: {result['cer']:.4f} | evaluated: {evaluated}")
    return result


def process_and_save(stimuli_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    wav_names = sorted(
        [f for f in os.listdir(stimuli_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not wav_names:
        raise RuntimeError(f"No WAV files found in {stimuli_dir}")

    expected_samples = int(round(TARGET_DURATION_S * TARGET_SR))
    prepend_samples = int(round(PREPEND_SILENCE_S * TARGET_SR))
    content_samples = expected_samples - prepend_samples

    for wav_name in wav_names:
        waveform, orig_sr = librosa.load(os.path.join(stimuli_dir, wav_name), sr=None, mono=True)

        if int(orig_sr) != SR:
            waveform = librosa.resample(waveform, orig_sr=int(orig_sr), target_sr=TARGET_SR)

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
        sf.write(os.path.join(output_dir, wav_name), out, TARGET_SR)
        print(f"Saved: {wav_name} ({len(out)/TARGET_SR:.2f}s)")

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
        waveform, _ = librosa.load(os.path.join(wav_dir, wav_name), sr=TARGET_SR, mono=True)
        waveforms.append(waveform)
        print(f"Loaded: {wav_name} ({len(waveform)/TARGET_SR:.2f}s)")
    print(f"\nLoaded {len(waveforms)} files from: {wav_dir}")
    return wav_names, waveforms


def compute_mel_spectrograms(wav_dir: str):
    #win_length = int(round(WIN_MS / 1000.0 * TARGET_SR))
    #hop_length = int(round(HOP_MS / 1000.0 * TARGET_SR))

    wav_names = sorted(
        [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not wav_names:
        raise RuntimeError(f"No WAV files found in {wav_dir}")

    mel_list = []
    for wav_name in wav_names:
        waveform, _ = librosa.load(os.path.join(wav_dir, wav_name), sr=TARGET_SR, mono=True)
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=TARGET_SR,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window="hann",
            center=False,
            power=2.0,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
        )
        log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)
        mel_list.append(log_mel)
        print(f"{wav_name}: mel shape {log_mel.shape}")

    mel_arr = np.stack(mel_list, axis=0)  # [N, N_MELS, T]
    out_path = os.path.join(wav_dir, "log_mel.npy")
    np.save(out_path, mel_arr)
    print(f"\nSaved mel spectrograms: {mel_arr.shape} -> {out_path}")
    return mel_arr



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

def vocode_grifflim(output_dir: str):
    mel_arr = pd.read_csv(os.path.join(AUDIODATA_DIR, "audio1_logmel.csv"), header=None).to_numpy(dtype=np.float32)
    mel_power = librosa.db_to_power(mel_arr, ref=1.0)
    wav = librosa.feature.inverse.mel_to_audio(
        M=mel_power,
        sr=TARGET_SR,
        n_fft=1024,
        hop_length=512,
        win_length=1024,
        window="hann",
        center=False,
        pad_mode="constant",
        power=2.0,
        n_iter=64,
        fmin=FMIN,
        fmax=FMAX,
    ).astype(np.float32)
    sf.write(os.path.join(output_dir, "check_griffinlim1.wav"), wav / (np.abs(wav).max() + 1e-9), TARGET_SR)

def vocode_hifi(output_dir: str):
    
    # [N, N_MELS, T]
    mel_arr = pd.read_csv("audiodata/audio1_logmel.csv", header=None).to_numpy(dtype=np.float32)
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path: sys.path.insert(0, _root)
    from models.vocoders import Generator as _HiFiG
    
    _hcfg = SimpleNamespace(**json.load(open(os.path.join(_root, "UNIVERSAL_V1", "config.json"))))
    _hg = _HiFiG(_hcfg).eval()
    _ckpt = torch.load(os.path.join(_root, "UNIVERSAL_V1", "g_02500000"), map_location="cpu")
    _hg.load_state_dict(_ckpt["generator"] if isinstance(_ckpt, dict) and "generator" in _ckpt else _ckpt)
    with torch.no_grad():
        _wav_out = _hg(torch.from_numpy(mel_arr).float().unsqueeze(0)).squeeze().numpy()
    sf.write(os.path.join(output_dir, "check_hifi1.wav"), _wav_out / (np.abs(_wav_out).max() + 1e-9), 22050)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare processed audio features and optional wav2vec CTC metrics.")
    parser.add_argument("--wav-dir", default=OUTPUT_DIR, help="Directory containing processed WAV files.")
    parser.add_argument("--audio-data-dir", default=AUDIODATA_DIR, help="Directory for saved mel CSV files.")
    parser.add_argument("--asr-backend", choices=["wav2vec", "hubert"], default="hubert", help="ASR backend family used for CTC scoring.")
    parser.add_argument("--ctc-model", default=None, help="Optional Hugging Face CTC ASR model name. If omitted, uses default for --asr-backend.")
    parser.add_argument("--ctc-output-csv", default=None, help="Optional CSV path for per-file CTC metrics.")
    args = parser.parse_args()

    selected_model = args.ctc_model if args.ctc_model else DEFAULT_ASR_MODELS[args.asr_backend]
    if args.ctc_output_csv is None:
        args.ctc_output_csv = os.path.join(args.audio_data_dir, f"{args.asr_backend}_ctc_metrics.csv")

    # wav_names_, _ = process_and_save(STIMULI_DIR, OUTPUT_DIR)
    # wav_names, _ = load_processed_wavs(args.wav_dir)
    # mels = compute_mel_spectrograms(args.wav_dir)
    # os.makedirs(args.audio_data_dir, exist_ok=True)
    # [pd.DataFrame(mels[i]).to_csv(os.path.join(args.audio_data_dir, f"{os.path.splitext(wav_names[i])[0]}_logmel.csv"), index=False, header=False) for i in range(len(wav_names))]

    
    compute_wav2vec_ctc_metrics(
        wav_dir=args.wav_dir,
        events_csv=EVENTS_CSV,
        model_name=selected_model,
        target_sr=TARGET_SR,
        output_csv=args.ctc_output_csv,
    )

    # mel_csv_arr = pd.read_csv(os.path.join(AUDIODATA_DIR, f"{os.path.splitext(wav_names[0])[0]}_logmel.csv")).to_numpy(dtype=np.float32)
    # mfccs = compute_mfccs(mels)
    # vocode_grifflim(OUTPUT_DIR)
