import os
import re
import importlib
from typing import Optional, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

import json
import sys
from types import SimpleNamespace

from preprocess.audio_prep import load_processed_wavs, load_word_labels, natural_key


PROJECT_ROOT = os.path.dirname(__file__)
EVENTS_CSV = os.path.join(PROJECT_ROOT, "events_codes.csv")
AUDIODATA_DIR = os.path.join(PROJECT_ROOT, "audiodata", "twos_16000")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "wav2vec2_finetuned")

W2V_MODEL_NAME = "facebook/wav2vec2-base-960h"
hf_TOKEN = ""
#check powershell echo $env:HF_TOKEN

def _set_seed(seed: int, deterministic: bool = True) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _normalize_label_for_ctc(label: str) -> str:
    cleaned = "".join(ch if ("A" <= ch <= "Z") or ch == " " else " " for ch in str(label).upper())
    return " ".join(cleaned.split())


def _build_supervised_pairs(wav_dir: str, events_csv: str) -> List[Tuple[str, str]]:
    labels_by_index = load_word_labels(events_csv)
    wav_names = sorted(
        [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )

    pairs: List[Tuple[str, str]] = []
    for wav_name in wav_names:
        match = re.search(r"(\d+)", wav_name)
        if match is None:
            continue

        idx = int(match.group(1))
        if idx not in labels_by_index:
            continue

        text = _normalize_label_for_ctc(labels_by_index[idx])
        if not text:
            continue
        pairs.append((wav_name, text))
    return pairs


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            sub = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + sub))
        prev = curr
    return prev[-1]


def _cer(reference: str, hypothesis: str) -> float:
    ref = "".join(ch for ch in str(reference).upper() if "A" <= ch <= "Z")
    hyp = "".join(ch for ch in str(hypothesis).upper() if "A" <= ch <= "Z")
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(_levenshtein_distance(ref, hyp)) / float(len(ref))


def pretrain_wav2vec_embeddings(
    wav_dir: str,
    events_csv: str = EVENTS_CSV,
    model_name: str = W2V_MODEL_NAME,
    target_sr: int = 16000,
    device: Optional[str] = None,
    require_cuda: bool = True,
    num_epochs: int = 37,
    batch_size: int = 5,
    learning_rate: float = 1e-5,
    freeze_feature_encoder: bool = True,
    grad_clip_norm: float = 1.0,
    val_ratio: float = 0.2,
    min_val_samples: int = 4,
    output_dir: Optional[str] = None,
    resume: bool = True,
    seed: int = 37,
) -> tuple:
    """Fine-tune wav2vec-base CTC on local WAVs and export mean pooled embeddings.

    Returns:
        wav_names: sorted WAV file names
        embeddings: np.ndarray [N, D]
        model: fine-tuned Hugging Face CTC model
    """

    transformers_mod = importlib.import_module("transformers")
    AutoModelForCTC = getattr(transformers_mod, "AutoModelForCTC")
    AutoProcessor = getattr(transformers_mod, "AutoProcessor")

    wav_names = sorted(
        [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not wav_names:
        raise RuntimeError(f"No WAV files found in {wav_dir}")

    supervised_pairs = _build_supervised_pairs(wav_dir, events_csv)
    if not supervised_pairs:
        raise RuntimeError(
            "No (wav, label) pairs were created. Ensure WAV names include an index "
            "(e.g., audio18.wav) and that events_codes.csv contains matching indices."
        )

    _set_seed(seed)
    cuda_available = torch.cuda.is_available()
    if device is None:
        if require_cuda and not cuda_available:
            raise RuntimeError("CUDA is required for this run, but no GPU was detected.")
        device = "cuda" if cuda_available else "cpu"
    elif str(device).lower().startswith("cuda") and not cuda_available:
        raise RuntimeError("Requested CUDA device but no GPU was detected.")

    print(f"Using device: {device}")

    if output_dir is None:
        output_dir = os.path.join(wav_dir, "wav2vec2_finetuned")
    os.makedirs(output_dir, exist_ok=True)

    resume_model = resume and os.path.exists(os.path.join(output_dir, "config.json"))
    if resume_model:
        processor = AutoProcessor.from_pretrained(output_dir)
        model = AutoModelForCTC.from_pretrained(output_dir).to(device)
        print(f"Resuming model from: {output_dir}")
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCTC.from_pretrained(model_name).to(device)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Loaded processor does not expose a tokenizer for CTC labels.")

    if freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_state_path = os.path.join(output_dir, "trainer_state.pt")
    best_dir = os.path.join(output_dir, "best_by_cer")
    start_epoch = 0
    best_val_cer = float("inf")
    if resume and os.path.exists(train_state_path):
        state = torch.load(train_state_path, map_location="cpu")
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0))
        best_val_cer = float(state.get("best_val_cer", float("inf")))
        print(f"Resuming optimizer/state from epoch {start_epoch}")

    rng = np.random.default_rng(seed=seed)
    shuffled = rng.permutation(len(supervised_pairs))
    val_size = int(round(len(supervised_pairs) * val_ratio))
    val_size = min(len(supervised_pairs) - 1, max(min_val_samples, val_size)) if len(supervised_pairs) > 1 else 0

    if val_size > 0:
        val_pairs = [supervised_pairs[i] for i in shuffled[:val_size]]
        train_pairs = [supervised_pairs[i] for i in shuffled[val_size:]]
    else:
        val_pairs = []
        train_pairs = supervised_pairs

    print(f"Train samples: {len(train_pairs)} | Val samples: {len(val_pairs)}")

    model.train()

    for epoch in range(start_epoch, num_epochs):
        order = rng.permutation(len(train_pairs))
        epoch_loss = 0.0
        seen = 0

        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_wavs = [train_pairs[i][0] for i in batch_indices]
            batch_texts = [train_pairs[i][1] for i in batch_indices]

            waveforms = []
            for wav_name in batch_wavs:
                wav_path = os.path.join(wav_dir, wav_name)
                waveform, wav_sr = librosa.load(wav_path, sr=None, mono=True)
                if int(wav_sr) != int(target_sr):
                    waveform = librosa.resample(
                        waveform,
                        orig_sr=int(wav_sr),
                        target_sr=int(target_sr),
                    )
                waveforms.append(waveform)

            inputs = processor(
                waveforms,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
            )

            label_batch = tokenizer(batch_texts, return_tensors="pt", padding=True)

            labels = label_batch.input_ids.masked_fill(label_batch.attention_mask.ne(1), -100).to(device)
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)

        mean_loss = epoch_loss / max(1, seen)

        val_cer = float("nan")
        if val_pairs:
            model.eval()
            cer_sum = 0.0
            with torch.inference_mode():
                for wav_name, ref_text in val_pairs:
                    wav_path = os.path.join(wav_dir, wav_name)
                    waveform, wav_sr = librosa.load(wav_path, sr=None, mono=True)
                    if int(wav_sr) != int(target_sr):
                        waveform = librosa.resample(waveform, orig_sr=int(wav_sr), target_sr=int(target_sr))

                    val_inputs = processor(
                        waveform,
                        sampling_rate=target_sr,
                        return_tensors="pt",
                        padding=True,
                    )
                    input_values = val_inputs.input_values.to(device)
                    attention_mask = val_inputs.attention_mask.to(device) if "attention_mask" in val_inputs else None

                    logits = model(input_values=input_values, attention_mask=attention_mask).logits
                    pred_ids = torch.argmax(logits, dim=-1)
                    pred_text = processor.batch_decode(pred_ids)[0]
                    cer_sum += _cer(ref_text, pred_text)

            val_cer = cer_sum / float(len(val_pairs))
            model.train()

        if val_pairs:
            print(f"Epoch {epoch + 1}/{num_epochs} | mean CTC loss: {mean_loss:.4f} | val CER: {val_cer:.4f}")
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                model.save_pretrained(best_dir)
                processor.save_pretrained(best_dir)
                print(f"New best val CER: {best_val_cer:.4f} -> saved to {best_dir}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} | mean CTC loss: {mean_loss:.4f}")

        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        torch.save(
            {
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "best_val_cer": best_val_cer,
            },
            train_state_path,
        )

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved fine-tuned model to: {output_dir}")

    embeddings = []
    model.eval()
    with torch.inference_mode():
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

            outputs = model.wav2vec2(input_values=input_values, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state[0]
            emb = hidden.mean(dim=0).cpu().numpy().astype(np.float32)
            embeddings.append(emb)
            print(f"{wav_name}: wav2vec embedding shape {emb.shape}")

    emb_arr = np.stack(embeddings, axis=0)
    out_path = os.path.join(output_dir, "wav2vec_embeddings.npy")
    np.save(out_path, emb_arr)
    print(f"\nSaved wav2vec embeddings: {emb_arr.shape} -> {out_path}")
    return wav_names, emb_arr, model

if __name__ == "__main__":
    # process_and_save(STIMULI_DIR, OUTPUT_DIR)
    wavs, _ = load_processed_wavs(AUDIODATA_DIR)
    word_labels = load_word_labels(EVENTS_CSV)

    wav_names, emb_arr, model = pretrain_wav2vec_embeddings(
        AUDIODATA_DIR,
        events_csv=EVENTS_CSV,
        model_name=W2V_MODEL_NAME,
        target_sr=16000,
        device="cuda",
        require_cuda=True,
        output_dir=OUTPUT_DIR,
        resume=True,
        seed=37,
    )
