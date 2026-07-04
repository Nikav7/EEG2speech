import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import List, Tuple, Dict
import csv

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch.utils.data import DataLoader
import soundfile as sf

from models import ntGAN
from models.eeg2melTransformer import TransformerMelSynth
from previous_scripts.dataset import EEGAudioDataset
from previous_scripts.utils import load_vocoder, mel_to_vocoder_input


test_path = os.path.join("eegdata")
audio_path = os.path.join("audiodata")
ckp = os.path.join("trainoutput_trans_riem_ctc205", "checkpoint_epoch_550.pt")

def _normalize_event_text_to_ctc_label(text: str) -> str:
    text = text.strip().strip("\"'").upper()
    cleaned = "".join(ch if ("A" <= ch <= "Z") or ch == " " else " " for ch in text)
    cleaned = " ".join(cleaned.split())
    return cleaned.replace(" ", "|") + "|" if cleaned else ""

def load_word_labels_from_events_csv(events_csv_path: str) -> List[str]:
    if not os.path.exists(events_csv_path):
        raise FileNotFoundError(f"events_codes CSV not found: {events_csv_path}")

    labels_by_code: Dict[int, str] = {}
    with open(events_csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                code = int(str(row[1]).strip())
            except ValueError:
                continue
            if code <= 0:
                continue
            label = _normalize_event_text_to_ctc_label(row[0])
            if label:
                labels_by_code[code] = label

    if not labels_by_code:
        raise ValueError(f"No valid event labels found in: {events_csv_path}")

    return [labels_by_code[code] for code in sorted(labels_by_code)]


def build_hparams(args, eeg_shape: Tuple[int, int, int], audio_shape: Tuple[int, int, int], n_classes: int):
    """
    Build hyperparameters for generator and discriminator.
    
    Args:
        args: Command line arguments
        eeg_shape: (channels, freq_bins, time_steps) or (1, freq_bins, time_steps)
        audio_shape: (channels, freq_bins, time_steps)
    """
    eeg_channels = int(eeg_shape[0])
    eeg_freq_bins = int(eeg_shape[1])
    audio_channels = int(audio_shape[0])
    audio_freq_bins = int(audio_shape[1])
    audio_time_steps = int(audio_shape[2])

    h_g = SimpleNamespace(
        in_ch=eeg_channels * eeg_freq_bins,
        out_ch=audio_channels * audio_freq_bins,
        out_freq_bins=audio_freq_bins,
        out_spec_channels=audio_channels,
        ch_init_upsample=args.g_ch_init,
        upsample_rates=args.g_upsample_rates,
        upsample_kernel_sizes=args.g_upsample_kernel_sizes,
        resblock_kernel_sizes=args.resblock_kernel_sizes,
        resblock_dilation_sizes=args.resblock_dilation_sizes,
    )

    h_d = SimpleNamespace(
        in_ch=audio_channels * audio_freq_bins,
        ch_init_downsample=args.d_ch_init,
        downsample_rates=args.d_downsample_rates,
        downsample_kernel_sizes=args.d_downsample_kernel_sizes,
        resblock_kernel_sizes=args.resblock_kernel_sizes,
        resblock_dilation_sizes=args.resblock_dilation_sizes,
        n_classes=int(n_classes),
        input_size=audio_time_steps,
    )

    transformer = SimpleNamespace(
        input_channels=eeg_channels,
        input_freq_bins=eeg_freq_bins,
        output_mel_bins=audio_freq_bins,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_dim_feedforward=512,
        transformer_num_layers=6,
        dropout=0.1,
    )
    return h_g, h_d, transformer


def _extract_model_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("model", "generator", "state_dict"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                state_dict = checkpoint_obj[key]
                break
        else:
            state_dict = checkpoint_obj
    else:
        state_dict = checkpoint_obj

    if isinstance(state_dict, dict) and state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    return state_dict




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load word labels from events CSV
    word_labels = load_word_labels_from_events_csv("events_codes.csv")
    print(f"Loaded {len(word_labels)} word labels from events_codes.csv")
    
    # Load test dataset
    test_dataset = EEGAudioDataset(
        eeg_csp_path=test_path,
        audio_mel_path=audio_path,
        speech_type="attempted_speech",
        split="test",
        subjects=["16"]
    )
    
    # shapes
    sample_eeg, sample_audio, _ = test_dataset[0]
    eeg_shape = tuple(sample_eeg.shape)
    audio_shape = tuple(sample_audio.shape)
    print(f"EEG shape: {eeg_shape}, Audio shape: {audio_shape}")
    
    # args for vocoder and model
    args = SimpleNamespace(
        vocoder_method="hifigan",
        vocoder_target_mels=80,
        vocoder_config=os.path.join("UNIVERSAL_V1", "config.json"),
        vocoder_pretrained=os.path.join("UNIVERSAL_V1", "g_02500000"),
        sample_rate_mel=None,
        # Generator 
        g_ch_init=512,
        g_upsample_rates=[1, 1, 1],
        g_upsample_kernel_sizes=[3, 3, 3],
        # Discriminator
        d_ch_init=32,
        d_downsample_rates=[3, 3, 3],
        d_downsample_kernel_sizes=[6, 6, 6],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )
    
    #model hparams
    n_classes = 74
    h_g, h_d, transformer_hparams = build_hparams(args, eeg_shape, audio_shape, n_classes)
    
    #generator
    generator = TransformerMelSynth(**vars(transformer_hparams)).to(device)
    #generator = ntGAN.Generator(h_g).to(device)
    checkpoint = torch.load(ckp, map_location=device)
    model_state = _extract_model_state_dict(checkpoint)
    generator.load_state_dict(model_state)
    generator.eval()
    print(f"Loaded checkpoint: {ckp}")
    
    #vocoder
    vocoder, sample_rate_mel = load_vocoder(args, device)
    print(f"Vocoder ready (sample_rate={sample_rate_mel})")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    
    # INFERENCE
    output_dir = "inference_outputs_trans"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (eeg, audio, metadata) in enumerate(test_loader):
            eeg = eeg.to(device)
            
            # Generate mel spectrogram
            pred = generator(eeg)  # [B, C, F, T]
            
            # Prepare for vocoder
            mel_input = mel_to_vocoder_input(pred, target_mels=args.vocoder_target_mels)
            
            # Vocod to waveform
            wav = vocoder(mel_input)  # [B, 1, samples]
            
            # Save wav files
            for i in range(wav.shape[0]):
                sample_idx = batch_idx * test_loader.batch_size + i
                wav_data = wav[i, 0].cpu().numpy()
                out_path = os.path.join(output_dir, f"output_{sample_idx:04d}.wav")
                sf.write(out_path, wav_data, sample_rate_mel)
            
            print(f"Batch {batch_idx + 1}/{len(test_loader)} complete")
    
    print(f"Inference complete. Outputs saved to {output_dir}")

    
