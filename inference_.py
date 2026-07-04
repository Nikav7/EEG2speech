import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import wavio

from models import ntGAN as networks
from models.vocoders import Generator as HiFiGANGenerator
from modules import AttrDict
from modules import DTW_align


EPSILON = np.finfo(float).eps


def _read_csv_matrix(file_path: str) -> np.ndarray:
	with open(file_path, "r", newline="") as handle:
		reader = csv.reader(handle)
		rows = [row for row in reader]
	return np.asarray(rows, dtype=np.float32)


def _normalize_array(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
	max_val = float(np.max(data))
	min_val = float(np.min(data))
	avg = (max_val + min_val) / 2.0
	std = max((max_val - min_val) / 2.0, float(EPSILON))
	normalized = ((data - avg) / std).astype(np.float32)
	return normalized, avg, std


def _denorm_mel(mel_norm: torch.Tensor, avg: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
	"""Inverse of dataset min-max normalization used in training."""
	# mel_norm: [B, C, T], avg/std: [B]
	avg = avg.to(mel_norm.device, dtype=mel_norm.dtype).view(-1, 1, 1)
	std = std.to(mel_norm.device, dtype=mel_norm.dtype).view(-1, 1, 1)
	std = torch.where(std == 0, torch.ones_like(std), std)
	return (mel_norm * std) + avg


def _extract_class_code_from_filename(file_name: str) -> int:
	base_name = os.path.basename(file_name)
	match = re.search(r"label(\d+)", base_name, flags=re.IGNORECASE)
	if not match:
		raise ValueError(f"Could not parse class code from file name: {file_name}")
	return int(match.group(1))


def _save_wav_wavio(file_path: str, wav_tensor: torch.Tensor, sample_rate: int) -> None:
	"""Save mono waveform tensor [-1, 1] as 16-bit PCM WAV via wavio."""
	wav_np = wav_tensor.detach().cpu().numpy()
	wav_np = np.clip(wav_np, -1.0, 1.0)
	wav_int16 = (wav_np * 32767.0).astype(np.int16)
	wavio.write(file_path, wav_int16, int(sample_rate), sampwidth=2)


@dataclass
class SampleInfo:
	file_name: str
	class_code: int
	input_tensor: torch.Tensor  # [C, T]
	target_mel_norm: torch.Tensor  # [C, T_target]
	target_avg: float
	target_std: float


def _build_samples(test_eeg_dir: str, audio_mel_dir: str) -> List[SampleInfo]:
	if not os.path.isdir(test_eeg_dir):
		raise FileNotFoundError(f"Test EEG directory not found: {test_eeg_dir}")
	if not os.path.isdir(audio_mel_dir):
		raise FileNotFoundError(f"Audio mel directory not found: {audio_mel_dir}")

	eeg_files = [
		os.path.join(test_eeg_dir, name)
		for name in sorted(os.listdir(test_eeg_dir))
		if name.lower().endswith(".csv")
	]
	if not eeg_files:
		raise RuntimeError(f"No CSV files found in: {test_eeg_dir}")

	samples: List[SampleInfo] = []
	for eeg_file in eeg_files:
		class_code = _extract_class_code_from_filename(eeg_file)
		audio_mel_path = os.path.join(audio_mel_dir, f"audio{class_code}_logmel.csv")
		if not os.path.isfile(audio_mel_path):
			raise FileNotFoundError(
				f"Missing target mel file for class {class_code}: {audio_mel_path}"
			)

		eeg_np = _read_csv_matrix(eeg_file)
		eeg_norm, _, _ = _normalize_array(eeg_np)
		target_mel = _read_csv_matrix(audio_mel_path)
		_, target_avg, target_std = _normalize_array(target_mel)
		target_mel_norm, _, _ = _normalize_array(target_mel)

		samples.append(
			SampleInfo(
				file_name=os.path.basename(eeg_file),
				class_code=class_code,
				input_tensor=torch.tensor(eeg_norm, dtype=torch.float32),
				target_mel_norm=torch.tensor(target_mel_norm, dtype=torch.float32),
				target_avg=float(target_avg),
				target_std=float(target_std),
			)
		)

	return samples


def _extract_model_state_dict(checkpoint_obj):
	if isinstance(checkpoint_obj, dict):
		for key in ("state_dict", "generator", "model"):
			if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
				state_dict = checkpoint_obj[key]
				break
		else:
			state_dict = checkpoint_obj
	else:
		state_dict = checkpoint_obj

	if isinstance(state_dict, dict) and state_dict and all(
		k.startswith("module.") for k in state_dict.keys()
	):
		state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

	return state_dict


def _load_generator(generator_config_path: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
	with open(generator_config_path, "r", encoding="utf-8") as handle:
		h_g = AttrDict(json.load(handle))

	model_g = networks.Generator(h_g).to(device)
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	state_dict = _extract_model_state_dict(checkpoint)
	model_g.load_state_dict(state_dict, strict=True)
	model_g.eval()
	return model_g


def _load_hifigan_vocoder(vocoder_ckpt_path: str, device: torch.device) -> torch.nn.Module:
	vocoder_config_path = os.path.join(os.path.dirname(vocoder_ckpt_path), "config.json")
	if not os.path.isfile(vocoder_config_path):
		raise FileNotFoundError(f"HiFi-GAN config.json not found: {vocoder_config_path}")

	with open(vocoder_config_path, "r", encoding="utf-8") as handle:
		h_v = AttrDict(json.load(handle))

	vocoder = HiFiGANGenerator(h_v).to(device)
	state = torch.load(vocoder_ckpt_path, map_location="cpu")
	if "generator" not in state:
		raise KeyError(f"Invalid HiFi-GAN checkpoint format, missing 'generator': {vocoder_ckpt_path}")
	vocoder.load_state_dict(state["generator"], strict=True)
	vocoder.eval()
	return vocoder


class GriffinLimVocoderMs(torch.nn.Module):
	"""Griffin-Lim vocoder with window/hop specified in milliseconds."""

	def __init__(self, sample_rate: int, n_mels: int, win_ms: float, hop_ms: float, n_iter: int = 32):
		super().__init__()
		win_length = max(2, int(round(float(win_ms) * float(sample_rate) / 1000.0)))
		hop_length = max(1, int(round(float(hop_ms) * float(sample_rate) / 1000.0)))
		n_fft = win_length

		n_stft = (n_fft // 2) + 1
		self.inv_mel = torchaudio.transforms.InverseMelScale(
			n_stft=n_stft,
			n_mels=int(n_mels),
			sample_rate=int(sample_rate),
		)
		self.griffin = torchaudio.transforms.GriffinLim(
			n_fft=n_fft,
			n_iter=int(n_iter),
			win_length=win_length,
			hop_length=hop_length,
			power=2.0,
		)

	def forward(self, mel: torch.Tensor) -> torch.Tensor:
		mel_in = mel.float()
		mel_linear = torchaudio.functional.DB_to_amplitude(mel_in, ref=1.0, power=1.0).clamp_min(1e-8)
		spec = self.inv_mel(mel_linear).clamp_min(1e-8)
		wav = self.griffin(spec)
		return wav.unsqueeze(1)


def _load_griffinlim_vocoder(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
	with open(args.generator_config, "r", encoding="utf-8") as handle:
		h_g = AttrDict(json.load(handle))
	n_mels = int(getattr(h_g, "out_ch", 80))
	vocoder = GriffinLimVocoderMs(
		sample_rate=int(args.sample_rate),
		n_mels=n_mels,
		win_ms=float(args.win_ms),
		hop_ms=float(args.hop_ms),
		n_iter=int(args.griffinlim_n_iter),
	).to(device)
	vocoder.eval()
	return vocoder


def run_inference(args: argparse.Namespace) -> None:
	device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		torch.cuda.set_device(device)

	print(f"Using device: {device}")
	print(f"Generator checkpoint: {args.generator_checkpoint}")
	print(f"Test EEG CSV folder: {args.test_eeg_dir}")
	print(f"Vocoder type: {args.vocoder_type}")
	if args.vocoder_type == "hifigan":
		print(f"Vocoder checkpoint (UNIVERSAL_V1): {args.vocoder_checkpoint}")
	else:
		print(f"Griffin-Lim params: WIN_MS={args.win_ms}, HOP_MS={args.hop_ms}, n_iter={args.griffinlim_n_iter}")

	model_g = _load_generator(args.generator_config, args.generator_checkpoint, device)
	if args.vocoder_type == "griffinlim":
		vocoder = _load_griffinlim_vocoder(args, device)
	else:
		vocoder = _load_hifigan_vocoder(args.vocoder_checkpoint, device)
	samples = _build_samples(args.test_eeg_dir, args.audio_mel_dir)

	os.makedirs(args.output_dir, exist_ok=True)
	mel_out_dir = os.path.join(args.output_dir, "mel_csv")
	wav_out_dir = args.wav_output_dir
	os.makedirs(mel_out_dir, exist_ok=True)
	os.makedirs(wav_out_dir, exist_ok=True)

	with torch.no_grad():
		for idx, sample in enumerate(samples):
			x = sample.input_tensor.unsqueeze(0).to(device)  # [1, C, T]
			pred_mel_norm = model_g(x)
			target_mel_norm = sample.target_mel_norm.unsqueeze(0).to(device)
			pred_mel_norm = DTW_align(pred_mel_norm, target_mel_norm)
			pred_mel = _denorm_mel(
				pred_mel_norm,
				avg=torch.tensor([sample.target_avg], dtype=torch.float32),
				std=torch.tensor([sample.target_std], dtype=torch.float32),
			)

			wav = vocoder(pred_mel)
			wav = torch.clamp(wav, min=-1.0, max=1.0)
			wav_1d = wav.squeeze(0).squeeze(0)

			# Improve audibility while avoiding clipping.
			if bool(args.normalize_audio):
				peak = torch.max(torch.abs(wav_1d))
				if torch.isfinite(peak) and peak.item() > 1e-6:
					wav_1d = (wav_1d / peak) * float(args.peak_level)

			# Force fixed target duration (default 2.0s).
			target_samples = int(round(float(args.target_duration_s) * int(args.sample_rate)))
			if wav_1d.numel() < target_samples:
				wav_1d = F.pad(wav_1d, (0, target_samples - wav_1d.numel()))
			elif wav_1d.numel() > target_samples:
				wav_1d = wav_1d[:target_samples]

			wav_1d = wav_1d.detach().cpu()

			stem = os.path.splitext(sample.file_name)[0].replace("epoch", "trial")
			display_name = sample.file_name.replace("epoch", "trial")
			mel_np = pred_mel.squeeze(0).detach().cpu().numpy()
			np.savetxt(os.path.join(mel_out_dir, f"{stem}_pred_mel.csv"), mel_np, delimiter=",")
			_save_wav_wavio(
				os.path.join(wav_out_dir, f"{stem}_pred.wav"),
				wav_1d,
				int(args.sample_rate),
			)

			print(
				f"[{idx + 1:03d}/{len(samples):03d}] {display_name} -> "
				f"class={sample.class_code} mel_frames={pred_mel.shape[-1]} wav_sec={wav_1d.numel()/int(args.sample_rate):.3f} saved mel+wav"
			)

	print(f"Inference finished. Mel outputs saved under: {mel_out_dir}")
	print(f"Inference finished. Wav outputs saved under: {wav_out_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="EEG-to-Speech inference with generator + HiFi-GAN vocoder")
	project_root = os.path.dirname(os.path.abspath(__file__))
	parser.add_argument(
		"--generator-checkpoint",
		type=str,
		default=r"C:\Users\hssn_\Desktop\EEG2speech\TrainResult22kHz_540\subj18\imagined_speech\savemodel\BEST_checkpoint_g.pt",
		help="Path to trained generator checkpoint",
	)
	parser.add_argument(
		"--generator-config",
		type=str,
		default=r"C:\Users\hssn_\Desktop\EEG2speech\models\config_G.json",
		help="Path to generator config JSON",
	)
	parser.add_argument(
		"--test-eeg-dir",
		type=str,
		default=r"C:\Users\hssn_\Desktop\EEG2speech\eegdata\subj18\imagined_speech\test",
		help="Folder with test EEG CSV files",
	)
	parser.add_argument(
		"--audio-mel-dir",
		type=str,
		default=r"C:\Users\hssn_\Desktop\EEG2speech\audiodata\logmel22",
		help="Folder with audioN_logmel.csv templates used for denormalization",
	)
	parser.add_argument(
		"--vocoder-type",
		type=str,
		default="griffinlim",
		choices=["hifigan", "griffinlim"],
		help="Vocoder backend to synthesize waveform from mel",
	)
	parser.add_argument(
		"--vocoder-checkpoint",
		type=str,
		default=r"C:\Users\hssn_\Desktop\EEG2speech\UNIVERSAL_V1\g_02500000",
		help="HiFi-GAN UNIVERSAL_V1 checkpoint path",
	)
	parser.add_argument(
		"--sample-rate",
		type=int,
		default=22050,
		help="Output waveform sample rate (UNIVERSAL_V1 default is 22050)",
	)
	parser.add_argument(
		"--win-ms",
		type=float,
		default=250.0,
		help="Window size (ms) for Griffin-Lim vocoding",
	)
	parser.add_argument(
		"--hop-ms",
		type=float,
		default=25.0,
		help="Hop size (ms) for Griffin-Lim vocoding",
	)
	parser.add_argument(
		"--griffinlim-n-iter",
		type=int,
		default=32,
		help="Number of Griffin-Lim iterations",
	)
	parser.add_argument(
		"--target-duration-s",
		type=float,
		default=2.0,
		help="output waveform duration in seconds",
	)
	parser.add_argument(
		"--normalize-audio",
		type=int,
		choices=[0, 1],
		default=1,
		help="Peak-normalize output audio for audibility",
	)
	parser.add_argument(
		"--peak-level",
		type=float,
		default=0.98,
		help="Peak amplitude used when normalize-audio=1",
	)
	parser.add_argument(
		"--gpu",
		type=int,
		default=0,
		help="CUDA GPU index if CUDA is available",
	)
	parser.add_argument(
		"--wav-output-dir",
		type=str,
		default=os.path.join(project_root, "inference22kHz"),
		help="Directory to save generated WAV outputs",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=r"C:\Users\hssn_\Desktop\EEG2speech\TrainResult22kHz_540\subj18\imagined_speech\inference_best_g",
		help="Directory to save predicted mel CSV outputs",
	)
	return parser.parse_args()


if __name__ == "__main__":
	run_inference(parse_args())
