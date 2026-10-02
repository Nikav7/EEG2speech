import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import glob
import re
import librosa
from torch.nn.utils.parametrizations import weight_norm

def audio_denorm(data):
    max_audio = 32768.0
    
    data = np.array(data * max_audio).astype(np.float32)
       
    return data


def data_denorm(data, avg, std):
    
    std = std.type(torch.cuda.FloatTensor)
    avg = avg.type(torch.cuda.FloatTensor)
    
    # if std == 0, change to 1.0 for nothing happen
    std = torch.where(std==torch.tensor(0,dtype=torch.float32).cuda(), torch.tensor(1,dtype=torch.float32).cuda(), std)
 
    # change the size of std and avg
    std = torch.permute(std.repeat(data.shape[1],data.shape[2],1),[2,0,1])
    avg = torch.permute(avg.repeat(data.shape[1],data.shape[2],1),[2,0,1])
    
    data = torch.mul(data, std) + avg
       
    return data


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()

    return fig
    
def imgSave(dir, file_name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.tight_layout()
    plt.savefig(dir + file_name)
    plt.clf()


def word_index(word_label, bundle):
    labels_ = ''.join(list(bundle.get_labels()))
    word_indices = np.zeros((len(word_label), 15), dtype=np.int64)
    word_length = np.zeros((len(word_label), ), dtype=np.int64)
    for w in range(len(word_label)):
        word = word_label[w]
        label_idx = []
        for ww in range(len(word)):
            label_idx.append(labels_.find(word[ww]))
        word_indices[w,:len(label_idx)] = torch.tensor(label_idx)
        word_length[w] = len(label_idx)
        
    return word_indices, word_length


######################################################################
############                  HiFiGAN                   ##############
######################################################################
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)



####################

def natural_key(name):
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)), name) if m else (10**9, name)

def load_wavs(wav_dir: str, sr: int):
    wav_names = sorted(
        [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")],
        key=natural_key,
    )
    if not wav_names:
        raise RuntimeError(f"No WAV files found in {wav_dir}")
    waveforms = []
    for wav_name in wav_names:
        waveform, _ = librosa.load(os.path.join(wav_dir, wav_name), sr=sr, mono=True)
        waveforms.append(waveform)
        print(f"Loaded: {wav_name} ({len(waveform)/sr:.2f}s)")
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

