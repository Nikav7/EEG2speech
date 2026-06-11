import os

# TensorBoard loads TensorFlow internally; keep startup logs quiet and deterministic.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import torch
from models import ntGAN as networks
from models.vocoders import Generator as model_HiFi
from models.vocoders import GriffinLimVocoder, TorchaudioHiFiGAN16k
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss
from modules import mel2wav_vocoder, perform_STT
from utils import data_denorm, word_index
import torch.nn as nn
import torch.nn.functional as F
from dataset import myDataset

import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics.text import CharErrorRate
import json
import argparse
import csv
import wavio
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(state, is_best, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(save_dir, f"BEST_{filename}")
        torch.save(state, best_path)


METRICS_CSV_COLUMNS = [
    "epoch",
    "lr_g",
    "lr_d",
    "train_loss_g",
    "train_loss_g_recon",
    "train_loss_g_valid",
    "train_loss_g_ctc",
    "train_acc_g_valid",
    "train_cer_gt",
    "train_cer_recon",
    "train_loss_d",
    "train_acc_d_real",
    "train_acc_d_fake",
    "val_loss_g",
    "val_loss_g_recon",
    "val_loss_g_valid",
    "val_loss_g_ctc",
    "val_acc_g_valid",
    "val_cer_gt",
    "val_cer_recon",
    "val_loss_d",
    "val_acc_d_real",
    "val_acc_d_fake",
    "best_val_loss_so_far",
    "is_best",
    "epoch_time_sec",
]


def _ensure_metrics_csv(metrics_csv_path):
    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    should_write_header = (not os.path.exists(metrics_csv_path)) or os.path.getsize(metrics_csv_path) == 0
    if should_write_header:
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(METRICS_CSV_COLUMNS)


def _append_metrics_csv(metrics_csv_path, row_dict):
    with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
        writer.writerow(row_dict)


def _safe_column_mean(array_like, col_idx):
    arr = np.array(array_like)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr[:, col_idx]))


def _assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        total = tensor.numel()
        finite = torch.isfinite(tensor).sum().item()
        raise RuntimeError(f"Non-finite values detected in {name}: finite={finite}/{total}")


def _validate_ctc_inputs(emission, targets, input_lengths, target_lengths):
    """Validate CTC tensor shapes/ranges on CPU before launching CUDA kernels."""
    if emission.ndim != 3:
        raise RuntimeError(f"CTC emission must be 3D [B, T, C], got shape={tuple(emission.shape)}")

    bsz, t_steps, n_classes = int(emission.shape[0]), int(emission.shape[1]), int(emission.shape[2])

    if targets.ndim != 2:
        raise RuntimeError(f"CTC targets must be 2D [B, S], got shape={tuple(targets.shape)}")
    if int(targets.shape[0]) != bsz:
        raise RuntimeError(f"CTC batch mismatch: emission B={bsz}, targets B={int(targets.shape[0])}")

    if input_lengths.numel() != bsz or target_lengths.numel() != bsz:
        raise RuntimeError(
            f"CTC length size mismatch: input_lengths={int(input_lengths.numel())}, "
            f"target_lengths={int(target_lengths.numel())}, batch={bsz}"
        )

    input_lengths_cpu = input_lengths.detach().to("cpu")
    target_lengths_cpu = target_lengths.detach().to("cpu")
    targets_cpu = targets.detach().to("cpu")

    if int(torch.min(input_lengths_cpu).item()) <= 0:
        raise RuntimeError(f"CTC invalid input_lengths: min={int(torch.min(input_lengths_cpu).item())}")
    if int(torch.max(input_lengths_cpu).item()) > t_steps:
        raise RuntimeError(
            f"CTC invalid input_lengths: max={int(torch.max(input_lengths_cpu).item())} exceeds T={t_steps}"
        )
    if int(torch.min(target_lengths_cpu).item()) < 0:
        raise RuntimeError(f"CTC invalid target_lengths: min={int(torch.min(target_lengths_cpu).item())}")
    if int(torch.max(target_lengths_cpu).item()) > int(targets_cpu.shape[1]):
        raise RuntimeError(
            f"CTC invalid target_lengths: max={int(torch.max(target_lengths_cpu).item())} exceeds S={int(targets_cpu.shape[1])}"
        )
    if (target_lengths_cpu > input_lengths_cpu).any().item():
        bad_idx = int((target_lengths_cpu > input_lengths_cpu).nonzero(as_tuple=False)[0].item())
        raise RuntimeError(
            f"CTC invalid lengths at batch idx {bad_idx}: target_len={int(target_lengths_cpu[bad_idx].item())} "
            f"> input_len={int(input_lengths_cpu[bad_idx].item())}"
        )

    # Check only the valid target positions per sample.
    for bi in range(bsz):
        cur_len = int(target_lengths_cpu[bi].item())
        if cur_len == 0:
            continue
        cur = targets_cpu[bi, :cur_len]
        if int(torch.min(cur).item()) < 0 or int(torch.max(cur).item()) >= n_classes:
            raise RuntimeError(
                f"CTC target index out of range at batch idx {bi}: "
                f"min={int(torch.min(cur).item())}, max={int(torch.max(cur).item())}, classes={n_classes}"
            )


def _labels_from_target_cl(target_cl, n_classes):
    if target_cl.ndim >= 2 and target_cl.shape[1] > 1:
        labels = torch.argmax(target_cl, dim=1).long()
    else:
        labels = target_cl.view(-1).long()
        if torch.min(labels).item() >= 1 and torch.max(labels).item() <= n_classes:
            labels = labels - 1
    return labels


def _load_audio_waveforms(audio_wav_path, target_sample_rate):
    if not os.path.isdir(audio_wav_path):
        raise FileNotFoundError(f"Audio waveform directory not found: {audio_wav_path}")

    wav_cache = {}
    for fname in sorted(os.listdir(audio_wav_path)):
        if not fname.lower().endswith('.wav'):
            continue
        stem = os.path.splitext(fname)[0]
        if not stem.lower().startswith('audio'):
            continue
        class_code = int(stem.replace('audio', ''))
        wav_obj = wavio.read(os.path.join(audio_wav_path, fname))
        sample_rate = int(wav_obj.rate)
        wav_np = np.asarray(wav_obj.data)

        if np.issubdtype(wav_np.dtype, np.integer):
            info = np.iinfo(wav_np.dtype)
            scale = float(max(abs(info.min), info.max))
            wav_np = wav_np.astype(np.float32) / scale
        else:
            wav_np = wav_np.astype(np.float32)

        if wav_np.ndim == 1:
            wav_np = wav_np[:, None]

        # [time, channels] -> [channels, time]
        wav = torch.from_numpy(wav_np.T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, target_sample_rate)
        wav_cache[class_code] = wav.squeeze(0).float().cpu()

    if not wav_cache:
        raise RuntimeError(f"No wav files found under: {audio_wav_path}")
    return wav_cache


def _build_voice_batch(labels, wav_cache, device):
    voice_list = []
    max_len = 0
    for lbl in labels.tolist():
        class_code = int(lbl) + 1
        if class_code not in wav_cache:
            raise KeyError(f"Missing waveform for class code {class_code} in audio cache.")
        wav = wav_cache[class_code]
        voice_list.append(wav)
        max_len = max(max_len, wav.shape[0])

    padded = []
    for wav in voice_list:
        if wav.shape[0] < max_len:
            pad = max_len - wav.shape[0]
            wav = F.pad(wav, (0, pad))
        elif wav.shape[0] > max_len:
            wav = wav[:max_len]
        padded.append(wav)

    return torch.stack(padded, dim=0).to(device)


def _stt_forward_chunked(model_stt, wave, chunk_size, requires_grad):
    """Run STT in smaller chunks to reduce peak GPU memory."""
    batch_size = int(wave.size(0))
    if chunk_size is None or chunk_size <= 0:
        chunk_size = batch_size

    emissions = []
    if requires_grad:
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            em, _ = model_stt(wave[start:end])
            emissions.append(em)
    else:
        with torch.no_grad():
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                em, _ = model_stt(wave[start:end])
                emissions.append(em)

    return torch.cat(emissions, dim=0)

def train(args, train_loader, models, criterions, optimizers, epoch, trainValid=True, inference=False):
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param models: tuple containing the generator and discriminator models
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: losses
    '''
    (optimizer_g, optimizer_d) = optimizers
    
    # switch to train mode
    assert type(models) == tuple, "More than two models should be inputed (generator and discriminator)"

    epoch_loss_g = []
    epoch_loss_d = []
    
    epoch_acc_g = []
    epoch_acc_d = []
    
    epoch_loss_g_ns = []
    epoch_loss_d_ns = []
    
    epoch_acc_g_ns = []
    epoch_acc_d_ns = []

    total_batches = len(train_loader)
    
    for i, (input, target, target_cl, data_info) in enumerate(train_loader):    

        print("\rBatch [%5d / %5d]"%(i,total_batches), sep=' ', end='', flush=True)
        
        input = input.cuda()
        target = target.cuda()
        target_cl = target_cl.cuda()
        labels = _labels_from_target_cl(target_cl, len(args.classname))
        voice = _build_voice_batch(labels, args.audio_wavs, input.device)
        
        # extract unseen
        idx_unseen=[]
        idx_seen=[]
        for j in range(len(labels)):
            lbl_idx = int(labels[j].item())
            if args.classname[lbl_idx] == args.unseen:
                idx_unseen.append(j)
            else:
                idx_seen.append(j)
        
        input_ns = input[idx_unseen]
        target_ns = target[idx_unseen]
        target_cl_ns = target_cl[idx_unseen]
        labels_ns = labels[idx_unseen]
        voice_ns = voice[idx_unseen]
        data_info_ns = [data_info[0][idx_unseen],data_info[1][idx_unseen]]
        
        input = input[idx_seen]
        target = target[idx_seen]
        target_cl = target_cl[idx_seen]
        labels = labels[idx_seen]
        voice = voice[idx_seen]
        data_info = [data_info[0][idx_seen],data_info[1][idx_seen]]
        
        # # need to remove
        # models = (model_g, model_d, vocoder, model_STT, decoder_STT)
        # criterions = (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER)
        # trainValid = True
        
        # general training         
        if len(input) != 0:
            # train generator
            mel_out, e_loss_g, e_acc_g = train_G(args, 
                                                 input, target, voice, labels,
                                                 models, criterions, optimizer_g, 
                                                 data_info, 
                                                 trainValid,
                                                 phase_is_train=trainValid,
                                                 batch_idx=i)
            epoch_loss_g.append(e_loss_g)
            epoch_acc_g.append(e_acc_g)
        
            # train discriminator
            e_loss_d, e_acc_d = train_D(args, 
                                        mel_out, target, labels,
                                        models, criterions, optimizer_d, 
                                        trainValid)
            epoch_loss_d.append(e_loss_d)
            epoch_acc_d.append(e_acc_d)
        
        # Unseen words training
        if len(input_ns) != 0 :
            # Unseen train generator
            mel_out_ns, e_loss_g_ns, e_acc_g_ns = train_G(args, 
                                                          input_ns, target_ns, voice_ns, labels_ns,
                                                          models, criterions, optimizer_g, 
                                                          data_info_ns,
                                                          False,
                                                          phase_is_train=trainValid,
                                                          batch_idx=i)
            epoch_loss_g_ns.append(e_loss_g_ns)
            epoch_acc_g_ns.append(e_acc_g_ns)
            
            # Unseen train discriminator
            e_loss_d_ns, e_acc_d_ns = train_D(args, 
                                              mel_out_ns, target_ns, labels_ns,
                                              models, criterions, optimizer_d, 
                                              False)
            epoch_loss_d_ns.append(e_loss_d_ns)
            epoch_acc_d_ns.append(e_acc_d_ns)

    epoch_loss_g = np.array(epoch_loss_g)
    epoch_acc_g = np.array(epoch_acc_g)
    epoch_loss_d = np.array(epoch_loss_d)
    epoch_acc_d = np.array(epoch_acc_d)
    
    epoch_loss_g_ns = np.array(epoch_loss_g_ns)
    epoch_acc_g_ns = np.array(epoch_acc_g_ns)
    epoch_loss_d_ns = np.array(epoch_loss_d_ns)
    epoch_acc_d_ns = np.array(epoch_acc_d_ns)
    
    
    args.loss_g = _safe_column_mean(epoch_loss_g, 0)
    args.loss_g_recon = _safe_column_mean(epoch_loss_g, 1)
    args.loss_g_valid = _safe_column_mean(epoch_loss_g, 2)
    args.loss_g_ctc = _safe_column_mean(epoch_loss_g, 3)
    args.acc_g_valid = _safe_column_mean(epoch_acc_g, 0)
    args.cer_gt = _safe_column_mean(epoch_acc_g, 1)
    args.cer_recon = _safe_column_mean(epoch_acc_g, 2)

    args.loss_d = _safe_column_mean(epoch_loss_d, 0)
    args.loss_d_valid = _safe_column_mean(epoch_loss_d, 1)
    args.loss_d_cl = _safe_column_mean(epoch_loss_d, 2)
    args.acc_d_real = _safe_column_mean(epoch_acc_d, 0)
    args.acc_d_fake = _safe_column_mean(epoch_acc_d, 1)
    args.acc_cl_real = _safe_column_mean(epoch_acc_d, 2)
    args.acc_cl_fake = _safe_column_mean(epoch_acc_d, 3)

    # Unseen
    args.loss_g_ns = _safe_column_mean(epoch_loss_g_ns, 0)
    args.loss_g_recon_ns = _safe_column_mean(epoch_loss_g_ns, 1)
    args.loss_g_valid_ns = _safe_column_mean(epoch_loss_g_ns, 2)
    args.loss_g_ctc_ns = _safe_column_mean(epoch_loss_g_ns, 3)
    args.acc_g_valid_ns = _safe_column_mean(epoch_acc_g_ns, 0)
    args.cer_gt_ns = _safe_column_mean(epoch_acc_g_ns, 1)
    args.cer_recon_ns = _safe_column_mean(epoch_acc_g_ns, 2)

    args.loss_d_ns = _safe_column_mean(epoch_loss_d_ns, 0)
    args.loss_d_valid_ns = _safe_column_mean(epoch_loss_d_ns, 1)
    args.loss_d_cl_ns = _safe_column_mean(epoch_loss_d_ns, 2)
    args.acc_d_real_ns = _safe_column_mean(epoch_acc_d_ns, 0)
    args.acc_d_fake_ns = _safe_column_mean(epoch_acc_d_ns, 1)
    args.acc_cl_real_ns = _safe_column_mean(epoch_acc_d_ns, 2)
    args.acc_cl_fake_ns = _safe_column_mean(epoch_acc_d_ns, 3)
    
    # tensorboard
    if trainValid:
        tag = 'train'
    else:
        tag = 'valid'
        
    if not inference and args.writer is not None:
        try:
            args.writer.add_scalar("Loss_G/{}".format(tag), args.loss_g, epoch)
            args.writer.add_scalar("CER/{}".format(tag), args.cer_recon, epoch)
            
            args.writer.add_scalar("Loss_G_recon/{}".format(tag), args.loss_g_recon, epoch)
            args.writer.add_scalar("Loss_G_valid/{}".format(tag), args.loss_g_valid, epoch)
            args.writer.add_scalar("Loss_G_ctc/{}".format(tag), args.loss_g_ctc, epoch)
            
            args.writer.add_scalar("ACC_D_real/{}".format(tag), args.acc_d_real, epoch)
            args.writer.add_scalar("ACC_D_fake/{}".format(tag), args.acc_d_fake, epoch)
            
            args.writer.add_scalar("Loss_G_unseen/{}".format(tag), args.loss_g_ns, epoch)
            args.writer.add_scalar("CER_unseen/{}".format(tag), args.cer_recon_ns, epoch)
        except Exception as exc:
            print(f"TensorBoard logging disabled after runtime error: {exc}")
            args.writer = None

    print('\n[%3d/%3d] CER-gt: %.4f CER-recon: %.4f / ACC_R: %.4f ACC_F: %.4f / g-RMSE: %.4f g-lossValid: %.4f g-lossCTC: %.4f' 
          % (i, total_batches, 
             args.cer_gt, args.cer_recon, 
             args.acc_d_real, args.acc_d_fake, 
             args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc))
        
        
    return (args.loss_g, args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc, args.acc_g_valid, args.cer_gt, args.cer_recon, 
            args.loss_d, args.acc_d_real, args.acc_d_fake)


def train_G(args, input, target, voice, labels, models, criterions, optimizer_g, data_info, trainValid, phase_is_train=True, batch_idx=0):

    (model_g, model_d, vocoder, model_STT, decoder_STT) = models
    (criterion_recon, criterion_ctc, criterion_adv, _, CER) =  criterions
    
    if trainValid:
        model_g.train()
        model_d.train()
    else:
        model_g.eval()
        model_d.eval()
    # Vocoder and STT are always frozen; keep in eval mode to disable dropout and reduce activation memory.
    vocoder.eval()
    model_STT.eval()
    
    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(input), 1), dtype=torch.float32).cuda()
    
    ###############################
    # Train Generator
    ###############################
    
    if trainValid:
        for p in model_g.parameters():
            p.requires_grad = True   # unfreeze G
        for p in model_d.parameters():
            p.requires_grad = False  # freeze D
        for p in vocoder.parameters():
            p.requires_grad = False  # freeze vocoder
        for p in model_STT.parameters():
            p.requires_grad = False  # freeze model_STT
            
        # set zero grad    
        optimizer_g.zero_grad()
        
        # Run Generator
        output = model_g(input)
    else:
        with torch.no_grad():
            # run generator
            output = model_g(input)
    _assert_finite("generator_output", output)
    
    # DTW
    mel_out = output.clone()
    mel_out = DTW_align(mel_out, target)
    #_assert_finite("mel_out_after_dtw", mel_out)
    #mel_out = torch.clamp(mel_out, min=args.mel_clamp_min, max=args.mel_clamp_max)
    #_assert_finite("mel_out_after_clamp", mel_out)
    
    # Run Discriminator
    g_valid, _ = model_d(mel_out)
    
    # generator loss
    loss_recon = criterion_recon(mel_out, target)
    
    # GAN loss
    loss_valid = criterion_adv(g_valid, valid)
    
    # accuracy    args.l_g = h_g.l_g
    acc_g_valid = (g_valid.round() == valid).float().mean()
    
    ###############################
    # Loss from Vocoder - STT
    ###############################
    # out_DTW
    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    _assert_finite("output_denorm", output_denorm)
    #output_denorm = torch.clamp(output_denorm, min=args.vocoder_mel_min, max=args.vocoder_mel_max)
    #_assert_finite("output_denorm_after_clamp", output_denorm)

    # Optionally compute the vocoder(just HiFi)+STT+CTC branch on CPU to reduce GPU memory pressure.
    ctc_device = getattr(args, "ctc_torch_device", output_denorm.device)
    if isinstance(ctc_device, str):
        ctc_device = torch.device(ctc_device)
    
    gt_label=[]
    gt_label_idx=[]
    gt_length=[]
    for j in range(len(target)):
        gt_label.append(args.word_label[labels[j].item()])
        gt_label_idx.append(args.word_index[labels[j].item()])
        gt_length.append(args.word_length[labels[j].item()])
    gt_label_idx = torch.tensor(np.array(gt_label_idx),dtype=torch.int64)
    gt_length = torch.tensor(gt_length,dtype=torch.int64)
    
    
    emission_recon = None
    # recon
    ##### VOCODING
    output_denorm_ctc = output_denorm.to(ctc_device)
    wav_recon = vocoder(output_denorm_ctc)
    _assert_finite("wav_recon_from_vocoder", wav_recon)
    wav_recon = torch.clamp(wav_recon, min=-1.0, max=1.0)
    wav_recon = torch.reshape(wav_recon, (len(wav_recon), wav_recon.shape[-1]))

    #### resampling for STT (always match asr model sample rate)
    if int(args.sample_rate_mel) != int(args.sample_rate_STT):
        wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)

    ##### STT Wav2Vec 2.0
    stt_chunk_size = int(getattr(args, "stt_chunk_size", 4))
    emission_recon = _stt_forward_chunked(model_STT, wav_recon, stt_chunk_size, requires_grad=True)

    should_compute_cer = False
    if phase_is_train:
        # During training, compute CER on seen-word batches every N batches.
        if trainValid:
            interval = int(getattr(args, "cer_every_n_batches", 10))
            if interval > 0:
                should_compute_cer = (int(batch_idx) % interval) == 0
    else:
        should_compute_cer = bool(getattr(args, "compute_cer_in_val", True))

    emission_gt = None
    if should_compute_cer:
        stt_chunk_size = int(getattr(args, "stt_chunk_size", 4))
        emission_gt = _stt_forward_chunked(model_STT, voice.to(ctc_device), stt_chunk_size, requires_grad=False)

    if emission_recon is None and should_compute_cer:
        stt_chunk_size = int(getattr(args, "stt_chunk_size", 4))
        with torch.no_grad():
            output_denorm_cer = output_denorm.to(ctc_device)
            wav_recon_cer = vocoder(output_denorm_cer)
            wav_recon_cer = torch.clamp(wav_recon_cer, min=-1.0, max=1.0)
            wav_recon_cer = torch.reshape(wav_recon_cer, (len(wav_recon_cer), wav_recon_cer.shape[-1]))
            if int(args.sample_rate_mel) != int(args.sample_rate_STT):
                wav_recon_cer = torchaudio.functional.resample(wav_recon_cer, args.sample_rate_mel, args.sample_rate_STT)
            emission_recon = _stt_forward_chunked(model_STT, wav_recon_cer, stt_chunk_size, requires_grad=False)
   
    # CTC loss
    if emission_recon is not None:
        input_lengths = torch.full(
            size=(emission_recon.size(dim=0),),
            fill_value=emission_recon.size(dim=1),
            dtype=torch.long,
            device=emission_recon.device,
        )
        gt_label_idx_ctc = gt_label_idx.to(emission_recon.device)
        gt_length_ctc = gt_length.to(emission_recon.device)
        _validate_ctc_inputs(emission_recon, gt_label_idx_ctc, input_lengths, gt_length_ctc)
        emission_recon_ = emission_recon.log_softmax(2).transpose(0, 1)
        loss_ctc = F.ctc_loss(
            emission_recon_,
            gt_label_idx_ctc,
            input_lengths,
            gt_length_ctc,
            zero_infinity=True,
        )
        if loss_ctc.device != loss_recon.device:
            loss_ctc = loss_ctc.to(loss_recon.device)
    else:
        loss_ctc = loss_recon.new_zeros(())
    
    # total generator loss
    loss_g = args.l_g[0] * loss_recon + args.l_g[1] * loss_valid + args.l_g[2] * loss_ctc

    cer_device = emission_recon.device if emission_recon is not None else loss_recon.device
    cer_gt = torch.tensor(0.0, device=cer_device)
    cer_recon = torch.tensor(0.0, device=cer_device)
    if should_compute_cer:
        # decoder STT
        transcript_gt = []
        transcript_recon = []

        emission_recon_detached = emission_recon.detach()
        for j in range(len(voice)):
            transcript = decoder_STT(emission_gt[j])
            transcript_gt.append(transcript)

            transcript = decoder_STT(emission_recon_detached[j])
            transcript_recon.append(transcript)

        cer_gt = CER(transcript_gt, gt_label)
        cer_recon = CER(transcript_recon, gt_label)

    if trainValid:
        loss_g.backward()
        optimizer_g.step()
        torch.cuda.empty_cache()

    e_loss_g = (loss_g.item(), loss_recon.item(), loss_valid.item(), loss_ctc.item())
    e_acc_g = (acc_g_valid.item(), cer_gt.item(), cer_recon.item())
    
    return mel_out, e_loss_g, e_acc_g
      
    
def train_D(args, mel_out, target, labels, models, criterions, optimizer_d, trainValid):
    
    (_, model_d, _, _, _) = models
    (_, _, criterion_adv, criterion_cl, _) =  criterions

    if trainValid:
        model_d.train()
    else:
        model_d.eval()
    
    # Adversarial ground truths 1:real, 0: fake
    valid = torch.ones((len(mel_out), 1), dtype=torch.float32).cuda()
    fake = torch.zeros((len(mel_out), 1), dtype=torch.float32).cuda()
    
    ###############################
    # Train Discriminator
    ###############################
    
    if trainValid:
        if args.pretrain and args.prefreeze:
            for total_ct, _ in enumerate(model_d.children()):
                ct=0
            for ct, child in enumerate(model_d.children()):
                if ct > total_ct-1: # unfreeze classifier 
                    for param in child.parameters():
                        param.requires_grad = True  # unfreeze D    
        else:
            for p in model_d.parameters():
                p.requires_grad = True  # unfreeze D   
                
        # set zero grad
        optimizer_d.zero_grad()

    # run model cl
    real_valid, real_cl = model_d(target)
    fake_valid, fake_cl = model_d(mel_out.detach())

    loss_d_real_valid = criterion_adv(real_valid, valid)
    loss_d_fake_valid = criterion_adv(fake_valid, fake)
    loss_d_real_cl = criterion_cl(real_cl, labels)
    
    loss_d_valid = 0.5 * (loss_d_real_valid + loss_d_fake_valid)
    loss_d_cl = loss_d_real_cl
    
    loss_d = args.l_d[0] * loss_d_cl + args.l_d[1] * loss_d_valid
    
    # accuracy
    acc_d_real = (real_valid.round() == valid).float().mean()
    acc_d_fake = (fake_valid.round() == fake).float().mean()
    preds_real = torch.argmax(real_cl,dim=1)
    acc_cl_real = (preds_real == labels).float().mean()
    preds_fake = torch.argmax(fake_cl,dim=1)
    acc_cl_fake = (preds_fake == labels).float().mean()
    
    if trainValid:
        loss_d.backward()
        optimizer_d.step()

    e_loss_d = (loss_d.item(), loss_d_valid.item(), loss_d_cl.item())
    e_acc_d = (acc_d_real.item(), acc_d_fake.item(), acc_cl_real.item(), acc_cl_fake.item())
    
    return e_loss_d, e_acc_d


def saveData(args, test_loader, models, epoch, losses):
    
    model_g = models[0].eval()

    input, target, target_cl, data_info = next(iter(test_loader))
    
    input = input.cuda()
    target = target.cuda()
    target_cl = target_cl.cuda()
    labels = _labels_from_target_cl(target_cl, len(args.classname))
    
    with torch.no_grad():
        # run the mdoel
        output = model_g(input)
    
    mel_out = output
    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    mel_recon_np = np.asarray(output_denorm[0].detach().cpu().numpy())

    # Save spectrogram only (no waveform synthesis/save).
    str_tar = args.word_label[labels[0].item()].replace("|", "_").replace(" ", "_")
    title = "Tar_{}".format(str_tar)
    save_path = args.savevoice + '/e{}_{}_mel.csv'.format(str(epoch), title)
    np.savetxt(save_path, mel_recon_np, delimiter=",")


def main(args):
    
    device = torch.device(f'cuda:{args.gpuNum[0]}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device: {} '.format(torch.cuda.current_device())) # check
    print('The number of available GPU:{}'.format(torch.cuda.device_count()))
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    args.vocoder_type = str(getattr(args, "vocoder_type", "hifigan")).lower()
    if args.vocoder_type not in ("hifigan", "griffinlim", "hifigan_16k"):
        raise ValueError(f"Unsupported vocoder_type: {args.vocoder_type}. Use 'hifigan', 'hifigan_16k', or 'griffinlim'.")

    ctc_device_name = str(getattr(args, "ctc_device", "cuda")).lower()
    if ctc_device_name not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported ctc_device: {args.ctc_device}. Use 'cuda' or 'cpu'.")
    if ctc_device_name == "cuda" and not torch.cuda.is_available():
        print("ctc_device=cuda requested but CUDA is unavailable. Falling back to CPU for vocoder+STT+CTC.")
        ctc_device_name = "cpu"
    args.ctc_torch_device = torch.device("cpu" if ctc_device_name == "cpu" else f"cuda:{args.gpuNum[0]}")

    # define generator
    config_file = os.path.join(args.model_config, 'config_G.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_g = AttrDict(json_config)
    model_g = networks.Generator(h_g).cuda()
    
    args.sample_rate_mel = args.sampling_rate
    
    # define discriminator
    config_file = os.path.join(args.model_config, 'config_D.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_d = AttrDict(json_config)
    model_d = networks.Discriminator(h_d).cuda()
    
    # Vocoder backend
    if args.vocoder_type == "hifigan_16k":
        args.sample_rate_mel = TorchaudioHiFiGAN16k.SAMPLE_RATE  # 16000
        vocoder = TorchaudioHiFiGAN16k(device=args.ctc_torch_device)
        print(f"Using torchaudio HiFi-GAN 16kHz vocoder (sample_rate={args.sample_rate_mel})")
    elif args.vocoder_type == "griffinlim":
        args.sample_rate_mel = 16000
        n_mels = int(getattr(h_g, "num_mels", 80))
        vocoder = GriffinLimVocoder(n_mels=n_mels, sample_rate=args.sample_rate_mel).to(args.ctc_torch_device)
        print(f"Using Griffin-Lim vocoder (n_mels={n_mels}, sample_rate={args.sample_rate_mel})")
    else:
        # HiFi-GAN (pretrained)
        # UNIVERSALv1,
        config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
        with open(config_file) as f:
            data = f.read()

        json_config = json.loads(data)
        h = AttrDict(json_config)

        vocoder = model_HiFi(h).to(args.ctc_torch_device)
        state_dict_g = torch.load(args.vocoder_pre) #, map_location=args.device)
        vocoder.load_state_dict(state_dict_g['generator'])
        print(f"Using HiFi-GAN vocoder checkpoint: {args.vocoder_pre}")
    
    # STT backend
    stt_backbone = str(getattr(args, "stt_backbone", "wav2vec2_base_960h")).lower()
    stt_bundle_map = {
        "wav2vec2_base_960h": torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H,
        "wav2vec2_large_960h": torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H,
        "hubert_large": torchaudio.pipelines.HUBERT_ASR_LARGE,
    }
    if stt_backbone not in stt_bundle_map:
        raise ValueError(
            f"Unsupported stt_backbone: {stt_backbone}. "
            "Use 'wav2vec2_base_960h', 'wav2vec2_large_960h', or 'hubert_large'."
        )

    bundle = stt_bundle_map[stt_backbone]
    model_STT = bundle.get_model().to(args.ctc_torch_device)
    args.sample_rate_STT = int(bundle.sample_rate)
    print(f"Using STT backbone: {stt_backbone} (sample_rate={args.sample_rate_STT}, device={args.ctc_torch_device})")
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    args.word_index, args.word_length = word_index(args.word_label, bundle)
    args.audio_wavs = _load_audio_waveforms(args.audio_wav_path, args.sample_rate_STT)
    
    # Parallel setting
    if len(args.gpuNum) > 1:
        model_g = nn.DataParallel(model_g, device_ids=args.gpuNum)
        model_d = nn.DataParallel(model_d, device_ids=args.gpuNum)
        if args.vocoder_type == "hifigan" and args.ctc_torch_device.type == "cuda":
            vocoder = nn.DataParallel(vocoder, device_ids=args.gpuNum)
        if args.ctc_torch_device.type == "cuda":
            model_STT = nn.DataParallel(model_STT, device_ids=args.gpuNum)
        print(f"Using DataParallel across GPUs: {args.gpuNum}")
    else:
        print(f"Single-GPU mode on cuda:{args.gpuNum[0]} (DataParallel disabled).")

    # loss function
    criterion_recon = RMSELoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    criterion_ctc = nn.CTCLoss().cuda()
    criterion_cl = nn.CrossEntropyLoss().cuda()
    CER = CharErrorRate().cuda()

    # optimizer
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.lr_d_decay, last_epoch=-1)

   # create the directory if not exist
    if not os.path.exists(args.logDir):
        os.mkdir(args.logDir)
        
    sub_tag = "subj" + "-".join(str(s) for s in args.sub)

    subDir = os.path.join(args.logDir, sub_tag)
    if not os.path.exists(subDir):
        os.mkdir(subDir)        
        
    saveDir = os.path.join(args.logDir, sub_tag, args.task)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    args.savevoice = saveDir + '/epovoice'
    if not os.path.exists(args.savevoice):
        os.mkdir(args.savevoice)

    args.savemodel = saveDir + '/savemodel'
    if not os.path.exists(args.savemodel):
        os.mkdir(args.savemodel)
        
    args.logs = saveDir + '/logs'
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
        
    # Load trained model
    start_epoch = 0
    if args.pretrain:
        loc_g = os.path.join(args.trained_model, sub_tag, 'BEST_checkpoint_g.pt')
        loc_d = os.path.join(args.trained_model, sub_tag, 'BEST_checkpoint_d.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))

    if args.resume:
        loc_g = os.path.join(args.savemodel, 'checkpoint_g.pt')
        loc_d = os.path.join(args.savemodel, 'checkpoint_d.pt')

        if os.path.isfile(loc_g):
            print("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
            start_epoch = checkpoint_g['epoch'] + 1
        else:
            print("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            print("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(loc_d))

    # TensorBoard setting (optional-safe)
    try:
        args.writer = SummaryWriter(args.logs)
    except Exception as exc:
        print(f"TensorBoard disabled: {exc}")
        args.writer = None

    # CSV metrics are always enabled for robust training logs.
    args.metrics_csv = os.path.join(args.logs, "metrics.csv")
    _ensure_metrics_csv(args.metrics_csv)
    
    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)

    trainset = myDataset(
        mode=0,
        data=args.dataLoc,
        task=args.task,
        recon=args.recon,
        audio_mel_path=args.audio_mel_path,
        subjects=args.sub,
        eeg_csp_path=args.eeg_csp_path,
        split='train',
        speech_type=args.task,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    valset = myDataset(
        mode=2,
        data=args.dataLoc,
        task=args.task,
        recon=args.recon,
        audio_mel_path=args.audio_mel_path,
        subjects=args.sub,
        eeg_csp_path=args.eeg_csp_path,
        split='val',
        speech_type=args.task,
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)

    epoch = start_epoch
    lr_g = 0
    lr_d = 0
    best_loss = 1000
    is_best = False
    epochs_since_improvement = 0
    
    for epoch in range(start_epoch, args.max_epochs):
        
        start_time = time.time()
        
        for param_group in optimizer_g.param_groups:
            lr_g = param_group['lr']
        for param_group in optimizer_d.param_groups:
            lr_d = param_group['lr']

        print("Epoch : %d/%d" %(epoch, args.max_epochs) )
        print("Learning rate for G: %.9f" %lr_g)
        print("Learning rate for D: %.9f" %lr_d)
        Tr_losses = train(args, train_loader, 
                          (model_g, model_d, vocoder, model_STT, decoder_STT), 
                          (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER), 
                          (optimizer_g, optimizer_d), 
                          epoch,
                          True) 
        
        Val_losses = train(args, val_loader, 
                           (model_g, model_d, vocoder, model_STT, decoder_STT), 
                           (criterion_recon, criterion_ctc, criterion_adv, criterion_cl, CER), 
                           ([],[]), 
                           epoch,
                           False)
        # Step schedulers after optimizer.step() calls in the epoch.
        scheduler_g.step()
        scheduler_d.step()
        
        # Save checkpoint
        state_g = {'arch': str(model_g),
                 'state_dict': model_g.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_g.state_dict()}
        
        state_d = {'arch': str(model_d),
                 'state_dict': model_d.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_d.state_dict()}
        
        # Did validation loss improve?
        loss_total =  Val_losses[0]
        is_best = loss_total < best_loss
        best_loss = min(loss_total, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
        save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')

        if (epoch % 10) == 0:
            saveData(args, val_loader, (model_g, model_d, vocoder, model_STT, decoder_STT), epoch, (Tr_losses,Val_losses))

        time_taken = time.time() - start_time

        metrics_row = {
            "epoch": int(epoch),
            "lr_g": float(lr_g),
            "lr_d": float(lr_d),
            "train_loss_g": float(Tr_losses[0]),
            "train_loss_g_recon": float(Tr_losses[1]),
            "train_loss_g_valid": float(Tr_losses[2]),
            "train_loss_g_ctc": float(Tr_losses[3]),
            "train_acc_g_valid": float(Tr_losses[4]),
            "train_cer_gt": float(Tr_losses[5]),
            "train_cer_recon": float(Tr_losses[6]),
            "train_loss_d": float(Tr_losses[7]),
            "train_acc_d_real": float(Tr_losses[8]),
            "train_acc_d_fake": float(Tr_losses[9]),
            "val_loss_g": float(Val_losses[0]),
            "val_loss_g_recon": float(Val_losses[1]),
            "val_loss_g_valid": float(Val_losses[2]),
            "val_loss_g_ctc": float(Val_losses[3]),
            "val_acc_g_valid": float(Val_losses[4]),
            "val_cer_gt": float(Val_losses[5]),
            "val_cer_recon": float(Val_losses[6]),
            "val_loss_d": float(Val_losses[7]),
            "val_acc_d_real": float(Val_losses[8]),
            "val_acc_d_fake": float(Val_losses[9]),
            "best_val_loss_so_far": float(best_loss),
            "is_best": int(is_best),
            "epoch_time_sec": float(time_taken),
        }
        _append_metrics_csv(args.metrics_csv, metrics_row)

        print("Time: %.2f\n"%time_taken)
        
    if args.writer is not None:
        args.writer.flush()

if __name__ == '__main__':

    dataDir = './eegdata'
    audioDir = './audiodata'
    audioWavDir = './audiodata/twos_16000'
    logDir = './TrainResult'
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--vocoder_pre', type=str, default='UNIVERSAL_V1/g_02500000', help='pretrained vocoder file path')
    parser.add_argument('--vocoder_type', type=str, default='hifigan_16k', choices=['hifigan', 'hifigan_16k', 'griffinlim'], help='vocoder backend to synthesize waveform from mel')
    parser.add_argument('--trained_model', type=str, default=None, help='trained model for G & D folder path')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--dataLoc', type=str, default=dataDir)
    parser.add_argument('--eeg_csp_path', type=str, default=dataDir, help='root EEG directory (contains subjXX folders)')
    parser.add_argument('--audio_mel_path', type=str, default=audioDir, help='audio mel directory with audio*_logmel.csv')
    parser.add_argument('--audio_wav_path', type=str, default=audioWavDir, help='audio waveform directory with audio*.wav')
    parser.add_argument('--config', type=str, default='./config_params.json')
    parser.add_argument('--logDir', type=str, default=logDir)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--prefreeze', type=bool, default=False)
    parser.add_argument('--gpuNum', nargs='+', type=int, default=[0])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sub', nargs='+', type=int, default=[18])
    parser.add_argument('--task', type=str, default='imagined_speech')

    parser.add_argument('--recon', type=str, default='Y_mel')
    parser.add_argument('--unseen', type=str, default='stop')
    parser.add_argument('--stt_chunk_size', type=int, default=2)
    parser.add_argument('--stt_backbone', type=str, default='wav2vec2_base_960h', choices=['wav2vec2_base_960h', 'wav2vec2_large_960h', 'hubert_large'])
    parser.add_argument('--ctc_device', type=str, default='cpu', choices=['cuda', 'cpu'], help='device for vocoder+STT+CTC branch')
    parser.add_argument('--cer_every_n_batches', type=int, default=6)
    parser.add_argument('--compute_cer_in_val', type=bool, default=True)
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    main(args)        
    
     
