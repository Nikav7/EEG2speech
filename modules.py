import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


###################################   DTW    ####################################
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0,1:] = np.inf
    dtw[1:,0] = np.inf
    eps = 1e-4
    for i in range(1,costs.shape[0]):
        for j in range(1,costs.shape[1]):
            dtw[i,j] = costs[i,j] + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return dtw

def align_from_distances(distance_matrix, debug=False):
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i-1,j),(i,j-1),(i-1,j-1)], key=lambda x: dtw[x[0],x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)),results] = 1
        plt.matshow(visual)
        plt.show()

    return results

def DTW_align(input, target):
    # Return a tensor aligned to target time length: (B, C, T_target)
    aligned = torch.empty_like(target)
    for j in range(len(input)):
        dists = torch.cdist(torch.transpose(input[j], 1, 0), torch.transpose(target[j], 1, 0))
        alignment = align_from_distances(dists.T.cpu().detach().numpy())
        aligned[j, :, :] = input[j, :, alignment]

    return aligned


def DTW_alignGPU(input_tensor, target_tensor):
    """
    Computes DTW on the GPU safely without breaking the PyTorch AMP Scaler graph.
    
    Args:
        input_tensor (Tensor): Shape (B, C, N) -> e.g., audio_fake[:, 0]
        target_tensor (Tensor): Shape (B, C, M) -> e.g., audio[:, 0]
    Returns:
        Tensor: Warped version of input_tensor matching the M timesteps of target_tensor.
    """
    device = input_tensor.device
    B, C, N = input_tensor.shape
    _, _, M = target_tensor.shape

    # 1. Compute path entirely detached from autograd to prevent graph pollution
    with torch.no_grad():
        x_t = input_tensor.permute(0, 2, 1)    # (B, N, C)
        y_t = target_tensor.permute(0, 2, 1)   # (B, M, C)
        
        # Safe distance matrix expansion
        x_norms = torch.sum(x_t ** 2, dim=-1, keepdim=True)
        y_norms = torch.sum(y_t ** 2, dim=-1, keepdim=True).transpose(1, 2)
        xy_prod = torch.bmm(x_t, y_t.transpose(1, 2))
        dist_mat = torch.clamp(x_norms + y_norms - 2 * xy_prod, min=1e-12)
        
        INF = 1e8
        cost_mat = torch.full((B, N, M), INF, device=device)
        cost_mat[:, 0, 0] = dist_mat[:, 0, 0]
        
        # Dynamic Programming via Anti-Diagonals
        for d in range(1, N + M - 1):
            i_min = max(0, d - M + 1)
            i_max = min(N - 1, d)
            if i_min > i_max: continue
                
            i_indices = torch.arange(i_min, i_max + 1, device=device)
            j_indices = d - i_indices
            
            im1 = torch.clamp(i_indices - 1, min=0)
            jm1 = torch.clamp(j_indices - 1, min=0)
            
            v1 = cost_mat[:, im1, j_indices]
            v2 = cost_mat[:, i_indices, jm1]
            v3 = cost_mat[:, im1, jm1]
            
            v1 = torch.where(i_indices.unsqueeze(0) > 0, v1, torch.tensor(INF, device=device))
            v2 = torch.where(j_indices.unsqueeze(0) > 0, v2, torch.tensor(INF, device=device))
            v3 = torch.where((i_indices.unsqueeze(0) > 0) & (j_indices.unsqueeze(0) > 0), v3, torch.tensor(INF, device=device))
            
            min_prev = torch.min(torch.stack([v1, v2, v3], dim=-1), dim=-1)[0]
            cost_mat[:, i_indices, j_indices] = dist_mat[:, i_indices, j_indices] + min_prev

        # Backtracking to find target frame index mapping for each output step
        aligned_indices_batch = []
        for b in range(B):
            # CHANGED: Force length to M to track choices along the target tensor's timeline
            path = [0] * M
            i, j = N - 1, M - 1
            
            while i > 0 or j > 0:
                # CHANGED: Force M by assigning generated frames (i) to target index slots (j)
                path[j] = i  
                if i == 0:
                    j -= 1
                elif j == 0:
                    i -= 1
                else:
                    v1 = cost_mat[b, i-1, j]
                    v2 = cost_mat[b, i, j-1]  # Fixed typo typo here
                    v3 = cost_mat[b, i-1, j-1]
                    
                    m = min(v1, v2, v3)
                    if m == v3:
                        i -= 1
                        j -= 1
                    elif m == v1:
                        i -= 1
                    else:
                        j -= 1
            aligned_indices_batch.append(torch.tensor(path, dtype=torch.long, device=device))
            
        alignment_idx = torch.stack(aligned_indices_batch, dim=0) # Shape: (B, M)

    # 2. RUN GRADIENT SAFE EXTREMA LOOKUP
    # CHANGED: Select frames from input_tensor (audio_fake) so the output is exactly M long,
    # keeping the gradient flow context connected for the generator parameters.
    aligned_batch = []
    for b in range(B):
        aligned_batch.append(input_tensor[b].index_select(1, alignment_idx[b]))
        
    return torch.stack(aligned_batch, dim=0)

#####################################################################################
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    
######################################################################

#from nt code

def mel2wav_vocoder(mel, vocoder, mini_batch=2):
    waves = []
    for j in range(len(mel)//mini_batch):
        wave_ = vocoder(mel[mini_batch*j:mini_batch*j+mini_batch])
        waves.append(wave_.cpu().detach().numpy())
    wav_recon = torch.Tensor(np.array(waves)).cuda()
    wav_recon = torch.reshape(wav_recon, (len(mel),wav_recon.shape[-1]))
    
    return wav_recon


def perform_STT(wave, model_STT, decoder_STT, gt_label, mini_batch=2):
    # model STT
    emission = []
    with torch.inference_mode():
        for j in range(len(wave)//mini_batch):
            em_, _ = model_STT(wave[mini_batch*j:mini_batch*j+mini_batch])
            emission.append(em_.cpu().detach().numpy())
    emission_recon = torch.Tensor(np.array(emission)).cuda()
    emission_recon = torch.reshape(emission_recon, (len(wave),emission_recon.shape[-2],emission_recon.shape[-1]))
    
    # decoder STT
    transcripts = []
    # corr_num=0
    for j in range(len(wave)):
        transcript = decoder_STT(emission_recon[j])    
        transcripts.append(transcript)
        
    #     if transcript == gt_label[j]:
    #         corr_num = corr_num + 1

    # acc_word = corr_num / len(wave)
        
    return transcripts#, emission_recon, acc_word
