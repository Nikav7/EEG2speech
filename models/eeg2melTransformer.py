# multi-layered transformer inspired by the work of Wairagkar, 2025.
# Adapted for Riemannian Tangent Space EEG -> mel-spectrogram sequence synthesis.
# EEG Input: 27 chans, 2000 timepoints, 160ms windows, 22ms stride over 1000Hz 2s Epoch -> 85 time steps)
# PyTorch version Veronica Valente (RUG), 2025/2026

#PIPELINE: 
# [EEG Timeline] -> [Transformer Generator] -> [Predicted Spectrogram]
# [Generated denormalized Mel] -> [vocoding, e.g. HiFi-GAN, default griffin-lim] -> [Waveform Tensor] -> [Wav2Vec2] -> [Continuous Logits] -> [CTC Loss] -> (Backpropagation)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal Positional Encoding.
    Expects input shape [batch_size, seq_len, d_model].
    """
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Input sequence length ({x.size(1)}) exceeds max_len ({self.pe.size(1)})."
            )
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# --- The Model ---
class TransformerMelSynth(nn.Module):
    """
    Transformer-based EEG Riemannian Tangent Space -> mel-spectrogram sequence generator.

    Input:  (B, 1, tangent_space_dim, 85)   -- Calculated via Option B windowing
    Output: (B, 1, output_mel_bins, 85)     -- Multi-class equivalent Mel Spectrogram frames
    
    The sequence dimension T=85 is directly mapped across the whole network architecture.
    """

    def __init__(self,
                 n_channels: int = 27,             # Physical EEG channel count 
                 output_mel_bins: int = 80,
                 transformer_d_model: int = 128,
                 transformer_nhead: int = 8,
                 transformer_dim_feedforward: int = 512,
                 transformer_num_layers: int = 6,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000):
        super().__init__()

        self.n_channels = n_channels
        self.output_mel_bins = output_mel_bins
        self.transformer_d_model = transformer_d_model

        # Calculate exact vector length of the upper triangle of the manifold
        self.tangent_space_dim = int((n_channels * (n_channels + 1)) / 2)

        # Project flattened Tangent Space vector to transformer embedding dimension
        self.input_projection = nn.Sequential(
            nn.Linear(self.tangent_space_dim, transformer_d_model * 2),
            nn.GELU(),
            nn.Linear(transformer_d_model * 2, transformer_d_model),
        )

        self.positional_encoding = PositionalEncoding(
            transformer_d_model, dropout=dropout, max_len=max_seq_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_num_layers
        )

        # Per-time-step output projection -> mel bins
        self.output_projection = nn.Linear(transformer_d_model, output_mel_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, tangent_space_dim, 85)

        Returns:
            (B, 1, output_mel_bins, 85)
        """
        B, C, F, T = x.shape  # Expecting C=1, F=tangent_space_dim, T=85
        
        # Dimension validation check
        if F != self.tangent_space_dim:
            raise ValueError(
                f"Expected feature dimension {self.tangent_space_dim} for {self.n_channels} channels, got {F}."
            )

        # Remove dummy dimensions and swap time axes: (B, 1, F, T) -> (B, T, F)
        x = x.squeeze(1).permute(0, 2, 1)  # (B, 85, tangent_space_dim)

        # Project to d_model
        x = self.input_projection(x)                  # (B, 85, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)               # (B, 85, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)               # (B, 85, d_model)

        # Per-step output projection
        x = self.output_projection(x)                 # (B, 85, output_mel_bins)

        # Tanh activation function enforces consistent range scaling
        x = torch.tanh(x)

        # Re-arrange back to the original format layout: (B, 1, output_mel_bins, 85)
        x = x.permute(0, 2, 1).unsqueeze(1)          

        return x


# --- Helpers ---
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Test code using Option B Dimensions ---
if __name__ == "__main__":
    BATCH = 16
    EEG_CHANNELS = 27  # Standard channel resolution
    
    # (27 * 28) / 2 = 378 dimensional spatial features
    EXPECTED_TS_DIM = int((EEG_CHANNELS * (EEG_CHANNELS + 1)) / 2) 
    
    # Enforcing exactly Option B window slicing configuration
    TIME_STEPS = 85       
    OUTPUT_MEL_BINS = 80

    model = TransformerMelSynth(
        n_channels=EEG_CHANNELS,
        output_mel_bins=OUTPUT_MEL_BINS,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_dim_feedforward=512,
        transformer_num_layers=6,
        dropout=0.1,
    )

    print(f"EEG Channels: {EEG_CHANNELS} -> Input Dimension: {EXPECTED_TS_DIM}")
    print(f"Enforced Time Sequence: {TIME_STEPS} steps")
    print(f"Trainable parameters: {count_parameters(model):,}")


    dummy_input = torch.randn(BATCH, 1, EXPECTED_TS_DIM, TIME_STEPS)
    print(f"\nInput shape:  {tuple(dummy_input.shape)}")


    output = model(dummy_input)
    print(f"Output shape: {tuple(output.shape)}")
