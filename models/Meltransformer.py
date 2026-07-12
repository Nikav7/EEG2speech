# multi-layered (frame-by-frame) transformer inspired by the work of Wairagkar, 2025.
# Adapted for CSP EEG -> mel-spectrogram seq2seq synthesis.
# PyTorch version Veronica Valente (RUG), 2025/2026

#PIPELINE: 
# [EEG Timeline] -> [Transformer Generator] -> [Predicted Spectrogram]
# [Generated Mel] -> [vocoding, e.g. HiFi-GAN, default griffin-lim] -> [Waveform Tensor] -> [Wav2Vec2] -> [Continuous Logits] -> [CTC Loss] -> (Backpropagation)



import torch
import torch.nn as nn
import torch.nn.functional 
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
    Transformer-based EEG CSP -> mel-spectrogram sequence generator.

    Input:  (B, input_channels, input_freq_bins, T)  -- same 4-D format as ntGAN
    Output: (B, 1, output_mel_bins, T)               -- same 4-D format as ntGAN

    The time axis T is shared between EEG and mel (no temporal resampling).
    The Transformer treats T as the sequence dimension and
    (input_channels * input_freq_bins) as the per-step feature dimension.
    """

    def __init__(self,
                 input_channels: int = 1,
                 input_freq_bins: int = 32,
                 output_mel_bins: int = 80,
                 transformer_d_model: int = 128,
                 transformer_nhead: int = 8,
                 transformer_dim_feedforward: int = 512,
                 transformer_num_layers: int = 6,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000):
        super().__init__()

        self.input_channels = input_channels
        self.input_freq_bins = input_freq_bins
        self.output_mel_bins = output_mel_bins
        self.transformer_d_model = transformer_d_model

        input_dim = input_channels * input_freq_bins  # flattened spatial dim

        # Project flattened CSP features to transformer embedding dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, transformer_d_model * 2),
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
            x: (B, input_channels, input_freq_bins, T)

        Returns:
            (B, 1, output_mel_bins, T)
        """
        B, C, F, T = x.shape
        #print(f"Input shape: {x.shape}")
        # Interpolate only the time dimension if needed
        if T != 85:
            # x: (B, C, F, T) -> interpolate last dim (T) to 85, keep F unchanged
            x = torch.nn.functional.interpolate(x, size=(F, 85), mode='bilinear', align_corners=False)
            T = 85
        # Flatten spatial dims and permute: (B, C, F, T) -> (B, T, C*F)
        x = x.reshape(B, C * F, T).permute(0, 2, 1)  # (B, T, C*F)

        # Project to d_model
        x = self.input_projection(x)                  # (B, T, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)               # (B, T, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)               # (B, T, d_model)

        # Per-step output
        x = self.output_projection(x)                 # (B, T, output_mel_bins)

        # Tanh, try Relu if it doesn't work good, however tanh is common for spectrogram outputs to keep values in a reasonable range
        x = torch.tanh(x)

        # Reshape to (B, 1, output_mel_bins, T)
        x = x.permute(0, 2, 1).unsqueeze(1)          # (B, 1, output_mel_bins, T)

        return x


# --- Helpers ---
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Quick test ---
if __name__ == "__main__":
    # Typical CSP shapes from CSPAudioDataset: (B, 1, csp_components, T)
    BATCH = 16
    INPUT_CHANNELS = 1
    CSP_COMPONENTS = 105   # CSP components count
    TIME_STEPS = 85       # time dimension
    OUTPUT_MEL_BINS = 80

    model = TransformerMelSynth(
        input_channels=INPUT_CHANNELS,
        input_freq_bins=CSP_COMPONENTS,
        output_mel_bins=OUTPUT_MEL_BINS,
        transformer_d_model=128,
        transformer_nhead=8,
        transformer_dim_feedforward=512,
        transformer_num_layers=6,
        dropout=0.1,
    )

    print("--- Model Architecture ---")
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    dummy_input = torch.randn(BATCH, INPUT_CHANNELS, CSP_COMPONENTS, TIME_STEPS)
    print(f"\nInput shape:  {tuple(dummy_input.shape)}")

    try:
        output = model(dummy_input)
        print(f"Output shape: {tuple(output.shape)}")
        assert output.shape == (BATCH, 1, OUTPUT_MEL_BINS, TIME_STEPS), "Shape mismatch!"
        print("Shape check passed.")
    except Exception as e:
        print(f"Error: {e}")
