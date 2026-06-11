import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm
import math

LRELU_SLOPE = 0.1

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


def remove_weight_norm(module: nn.Module) -> None:
    """Backward-compatible remover for parametrizations.weight_norm."""
    try:
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    except (ValueError, AttributeError):
        # Module may not have a weight-norm parametrization.
        pass


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1,3,5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1, 
                       dilation=dilation[0],                               
                       padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1,                                
                       dilation=dilation[1],                               
                       padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1,                                
                       dilation=dilation[2],                               
                       padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels, 
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels, 
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
            
            
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.i_mid = 0
        self.i_mid_gru = 1
        
        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch, 
                   h.ch_init_upsample//2,
                   3, 1, 
                   padding=get_padding(3,1)))
        
        
        self.GRU = nn.GRU(h.ch_init_upsample//2, 
                          h.ch_init_upsample//4, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, 
                                       h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.ch_init_upsample//(2**i), 
                                h.ch_init_upsample//(2**(i+1)),
                                k, u, padding=(k-u)//2)))
            
        self.conv_mid1 = weight_norm(
            Conv1d(h.ch_init_upsample//(2**self.i_mid), 
                   h.ch_init_upsample//(2**self.i_mid), 
                   3, 1, 
                   padding=0))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.ch_init_upsample//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, 
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))

        self.conv_post = weight_norm(
            Conv1d(ch, 
                   h.out_ch, 
                   9, 1, 
                   padding=get_padding(9,1)))
        
        self.conv_pre.apply(init_weights)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.conv_mid1.apply(init_weights)

    def forward(self, x):
        input_was_4d = (x.dim() == 4)
        input_freq_bins = None
        input_time_steps = None
        target_time_steps = int(getattr(self.h, 'out_time_steps', 85))
        if input_was_4d:
            batch_size, spec_channels, freq_bins, time_steps = x.shape
            input_freq_bins = freq_bins
            input_time_steps = time_steps
            x = x.reshape(batch_size, spec_channels * freq_bins, time_steps)

        x = self.conv_pre(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)

        for i in range(self.num_upsamples):
            # to match the output size
            if i == self.i_mid:
                x = self.conv_mid1(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        if input_was_4d:
            out_freq_bins = int(getattr(self.h, 'out_freq_bins', input_freq_bins))
            out_spec_channels = int(getattr(self.h, 'out_spec_channels', 1))
            if x.size(1) != out_spec_channels * out_freq_bins:
                if x.size(1) % out_freq_bins != 0:
                    raise ValueError(
                        f"Generator output channels ({x.size(1)}) cannot be reshaped to spectrogram "
                        f"with out_freq_bins={out_freq_bins}. Set h.out_ch/out_freq_bins consistently."
                    )
                out_spec_channels = x.size(1) // out_freq_bins
            x = x.reshape(x.size(0), out_spec_channels, out_freq_bins, x.size(2))
            if x.size(-1) != target_time_steps:
                x = F.interpolate(
                    x,
                    size=(x.size(2), target_time_steps),
                    mode='bilinear',
                    align_corners=False,
                )
        # elif x.size(-1) != target_time_steps:
        #     x = F.interpolate(x, size=target_time_steps, mode='linear', align_corners=False)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        remove_weight_norm(self.conv_mid1)


class Discriminator(torch.nn.Module):
    def __init__(self, h):
        super(Discriminator, self).__init__()
        self.h = h
        self.ch_init_downsample = h.ch_init_downsample
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_downsamples = len(h.downsample_rates)
        self.n_classes = h.n_classes
        self.input_size = int(getattr(h, "input_size", 0))
        self.m = 1
        
        for j in range(len(h.downsample_rates)):
            self.m = self.m * h.downsample_rates[j]
        
        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch, 
                   h.ch_init_downsample,
                   3, 1, 
                   padding=get_padding(3,1)))
        
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates, 
                                       h.downsample_kernel_sizes)):
            self.downs.append(weight_norm(
                Conv1d(h.ch_init_downsample*(2**i), 
                       h.ch_init_downsample*(2**(i+1)),
                       k, u, padding=math.ceil((k-u)/2))))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h.ch_init_downsample*(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, 
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))
        
        self.GRU = nn.GRU(ch, ch//2,
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        self.conv_post = weight_norm(Conv1d(ch, ch, 9, 1, padding=get_padding(9,1)))
        
        # FC Layer (input-size agnostic)
        self.adv_classifier = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid(),
        )
        self.aux_classifier = nn.LazyLinear(h.n_classes)
        
        self.conv_pre.apply(init_weights)
        self.downs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        if x.dim() == 4:
            batch_size, spec_channels, freq_bins, time_steps = x.shape
            x = x.reshape(batch_size, spec_channels * freq_bins, time_steps)

        x = self.conv_pre(x)

        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)

        # FC Layer
        x = x.reshape(x.size(0), -1)
        validity = self.adv_classifier(x)
        label = self.aux_classifier(x)
        
        return validity, label

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.downs:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
            

# substituting GRU with attention layer

# class Generator(torch.nn.Module):
#     def __init__(self, h):
#         super(Generator, self).__init__()
#         self.h = h
#         self.num_kernels = len(h.resblock_kernel_sizes)
#         self.num_upsamples = len(h.upsample_rates)
#         self.i_mid = 0
        
#         # Initial projection to the hidden dimension
#         # Assuming h.ch_init_upsample is your base channel size
#         hidden_dim = h.ch_init_upsample // 2
#         self.conv_pre = LazyConv1d(hidden_dim, 3, 1, padding=get_padding(3, 1))
        
#         # --- Transformer Encoder Layer ---
#         # nhead should divide hidden_dim evenly (e.g., if hidden_dim=256, nhead=8)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim, 
#             nhead=8, 
#             dim_feedforward=hidden_dim * 4,
#             dropout=0.1,
#             activation='relu',
#             batch_first=True 
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
#         # ---------------------------------

#         self.ups = nn.ModuleList()
#         for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
#             self.ups.append(weight_norm(
#                 ConvTranspose1d(h.ch_init_upsample // (2**i), 
#                                 h.ch_init_upsample // (2**(i+1)),
#                                 k, u, padding=(k-u)//2)))
        
#         # ... rest of __init__ (resblocks, conv_post, etc.) ...

#     def forward(self, x):
#         # handle 4D to 3D reshape

#         #Initial Convolution
#         x = self.conv_pre(x) # [Batch, Channels, Time]
        
#         #Transformer Processing
#         # Transformer expects [Batch, Seq_Len, Features], so we transpose
#         x = x.transpose(1, 2) 
#         x = self.transformer_encoder(x)
#         x = x.transpose(1, 2) # Back to [Batch, Channels, Time]

#         #Upsampling and ResBlocks
#         for i in range(self.num_upsamples):
#             if i == self.i_mid:
#                 x = self.conv_mid1(x)
#             x = F.leaky_relu(x, LRELU_SLOPE)
#             x = self.ups[i](x)
            
#             xs = None
#             for j in range(self.num_kernels):
#                 if xs is None:
#                     xs = self.resblocks[i*self.num_kernels+j](x)
#                 else:
#                     xs += self.resblocks[i*self.num_kernels+j](x)
#             x = xs / self.num_kernels
            
#         # output layer(s)
#         return x