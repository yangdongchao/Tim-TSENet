import torch
from torch import nn
import torch.nn.functional as F
import sys
import pickle
sys.path.append('../')
from utils.util import check_parameters
from model.PANNS import ResNet38, CNN10
from model.tsd import TSD

def init_kernel(frame_len,
                frame_hop,
                num_fft=None,
                window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    if not num_fft:
        # FFT points
        fft_size = 2 ** math.ceil(math.log2(frame_len))
    else:
        fft_size = num_fft
    # window [window_length]
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    # window_length, F, 2 (real+imag)
    kernel = torch.rfft(torch.eye(fft_size) / S_, 1)[:frame_len]
    # 2, F, window_length
    kernel = torch.transpose(kernel, 0, 2) * window
    # 2F, 1, window_length
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 num_fft=None):
        super(STFTBase, self).__init__()
        K = init_kernel(
            frame_len,
            frame_hop,
            num_fft=num_fft,
            window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self):
        self.K.requires_grad = False

    def unfreeze(self):
        self.K.requires_grad = True

    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))

    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(
            self.window, self.stride, self.K.requires_grad, self.K.shape)


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Accept raw waveform and output magnitude and phase
        x: input signal, N x 1 x S or N x S
        m: magnitude, N x F x T
        p: phase, N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        self.check_nan()
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        # N x 2F x T
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        # N x F x T
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        m, p: N x F x T
        s: N x C x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        self.check_nan()
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        # N x 2F x T
        c = torch.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s



class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim)
    else:
        return nn.BatchNorm1d(dim)


class Conv1D(nn.Module):
    '''
       Build the Conv1D structure
       causal: if True is causal setting
    '''

    def __init__(self, in_channels=256, out_channels=512,
                 kernel_size=3, dilation=1, norm='gln', causal=False):
        super(Conv1D, self).__init__()
        self.causal = causal
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.PReLu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_channels)
        self.pad = (dilation * (kernel_size - 1)
                    ) // 2 if not causal else dilation * (kernel_size - 1)
        self.dwconv = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                                groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_channels)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1)

    def forward(self, x):
        """
          Input:
              x: [B x C x T], B is batch size, T is times
          Returns:
              x: [B, C, T]
        """
        # B x C x T -> B x C_o x T_o
        x_conv = self.conv1x1(x)
        x_conv = self.PReLu1(x_conv)
        x_conv = self.norm1(x_conv)
        # B x C_o x T_o
        x_conv = self.dwconv(x_conv)
        x_conv = self.PReLu2(x_conv)
        x_conv = self.norm2(x_conv)
        # B x C_o x T_o -> B x C x T
        if self.causal:
            x_conv = x_conv[:, :, :-self.pad]
        x_conv = self.end_conv1x1(x_conv)
        return x + x_conv

class Conv1D_emb(nn.Module):
    '''
       Build the Conv1D structure with embedding
       causal: if True is causal setting
    '''

    def __init__(self, in_channels=256, emb_channels=128, out_channels=512,
                 kernel_size=3, dilation=1, norm='gln', causal=False):
        super(Conv1D_emb, self).__init__()
        self.causal = causal
        self.conv1x1 = nn.Conv1d(in_channels+emb_channels, out_channels, kernel_size=1)
        self.PReLu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_channels)
        self.pad = (dilation * (kernel_size - 1)
                    ) // 2 if not causal else dilation * (kernel_size - 1)
        self.dwconv = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                                groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_channels)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1)

    def forward(self, x, emb):
        """
          Input:
              x: [B x C x T], B is batch size, T is times
              emb: [B x C']
          Returns:
              x: [B, C, T]
        """
        T = x.shape[-1]
        emb = torch.unsqueeze(emb, -1)
        # B x C' X T
        emb = emb.repeat(1, 1, T)
        # B x (C + C') X T
        x_ = torch.cat([x, emb], 1)
        # B x (C + C') X T -> B x C_o x T_o
        x_conv = self.conv1x1(x_)
        x_conv = self.PReLu1(x_conv)
        x_conv = self.norm1(x_conv)
        # B x C_o x T_o
        x_conv = self.dwconv(x_conv)
        x_conv = self.PReLu2(x_conv)
        x_conv = self.norm2(x_conv)
        # B x C_o x T_o -> B x C x T
        if self.causal:
            x_conv = x_conv[:, :, :-self.pad]
        x_conv = self.end_conv1x1(x_conv)
        return x + x_conv

class Encoder(nn.Module):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                kernel_size=kernel_size, stride=kernel_size // 2, groups=1)

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)

        return x


class Separation_TasNet(nn.Module):
    '''
       TasNet Separation part
       LayerNorm -> 1x1Conv -> 1-D Conv .... -> output
    '''

    def __init__(self, repeats=3, conv1d_block=8, in_channels=64, out_channels=128, emb_channels=128, final_channels=257,
                 out_sp_channels=512, kernel_size=3, norm='gln', causal=False, num_spks=2):
        super(Separation_TasNet, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        # self.conv1d_list = self._Sequential(
        #     repeats, conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
        #     kernel_size=kernel_size, norm=norm, causal=causal)
        self.conv_block_1_front = Conv1D_emb(in_channels=out_channels, emb_channels=emb_channels, out_channels=out_sp_channels,
                 kernel_size=kernel_size, dilation=1, norm=norm, causal=causal)
        self.conv_block_1_back = self._Sequential_block(conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
            kernel_size=kernel_size, norm=norm, causal=causal)
        self.conv_block_2_front = Conv1D_emb(in_channels=out_channels, emb_channels=emb_channels, out_channels=out_sp_channels,
                 kernel_size=kernel_size, dilation=1, norm=norm, causal=causal)
        self.conv_block_2_back = self._Sequential_block(conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
            kernel_size=kernel_size, norm=norm, causal=causal)
        self.conv_block_3_front = Conv1D_emb(in_channels=out_channels, emb_channels=emb_channels, out_channels=out_sp_channels,
                 kernel_size=kernel_size, dilation=1, norm=norm, causal=causal)
        self.conv_block_3_back = self._Sequential_block(conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
            kernel_size=kernel_size, norm=norm, causal=causal)
        # self.conv_block_4_front = Conv1D_emb(in_channels=out_channels, emb_channels=emb_channels, out_channels=out_sp_channels,
        #          kernel_size=kernel_size, dilation=1, norm=norm, causal=causal)
        # self.conv_block_4_back = self._Sequential_block(conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
        #     kernel_size=kernel_size, norm=norm, causal=causal)

        self.PReLu = nn.PReLU()
        self.norm = select_norm('cln', in_channels)
        self.end_conv1x1 = nn.Conv1d(out_channels, num_spks * final_channels, 1)
        # self.activation = nn.Sigmoid()
        self.num_spks = num_spks

    def _Sequential_block(self, num_blocks, **block_kwargs):
        '''
           Sequential 1-D Conv Block
           input:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        Conv1D_lists = [Conv1D(
            **block_kwargs, dilation=(2 ** i)) for i in range(num_blocks)]

        return nn.Sequential(*Conv1D_lists)

    def _Sequential(self, num_repeats, num_blocks, **block_kwargs):
        '''
           Sequential repeats
           input:
                 num_repeats: Number of repeats
                 num_blocks: Number of block in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        repeats_lists = [self._Sequential_block(
            num_blocks, **block_kwargs) for i in range(num_repeats)]
        return nn.Sequential(*repeats_lists)

    def forward(self, x, emb):
        """
           Input:
               x: [B x C x T], B is batch size, T is times
               emb: [B x C x T], B is batch size, T is times
           Returns:
               x: [num_spks, B, N, T]
         """
        # B x C x T
        x = self.norm(x)
        x = self.conv1x1(x)
        # B x C x T
        x = self.conv_block_1_front(x, emb)
        x = self.conv_block_1_back(x)
        x = self.conv_block_2_front(x, emb)
        x = self.conv_block_2_back(x)
        x = self.conv_block_3_front(x, emb)
        x = self.conv_block_3_back(x)
        # x = self.conv_block_4_front(x, emb)
        # x = self.conv_block_4_back(x)
        # B x N x T
        x = self.PReLu(x)
        x = self.end_conv1x1(x)
        # x = self.activation(x)
        return x


class TSDNet(nn.Module):
    '''
       TSDNet module
    '''

    def __init__(self, nFrameLen=512, nFrameShift=256, cls_num=50):
        super(TSDNet, self).__init__()
        self.PReLu = nn.PReLU()
        self.encoder_ref = CNN10(sample_rate=16000, window_size=1024,
                                    hop_size=320, mel_bins=64, fmin=50, fmax=8000,
                                    classes_num=527)
        self.cls1 = nn.Linear(128, 128)
        self.cls2 = nn.Linear(128, cls_num)
        self.init_ref()
        self.emb_fc = nn.Linear(512, 128)
        self.tsd = TSD(sample_rate=16000, window_size=nFrameLen,
                                    hop_size=nFrameShift, mel_bins=64, fmin=50, fmax=8000)
    def init_ref(self):
        device = torch.device('cuda')
        checkpoint_path = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/model/Cnn10_mAP=0.380.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.encoder_ref.load_state_dict(checkpoint['model'])

    def forward(self, x, ref):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
        """
        out_enc = self.encoder_ref(ref)
        emb = self.emb_fc(out_enc)
        emb = self.PReLu(emb)
        out_tsd_up, out_tsd_time = self.tsd(x, emb)
        x_cls = self.PReLu(self.cls1(emb))
        x_cls = F.dropout(x_cls, p=0.5, training=self.training)
        x_cls = self.cls2(x_cls)
        x_cls = F.log_softmax(x_cls, dim=-1)

        return x_cls, out_tsd_time


if __name__ == "__main__":
    conv = TSDNet().cuda()
    # encoder = Encoder(16, 512)
    x = torch.randn(4, 64000).cuda()
    label = torch.randn(4, 64000).cuda()
    ref = torch.randn(4, 64000).cuda()
    audio, lps, lab = conv(x, ref, label)
    print(audio[0].shape)
    # print("{:.3f}".format(check_parameters(conv)))

