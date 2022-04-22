import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from utils.util import check_parameters
from model.PANNS import CNN10
from model.tsd import TSD, TSD2, TSD2_tse, TSD_plus, TSD_plus_sim, TSD_IS,TSD_regresion,TSD_regresion_two_cls
import math

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
                 kernel_size=3, dilation=1, norm='gln', causal=False, fusion='concat', usingEmb=True, usingTsd=False):
        super(Conv1D_emb, self).__init__()
        self.causal = causal
        self.usingTsd = usingTsd
        self.usingEmb = usingEmb
        self.fusion = fusion # concat, add, multiply
        if usingEmb:
            if fusion == 'concat':
                if not usingTsd:
                    self.conv1x1 = nn.Conv1d(in_channels + emb_channels, out_channels, kernel_size=1)
                else:
                    self.conv1x1 = nn.Conv1d(in_channels + emb_channels + 1, out_channels, kernel_size=1)
            elif fusion == 'add':
                self.preCNN = nn.Conv1d(emb_channels, in_channels, kernel_size=1)
                if not usingTsd:
                    self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
                else:
                    self.conv1x1 = nn.Conv1d(in_channels + 1, out_channels, kernel_size=1)
            elif fusion == 'multiply':
                self.preCNN = nn.Conv1d(emb_channels, in_channels, kernel_size=1)
                if not usingTsd:
                    self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
                else:
                    self.conv1x1 = nn.Conv1d(in_channels + 1, out_channels, kernel_size=1)
        else:
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

    def forward(self, x, emb=None, tsd=None):
        """
          Input:
              x: [B x C x T], B is batch size, T is times
              emb: [B x C']
              tsd: [B x 1 x T]
          Returns:
              x: [B, C, T]
        """
        T = x.shape[-1]
        emb = torch.unsqueeze(emb, -1)
        # B x C' X T
        emb = emb.repeat(1, 1, T)
        # B x (C + C') X T
        if self.usingEmb:
            if self.fusion == 'concat':
                if not self.usingTsd:
                    x_ = torch.cat([x, emb], 1)
                else:
                    x_ = torch.cat([x, emb, tsd], 1)
            elif self.fusion == 'add':
                x_ = self.PReLu1(self.preCNN(emb)) + x
                if not self.usingTsd:
                    x_ = x_
                else:
                    x_ = torch.cat([x_, tsd], 1)
            elif self.fusion == 'multiply':
                x_ = self.PReLu1(self.preCNN(emb)) * x
                if not self.usingTsd:
                    x_ = x_
                else:
                    x_ = torch.cat([x_, tsd], 1)
        else:
            x_ = x
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

class ExtractionNet(nn.Module):
    '''
       TasNet Separation part
       LayerNorm -> 1x1Conv -> 1-D Conv .... -> output
    '''

    def __init__(self, conv1d_block=8, in_channels=64, out_channels=128, emb_channels=128, final_channels=257,
                 out_sp_channels=512, kernel_size=3, norm='gln', causal=False, num_spks=1, fusion='concat', usingEmb=[True,True,True], usingTsd=[False,False,False]):
        super(ExtractionNet, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv_block_1_front = Conv1D_emb(in_channels=out_channels, emb_channels=emb_channels, out_channels=out_sp_channels,
                 kernel_size=kernel_size, dilation=1, norm=norm, causal=causal, fusion=fusion, usingEmb=usingEmb[0], usingTsd=usingTsd[0])
        self.conv_block_1_back = self._Sequential_block(conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
            kernel_size=kernel_size, norm=norm, causal=causal)
        self.conv_block_2_front = Conv1D_emb(in_channels=out_channels, emb_channels=emb_channels, out_channels=out_sp_channels,
                 kernel_size=kernel_size, dilation=1, norm=norm, causal=causal, fusion=fusion, usingEmb=usingEmb[1], usingTsd=usingTsd[1])
        self.conv_block_2_back = self._Sequential_block(conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
            kernel_size=kernel_size, norm=norm, causal=causal)
        self.conv_block_3_front = Conv1D_emb(in_channels=out_channels, emb_channels=emb_channels, out_channels=out_sp_channels,
                 kernel_size=kernel_size, dilation=1, norm=norm, causal=causal, fusion=fusion, usingEmb=usingEmb[2], usingTsd=usingTsd[2])
        self.conv_block_3_back = self._Sequential_block(conv1d_block, in_channels=out_channels, out_channels=out_sp_channels,
            kernel_size=kernel_size, norm=norm, causal=causal)

        self.PReLu = nn.PReLU()
        self.norm = select_norm('cln', in_channels)
        self.end_conv1x1 = nn.Conv1d(out_channels, num_spks * final_channels, 1)
        self.activation = nn.Sigmoid()
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

    def forward(self, x, emb=None, tsd=None):
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
        x = self.PReLu(x)
        # B x C x T
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block_1_front(x, emb, tsd)
        x = self.conv_block_1_back(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block_2_front(x, emb, tsd)
        x = self.conv_block_2_back(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block_3_front(x, emb, tsd)
        x = self.conv_block_3_back(x)
        x = F.dropout(x, p=0.2, training=self.training)
        # B x N x T
        x = self.PReLu(x)
        x = self.end_conv1x1(x)
        x = self.activation(x)
        return x

class TSENet(nn.Module):
    '''
       TSENet module
       N	Number of ﬁlters in autoencoder
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''

    def __init__(self,
                 N=512,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 norm="gln",
                 num_spks=1,
                 causal=False,
                 cls_num=50,
                 nFrameLen=512,
                 nFrameShift=256,
                 nFFT=512,
                 fusion='concat',
                 usingEmb=[True,True,True],
                 usingTsd=[False,False,False],
                 CNN10_settings=[16000,1024,320,64,50,8000,527,512,128],
                 fixCNN10=False,
                 fixTSDNet=True,
                 pretrainedCNN10=None,
                 pretrainedTSDNet=None,
                 threshold=0.5,
                 ):
        super(TSENet, self).__init__()
        self.device = torch.device('cuda')
        self.stft = STFT(frame_len=nFrameLen, frame_hop=nFrameShift, num_fft=nFFT)
        self.istft = iSTFT(frame_len=nFrameLen, frame_hop=nFrameShift, num_fft=nFFT)
        self.front_CNN = nn.Conv1d(nFrameShift+1, N, 1)
        self.PReLu = nn.PReLU()
        self.extractor = ExtractionNet(conv1d_block=X, in_channels=N,
                                            out_channels=B, final_channels=nFrameShift + 1, out_sp_channels=H, kernel_size=P,
                                            norm=norm, causal=causal, num_spks=num_spks, fusion=fusion,
                                            usingEmb=usingEmb, usingTsd=usingTsd)
        self.num_spks = num_spks
        self.conditioner = CNN10(sample_rate=CNN10_settings[0], window_size=CNN10_settings[1],
                                    hop_size=CNN10_settings[2], mel_bins=CNN10_settings[3], fmin=CNN10_settings[4], fmax=CNN10_settings[5],
                                    classes_num=CNN10_settings[6])
        self.cls1 = nn.Linear(CNN10_settings[7], CNN10_settings[8])
        self.cls2 = nn.Linear(CNN10_settings[8], cls_num)
        self.fixCNN10 = fixCNN10
        self.fixTSDNet = fixTSDNet
        self.pretrainedCNN10 = pretrainedCNN10
        self.pretrainedTSDNet = pretrainedTSDNet
        self.usingEmb = usingEmb
        self.usingTsd = usingTsd
        self.threshold = threshold
        self.init_conditioner()
        self.emb_fc = nn.Linear(CNN10_settings[7], CNN10_settings[8])
        self.onehot = nn.Embedding(cls_num, CNN10_settings[8])
        if usingTsd[0] or usingTsd[1] or usingTsd[2]:
            self.tsdnet = TSDNet(nFrameLen=nFrameLen, nFrameShift=nFrameShift, cls_num=cls_num, CNN10_settings=CNN10_settings)
            self.init_TSDNet()
        self.epsilon = 1e-20

    def init_conditioner(self):
        if self.pretrainedCNN10:
            device = torch.device('cuda')
            checkpoint = torch.load(self.pretrainedCNN10, map_location=device)
            self.conditioner.load_state_dict(checkpoint['model'])
        if self.fixCNN10:
            for p in self.conditioner.parameters():
                p.requires_grad = False

    def init_TSDNet(self):
        if self.pretrainedTSDNet:
            device = torch.device('cuda')
            dicts = torch.load(self.pretrainedTSDNet, map_location=device)
            self.tsdnet.load_state_dict(dicts["model_state_dict"])
        if self.fixTSDNet:
            for p in self.tsdnet.parameters():
                p.requires_grad = False

    def forward(self, x, ref, cls_index, label=None, inf=False):
        """
          Input:
              x: [B, T], B is batch size, T is times
              ref: [B, T], B is batch size, T is times
          Returns:
              audio: [B, T]
        """
        # B x T -> B x C x T
        x_magnitude, x_phase = self.stft(x)
        x_encoder = torch.log(x_magnitude ** 2 + self.epsilon)  # bs, 257, 249
        if not inf:
            label_magnitude, label_phase = self.stft(label)

        if self.usingTsd[0] or self.usingTsd[1] or self.usingTsd[2]:
            _, _, out_tsd_up = self.tsdnet(x, ref)
            tsdMask = torch.zeros(x_magnitude.shape[0], x_magnitude.shape[2]).cuda()
            tsdMask[out_tsd_up > self.threshold] = 1.
            tsdMask = tsdMask[:, None, :]
        else:
            tsdMask = None
        # B x T -> B x C -> B x C x T
        out_enc = self.conditioner(ref)
        emb = self.emb_fc(out_enc)
        emb = self.PReLu(emb)
        x_cls = self.PReLu(self.cls1(out_enc))
        x_cls = F.dropout(x_cls, p=0.5, training=self.training)
        x_cls = self.cls2(x_cls)
        x_cls = F.log_softmax(x_cls, dim=-1)

        emb_onehot = self.onehot(cls_index)
        emb_onehot = F.dropout(emb_onehot, p=0.2, training=self.training)
        x_encoder = self.PReLu(self.front_CNN(x_encoder))
        # mask = self.extractor(x_encoder, emb, tsdMask)
        mask = self.extractor(x_encoder, emb_onehot, tsdMask)
        x_ex = x_magnitude * mask
        gt = label_magnitude / (x_magnitude + self.epsilon) * torch.cos(label_phase - x_phase) # PSM
        gt = torch.clamp(gt, min=0., max=1.) # Truncated to [0,1]
        audio_encoder = self.istft(x_ex, x_phase)
        audio = [audio_encoder[:, 0]]

        return audio, mask, gt, x_cls, emb, emb_onehot

class TSDNet(nn.Module):
    '''
       TSDNet module
    '''
    def __init__(self, nFrameLen=512, nFrameShift=256, cls_num=41, CNN10_settings=[16000,1024,320,64,50,8000,527,512,128], pretrainedCNN10=None):
        super(TSDNet, self).__init__()
        self.PReLu = nn.PReLU()
        self.conditioner = CNN10(sample_rate=16000, window_size=1024,
                                    hop_size=320, mel_bins=64, fmin=50, fmax=8000,
                                    classes_num=527)
        self.cls1 = nn.Linear(128, 128)
        self.cls2 = nn.Linear(128, cls_num)
        self.pretrainedCNN10 = pretrainedCNN10
        self.init_ref()
        self.emb_fc = nn.Linear(512, 128)
        # print(CNN10_settings)
        self.tsd = TSD2(sample_rate=CNN10_settings[0], window_size=nFrameLen,
                                    hop_size=nFrameShift, mel_bins=CNN10_settings[3], fmin=CNN10_settings[4], fmax=CNN10_settings[5])
        # print('self.tsd ',self.tsd)
        self.init_fc_layer(self.cls1) # new add
        self.init_fc_layer(self.cls2) # new add
        self.init_fc_layer(self.emb_fc) # new add
        # assert 1==2
    def init_ref(self):
        if self.pretrainedCNN10:
            device = torch.device('cuda')
            checkpoint = torch.load(self.pretrainedCNN10, map_location=device)
            self.conditioner.load_state_dict(checkpoint['model'])
    
    def init_rnn_layer(self, layer):
        for name, param in layer.named_parameters():
            if name.startswith("weight"):
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)

    def init_fc_layer(self, layer):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.)
    
    def forward(self, x, ref):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
        """
        out_enc = self.conditioner(ref)
        emb = self.emb_fc(out_enc)
        emb = self.PReLu(emb)
        out_tsd_time, out_tsd_up, _ = self.tsd(x, emb)
        x_cls = self.PReLu(self.cls1(emb))
        x_cls = F.dropout(x_cls, p=0.5, training=self.training)
        x_cls = self.cls2(x_cls)
        x_cls = F.log_softmax(x_cls, dim=-1)

        return x_cls, out_tsd_time, out_tsd_up

class TSDNet_tse(nn.Module):
    '''
       TSDNet module
    '''
    def __init__(self, nFrameLen=512, nFrameShift=256, cls_num=41, CNN10_settings=[16000,1024,320,64,50,8000,527,512,128], pretrainedCNN10=None, use_frame=False, only_ref=True):
        super(TSDNet_tse, self).__init__()
        self.PReLu = nn.PReLU()
        self.conditioner = CNN10(sample_rate=16000, window_size=1024,
                                    hop_size=320, mel_bins=64, fmin=50, fmax=8000,
                                    classes_num=527)
        self.cls1 = nn.Linear(128, 128)
        self.cls2 = nn.Linear(128, cls_num)
        self.pretrainedCNN10 = pretrainedCNN10
        self.init_ref()
        self.emb_fc = nn.Linear(512, 128)
        self.only_ref = only_ref
        # print(CNN10_settings)
        self.tsd = TSD2_tse(sample_rate=CNN10_settings[0], window_size=nFrameLen,
                                    hop_size=nFrameShift, mel_bins=CNN10_settings[3], fmin=CNN10_settings[4], fmax=CNN10_settings[5], use_frame=use_frame)
        # print('self.tsd ',self.tsd)
        # assert 1==2
        self.init_fc_layer(self.cls1) # new add
        self.init_fc_layer(self.cls2) # new add
        self.init_fc_layer(self.emb_fc) # new add
    def init_ref(self):
        if self.pretrainedCNN10:
            device = torch.device('cuda')
            checkpoint = torch.load(self.pretrainedCNN10, map_location=device)
            self.conditioner.load_state_dict(checkpoint['model'])
    
    def init_rnn_layer(self, layer):
        for name, param in layer.named_parameters():
            if name.startswith("weight"):
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)

    def init_fc_layer(self, layer):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.)
    def forward(self, x, ref, tse_audio):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
        """
        # clip level condition
        # clip_ref = torch.cat([ref, tse_audio],dim=1)
        if self.only_ref:
            clip_ref = ref # 只用ref
        else:
            clip_ref = torch.cat([ref, tse_audio],dim=1)
        clip_enc = self.conditioner(clip_ref)
        #print('clip_enc ',clip_enc.shape)
        clip_emb = self.emb_fc(clip_enc)
        clip_emb = self.PReLu(clip_emb)
        # frame level condition
        frame_enc = self.conditioner.extract_frame(tse_audio)
        frame_emb = self.emb_fc(frame_enc)
        frame_emb = self.PReLu(frame_emb)

        out_tsd_time, out_tsd_up = self.tsd(x, clip_emb, frame_emb)
        x_cls = self.PReLu(self.cls1(clip_emb))
        x_cls = F.dropout(x_cls, p=0.5, training=self.training)
        x_cls = self.cls2(x_cls)
        x_cls = F.log_softmax(x_cls, dim=-1)

        return x_cls, out_tsd_time, out_tsd_up


class TSDNet_one_hot(nn.Module):
    '''
       TSDNet module
    '''
    def __init__(self, nFrameLen=512, nFrameShift=256, cls_num=41, CNN10_settings=[16000,1024,320,64,50,8000,527,512,128], pretrainedCNN10=None):
        super(TSDNet_one_hot, self).__init__()
        self.PReLu = nn.PReLU()
        # self.conditioner = CNN10(sample_rate=16000, window_size=1024,
        #                             hop_size=320, mel_bins=64, fmin=50, fmax=8000,
        #                             classes_num=527)
        # self.cls1 = nn.Linear(128, 128)
        # self.cls2 = nn.Linear(128, cls_num)
        # self.pretrainedCNN10 = pretrainedCNN10
        # self.init_ref()
        # self.emb_fc = nn.Linear(512, 128)
        self.conditioner_one_hot = nn.Embedding(cls_num,128)
        # print(CNN10_settings)
        self.tsd = TSD(sample_rate=CNN10_settings[0], window_size=nFrameLen,
                                    hop_size=nFrameShift, mel_bins=CNN10_settings[3], fmin=CNN10_settings[4], fmax=CNN10_settings[5])
        # print('self.tsd ',self.tsd)
        # assert 1==2
    def init_ref(self):
        if self.pretrainedCNN10:
            device = torch.device('cuda')
            checkpoint = torch.load(self.pretrainedCNN10, map_location=device)
            self.conditioner.load_state_dict(checkpoint['model'])

    def forward(self, x, ref, onehot=None):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
        """
        # out_enc = self.conditioner(ref)
        # emb = self.emb_fc(out_enc)
        # emb = self.PReLu(emb)
        emb_onehot = self.conditioner_one_hot(onehot)
        # print('emb_onehot ',emb_onehot.shape)
        # assert 1==2
        out_tsd_time, out_tsd_up, sim_cos = self.tsd(x, emb_onehot)
        # x_cls = self.PReLu(self.cls1(emb))
        # x_cls = F.dropout(x_cls, p=0.5, training=self.training)
        # x_cls = self.cls2(x_cls)
        # x_cls = F.log_softmax(x_cls, dim=-1)
        x_cls = torch.zeros(1).cuda()
        return x_cls, out_tsd_time, out_tsd_up, sim_cos # st,ed


class TSDNet_plus_one_hot(nn.Module):
    '''
       TSDNet module
    '''
    def __init__(self, nFrameLen=512, nFrameShift=256, cls_num=41, CNN10_settings=[16000,1024,320,64,50,8000,527,512,128], pretrainedCNN10=None):
        super(TSDNet_plus_one_hot, self).__init__()
        self.PReLu = nn.PReLU()
        # self.conditioner = CNN10(sample_rate=16000, window_size=1024,
        #                             hop_size=320, mel_bins=64, fmin=50, fmax=8000,
        #                             classes_num=527)
        # self.cls1 = nn.Linear(128, 128)
        # self.cls2 = nn.Linear(128, cls_num)
        # self.pretrainedCNN10 = pretrainedCNN10
        # self.init_ref()
        # self.emb_fc = nn.Linear(512, 128)
        self.conditioner_one_hot = nn.Embedding(cls_num,128)
        # print(CNN10_settings)
        self.tsd = TSD_plus_sim(sample_rate=CNN10_settings[0], window_size=nFrameLen,
                                    hop_size=nFrameShift, mel_bins=CNN10_settings[3], fmin=CNN10_settings[4], fmax=CNN10_settings[5])
        # print('self.tsd ',self.tsd)
        # assert 1==2
    def init_ref(self):
        if self.pretrainedCNN10:
            device = torch.device('cuda')
            checkpoint = torch.load(self.pretrainedCNN10, map_location=device)
            self.conditioner.load_state_dict(checkpoint['model'])

    def forward(self, x, ref,onehot=None):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
        """
        # out_enc = self.conditioner(ref)
        # emb = self.emb_fc(out_enc)
        # emb = self.PReLu(emb)
        emb_onehot = self.conditioner_one_hot(onehot)
        # print('emb_onehot ',emb_onehot.shape)
        # assert 1==2
        out_tsd_up, out_tsd_time,sim_cos = self.tsd(x, emb_onehot)
        # x_cls = self.PReLu(self.cls1(emb))
        # x_cls = F.dropout(x_cls, p=0.5, training=self.training)
        # x_cls = self.cls2(x_cls)
        # x_cls = F.log_softmax(x_cls, dim=-1)
        x_cls = torch.zeros(1).cuda()

        return x_cls, out_tsd_time, out_tsd_up, sim_cos



if __name__ == "__main__":
    conv = Conv_TasNet().cuda()
    # encoder = Encoder(16, 512)
    x = torch.randn(4, 64000).cuda()
    label = torch.randn(4, 64000).cuda()
    ref = torch.randn(4, 64000).cuda()
    audio, lps, lab = conv(x, ref, label)
    print(audio[0].shape)
    # print("{:.3f}".format(check_parameters(conv)))






