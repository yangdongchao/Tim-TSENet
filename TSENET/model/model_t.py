import torch
import torch.nn as nn
import torch.nn.functional as F

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
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
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
    if norm not in ['gln', 'cln', 'bn']:
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)

class Conv1D_e(nn.Module):
    '''
       Build the Conv1D structure
       causal: if True is causal setting
    '''
    def __init__(self, in_channels=256, out_channels=512,
                 kernel_size=3, dilation=1, norm='gln', causal=False):
        super(Conv1D_e, self).__init__()
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


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    '''
       This module can be seen as the gradient of Conv1d with respect to its input. 
       It is also known as a fractionally-strided convolution 
       or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(nn.Module):
    '''
       Consider only residual links
    '''

    def __init__(self, in_channels=256, out_channels=512,
                 kernel_size=3, dilation=1, norm='gln', causal=False):
        super(Conv1D_Block, self).__init__()
        # conv 1 x 1
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        # not causal don't need to padding, causal need to pad+1 = kernel_size
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise convolution
        self.dwconv = Conv1D(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        self.Sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

    def forward(self, x):
        # x: N x C x L
        # N x O_C x L
        c = self.conv1x1(x)
        # N x O_C x L
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        # causal: N x O_C x (L+pad)
        # noncausal: N x O_C x L
        c = self.dwconv(c)
        # N x O_C x L
        if self.causal:
            c = c[:, :, :-self.pad]
        c = self.PReLU_2(c)
        c = self.norm_2(c)
        c = self.Sc_conv(c)
        return x+c


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
        Conv1D_lists = [Conv1D_e(
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
        # print('x ',x.shape)
        x = self.PReLu(x)
        x = self.end_conv1x1(x)
        # print('x ', x.shape)
        # assert 1==2
        x = self.activation(x)
        return x



class ConvTasNet(nn.Module):
    '''
       ConvTasNet module
       N	Number of ﬁlters in autoencoder
       L	Length of the ﬁlters (in samples)
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       Sc	Number of channels in skip-connection paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''
    def __init__(self,
                 N=128,
                 L=20,
                 B=128,
                 H=256,
                 P=3,
                 X=8,
                 R=3,
                 norm="gln",
                 num_spks=1,
                 activate="relu",
                 causal=False,
                 cls_num=41,
                 nFrameLen=512,
                 nFrameShift=256, # 
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
        super(ConvTasNet, self).__init__()
        # n x 1 x T => n x N x T
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        # n x N x T  Layer Normalization of Separation
        self.LayerN_S = select_norm('cln', N)
        # n x B x T  Conv 1 x 1 of  Separation
        self.BottleN_S = Conv1D(N, B, 1)
        # Separation block
        # n x B x T => n x B x T
        # self.separation = self._Sequential_repeat(
        #     R, X, in_channels=B, out_channels=H, kernel_size=P, norm=norm, causal=causal)
        self.extractor = ExtractionNet(conv1d_block=X, in_channels=N,
                                        out_channels=B, final_channels=N, out_sp_channels=H, kernel_size=P,
                                        norm=norm, causal=causal, num_spks=num_spks, fusion=fusion,
                                        usingEmb=usingEmb, usingTsd=usingTsd)
        self.conditioner_one_hot = nn.Embedding(cls_num, 128)
        self.fixCNN10 = fixCNN10
        self.fixTSDNet = fixTSDNet
        self.pretrainedCNN10 = pretrainedCNN10
        self.pretrainedTSDNet = pretrainedTSDNet
        self.usingEmb = usingEmb
        self.usingTsd = usingTsd
        self.threshold = threshold
        # self.init_conditioner() # init conditioner modual
        self.emb_fc = nn.Linear(CNN10_settings[7], CNN10_settings[8]) # produce embedding
        # if usingTsd[0] or usingTsd[1] or usingTsd[2]: # if we decide to use tsdNet
        #     self.tsdnet = TSDNet(nFrameLen=nFrameLen, nFrameShift=nFrameShift, cls_num=cls_num, CNN10_settings=CNN10_settings)
        #     self.init_TSDNet() # init it
        self.epsilon = 1e-20
        # n x B x T => n x 2*N x T
        self.gen_masks = Conv1D(B, num_spks*N, 1)
        # n x N x T => n x 1 x L
        self.decoder = ConvTrans1D(N, 1, L, stride=L//2)
        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation_type = activate
        self.activation = active_f[activate]
        self.num_spks = num_spks

    def init_conditioner(self):
        if self.pretrainedCNN10:
            device = torch.device('cuda')
            checkpoint = torch.load(self.pretrainedCNN10, map_location=device)
            self.conditioner.load_state_dict(checkpoint['model'])
        if self.fixCNN10: # if fix it
            for p in self.conditioner.parameters():
                p.requires_grad = False

    def _Sequential_block(self, num_blocks, **block_kwargs):
        '''
           Sequential 1-D Conv Block
           input:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        Conv1D_Block_lists = [Conv1D_Block(
            **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

        return nn.Sequential(*Conv1D_Block_lists)

    def _Sequential_repeat(self, num_repeats, num_blocks, **block_kwargs):
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

    def forward(self, x, one_hot):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # x: n x 1 x L => n x N x T
        tsdMask = None
        # print('x ',x.shape)
        w = self.encoder(x)
        # print('w ',w.shape)
        # n x N x L => n x B x L
        e = self.LayerN_S(w)
        # print('e ', e.shape)
        e = self.BottleN_S(e)
        # print('e1 ', e.shape)        
        # conditional part
        x_cls = None
        # print('one_hot ',one_hot.shape)
        emb_one_hot = self.conditioner_one_hot(one_hot)
        # print('emb_one_hot ',emb_one_hot.shape)
        # n x B x L => n x B x L
        m = self.extractor(e, emb_one_hot, tsdMask)
        # print('m ',m.shape)
        # assert 1==2
        # n x B x L => n x num_spk*N x L
        # m = self.gen_masks(e)
        # n x N x L x num_spks
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        # num_spks x n x N x L
        # m = self.activation(torch.stack(m, dim=0))
        gt = None
        d = [w*m[i] for i in range(self.num_spks)]
        # print('d ',d[0].shape)
        #d = w*m
        # decoder part num_spks x n x L
        # audio_encoder = self.istft(x_ex, x_phase) # reconstruct predict audio
        # audio = [audio_encoder[:, 0]]
        s = [self.decoder(d[i], squeeze=True) for i in range(self.num_spks)]
        # print('s ',s[0].shape)
        return s, m, gt, x_cls

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_convtasnet():
    x = torch.randn(2, 160000)
    one_hot = torch.zeros(2)
    nnet = ConvTasNet()
    s = nnet(x,one_hot.long())
    print(str(look_parameters(nnet))+' Mb')
    print(s[1].shape)


if __name__ == "__main__":
    test_convtasnet()
