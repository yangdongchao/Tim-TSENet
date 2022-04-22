import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def frame_shift(features):
    batch_size, _, _, _ = features.shape
    shifted_feature = []
    for idx in range(batch_size):
        shift = int(random.gauss(0, 10))
        shifted_feature.append(torch.roll(features[idx], shift, dims=2))
    return torch.stack(shifted_feature)

class TimeShift(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            shift = torch.empty(1).normal_(self.mean, self.std).int().item()
            x = torch.roll(x, shift, dims=2)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)

class Cnn10(nn.Module):
    def __init__(self,scale=2):  
        super(Cnn10, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.scale = scale
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Input: (batch_size, data_length)"""
        if self.scale == 8:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (2,4)
            pool_size4 = (1,4)
        elif self.scale == 4:
            pool_size1 = (2,2)
            pool_size2 = (2,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        elif self.scale == 2:
            pool_size1 = (2,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        else:
            pool_size1 = (1,2)
            pool_size2 = (1,2)
            pool_size3 = (1,4)
            pool_size4 = (1,4)
        x = self.conv_block1(input, pool_size=pool_size1, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=pool_size2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=pool_size3, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=pool_size4, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x

class conv1d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding='VALID', dilation=1):
        super(conv1d, self).__init__()
        if padding == 'VALID':
            dconv_pad = 0
        elif padding == 'SAME':
            dconv_pad = dilation * ((kernel_size - 1) // 2)
        else:
            raise ValueError("Padding Mode Error!")
        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=dconv_pad)
        self.act = nn.ReLU()
        self.init_layer(self.conv)

    def init_layer(self, layer): # relu
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        out = self.act(self.conv(x))
        return out

class Fusion(nn.Module):
    def __init__(self, inputdim_1, inputdim_2, n_fac):
        super().__init__()
        self.fuse_layer1 = conv1d(inputdim_1, inputdim_1*n_fac,1) # 128*4
        self.fuse_layer2 = conv1d(inputdim_2, inputdim_1*n_fac,1) # 128*4
        # self.fuse_layer1.apply(init_weights) # 2022/2/12 new add to solve the problem of initiaze
        # self.fuse_layer2.apply(init_weights) # 2022/2/12 new add to solve the problem of initiaze
        self.avg_pool = nn.AvgPool1d(n_fac, stride=n_fac) # 沿着最后一个维度进行pooling
    def forward(self,embedding, mix_embed):
        embedding = embedding.permute(0,2,1)
        fuse1_out = self.fuse_layer1(embedding) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse1_out = fuse1_out.permute(0,2,1)

        mix_embed = mix_embed.permute(0,2,1)
        fuse2_out = self.fuse_layer2(mix_embed) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse2_out = fuse2_out.permute(0,2,1)
        as_embs = torch.mul(fuse1_out, fuse2_out) # 相乘 [2, 501, 2560]
        # (10, 501, 512)
        as_embs = self.avg_pool(as_embs) # [2, 501, 512] 相当于 2560//5
        return as_embs

class TSD(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(TSD, self).__init__()
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        self.gru = nn.GRU(128, 128, bidirectional=True, batch_first=True) # 先用一个gru试试
        self.fc = nn.Linear(256, 256)
        self.fusion = Fusion(128,4)
        self.outputlayer = nn.Linear(256, 2)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
        self.bn0.apply(init_bn)

    def forward(self, input, emb):
        """
        Input: (batch_size, data_length)"""
        # print('input ',input.shape)
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) # torch.Size([32, 1, 624, 128])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print('x ',x.shape)
        batch, ch, time, dim = x.shape # (b,1,t,d)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2) # (b,156,128)
        # print('x ',x.shape)
        # assert 1==2
        emb = emb.unsqueeze(1)
        emb = emb.repeat(1, x.shape[1], 1)
        sim_cos = self.cos(x,emb)
        # print('x ',x.shape)
        # print('emb ',emb.shape)
        # print('sim_cos ',sim_cos)
        # assert 1==2
        x = self.fusion(emb,x) # 512
        #x = torch.cat((x, emb), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) # torch.Size([16, 161, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 156, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 156]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # (16, 624, 2)
        return decision_time[:,:,0], decision_up[:,:,0], sim_cos

class TSD2(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(TSD2, self).__init__()
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        self.gru = nn.GRU(128, 128, 2, bidirectional=True, batch_first=True) # 先用一个gru试试
        self.fc = nn.Linear(256, 256)
        self.fusion = Fusion(128,128,4)
        self.outputlayer = nn.Linear(256, 2)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.features.apply(init_weights)
        self.fc.apply(init_weights) # 2022/2/12 new add to solve the problem of initiaze
        self.init_rnn_layer(self.gru) # 2022/2/12 new add to solve the problem of initiaze
        self.outputlayer.apply(init_weights)
        self.bn0.apply(init_bn)

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

    def forward(self, input, emb):
        """
        Input: (batch_size, data_length)"""
        # print('input ',input.shape)
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) # torch.Size([32, 1, 624, 128])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print('x ',x.shape)
        batch, ch, time, dim = x.shape # (b,1,t,d)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2) # (b,156,128)
        # print('x ',x.shape)
        # assert 1==2
        emb = emb.unsqueeze(1)
        emb = emb.repeat(1, x.shape[1], 1)
        sim_cos = self.cos(x,emb)
        # print('x ',x.shape)
        # print('emb ',emb.shape)
        # print('sim_cos ',sim_cos)
        # assert 1==2
        x = self.fusion(emb,x) # 512
        #x = torch.cat((x, emb), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) # torch.Size([16, 161, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 156, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 156]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # (16, 624, 2)
        return decision_time[:,:,0], decision_up[:,:,0], sim_cos


class TSD2_tse(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, use_frame):
        super(TSD2_tse, self).__init__()
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        self.use_frame = use_frame
        if self.use_frame:
            self.gru = nn.GRU(128, 128, 2, bidirectional=True, batch_first=True) 
            self.fusion = Fusion(128,256,4) # embed,mix
        else:
            self.gru = nn.GRU(128, 128, 2, bidirectional=True, batch_first=True) 
            self.fusion = Fusion(128,128,4)
        self.fc = nn.Linear(256, 256)
        self.outputlayer = nn.Linear(256, 2)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.fc.apply(init_weights) # 2022/2/12 new add to solve the problem of initiaze
        self.init_rnn_layer(self.gru) # 2022/2/12 new add to solve the problem of initiaze
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
        self.bn0.apply(init_bn)
    
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
    def forward(self, input, emb, frame_emb):
        """
        Input: (batch_size, data_length)"""
        # print('input ',input.shape)
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) # torch.Size([32, 1, 624, 128])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print('x ',x.shape)
        batch, ch, time, dim = x.shape # (b,1,t,d)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2) # (b,312,128)
        frame_emb_up = torch.nn.functional.interpolate(
                frame_emb.transpose(1, 2), # [b, 250, 128]
                x.shape[1], # 501
                mode='linear',
                align_corners=False).transpose(1, 2)
        # print('frame_emb_up ',frame_emb_up.shape)
        # assert 1==2

        # if we decide use frame level
        if self.use_frame:
            # sim_cos = self.cos(x,emb)
            x = torch.cat((x, frame_emb_up), dim=2)
        # else:
        #     sim_cos = self.cos(x,emb)
        # assert 1==2
        emb = emb.unsqueeze(1)
        emb = emb.repeat(1, x.shape[1], 1)
        # print('x ',x.shape)
        # print('emb ',emb.shape)
        # print('sim_cos ',sim_cos)
        # assert 1==2
        x = self.fusion(emb,x) # 512
        #x = torch.cat((x, emb), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) # torch.Size([16, 161, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 156, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 156]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # (16, 624, 2)
        return decision_time[:,:,0], decision_up[:,:,0]


class TSD_L(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(TSD_L, self).__init__()
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            nn.Dropout(0.3),
        )
        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, 256)
        self.outputlayer = nn.Linear(256, 2)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
        self.bn0.apply(init_bn)

    def forward(self, input, emb):
        """
        Input: (batch_size, data_length)"""
        # print('input ',input.shape)
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) # torch.Size([32, 1, 624, 128])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print('x ',x.shape)
        batch, ch, time, dim = x.shape # (b,1,t,d)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2) # (b,156,128)
        # print('x ',x.shape)
        # assert 1==2
        emb = emb.unsqueeze(1)
        emb = emb.repeat(1, x.shape[1], 1)
        sim_cos = self.cos(x,emb)
        # print('x ',x.shape)
        # print('emb ',emb.shape)
        # print('sim_cos ',sim_cos)
        # assert 1==2
        x = torch.cat((x, emb), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) # torch.Size([16, 161, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 156, 2])
        # decision_up = torch.nn.functional.interpolate(
        #         decision_time.transpose(1, 2), # [16, 2, 156]
        #         time, # 501
        #         mode='linear',
        #         align_corners=False).transpose(1, 2) # (16, 624, 2)
        return decision_time[:,:,0], decision_time[:,:,0],sim_cos

class TSD_plus(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(TSD_plus, self).__init__()
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.features = Cnn10()
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024, 256)
        self.outputlayer = nn.Linear(256, 2)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
        self.bn0.apply(init_bn)

    def forward(self, input, emb):
        """
        Input: (batch_size, data_length)"""
        # print('input ',input.shape)
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) # torch.Size([32, 1, 624, 128])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print('x ',x.shape)
        batch, ch, time, dim = x.shape # (b,1,t,d)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2) # (b,156,128)
        # print('x ',x.shape) # 512
        # assert 1==2
        emb = emb.unsqueeze(1)
        emb = emb.repeat(1, x.shape[1], 1)
        x = torch.cat((x, emb), dim=2) # [B, T, 128 + emb_dim]

        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) # torch.Size([16, 161, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 156, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 156]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # (16, 624, 2)

        return decision_up[:,:,0], decision_time[:,:,0],x

class TSD_plus_sim(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(TSD_plus_sim, self).__init__()
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=60, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.time_shift = TimeShift(0, 50)
        self.features = Cnn10()
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        self.sim_fc = nn.Linear(512,128)
        self.fc = nn.Linear(1024, 256)
        self.outputlayer = nn.Linear(256, 2)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
        self.bn0.apply(init_bn)

    def forward(self, input, emb):
        """
        Input: (batch_size, data_length)"""
        # print('input ',input.shape)
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) # torch.Size([32, 1, 624, 128])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            # print('x, ', x.shape)
            x = self.time_shift(x)
            x = self.spec_augmenter(x)
        # print('x ',x.shape)
        batch, ch, time, dim = x.shape # (b,1,t,d)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2) # (b,156,128)
        # print('x ',x.shape) # 512
        # assert 1==2
        emb = emb.unsqueeze(1)
        emb = emb.repeat(1, x.shape[1], 1)
        x_sim = self.sim_fc(x)
        sim_cos = self.cos(x_sim, emb)
        x = torch.cat((x, emb), dim=2) # [B, T, 128 + emb_dim]

        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) # torch.Size([16, 161, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 156, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 156]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # (16, 624, 2)

        return decision_up[:,:,0], decision_time[:,:,0],sim_cos

class TSD_IS(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super().__init__()
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.features = nn.Sequential(
            Block2D(1, 64),
            nn.MaxPool2d((2, 4)),
            Block2D(64, 64),
            nn.MaxPool2d((1, 4)),
            Block2D(64, 64),
            nn.MaxPool2d((1, 4)))
        # with torch.no_grad():
        #     rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
        #     rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(64, 62, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(124+128, 124, bidirectional=True, batch_first=True)
        # self.fc = nn.Linear(248,2)
        self.outputlayer = nn.Linear(248, 2)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.features.apply(init_weights)
        self.bn0.apply(init_bn)
        self.outputlayer.apply(init_weights)

    def forward(self,input,embedding):
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) # torch.Size([32, 1, 624, 128])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        batch, ch, time, dim = x.shape # (b,1,t,d)
        # x = x.unsqueeze(1) # (b,1,t,d) 
        # print('x ',x.shape)
        x = self.features(x) # 
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
            self.gru2.flatten_parameters()
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,250,64)
        # print('x ',x.shape)
        # assert 1==2
        x, _ = self.gru(x)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        sim_cos = torch.zeros(1).cuda()
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        x, _ = self.gru2(x) #  x  torch.Size([16, 125, 256])
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_up[:,:,0], decision_time[:,:,0],sim_cos

class TSD_regresion_two_cls(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(TSD_regresion_two_cls, self).__init__()
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64,16)
        self.st_fc = nn.Linear(16*312, 10) # 
        self.ed_fc = nn.Linear(16*312,11)
        self.PReLu1 = nn.PReLU()
        # self.outputlayer = nn.Linear(16*312, 2)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.features.apply(init_weights)
        # self.outputlayer.apply(init_weights)
        self.bn0.apply(init_bn)

    def forward(self, input, emb):
        """
        Input: (batch_size, data_length)"""
        # print('input ',input.shape)
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) # torch.Size([32, 1, 624, 128])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print('x ',x.shape)
        batch, ch, time, dim = x.shape # (b,1,t,d)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2) # (b,156,128)
        # print('x ',x.shape)
        emb = emb.unsqueeze(1)
        emb = emb.repeat(1, x.shape[1], 1)
        sim_cos = self.cos(x,emb)
        # print('x ',x.shape)
        # print('emb ',emb.shape)
        # print('sim_cos ',sim_cos)
        # assert 1==2
        x = torch.cat((x, emb), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) # torch.Size([16, 161, 256])
        # print('x ',x.shape)
        # assert 1==2
        x = self.fc(x)
        x = self.PReLu1(x)
        x = self.fc2(x)
        x = x.flatten(-2)
        x_st = self.st_fc(x)
        x_ed = self.ed_fc(x)
        decision_time = x_st
        decision_up = x_ed
        return decision_up, decision_time, sim_cos

class TSD_regresion(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(TSD_regresion, self).__init__()
        window = 'hann'
        center = False
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64,16)
        self.PReLu1 = nn.PReLU()
        self.outputlayer = nn.Linear(16*312, 2)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
        self.bn0.apply(init_bn)

    def forward(self, input, emb):
        """
        Input: (batch_size, data_length)"""
        # print('input ',input.shape)
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins) # torch.Size([32, 1, 624, 128])
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print('x ',x.shape)
        batch, ch, time, dim = x.shape # (b,1,t,d)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2) # (b,156,128)
        # print('x ',x.shape)
        emb = emb.unsqueeze(1)
        emb = emb.repeat(1, x.shape[1], 1)
        sim_cos = self.cos(x,emb)
        # print('x ',x.shape)
        # print('emb ',emb.shape)
        # print('sim_cos ',sim_cos)
        # assert 1==2
        x = torch.cat((x, emb), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) # torch.Size([16, 161, 256])
        # print('x ',x.shape)
        # assert 1==2
        x = self.fc(x)
        x = self.PReLu1(x)
        x = self.fc2(x)
        x = x.flatten(-2)
        x = self.outputlayer(x)
        decision_time = x
        decision_up = x
        return decision_up, decision_time, sim_cos