import sys
sys.path.append('../')
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
import numpy as np
import soundfile as sf
import torchaudio
from utils.util import handle_scp, handle_scp_inf
from model.model import STFT
import os
import pickle
import math
nFrameLen = 512
nFrameShift = 256
nFFT = 512
stft = STFT(frame_len=nFrameLen, frame_hop=nFrameShift, num_fft=nFFT)

def time_to_frame(tm,M):
    ans = int(tm/(10.0/M))
    if ans < 0:
        ans = 0
    if ans > M:
        ans = M
    return ans


def read_wav(fname, return_rate=False):
    '''
         Read wavfile using Pytorch audio
         input:
               fname: wav file path
               return_rate: Whether to return the sampling rate
         output:
                src: output tensor of size C x L
                     L is the number of audio frames
                     C is the number of channels.
                sr: sample rate
    '''
    src, sr = torchaudio.load(fname, channels_first=True)
    if return_rate:
        return src.squeeze(), sr
    else:
        return src.squeeze()


class Datasets(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''
    def __init__(self, mix_scp=None, s1_scp=None, ref_scp=None, inf_scp=None, sr=16000, cls_num=50, audio_length=10, hop_size=256):
        super(Datasets, self).__init__()
        self.mix_audio = handle_scp(mix_scp)
        self.s1_audio = handle_scp(s1_scp)
        self.ref_audio = handle_scp(ref_scp)
        self.clss, self.onsets, self.offsets = handle_scp_inf(inf_scp)
        self.sr = sr
        self.cls_num = cls_num
        self.audio_length = audio_length
        self.samples = sr * audio_length
        self.max_frame = (self.samples // hop_size - 1) // 2
        self.key = list(self.mix_audio.keys())


    def __len__(self):
        return len(self.key)

    def __getitem__(self, index):
        index = self.key[index]
        s1_index = index.replace('.wav', '_lab.wav')
        ref_index = index.replace('.wav', '_re.wav')

        mix = read_wav(self.mix_audio[index])
        s1 = read_wav(self.s1_audio[s1_index])
        ref = read_wav(self.ref_audio[ref_index])
        cls = torch.zeros(self.cls_num)
        cls[self.clss[index]] = 1.
        tsd_lab = torch.zeros(self.max_frame)
        sim_lab = torch.zeros(self.max_frame)
        real_time = torch.zeros(2)
        tmp_st = math.floor(self.onsets[index])
        if tmp_st < 0:
            tmp_st = 0
        tmp_ed = math.ceil(self.offsets[index])
        if tmp_ed > 10:
            tmp_ed = 10
        real_time[0] = tmp_st
        real_time[1] = tmp_ed
        M = 156
        start_frame = round(self.max_frame * self.onsets[index] / self.audio_length) if round(self.max_frame * self.onsets[index] / self.audio_length) >= 0 else 0
        end_frame = round(self.max_frame * self.offsets[index] / self.audio_length) if round(self.max_frame * self.offsets[index] / self.audio_length) < self.max_frame else self.max_frame - 1
        L_st = time_to_frame(self.onsets[index], M)
        L_ed = time_to_frame(self.offsets[index], M)
        L_lab = torch.zeros(M)
        L_lab[L_st:L_ed] = 1.0
        tsd_lab[start_frame:end_frame] = 1.
        sim_lab[start_frame:end_frame] = 1.0
        if start_frame>0:
            sim_lab[0:start_frame] = -1.0
        if end_frame < self.max_frame:
            sim_lab[end_frame:] = -1.0
        onset = self.onsets[index]
        offset = self.offsets[index]
        return mix, s1, ref, cls, onset, offset, tsd_lab, sim_lab, L_lab

    def get_mean_std(self):
        preNormFile = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps/norm.info'
        # if os.path.exists(preNormFile):
        #     print('normfile exists, just load it!')
        #     return
        fea_norm_cal = []
        lab_norm_cal = []
        print('Calculate mean and std.')
        num = 0
        for i in self.key:
            mix = read_wav(self.mix_audio[i])
            print(self.mix_audio[i])
            s1 = read_wav(self.mix_audio[i].replace('.wav', '_lab.wav'))
            fea, _= stft(mix[None, :]) #1,f,t
            lab, _= stft(s1[None, :])  #1,f,t
            fea = torch.log(fea ** 2 + 1e-20)
            lab = torch.log(lab ** 2 + 1e-20)
            fea = fea[0].numpy()
            lab = lab[0].numpy()
            if num == 0:
                fea_norm_cal = fea
                lab_norm_cal = lab
            else:
                fea_norm_cal = np.concatenate((fea_norm_cal,fea), -1)
                lab_norm_cal = np.concatenate((lab_norm_cal,lab), -1)

            num += 1
            if num > 5000:
                break

        n_frame = np.shape(fea_norm_cal)[-1]
        self.fea_mean = np.mean(fea_norm_cal, axis=-1)
        self.lab_mean = np.mean(lab_norm_cal, axis=-1)
        print(n_frame)
        print('fea_mean and fea_std size: {}'.format(np.shape(self.fea_mean)[0]))
        print('lab_mean and lab_std size: {}'.format(np.shape(self.lab_mean)[0]))

        for i in range(n_frame):
            if i == 0:
                self.fea_std = np.square(fea_norm_cal[:,i] - self.fea_mean)
                self.lab_std = np.square(lab_norm_cal[:,i] - self.lab_mean)
            else:
                self.fea_std += np.square(fea_norm_cal[:,i] - self.fea_mean)
                self.lab_std += np.square(lab_norm_cal[:,i] - self.lab_mean)
        self.fea_std = np.sqrt(self.fea_std / n_frame)
        self.lab_std = np.sqrt(self.lab_std / n_frame)

        print(f'restore mean and std in {preNormFile}')
        export_dict = {
            'feaMean': self.fea_mean.astype(np.float32),
            'feaStd': self.fea_std.astype(np.float32),
            'labMean': self.lab_mean.astype(np.float32),
            'labStd': self.lab_std.astype(np.float32)
            }
        with open(preNormFile, 'wb') as fid:
            pickle.dump(export_dict, fid)

class Datasets_tse(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''
    def __init__(self, mix_scp=None, s1_scp=None, ref_scp=None, inf_scp=None, tse_scp=None, sr=16000, cls_num=50, audio_length=10, hop_size=256):
        super(Datasets_tse, self).__init__()
        self.mix_audio = handle_scp(mix_scp)
        self.s1_audio = handle_scp(s1_scp)
        self.ref_audio = handle_scp(ref_scp)
        self.tse_audio = handle_scp(tse_scp)
        self.clss, self.onsets, self.offsets = handle_scp_inf(inf_scp)
        self.sr = sr
        self.cls_num = cls_num
        self.audio_length = audio_length
        self.samples = sr * audio_length
        self.max_frame = (self.samples // hop_size - 1) // 2
        self.key = list(self.mix_audio.keys())


    def __len__(self):
        return len(self.key)

    def __getitem__(self, index):
        index = self.key[index]
        s1_index = index.replace('.wav', '_lab.wav')
        ref_index = index.replace('.wav', '_re.wav')
        tse_index = index.replace('.wav','_tse.wav')

        mix = read_wav(self.mix_audio[index])
        s1 = read_wav(self.s1_audio[s1_index])
        ref = read_wav(self.ref_audio[ref_index])
        tse = read_wav(self.tse_audio[tse_index])
        cls = torch.zeros(self.cls_num)
        cls[self.clss[index]] = 1.
        tsd_lab = torch.zeros(self.max_frame)
        sim_lab = torch.zeros(self.max_frame)
        real_time = torch.zeros(2)
        tmp_st = math.floor(self.onsets[index])
        if tmp_st < 0:
            tmp_st = 0
        tmp_ed = math.ceil(self.offsets[index])
        if tmp_ed > 10:
            tmp_ed = 10
        real_time[0] = tmp_st
        real_time[1] = tmp_ed
        M = 156
        start_frame = round(self.max_frame * self.onsets[index] / self.audio_length) if round(self.max_frame * self.onsets[index] / self.audio_length) >= 0 else 0
        end_frame = round(self.max_frame * self.offsets[index] / self.audio_length) if round(self.max_frame * self.offsets[index] / self.audio_length) < self.max_frame else self.max_frame - 1
        L_st = time_to_frame(self.onsets[index], M)
        L_ed = time_to_frame(self.offsets[index], M)
        L_lab = torch.zeros(M)
        L_lab[L_st:L_ed] = 1.0
        tsd_lab[start_frame:end_frame] = 1.
        sim_lab[start_frame:end_frame] = 1.0
        if start_frame>0:
            sim_lab[0:start_frame] = -1.0
        if end_frame < self.max_frame:
            sim_lab[end_frame:] = -1.0
        onset = self.onsets[index]
        offset = self.offsets[index]
        return mix, s1, ref, tse, cls, onset, offset, tsd_lab, sim_lab, L_lab

if __name__ == "__main__":
    datasets = Datasets("/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps/tr_mix.scp",
                        "/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps/tr_s1.scp",
                        "/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps/tr_re.scp",
                        "/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps/tr_inf.scp",
                        "/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps/tr_tse.scp",
                        16000,
                        50,
                        10)

    print(datasets.key)
    # datasets.get_mean_std()

