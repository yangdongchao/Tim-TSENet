import sys
sys.path.append('../')

from data_loader.AudioData import AudioReader
import torch
from torch.utils.data import Dataset

import numpy as np


class Datasets(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
       chunk_size (int, optional): split audio size (default: 32000(4 s))
       least_size (int, optional): Minimum split size (default: 16000(2 s))
    '''

    def __init__(self, mix_scp=None, s1_scp=None, ref_scp=None, sample_rate=32000, chunk_size=64000, least_size=64000):
        super(Datasets, self).__init__()
        self.mix_audio = AudioReader(
            mix_scp, sample_rate=sample_rate, chunk_size=chunk_size, least_size=least_size).audio
        self.ref_audio = AudioReader(
            ref_scp, sample_rate=sample_rate, chunk_size=chunk_size, least_size=least_size).audio
        self.s1_audio = AudioReader(
            s1_scp, sample_rate=sample_rate, chunk_size=chunk_size, least_size=least_size).audio

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        return self.mix_audio[index], self.s1_audio[index], self.ref_audio[index]


if __name__ == "__main__":
    dataset = Datasets("/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tto_mix.scp",
                      "/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tto_s1.scp", "/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tto_re.scp")
    for i in dataset.mix_audio:
        print(i.shape)
        if i.shape[0] != 64000:
            print('fail')

