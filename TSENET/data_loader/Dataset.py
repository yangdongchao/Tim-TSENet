import sys
sys.path.append('../')
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
from utils.util import handle_scp, handle_scp_inf

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
       inf_scp: include the onset and offset information?
    '''
    def __init__(self, mix_scp=None, s1_scp=None, ref_scp=None, inf_scp=None, sr=16000, cls_num=50, audio_length=10, nFrameShift=256):
        super(Datasets, self).__init__()
        self.mix_audio = handle_scp(mix_scp)
        self.s1_audio = handle_scp(s1_scp)
        self.ref_audio = handle_scp(ref_scp)
        self.clss, self.onsets, self.offsets = handle_scp_inf(inf_scp)
        self.sr = sr
        self.cls_num = cls_num # class num
        self.audio_length = audio_length # s
        self.nFrameShift = nFrameShift # 
        self.key = list(self.mix_audio.keys()) # mixture audio name

    def __len__(self):
        return len(self.key)

    def __getitem__(self, index):
        index = self.key[index] # get name?
        s1_index = index.replace('.wav', '_lab.wav') # get clean wav name
        ref_index = index.replace('.wav', '_re.wav') # get reference wav  name
        mix = read_wav(self.mix_audio[index])
        s1 = read_wav(self.s1_audio[s1_index])
        ref = read_wav(self.ref_audio[ref_index])
        cls = torch.zeros(self.cls_num) # ready for one-hot 
        cls[self.clss[index]] = 1. # 
        onset = self.onsets[index] # get oneset time
        offset = self.offsets[index] # get offset time
        max_frame = self.sr * self.audio_length // self.nFrameShift - 2 # 
        onset_frame = round(onset * (self.sr // self.nFrameShift - 1)) if round(onset * (self.sr // self.nFrameShift - 1)) >= 0 else 0
        # time transfer to the number of frame
        offset_frame = round(offset * (self.sr // self.nFrameShift - 1)) if round(
            offset * (self.sr // self.nFrameShift - 1)) < max_frame else max_frame
        framelab = torch.zeros(max_frame + 1) # frame-level label
        for i in range(onset_frame, offset_frame + 1):
            framelab[i] = 1.
        return mix, s1, ref, cls, onset, offset, framelab


if __name__ == "__main__":
    datasets = Datasets("/apdcephfs/share_1316500/donchaoyang/tsss/Dual-Path-RNN-Pytorch/scps/tr_mix.scp",
                        "/apdcephfs/share_1316500/donchaoyang/tsss/Dual-Path-RNN-Pytorch/scps/tr_s1.scp",
                        "/apdcephfs/share_1316500/donchaoyang/tsss/Dual-Path-RNN-Pytorch/scps/tr_re.scp",
                        "/apdcephfs/share_1316500/donchaoyang/tsss/Dual-Path-RNN-Pytorch/scps/tr_inf.scp",
                        16000,
                        50,
                        8,
                        256)
    print(datasets.key)



