import os
import torch
from data_loader.AudioReader import AudioReader, write_wav
import argparse
from model.model import TSENet,TSENet_one_hot
from logger.set_logger import setup_logger
import logging
from config.option import parse
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

class Separation():
    def __init__(self, mix_path, s1_path, ref_path, inf_path, yaml_path, model, gpuid):
        super(Separation, self).__init__()
        self.mix = handle_scp(mix_path)
        self.s1 = handle_scp(s1_path)
        self.ref = handle_scp(ref_path)
        self.clss, self.onsets, self.offsets = handle_scp_inf(inf_path)
        self.key = list(self.mix.keys())
        opt = parse(yaml_path)
        net = TSENet(N=opt['TSENet']['N'],
                 B=opt['TSENet']['B'],
                 H=opt['TSENet']['H'],
                 P=opt['TSENet']['P'],
                 X=opt['TSENet']['X'],
                 R=opt['TSENet']['R'],
                 norm=opt['TSENet']['norm'],
                 num_spks=opt['TSENet']['num_spks'],
                 causal=opt['TSENet']['causal'],
                 cls_num=opt['TSENet']['class_num'],
                 nFrameLen=opt['datasets']['audio_setting']['nFrameLen'],
                 nFrameShift=opt['datasets']['audio_setting']['nFrameShift'],
                 nFFT=opt['datasets']['audio_setting']['nFFT'],
                 fusion=opt['TSENet']['fusion'],
                 usingEmb=opt['TSENet']['usingEmb'],
                 usingTsd=opt['TSENet']['usingTsd'],
                 CNN10_settings=opt['TSENet']['CNN10_settings'],
                 fixCNN10=opt['TSENet']['fixCNN10'],
                 fixTSDNet=opt['TSENet']['fixTSDNet'],
                 pretrainedCNN10=opt['TSENet']['pretrainedCNN10'],
                 pretrainedTSDNet=opt['TSENet']['pretrainedTSDNet'],
                 threshold=opt['TSENet']['threshold'])
        dicts = torch.load(model, map_location='cpu')
        net.load_state_dict(dicts["model_state_dict"])
        setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
        self.logger = logging.getLogger(opt['logger']['name'])
        self.logger.info('Load checkpoint from {}, epoch {: d}'.format(model, dicts["epoch"]))
        self.net=net.cuda()
        self.device=torch.device('cuda:{}'.format(
            gpuid[0]) if len(gpuid) > 0 else 'cpu')
        self.gpuid=tuple(gpuid)
        self.sr = opt['datasets']['audio_setting']['sample_rate']
        self.audio_length = opt['datasets']['audio_setting']['audio_length']
        self.cls_num = opt['TSENet']['class_num']
        self.nFrameShift = opt['datasets']['audio_setting']['nFrameShift']

    def inference(self, file_path, max_num=8000):
        with torch.no_grad():
            for i in range(min(len(self.key), max_num)):
                index = self.key[i]
                s1_index = index.replace('.wav', '_lab.wav')
                ref_index = index.replace('.wav', '_re.wav')
                mix = read_wav(self.mix[index])
                ref = read_wav(self.ref[ref_index])
                s1 = read_wav(self.s1[s1_index])
                cls = torch.zeros(self.cls_num)
                cls[self.clss[index]] = 1.
                cls_index = cls.argmax(0)
                cls_index = cls_index.to(self.device)
                onset = self.onsets[index]
                offset = self.offsets[index]
                max_frame = self.sr * self.audio_length // self.nFrameShift - 2
                onset_frame = round(onset * (self.sr // self.nFrameShift - 1)) if round(
                    onset * (self.sr // self.nFrameShift - 1)) >= 0 else 0
                offset_frame = round(offset * (self.sr // self.nFrameShift - 1)) if round(
                    offset * (self.sr // self.nFrameShift - 1)) < max_frame else max_frame
                framelab = torch.zeros(max_frame + 1)
                for i in range(onset_frame, offset_frame + 1):
                    framelab[i] = 1.
                framelab = framelab[None,:]
                self.logger.info("Compute on utterance {}...".format(index))
                mix = mix.to(self.device)
                ref = ref.to(self.device)
                s1 = s1.to(self.device)
                framelab = framelab.to(self.device)
                if mix.dim() == 1:
                    mix = torch.unsqueeze(mix, 0)
                if ref.dim() == 1:
                    ref = torch.unsqueeze(ref, 0)
                if s1.dim() == 1:
                    s1 = torch.unsqueeze(s1, 0)
                #out, lps, lab, est_cls 
                #ests, lps, lab, est_cls = self.net(mix, ref,cls_index.long(), s1)
                ests, lps, lab, est_cls = self.net(mix, ref, s1)
                spks=[torch.squeeze(s.detach().cpu()) for s in ests]
                a = 0
                for s in spks:
                    s = s[:mix.shape[1]]
                    s = s.unsqueeze(0)
                    a += 1
                    os.makedirs(file_path+'/sound'+str(a), exist_ok=True)
                    filename=file_path+'/sound'+str(a)+'/'+index
                    write_wav(filename, s, 16000)
            self.logger.info("Compute over {:d} utterances".format(len(self.mix)))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-mix_scp', type=str, default='/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018/scps/val_mix.scp', help='Path to mix scp file.')
    parser.add_argument(
        '-s1_scp', type=str, default='/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018/scps/val_s1.scp', help='Path to s1 scp file.')
    parser.add_argument(
        '-ref_scp', type=str, default='/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018/scps/val_re.scp', help='Path to ref scp file.')
    parser.add_argument(
        '-inf_scp', type=str, default='/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018/scps/val_inf.scp', help='Path to inf file.')
    parser.add_argument(
        '-yaml', type=str, default='./config/TSENet/train.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='/apdcephfs/share_1316500/donchaoyang/tsss/TSE_exp/checkpoint_fsd2018_audio/TSENet_loss_one_hot_loss_7/best.pt', help="Path to model file.")
    parser.add_argument(
        '-max_num', type=str, default=20, help="Max number for testing samples.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='/apdcephfs/share_1316500/donchaoyang/tsss/TSENet/result/TSENet/baseline', help='save result path')
    args=parser.parse_args()
    gpuid=[int(i) for i in args.gpuid.split(',')]
    separation=Separation(args.mix_scp, args.s1_scp, args.ref_scp, args.inf_scp, args.yaml, args.model, gpuid)
    separation.inference(args.save_path, args.max_num)


if __name__ == "__main__":
    main()

