import os
import torch
from data_loader.AudioReader import AudioReader, write_wav
import argparse
from torch.nn.parallel import data_parallel
from model.model import TSDNet,TSDNet_one_hot, TSDNet_tse
from logger.set_logger import setup_logger
import logging
from config.option import parse
import torchaudio
from utils.util import handle_scp, handle_scp_inf
from torch.utils.data import DataLoader as Loader
from data_loader.Dataset_light import Datasets
from model import model
from logger import set_logger
from config import option
import argparse
import torch
import time
import soundfile as sf
import metrics # import metrics.py file
import tsd_utils as utils # import utils.py
import pandas as pd
import numpy as np
from tabulate import tabulate
import datetime
import uuid
from pathlib import Path

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
    def __init__(self, mix_scp, ref_scp, inf_scp, tse_scp, yaml_path, model, gpuid, pred_file='./tsd_result.tsv'):
        super(Separation, self).__init__()
        self.mix_audio = handle_scp(mix_scp)
        self.ref_audio = handle_scp(ref_scp)
        self.tse_audio = handle_scp(tse_scp)
        self.clss,_,_ = handle_scp_inf(inf_scp)
        self.key = list(self.mix_audio.keys())
        opt = parse(yaml_path)
        tsdnet = TSDNet_tse(nFrameLen=opt['datasets']['audio_setting']['nFrameLen'],
                            nFrameShift=opt['datasets']['audio_setting']['nFrameShift'],
                            cls_num = opt['datasets']['audio_setting']['class_num'],
                            CNN10_settings=opt['Conv_Tasnet']['CNN10_settings'],
                            pretrainedCNN10='/apdcephfs/private_donchaoyang/tsss/Dual-Path-RNN-Pytorch2/model/Cnn10_mAP=0.380.pth',
                            use_frame = opt['use_frame'],
                            only_ref = opt['only_ref']
                            )
        dicts = torch.load(model, map_location='cpu')
        tsdnet.load_state_dict(dicts["model_state_dict"])
        setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
        self.logger = logging.getLogger(opt['logger']['name'])
        self.logger.info('Load checkpoint from {}, epoch {: d}'.format(model, dicts["epoch"]))
        self.tsdnet=tsdnet.cuda()
        self.device=torch.device('cuda:{}'.format(
            gpuid[0]) if len(gpuid) > 0 else 'cpu')
        self.pred_file = pred_file
        self.label_path = opt['label_path']
        self.save_tsv_path = os.path.join(opt['save_tsv_path'], opt['name'],
            "{}_{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'), uuid.uuid1().hex))
        Path(self.save_tsv_path).mkdir(exist_ok=True, parents=True) # make dir
            

    def test(self):
        self.tsdnet.eval()
        time_predictions = []
        class_result_file = 'class_result_{}.txt'
        event_file = 'event_{}.txt'
        segment_file = 'segment_{}.txt'
        with torch.no_grad():
            for i in range(len(self.key)):
                index = self.key[i]
                ref_index = index.replace('.wav', '_re.wav')
                tse_index = index.replace('.wav','_tse.wav')
                cls = str(self.clss[index])
                cls = 'class_' + cls
                mix = read_wav(self.mix_audio[index])
                ref = read_wav(self.ref_audio[ref_index])
                tse_audio = read_wav(self.tse_audio[tse_index])
                mix = mix.to(self.device)
                ref = ref.to(self.device)
                tse_audio = tse_audio.to(self.device)
                mix = mix[None,:]
                ref = ref[None,:]
                tse_audio = tse_audio[None,:]
                x_cls, out_tsd_time, out_tsd_up = self.tsdnet(mix, ref, tse_audio)
                # x_cls: <bs,50>
                # out_tsd_time: <bs,t/2>
                # out_tsd_up: <bs,t>
                pred = out_tsd_up.detach().cpu().numpy() # transpose to numpy
                # pred = pred[:,:,0]
                # print(pred.shape)
                thres = 0.5
                window_size = 1
                filtered_pred = utils.median_filter(pred, window_size=window_size, threshold=thres)
                decoded_pred = [] #
                decoded_pred_ = utils.decode_with_timestamps(str(cls),filtered_pred[0,:])
                if len(decoded_pred_) == 0: # neg deal
                    decoded_pred_.append((str(cls),0,0))
                decoded_pred.append(decoded_pred_)
                for num_batch in range(len(decoded_pred)): # when we test our model,the batch_size is 1
                    #print('len(decoded_pred) ',len(decoded_pred))
                    filename = index.split('/')[-1]
                    # Save each frame output, for later visualization
                    label_prediction = decoded_pred[num_batch] # frame predict
                    for event_label, onset, offset in label_prediction:
                        time_predictions.append({
                            'filename': filename,
                            'onset': onset,
                            'offset': offset,
                            'event_label': str(event_label)}) # get real predict results,including event_label,onset,offset
        assert len(time_predictions) > 0, "No outputs, lower threshold?"
        pred_df = pd.DataFrame(time_predictions, columns=['filename', 'onset', 'offset','event_label']) # it store the happen event and its time information
        time_ratio = 10.0/pred.shape[1] # calculate time
        pred_df = utils.predictions_to_time(pred_df, ratio=time_ratio) # transform the number of frame to real time
        label_path = self.label_path
        test_data_filename = os.path.splitext(os.path.basename(label_path))[0]
        print('test_data_filename ',test_data_filename)
        pred_file = 'hard_predictions_{}.txt'
        if pred_file: # it name is hard_predictions...
            pred_df.to_csv(os.path.join(self.save_tsv_path, pred_file.format(test_data_filename)),
                                        index=False, sep="\t")
        strong_labels_df = pd.read_csv(self.label_path, sep='\t') # get
        if not np.issubdtype(strong_labels_df['filename'].dtype, np.number):
            strong_labels_df['filename'] = strong_labels_df['filename'].apply(os.path.basename)
        sed_eval = True
        if sed_eval:
            event_result, segment_result = metrics.compute_metrics(
                strong_labels_df, pred_df, time_resolution=0.2)  # calculate f1
            print("Event Based Results:\n{}".format(event_result))
            event_results_dict = event_result.results_class_wise_metrics()
            class_wise_results_df = pd.DataFrame().from_dict({
                f: event_results_dict[f]['f_measure']
                for f in event_results_dict.keys()}).T
            class_wise_results_df.to_csv(os.path.join(self.save_tsv_path, class_result_file.format(test_data_filename)), sep='\t')
            print("Class wise F1-Macro:\n{}".format(
                tabulate(class_wise_results_df, headers='keys', tablefmt='github')))
            if event_file:
                with open(os.path.join(self.save_tsv_path, event_file.format(test_data_filename)), 'w') as wp:
                    wp.write(event_result.__str__())
            print("=" * 100)
            print(segment_result)
            if segment_file:
                with open(os.path.join(self.save_tsv_path,
                                       segment_file.format(test_data_filename)), 'w') as wp:
                    wp.write(segment_result.__str__())
            event_based_results = pd.DataFrame(
                event_result.results_class_wise_average_metrics()['f_measure'], index=['event_based'])
            segment_based_results = pd.DataFrame(
                segment_result.results_class_wise_average_metrics()
                ['f_measure'], index=['segment_based'])
            result_quick_report = pd.concat((event_based_results, segment_based_results))
            # Add two columns
            with open(os.path.join(self.save_tsv_path, 'quick_report_{}.md'.format(test_data_filename)), 'w') as wp:
                print(tabulate(result_quick_report, headers='keys', tablefmt='github'), file=wp)
            print("Quick Report: \n{}".format(tabulate(result_quick_report, headers='keys', tablefmt='github')))




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-yaml', type=str, default='./config/Conv_Tasnet/train_tse.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='/apdcephfs/share_1316500/donchaoyang/tsss/TSD_exp/checkpoint_fsd2018_audio/TSDNet_audio_2gru_tse_ML2_fix_random_kaiming_norm_w_clip_w_frame/best.pt', help="Path to model file.")
    parser.add_argument(
        '-max_num', type=str, default=10000, help="Max number for testing samples.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='./result/Conv_Tasnet/', help='save result path')
    args=parser.parse_args()
    gpuid=[int(i) for i in args.gpuid.split(',')]
    separation=Separation('/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018/scps/val_mix.scp',
                          '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018/scps/val_re.scp',
                          '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018/scps/val_inf.scp',
                          '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018/scps/val_tse_7_1.scp',
                          args.yaml, args.model, gpuid)
    separation.test()


if __name__ == "__main__":
    main()
