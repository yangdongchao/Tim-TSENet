import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
import random
from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config
import soundfile as sf
import math
import pandas as pd
from pathlib import Path
event_ls = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", 
            "Burping_or_eructation", "Bus", "Cello", "Chime", 
            "Clarinet", "Computer_keyboard", "Cough", "Cowbell", 
            "Double_bass", "Drawer_open_or_close", "Electric_piano", 
            "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", 
            "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", 
            "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", 
            "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", 
            "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]
event_to_id = {label : i for i, label in enumerate(event_ls)}
print(event_to_id)
def get_file_label_dict(csv_path):
    #strong_csv = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/ft_local/FSD50K.ground_truth/dev.csv'
    print('strong_csv ',csv_path)
    DF_strong = pd.read_csv(csv_path,sep=',',usecols=[0,1])
    file_id = DF_strong['fname']
    labels = DF_strong['label']
    filename_ls = []
    label_ls = []
    for fname in file_id:
        filename_ls.append(fname)
    for label in labels:
        label_ls.append(label)
    dict_ls = {}
    for i in range(len(filename_ls)):
        dict_ls[filename_ls[i]] = str(event_to_id[label_ls[i]])
    return dict_ls


def region_selection(top_result_mat):
    region_index = np.zeros((top_result_mat.shape[-1], 2), dtype=np.int32)
    max_value = np.zeros(top_result_mat.shape[-1])
    for i in range(top_result_mat.shape[-1]):
        max_index = np.argmax(top_result_mat[:, i])
        max_value[i] = np.max(top_result_mat[:, i])
        if max_index < 100:
            max_index = 100
        elif max_index > 900:
            max_index = 900
        l_index = max_index - 100
        r_index = max_index + 100
        region_index[i, 0] = l_index
        region_index[i, 1] = r_index
    return region_index, max_value


def check_files():
    train_pth = '/apdcephfs/private_helinwang/tsss/ft_local/balanced_train_segments'
    eval_pth = '/apdcephfs/private_helinwang/tsss/ft_local/eval_segments'
    train_lst = []
    eval_lst = []
    for root, dirs, files in os.walk(train_pth):
        for name in files:
            train_lst.append(os.path.join(root, name))
    for root, dirs, files in os.walk(eval_pth):
        for name in files:
            eval_lst.append(os.path.join(root, name))

    for file in train_lst + eval_lst:
        try:
            (waveform, sr) = librosa.core.load(file, mono=True)
        except:
            print('{} Read Error'.format(file))
            os.system('rm -rf '+ file)
        else:
            if waveform.shape[0] != int(sr * 10):
                print('{} Wave Length Error: {} samples. Fix it.'.format(file, waveform.shape[0]))
                if waveform.shape[0] > int(sr * 10):
                    waveform = waveform[:int(sr * 10)]
                else:
                    waveform = np.concatenate((waveform, [0.] * (int(sr * 10) - waveform.shape[0])),0)
                sf.write(file, waveform, sr, subtype='PCM_24')
            else:
                print('{} No Error.'.format(file))

    print('Finished Checkout!')

def generate_mixed_data(args):
    sample_rate = args.sample_rate
    duration = args.duration
    sample_num = int(sample_rate*duration)
    num1 = int(sample_rate*2)
    train_dir = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/train'
    test_dir = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/test'
    val_dir = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/val'
    train_txt = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/train.txt'
    test_txt = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/test.txt'
    val_txt = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/val.txt'

    train_data_pth = '/apdcephfs/private_helinwang/tsss/tsss_data/train'
    test_data_pth = '/apdcephfs/private_helinwang/tsss/tsss_data/test'

    train_lst = []
    test_lst = []
    for root, dirs, files in os.walk(train_data_pth):
        for name in files:
            train_lst.append(os.path.join(root, name))
    for root, dirs, files in os.walk(test_data_pth):
        for name in files:
            test_lst.append(os.path.join(root, name))
    train_data_num = len(train_lst)
    test_data_num = len(test_lst)

    # generate train mixed data
    rs1 = random.sample(train_lst, train_data_num)
    rs1 = rs1 * 2
    rs2 = random.sample(train_lst, train_data_num) + random.sample(train_lst, train_data_num)
    flag_num = 1
    for i in range(len(rs1)):
        # resample
        (waveform1, _) = librosa.core.load(rs1[i], sr=sample_rate, mono=True)
        (waveform2, _) = librosa.core.load(rs2[i], sr=sample_rate, mono=True)
        # get random index
        temp = random.random()
        if temp > 0.5:
            index1 = random.randint(0, (sample_num - num1 - 1) // 2)
            index2 = random.randint((sample_num - num1 - 1) // 2, sample_num - num1 - 1)
        else:
            index2 = random.randint(0, (sample_num - num1 - 1) // 2)
            index1 = random.randint((sample_num - num1 - 1) // 2, sample_num - num1 - 1)
        wave = np.zeros(sample_num)
        wave1 = np.zeros(sample_num)
        wave2 = np.zeros(sample_num)
        # check and fix length
        if waveform1.shape[0] > num1:
            waveform1 = waveform1[:num1]
        elif waveform1.shape[0] < num1:
            waveform1 = np.concatenate((waveform1, [0.] * (num1 - waveform1.shape[0])), 0)
        if waveform2.shape[0] > num1:
            waveform2 = waveform2[:num1]
        elif waveform2.shape[0] < num1:
            waveform2 = np.concatenate((waveform2, [0.] * (num1 - waveform2.shape[0])), 0)
        # energy normalization
        wave1[index1:(index1 + num1)] = waveform1
        wave2[index2:(index2 + num1)] = waveform2
        wave2 = wave2*np.sum(wave1**2)/np.sum(wave2**2)
        wave = wave1 + wave2

        # waveform = waveform1 + waveform2
        file_name = 'train_' + str(flag_num) + '.wav'
        file_name_a = 'train_' + str(flag_num) + '_a.wav'
        file_name_b = 'train_' + str(flag_num) + '_b.wav'
        file_name_re = 'train_' + str(flag_num) + '_re.wav'
        file_pth = train_dir + '/' + file_name
        file_pth_a = train_dir + '/' + file_name_a
        file_pth_b = train_dir + '/' + file_name_b
        file_pth_re = train_dir + '/' + file_name_re
        sf.write(file_pth, wave, sample_rate, subtype='PCM_24')
        sf.write(file_pth_a, wave1, sample_rate, subtype='PCM_24')
        sf.write(file_pth_b, wave2, sample_rate, subtype='PCM_24')
        sf.write(file_pth_re, waveform1, sample_rate, subtype='PCM_24')
        print('Save to: {}'.format(file_pth))
        with open(train_txt, "a+") as f:
            f.write(file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[i].split('/')[-1] + '\t' + rs1[i].split('/')[-1] + '\n')
        flag_num += 1

    # generate val mixed data
    rs1 = random.sample(train_lst, int(0.25 * train_data_num))
    rs2 = random.sample(train_lst, int(0.25 * train_data_num))
    flag_num = 1
    for i in range(len(rs1)):
        # resample
        (waveform1, _) = librosa.core.load(rs1[i], sr=sample_rate, mono=True)
        (waveform2, _) = librosa.core.load(rs2[i], sr=sample_rate, mono=True)
        # get random index
        temp = random.random()
        if temp > 0.5:
            index1 = random.randint(0, (sample_num - num1 - 1) // 2)
            index2 = random.randint((sample_num - num1 - 1) // 2, sample_num - num1 - 1)
        else:
            index2 = random.randint(0, (sample_num - num1 - 1) // 2)
            index1 = random.randint((sample_num - num1 - 1) // 2, sample_num - num1 - 1)
        wave = np.zeros(sample_num)
        wave1 = np.zeros(sample_num)
        wave2 = np.zeros(sample_num)
        # check and fix length
        if waveform1.shape[0] > num1:
            waveform1 = waveform1[:num1]
        elif waveform1.shape[0] < num1:
            waveform1 = np.concatenate((waveform1, [0.] * (num1 - waveform1.shape[0])), 0)
        if waveform2.shape[0] > num1:
            waveform2 = waveform2[:num1]
        elif waveform2.shape[0] < num1:
            waveform2 = np.concatenate((waveform2, [0.] * (num1 - waveform2.shape[0])), 0)
        # energy normalization
        wave1[index1:(index1 + num1)] = waveform1
        wave2[index2:(index2 + num1)] = waveform2
        wave2 = wave2 * np.sum(wave1 ** 2) / np.sum(wave2 ** 2)
        wave = wave1 + wave2

        # waveform = waveform1 + waveform2
        file_name = 'val_' + str(flag_num) + '.wav'
        file_name_a = 'val_' + str(flag_num) + '_a.wav'
        file_name_b = 'val_' + str(flag_num) + '_b.wav'
        file_name_re = 'val_' + str(flag_num) + '_re.wav'
        file_pth = val_dir + '/' + file_name
        file_pth_a = val_dir + '/' + file_name_a
        file_pth_b = val_dir + '/' + file_name_b
        file_pth_re = val_dir + '/' + file_name_re
        sf.write(file_pth, wave, sample_rate, subtype='PCM_24')
        sf.write(file_pth_a, wave1, sample_rate, subtype='PCM_24')
        sf.write(file_pth_b, wave2, sample_rate, subtype='PCM_24')
        sf.write(file_pth_re, waveform1, sample_rate, subtype='PCM_24')
        print('Save to: {}'.format(file_pth))
        with open(val_txt, "a+") as f:
            f.write(file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[i].split('/')[-1] + '\t' + rs1[i].split('/')[-1] + '\n')
        flag_num += 1

    # generate test mixed data
    rs1 = random.sample(test_lst, int(0.25 * test_data_num))
    rs2 = random.sample(test_lst, int(0.25 * test_data_num))
    flag_num = 1
    for i in range(len(rs1)):
        # resample
        (waveform1, _) = librosa.core.load(rs1[i], sr=sample_rate, mono=True)
        (waveform2, _) = librosa.core.load(rs2[i], sr=sample_rate, mono=True)
        # get random index
        temp = random.random()
        if temp > 0.5:
            index1 = random.randint(0, (sample_num - num1 - 1) // 2)
            index2 = random.randint((sample_num - num1 - 1) // 2, sample_num - num1 - 1)
        else:
            index2 = random.randint(0, (sample_num - num1 - 1) // 2)
            index1 = random.randint((sample_num - num1 - 1) // 2, sample_num - num1 - 1)
        wave = np.zeros(sample_num)
        wave1 = np.zeros(sample_num)
        wave2 = np.zeros(sample_num)
        # check and fix length
        if waveform1.shape[0] > num1:
            waveform1 = waveform1[:num1]
        elif waveform1.shape[0] < num1:
            waveform1 = np.concatenate((waveform1, [0.] * (num1 - waveform1.shape[0])), 0)
        if waveform2.shape[0] > num1:
            waveform2 = waveform2[:num1]
        elif waveform2.shape[0] < num1:
            waveform2 = np.concatenate((waveform2, [0.] * (num1 - waveform2.shape[0])), 0)
        # energy normalization
        wave1[index1:(index1 + num1)] = waveform1
        wave2[index2:(index2 + num1)] = waveform2
        wave2 = wave2 * np.sum(wave1 ** 2) / np.sum(wave2 ** 2)
        wave = wave1 + wave2

        # waveform = waveform1 + waveform2
        file_name = 'test_' + str(flag_num) + '.wav'
        file_name_a = 'test_' + str(flag_num) + '_a.wav'
        file_name_b = 'test_' + str(flag_num) + '_b.wav'
        file_name_re = 'test_' + str(flag_num) + '_re.wav'
        file_pth = test_dir + '/' + file_name
        file_pth_a = test_dir + '/' + file_name_a
        file_pth_b = test_dir + '/' + file_name_b
        file_pth_re = test_dir + '/' + file_name_re
        sf.write(file_pth, wave, sample_rate, subtype='PCM_24')
        sf.write(file_pth_a, wave1, sample_rate, subtype='PCM_24')
        sf.write(file_pth_b, wave2, sample_rate, subtype='PCM_24')
        sf.write(file_pth_re, waveform1, sample_rate, subtype='PCM_24')
        print('Save to: {}'.format(file_pth))
        with open(test_txt, "a+") as f:
            f.write(file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[i].split('/')[-1] + '\t' + rs1[i].split('/')[-1] + '\n')
        flag_num += 1

    print('Finished Mixed Data!')

def generate_mixed_offset_data(args):
    sample_rate = args.sample_rate
    sample_num = int(sample_rate * 5)
    num1 = int(sample_rate * 3)
    test_dir = '/apdcephfs/share_1316500/helinwang/data/tsss/esc_data'
    test_txt = '/apdcephfs/share_1316500/helinwang/data/tsss/esc_data/test.txt'
    test_data_pth = '/apdcephfs/private_helinwang/tsss/ft_local/ESC-50-master/audio'
    test_lst = []
    for root, dirs, files in os.walk(test_data_pth):
        for name in files:
            test_lst.append(os.path.join(root, name))

    flag_num = 1
    test_data_num = len(test_lst)
    rs1 = random.sample(test_lst, test_data_num) + random.sample(test_lst, test_data_num) + random.sample(test_lst, test_data_num) +random.sample(test_lst, test_data_num)
    for i in range(len(rs1)):
        cls1 = rs1[i].split('-')[-1]
        while True:
            rs2 = random.sample(test_lst, 1)
            cls2 = rs2[0].split('-')[-1]
            if cls2 == cls1:
                break
        while True:
            rs3 = random.sample(test_lst, 1)
            cls3 = rs3[0].split('-')[-1]
            if cls3 != cls1:
                break
        (waveform1, _) = librosa.core.load(rs1[i], sr=sample_rate, mono=True)
        (waveform2, _) = librosa.core.load(rs2[0], sr=sample_rate, mono=True)
        (waveform3, _) = librosa.core.load(rs3[0], sr=sample_rate, mono=True)
        # get random index
        temp = random.random()
        if temp > 0.5:
            index1 = random.randint(0, (sample_num - num1 - 1) // 2)
            index2 = random.randint((sample_num - num1 - 1) // 2, sample_num - num1 - 1)
        else:
            index2 = random.randint(0, (sample_num - num1 - 1) // 2)
            index1 = random.randint((sample_num - num1 - 1) // 2, sample_num - num1 - 1)
        wave = np.zeros(sample_num)
        wave1 = np.zeros(sample_num)
        wave2 = np.zeros(sample_num)
        # energy normalization
        wave1[index1:(index1 + num1)] = waveform1[index1:(index1 + num1)]
        wave2[index2:(index2 + num1)] = waveform3[index2:(index2 + num1)]
        wave2 = wave2 * np.sum(wave1 ** 2) / np.sum(wave2 ** 2)
        wave = wave1 + wave2
        # waveform = waveform1 + waveform3

        file_name = 'test_offset_' + str(flag_num) + '.wav'
        file_name_a = 'test_offset_' + str(flag_num) + '_a.wav'
        file_name_b = 'test_offset_' + str(flag_num) + '_b.wav'
        file_name_re = 'test_offset_' + str(flag_num) + '_re.wav'
        file_pth = test_dir + '/' + file_name
        file_pth_a = test_dir + '/' + file_name_a
        file_pth_b = test_dir + '/' + file_name_b
        file_pth_re = test_dir + '/' + file_name_re
        sf.write(file_pth, wave, sample_rate, subtype='PCM_24')
        sf.write(file_pth_a, wave1, sample_rate, subtype='PCM_24')
        sf.write(file_pth_b, wave2, sample_rate, subtype='PCM_24')
        sf.write(file_pth_re, waveform2, sample_rate, subtype='PCM_24')
        print('Save to: {}'.format(file_pth))
        with open(test_txt, "a+") as f:
            f.write(file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs3[0].split('/')[-1]+ '\t' + rs2[0].split('/')[-1] + '\n')
        flag_num += 1

    print('Finished Mixed Offset Data!')

def data_pre(audio, audio_length, fs, audio_skip):
    stride = round(audio_skip * fs / 2)
    loop = round((audio_length * fs) // stride - 1)
    i = 0
    out = audio
    while i < loop:
        win_data = out[i*stride: (i+2)*stride]
        maxamp = np.max(np.abs(win_data))
        if maxamp < 0.0005:
            loop = loop - 2
            out[i*stride: (loop+1)*stride] = out[(i+2)*stride: (loop+3)*stride]
        else:
            i = i + 1
    length = (audio_length * fs) // stride - loop - 1
    if length == 0:
        return out
    else:
        return out[:(loop + 1) * stride]
# out of domain
def generate_data_train(args):
    sample_rate = args.sample_rate
    sample_num = round(sample_rate * 10)
    csv_path = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/ft_local/FSDKaggle2018.meta/train_post_competition.csv'
    train_dir = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018_all_n/train'
    train_txt = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018_all_n/train.txt'
    data_pth = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/ft_local/FSDKaggle2018.audio_train'
    back_pth = '/apdcephfs/private_donchaoyang/tsss/ft_local/TAU-urban-acoustic-scenes-2019-development/audio'
    data_lst = []
    back_lst = []
    name_dict = get_file_label_dict(csv_path)# get dict {filename: class_name} 
    for root, dirs, files in os.walk(data_pth): # 获得目录下所有的音频文件
        for name in files:
            data_lst.append(os.path.join(root, name))

    for root, dirs, files in os.walk(back_pth):
        for name in files:
            back_lst.append(os.path.join(root, name))

    flag_num = 1
    data_num = len(data_lst)
    print(data_num)
    mix_lst = [1,2,3] 
    rs1 = random.sample(data_lst, data_num)+\
          random.sample(data_lst, data_num)+\
          random.sample(data_lst, data_num)+\
          random.sample(data_lst, data_num)+\
          random.sample(data_lst, data_num) # 每个音频都有5次成为目标声音的机会 
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num) 
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+\
        #   random.sample(data_lst, data_num)+ \
        #   random.sample(data_lst, data_num) + \
        #   random.sample(data_lst, data_num) + \
        #   random.sample(data_lst, data_num) + \
        #   random.sample(data_lst, data_num) + \
        #   random.sample(data_lst, data_num)
    # print(len(rs1))
    # assert 1==2
    for i in range(len(rs1)):
        real_name = Path(rs1[i]).name
        cls1 = name_dict[real_name] # get class label
        while True:
            rs2 = random.sample(data_lst, 1) # 随机选择一个audio
            rs2_real = Path(rs2[0]).name
            cls2 = name_dict[rs2_real]
            if cls2 == cls1 and rs2_real != real_name: # 保证不选到相同的音频
                break
        while True:
            rs3 = random.sample(data_lst, 1)
            rs3_real = Path(rs3[0]).name
            cls3 = name_dict[rs3_real]
            if cls3 != cls1:
                break
        while True:
            rs4 = random.sample(data_lst, 1)
            rs4_real = Path(rs4[0]).name
            cls4 = name_dict[rs4_real]
            if cls4 != cls1:
                break
        while True:
            rs5 = random.sample(data_lst, 1)
            rs5_real = Path(rs5[0]).name
            cls5 = name_dict[rs5_real]
            if cls5 != cls1:
                break
        rs6 = random.sample(back_lst, 1)
        rs7 = random.sample(mix_lst, 1)
        mix_num = rs7[0]
        (waveform1_, _) = librosa.core.load(rs1[i], sr=sample_rate, mono=True) # target sound
        (waveform2, _) = librosa.core.load(rs2[0], sr=sample_rate, mono=True)
        (waveform3_, _) = librosa.core.load(rs3[0], sr=sample_rate, mono=True)
        (waveform4_, _) = librosa.core.load(rs4[0], sr=sample_rate, mono=True)
        (waveform5_, _) = librosa.core.load(rs5[0], sr=sample_rate, mono=True)
        (background, _) = librosa.core.load(rs6[0], sr=sample_rate, mono=True)
        if waveform1_.shape[0] >= sample_num:
            waveform1_ = waveform1_[:sample_num-100]
        if waveform2.shape[0] > sample_num:
            waveform2 = waveform2[:sample_num]
        if waveform3_.shape[0] >= sample_num:
            waveform3_ = waveform3_[:sample_num-100]
        if waveform4_.shape[0] >= sample_num:
            waveform4_ = waveform4_[:sample_num-100]
        if waveform5_.shape[0] >= sample_num:
            waveform5_ = waveform5_[:sample_num-100]
        
        if background.shape[0] > sample_num:
            background = background[:sample_num]
        elif background.shape[0] < sample_num:
            background = np.concatenate((background, [0.] * (sample_num - background.shape[0])), 0)
        waveform1 = data_pre(waveform1_, audio_length=int(waveform1_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        waveform3 = data_pre(waveform3_, audio_length=int(waveform3_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        waveform4 = data_pre(waveform4_, audio_length=int(waveform4_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        waveform5 = data_pre(waveform5_, audio_length=int(waveform5_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        num1 = waveform1.shape[0]
        num3 = waveform3.shape[0]
        num4 = waveform4.shape[0]
        num5 = waveform5.shape[0]
        index1 = random.randint(0, sample_num - num1 - 1) # 在0-10s中，随机选择起始位置，将声音放进去
        index3 = random.randint(0, sample_num - num3 - 1)
        index4 = random.randint(0, sample_num - num4 - 1)
        index5 = random.randint(0, sample_num - num5 - 1)
        onset = 10.0 * index1 / sample_num
        offset = 10.0 * (index1 + num1) / sample_num
        onset3 = 10.0 * index3 / sample_num
        offset3 = 10.0 * (index3 + num3) / sample_num
        onset4 = 10.0 * index4 / sample_num
        offset4 = 10.0 * (index4 + num4) / sample_num
        onset5 = 10.0 * index5 / sample_num
        offset5 = 10.0 * (index5 + num5) / sample_num
        wave1 = np.zeros(sample_num)
        wave2 = np.zeros(sample_num)
        wave3 = np.zeros(sample_num)
        wave4 = np.zeros(sample_num)
        wave5 = np.zeros(sample_num)
        snr = 10. ** (float(random.uniform(-5, 10)) / 20.) # 随机生成一定的信噪比
        waveform3 = waveform3 * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(waveform3) ** 2)) * snr) # add snr
        waveform4 = waveform4 * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(waveform4) ** 2)) * snr)
        waveform5 = waveform5 * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(waveform5) ** 2)) * snr)
        wave1[index1:index1 + num1] = waveform1
        if num1 == 0:
            continue
        wave3[index3:index3 + num3] = waveform3
        wave4[index4:index4 + num4] = waveform4
        wave5[index5:index5 + num5] = waveform5
        wave2[:waveform2.shape[0]] = waveform2
        snr2 = 10. ** (float(random.uniform(5, 20)) / 20.)
        background = background * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(background) ** 2)) * snr2) # 背景声音也加入一个随机的信噪比
        if mix_num == 1: # 一种干扰声音
            wave = background + wave1 + wave3
        elif mix_num == 2: # 2
            wave = background + wave1 + wave3 + wave4
        else: # 3
            wave = background + wave1 + wave3 + wave4 + wave5


        file_name = 'train_' + str(flag_num) + '.wav'
        file_name_lab = 'train_' + str(flag_num) + '_lab.wav'
        file_name_re = 'train_' + str(flag_num) + '_re.wav'
        file_pth = train_dir + '/' + file_name
        file_pth_lab = train_dir + '/' + file_name_lab
        file_pth_re = train_dir + '/' + file_name_re
        sf.write(file_pth, wave, sample_rate, subtype='PCM_16') # save mixture
        sf.write(file_pth_lab, wave1, sample_rate, subtype='PCM_16') # save clean audio
        sf.write(file_pth_re, wave2, sample_rate, subtype='PCM_16') # save reference audio
        print('Save to: {}'.format(file_pth))
        if mix_num == 1: # 保存基本信息
            with open(train_txt, "a+") as f:
                f.write(
                    file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[0].split('/')[-1] + '\t' + rs6[0].split('/')[-1] + '\t' +
                    rs5[0].split('/')[-1] + '\t' + str(onset) + '\t' + str(offset) + '\t' + cls1 + '\t' + str(mix_num) + '\t' +
                    str(onset3) + '\t' + str(offset3) + '\t' + cls3 + '\n')
        elif mix_num == 2:
            with open(train_txt, "a+") as f:
                f.write(
                    file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[0].split('/')[-1] + '\t' + rs6[0].split('/')[-1] + '\t' +
                    rs5[0].split('/')[-1] + '\t' + str(onset) + '\t' + str(offset) + '\t' + cls1 + '\t' + str(mix_num) + '\t' +
                    str(onset3) + '\t' + str(offset3) + '\t' + cls3 + '\t' +
                    str(onset4) + '\t' + str(offset4) + '\t' + cls4 + '\n')
        else:
            with open(train_txt, "a+") as f:
                f.write(
                    file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[0].split('/')[-1] + '\t' + rs6[0].split('/')[-1] + '\t' +
                    rs5[0].split('/')[-1] + '\t' + str(onset) + '\t' + str(offset) + '\t' + cls1 + '\t' + str(mix_num) + '\t' +
                    str(onset3) + '\t' + str(offset3) + '\t' + cls3 + '\t' +
                    str(onset4) + '\t' + str(offset4) + '\t' + cls4 + '\t' +
                    str(onset5) + '\t' + str(offset5) + '\t' + cls5 + '\n')
        flag_num += 1

    print('Finished Training Data Generation!')

def generate_data_val(args):
    sample_rate = args.sample_rate
    sample_num = round(sample_rate * 10)
    csv_path = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/ft_local/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv'
    train_dir = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018_all_n/val'
    train_txt = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018_all_n/val.txt'
    data_pth = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/ft_local/FSDKaggle2018.audio_test'
    back_pth = '/apdcephfs/private_donchaoyang/tsss/ft_local/TAU-urban-acoustic-scenes-2019-development/audio'
    data_lst = []
    back_lst = []
    name_dict = get_file_label_dict(csv_path)
    # print(name_dict)
    # assert 1==2
    for root, dirs, files in os.walk(data_pth):
        for name in files:
            data_lst.append(os.path.join(root, name))

    for root, dirs, files in os.walk(back_pth):
        for name in files:
            back_lst.append(os.path.join(root, name))

    flag_num = 1
    data_num = len(data_lst)
    mix_lst = [1,2,3]
    rs1 = random.sample(data_lst, data_num) 

    for i in range(len(rs1)):
        real_name = Path(rs1[i]).name
        cls1 = name_dict[real_name] # get class label
        while True:
            rs2 = random.sample(data_lst, 1)
            real_name2 = Path(rs2[0]).name
            cls2 = name_dict[real_name2] # get class label
            if cls2 == cls1 and real_name2 != real_name:
                break
        while True:
            rs3 = random.sample(data_lst, 1)
            real_name3 = Path(rs3[0]).name
            cls3 = name_dict[real_name3] # get class label
            if cls3 != cls1:
                break
        while True:
            rs4 = random.sample(data_lst, 1)
            real_name4 = Path(rs4[0]).name
            cls4 = name_dict[real_name4] # get class label
            if cls4 != cls1:
                break
        while True:
            rs5 = random.sample(data_lst, 1)
            real_name5 = Path(rs5[0]).name
            cls5 = name_dict[real_name5] # get class label
            if cls5 != cls1:
                break
        rs6 = random.sample(back_lst, 1)
        rs7 = random.sample(mix_lst, 1)
        mix_num = rs7[0]
        (waveform1_, _) = librosa.core.load(rs1[i], sr=sample_rate, mono=True)
        (waveform2, _) = librosa.core.load(rs2[0], sr=sample_rate, mono=True)
        (waveform3_, _) = librosa.core.load(rs3[0], sr=sample_rate, mono=True)
        (waveform4_, _) = librosa.core.load(rs4[0], sr=sample_rate, mono=True)
        (waveform5_, _) = librosa.core.load(rs5[0], sr=sample_rate, mono=True)
        (background, _) = librosa.core.load(rs6[0], sr=sample_rate, mono=True)

        if waveform1_.shape[0] >= sample_num:
            waveform1_ = waveform1_[:sample_num-100]
        if waveform2.shape[0] > sample_num:
            waveform2 = waveform2[:sample_num]
        if waveform3_.shape[0] >= sample_num:
            waveform3_ = waveform3_[:sample_num-100]
        if waveform4_.shape[0] >= sample_num:
            waveform4_ = waveform4_[:sample_num-100]
        if waveform5_.shape[0] >= sample_num:
            waveform5_ = waveform5_[:sample_num-100]
        
        if background.shape[0] > sample_num:
            background = background[:sample_num]
        elif background.shape[0] < sample_num:
            background = np.concatenate((background, [0.] * (sample_num - background.shape[0])), 0)
        waveform1 = data_pre(waveform1_, audio_length=int(waveform1_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        waveform3 = data_pre(waveform3_, audio_length=int(waveform3_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        waveform4 = data_pre(waveform4_, audio_length=int(waveform4_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        waveform5 = data_pre(waveform5_, audio_length=int(waveform5_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)

        num1 = waveform1.shape[0]
        num3 = waveform3.shape[0]
        num4 = waveform4.shape[0]
        num5 = waveform5.shape[0]
        index1 = random.randint(0, sample_num - num1 - 1)
        index3 = random.randint(0, sample_num - num3 - 1)
        index4 = random.randint(0, sample_num - num4 - 1)
        index5 = random.randint(0, sample_num - num5 - 1)
        onset = 10.0 * index1 / sample_num
        offset = 10.0 * (index1 + num1) / sample_num
        onset3 = 10.0 * index3 / sample_num
        offset3 = 10.0 * (index3 + num3) / sample_num
        onset4 = 10.0 * index4 / sample_num
        offset4 = 10.0 * (index4 + num4) / sample_num
        onset5 = 10.0 * index5 / sample_num
        offset5 = 10.0 * (index5 + num5) / sample_num
        wave1 = np.zeros(sample_num)
        wave2 = np.zeros(sample_num)
        wave3 = np.zeros(sample_num)
        wave4 = np.zeros(sample_num)
        wave5 = np.zeros(sample_num)
        snr = 10. ** (float(random.uniform(-5, 10)) / 20.)
        waveform3 = waveform3 * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(waveform3) ** 2)) * snr)
        waveform4 = waveform4 * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(waveform4) ** 2)) * snr)
        waveform5 = waveform5 * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(waveform5) ** 2)) * snr)
        wave1[index1:index1 + num1] = waveform1
        if num1 == 0:
            continue
        wave3[index3:index3 + num3] = waveform3
        wave4[index4:index4 + num4] = waveform4
        wave5[index5:index5 + num5] = waveform5
        wave2[:waveform2.shape[0]] = waveform2

        snr2 = 10. ** (float(random.uniform(5, 20)) / 20.)
        background = background * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(background) ** 2)) * snr2)
        if mix_num == 1:
            wave = background + wave1 + wave3
        elif mix_num == 2:
            wave = background + wave1 + wave3 + wave4
        else:
            wave = background + wave1 + wave3 + wave4 + wave5

        file_name = 'val_' + str(flag_num) + '.wav'
        file_name_lab = 'val_' + str(flag_num) + '_lab.wav'
        file_name_re = 'val_' + str(flag_num) + '_re.wav'
        file_pth = train_dir + '/' + file_name
        file_pth_lab = train_dir + '/' + file_name_lab
        file_pth_re = train_dir + '/' + file_name_re
        sf.write(file_pth, wave, sample_rate, subtype='PCM_16')
        sf.write(file_pth_lab, wave1, sample_rate, subtype='PCM_16')
        sf.write(file_pth_re, wave2, sample_rate, subtype='PCM_16')
        print('Save to: {}'.format(file_pth))
        if mix_num == 1:
            with open(train_txt, "a+") as f:
                f.write(
                    file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[0].split('/')[-1] + '\t' + rs6[0].split('/')[-1] + '\t' +
                    rs5[0].split('/')[-1] + '\t' + str(onset) + '\t' + str(offset) + '\t' + cls1 + '\t' + str(mix_num) + '\t' +
                    str(onset3) + '\t' + str(offset3) + '\t' + cls3 + '\n')
        elif mix_num == 2:
            with open(train_txt, "a+") as f:
                f.write(
                    file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[0].split('/')[-1] + '\t' + rs6[0].split('/')[-1] + '\t' +
                    rs5[0].split('/')[-1] + '\t' + str(onset) + '\t' + str(offset) + '\t' + cls1 + '\t' + str(mix_num) + '\t' +
                    str(onset3) + '\t' + str(offset3) + '\t' + cls3 + '\t' +
                    str(onset4) + '\t' + str(offset4) + '\t' + cls4 + '\n')
        else:
            with open(train_txt, "a+") as f:
                f.write(
                    file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[0].split('/')[-1] + '\t' + rs6[0].split('/')[-1] + '\t' +
                    rs5[0].split('/')[-1] + '\t' + str(onset) + '\t' + str(offset) + '\t' + cls1 + '\t' + str(mix_num) + '\t' +
                    str(onset3) + '\t' + str(offset3) + '\t' + cls3 + '\t' +
                    str(onset4) + '\t' + str(offset4) + '\t' + cls4 + '\t' +
                    str(onset5) + '\t' + str(offset5) + '\t' + cls5 + '\n')
        flag_num += 1

    print('Finished Val Data Generation!')

def generate_data_test(args):
    sample_rate = args.sample_rate
    sample_num = round(sample_rate * 10)
    csv_path = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/ft_local/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv'
    train_dir = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018_all_n/test'
    train_txt = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018_all_n/test.txt'
    data_pth = '/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/ft_local/FSDKaggle2018.audio_test'
    back_pth = '/apdcephfs/private_donchaoyang/tsss/ft_local/TAU-urban-acoustic-scenes-2019-development/audio'
    data_lst = []
    back_lst = []
    name_dict = get_file_label_dict(csv_path)
    # print(name_dict)
    # assert 1==2
    for root, dirs, files in os.walk(data_pth):
        for name in files:
            data_lst.append(os.path.join(root, name))

    for root, dirs, files in os.walk(back_pth):
        for name in files:
            back_lst.append(os.path.join(root, name))

    flag_num = 1
    data_num = len(data_lst)
    mix_lst = [1,2,3]
    rs1 = random.sample(data_lst, data_num) 

    for i in range(len(rs1)):
        real_name = Path(rs1[i]).name
        cls1 = name_dict[real_name] # get class label
        while True:
            rs2 = random.sample(data_lst, 1)
            real_name2 = Path(rs2[0]).name
            cls2 = name_dict[real_name2] # get class label
            if cls2 == cls1 and real_name2 != real_name:
                break
        while True:
            rs3 = random.sample(data_lst, 1)
            real_name3 = Path(rs3[0]).name
            cls3 = name_dict[real_name3] # get class label
            if cls3 != cls1:
                break
        while True:
            rs4 = random.sample(data_lst, 1)
            real_name4 = Path(rs4[0]).name
            cls4 = name_dict[real_name4] # get class label
            if cls4 != cls1:
                break
        while True:
            rs5 = random.sample(data_lst, 1)
            real_name5 = Path(rs5[0]).name
            cls5 = name_dict[real_name5] # get class label
            if cls5 != cls1:
                break
        rs6 = random.sample(back_lst, 1)
        rs7 = random.sample(mix_lst, 1)
        mix_num = rs7[0]
        (waveform1_, _) = librosa.core.load(rs1[i], sr=sample_rate, mono=True)
        (waveform2, _) = librosa.core.load(rs2[0], sr=sample_rate, mono=True)
        (waveform3_, _) = librosa.core.load(rs3[0], sr=sample_rate, mono=True)
        (waveform4_, _) = librosa.core.load(rs4[0], sr=sample_rate, mono=True)
        (waveform5_, _) = librosa.core.load(rs5[0], sr=sample_rate, mono=True)
        (background, _) = librosa.core.load(rs6[0], sr=sample_rate, mono=True)

        if waveform1_.shape[0] >= sample_num:
            waveform1_ = waveform1_[:sample_num-100]
        if waveform2.shape[0] > sample_num:
            waveform2 = waveform2[:sample_num]
        if waveform3_.shape[0] >= sample_num:
            waveform3_ = waveform3_[:sample_num-100]
        if waveform4_.shape[0] >= sample_num:
            waveform4_ = waveform4_[:sample_num-100]
        if waveform5_.shape[0] >= sample_num:
            waveform5_ = waveform5_[:sample_num-100]
        
        if background.shape[0] > sample_num:
            background = background[:sample_num]
        elif background.shape[0] < sample_num:
            background = np.concatenate((background, [0.] * (sample_num - background.shape[0])), 0)
        waveform1 = data_pre(waveform1_, audio_length=int(waveform1_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        waveform3 = data_pre(waveform3_, audio_length=int(waveform3_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        waveform4 = data_pre(waveform4_, audio_length=int(waveform4_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)
        waveform5 = data_pre(waveform5_, audio_length=int(waveform5_.shape[0]//16000), fs=sample_rate, audio_skip=0.2)

        num1 = waveform1.shape[0]
        num3 = waveform3.shape[0]
        num4 = waveform4.shape[0]
        num5 = waveform5.shape[0]
        index1 = random.randint(0, sample_num - num1 - 1)
        index3 = random.randint(0, sample_num - num3 - 1)
        index4 = random.randint(0, sample_num - num4 - 1)
        index5 = random.randint(0, sample_num - num5 - 1)
        onset = 10.0 * index1 / sample_num
        offset = 10.0 * (index1 + num1) / sample_num
        onset3 = 10.0 * index3 / sample_num
        offset3 = 10.0 * (index3 + num3) / sample_num
        onset4 = 10.0 * index4 / sample_num
        offset4 = 10.0 * (index4 + num4) / sample_num
        onset5 = 10.0 * index5 / sample_num
        offset5 = 10.0 * (index5 + num5) / sample_num
        wave1 = np.zeros(sample_num)
        wave2 = np.zeros(sample_num)
        wave3 = np.zeros(sample_num)
        wave4 = np.zeros(sample_num)
        wave5 = np.zeros(sample_num)
        snr = 10. ** (float(random.uniform(-5, 10)) / 20.)
        waveform3 = waveform3 * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(waveform3) ** 2)) * snr)
        waveform4 = waveform4 * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(waveform4) ** 2)) * snr)
        waveform5 = waveform5 * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(waveform5) ** 2)) * snr)
        wave1[index1:index1 + num1] = waveform1
        if num1 == 0:
            continue
        wave3[index3:index3 + num3] = waveform3
        wave4[index4:index4 + num4] = waveform4
        wave5[index5:index5 + num5] = waveform5
        wave2[:waveform2.shape[0]] = waveform2

        snr2 = 10. ** (float(random.uniform(5, 20)) / 20.)
        background = background * math.sqrt(np.mean(np.abs(waveform1) ** 2)) / (
                math.sqrt(np.mean(np.abs(background) ** 2)) * snr2)
        if mix_num == 1:
            wave = background + wave1 + wave3
        elif mix_num == 2:
            wave = background + wave1 + wave3 + wave4
        else:
            wave = background + wave1 + wave3 + wave4 + wave5

        file_name = 'test_' + str(flag_num) + '.wav'
        file_name_lab = 'test_' + str(flag_num) + '_lab.wav'
        file_name_re = 'test_' + str(flag_num) + '_re.wav'
        file_pth = train_dir + '/' + file_name
        file_pth_lab = train_dir + '/' + file_name_lab
        file_pth_re = train_dir + '/' + file_name_re
        sf.write(file_pth, wave, sample_rate, subtype='PCM_16')
        sf.write(file_pth_lab, wave1, sample_rate, subtype='PCM_16')
        sf.write(file_pth_re, wave2, sample_rate, subtype='PCM_16')
        print('Save to: {}'.format(file_pth))
        if mix_num == 1:
            with open(train_txt, "a+") as f:
                f.write(
                    file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[0].split('/')[-1] + '\t' + rs6[0].split('/')[-1] + '\t' +
                    rs5[0].split('/')[-1] + '\t' + str(onset) + '\t' + str(offset) + '\t' + cls1 + '\t' + str(mix_num) + '\t' +
                    str(onset3) + '\t' + str(offset3) + '\t' + cls3 + '\n')
        elif mix_num == 2:
            with open(train_txt, "a+") as f:
                f.write(
                    file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[0].split('/')[-1] + '\t' + rs6[0].split('/')[-1] + '\t' +
                    rs5[0].split('/')[-1] + '\t' + str(onset) + '\t' + str(offset) + '\t' + cls1 + '\t' + str(mix_num) + '\t' +
                    str(onset3) + '\t' + str(offset3) + '\t' + cls3 + '\t' +
                    str(onset4) + '\t' + str(offset4) + '\t' + cls4 + '\n')
        else:
            with open(train_txt, "a+") as f:
                f.write(
                    file_name + '\t' + rs1[i].split('/')[-1] + '\t' + rs2[0].split('/')[-1] + '\t' + rs6[0].split('/')[-1] + '\t' +
                    rs5[0].split('/')[-1] + '\t' + str(onset) + '\t' + str(offset) + '\t' + cls1 + '\t' + str(mix_num) + '\t' +
                    str(onset3) + '\t' + str(offset3) + '\t' + cls3 + '\t' +
                    str(onset4) + '\t' + str(offset4) + '\t' + cls4 + '\t' +
                    str(onset5) + '\t' + str(offset5) + '\t' + cls5 + '\n')
        flag_num += 1
    print('Finished Test Data Generation!')

def save_data(args):
    train_dir = '/apdcephfs/share_1316500/helinwang/data/tsss/data/train'
    test_dir = '/apdcephfs/share_1316500/helinwang/data/tsss/data/test'
    noise_dir = '/apdcephfs/share_1316500/helinwang/data/tsss/data/noise'
    train_pth = '/apdcephfs/private_helinwang/tsss/ft_local/ESC-50-master/train'
    test_pth = '/apdcephfs/private_helinwang/tsss/ft_local/ESC-50-master/test/audio'
    noise_pth = '/apdcephfs/share_1316500/helinwang/data/ft_local/TAU-urban-acoustic-scenes-2019-development/audio'
    sample_rate = args.sample_rate
    train_dict = {}
    test_dict = {}
    noise_dict = {}
    for root, dirs, files in os.walk(train_pth):
        for name in files:
            path_ = os.path.join(root, name)
            path_2 = os.path.join(train_dir, name)
            train_dict[path_] = path_2

    for root, dirs, files in os.walk(test_pth):
        for name in files:
            path_ = os.path.join(root, name)
            path_2 = os.path.join(test_dir, name)
            test_dict[path_] = path_2

    for root, dirs, files in os.walk(noise_pth):
        for name in files:
            path_ = os.path.join(root, name)
            path_2 = os.path.join(noise_dir, name)
            noise_dict[path_] = path_2

    for i in train_dict.keys():
        (wave, _) = librosa.core.load(i, sr=sample_rate, mono=True)
        wave = data_pre(wave, audio_length=5, fs=sample_rate, audio_skip=0.2)
        sf.write(train_dict[i], wave, sample_rate, subtype='PCM_24')
        print('Save to: {}'.format(train_dict[i]))

    for i in test_dict.keys():
        (wave, _) = librosa.core.load(i, sr=sample_rate, mono=True)
        wave = data_pre(wave, audio_length=5, fs=sample_rate, audio_skip=0.2)
        sf.write(test_dict[i], wave, sample_rate, subtype='PCM_24')
        print('Save to: {}'.format(test_dict[i]))

    for i in noise_dict.keys():
        (wave, _) = librosa.core.load(i, sr=sample_rate, mono=True)
        if wave.shape[0] > round(sample_rate*10):
            wave = wave[:round(sample_rate*10)]
        elif wave.shape[0] < round(sample_rate*10):
            wave = np.concatenate((wave, [0.] * (round(sample_rate*10) - wave.shape[0])), 0)
        sf.write(noise_dict[i], wave, sample_rate, subtype='PCM_24')
        print('Save to: {}'.format(noise_dict[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')
    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=16000)
    parser_at.add_argument('--duration', type=int, default=4)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000)
    # parser_at.add_argument('--model_type', type=str, required=True)
    # parser_at.add_argument('--checkpoint_path', type=str, required=True)
    # parser_at.add_argument('--audio_path', type=str, required=True)
    # parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=16000)
    parser_sed.add_argument('--duration', type=int, default=4)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000)
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--audio_path', type=str, required=True)
    parser_sed.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    # check_files()
    # if args.mode == 'audio_tagging':
    #     audio_tagging(args)
    #
    # elif args.mode == 'sound_event_detection':
    #     sound_event_detection(args)
    #
    # else:
    #     raise Exception('Error argument!')
    # generate_mixed_data(args)
    # generate_mixed_offset_data(args)

    generate_data_train(args)
    generate_data_val(args)
    generate_data_test(args)
    #save_data(args)



