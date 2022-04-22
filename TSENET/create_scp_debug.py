import os

train_mix_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tr_mix.scp'
train_s1_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tr_s1.scp'
train_s2_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tr_s2.scp'
train_re_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tr_re.scp'

test_mix_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tt_mix.scp'
test_s1_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tt_s1.scp'
test_s2_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tt_s2.scp'
test_re_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tt_re.scp'

val_mix_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/val_mix.scp'
val_s1_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/val_s1.scp'
val_s2_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/val_s2.scp'
val_re_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/val_re.scp'

test_offset_mix_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tto_mix.scp'
test_offset_s1_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tto_s1.scp'
test_offset_s2_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tto_s2.scp'
test_offset_re_scp = '/apdcephfs/private_helinwang/tsss/Dual-Path-RNN-Pytorch/scps_debug/tto_re.scp'


train_mix = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/train'
test_mix = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/test'
vl_mix = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/val'
test_offset_mix = '/apdcephfs/share_1316500/helinwang/data/tsss/tsss_mixed/test_offset'


tr_mix = open(train_mix_scp,'w')
tr_s1 = open(train_s1_scp,'w')
tr_s2 = open(train_s2_scp,'w')
tr_re = open(train_re_scp,'w')
num1 = 0
num2 = 0
num3 = 0
num4 = 0
for root, dirs, files in os.walk(train_mix):
    files.sort()
    for file in files:
        if 'a.wav' in file and num1 < 30:
            tr_s1.write(file+" "+root+'/'+file)
            tr_s1.write('\n')
            num1 += 1
            tr_re.write(file + " " + root + '/' + file)
            tr_re.write('\n')
            num2 += 1
        elif 'b.wav' in file and num3 < 30:
            tr_s2.write(file + " " + root + '/' + file)
            tr_s2.write('\n')
            num3 += 1
        elif num4 < 30:
            tr_mix.write(file + " " + root + '/' + file)
            tr_mix.write('\n')
            num4 += 1
tr_mix.close()
tr_s1.close()
tr_s2.close()
tr_re.close()

tt_mix = open(test_mix_scp,'w')
tt_s1 = open(test_s1_scp,'w')
tt_s2 = open(test_s2_scp,'w')
tt_re = open(test_re_scp,'w')
num1 = 0
num2 = 0
num3 = 0
num4 = 0
for root, dirs, files in os.walk(test_mix):
    files.sort()
    for file in files:
        if 'a.wav' in file and num1 < 30:
            tt_s1.write(file+" "+root+'/'+file)
            tt_s1.write('\n')
            num1 += 1
            tt_re.write(file + " " + root + '/' + file)
            tt_re.write('\n')
            num2 += 1
        elif 'b.wav' in file and num3 < 30:
            tt_s2.write(file + " " + root + '/' + file)
            tt_s2.write('\n')
            num3 += 1
        elif num4 < 30:
            tt_mix.write(file + " " + root + '/' + file)
            tt_mix.write('\n')
            num4 += 1
tt_mix.close()
tt_s1.close()
tt_s2.close()
tt_re.close()

val_mix = open(val_mix_scp,'w')
val_s1 = open(val_s1_scp,'w')
val_s2 = open(val_s2_scp,'w')
val_re = open(val_re_scp,'w')
num1 = 0
num2 = 0
num3 = 0
num4 = 0
for root, dirs, files in os.walk(vl_mix):
    files.sort()
    for file in files:
        if 'a.wav' in file and num1 < 30:
            val_s1.write(file+" "+root+'/'+file)
            val_s1.write('\n')
            num4 += 1
            val_re.write(file + " " + root + '/' + file)
            val_re.write('\n')
            num2 += 1
        elif 'b.wav' in file and num3 < 30:
            val_s2.write(file + " " + root + '/' + file)
            val_s2.write('\n')
            num3 += 1
        elif num4 < 30:
            val_mix.write(file + " " + root + '/' + file)
            val_mix.write('\n')
            num4 += 1
val_mix.close()
val_s1.close()
val_s2.close()
val_re.close()

tto_mix = open(test_offset_mix_scp,'w')
tto_s1 = open(test_offset_s1_scp,'w')
tto_s2 = open(test_offset_s2_scp,'w')
tto_re = open(test_offset_re_scp,'w')
num1 = 0
num2 = 0
num3 = 0
num4 = 0
for root, dirs, files in os.walk(test_offset_mix):
    files.sort()
    for file in files:
        if 'a.wav' in file and num1 < 30:
            tto_s1.write(file+" "+root+'/'+file)
            tto_s1.write('\n')
            num1 += 1
        elif 'b.wav' in file and num2 < 30:
            tto_s2.write(file + " " + root + '/' + file)
            tto_s2.write('\n')
            num2 += 1
        elif 're.wav' in file and num3 < 30:
            tto_re.write(file + " " + root + '/' + file)
            tto_re.write('\n')
            num3 += 1
        elif num4 < 30:
            tto_mix.write(file + " " + root + '/' + file)
            tto_mix.write('\n')
            num4 += 1
tto_mix.close()
tto_s1.close()
tto_s2.close()
tto_re.close()
