import os
from random import shuffle

train_mix_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tr_mix.scp'
train_s1_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tr_s1.scp'
train_re_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tr_re.scp'

test_mix_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tt_mix.scp'
test_s1_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tt_s1.scp'
test_re_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tt_re.scp'

val_mix_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/val_mix.scp'
val_s1_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/val_s1.scp'
val_re_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/val_re.scp'

train_mix = '/apdcephfs/share_1316500/helinwang/data/tsss/esc_data_new2/train'
test_mix = '/apdcephfs/share_1316500/helinwang/data/tsss/esc_data_new2/test'
vl_mix = '/apdcephfs/share_1316500/helinwang/data/tsss/esc_data_new2/val'

tr_mix = open(train_mix_scp,'w')
tr_s1 = open(train_s1_scp,'w')
tr_re = open(train_re_scp,'w')
for root, dirs, files in os.walk(train_mix):
    bags = []
    for file in files:
        if 'lab.wav' not in file and 're.wav' not in file:
            bags.append(file)
    leng = len(bags) // 10
    shuffle(bags)
    bags = bags[:leng]
    bags.sort()
    for file in bags:
        tr_s1.write(file.split('.wav')[0] + "_lab.wav " + root + '/' + file.split('.wav')[0] + "_lab.wav")
        tr_s1.write('\n')
        tr_re.write(file.split('.wav')[0] + "_re.wav " + root + '/' + file.split('.wav')[0] + "_re.wav")
        tr_re.write('\n')
        tr_mix.write(file + " " + root + '/' + file)
        tr_mix.write('\n')
tr_mix.close()
tr_s1.close()
tr_re.close()

tt_mix = open(test_mix_scp,'w')
tt_s1 = open(test_s1_scp,'w')
tt_re = open(test_re_scp,'w')
for root, dirs, files in os.walk(test_mix):
    files.sort()
    for file in files:
        if 'lab.wav' in file:
            tt_s1.write(file+" "+root+'/'+file)
            tt_s1.write('\n')
        elif 're.wav' in file:
            tt_re.write(file + " " + root + '/' + file)
            tt_re.write('\n')
        else:
            tt_mix.write(file + " " + root + '/' + file)
            tt_mix.write('\n')
tt_mix.close()
tt_s1.close()
tt_re.close()

val_mix = open(val_mix_scp,'w')
val_s1 = open(val_s1_scp,'w')
val_re = open(val_re_scp,'w')
for root, dirs, files in os.walk(vl_mix):
    files.sort()
    for file in files:
        if 'lab.wav' in file:
            val_s1.write(file+" "+root+'/'+file)
            val_s1.write('\n')
        elif 're.wav' in file:
            val_re.write(file + " " + root + '/' + file)
            val_re.write('\n')
        else:
            val_mix.write(file + " " + root + '/' + file)
            val_mix.write('\n')
val_mix.close()
val_s1.close()
val_re.close()

