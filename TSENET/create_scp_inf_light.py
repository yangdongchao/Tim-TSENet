#
train_inf_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tr_inf.scp'
test_inf_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tt_inf.scp'
val_inf_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/val_inf.scp'
train_txt = '/apdcephfs/share_1316500/helinwang/data/tsss/esc_data_new2/train.txt'
test_txt = '/apdcephfs/share_1316500/helinwang/data/tsss/esc_data_new2/test.txt'
val_txt = '/apdcephfs/share_1316500/helinwang/data/tsss/esc_data_new2/val.txt'
train_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tr_mix.scp'
test_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/tt_mix.scp'
val_scp = '/apdcephfs/private_helinwang/tsss/TSENet/scps_light/val_mix.scp'


with open(train_txt, 'r') as f:
    data = f.readlines()
tr_inf = open(train_inf_scp,'w')
file_list = []
line = 0
lines = open(train_scp, 'r').readlines()
for l in lines:
    scp_parts = l.strip().split()
    line += 1
    key, value = scp_parts
    file_list.append(key)

for f in file_list:
    for i in range(len(data)):
        file = data[i].split('\t')[0]
        if f == file:
            cls = data[i].split('\t')[7]
            onset = data[i].split('\t')[5]
            offset = data[i].split('\t')[6]
            tr_inf.write(file + " " + cls + " " + onset + " " + offset)
            tr_inf.write('\n')

tr_inf.close()


with open(test_txt, 'r') as f:
    data = f.readlines()
tt_inf = open(test_inf_scp,'w')
file_list = []
line = 0
lines = open(test_scp, 'r').readlines()
for l in lines:
    scp_parts = l.strip().split()
    line += 1
    key, value = scp_parts
    file_list.append(key)

for f in file_list:
    for i in range(len(data)):
        file = data[i].split('\t')[0]
        if f == file:
            cls = data[i].split('\t')[7]
            onset = data[i].split('\t')[5]
            offset = data[i].split('\t')[6]
            tt_inf.write(file + " " + cls + " " + onset + " " + offset)
            tt_inf.write('\n')

tt_inf.close()


with open(val_txt, 'r') as f:
    data = f.readlines()
val_inf = open(val_inf_scp,'w')
file_list = []
line = 0
lines = open(val_scp, 'r').readlines()
for l in lines:
    scp_parts = l.strip().split()
    line += 1
    key, value = scp_parts
    file_list.append(key)
for f in file_list:
    for i in range(len(data)):
        file = data[i].split('\t')[0]
        if f == file:
            cls = data[i].split('\t')[7]
            onset = data[i].split('\t')[5]
            offset = data[i].split('\t')[6]
            val_inf.write(file + " " + cls + " " + onset + " " + offset)
            val_inf.write('\n')

val_inf.close()


