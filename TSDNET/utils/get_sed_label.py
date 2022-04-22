import pandas as pd
save_name = []
save_onset = []
save_offset = []
save_event = []
with open('/apdcephfs/share_1316500/donchaoyang/tsss/tsss_data/fsd_2018/scps/val_inf_new.scp', 'r') as file:
    for line in file:
        split_line = line.strip().split(' ')
        save_name.append(split_line[0])
        #target_audio = split_line[1]
        class_event = split_line[1]
        save_event.append('class_'+class_event)
        save_onset.append(split_line[2])
        save_offset.append(split_line[3])
        #print(class_event)
        # print('split_line ',split_line)
        # assert 1==2
dict = {'filename': save_name, 'onset': save_onset, 'offset': save_offset, 'event_label': save_event}
df = pd.DataFrame(dict)
df.to_csv('strong_label_fsd2018_val_new.tsv',index=False,sep='\t')