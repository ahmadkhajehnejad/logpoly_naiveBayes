import pandas as pd
import numpy as np

device = 'Danmini_Doorbell'
data_root_path = 'data/iot_botnet_attack/' + device + '/'

attacks_paths = ['gafgyt_attacks/combo.csv', 'gafgyt_attacks/junk.csv', 'gafgyt_attacks/scan.csv',
                 'gafgyt_attacks/tcp.csv', 'gafgyt_attacks/udp.csv', 'mirai_attacks/ack.csv',
                 'mirai_attacks/scan.csv', 'mirai_attacks/syn.csv', 'mirai_attacks/udp.csv',
                 'mirai_attacks/udpplain.csv']
benign_path = 'benign_traffic.csv'
data = pd.read_csv(data_root_path + benign_path)
data['label'] = 0

for i, p in enumerate(attacks_paths):
    data_tmp = pd.read_csv(data_root_path + p)
    print(i, p, len(data_tmp))
    data_tmp['label'] = i+1
    #data_tmp['label'] = 1
    data = data.append(data_tmp)

data = data.sample(frac=1).reset_index(drop=True)

feature_types = pd.DataFrame(columns = list(data.columns)[:-1], data=[['c'] * (len(data.columns)-1)])
feature_types.to_csv('data/iot_botnet_attack_' + device + '_11classes_feature_types.csv', index=None, header= None)

data.to_csv('data/iot_botnet_attack_' + device + '_11classes.csv', index=None, header= None)
