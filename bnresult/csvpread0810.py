"""
浠巄n鎺ㄦ柇鐨刢sv缁撴灉涓鍙栨潯浠惰〃
璇诲彇缁撴灉鐩爣锛�
conditional_prob_table = {
                'risk1': {
                    'threat1': 0.295,
                    'threat2': 0.705,
                    'threat3': 0,

                    'drive1': 0.7192,
                    'drive2': 0.0829,
                    'drive3': 0.1979,
                },
                'risk2': {
                    'threat1': 0.2959,

                    'threat2': 0.7026,
                    'threat3': 0.0016,
                    'drive1': 0.8016,
                    'drive2': 0.067,
                    'drive3': 0.1314,
                },
                'risk3': {
                    'threat1': 0.2991,

                    'threat2': 0.6991,
                    'threat3': 0.0017,
                    'drive1': 0.9597,
                    'drive2': 0.0265,
                    'drive3': 0.0138,
                },
                'drive1': {
                    'threat1': 0.3439,

                    'threat2': 0.6555,
                    'threat3': 0.0006,
                    'emo1': 0.6118,
                    'emo2': 0.2919,
                    'emo3': 0.0963,
                    'radi1': 0.6064,

                    'radi2': 0.3628,
                    'radi3': 0.0309,
                },
                'drive2': {
                    'threat1': 0.0101,

                    'threat2': 0.9878,
                    'threat3': 0.0021,
                    'emo1': 0.4882,
                    'emo2': 0.2329,
                    'emo3': 0.2789,
                    'radi1': 0.3875,

                    'radi2': 0.5951,
                    'radi3': 0.0174,
                },
                'drive3': {
                    'threat1': 0.1038,

                    'threat2': 0.8918,
                    'threat3': 0.0043,
                    'emo1': 0.6565,
                    'emo2': 0.3132,
                    'emo3': 0.0302,
                    'radi1': 0.7068,

                    'radi2': 0.2696,
                    'radi3': 0.0236,
                },
            }
"""

import pandas as pd
import re
import pickle

# 读取xlsx文件
global scene
global modelname
scene = 'ghost'
modelname = "cpgcn0810"
read_path = "F:\\TLY\\g2\\SOR\\code\\GCN+LSTM\\GCN-LSTM0704\\bnresult\\xlsx\\"+ scene + "\\" + modelname + ".xlsx"
# path_route = "F:\\TLY\\g2\\SOR\\code\\GCN+LSTM\\GCN-LSTM0704\\bnresult0710\\xlsx"
data = pd.read_excel(read_path, sheet_name=None)
# 鑾峰彇鎵�鏈夊瓙琛ㄥ悕绉�
sheet_names = list(data.keys())


#棣栧厛浠庤〃涓垱寤轰竴涓┖鐨勮濉殑dict
prodict = {}

#璇诲彇鎵�鏈夌殑鑺傜偣
node_list = []
for sheet_name in sheet_names:
    s_list = sheet_name.split("to")
    # 鑾峰彇鍒嗗壊鍚庣殑瀛楃涓�
    s1 = s_list[0].strip('_')  # 鍘婚櫎棣栧熬_
    s2 = s_list[1].strip('_')
    node_list.append(s1)
    node_list.append(s2)
node_list = list(set(node_list))
# prodict = {key: {} for key in node_list}
print('宸插畬鎴�')

#鍒涘缓姣忎釜閿�肩殑閿�硷紝瀹冨紑澶寸殑鏂囦欢锛岀涓�琛屾槸浠栫殑鏁伴噺
node_listquan = []
for node in node_list:
    node_temp = []
    if node == 'risk_gradet-1':
        num_list = [0,1,2]
        for num in num_list:
            strtt = node + str(num)
            node_temp.append(strtt)
    else:
        for sheet_name in sheet_names:
            str1 = node
            str2 = sheet_name
            pair = str2[:len(str1)]
            if pair == str1:
                num_list = data[str2].columns.tolist()
                for num in num_list:
                    strtt = node + str(num)
                    node_temp.append(strtt)
                break
    node_listquan.extend(node_temp)
    print(node)

print('yiwang')
prodict = {key: {} for key in node_listquan}
for condition in node_listquan:
    for event in node_listquan:
        ree = re.split(r'(\D+)', condition)
        condition_name = result = re.split(r'(\D+)', condition)[1]
        condition_num = result = re.split(r'(\D+)', condition)[2]
        event_name = result = re.split(r'(\D+)', event)[1]
        event_num = result = re.split(r'(\D+)', event)[2]
        procsv_name = condition_name + '_to_' + event_name
        if procsv_name in sheet_names:
            prodata = data[procsv_name]
            valuet = prodata.iloc[int(event_num), int(condition_num)]
            prodict[condition].update({event:prodata.iloc[int(event_num), int(condition_num)]})
print('宸插畬鎴恉ict鍒朵綔')
import pickle

# 存储转换完成的pkl文件
dict_save_path = "F:\\TLY\\g2\\SOR\\code\\GCN+LSTM\\GCN-LSTM0704\\bnresult\\pkl\\"+ scene + "\\" + modelname + ".pkl"
with open(dict_save_path, 'wb') as f:
    pickle.dump(prodict, f)

# 浠庢枃浠跺姞杞藉瓧鍏�
dict_load_path = "F:\\TLY\\g2\\SOR\\code\\GCN+LSTM\\GCN-LSTM0704\\bnresult\\pkl\\"+ scene + "\\" + modelname + ".pkl"
with open(dict_load_path, 'rb') as f:
    loaded_dict = pickle.load(f)


print("鍘昏瀹屾垚")

# 鎵"""
浠巄n鎺ㄦ柇鐨刢sv缁撴灉涓鍙栨潯浠惰〃
璇诲彇缁撴灉鐩爣锛�
conditional_prob_table = {
                'risk1': {
                    'threat1': 0.295,
                    'threat2': 0.705,
                    'threat3': 0,

                    'drive1': 0.7192,
                    'drive2': 0.0829,
                    'drive3': 0.1979,
                },
                'risk2': {
                    'threat1': 0.2959,

                    'threat2': 0.7026,
                    'threat3': 0.0016,
                    'drive1': 0.8016,
                    'drive2': 0.067,
                    'drive3': 0.1314,
                },
                'risk3': {
                    'threat1': 0.2991,

                    'threat2': 0.6991,
                    'threat3': 0.0017,
                    'drive1': 0.9597,
                    'drive2': 0.0265,
                    'drive3': 0.0138,
                },
                'drive1': {
                    'threat1': 0.3439,

                    'threat2': 0.6555,
                    'threat3': 0.0006,
                    'emo1': 0.6118,
                    'emo2': 0.2919,
                    'emo3': 0.0963,
                    'radi1': 0.6064,

                    'radi2': 0.3628,
                    'radi3': 0.0309,
                },
                'drive2': {
                    'threat1': 0.0101,

                    'threat2': 0.9878,
                    'threat3': 0.0021,
                    'emo1': 0.4882,
                    'emo2': 0.2329,
                    'emo3': 0.2789,
                    'radi1': 0.3875,

                    'radi2': 0.5951,
                    'radi3': 0.0174,
                },
                'drive3': {
                    'threat1': 0.1038,

                    'threat2': 0.8918,
                    'threat3': 0.0043,
                    'emo1': 0.6565,
                    'emo2': 0.3132,
                    'emo3': 0.0302,
                    'radi1': 0.7068,

                    'radi2': 0.2696,
                    'radi3': 0.0236,
                },
            }
"""

import pandas as pd
import re
import pickle

# 读取xlsx文件
global scene
global modelname
scene = 'ghost'
modelname = "cpgcn0810"
read_path = "F:\\TLY\\g2\\SOR\\code\\GCN+LSTM\\GCN-LSTM0704\\bnresult\\xlsx\\"+ scene + "\\" + modelname + ".xlsx"
# path_route = "F:\\TLY\\g2\\SOR\\code\\GCN+LSTM\\GCN-LSTM0704\\bnresult0710\\xlsx"
data = pd.read_excel(read_path, sheet_name=None)
# 鑾峰彇鎵�鏈夊瓙琛ㄥ悕绉�
sheet_names = list(data.keys())


#棣栧厛浠庤〃涓垱寤轰竴涓┖鐨勮濉殑dict
prodict = {}

#璇诲彇鎵�鏈夌殑鑺傜偣
node_list = []
for sheet_name in sheet_names:
    s_list = sheet_name.split("to")
    # 鑾峰彇鍒嗗壊鍚庣殑瀛楃涓�
    s1 = s_list[0].strip('_')  # 鍘婚櫎棣栧熬_
    s2 = s_list[1].strip('_')
    node_list.append(s1)
    node_list.append(s2)
node_list = list(set(node_list))
# prodict = {key: {} for key in node_list}
print('宸插畬鎴�')

#鍒涘缓姣忎釜閿�肩殑閿�硷紝瀹冨紑澶寸殑鏂囦欢锛岀涓�琛屾槸浠栫殑鏁伴噺
node_listquan = []
for node in node_list:
    node_temp = []
    if node == 'risk_gradet-1':
        num_list = [0,1,2]
        for num in num_list:
            strtt = node + str(num)
            node_temp.append(strtt)
    else:
        for sheet_name in sheet_names:
            str1 = node
            str2 = sheet_name
            pair = str2[:len(str1)]
            if pair == str1:
                num_list = data[str2].columns.tolist()
                for num in num_list:
                    strtt = node + str(num)
                    node_temp.append(strtt)
                break
    node_listquan.extend(node_temp)
    print(node)

print('yiwang')
prodict = {key: {} for key in node_listquan}
for condition in node_listquan:
    for event in node_listquan:
        ree = re.split(r'(\D+)', condition)
        condition_name = result = re.split(r'(\D+)', condition)[1]
        condition_num = result = re.split(r'(\D+)', condition)[2]
        event_name = result = re.split(r'(\D+)', event)[1]
        event_num = result = re.split(r'(\D+)', event)[2]
        procsv_name = condition_name + '_to_' + event_name
        if procsv_name in sheet_names:
            prodata = data[procsv_name]
            valuet = prodata.iloc[int(event_num), int(condition_num)]
            prodict[condition].update({event:prodata.iloc[int(event_num), int(condition_num)]})
print('宸插畬鎴恉ict鍒朵綔')
import pickle

# 存储转换完成的pkl文件
dict_save_path = "F:\\TLY\\g2\\SOR\\code\\GCN+LSTM\\GCN-LSTM0704\\bnresult\\pkl\\"+ scene + "\\" + modelname + ".pkl"
with open(dict_save_path, 'wb') as f:
    pickle.dump(prodict, f)

# 浠庢枃浠跺姞杞藉瓧鍏�
dict_load_path = "F:\\TLY\\g2\\SOR\\code\\GCN+LSTM\\GCN-LSTM0704\\bnresult\\pkl\\"+ scene + "\\" + modelname + ".pkl"
with open(dict_load_path, 'rb') as f:
    loaded_dict = pickle.load(f)


print("鍘昏瀹屾垚")

# 鎵撳嵃鍔犺浇鐨勫瓧鍏�
print(loaded_dict)
