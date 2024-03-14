from keras.optimizers import SGD, Adadelta, RMSprop, Adam   #SGD=随机梯度下降；Adadelta=优化方法，这些都是优化算法；
from keras.models import Sequential
from keras.layers import Masking, Embedding   #keras的一层，Embedding层只能作为第一层
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras import backend as K
import numpy as np
import pandas as pd
import csv
import os
import math
from keras.models import load_model  #加载模型
from sklearn.model_selection import train_test_split
from utils2 import *
from graph8 import GraphConvolution
from keras.regularizers import l2
from keras.layers import Permute, Dense, TimeDistributed, Conv2D, MaxPooling2D, Multiply, Dropout, Flatten, Reshape, \
    LSTM, UpSampling2D, concatenate, Input
from keras.models import Model
from keras.callbacks import EarlyStopping  #callback=回调函数  earlystoping=提前停止训练（可以达到当训
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import h5py
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
# from graphgen import *
# from utils88 import *
import keras as K
import keras.backend as KB
from graph8 import GraphConvolution
from keras.regularizers import l2
from keras.layers import Permute, Dense, TimeDistributed, Conv2D, MaxPooling2D, Multiply, Dropout, Flatten, Reshape, \
    LSTM, UpSampling2D, concatenate, Input, Bidirectional, Lambda, RepeatVector
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping  #callback=回调函数  earlystoping=提前停止训练（可以达到当训
import networkx as nx
import matplotlib.pyplot as plt
import math
import pickle
import sklearn.preprocessing as pre
import pickle
# from SpatialSelfAttentionLayer import SpatialSelfAttentionLayer
from metrixcompu import zhibiao
import itertools
import h5py
import time


SINGLE_ATTENTION_VECTOR = False
"""
加载数据：
加载模型：
输出预测的指标：
绘制轨迹图：
"""
n_step_in = 75
n_step_out = 25
n_node_in = 2
n_node_out = 1
n_feature_in = 4
n_feature_out = 2
#psyn_node_in = 8
psyn_node_in = 15
psyn_feature_in = 1
MAX_DEGREE = 2  # maximum polynomial degree
support = MAX_DEGREE + 1
scene = 'brake'
ceshi = '5_10_2_1'
modelname = 'cpsorgcn0818'
# modelname_save = 'cpgcn0810'
# ceshimodel = 'cpgcn-3s-1s'
modelname_save = 'cpsorgcn0818'
ceshimodel = 'cpsorgcn-3s-1s'
"""
几个重要路径汇总
测试数据的load路径：'../data/traindata/'+ scene +'/'+ modelname + '/'+ ceshi +'/Y_test_splice_normalized.npy'
model文件的load路径：'../modelresult/' + scene + '/' + modelname_save
                    model_savepath + '/' + ceshimodel + '.h5'
real and pre 储存路径：'../predictresult/' + scene + '/' + modelname + '/' + ceshi
scaler文件的load路径：
"""
###############################################数据加载#####################################################
def arraytolist_graph(arr1):
    result = []
    for i in range(arr1.shape[0]):
        inner_result = []
        for j in range(arr1.shape[1]):
            inner_result.append(arr1[i,j])
        result.append(inner_result)
    return result
def arraytolist_feature(arr1):
    result = []
    for i in range(arr1.shape[0]):
        result.append(arr1[i])
    return result
name_list_feature = ['Xtestphy_splice_normalized','Xtestpsy_splice_normalized']
name_list_graph = ['graph_test_phy_splice','graph_test_psy_splice']
feature_dict = {}
for name in name_list_feature:
    # filename = '../data/traindata/' + scene + '/' + name + '.npy'
    filename = '../data/traindata/'+ scene +'/'+ modelname + '/' + ceshi + '/' + name + '.npy'
    data_temp = np.load(filename, allow_pickle=True)
    # data_temp = arraytolist_feature(data_temp)
    feature_dict[name] = data_temp

graph_dict = {}
for name in name_list_graph:
    filename = '../data/traindata/'+ scene +'/'+ modelname + '/'+ ceshi + '/' + name + '.npy'
    # filename = '../data/traindata/' + scene + '/' + name + '.npy'
    data_tempp = np.load(filename,allow_pickle=True)
    # data_tempp = arraytolist_graph(data_tempp)
    graph_dict[name] = data_tempp
Xtestphy_splice_normalized = feature_dict['Xtestphy_splice_normalized']
Xtestpsy_splice = feature_dict['Xtestpsy_splice_normalized']
graph_test_phy_splice = list(graph_dict['graph_test_phy_splice'])
graph_test_psy_splice = list(graph_dict['graph_test_psy_splice'])
X_testinput = [Xtestphy_splice_normalized, graph_test_phy_splice, Xtestpsy_splice, graph_test_psy_splice]
y_test_route = '../data/traindata/'+ scene +'/'+ modelname + '/'+ ceshi +'/Y_test_splice_normalized.npy'
# y_test_route = '../data/traindata/'+ scene +'/Y_test_splice_normalized.npy'
Y_test_splice_normalized = np.load(y_test_route)
########################################### 读取训练好的模型 #################################################
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #设置环境变量，设置结果显示的级别，解释在word里
# n_feature = 20  #输入特征数
# n_input = 12 #输入轨迹点的个数
# # 定义自定义对象字典
custom_objects = {'GraphConvolution': GraphConvolution}
model_savepath = '../modelresult/' + scene + '/' + modelname_save
modelload = model_savepath + '/' + ceshimodel + '.h5'
model = load_model(modelload, custom_objects = custom_objects)   #读取模型“cnnlstm（1223）.h5”
model.summary()  ##打印出模型概况，它实际调用的是keras.utils.print_summary
p = model.predict(X_testinput)

# 存储为npz文件
# 存储为npz文件
pre_save = '../predictresult/' + scene + '/' + modelname + '/' + ceshi
try:
    path_pre = pre_save + '/y_pre.npz'
    path_real = pre_save + '/y_real.npz'
    np.savez(path_pre,arr=p)
    np.savez(path_real,arr = Y_test_splice_normalized)
except:
    os.mkdir(pre_save)
    path_pre = pre_save + '/y_pre.npz'
    path_real = pre_save + '/y_real.npz'
    np.savez(path_pre, arr=p)
    np.savez(path_real, arr=Y_test_splice_normalized)
#################################计算指标
n_feature_out = 2
dim1,dim2 = p.shape[:2]
scaler_xy_save = '../data/scaler/' + scene + '_' + 'scaler_xy.pickle'
with open(scaler_xy_save, 'rb') as f:
    scaler_xy = pickle.load(f)
result_arr = np.reshape(p,(-1,n_feature_out))
p  = scaler_xy.inverse_transform(result_arr)
p = np.reshape(p, (dim1,dim2,1,n_feature_out))
r = np.reshape(Y_test_splice_normalized,(-1,n_feature_out))
r  = scaler_xy.inverse_transform(r)
r = np.reshape(r,(dim1,dim2,1,n_feature_out))
dimx1,dimx2,dimx3 = Xtestphy_splice_normalized.shape[:3]
Xtestphy_splice_normalized = np.reshape(Xtestphy_splice_normalized,(-1,n_feature_out))
Xtestphy_splice_normalized  = scaler_xy.inverse_transform(Xtestphy_splice_normalized)
Xtestphy_splice_normalized = np.reshape(Xtestphy_splice_normalized,(dimx1,dimx2,dimx3,n_feature_out+2))
# tester_pred = list(p[:,:,0,:])
# # npc_pred = list(p[:,:,1,:])
# tester_real = list(r[:,:,0,:])
# # npc_real = list(r[:,:,1,:])
# tester_zhibiao = []
# # npc_zhibiao = []
# for predicted_trajectories, true_trajectories in zip(tester_pred,tester_real):
#     tester_temp = zhibiao(predicted_trajectories, true_trajectories)
#     tester_zhibiao.append(tester_temp)
#
# # for predicted_trajectories, true_trajectories in zip(npc_pred,npc_real):
# #     npc_temp = zhibiao(predicted_trajectories, true_trajectories)
# #     npc_zhibiao.append(npc_temp)
#
# tester = np.array(tester_zhibiao)
# # npc = np.array(npc_zhibiao)
# tester_result = np.zeros((4,1))
# # npc_result = np.zeros((4,1))
# for i in range(4):
#     tester_result[i] = np.mean(tester[:, i])
#     # npc_result[i] = np.mean(npc[:, i])
# print(tester_result)
# # print(npc_result)
# print("指标计算完成")
#################################绘制轨迹图
s = 1
for i in range(p.shape[0]):
# for i in range(100):
    if i%5 == 0 :
        #5_17_0
        # x1 = p[i][0][0][0]-10
        # y1 = p[i][0][0][1]-0.1
        # x1 = p[i][0][0][0]-26
        # y1 = p[i][0][0][1]-0.3
        x1 = p[i][0][0][0]
        y1 = p[i][0][0][1]
        if i == 0:
            x_y1 = np.array([x1, y1])
        else:
            x_y1 = np.vstack((x_y1, [x1, y1]))

df_pre = pd.DataFrame(x_y1, columns=['x', 'y'])

# for i in range(p2.shape[0]):
# # for i in range(100):
#     if i%1 == 0 :
#         x2 = p2[i][0][0][0]
#         y2 = p2[i][0][0][1]
#         if i == 0:
#             x_y2 = np.array([x2, y2])
#         else:
#             x_y2 = np.vstack((x_y2, [x2, y2]))
#
# df_pre2 = pd.DataFrame(x_y2, columns=['x', 'y'])

for i in range(r.shape[0]):
# for i in range(100):
    if i%5 ==0 :
        x3 = r[i][0][0][0]
        y3 = r[i][0][0][1]
        if i == 0:
            x_y3 = np.array([x3, y3])
        else:
            x_y3 = np.vstack((x_y3, [x3, y3]))

df_real = pd.DataFrame(x_y3, columns=['x', 'y'])

for i in range(Xtestphy_splice_normalized.shape[0]):
# for i in range(100):
    if i%5 ==0 :
        x4 = Xtestphy_splice_normalized[i][-1][1][0]
        y4 = Xtestphy_splice_normalized[i][-1][1][1]
        if i == 0:
            x_y4 = np.array([x4, y4])
        else:
            x_y4 = np.vstack((x_y4, [x4, y4]))
df_x = pd.DataFrame(x_y4, columns=['x', 'y'])

#绘制x、y的折线图
plt.figure(figsize = (10,5))
plt.plot(df_pre['x'], df_pre['y'], marker = 'o',color = 'r',label = 'subject-predict')
# plt.plot(df_pre2['x'], df_pre2['y'], 'g', label='pre-2')
plt.plot(df_real['x'], df_real['y'], marker = 'o', color = 'b', label='subject-real')
plt.plot(df_x['x'], df_x['y'], marker = 'o', color = 'g', label='npc')
plt.title('test cumulative count', fontsize=30)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
#2_17_0
# plt.xlim(120, 260)
# plt.ylim(-0.5,2)
#brake 5_1-1
# plt.xlim(-10, 100)
# plt.ylim(-2,2)
#brake 5_10_2
plt.xlim(-10, 100)
plt.ylim(-2.5,1.5)
plt.legend()
figsave_path = '../predictresult/' + scene + '/' + modelname + '/' + ceshi + '_trait.png'
plt.savefig(figsave_path)
plt.show()