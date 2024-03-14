#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import numpy as np
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
# import sklearn.preprocessing as pre
import pickle
# from SpatialSelfAttentionLayer import SpatialSelfAttentionLayer
import numpy as np
import os
import itertools
import h5py
import time
import tracemalloc

# from datagen11 import *
os.environ['CUDA_VISIBLE_DEVICES' ]='0' # 指定哪块GPU训练
config=tf.compat.v1.ConfigProto()# 设置最大有GPU不超过显存的8% (可选# config.gpu options.per_process_gpu memory fraction=0.8
config.gpu_options.allow_growth = True # 设置动态分配GPU内存
sess=tf.compat.v1.Session( config=config)
SINGLE_ATTENTION_VECTOR = False
"""
Input:
1,in physical level:
G_nodes_list:(每个时刻的交互对象列表)，原始数据再处理
G_nodes_edges:都为1
G运算得到support级运算得到GG列表list3 (None, nodes, nodes)
F(None, 5,2, 4):每时刻的每个点的xyva

2,in psychological level:
G_nodes_list: 根据BN网固定
G_nodes_edges: 客观——主观指定，概率表查询
G运算得到support级运算得到GG列表list3 (None, nodes, nodes)
F（None，5，15,1）：每个时刻节点的值，原始数据处理得到

output(y)
每个时刻本车的xy
"""
# from tensorflow.keras.utils import plot_model
"""
n_step_in = 5
n_step_out = 2
n_node_out = 1
n_feature_in = 4
n_feature_out = 2
psyn_node_in = 9
psyn_feature_in = 1
Xtrainphy,Xvalphy,Xtestphy
Xtrainpsy,Xvalpsy,Xtestpsy
graph_train_phy,graph_val_phy,graph_test_phy
graph_train_phy,graph_val_phy,graph_test_phy
"""
n_step_in = 75
n_step_out = 25
n_node_in = 2
n_node_out = 1
n_feature_in = 4
n_feature_out = 2
#psyn_node_in = 8
psyn_node_in = 21
psyn_feature_in = 1
MAX_DEGREE = 2  # maximum polynomial degree
support = MAX_DEGREE + 1
scene = 'cutin'
ceshi = 'cpsorgcn-1s-1s'
modelname = 'cpsorgcn0918'
tracemalloc.start()
def myreshape(data):
    new_data = []
    for x in data:
        seq_len = x.shape[0]
        dim1 = x.shape[1]
        dim2 = x.shape[2]
        dim0 = int(seq_len/n_step_in)
        # dim0 = seq_len
        # print(dim0)
        x_new = x[:dim0*n_step_in,:,:]
        x_new = tf.expand_dims(x_new, axis=0)
        # print(x_new.shape)
        x_new = KB.reshape(x_new,(dim0,n_step_in,dim1,dim2))
        new_data.append(x_new)
    print(new_data[0].shape)
    return new_data,dim0*n_step_in

def myreshape1(data):
    new_data = []
    # data = np.array(data)
    # data1 = data.transpose((1, 0, 2, 3))
    for x in data:
        seq_len = x.shape[0]
        dim1 = x.shape[1]
        dim2 = x.shape[2]
        dim0 = seq_len
        temp = []
        for k in range((seq_len-n_step_in)):
            x_new = x[k:k+n_step_in,:,:]
            x_new = tf.expand_dims(x_new, axis=0)
            temp.append(x_new)
        temp = concatenate(temp,0)
        # print(temp.shape)
    new_data.append(temp)
    print('finished one!')
    return new_data,seq_len-n_step_in

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = concatenate([inputs,a_probs],axis=-1)
    return output_attention_mul
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


name_list_feature = ['Xtrainphy_splice_normalized','Xvalphy_splice_normalized','Xtestphy_splice_normalized',
                        'Xtrainpsy_splice_normalized','Xvalpsy_splice_normalized','Xtestpsy_splice_normalized']

name_list_graph = ['graph_train_phy_splice','graph_val_phy_splice','graph_test_phy_splice',
                   'graph_train_psy_splice', 'graph_val_psy_splice', 'graph_test_psy_splice']
feature_dict = {}
ceshiname = modelname +'/'+ ceshi
for name in name_list_feature:
    # filename = '../data/traindata/' + scene + '/' + name + '.npy'
    filename = '../data/traindata/'+ scene +'/'+ ceshiname + '/' + name + '.npy'
    data_temp = np.load(filename, allow_pickle=True)
    # data_temp = arraytolist_feature(data_temp)
    feature_dict[name] = data_temp

graph_dict = {}
for name in name_list_graph:
    filename = '../data/traindata/'+ scene +'/'+ ceshiname + '/' + name + '.npy'
    # filename = '../data/traindata/' + scene + '/' + name + '.npy'
    data_tempp = np.load(filename,allow_pickle=True)
    # data_tempp = arraytolist_graph(data_tempp)
    graph_dict[name] = data_tempp

Xtrainphy_splice_normalized = feature_dict['Xtrainphy_splice_normalized']
Xvalphy_splice_normalized = feature_dict['Xvalphy_splice_normalized']
Xtestphy_splice_normalized = feature_dict['Xtestphy_splice_normalized']
Xtrainpsy_splice =  feature_dict['Xtrainpsy_splice_normalized']
Xvalpsy_splice = feature_dict['Xvalpsy_splice_normalized']
Xtestpsy_splice = feature_dict['Xtestpsy_splice_normalized']
graph_train_phy_splice = list(graph_dict['graph_train_phy_splice'])
graph_val_phy_splice = list(graph_dict['graph_val_phy_splice'])
graph_test_phy_splice = list(graph_dict['graph_test_phy_splice'])
graph_train_psy_splice = list(graph_dict['graph_train_psy_splice'])
graph_val_psy_splice = list(graph_dict['graph_val_psy_splice'])
graph_test_psy_splice = list(graph_dict['graph_test_psy_splice'])

X_input = [Xtrainphy_splice_normalized, graph_train_phy_splice, Xtrainpsy_splice, graph_train_psy_splice]
X_valinput = [Xvalphy_splice_normalized, graph_val_phy_splice, Xvalpsy_splice, graph_val_psy_splice]
X_testinput = [Xtestphy_splice_normalized, graph_test_phy_splice, Xtestpsy_splice, graph_test_psy_splice]
y_train_route = '../data/traindata/'+ scene +'/'+ ceshiname +'/Y_train_splice_normalized.npy'
# y_train_route = '../data/traindata/'+ scene +'/Y_train_splice_normalized.npy'
Y_train_splice_normalized = np.load(y_train_route)
y_val_route = '../data/traindata/'+ scene +'/'+ ceshiname +'/Y_val_splice_normalized.npy'
# y_val_route = '../data/traindata/'+ scene +'/Y_val_splice_normalized.npy'
Y_val_splice_normalized = np.load(y_val_route)
y_test_route = '../data/traindata/'+ scene +'/'+ ceshiname +'/Y_test_splice_normalized.npy'
# y_test_route = '../data/traindata/'+ scene +'/Y_test_splice_normalized.npy'
Y_test_splice_normalized = np.load(y_test_route)

###########################################模型构建——物理层##################################################

#(None,50,2,2)
Gn_phy = [Input(batch_shape=(None,n_step_in, n_node_in, n_node_in), sparse=False) for _ in range(support)]
#(None,50, 24)
X_in_phy = Input(shape=(n_step_in,n_node_in, n_feature_in))
#(None，50, 2, 64)
H_phy = GraphConvolution(64, support, activation='relu',kernel_regularizer=l2(5e-4))([X_in_phy] + [[Gn_phy]])
H_phy = Dropout(rate=0.1)(H_phy)
H_phy = GraphConvolution(32, support, activation='relu',kernel_regularizer=l2(5e-4))([X_in_phy] + [[Gn_phy]])
H_phy = Dropout(rate=0.1)(H_phy)
H_phy = GraphConvolution(16, support, activation='relu',kernel_regularizer=l2(5e-4))([X_in_phy] + [[Gn_phy]])
H_phy = Dropout(rate=0.1)(H_phy)
H_phy = GraphConvolution(8, support, activation='relu',kernel_regularizer=l2(5e-4))([X_in_phy] + [[Gn_phy]])
H_phy = Dropout(rate=0.1)(H_phy)
H_phy = GraphConvolution(4, support, activation='relu',kernel_regularizer=l2(5e-4))([X_in_phy] + [[Gn_phy]])
H_phy = Dropout(rate=0.1)(H_phy)
#(None,50,2,1)
YY_phy = GraphConvolution(1, support, activation='relu')([H_phy] + [[Gn_phy]])  # new[None, None, 256, 1]
model1 = tf.keras.Model(
    inputs=[X_in_phy, Gn_phy],
    outputs=YY_phy
)
# model1.compile(optimizer='adam', loss='mse') #编译
# time.sleep(100)


# history1 = model1.fit(X_input_phy,
#                     Y_train_splice_normalized,
#                     batch_size=64, epochs=100, verbose=1)

#50
yphydim2 = YY_phy.shape[1]
#2
yphydim3 = YY_phy.shape[2]
#None,50,2
g_reshape_phy = Reshape((yphydim2, yphydim3))(YY_phy)

#list3 (None,50,14,14)
Gn_psy = [Input(batch_shape=(None,n_step_in, psyn_node_in, psyn_node_in), sparse=False) for _ in range(support)]
#None 50,14,1
X_in_psy = Input(shape=(n_step_in,psyn_node_in, psyn_feature_in))
#None 50 14 16
H_psy = GraphConvolution(64, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in_psy] + [[Gn_psy]])
H_psy = Dropout(rate=0.1)(H_psy)
H_psy = GraphConvolution(32, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in_psy] + [[Gn_psy]])
H_psy = Dropout(rate=0.1)(H_psy)
H_psy = GraphConvolution(16, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in_psy] + [[Gn_psy]])
H_psy = Dropout(rate=0.1)(H_psy)
H_psy = GraphConvolution(8, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in_psy] + [[Gn_psy]])
H_psy = Dropout(rate=0.1)(H_psy)
H_psy = GraphConvolution(4, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in_psy] + [[Gn_psy]])
H_psy = Dropout(rate=0.1)(H_psy)
#None,50,14,1
YY_psy = GraphConvolution(1, support, activation='relu')([H_psy] + [[Gn_psy]])
model2 = tf.keras.Model(
    inputs=[X_in_psy, Gn_psy],
    outputs=YY_psy
)
ypsydim2 = YY_psy.shape[1]
ypsydim3 = YY_psy.shape[2]
g_reshape_psy = Reshape((ypsydim2, ypsydim3))(YY_psy)
Y = concatenate([g_reshape_phy, g_reshape_psy, X_in_phy[:,:,0],X_in_psy[:,:,:,0]], axis=-1)
bilstm = LSTM(128, dropout=0.1, return_sequences=True,input_shape=(n_step_in, n_node_in + psyn_node_in+25))(Y)
lstm_out = concatenate([bilstm, X_in_phy[:,:,0],X_in_psy[:,:,:,0]], axis=-1)
lstm_out = Dropout(0.1)(lstm_out)
attention_mul = attention_3d_block(lstm_out)
attention_mul = Flatten()(attention_mul)
dense = Dense(50)(attention_mul)
dropout = Dropout(0.2)(dense)
dense = Dense(n_step_out*n_node_out*n_feature_out)(dropout)
dense2 = Reshape((n_step_out,n_node_out,n_feature_out))(dense)

inputss = model1.input + model2.input
model = Model(inputs=inputss, outputs=dense2)
model.compile(optimizer='adam', loss='mse') #编译
model.summary() #打印出模型概况
early_stopping = EarlyStopping(monitor='val_loss', patience=20) #停止条件  monitor='val_loss'表示监视条件   patience=2表示当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。一般情况设置此值较大些，因为神经网络学习本来就不稳定，只有加大些才能获得“最优”解。
history = model.fit(X_input,
                    Y_train_splice_normalized,
                    batch_size=64, epochs=100, verbose=1,
                    validation_data=(X_valinput, Y_val_splice_normalized),
                    callbacks=[early_stopping])

model_savepath = '../modelresult/' + scene + '/' + modelname
try:
    model.save(model_savepath + '/' + ceshi + '.h5')
except:
    os.mkdir(model_savepath)
    model.save(model_savepath + '/' + ceshi + '.h5')
# model.save('../modelresult/gcn-4s-2s-SOR-emo.h5') #保存模型 ，保存格式是h5

with open(r'../data/result_Gcnlstm.csv', 'w') as ff: #以写入的方式打开file文件，并将文件存储到变量中
    ff.write(str(history.history))  ##write表示将字符串str写入到文件，str()将对象转化为适合人阅读的格式

print('done!')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
pre_save = '../predictresult/' + scene + '/' + modelname + '/' + ceshi
try:
    path_loss = pre_save + '/loss.jpg'
    plt.savefig(path_loss)
except:
    os.mkdir(pre_save)
    path_loss = pre_save + '/loss.jpg'
    plt.savefig(path_loss)
plt.show()
# print(val_demand_rmse)
# print(val_supply_rmse)

# Evaluate model on the test data
#eval_results = model.evaluate(graph, y_test,
#                              sample_weight=test_mask,
#                              batch_size=A.shape[0])

eval_results = model.evaluate(X_testinput,
                              Y_test_splice_normalized,batch_size=2)
print('Metrics:', model.metrics_names)
print(eval_results)
p = model.predict(X_testinput)
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
print('ddddd')




