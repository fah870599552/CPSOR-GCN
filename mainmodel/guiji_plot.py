
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
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
pinghua = 7
ceshi = '25_14_0_1'
# ceshi1 = '5_17_0_f_1'
# ceshi2 = '5_17_0_a_1'
modelname = 'cpgcn0918'
modelname_save = 'cpgcn0918'
ceshimodel = 'cpsorgcn-1s-1s'
# modelname_save = 'cpsorgcn0918'
# ceshimodel = 'cpsorgcn-3s-1s'
"""
几个重要路径汇总
测试数据的load路径：'../data/traindata/'+ scene +'/'+ modelname + '/'+ ceshi +'/Y_test_splice_normalized.npy'
model文件的load路径：'../modelresult/' + scene + '/' + modelname_save
                    model_savepath + '/' + ceshimodel + '.h5'
real and pre 储存路径：'../predictresult/' + scene + '/' + modelname + '/' + ceshi
scaler文件的load路径：
"""
#################################加载预测结果的数据文件
##############################################数据加载#####################################################
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
#################################加载pq数组
pre_save1 = '../predictresult/' + scene + '/' + modelname + '/' + ceshi
path_pre = pre_save1 + '/y_pre.npz'
path_real = pre_save1 + '/y_real.npz'
p_data = np.load(path_pre)
r_data = np.load(path_real)
p = p_data[list(p_data.keys())[0]]
r = r_data[list(r_data.keys())[0]]
#
# pre_save1 = '../predictresult/' + scene + '/' + modelname + '/' + ceshi1
# path_pre1 = pre_save1 + '/y_pre.npz'
# p_data1 = np.load(path_pre1)
# p_1 = p_data1[list(p_data1.keys())[0]]
#
# pre_save2 = '../predictresult/' + scene + '/' + modelname + '/' + ceshi2
# path_pre2 = pre_save2 + '/y_pre.npz'
# p_data2 = np.load(path_pre2)
# p_2 = p_data2[list(p_data2.keys())[0]]
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

# result_arr1 = np.reshape(p_1,(-1,n_feature_out))
# p_1  = scaler_xy.inverse_transform(result_arr1)
# p_1 = np.reshape(p_1, (dim1,dim2,1,n_feature_out))
#
# result_arr2 = np.reshape(p_2,(-1,n_feature_out))
# p_2  = scaler_xy.inverse_transform(result_arr2)
# p_2 = np.reshape(p_2, (dim1,dim2,1,n_feature_out))
dimx1,dimx2,dimx3 = Xtestphy_splice_normalized.shape[:3]
Xtestphy_splice_normalized = np.reshape(Xtestphy_splice_normalized,(-1,n_feature_out))
Xtestphy_splice_normalized  = scaler_xy.inverse_transform(Xtestphy_splice_normalized)
Xtestphy_splice_normalized = np.reshape(Xtestphy_splice_normalized,(dimx1,dimx2,dimx3,n_feature_out+2))
#################################绘制轨迹图
s = 1
for i in range(p.shape[0]):

    if i%pinghua == 0 :
        x1 = p[i][-1][0][0]
        y1 = p[i][-1][0][1]
        if i == 0:
            x_y1 = np.array([x1, y1])
        else:
            x_y1 = np.vstack((x_y1, [x1, y1]))

df_pre = pd.DataFrame(x_y1, columns=['x', 'y'])

# for i in range(p_1.shape[0]):
#
#     if i%pinghua == 0 :
#         x11 = p_1[i][-1][0][0]
#         y11 = p_1[i][-1][0][1]
#         if i == 0:
#             x_y11 = np.array([x11, y11])
#         else:
#             x_y11 = np.vstack((x_y11, [x11, y11]))
#
# df_pre1 = pd.DataFrame(x_y11, columns=['x', 'y'])
#
# for i in range(p_2.shape[0]):
# # for i in range(100):
#     if i%pinghua == 0 :
#         x12 = p_2[i][-1][0][0]
#         y12 = p_2[i][-1][0][1]
#         if i == 0:
#             x_y12 = np.array([x12, y12])
#         else:
#             x_y12 = np.vstack((x_y12, [x12, y12]))
#
# df_pre2 = pd.DataFrame(x_y12, columns=['x', 'y'])


for i in range(r.shape[0]):
# for i in range(100):
    if i%pinghua==0 :
        x3 = r[i][-1][0][0]
        y3 = r[i][-1][0][1]
        if i == 0:
            x_y3 = np.array([x3, y3])
        else:
            x_y3 = np.vstack((x_y3, [x3, y3]))

df_real = pd.DataFrame(x_y3, columns=['x', 'y'])

for i in range(Xtestphy_splice_normalized.shape[0]):
# for i in range(100):
    if i%pinghua ==0 :
        x4 = Xtestphy_splice_normalized[i][-1][1][0]
        y4 = Xtestphy_splice_normalized[i][-1][1][1]
        if i == 0:
            x_y4 = np.array([x4, y4])
        else:
            x_y4 = np.vstack((x_y4, [x4, y4]))
df_x = pd.DataFrame(x_y4, columns=['x', 'y'])

#绘制x、y的折线图
def html_color_to_rgb(html_color):
    html_color = html_color.lstrip('#')
    return tuple(int(html_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
color_pred2 = html_color_to_rgb('FA4224')
color_pred1 = html_color_to_rgb('82B0D2')
color_pred = html_color_to_rgb('FFB100')
color_real = html_color_to_rgb('214151')
color_npc = html_color_to_rgb('A2D0C1')
linewidth = 3
markersize = 9
plt.figure(figsize=(9, 6))
plt.rc('font', family='Times New Roman', size=19)
plt.plot(df_real['x'], df_real['y'], marker='o', color=color_real, linestyle='-', linewidth=linewidth, alpha=0.6,
         markeredgewidth=0,markeredgecolor=(color_real[0],color_real[1],color_real[2], 0.1),markersize=markersize, label='real')
# Plot subject-predict with transparent lines and empty circles
plt.plot(df_pre['x'], df_pre['y'], marker='o', color=color_pred, linestyle='-', linewidth=linewidth, alpha=0.6,
         markeredgewidth=0,markeredgecolor=(color_pred[0],color_pred[1],color_pred[2], 0.1), markersize=markersize,label='prediction')
# plt.plot(df_pre1['x'], df_pre1['y'], marker='o', color=color_pred1, linestyle='-', linewidth=linewidth, alpha=0.6,
#          markeredgewidth=0,markeredgecolor=(color_pred1[0],color_pred1[1],color_pred1[2], 0.1), markersize=markersize,label='prediction')
# # Plot subject-real with transparent lines and empty circles
# plt.plot(df_pre2['x'], df_pre2['y'], marker='o', color=color_pred2, linestyle='-', linewidth=linewidth, alpha=0.6,
#          markeredgewidth=0,markeredgecolor=(color_pred2[0],color_pred2[1],color_pred2[2], 0.1), markersize=markersize,label='anger')
plt.xlabel('x')
plt.xlabel('x')
plt.ylabel('y')
# 24_5_1
# plt.xlim(-50, 580)
# plt.ylim(-0.5, 0.9)
#24_10_0
# plt.xlim(-30, 380)
# plt.ylim(-0.3, 0.7)
#24_1_2
# plt.xlim(-50, 530)
# plt.ylim(-0.25, 1.75)
# plt.xlim(-35, 5)
# plt.ylim(-12, 5)
plt.legend()
plt.legend(loc= 3)
figsave_path = '../predictresult/' + scene + '/' + modelname + '/' + ceshi + '_guiji0121.jpg'
plt.savefig(figsave_path,dpi = 800)
plt.show()