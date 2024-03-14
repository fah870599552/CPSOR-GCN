"""
tf1.4的环境运行
算每个点的RMSE，MAE,ADE,FDE
"""
import numpy as np
import math
import pickle
import numpy as np
n_feature_out =2

def calculate_RMSE(predicted_trajectories, true_trajectories):
    errors = predicted_trajectories - true_trajectories
    squared_errors = np.square(errors)
    mean_squared_error = np.mean(squared_errors)
    RMSE = np.sqrt(mean_squared_error)
    return RMSE


def calculate_MAE(predicted_trajectories, true_trajectories):
    errors = np.abs(predicted_trajectories - true_trajectories)
    MAE = np.mean(errors)
    return MAE


def calculate_ADE(predicted_trajectories, true_trajectories):
    pred_list = list(predicted_trajectories)
    true_list = list(true_trajectories)
    RMSE_t = []
    for pred, true in zip(pred_list, true_list):
        pred = np.array([pred])
        true = np.array([true])
        RMSE_temp = calculate_RMSE(pred, true)
        RMSE_t.append(RMSE_temp)
    RMSE_t = np.array(RMSE_t)
    ADE = np.mean(RMSE_t)
    return ADE


def calculate_FDE(predicted_trajectories, true_trajectories):
    pred_list = list(predicted_trajectories)
    true_list = list(true_trajectories)
    RMSE_t = []
    for pred, true in zip(pred_list, true_list):
        pred = np.array([pred])
        true = np.array([true])
        RMSE_temp = calculate_RMSE(pred, true)
        RMSE_t.append(RMSE_temp)
    RMSE_t = np.array(RMSE_t)
    FDE = RMSE_t[-1]
    return FDE


# 示例数据
# predicted_trajectories = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
#                                    [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])
# true_trajectories = np.array([[1.5, 2.5], [2.5, 3.5], [3.5, 4.5],
#                               [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])


# 计算RMSE
def zhibiao(predicted_trajectories, true_trajectories):
    RMSE = calculate_RMSE(predicted_trajectories, true_trajectories)

    # 计算MAE
    MAE = calculate_MAE(predicted_trajectories, true_trajectories)

    # 计算ADE
    ADE = calculate_ADE(predicted_trajectories, true_trajectories)

    # 计算FDE
    FDE = calculate_FDE(predicted_trajectories, true_trajectories)

    print("RMSE:", RMSE)
    print("MAE:", MAE)
    print("ADE:", ADE)
    print("FDE:", FDE)
    return [RMSE, MAE, ADE, FDE]


#导入P值以及真实值Y_train_splice_normalized
def main():
    scene = 'brake'
    ceshi = 'cpsorgcn-3s-1s'
    modelname = 'cpsorgcn0918'
    pre_save = '../predictresult/' + scene + '/' + modelname + '/' + ceshi
    path_pre = pre_save + '/y_pre.npz'
    path_real = pre_save + '/y_real.npz'
    data = np.load(path_pre)
    # 通过数组名称获取数组
    p = data['arr']
    dim1,dim2 = p.shape[:2]
    data2 = np.load(path_real)
    Y_train_splice_normalized = data2['arr']
    #inverse的操作要改一下
    scaler_xy_save = '../data/scaler/' + scene + '_' + 'scaler_xy.pickle'
    with open(scaler_xy_save, 'rb') as f:
        Y_sc = pickle.load(f)
    result_arr = np.reshape(p,(-1,n_feature_out))
    p  = Y_sc.inverse_transform(result_arr)
    p = np.reshape(p, (dim1,dim2,1,n_feature_out))
    Y_train_splice_normalized = np.reshape(Y_train_splice_normalized,(-1,n_feature_out))
    r  = Y_sc.inverse_transform(Y_train_splice_normalized)
    r = np.reshape(r,(dim1,dim2,1,n_feature_out))
    tester_pred = list(p[:,:,0,:])
    # npc_pred = list(p[:,:,1,:])
    tester_real = list(r[:,:,0,:])
    # npc_real = list(r[:,:,1,:])
    tester_zhibiao = []
    # npc_zhibiao = []
    for predicted_trajectories, true_trajectories in zip(tester_pred,tester_real):
        tester_temp = zhibiao(predicted_trajectories, true_trajectories)
        tester_zhibiao.append(tester_temp)

    # for predicted_trajectories, true_trajectories in zip(npc_pred,npc_real):
    #     npc_temp = zhibiao(predicted_trajectories, true_trajectories)
    #     npc_zhibiao.append(npc_temp)

    tester = np.array(tester_zhibiao)
    # npc = np.array(npc_zhibiao)
    tester_result = np.zeros((4,1))
    # npc_result = np.zeros((4,1))
    for i in range(4):
        tester_result[i] = np.mean(tester[:, i])
        # npc_result[i] = np.mean(npc[:, i])
    print(tester_result)
    # print(npc_result)


# rmse_zhuche_v = RMSE(q,p)
# mae_zhuche_v = MAE(q,p)
# print(rmse_zhuche_v,mae_zhuche_v)
if __name__ =="__main__":
    main()