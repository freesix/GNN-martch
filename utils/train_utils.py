import torch 
import torch.distributed as dist 
import numpy as np 
import cv2 

#分析匹配对序列
#分析的是pair_num.txt文件中信息
def parse_pair_seq(pair_num_list):

    pair_num=int(pair_num_list[0,1]) 
    pair_num_list=pair_num_list[1:] 
    pair_seq_list=[]
    cursor=0
    accu_pair_num={}

    for line in pair_num_list:
        seq, seq_pair_num=line[0], int(line[1])
        for _ in range(seq_pair_num):
            pair_seq_list.append(seq)
        accu_pair_num[seq]=cursor
        cursor+=seq_pair_num

    assert pair_num==cursor 
    return pair_seq_list, accu_pair_num

#将数据tensor类型数据加载进cuda
def tocuda(data):
    for key in data.keys():
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda()
    return data 


def reduce_tensor(tensor,op='mean'):
    rt = tensor.detach() #分离tensor，获得一个和原tensor一样但没有梯度信息的tensor
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) #多进程计算的归约操作，在所有进程之间共享和同步张量的值，以实现全局梯度的计算，他能够将所有进程中的同一张量进行操作，如求和等，并广播给所有进程。
    if op=='mean':
        rt/=dist.get_world_size() #进程数量
    return rt


def get_rnd_homography(batch_size, pert_ratio=0.25):
    corners = np.array([[-1,1], [1,1], [-1,-1], [1,-1]], dtype=np.float32)
    homo_tower=[]
    for _ in range(batch_size):
        rnd_pert = np.random.uniform(-2*pert_ratio, 2*pert_ratio, (4,2)).astype(np.float32) #产生一个指定范围内指定大小均匀分布数据
        print(rnd_pert)
        pert_corners = corners + rnd_pert
        M = cv2.getPerspectiveTransform(corners, pert_corners) #计算透视变换矩阵，输入参数分别为源和目标图像的四个点,返回一个3x3的变换矩阵
        homo_tower.append(M)
    homo_tower = np.stack(homo_tower, axis=0)
    return homo_tower 

