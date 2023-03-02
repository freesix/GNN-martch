import argparse 

def str2bool(v):
    return v.lower() in ('true', "1")

arg_list = []
parser = argparse.ArgumentParser()

#将一些参数加入相应的组别
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg

#网络的一些参数
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--model_name", type=str, default='SGM', help="训练模型名称"
)
net_arg.add_argument(
    "--config_path", type=str, default='/home/freesix/SGM-1/train/configs/sgm.yaml', help="模型配置文件路径"
)


#数据的一些参数
data_arg = add_argument_group("Data")
data_arg.add_argument("--rawdata_path", type=str, default='/mnt/h/0/code/codeData/GL3D-2/data/', help="path for rawdata")
data_arg.add_argument("--dataset_path", type=str, default='/mnt/h/0/code/codeData/GL3D-2/dataset_dump_dir/', help="path for dataset")
data_arg.add_argument("--desc_path", type=str, default='/mnt/h/0/code/codeData/GL3D-2/feature_dump_dir/', help="path for descriptor dir")
data_arg.add_argument("--num_kpt", type=int, default=1000, help="用于训练的特征点数量")
data_arg.add_argument("--input_normalize", type=str, default='img', help="输入特征的normalize方式，img or intrinsic")
data_arg.add_argument("--data_aug", type=str2bool, default=True, help="是否使用特征点相关单应性增强")
data_arg.add_argument("--desc_suffix", type=str, default='_root_1000.hdf5', help="特征文件后缀")


#损失计算相关的一些参数
loss_arg = add_argument_group("loss")
loss_arg.add_argument("--momentum", type=float, default=0.9, help="torch损失函数中的冲量")
loss_arg.add_argument("--seed_loss_weight", type=float, default=250, help="之心损失的权重for sgm，一个超参数")
loss_arg.add_argument("--inlier_th", type=float, default=5e-3, help="内在关系的阈值")


#训练相关参数
train_arg = add_argument_group("Train")
train_arg.add_argument("--train-lr", type=float, default=1e-4, help="学习率")
train_arg.add_argument("--train_batch_size", type=int, default=16, help="batch size")
train_arg.add_argument("--gpu_id", type=str, default='0', help="CUDA_VISIBLE_DEVICES的id")
train_arg.add_argument("--train_iter", type=int, default=1000000, help="训练轮次")
train_arg.add_argument("--log_base", type=str, default='./log/', help="log的路径")
train_arg.add_argument("--val_intv", type=int, default=20000, help="验证的间隔")
train_arg.add_argument("--save_intv", type=int, default=1000, help="总结的间隔")
train_arg.add_argument("--log_intv", type=int, default=100, help="log的间隔")
train_arg.add_argument("--decay_rate", type=float, default=0.999996, help="学习率衰减率")
train_arg.add_argument("--decay_iter", type=float, default=300000, help="学习率衰减轮次")
train_arg.add_argument("--local_rank", type=int, default=0, help="分布式计算的局部序列")
train_arg.add_argument("--train_vis_floder", type=str, default='.', help="训练期间可视化目录")



#可视化相关
vis_arg = add_argument_group('Visualization')
vis_arg.add_argument("--tqdm_width", type=int, default=79, help="进度条的宽度")

#获取配置
def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
#打印可能传入的未定义参数
def print_unage():
    parser.print_usage()