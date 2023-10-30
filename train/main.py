import yaml
from collections import namedtuple
from config import get_config, print_unage
from dataset import Offline_Dataset
import os
import torch
import torch.utils.data 
import torch.distributed as dist
import torch.multiprocessing as mp
from sgmnet.match_model import matcher as SGM_Modle
from train import train
from distributed_utils import is_main_process,setup_for_distributed #修改分布式下的print函数
# torch.backends.cudnn.enable=True
# torch.backends.cudnn.benchmark=True

def main(local_rank, ngpus_per_node, config):
    with open(config.config_path) as f:
        model_config = yaml.safe_load(f) # 读取模型配置文件
    model_config=namedtuple('model_config',model_config.keys())(*model_config.values()) # 将配置文件转换为命名元组
    
    if config.model_name=='SGM': #选择模型(如果有多个模型)
        model = SGM_Modle(model_config)
    else:
        raise NotImplementedError
    # 多GPU训练 loacl_rank为每个进程的GPU编号 rank为每个进程的编号
    config.local_rank = local_rank
    config.rank = config.node_rank * ngpus_per_node + local_rank
        
    #初始化多线程计算
    os.environ['MASTER_ADDR'] = 'localhost' #设置主机名
    os.environ['MASTER_PORT'] = '5678' #设置端口
    dist.init_process_group(backend='nccl', init_method='env://', rank=config.rank, world_size=config.world_size)   
    dist.barrier() #同步所有进程
    # setup_for_distributed(config.rank == 0)
    # torch.cuda.set_device(config.local_rank)
    device = torch.device("cuda:{}".format(config.local_rank))
    #将模型载入cuda并配置多进程设置
    model.to(device)

    #初始化多线程计算组
     
    # 这里的find_unused_parameters参数是因为网络层中有些参数未参与反向传播，后续设计将其考虑进去
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],output_device=config.local_rank, find_unused_parameters=True)
    if is_main_process():
        os.system('nvidia-smi')
    #dataloader 载入数据
    train_dataset = Offline_Dataset(config, 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler,batch_size=config.train_batch_size//torch.distributed.get_world_size(),drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,pin_memory=False, batch_sampler=train_batch_sampler, collate_fn=train_dataset.collate_fn)

    valid_dataset = Offline_Dataset(config, 'valid')
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.train_batch_size//torch.distributed.get_world_size(),
                                                pin_memory=False,collate_fn=valid_dataset.collate_fn,sampler=valid_sampler)
    
    if is_main_process():
        print('开始训练')
    train(model,train_loader,valid_loader,config,model_config,train_sampler)
    # print(train_loader)

if __name__ == '__main__':

    config, unparsed = get_config()
    # with open(config.config_path) as f:
    #     model_config = yaml.safe_load(f)
    # model_config=namedtuple('model_config',model_config.keys())(*model_config.values())
    if len(unparsed) > 0:
        print_unage()
        exit(1)
    config.world_size = config.ngpus_per_node*config.nodes # 设置多进程的进程数(每个节点的GPU数*节点数)
    # 多GPU训练
    mp.spawn(main,nprocs=config.ngpus_per_node,args=(config.ngpus_per_node,config))