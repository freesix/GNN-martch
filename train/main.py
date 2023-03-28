import yaml
from collections import namedtuple
from config import get_config, print_unage
from dataset import Offline_Dataset
import os
import torch
import torch.utils.data 
import torch.distributed as dist
from sgmnet.match_model import matcher as SGM_Modle
from train import train
# torch.backends.cudnn.enable=True
# torch.backends.cudnn.benchmark=True

def main(config, model_config):
    if config.model_name=='SGM': #选择模型(如果有多个模型)
        model = SGM_Modle(model_config)
    else:
        raise NotImplementedError

        
    #初始化多线程计算
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    torch.cuda.set_device(config.local_rank)
    device = torch.device(f'cuda:{config.local_rank}')
    #将模型载入cuda并配置多进程设置
    model.to(device)

    #初始化多线程计算组
    dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=int(os.environ['WORLD_SIZE']) if 
                            'WORLD_SIZE' in os.environ else 1)    
    # 这里的find_unused_parameters参数是因为网络层中有些参数未参与反向传播，后续设计将其考虑进去
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],find_unused_parameters=True)
    

    if config.local_rank==0:
        os.system('nvidia-smi')
    #dataloader
    train_dataset = Offline_Dataset(config, 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size//torch.distributed.get_world_size(),
                    num_workers=8//dist.get_world_size(), pin_memory=False, sampler=train_sampler, collate_fn=train_dataset.collate_fn)

    valid_dataset = Offline_Dataset(config, 'valid')
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.train_batch_size,
                    num_workers=8//dist.get_world_size(), pin_memory=False,collate_fn=valid_dataset.collate_fn,sampler=valid_sampler)
    
    if config.local_rank==0:
        print('开始训练')
        train(model,train_loader,valid_loader,config,model_config)
    # print(train_loader)

if __name__ == '__main__':

    config, unparsed = get_config()
    with open(config.config_path) as f:
        model_config = yaml.safe_load(f)
    model_config=namedtuple('model_config',model_config.keys())(*model_config.values())
    if len(unparsed) > 0:
        print_unage()
        exit(1)

    main(config, model_config)