import yaml
from config import get_config, print_unage
from dataset import Offline_Dataset
import os
import torch
import torch.utils.data 
import torch.distributed as dist



def main(config, model_config):
    # if config.modle_name=='SGM': #选择模型(如果有多个模型)
        # model_config = SGM_Modle(model_config)
    #初始化多线程计算
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    torch.cuda.set_device(config.local_rank)
    device = torch.device(f'cuda:{config.local_rank}')
    #将模型载入cuda并配置多进程设置

    #初始化多线程计算组
    dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=int(os.environ['WORLD_SIZE']) if 
                            'WORLD_SIZE' in os.environ else 1)    
    if config.local_rank==0:
        os.system('nvidia-smi')
    #dataloader
    train_dataset = Offline_Dataset(config, 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size//torch.distributed.get_world_size(),
                                               num_workers=8//dist.get_world_size(), pin_memory=False, sampler=train_sampler, collate_fn=train_dataset.collate_fn)
    
    print(train_loader)

if __name__ == '__main__':

    config, unparsed = get_config()
    with open(config.config_path) as f:
        model_config = yaml.safe_load(f)
        print(model_config)
    
    if len(unparsed) > 0:
        print_unage()
        exit(1)

    main(config, model_config)