import torch
import torch.nn as nn

class matcher(nn.module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.seed_top_k=config.seed_top_k #topK的数量
        self.conf_bar=config.conf_bar #[1, 0.1]置信区间
        self.seed_radius_coe=config.seed_radius_coe #NMS半径超参数
        self.use_score_encoding=config.use_score_encoding #
        self.detach_iter=config.detach_iter #分离的迭代次数
        self.seedlayer=config.seedlayer # 种子层数
        self.layer_num=config.layer_num #网络层数
        self.sink_iter=config.sink_iter #sinkhorn算法迭代次数

        # 坐标位置编码 
        self.position_encoder = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1) if config.use_score_encoding else nn.Conv1d(2, 32, kernel_size=1),
                                              nn.SyncBatchNorm(32),nn.ReLU(),
                                              nn.Conv1d(32, 64, kernel_size=1), nn.SyncBatchNorm(64), nn.ReLU(),
                                              nn.Conv1d(64, 128, kernel_size=1), nn.SyncBatchNorm(128), nn.ReLU(),
                                              nn.Conv1d(128, 256, kernel_size=1), nn.SyncBatchNorm(256), nn.ReLU(),
                                              nn.Conv1d(256, config.net_channels, kernel_size=1))