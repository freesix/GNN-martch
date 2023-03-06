import torch
import torch.nn as nn

#带权重的注意力传播
class attention_propagantion(nn.Module):
    def __init__(self,channel,head):
        self.head = head #注意力头数
        self.head_dim = channel//head #运用注意力多头机制后特征维度
        self.query_filter,self.key_filter,self.value_filter=nn.Conv1d(channel,channel,kernel_size=1),nn.Conv1d(channel,channel,kernel_size=1), \
                                                            nn.Conv1d(channel,channel,kernel_size=1)
        self.mh_filter=nn.Conv1d(channel,channel,kernel_size=1)
        self.cat_filter=nn.Sequential(nn.Conv1d(2*channel,2*channel,kernel_size=1), nn.SyncBatchNorm(2*channel), nn.ReLU(), 
                                      nn.Conv1d(2*channel, channel, kernel_size=1))
        
    def forward(self,desc1,desc2,weight_v=None):
        batch_size = desc1.shape[0]
        query,key,value=self.query_filter(desc1).view(batch_size,self.head,self.head_dim,-1), self.key_filter(desc2).view(batch_size,
                            self.head, self.head_dim,-1), self.value_filter(batch_size,self.head,self.head_dim,-1)
        if weight_v is not None:
            value = value*weight_v.view(batch_size,1,1,-1)
        score=torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query,key)/ self.head_dim **0.5, dim=-1)
        add_value = torch.einsum('bhnm,bhdm->bhdn',score,value).reshape(batch_size,self.head_dim*self.head,-1)
        add_value = self.nh_filter(add_value)
        desc1_new=desc1+self.cat_filter(torch.cat([desc1,add_value],dim=1))
        return desc1_new
    
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