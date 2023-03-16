import torch
import torch.nn as nn

#种子模块
def seeding(nn_index1,nn_index2,x1,x2,topk,match_score,confbar,nms_radius,use_mc=True,test=False):

    if use_mc: #检查是否有非相互匹配
        mask_not_mutual=nn_index2.gather(dim=-1,index=nn_index1)!=torch.arange(nn_index1.shape[1],device='cuda')
        match_score[mask_not_mutual]=-1
    # 非极大值抑制算法
    pos_dismat1=((x1.norm(p=2,dim=-1)**2).unsqueeze_(-1)+(x1.norm(p=2,dim=-1)**2).unsqueeze_(-2)-2*(x1@x1.transpose(1,2))).abs_().sqrt_() #abs_()和abs()的区别在于abs_()会在本地创建一个张量，改变张量本身的值
    x2=x2.gather(index=nn_index1.unsqueeze(-1).expand(-1,-1,2),dim=1) #获取x2在nn_index1索引中的值
    pos_dismat2=((x2.norm(p=2,dim=-1)**2).unsqueeze_(-1)+(x2.norm(p=2,dim=-1)**2).unsqueeze_(-2)-2*(x2@x2.transpose(1,2))).abs_().sqrt_()
    radius1, radius2=nms_radius*pos_dismat1.mean(dim=(1,2),keepdim=True), nms_radius*pos_dismat2.mean(dim=(1,2),keepdim=True)
    nms_mask=(pos_dismat1>=radius1) & (pos_dismat2>=radius2)
    mask_not_local_max=(match_score.unsqueeze(-1)>=match_score.unsqueeze(-2)) | nms_mask
    mask_not_local_max=~(mask_not_local_max.min(dim=-1).values)
    match_score[mask_not_local_max] = -1

    match_score[match_score<confbar] = -1 #置信度小于设定值的赋-1
    mask_survive=match_score>0
    if test:
        topk=min(mask_survive.sum(dim=1)[0]+2,topk)
    _,topindex=torch.topk(match_score,topk,dim=-1) #b*k
    seed_index1,seed_index2=topindex,nn_index1.gather(index=topindex,dim=-1)
    return seed_index1,seed_index2,pos_dismat1,pos_dismat2


#领域聚合
def domain(seed_index1,seed_index2,pos_dismat1,pos_dismat2,domain_topk,domain_radiues,desc1,desc2):
    seed_dismat1, seed_dismat2=pos_dismat1.gather(dim=-1,index=seed_index1.unsqueeze(1).expand(-1,1000,-1)),\
                                     pos_dismat2.gather(dim=-1,index=seed_index2.unsqueeze(1).expand(-1,1000,-1))
    values1, indices1 = torch.topk(seed_dismat1,k=domain_topk,largest=False,dim=1,sorted=True)
    values2, indices2 = torch.topk(seed_dismat2,k=domain_topk,largest=False,dim=1,sorted=True)
    dis1_radiues,dis2_radiues=torch.max(seed_dismat1)*domain_radiues,torch.max(seed_dismat2)*domain_radiues

    values1_nearest, values2_nearest=values1[:,1,:].unsqueeze(1).expand(-1,10,-1), values2[:,1,:].unsqueeze(1).expand(-1,10,-1)
    indices1_nearest, indices2_nearest=indices1[:,1,:].unsqueeze(1).expand(-1,10,-1), indices2[:,1,:].unsqueeze(1).expand(-1,10,-1)
    
    values1[values1>dis1_radiues]=values1_nearest[values1>dis1_radiues]
    values2[values2>dis2_radiues]=values2_nearest[values2>dis2_radiues]
    indices1[values1>dis1_radiues]=indices1_nearest[values1>dis1_radiues]
    indices2[values2>dis2_radiues]=indices2_nearest[values2>dis2_radiues]
        
    desc1_domain=torch.tensor([],device='cuda')
    for i in range(indices1.shape[-1]):
        domain_desc1=desc1.gather(dim=-1, index=indices1[:,:,i].squeeze(-1).unsqueeze(1).expand(-1,128,-1))
        domain_desc1=domain_desc1.unsqueeze(-1)
        desc1_domain=torch.cat((desc1_domain,domain_desc1),dim=-1)
    return desc1_domain
        
            
    
    
    


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
    
    
class PointCN(nn.Module):
    def __init__(self, channels,out_channels):
        nn.Module.__init__(self)
        self.shot_cut = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(),
            nn.Conv1d(channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x) + self.shot_cut(x)


    
class hybrid_block(nn.module):
    def __init__(self, channel, head):
        nn.Module.__init__(self)
        self.head = head
        self.channel = channel
        self.attention_block_down = attention_propagantion(channel, head)
        self.cluster_filter = nn.Sequential(nn.Conv1d(2*channel,2*channel, kernel_size=1), nn.SyncBatchNorm(2*channel), nn.ReLU(),
                                            nn.Conv1d(2*channel,2*channel, kernel_size=1))
        self.cross_filter=attention_propagantion(channel, head)
        self.confidence_filter = PointCN(2*channel,1)
        self.attention_block_self=attention_propagantion(channel,head)
        self.attention_block_up=attention_propagantion(channel,head)

    def forward(self,desc1,desc2,seed_index1,seed_index2):
        '''
        seed_index1:   16*128
        desc1:         16*128*1000
        '''
        cluster1, cluster2 = desc1.gather(dim=-1, index=seed_index1.unsqueeze(1).expand(-1, self.channel, -1)), \
                             desc2.gather(dim=-1, index=seed_index2.unsqueeze(1).expand(-1, self.channel, -1)) #根据种子序列取得对应特征值
        cluster1, cluster2 = self.attention_block_down(cluster1, desc1), self.attention_block_down(cluster2, desc2)
        concate_cluster = self.cluster_filter(torch.cat([cluster1, cluster2], dim=1))

        cluster1, cluster2 = self.cross_filter(concate_cluster[:, :self.channel], concate_cluster[:, self.channel:]), \
                             self.cross_filter(concate_cluster[:, self.channel:], concate_cluster[:, :self.channel])
        cluster1, cluster2 = self.attention_block_self(cluster1,cluster1), self.attention_block_self(cluster2, cluster2)

        seed_weight=self.confidence_filter(torch.cat([cluster1,cluster2],dim=1))
        seed_weight=torch.sigmoid(seed_weight).squeeze(1)
        desc1_new, desc2_new=self.attention_block_up(desc1,cluster1,seed_weight), self.attention_block_up(desc2, cluster2,seed_weight)
        return desc1_new,desc2_new,seed_weight

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
        
        self.hybrid_block=nn.Sequential(*[hybrid_block(config.net_channels, config.head) for _ in range(config.layer_num)])
        self.final_project=nn.Conv1d(config.net_channels, config.net_channels, kernel_size=1)
        self.dustbin=nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

        #reseeding
        if len(config.seedlayer)!=1:
            self.mid_ditbin=nn.ParameterDict({str(i):nn.Parameter(torch.tensor(2,dtype=torch.float32)) for i in config.seedlayer[1:]})
            self.mid_final_project=nn.Conv1d(config.net_channels, config.net_channels, kernel_size=1)
        
    def forward(self,data,test_mode=True):
        x1, x2, desc1, desc2 = data['x1'][:,:,:2], data['x2'][:,:,:2], data['desc1'], data['desc2']
        desc1, desc2=torch.nn.functional.normalize(desc1,dim=-1), torch.nn.functional.normalize(desc2,dim=-1) #对描述子特征在最后一维进行L2范数归一化
        if test_mode:
            encode_x1,encode_x2=data['x1'],data['x2']
        else:
            encode_x1,encode_x2=data['aug_x1'],data['aug_x2']

        desc_dismat=(2-2*torch.matmul(desc1,desc2.transpose(1,2))).sqrt() #计算两幅图特征点特征之间的距离
        values, nn_index=torch.topk(desc_dismat,k=2,largest=False,dim=-1,sorted=True) #取出图B中分别与图A中最近的2个点索引和距离值
        nn_index2=torch.min(desc_dismat,dim=-1).indices.squeeze(1) #取出desc_dismat中沿最后一维最小值的索引
        inverse_ratio_score, nn_index1=values[:,:,1]/values[:,:,0], nn_index[:,:,0]

        seed_index1, seed_index2, pos_dismat1, pos_dismat2=seeding(nn_index1,nn_index2,x1,x2,self.seed_top_k[0],inverse_ratio_score,
                                                                   self.conf_bar[0],self.seed_radius_coe,test=test_mode) #获取种子索引
        
        # 位置编码
        desc1,desc2=desc1.transpose(1,2), desc2.transpose(1,2) #交换channel和feature of number，遵循torch默认channel在前
        if not self.use_score_encoding:
            encode_x1,encode_x2 = encode_x1[:,:,:2], encode_x2[:,:,:2]
        encode_x1,encode_x2 = encode_x1.transpose(1,2), encode_x2.transpose(1,2)
        x1_pos_embedding, x2_pos_embedding =self.position_encoder(encode_x1), self.position_encoder(encode_x2) 

            