import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
eps=2e-7 #防止sink_horn算法过程中分母为0的因子

def sinkhorn(M,r,c,iteration):
    p = torch.softmax(M, dim=-1)
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iteration):
        u = r / ((p * v.unsqueeze(-2)).sum(-1) + eps)
        v = c / ((p * u.unsqueeze(-1)).sum(-2) + eps)
    p = p * u.unsqueeze(-1) * v.unsqueeze(-2)
    return p

def sink_algorithm(M,dustbin,iteration):
    M = torch.cat([M, dustbin.expand([M.shape[0], M.shape[1], 1])], dim=-1)
    M = torch.cat([M, dustbin.expand([M.shape[0], 1, M.shape[2]])], dim=-2)
    r = torch.ones([M.shape[0], M.shape[1] - 1],device=("cuda:{}".format(dist.get_rank())))
    r = torch.cat([r, torch.ones([M.shape[0], 1],device=("cuda:{}".format(dist.get_rank()))) * M.shape[1]], dim=-1)
    c = torch.ones([M.shape[0], M.shape[2] - 1],device=("cuda:{}".format(dist.get_rank())))
    c = torch.cat([c, torch.ones([M.shape[0], 1],device=("cuda:{}".format(dist.get_rank()))) * M.shape[2]], dim=-1)
    

    p=sinkhorn(M,r,c,iteration)
    return p

#种子模块
def seeding(nn_index1,nn_index2,x1,x2,topk,match_score,confbar,nms_radius,use_mc=True,test=False):

    if use_mc: #检查是否有非相互匹配
        mask_not_mutual=nn_index2.gather(dim=-1,index=nn_index1)!=torch.arange(nn_index1.shape[1],device=("cuda:{}".format(dist.get_rank())))
        match_score[mask_not_mutual]=-1
        # test1=match_score[match_score==-1]
        # print(sum(test1))

    # 非极大值抑制算法
    pos_dismat1=((x1.norm(p=2,dim=-1)**2).unsqueeze_(-1)+(x1.norm(p=2,dim=-1)**2).unsqueeze_(-2)-2*(x1@x1.transpose(1,2))).abs_().sqrt_() #abs_()和abs()的区别在于abs_()会在本地创建一个张量，改变张量本身的值
    x2=x2.gather(index=nn_index1.unsqueeze(-1).expand(-1,-1,2),dim=1) #获取x2在nn_index1索引中的值
    pos_dismat2=((x2.norm(p=2,dim=-1)**2).unsqueeze_(-1)+(x2.norm(p=2,dim=-1)**2).unsqueeze_(-2)-2*(x2@x2.transpose(1,2))).abs_().sqrt_() #求取
    radius1, radius2=nms_radius*pos_dismat1.mean(dim=(1,2),keepdim=True), nms_radius*pos_dismat2.mean(dim=(1,2),keepdim=True)
    nms_mask=(pos_dismat1>=radius1) & (pos_dismat2>=radius2)
    mask_not_local_max=(match_score.unsqueeze(-1)>=match_score.unsqueeze(-2))|nms_mask
    mask_not_local_max=~(mask_not_local_max.min(dim=-1).values)
    match_score[mask_not_local_max] = -1 #对于非局部最大值舍弃

    match_score[match_score<confbar] = -1 #置信度小于设定值的赋-1
    mask_survive=match_score>0
    # if test:
    #     topk=min(mask_survive.sum(dim=1)[0]+2,topk) #筛选符合非极大值抑制的点
    _,topindex=torch.topk(match_score,topk,dim=-1) #b*k
    seed_index1,seed_index2=topindex,nn_index1.gather(index=topindex,dim=-1)
    return seed_index1,seed_index2,pos_dismat1,pos_dismat2


#领域聚合        
def domain(pos_dismat1,pos_dismat2,domain_topk,domain_radiues,desc1,desc2):
    # seed_dismat1, seed_dismat2=pos_dismat1.gather(dim=-1,index=seed_index1.unsqueeze(1).expand(-1,1000,-1)),\
    #                                  pos_dismat2.gather(dim=-1,index=seed_index2.unsqueeze(1).expand(-1,1000,-1))
    values1, indices1 = torch.topk(pos_dismat1,k=domain_topk,largest=False,dim=1,sorted=True) 
    values2, indices2 = torch.topk(pos_dismat2,k=domain_topk,largest=False,dim=1,sorted=True)
    dis1_radiues,dis2_radiues=torch.max(pos_dismat1)*domain_radiues,torch.max(pos_dismat2)*domain_radiues 
    # 离描述子最近一个邻近描述子的距离值和索引
    # values1_nearest, values2_nearest=values1[:,1,:].unsqueeze(1).expand(-1,10,-1), values2[:,1,:].unsqueeze(1).expand(-1,10,-1)
    indices1_nearest, indices2_nearest=indices1[:,1,:].unsqueeze(1).expand(-1,10,-1), indices2[:,1,:].unsqueeze(1).expand(-1,10,-1)
    # 将不在领域半径内的描述子替换成最邻近的那一个描述子
    # values1[values1>dis1_radiues]=values1_nearest[values1>dis1_radiues]
    # values2[values2>dis2_radiues]=values2_nearest[values2>dis2_radiues]
    indices1[values1>dis1_radiues]=indices1_nearest[values1>dis1_radiues]
    indices2[values2>dis2_radiues]=indices2_nearest[values2>dis2_radiues]

    desc1_domain=torch.tensor([],device=("cuda:{}".format(dist.get_rank())))
    for i in range(indices1.shape[-1]):
        domain_desc1=desc1.gather(dim=-1, index=indices1[:,:,i].squeeze(-1).unsqueeze(1).expand(-1,128,-1))
        domain_desc1=domain_desc1.unsqueeze(-1)
        desc1_domain=torch.cat([desc1_domain,domain_desc1],dim=-1)
    desc2_domain=torch.tensor([],device=("cuda:{}".format(dist.get_rank())))
    for i in range(indices1.shape[-1]):
        domain_desc2=desc2.gather(dim=-1, index=indices2[:,:,i].squeeze(-1).unsqueeze(1).expand(-1,128,-1))
        domain_desc2=domain_desc2.unsqueeze(-1)
        desc2_domain=torch.cat([desc2_domain,domain_desc2],dim=-1)
    return desc1_domain, desc2_domain            
    

# 领域注意力聚合
class domain_attention(nn.Module):
    def __init__(self, channel):
        nn.Module.__init__(self)
        self.channel=channel
        self.query_filter,self.key_filter,self.value_filter=nn.Conv2d(channel,channel,kernel_size=(1,1)),nn.Conv2d(channel,channel,kernel_size=(1,1)),\
                                                            nn.Conv2d(channel,channel,kernel_size=(1,1))
        self.nh_filter=nn.Conv2d(channel,channel,kernel_size=(1,1))
        self.cat_filter=nn.Sequential(nn.Conv2d(2*channel,2*channel,kernel_size=(1,1)), nn.SyncBatchNorm(2*channel), nn.ReLU(),
                                      nn.Conv2d(2*channel,channel,kernel_size=(1,1)))

    def forward(self,desc,weight_v=None):
        batch_size=desc.shape[0]
        query_desc=desc[:,:,0,:].unsqueeze(-2)
        query,key,value=self.query_filter(query_desc),self.key_filter(desc),self.value_filter(desc)
        if weight_v is not None:
            value=value*weight_v.view(batch_size,1,1,-1)
        socre=torch.softmax(torch.einsum('bctn,bckn->bctk',query,key),dim=-1)
        add_value=torch.einsum('bctk,bckn->bctn',socre,value)
        add_value=self.nh_filter(add_value)
        desc_new=query_desc+self.cat_filter(torch.cat([query_desc,add_value],dim=1))
        return desc_new.squeeze(-2)

class Separate_out_2(nn.Module):
    def __init__(self, channels, out_channels):
        nn.Module.__init__(self)
        self.shot_cut = nn.Conv2d(channels, out_channels ,kernel_size=(1,1))
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels,eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(),
            nn.Conv2d(channels,channels,kernel_size=(1,1)),
            nn.InstanceNorm2d(channels,eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(),
            nn.Conv2d(channels,out_channels,kernel_size=(1,1)) 
        )

    def forward(self,x):
        out = self.conv(x) + self.shot_cut(x)
        return out
    
class Separate_out(nn.Module):
    def __init__(self, channels, out_channels):
        nn.Module.__init__(self)
        self.shot_cut = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.SyncBatchNorm(channels), #分布式计算上的批标准化层
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(),
            nn.Conv1d(channels, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        evalu_score=self.conv(x)+self.shot_cut(x)
        # # evalu_score=torch.sigmoid(evalu_score).squeeze(1) #得出得分
        # values, indics = torch.topk(evalu_score,k=separate_num,dim=1,sorted=False)
        # evalu_score[evalu_score.gather(dim=-1,index=indics)]=0
        # evalu_score=torch.sigmoid(values)
        # separate_index1 = seed_index1.gater(dim=-1, index=indics)
        # separate_index2 = seed_index2.gaher(dim=-1, index=indics)
        # # new_desc1 = desc1.gather(dim=-1, index=separate_index1.unsqueeze(1).expand(-1,channel,-1))
        # # new_desc2 = desc2.gather(dim=-1, index=separate_index2.unsqueeze(1).expand(-1,channel,-1))
        # new_desc1 = desc1.gather(dim=-1,index=indics.unsqueeze(1).expand(-1,channel,-1))
        # new_desc2 = desc2.gather(dim=-1,index=indics.unsqueeze(1).expand(-1,channel,-1))
        return evalu_score
   

class attention_propagantion(nn.Module):

    def __init__(self,channel,head):
        nn.Module.__init__(self)
        self.head=head
        self.head_dim=channel//head
        self.query_filter,self.key_filter,self.value_filter=nn.Conv1d(channel,channel,kernel_size=1),nn.Conv1d(channel,channel,kernel_size=1),\
                                                            nn.Conv1d(channel,channel,kernel_size=1)
        self.mh_filter=nn.Conv1d(channel,channel,kernel_size=1)
        self.cat_filter=nn.Sequential(nn.Conv1d(2*channel,2*channel, kernel_size=1), nn.SyncBatchNorm(2*channel), nn.ReLU(),
                                      nn.Conv1d(2*channel, channel, kernel_size=1))

    def forward(self,desc1,desc2,weight_v=None):
        #desc1(q) attend to desc2(k,v)
        batch_size=desc1.shape[0]
        query,key,value=self.query_filter(desc1).view(batch_size,self.head,self.head_dim,-1),self.key_filter(desc2).view(batch_size,self.head,self.head_dim,-1),\
                        self.value_filter(desc2).view(batch_size,self.head,self.head_dim,-1)
        if weight_v is not None:
            value=value*weight_v.view(batch_size,1,1,-1)
        score=torch.softmax(torch.einsum('bhdn,bhdm->bhnm',query,key)/ self.head_dim ** 0.5,dim=-1)
        add_value=torch.einsum('bhnm,bhdm->bhdn',score,value).reshape(batch_size,self.head_dim*self.head,-1)
        add_value=self.mh_filter(add_value)
        desc1_new=desc1+self.cat_filter(torch.cat([desc1,add_value],dim=1))
        return desc1_new
    
    
#带权重的注意力传播
class attention_learn(nn.Module):
    def __init__(self,channel):
        nn.Module.__init__(self)
        self.channel=channel
        self.query_filter,self.key_filter,self.value_filter=nn.Conv1d(channel,channel,kernel_size=1),nn.Conv1d(channel,channel,kernel_size=1), \
                                                            nn.Conv1d(channel,channel,kernel_size=1)
        self.mh_filter=nn.Conv1d(channel,channel,kernel_size=1)
        self.cat_filter=nn.Sequential(nn.Conv1d(2*channel,2*channel,kernel_size=1), nn.SyncBatchNorm(2*channel), nn.ReLU(), 
                                      nn.Conv1d(2*channel, channel, kernel_size=1))
        self.separate=Separate_out_2(2*channel,1)
        
    def forward(self,desc1,desc2):
        batch_size = desc1.shape[0]
        query,key,value=self.query_filter(desc1).unsqueeze(-1).expand(batch_size,self.channel,-1,desc2.shape[-1]),\
                        self.key_filter(desc2).unsqueeze(-2).expand(batch_size,self.channel,desc1.shape[-1],-1),\
                        self.value_filter(desc2)
        score=self.separate(torch.cat([query,key],dim=1)).squeeze(1)
        score=torch.softmax(score,dim=-1)
        add_value = torch.einsum('bnm,bdm->bdn',score,value)
        add_value = self.mh_filter(add_value)
        desc1_new=desc1+self.cat_filter(torch.cat([desc1,add_value],dim=1))
        return desc1_new,score



    
                
    
class hybrid_block(nn.Module):
    def __init__(self, channel, head):
        nn.Module.__init__(self)
        self.head = head
        self.channel = channel
        self.attention_block_down = attention_propagantion(channel, head)
        self.cluster_filter = nn.Sequential(nn.Conv1d(2*channel,2*channel, kernel_size=1), nn.SyncBatchNorm(2*channel), nn.ReLU(),
                                            nn.Conv1d(2*channel,2*channel, kernel_size=1))
        self.cross_filter=attention_propagantion(channel, head)
        self.attention_block_self=attention_propagantion(channel,head)
        self.attention_block_up=attention_propagantion(channel,head)
        self.attention_learn = attention_learn(channel)

    def forward(self,desc1,desc2,seed_index1,seed_index2):
        '''
        seed_index1:   16*128
        desc1:         16*128*1000
        '''
        cluster1, cluster2 = desc1.gather(dim=-1, index=seed_index1.unsqueeze(1).expand(-1, self.channel, -1)), \
                             desc2.gather(dim=-1, index=seed_index2.unsqueeze(1).expand(-1, self.channel, -1)) #根据种子序列取得对应特征值
        cluster1, cluster2 = self.attention_block_down(cluster1, desc1), self.attention_block_down(cluster2, desc2) #将所有特征值聚合到种子特征上

        concate_cluster=self.cluster_filter(torch.cat([cluster1,cluster2],dim=1))
        #filtering
        cluster1,cluster2=self.cross_filter(concate_cluster[:,:self.channel],concate_cluster[:,self.channel:]),\
                        self.cross_filter(concate_cluster[:,self.channel:],concate_cluster[:,:self.channel])
        cluster1,cluster2=self.attention_block_self(cluster1,cluster1),self.attention_block_self(cluster2,cluster2)
        #unpooling
        desc1_new,seed_weight1=self.attention_learn(cluster1,cluster2)
        desc2_new,seed_weight2=self.attention_learn(cluster2,cluster1)
        
        # seed_weight=self.confidence_filter(torch.cat([cluster1,cluster2],dim=1))
        # seed_weight=torch.sigmoid(seed_weight).squeeze(1)
        desc1_new,desc2_new=self.attention_block_up(desc1,cluster1),self.attention_block_up(desc2,cluster2)
        return desc1_new,desc2_new,seed_weight1,seed_weight2
        

class matcher(nn.Module):
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
        # self.domain_topk=config.domain_topk #邻域聚合数量
        # self.domain_radiues=config.domain_radiues
        # self.separate_num1=config.separate_num

        # 坐标位置编码 
        self.position_encoder = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1) if config.use_score_encoding else nn.Conv1d(2, 32, kernel_size=1),
                                              nn.SyncBatchNorm(32),nn.ReLU(),
                                              nn.Conv1d(32, 64, kernel_size=1), nn.SyncBatchNorm(64), nn.ReLU(),
                                              nn.Conv1d(64, 128, kernel_size=1), nn.SyncBatchNorm(128), nn.ReLU(),
                                              nn.Conv1d(128, 256, kernel_size=1), nn.SyncBatchNorm(256), nn.ReLU(),
                                              nn.Conv1d(256, config.net_channels, kernel_size=1))
        
        # 领域聚合
        self.domain_encoder=domain_attention(config.net_channels)
        
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


        desc_dismat=(2-2*torch.matmul(desc1,desc2.transpose(1,2))).sqrt_()
        values,nn_index=torch.topk(desc_dismat,k=2,largest=False,dim=-1,sorted=True)
        nn_index2=torch.min(desc_dismat,dim=1).indices.squeeze(1)
        inverse_ratio_score,nn_index1=values[:,:,1]/values[:,:,0],nn_index[:,:,0]#get inverse score
   
        #initial seeding
        seed_index1,seed_index2,_,_=seeding(nn_index1,nn_index2,x1,x2,self.seed_top_k[0],inverse_ratio_score,self.conf_bar[0],\
                                self.seed_radius_coe,test=test_mode) 
        # seed_index1,seed_index2,pos_dismat1,pos_dismat2=seeding(nn_index1,nn_index2,x1,x2,self.seed_top_k[0],inverse_ratio_score,self.conf_bar[0],\
        #                         self.seed_radius_coe,test=test_mode) 


 
        # 位置编码
        desc1,desc2=desc1.transpose(1,2), desc2.transpose(1,2) #交换channel和feature of number，遵循torch默认channel在前
        #获取领域的描述子们
        # domain_desc1,domain_desc2=domain(pos_dismat1,pos_dismat2,self.domain_topk,self.domain_radiues,desc1,desc2)
        #领域聚合
        # desc1_embedding, desc2_embedding=self.domain_encoder(domain_desc1), self.domain_encoder(domain_desc2)
        # desc1_embedding, desc2_embedding=torch.nn.functional.normalize(desc1_embedding,dim=-1),torch.nn.functional.normalize(desc2_embedding,dim=-1)
        
        if not self.use_score_encoding:
            encode_x1,encode_x2 = encode_x1[:,:,:2], encode_x2[:,:,:2]
        encode_x1,encode_x2 = encode_x1.transpose(1,2), encode_x2.transpose(1,2)
        
        x1_pos_embedding, x2_pos_embedding =self.position_encoder(encode_x1), self.position_encoder(encode_x2) 
        # x1_pos_embedding,x2_pos_embedding=torch.nn.functional.normalize(x1_pos_embedding,dim=-1),torch.nn.functional.normalize(x2_pos_embedding,dim=-1)
        aug_desc1,aug_desc2=x1_pos_embedding+desc1, x2_pos_embedding+desc2 #最终拥有邻域、坐标、描述子的特征
        # aug_desc1,aug_desc2=desc1_embedding+x1_pos_embedding+desc1, desc2_embedding+x2_pos_embedding+desc2 #最终拥有邻域、坐标、描述子的特征


        seed_weight1_tower,seed_weight2_tower,mid_p_tower,seed_index_tower,nn_index_tower,separate1_index_tower,separate2_index_tower=[],[],[],[],[],[],[]
        seed_index_tower.append(torch.stack([seed_index1,seed_index2],dim=-1)) #保存每轮次的种子索引
        
        nn_index_tower.append(nn_index) # 保存图B中分别与图A中特征点最近的两个索引
        

        seed_para_index=0
        for i in range(self.layer_num):
            #mid seeding
            if i in self.seedlayer and i!= 0:
                seed_para_index+=1
                aug_desc1,aug_desc2=self.mid_final_project(aug_desc1),self.mid_final_project(aug_desc2)
                M=torch.matmul(aug_desc1.transpose(1,2),aug_desc2)
                p=sink_algorithm(M,self.mid_dustbin[str(i)],self.sink_iter[seed_para_index-1])
                mid_p_tower.append(p)
                #rematching with p
                values,nn_index=torch.topk(p[:,:-1,:-1],k=1,dim=-1)
                nn_index2=torch.max(p[:,:-1,:-1],dim=1).indices.squeeze(1)
                p_match_score,nn_index1=values[:,:,0],nn_index[:,:,0]
                #reseeding
                seed_index1, seed_index2,_,_ = seeding(nn_index1,nn_index2,x1,x2,self.seed_top_k[seed_para_index],p_match_score,\
                                                    self.conf_bar[seed_para_index],self.seed_radius_coe,test=test_mode)
                seed_index_tower.append(torch.stack([seed_index1, seed_index2],dim=-1)), nn_index_tower.append(nn_index1)
                if not test_mode and data['step']<self.detach_iter:
                    aug_desc1,aug_desc2=aug_desc1.detach(),aug_desc2.detach()
            aug_desc1,aug_desc2,seed_weight1,seed_weight2 = self.hybrid_block[i](aug_desc1,aug_desc2,seed_index1,seed_index2)
            seed_weight1_tower.append(seed_weight1),seed_weight2_tower.append(seed_weight2)

       
        
        aug_desc1,aug_desc2 = self.final_project(aug_desc1), self.final_project(aug_desc2)
        cmat = torch.matmul(aug_desc1.transpose(1,2), aug_desc2) #欧式距离
        p = sink_algorithm(cmat, self.dustbin,self.sink_iter[-1])
    
        return {'p':p,'seed_weight1_conf':seed_weight1_tower,'seed_weight2_conf':seed_weight2_tower,'seed_index':seed_index_tower,\
                'mid_p':mid_p_tower,'nn_index':nn_index_tower}

            

                

        

            