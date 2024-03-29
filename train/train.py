import torch
import torch.optim as optim
from tqdm import trange
import os
from tensorboardX import SummaryWriter
import numpy as np
import cv2
from torch.cuda.amp.grad_scaler import GradScaler 
import torch.distributed as dist
from loss import GSNLoss
from valid import valid,dump_train_vis
from distributed_utils import is_main_process

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))#文件根目录的绝对路径
sys.path.insert(0, ROOT_DIR)


from utils import train_utils
def train_step(optimizer, model, match_loss, data,step,pre_avg_loss):
    scaler = GradScaler()
    data['step']=step
    optimizer.zero_grad() #清空梯度
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        result=model(data,test_mode=False) # 模型训练
        loss_res=match_loss.run(data,result) # 计算损失
    
    
    scaler.scale(loss_res['total_loss']).backward() #损失加入反向传
    #apply reduce on all record tensor
    for key in loss_res.keys():
        loss_res[key]=train_utils.reduce_tensor(loss_res[key],'mean') #将分布式训练的损失函数梯度合并更新梯度然后广播回各个分布式设备
  
    if loss_res['total_loss']<7*pre_avg_loss or step<200 or pre_avg_loss==0:
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        unusual_loss=False
    else:
        optimizer.zero_grad()
        unusual_loss=True
    
    return loss_res,unusual_loss


def train(model, train_loader, valid_loader, config,model_config, train_sampler):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr)#初始化优化器
    
    if config.model_name=='SGM':#根据模型选择loss计算方式
        match_loss = GSNLoss(config,model_config) 
    else:
        raise NotImplementedError
    
    checkpoint_path = os.path.join(config.log_base, 'checkpoint.pth')#config.log_base为log路径,也为断点路径
    config.resume = os.path.isfile(checkpoint_path)#是否存在断点
    if config.resume:
        # if get_rank == 0: #暂时注释掉这行是测试print是否only on master
        print('==> 从断点恢复')
        checkpoint = torch.load(checkpoint_path,map_location='cuda:{}'.format(config.local_rank))# 加载模型文件到GPU上

        model.load_state_dict(checkpoint['state_dict']) #加载模型的断点前训练权重
        best_acc = checkpoint['best_acc']
        start_step = checkpoint['step']
        optimizer.load_state_dict(checkpoint['optimizer']) #加载优化器的断点前权重
    else: # 没有断点文件，就初始化训练步骤等参数从头开始训练
        best_acc = -1
        start_step = 0
    
    if is_main_process():
        writer=SummaryWriter(os.path.join(config.log_base,'log_file'))#运行的log数据均存于此(初始化summarywriter)

    train_sampler.set_epoch(start_step*config.train_batch_size//len(train_loader.dataset))#设置ddp采样 
    train_loader_iter = iter(train_loader) # 创建一个迭代对象
    pre_avg_loss=0
    #start_step:开始的步骤，中断时记录中断步骤。train_iter:迭代次数
    if is_main_process():
        progress_bar=trange(start_step, config.train_iter,ncols=config.tqdm_width)
    else:
        progress_bar=range(start_step, config.train_iter)#进度条参数
    for step in progress_bar: #所有训练轮次循环
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            # if get_rank == 0:
            print('epoch: ',step*config.train_batch_size//len(train_loader.dataset))
            train_sampler.set_epoch(step*config.train_batch_size//len(train_loader.dataset))
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
    
        train_data = train_utils.tocuda(train_data)#将数据重新载入gpu
        # lr=((step/config.decay_iter) * config.train_lr) if step < config.decay_iter else (config.train_lr*config.decay_rate**(step-config.decay_iter))
        lr=min(config.train_lr*config.decay_rate**(step-config.decay_iter),config.train_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # run training
        loss_res,unusual_loss = train_step(optimizer, model, match_loss, train_data,step-start_step,pre_avg_loss)
        if (step-start_step)<=200:
            pre_avg_loss=loss_res['total_loss'].data
        if (step-start_step)>200 and not unusual_loss:
            pre_avg_loss=pre_avg_loss.data*0.9+loss_res['total_loss'].data*0.1
        if unusual_loss and is_main_process():
            print('unusual loss! pre_avg_loss: ',pre_avg_loss,'cur_loss: ',loss_res['total_loss'].data)
        #log
        if is_main_process() and step%config.log_intv==0 and not unusual_loss:
            writer.add_scalar('TotalLoss',loss_res['total_loss'],step)
            writer.add_scalar('CorrLoss',loss_res['loss_corr'],step)
            writer.add_scalar('InCorrLoss', loss_res['loss_incorr'], step)
            writer.add_scalar('dustbin', model.module.dustbin, step)
            writer.add_scalar('acc_corr',loss_res['acc_corr'], step)
            writer.add_scalar('acc_incorr',loss_res['acc_incorr'], step)
            # writer.add_scalar('acc_mid_corr',loss_res['mid_acc_corr'], step)

         
            if config.model_name=='SGM':
                writer.add_scalar('Separ1ConfLoss', loss_res['loss_separ1_conf'], step)
                writer.add_scalar('Separ2ConfLoss', loss_res['loss_separ2_conf'], step)
                writer.add_scalar('loss_nomatch1', loss_res['loss_nomatch1'], step)
                writer.add_scalar('loss_nomatch2', loss_res['loss_nomatch2'], step)
                # writer.add_scalar('MidCorrLoss', loss_res['loss_corr_mid'].sum(), step)
                # writer.add_scalar('MidInCorrLoss', loss_res['loss_incorr_mid'].sum(), step)
            

        # valid ans save
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            total_loss,acc_corr,acc_incorr,separ1_precision,separ1_recall,separ2_precision,separ2_recall,total_precision_tower,\
                total_recall_tower,weight1,weight2=valid(valid_loader, model, match_loss, config,model_config)
            # total_loss,acc_corr,acc_incorr,separ1_precision,separ1_recall,separ2_precision,separ2_recall,total_precision_tower,\
            #     total_recall_tower,acc_mid,weight1,weight2=valid(valid_loader, model, match_loss, config,model_config)

            if is_main_process():
                writer.add_scalar('ValidAcc', acc_corr, step)
                writer.add_scalar('ValidLoss', total_loss, step)
                writer.add_image('weight1',weight1*255,step)
                writer.add_image('weight2',weight2*255,step)
                    
                
                if config.model_name=='SGM':
                    for i in range(len(separ1_recall)):
                        writer.add_scalar('separ1_conf_pre_%d'%i,separ1_precision[i],step)
                        writer.add_scalar('separ1_conf_recall_%d' % i, separ1_recall[i], step)
                        writer.add_scalar('separ2_conf_pre_%d'%i,separ2_precision[i],step)
                        writer.add_scalar('separ2_conf_recall_%d'%i, separ2_recall[i], step)
                    # for i in range(len(acc_mid)):
                    #     writer.add_scalar('acc_mid%d'%i,acc_mid[i],step)
                    print('acc_corr: ',acc_corr.data,'acc_incorr: ',acc_incorr.data,'separ1_conf_pre: ',separ1_precision.mean().data,'separ1_conf_recall: ',\
                          separ1_recall.mean().data,'separ2_conf_per: ',separ2_precision.mean().data,'separ2_conf_recall: ',separ2_recall.mean().data,\
                          'total_conf_pre: ',total_precision_tower.mean().data,'total_conf_recall: ',total_recall_tower.mean().data)
                    # print('acc_corr: ',acc_corr.data,'acc_incorr: ',acc_incorr.data,'separ1_conf_pre: ',separ1_precision.mean().data,'separ1_conf_recall: ',\
                    #       separ1_recall.mean().data,'separ2_conf_per: ',separ2_precision.mean().data,'separ2_conf_recall: ',separ2_recall.mean().data,\
                    #       'total_conf_pre: ',total_precision_tower.mean().data,'total_conf_recall: ',total_recall_tower.mean().data,'acc_mid: ',acc_mid.mean().data)

                else:
                     print('acc_corr: ',acc_corr.data,'acc_incorr: ',acc_incorr.data)
                
                #saving best
                if acc_corr > best_acc: #验证模型，当准确率高于上次验证结果则保存最佳模型(best_acc初始化为-1)
                    print("Saving best model with va_res = {}".format(acc_corr))
                    best_acc = acc_corr
                    save_dict={'step': step + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()}
                    save_dict.update(save_dict)
                    torch.save(save_dict, os.path.join(config.log_base, 'model_best.pth'))
                    # test_data = {
                    #     'x1':torch.rand(1,1000,2).cuda()-0.5,
                    #     'x2':torch.rand(1,1000,2).cuda()-0.5,
                    #     'desc1': torch.rand(1,1000,128).cuda(),
                    #     'desc2': torch.rand(1,1000,128).cuda()
                    #     }
                    # trace_model = torch.jit.script(model, example_inputs=test_data)
                    # trace_model.save(os.path.join(config.log_base, "model_best.pt"))

        if b_save: #保存断点模型
            if is_main_process():
                save_dict={'step': step + 1,
                'state_dict': model.module.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()}
                torch.save(save_dict, checkpoint_path)#保存模型
            
            #draw match results
            model.eval()
            with torch.no_grad():
                if is_main_process():
                    if not os.path.exists(os.path.join(config.train_vis_folder,'train_vis')):
                        os.mkdir(os.path.join(config.train_vis_folder,'train_vis'))
                    if not os.path.exists(os.path.join(config.train_vis_folder,'train_vis',config.log_base)):
                        os.mkdir(os.path.join(config.train_vis_folder,'train_vis',config.log_base))
                    os.mkdir(os.path.join(config.train_vis_folder,'train_vis',config.log_base,str(step)))
                res=model(train_data) 
                dump_train_vis(res,train_data,step,config)
            model.train()
    
    if is_main_process():
        writer.close()
