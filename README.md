# 这是论文代码   
## 数据准备
[GL3D](https://github.com/lzx551402/GL3D)数据集，按照文档配置好
* 进入dumpdata目录，进入configs目录配置好gl3d.yaml文件，里面参数按照说明配好，运行```python3 dump.py```  
## 开始训练
* 进入train目录，进入configs目录配置好sgm.yaml，然后找到config.py文件配置好
* 运行```python3 main.py```，如果多GPU或者多机器分布式训练运行```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 main.py --nodes 1 --ngpus_per_node 6``` 这个例子是以一台机器6个GPU为例
## 文件目录
* components
    > evaluators.py 做demo时候用的验证代码---暂时没写  
    > extoractors.py 特征点提取，目前只有SIFT方法  
    > load_comonent.py 数据准备流程封装  
    > matchers.py demo相关  
    > reders.py demo相关，当时想做demo来着  
* dumpdata
    > configs 存放数据集准备的配置文件，如gl3d数据 
    * dumper 
        > base_train.py 数据准备的基类和方法，可以理解为不同数据集处理方法的基类  
        > gl3d_train.py gl3d数据集的处理类和方法，从base_dumper.py中继承过来  
    > dump.py 数据预处理
* evaluation 
    > configs 存储验证的配置文件
    > eval_cost.py 验证模型一次时间
    > evaluate.py 验证可视化相关
* sgmnet 
    > match_model_copy.py 略
    > match_model.py 网络模型
* superpoint
    > superpoint.py 深度学习提取特征点方法，这里是superpoint方法，copy而来
* train
    > configs 训练配置文件
    > log 训练保存日志、断点、最佳模型--训练后自动生成文件夹  
    > train_vis 验证集验证结果保存--自动生成  
    > config.py 配置参数
    > dataset.py 数据组织代码，不同数据集预处理后有关信息可能不同，训练过程中按需组织和提取相关信息  
    > distributed_utils.py 分布式训练相关函数  
    > loss.py 损失函数  
    > main.py 主函数  
    > train.py 训练代码，训练过程  
    > valid.py 验证代码
* utils 
    > data_utils.py 数据处理过程中，深度信息和图像信息对齐，计算争取错误匹配(打标签用)  
    > evaluation_utils.py 验证相关杂项，画图，归一化等
    > train_utils.py 训练相关杂项
    > transformations.py 数据格式转换有关  
* convert.py 实验静态图转换有关--不用管