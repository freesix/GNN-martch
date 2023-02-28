from abc import ABCMeta, abstractmethod 
import os 
import h5py 
import numpy as np 
from tqdm import trange 
from torch.multiprocessing import Pool, set_start_method
set_start_method('spawn', force=True)#多进程方法，此处为spawn，此时父进程会启动一个新的python解释器，子进程只继承运行进程对象的方法所需资源

import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))#定义顶层文件路径
sys.path.insert(0, ROOT_DIR)#将顶层文件路径插入系统路径，使得顶层的components文件可以导入

from components import load_component


#获取数据和组织数据的类
class BaseDumper(metaclass=ABCMeta):
    def __init__(self,config):
        self.config = config
        self.img_seq = []
        self.dump_seq = []

    @abstractmethod
    def get_seqs(self):
        raise NotImplementedError 

    @abstractmethod
    def format_dump_folder():
        raise NotImplementedError 



    def initialize(self):
        # self.extractor：提取特征点类
        self.extractor = load_component('extractor', self.config['extractor']['name'], self.config['extractor'])
        self.get_seqs() #实现在相应文件
        self.format_dump_folder()
 
         
    def extract(self, index):
        img_path, dump_path = self.img_seq[index],self.dump_seq[index]
        if not self.config['extractor']['overwrite'] and os.path.exists(dump_path):
            return
        kp, desc = self.extractor.run(img_path) #调用提取特征点方法
        #将提取特征点信息写入相应文件
        self.write_feature(kp, desc, dump_path)

    #实现特征点提取后写入方法
    def write_feature(self, pts, desc, filename):
        with h5py.File(filename, "w") as ifp: #初始化File类
            ifp.create_dataset('keypoints', pts.shape, dtype=np.float32) #定义成员大小和数据类型
            ifp.create_dataset('descriptors', desc.shape, dtype=np.float32) 
            ifp["keypoints"][:] = pts
            ifp["descriptors"][:] = desc
    
    '''提取序列中所有图片特征点
    实现多线程提取特征点'''
    def dump_feature(self):
        print('提取特征点中') 
        self.num_img = len(self.dump_seq) #self.dump_seq是从具体get_seqs()中得到
        pool = Pool(self.config['extractor']['num_process']) # 初始化线程池
        iteration_num =self.num_img//self.config['extractor']['num_process'] #计算迭代次数
        if self.num_img%self.config['extractor']['num_process']!=0:#除不尽则增加一次迭代次数
            iteration_num+=1
        for index in trange(iteration_num): #显示多线程提取特征点进度条
            indicies_list = range(index*self.config['extractor']['num_process'],min((index+1)*self.config['extractor']['num_process'],self.num_img))
            pool.map(self.extract, indicies_list)
        pool.close()
        pool.join()






