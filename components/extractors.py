import cv2 
import numpy as np 
import torch
import os 
import sys 

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

from superpoint import SuperPoint #suerpoint提取特征点类


def resize(img,resize):
    img_h,img_w = img.shape[0], img.shape[1]
    cur_size = max(img_h, img_w)
    if len(resize)==1:
        scale1, scale2 = resize[0]/cur_size,resize[0]/cur_size #要进行resize
    else:
        scale1, scale2 = resize[0]/img_h, resize[1]/img_w #不进行resize
    new_h,new_w = int(img_h*scale1),int(img_w*scale2) #按比例缩小
    new_img =cv2.resize(img.astype('float32'), (new_w,new_h)).astype('uint8')
    scale = np.asarray([scale2,scale1])
    return new_img,scale 


class ExtractSIFT:
    def __init__(self,config,root=True):
        self.num_kp = config['num_kpt'] #特征点个数
        self.contrastThreshold = config['det_th'] #滤除特征对比度小的点，SIFT算法中
        self.resize = config['resize'] #是否将图像resize
        self.root = root #私有对象，是否用rootSIFT

    def run(self, img_path):
        #初始化一个SIFT类
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.num_kp, contrastThreshold=self.contrastThreshold) 
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        scale = [1,1]
        if self.resize[0]!=-1:
            img, scale = resize(img, self.resize)
        cv_kp, desc = self.sift.detectAndCompute(img, None)
        kp = np.array([[_kp.pt[0]/scale[1], _kp.pt[1]/scale[0], _kp.response] for _kp in cv_kp])#这里对特征点坐标处理
        index = np.flip(np.argsort(kp[:,2]))#np.flip按照指点维度翻转，未指定则所有维度均翻转
        kp,desc = kp[index], desc[index]
        if self.root:
            desc = np.sqrt(abs(desc/(np.linalg.norm(desc,axis=-1,ord=1)[:,np.newaxis]+1e-8)))
        return kp[:self.num_kp], desc[:self.num_kp]
    

