import os 
import h5py 
import numpy as np
#用于读取h5py文件信息
file_path = '/mnt/h/0/code/codeData/GL3D-2/feature_dump_dir/' #特征文件夹路径
data_path = '/mnt/h/0/code/codeData/GL3D-2/data/'

# basename_path = os.path.join(data_path, 'basename.txt')
# print(basename_path)
files = os.listdir(file_path)

Fea_list = []
for i, f in enumerate(files):
    feature_files = os.path.join(file_path,f)
    basename_path = os.path.join(data_path,f,'basenames.txt')
    images = np.loadtxt(basename_path)
    print(images)
    fea_list = h5py.File(feature_files, 'r') 
    Fea_list[i] = fea_list

