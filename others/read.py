import os 
import h5py 
import numpy as np
#用于读取h5py文件信息
feature_path = '/mnt/h/0/code/codeData/GL3D-2/feature_dump_dir/' #特征文件夹路径
data_path = '/mnt/h/0/code/codeData/GL3D-2/data/'

# files = os.listdir(file_path)
'''
#path:path for h5py
#name:h5py name
'''
def read_h5py(path, name, image):
    with h5py.File(os.path.join(path,name,image), 'r') as f:
        for key, value in f.items():
            print(f"{key}")

# path = feature_path 
# name = "00000000000000000000000f"    
# image = "00000000.jpg_root_1000.hdf5"
# read_h5py(path, name, image)

dataset_path = '/mnt/h/0/code/codeData/GL3D-2/dataset_dump_dir/'
dataset_name = "00000000000000000000000f" 
info_file = "info.h5py"
read_h5py(dataset_path,dataset_name,info_file)

# def sinkhorn(mu, nu, C, epsilon, max_iter=1000, tol=1e-6):

#     n,m = C.shape

#     u = np.ones(n)
#     v = np.ones(m)

#     for k in range(max_iter):
#         K = np.exp(-C/epsilon)
#         gamma = np.outer(u, v) * K / (np.sum(K * np.outer(u, v)+epsilon))

#         u_new = mu /np.dot(K, v)
#         if np.allclose(u, u_new, atol=tol):
#             break
#         u = u_new

#         v_new = nu /np.dot(K.T, u)
#         if np.allclose(v, v_new, atol=tol):
#             break
#         v = v_new
#     return gamma


# a = np.random.rand(20)
# b = np.random.rand(23)
# C = np.random.rand(20, 23)
# eps = 1e-8

# gam = sinkhorn(a,b,C,eps)
# print(gam)
