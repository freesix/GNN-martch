import os
import shutil
# #源文件夹
source_file = '/mnt/h/0/code/codeData/GL3D-2/data/'
target_file = '/home/freesix/GL3D-2/data/'
file_list = []
subfile_list = os.listdir(source_file)#子文件名称
for i in subfile_list:
    file_list.append(source_file+i)#完全路径名称

for j in os.listdir(target_file):
    if j in subfile_list:
        # print(os.path.join(target_file,j,'basename.txt'))
        # print(source_file+j)
        shutil.copy(os.path.join(target_file,j,'basenames.txt'),source_file+j)
