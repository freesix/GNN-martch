import  numpy as np


def norm_kpt(K, kp):
    kp = np.concatenate([kp, np.ones([kp.shape[0], 1])], axis=1)#转换为齐次坐标
    kp = np.matmul(kp, np.linalg.inv(K).T)[:, :2]  #将kp除以尺度变换信息K
    return kp
    
def unnorm_kp(K,kp):
    kp = np.concatenate([kp, np.ones([kp.shape[0], 1])], axis=1) #给特征点坐标增加一列全1,将坐标转换为齐次坐标
    kp = np.matmul(kp,K.T)[:, :2] #将kp乘以尺度变换信息K
    return kp

def interpolate_depth(pos, depth): #depth深度信息，pos：增加尺度后的坐标信息
        # pos:[y,x] #特征点的像素坐标系
        ids = np.array(range(0, pos.shape[0]))#给坐标编id
    
        h, w = depth.shape #深度图像的长宽[250,250]

        i = pos[:, 0] #y
        j = pos[:, 1] #x
        valid_corner=np.logical_and(np.logical_and(i>0,i<h-1),np.logical_and(j>0,j<w-1)) #排除实际图像映射到深度图像后的特征点坐标不在深度图坐标范围内的特征点
        i,j=i[valid_corner],j[valid_corner]
        ids = ids[valid_corner]
        #将特征点坐标值分别向上向下取整
        i_top_left = np.floor(i).astype(np.int32)#np.floor()向下取整
        j_top_left = np.floor(j).astype(np.int32)

        i_top_right = np.floor(i).astype(np.int32)
        j_top_right = np.ceil(j).astype(np.int32) #np.ceil()向上取整

        i_bottom_left = np.ceil(i).astype(np.int32)
        j_bottom_left = np.floor(j).astype(np.int32)

        i_bottom_right = np.ceil(i).astype(np.int32)
        j_bottom_right = np.ceil(j).astype(np.int32)
        
        #特征点映射到深度图中某个体素所对应的深度,通俗来说就是将落在此个体素内特征点的深度用对应四边形四个角所在深度表示
        depth_top_left,depth_top_right,depth_down_left,depth_down_right=depth[i_top_left, j_top_left],depth[i_top_right, j_top_right],\
                                                             depth[i_bottom_left, j_bottom_left],depth[i_bottom_right, j_bottom_right]
        #排除不正确的深度信息
        valid_depth = np.logical_and(
            np.logical_and(
                depth_top_left > 0,
                depth_top_right > 0
            ),
            np.logical_and(
                depth_down_left > 0,
                depth_down_left > 0
            )
        )
        ids=ids[valid_depth]
        depth_top_left,depth_top_right,depth_down_left,depth_down_right=depth_top_left[valid_depth],depth_top_right[valid_depth],\
                                                                        depth_down_left[valid_depth],depth_down_right[valid_depth]

        i,j,i_top_left,j_top_left=i[valid_depth],j[valid_depth],i_top_left[valid_depth],j_top_left[valid_depth]
        
        # 插值(分别计算特征点和所在深度体素的四个角围成的四个矩形的面积)
        dist_i_top_left = i - i_top_left.astype(np.float32)
        dist_j_top_left = j - j_top_left.astype(np.float32)
        w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
        w_top_right = (1 - dist_i_top_left) * dist_j_top_left
        w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
        w_bottom_right = dist_i_top_left * dist_j_top_left
        #根据四个矩形面积赋予不同权重计算特征点的平均深度
        interpolated_depth = (
            w_top_left * depth_top_left +
            w_top_right * depth_top_right+
            w_bottom_left * depth_down_left +
            w_bottom_right * depth_down_right
        )
        return [interpolated_depth, ids]

def reprojection(depth_map,kpt,dR,dt,K1_img2depth,K1,K2):
    #warp kpt from img1 to img2
    def swap_axis(data):
        return np.stack([data[:, 1], data[:, 0]], axis=-1)

    kp_depth = unnorm_kp(K1_img2depth,kpt)#给图像坐标增加尺度信息，转换到深度图像尺度
    uv_depth = swap_axis(kp_depth) #(x,y)->(y,x)
    z,valid_idx = interpolate_depth(uv_depth, depth_map)#z:深度，对应特征点id

    norm_kp=norm_kpt(K1,kpt) 
    norm_kp_valid = np.concatenate([norm_kp[valid_idx, :], np.ones((len(valid_idx), 1))], axis=-1) #转换为齐次坐标
    xyz_valid = norm_kp_valid * z.reshape(-1, 1)
    xyz2 = np.matmul(xyz_valid, dR.T) + dt.reshape(1, 3) #世界坐标系
    xy2 = xyz2[:, :2] / xyz2[:, 2:] #归一化(用x,y坐标除以最后一个维度)
    kp2, valid = np.ones(kpt.shape) * 1e5, np.zeros(kpt.shape[0])
    kp2[valid_idx] = unnorm_kp(K2,xy2)
    valid[valid_idx] = 1
    return kp2, valid.astype(bool)

def reprojection_2s(kp1, kp2,depth1, depth2, K1, K2, dR, dt, size1,size2):
    #size:H*W
    depth_size1,depth_size2 = [depth1.shape[0], depth1.shape[1]], [depth2.shape[0], depth2.shape[1]] #深度图像大小，长x宽
    scale_1= [float(depth_size1[0]) / size1[0], float(depth_size1[1]) / size1[1], 1] #深度图除以实际图像尺度[250/1000,250/1000,1]
    scale_2= [float(depth_size2[0]) / size2[0], float(depth_size2[1]) / size2[1], 1]
    K1_img2depth, K2_img2depth = np.diag(np.asarray(scale_1)), np.diag(np.asarray(scale_2)) #以尺度向量生成一个3x3的对角矩阵
    kp1_2_proj, valid1_2 = reprojection(depth1, kp1, dR, dt, K1_img2depth,K1,K2)
    kp2_1_proj, valid2_1 = reprojection(depth2, kp2, dR.T, -np.matmul(dR.T, dt), K2_img2depth,K2,K1)
    return [kp1_2_proj,kp2_1_proj],[valid1_2,valid2_1]

def make_corr(kp1,kp2,desc1,desc2,depth1,depth2,K1,K2,dR,dt,size1,size2,corr_th,incorr_th,check_desc=False):
    #make reprojection
    [kp1_2,kp2_1],[valid1_2,valid2_1]=reprojection_2s(kp1,kp2,depth1,depth2,K1,K2,dR,dt,size1,size2)
    num_pts1, num_pts2 = kp1.shape[0], kp2.shape[0]
    #计算映射前后对应点的距离之差
    dis_mat1=np.sqrt(abs((kp1 ** 2).sum(1,keepdims=True) + (kp2_1 ** 2).sum(1,keepdims=False)[np.newaxis] - 2 * np.matmul(kp1, kp2_1.T))) #2映射到1
    dis_mat2 =np.sqrt(abs((kp2 ** 2).sum(1,keepdims=True) + (kp1_2 ** 2).sum(1,keepdims=False)[np.newaxis] - 2 * np.matmul(kp2,kp1_2.T)))
    repro_error = np.maximum(dis_mat1,dis_mat2.T) #n1*n2 #逐元素比较得最大值
    
    # find corr index
    nn_sort1 = np.argmin(repro_error, axis=1)#返回每一行最小元素索引
    nn_sort2 = np.argmin(repro_error, axis=0) #返回每一列最小元素索引
    mask_mutual = nn_sort2[nn_sort1] == np.arange(kp1.shape[0]) #最近匹配都为对方的为True
    mask_inlier=np.take_along_axis(repro_error,indices=nn_sort1[:,np.newaxis],axis=-1).squeeze(1)<corr_th #在阈值内的为True
    mask = mask_mutual&mask_inlier
    corr_index=np.stack([np.arange(num_pts1)[mask], np.arange(num_pts2)[nn_sort1[mask]]], axis=-1) #正确匹配的索引
    
    if check_desc:
        #filter kpt in same pos using desc distance(e.g. DoG kpt)
        x1_valid, x2_valid = kp1[corr_index[:, 0]], kp2[corr_index[:, 1]]
        mask_samepos1=np.logical_and(x1_valid[:, 0,np.newaxis] == kp1[np.newaxis,:, 0],x1_valid[:, 1,np.newaxis] == kp1[np.newaxis,:, 1])
        mask_samepos2=np.logical_and(x2_valid[:, 0,np.newaxis]== kp2[np.newaxis,:, 0],x2_valid[:, 1,np.newaxis] == kp2[np.newaxis,:, 1])
        duplicated_mask=np.logical_or(mask_samepos1.sum(-1)>1,mask_samepos2.sum(-1)>1)
        duplicated_index=np.nonzero(duplicated_mask)[0]

        unique_corr_index=corr_index[~duplicated_mask]
        clean_duplicated_corr=[]
        for index in duplicated_index:
            cur_desc1, cur_desc2 = desc1[mask_samepos1[index]], desc2[mask_samepos2[index]]
            cur_desc_mat = np.matmul(cur_desc1, cur_desc2.T)
            cur_max_index =[np.argmax(cur_desc_mat)//cur_desc_mat.shape[1],np.argmax(cur_desc_mat)%cur_desc_mat.shape[1]]
            clean_duplicated_corr.append(np.stack([np.arange(num_pts1)[mask_samepos1[index]][cur_max_index[0]],
                        np.arange(num_pts2)[mask_samepos2[index]][cur_max_index[1]]]))
        
        clean_corr_index=unique_corr_index
        if len(clean_duplicated_corr)!=0:
            clean_duplicated_corr=np.stack(clean_duplicated_corr,axis=0)
            clean_corr_index=np.concatenate([clean_corr_index,clean_duplicated_corr],axis=0)
    else:
        clean_corr_index=corr_index
    # find incorr
    mask_incorr1 = np.min(dis_mat2.T[valid1_2], axis=-1) > incorr_th
    mask_incorr2 = np.min(dis_mat1.T[valid2_1], axis=-1) > incorr_th
    incorr_index1, incorr_index2 = np.arange(num_pts1)[valid1_2][mask_incorr1.squeeze()], \
                                    np.arange(num_pts2)[valid2_1][mask_incorr2.squeeze()]

    return clean_corr_index,incorr_index1,incorr_index2

