# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:57:22 2019

@author: muli
"""
import numpy as np
import scipy.io as scio
import random

path = '/data2/lab-shixian/project/ActivePointCloud/Dataset/ShapeNet/Preprocess/SampIndex_m-0.001.mat'
data = scio.loadmat(path)
     
file_idx_list = data['file_idx_list']           
data_idx_list = data['data_idx_list']  
pts_idx_list = data['pts_idx_list']  

print('file_idx_list:',file_idx_list.shape,file_idx_list.dtype)
print('data_idx_list:',data_idx_list.shape,data_idx_list.dtype)
print('pts_idx_list:',pts_idx_list.shape,pts_idx_list.dtype)



# print((a==b).all())

# randomMatrix=np.random.randint(0,2048,(12137,20))

# a = np.max(randomMatrix)
# b = np.min(randomMatrix)


num_rate = 0.500

pts_idx_list_new = np.zeros([len(pts_idx_list),int(num_rate*2048)],int)
for i in range(len(pts_idx_list)):
    pts_idx_list_new[i]=np.array(random.sample(range(0,2048),int(num_rate*2048)))


dataNew = '/data2/lab-shixian/project/ActivePointCloud/Dataset/ShapeNet/Preprocess/SampIndex_m-' + np.str(round(num_rate,3)) + '00.mat'

print(dataNew)
print('file_idx_list:',file_idx_list.shape,file_idx_list.dtype)
print('data_idx_list:',data_idx_list.shape,data_idx_list.dtype)
print('pts_idx_list:',pts_idx_list_new.shape,pts_idx_list_new.dtype)
#scio.savemat(dataNew, {'file_idx_list':data['file_idx_list'],'data_idx_list':data['data_idx_list'],'pts_idx_list':pts_idx_list_new})




data3 = scio.loadmat(dataNew)
file_idx_list3 = data3['file_idx_list']           
data_idx_list3 = data3['data_idx_list']  
pts_idx_list3 = data3['pts_idx_list']  


print('file_idx_list3:',file_idx_list3.shape,file_idx_list3.dtype)
print('data_idx_list3:',data_idx_list3.shape,data_idx_list3.dtype)
print('pts_idx_list3:',pts_idx_list3.shape,pts_idx_list3.dtype)


print((file_idx_list==file_idx_list3).all())
print((data_idx_list==data_idx_list3).all())
