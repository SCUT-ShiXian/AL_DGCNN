



import h5py
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse as arg
import json
import time
import scipy.io as scio
from datetime import datetime
import random
sys.path.append(os.path.abspath('./Util'))
sys.path.append(os.path.abspath('./ShapeNet'))
sys.path.append(os.path.abspath('./Strategies'))


import DataIO_ShapeNet as IO
import ShapeNet_DGCNN_util as util
import Tool
from Tool import printout
import Evaluation
import strategy_coreset as strategy
from strategy_coreset import kcenter_greedy


parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--m','-m',type=float,help='the ratio/percentage of points selected to be labelled (0.01=1%, '
                                               '0.05=5%, 0.1=10%, 1=100%)[default: 0.01]',default=0.01)


args = parser.parse_args()



def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    num_data = data.shape[0]
    data_idx = np.arange(0,num_data)

    return (data, label, seg, num_data, data_idx)



#### Load Sampled Point Index
save_path = os.path.expanduser(os.path.abspath('./Dataset/ShapeNet/Preprocess/'))
save_filepath = os.path.join(save_path, 'SampIndex_m-{:.3f}.mat'.format(args.m))
tmp = scio.loadmat(save_filepath)

pts_idx_list = tmp['pts_idx_list']
file_idx_list = np.zeros(shape=[pts_idx_list.shape[0]])
data_idx_list = np.arange(0,pts_idx_list.shape[0])


print('pts_idx_list---',pts_idx_list.shape, type(pts_idx_list))
print('file_idx_list---',file_idx_list.shape, np.max(file_idx_list))
print('data_idx_list---',data_idx_list.shape)


h5_base_path = '/data2/lab-shixian/project/ActivePointCloud/Dataset/ShapeNet/hdf5_data'
train_file_list = getDataFiles(os.path.join(h5_base_path, 'train_hdf5_file_list.txt'))


train_data = []
train_labels = []
train_seg = []
num_train = 0
train_data_idx = []
for cur_train_filename in train_file_list:
    print('cur_train_filename',cur_train_filename)
    cur_train_data, cur_train_labels, cur_train_seg, cur_num_train, cur_train_data_idx = load_h5_data_label_seg(os.path.join(h5_base_path,cur_train_filename))



    train_data.append(cur_train_data)
    train_labels.append(cur_train_labels)
    train_seg.append(cur_train_seg)
    train_data_idx.append(cur_train_data_idx+num_train)
    num_train += cur_num_train



    whole_train_data = np.concatenate(train_data)
    whole_train_labels = np.concatenate(train_labels)
    whole_train_seg = np.concatenate(train_seg)
    whole_train_data_idx = np.concatenate(train_data_idx)
    whole_num_train = num_train

print('len(train_data)------',whole_train_data.shape)
print('len(train_labels)------',whole_train_labels.shape)
print('len(train_seg)------',whole_train_seg.shape)
print('len(train_data_idx)------',whole_train_data_idx.shape, whole_train_data_idx[100:110])
print('num_train------',whole_num_train)


path_base = '/data2/lab-shixian/project/ActivePointCloud/Results/kcenter_greedy_m0.01_query20_round10_Epoch31_noshuffle/ShapeNet/'
path_mat = path_base + 'new_pts_idx_list.mat'
path_save = path_base + 'vis_files'
if not os.path.exists(path_save):
    os.makedirs(path_save)

data = scio.loadmat(path_mat)
pts_idx_list = data['pts_idx_list']
print('pts_idx_list:',pts_idx_list.shape,pts_idx_list.dtype)


pts_idx_list_random = pts_idx_list[:,:20]
pts_idx_list_kcenter = pts_idx_list[:,20:]


c_idx=random.sample(range(0,12137),30)
for i in range(len(c_idx)):
    points_whole = whole_train_data[c_idx[i]]
    points_random = (whole_train_data[c_idx[i]])[pts_idx_list_random[c_idx[i]]]
    points_kcenter = (whole_train_data[c_idx[i]])[pts_idx_list_kcenter[c_idx[i]]]

    print('points_whole:',points_whole.shape,points_whole.dtype)
    print('points_random:',points_random.shape,points_random.dtype)
    print('points_kcenter:',points_kcenter.shape,points_kcenter.dtype)

    np.savetxt(path_save + '/' + str(c_idx[i]) + 'whole.xyz', points_whole, fmt='%.6f')
    np.savetxt(path_save + '/' + str(c_idx[i]) + 'random.xyz', points_random, fmt='%.6f')
    np.savetxt(path_save + '/' + str(c_idx[i]) + 'kcenter.xyz', points_kcenter, fmt='%.6f')

