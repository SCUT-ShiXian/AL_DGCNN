import os
import sys
import numpy as np
import tensorflow as tf
from sklearn import metrics
import argparse as arg
import json
import time
import scipy.io as scio
from datetime import datetime
import h5py
import random
import bisect

sys.path.append(os.path.abspath('./Util'))
sys.path.append(os.path.abspath('./ShapeNet'))
sys.path.append(os.path.abspath('./Strategies'))

#import tf_util
import DataIO_ShapeNet as IO
#print('path',sys.path)
import ShapeNet_DGCNN_util as util
import Tool
from Tool import printout
import Evaluation
import strategy_coreset as strategy

parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--GPU',type=int,help='GPU to use',default=1)
parser.add_argument('--ExpSum','-es',type=int,help='Flag to indicate if export summary',default=1)    # bool
parser.add_argument('--SaveMdl','-sm',type=int,help='Flag to indicate if save learned model',default=1)    # bool
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=3e-3)
parser.add_argument('--Epoch','-ep',type=int,help='Number of epochs to train [default: 51]',default=31)
parser.add_argument('--gamma',type=float,help='L2 regularization coefficient',default=0.1)
parser.add_argument('--batchsize',type=int,help='Training batchsize [default: 1]',default=6)
parser.add_argument('--m','-m',type=float,help='the ratio/percentage of points selected to be labelled (0.01=1%, '
                                               '0.05=5%, 0.1=10%, 1=100%)[default: 0.01]',default=0.01)

parser.add_argument('--NUM_ROUND',type=int,help='SAMPLE NUM_ROUND',default=10)
parser.add_argument('--NUM_QUERY',type=int,help='SAMPLE NUM_QUERY',default=20)
parser.add_argument('--theta',type=int,help='shape score',default=0.2)
parser.add_argument('--log_name',type=str,help='log_name',default='gary')
parser.add_argument('--L_div',type=float,help='lambda diversity',default=1.0)
parser.add_argument('--L_uncer',type=float,help='lambda uncertainty',default=0.1)
parser.add_argument('--query_method',type=str,help='query_method',default='gary')

args = parser.parse_args()

num_round = args.NUM_ROUND
num_query = args.NUM_QUERY
theta = args.theta
query_method = args.query_method
if args.GPU != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)

#### Parameters
Model = 'DGCNN_RandomSamp'
l_div = args.L_div
l_uncer = args.L_uncer

##### Load Training/Testing Data
Loader = IO.ShapeNetIO(os.path.abspath('./Dataset/ShapeNet'),batchsize = args.batchsize)
Loader.LoadTrainValFiles()

##### Evaluation Object
Eval = Evaluation.ShapeNetEval()
## Number of categories
PartNum = Loader.NUM_PART_CATS
output_dim = PartNum
ShapeCatNum = Loader.NUM_CATEGORIES

#### Save Directories
BASE_PATH = os.path.expanduser(os.path.abspath('./Results/{}/ShapeNet/').format(args.log_name))
SUMMARY_PATH = os.path.join(BASE_PATH,Model,'Summary_m-{:.3f}'.format(args.m))
CAM_PATH = os.path.join(BASE_PATH,Model,'CAM_m-{:.3f}'.format(args.m))
PRED_PATH = os.path.join(BASE_PATH,Model,'Prediction_m-{:.3f}'.format(args.m))
CHECKPOINT_PATH = os.path.join(BASE_PATH,Model,'Checkpoint_m-{:.3f}'.format(args.m))

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

if not os.path.exists(SUMMARY_PATH):
    os.makedirs(SUMMARY_PATH)

if not os.path.exists(CAM_PATH):
    os.makedirs(CAM_PATH)

if not os.path.exists(PRED_PATH):
    os.makedirs(PRED_PATH)

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)


summary_filepath = os.path.join(SUMMARY_PATH, 'Summary.txt')

if args.ExpSum:
    fid = open(summary_filepath,'w')
    fid.close()

##### Initialize Training Operations
TrainOp = util.ShapeNet_IncompleteSup()
TrainOp.SetLearningRate(LearningRate=args.LearningRate, BatchSize=args.batchsize)

##### Define Network
TrainOp.DGCNN_SemiSup(batch_size=args.batchsize, point_num=2048)
##### load base network
# best_filepath = os.path.join(CHECKPOINT_PATH, 'Checkpoint_epoch-{}'.format('best'))
# TrainOp.RestoreCheckPoint(best_filepath)



#### Load Sampled Point Index
save_path = os.path.expanduser(os.path.abspath('./Dataset/ShapeNet/Preprocess/'))
save_filepath = os.path.join(save_path, 'SampIndex_m-{:.3f}.mat'.format(args.m))
tmp = scio.loadmat(save_filepath)

### superpoint id for each point
super_points_list_dir = 'Superpoints/super_points_list_mnfgeo500_lamda0.01_0to12137.npy'
super_points_list = np.load(super_points_list_dir)   #12137,2048


num_int = [1000, 1000, 3000, 5000]
num_sample = 12137
num_sp = 500
num_pts = 2048
##### Select annotated super-points

def Read_xyz2(path, len_line):
    r=open(path)
    last=""
   
    sourceInLine=r.readlines()
    for temp in sourceInLine:
        temp=temp.strip('\n') + ' '
        last+=temp 
        
    arr=last.split(' ') 
    r.close()
    del arr[-1]
    arr = np.array(arr)
    len_arr = int(len(arr)/len_line)
    arr = arr.reshape((len_arr,len_line))
    brr = np.zeros((len(arr), len_line))
    for i in range(len(arr)):
        for j in range(len_line):
            brr[i,j] = float(arr[i,j])
    brr = brr.astype(np.int64)
    return brr

#### number series [0~num_sample*num_sp]
data_wl_pick = 'Dataset/ShapeNet/Preprocess/sp_wl_pick_list_sp500.xyz' # [num_sample*num_sp]
####
sp_wl_pick_list = Read_xyz2(data_wl_pick, int(num_sample*num_sp)) 
sp_wl_list = np.zeros((num_sample, num_sp), int).reshape(-1) 
sp_wl_list[sp_wl_pick_list.reshape(-1)[:num_int[0]]] = 1
sp_wl_list = sp_wl_list.reshape(num_sample, num_sp) 


##Modify annotations based on superpoints
Loader.SP_SEG(super_points_list, num_sp)  


wl_pts_list = np.zeros((num_sample,num_pts), np.float32)
for i_ins in range(num_sample):
    sp_wl_one = sp_wl_list[i_ins]
    sp_one = super_points_list[i_ins]

    for i_sp in range(num_sp):
        if sp_wl_one[i_sp] == 1:
            wl_pts_list[i_ins, sp_one==i_sp] = 1.0
print('wl_pts_list', wl_pts_list.shape, np.sum(wl_pts_list))


for rd in range(0, num_round):

    TrainOp.reset_global_step()
    ##### Start Training Epochs
    if rd == (num_round - 1):
        num_Epoch = 201
    else:
        num_Epoch = args.Epoch

    for epoch in range(0, num_Epoch):

        if args.ExpSum:
            fid = open(summary_filepath, 'a')
        else:
            fid = None


        printout('\n\nstart {:d}-th round {:d}-th epoch at {}\n'.format(rd, epoch, time.ctime()),write_flag = args.ExpSum, fid=fid)

        #### Shuffle Training Data --- close
        sort = Loader.Shuffle_TrainSet()

        #### Train One Epoch
        train_avg_loss, train_avg_acc = TrainOp.TrainOneEpoch(Loader,wl_pts_list)

        printout('\nTrainingSet  Avg Loss {:.4f} Avg Acc {:.2f}%'.format(train_avg_loss, 100 * train_avg_acc), write_flag = args.ExpSum, fid = fid)
        

        #### Evaluate One Epoch
        if epoch % 5 ==0:
            # eval_avg_loss, eval_avg_acc, eval_perdata_miou, eval_pershape_miou = TrainOp.EvalOneEpoch(Loader, Eval)
            eval_avg_loss, eval_avg_acc, eval_perdata_miou, eval_pershape_miou = TrainOp.EvalOneEpoch(Loader, Eval)

            printout('\nEvaluationSet   avg loss {:.2f}   acc {:.2f}%   PerData mIoU {:.3f}%   PerShape mIoU {:.3f}%'.
                format(eval_avg_loss,100*eval_avg_acc, 100*np.mean(eval_perdata_miou), 100*np.mean(eval_pershape_miou)), write_flag = args.ExpSum, fid = fid)

            string = '\nEval PerShape IoU:'
            for iou in eval_pershape_miou:
                string += ' {:.2f}%'.format(100 * iou)
            printout(string, write_flag = args.ExpSum, fid = fid)

        if args.ExpSum:
            fid.close()

        if args.SaveMdl:

            save_filepath = os.path.join(CHECKPOINT_PATH, 'Checkpoint_epoch-{:d}'.format(epoch) + '_round-{:d}'.format(rd))
            best_filename = 'Checkpoint_epoch-{}'.format('best')
            round_filename = os.path.join(CHECKPOINT_PATH, 'Checkpoint_round-{:d}'.format(rd))

            TrainOp.SaveCheckPoint(save_filepath,best_filename,np.mean(eval_perdata_miou))
            TrainOp.SaveOneCheck(save_filepath, round_filename)



    if rd < (num_round-1):
        #### output feature and entropy 
        num_query = num_int[rd+1]
        DIS_FEATURE, ETP_bn, prds = TrainOp.EvalOneEpoch(Loader, Eval, 'train')
        ## feature (num_sample,pts,64)  entropy (num_sample,pts)
        ### active selecting
        sp_wl_list, active_idx, entropy_list, shape_scores = strategy.diversity_uncertainty_shape(num_sample, num_sp, super_points_list, sp_wl_list, num_query, DIS_FEATURE, ETP_bn, l_uncer)

        active_sppath = BASE_PATH + '/round_' + np.str(rd) + '_sp_wl_list.xyz'
        np.savetxt(active_sppath,sp_wl_list,fmt='%.d')

        #### annotate pts
        wl_pts_list = np.zeros((12137,2048), np.float32)
        for i_ins in range(num_sample):
            sp_wl_one = sp_wl_list[i_ins]
            sp_one = super_points_list[i_ins]

            for i_sp in range(num_sp):
                if sp_wl_one[i_sp] == 1:
                    wl_pts_list[i_ins, sp_one==i_sp] = 1.0
        print('wl_pts_list', wl_pts_list.shape, np.sum(wl_pts_list))
        print('sum(sp_wl_list)', np.sum(sp_wl_list))



