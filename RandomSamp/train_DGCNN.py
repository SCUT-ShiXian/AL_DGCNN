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

sys.path.append(os.path.abspath('./Util'))
sys.path.append(os.path.abspath('./ShapeNet'))



import DataIO_ShapeNet as IO
#print('path',sys.path)
import ShapeNet_DGCNN_util as util
import Tool
from Tool import printout
import Evaluation

parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--GPU',type=int,help='GPU to use',default=1)
parser.add_argument('--ExpSum','-es',type=int,help='Flag to indicate if export summary',default=1)    # bool
parser.add_argument('--SaveMdl','-sm',type=int,help='Flag to indicate if save learned model',default=1)    # bool
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=3e-3)
parser.add_argument('--Epoch','-ep',type=int,help='Number of epochs to train [default: 51]',default=51)
parser.add_argument('--gamma',type=float,help='L2 regularization coefficient',default=0.1)
parser.add_argument('--batchsize',type=int,help='Training batchsize [default: 1]',default=6)
parser.add_argument('--m','-m',type=float,help='the ratio/percentage of points selected to be labelled (0.01=1%, '
                                               '0.05=5%, 0.1=10%, 1=100%)[default: 0.01]',default=0.01)

                                               
parser.add_argument('--log_name',type=str,help='log_name',default='gary')

args = parser.parse_args()

if args.GPU != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)

#### Parameters
Model = 'DGCNN_RandomSamp'

# m = 1    # the ratio of points selected to be labelled

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
#dt = str(datetime.now().strftime("%Y%m%d"))
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
TrainOp.SetLearningRate(LearningRate=args.LearningRate,BatchSize=args.batchsize)

##### Define Network
TrainOp.DGCNN_SemiSup(batch_size=args.batchsize, point_num=2048)

#### Load Sampled Point Index
save_path = os.path.expanduser(os.path.abspath('./Dataset/ShapeNet/Preprocess/'))
save_filepath = os.path.join(save_path, 'SampIndex_m-{:.3f}.mat'.format(args.m))
tmp = scio.loadmat(save_filepath)

pts_idx_list = tmp['pts_idx_list']
file_idx_list = np.zeros(shape=[pts_idx_list.shape[0]])
data_idx_list = np.arange(0,pts_idx_list.shape[0])

##### Start Training Epochs
for epoch in range(0,args.Epoch):

    if args.ExpSum:
        fid = open(summary_filepath, 'a')
    else:
        fid = None

    printout('\n\nstart {:d}-th epoch at {}\n'.format(epoch, time.ctime()),write_flag = args.ExpSum, fid=fid)

    #### Shuffle Training Data
    Loader.Shuffle_TrainSet()

    #### Train One Epoch
    train_avg_loss, train_avg_acc = TrainOp.TrainOneEpoch(Loader,file_idx_list,data_idx_list,pts_idx_list)

    printout('\nTrainingSet  Avg Loss {:.4f} Avg Acc {:.2f}%'.format(
        train_avg_loss, 100 * train_avg_acc), write_flag = args.ExpSum, fid = fid)


    #### Evaluate One Epoch
    if epoch % 5 ==0:
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

        save_filepath = os.path.join(CHECKPOINT_PATH, 'Checkpoint_epoch-{:d}'.format(epoch))
        best_filename = 'Checkpoint_epoch-{}'.format('best')

        TrainOp.SaveCheckPoint(save_filepath,best_filename,np.mean(eval_pershape_miou))
