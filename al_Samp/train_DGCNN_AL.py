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
import h5py



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
from strategy_coreset import kcenter_greedy

parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--GPU',type=int,help='GPU to use',default=1)
parser.add_argument('--ExpSum','-es',type=int,help='Flag to indicate if export summary',default=1)    # bool
parser.add_argument('--SaveMdl','-sm',type=int,help='Flag to indicate if save learned model',default=1)    # bool
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=3e-3)
parser.add_argument('--Epoch','-ep',type=int,help='Number of epochs to train [default: 51]',default=6)
parser.add_argument('--gamma',type=float,help='L2 regularization coefficient',default=0.1)
parser.add_argument('--batchsize',type=int,help='Training batchsize [default: 1]',default=6)
parser.add_argument('--m','-m',type=float,help='the ratio/percentage of points selected to be labelled (0.01=1%, '
                                               '0.05=5%, 0.1=10%, 1=100%)[default: 0.01]',default=0.01)

parser.add_argument('--NUM_ROUND',type=int,help='SAMPLE NUM_ROUND',default=10)
parser.add_argument('--NUM_QUERY',type=int,help='SAMPLE NUM_QUERY',default=20)
parser.add_argument('--log_name',type=str,help='log_name',default='gary')
parser.add_argument('--L_div',type=float,help='lambda diversity',default=1.0)
parser.add_argument('--L_uncer',type=float,help='lambda uncertainty',default=0.1)
parser.add_argument('--query_method',type=str,help='query_method',default='gary')

args = parser.parse_args()

num_round = args.NUM_ROUND
num_query = args.NUM_QUERY
query_method = args.query_method
if args.GPU != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)

#### Parameters
Model = 'DGCNN_RandomSamp'
l_div = args.L_div
l_uncer = args.L_uncer
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


# adj = tf_util.pairwise_distance(whole_train_data)
# nn_idx = tf_util.knn(adj, k=20)   #b*n*20


#np.random.shuffle(pts_idx_list)
#ori_pts_idx_list = pts_idx_list



for rd in range(1, num_round):
    #print('Round {}'.format(rd))

    ##### Start Training Epochs
    for epoch in range(0,args.Epoch):
    #for epoch in range(0,1):
        if args.ExpSum:
            fid = open(summary_filepath, 'a')
        else:
            fid = None

        printout('\n\nstart {:d}-th round {:d}-th epoch at {}\n'.format(rd, epoch, time.ctime()),write_flag = args.ExpSum, fid=fid)

        #### Shuffle Training Data --- close
        sort = Loader.Shuffle_TrainSet()

        #### Train One Epoch
        train_avg_loss, train_avg_acc = TrainOp.TrainOneEpoch(Loader,file_idx_list,data_idx_list,pts_idx_list)

        printout('\nTrainingSet  Avg Loss {:.4f} Avg Acc {:.2f}%'.format(train_avg_loss, 100 * train_avg_acc), write_flag = args.ExpSum, fid = fid)
        
    
        #### Evaluate One Epoch
        if epoch % 5 ==0:
            eval_avg_loss, eval_avg_acc, eval_perdata_miou, eval_pershape_miou = TrainOp.EvalOneEpoch(Loader, Eval, 'val')

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

            TrainOp.SaveCheckPoint(save_filepath,best_filename,np.mean(eval_pershape_miou))
            TrainOp.SaveOneCheck(save_filepath, round_filename)

    ##adv
    if query_method == 'adversarial':
        max_iter = 50
        whole_adv_noise_dis = TrainOp.EvalAdversarialOneEquery(Loader, Eval, num_query, pts_idx_list, sort, 'train', max_iter)
        dataOri = BASE_PATH + '/whole_adv_noise_dis.mat'
        scio.savemat(dataOri, {'whole_adv_noise_dis':whole_adv_noise_dis[100]})
        new_idx_list = np.sort(whole_adv_noise_dis,axis=1)[:,:num_query]  #12137*num_query
        new_pts_idx_list = np.concatenate((pts_idx_list,new_idx_list),axis=1)
        
    elif query_method == 'adversarial_edges':
        max_iter = 10
        #UNFINISHED
        whole_adv_dis = TrainOp.EvalAdvEdgesEquery(Loader, Eval, num_query, pts_idx_list, index_knn, 'train', max_iter)
        new_idx_list = np.sort(whole_adv_noise_dis,axis=1)[:,:num_query]  #12137*num_query
        new_pts_idx_list = np.concatenate((pts_idx_list,new_idx_list),axis=1)     
        nn_idx 

    elif query_method == 'inputspc_coreset':
        #inputs space distance
        dis_feature = whole_train_data   
        #dis_feature = dis_feature[np.argsort(sort)]
        new_pts_idx_list = np.zeros([len(pts_idx_list),len(pts_idx_list[0,:]) + num_query], dtype=int)
        i, i_b, i_e = 0,0,100
        while i_e< len(dis_feature):
            new_pts_idx_list[i_b:i_e] = strategy.kcenter_greedy(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e])
            i_b += 100; i_e += 100; i+=1; print('i---',i)
        i_e = len(dis_feature)
        new_pts_idx_list[i_b:i_e] = strategy.kcenter_greedy(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e])

    elif query_method == 'feature_coreset':
        dis_feature, loss_seg_bn = TrainOp.EvalOneEpoch(Loader, Eval, 'train')
        new_pts_idx_list = np.zeros([len(pts_idx_list),len(pts_idx_list[0,:]) + num_query], dtype=int)
        i, i_b, i_e = 0,0,100
        while i_e< len(dis_feature):
            new_pts_idx_list[i_b:i_e] = strategy.kcenter_greedy(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e])
            i_b += 100; i_e += 100; i+=1; print('i---',i)
        i_e = len(dis_feature)
        new_pts_idx_list[i_b:i_e] = strategy.kcenter_greedy(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e])
    

    elif query_method == 'uncertainty':
        dis_feature, loss_seg_bn = TrainOp.EvalOneEpoch(Loader, Eval, 'train')
        new_pts_idx_list = np.zeros([len(pts_idx_list),len(pts_idx_list[0,:]) + num_query], dtype=int)
        i, i_b, i_e = 0,0,100
        while i_e< len(dis_feature):
            new_pts_idx_list[i_b:i_e] = strategy.uncertainty_query(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e], loss_seg_bn[i_b:i_e], l_uncer)
            i_b += 100; i_e += 100; i+=1; print('i---',i)
        i_e = len(dis_feature)
        new_pts_idx_list[i_b:i_e] = strategy.uncertainty_query(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e], loss_seg_bn[i_b:i_e], l_uncer)


    #kcenter_greedy(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e])
    #kcenter_greedy_diversity(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e], loss_seg_bn[i_b:i_e], l_div)
    #kcenter_greedy_uncertainty(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e], loss_seg_bn[i_b:i_e], l_uncer)
    #kcenter_graph(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e], loss_seg_bn[i_b:i_e], l_div, l_uncer)
    #uncertainty_query(num_query, dis_feature[i_b:i_e], pts_idx_list[i_b:i_e], loss_seg_bn[i_b:i_e], l_uncer)
    #adversatial_query()
    dataOri = BASE_PATH + '/pts_idx_list.mat'
    scio.savemat(dataOri, {'pts_idx_list_old':ori_pts_ipts_idx_listdx_list})
    dataNew = BASE_PATH + '/new_pts_idx_list.mat'
    scio.savemat(dataNew, {'pts_idx_list':new_pts_idx_list})
    pts_idx_list = new_pts_idx_list