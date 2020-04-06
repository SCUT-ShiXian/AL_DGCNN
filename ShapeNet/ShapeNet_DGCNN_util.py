import numpy as np
import tensorflow as tf
import os
import time
import Loss
import importlib
import copy
import scipy.sparse as sparse
import sys
import time

sys.path.append(os.path.expanduser('../Util'))
import Tool

import DGCNN_ShapeNet as network

class ShapeNet_IncompleteSup():

    def __init__(self):

        self.bestValCorrect = 0.


    def SetLearningRate(self,LearningRate,BatchSize):

        self.BASE_LEARNING_RATE = LearningRate
        self.BATCH_SIZE = BatchSize
        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_DECAY_RATE = 0.5
        self.DECAY_STEP = 16881 * 20
        self.DECAY_RATE = 0.5
        self.BN_DECAY_DECAY_STEP = float(self.DECAY_STEP*2)
        self.BN_DECAY_CLIP = 0.99


    def get_learning_rate(self):
        learning_rate = tf.train.exponential_decay(
            self.BASE_LEARNING_RATE,  # Base learning rate.
            self.batch * self.BATCH_SIZE,  # Current index into the dataset.
            self.DECAY_STEP,  # Decay step.
            self.DECAY_RATE,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 1e-5)  # CLIP THE LEARNING RATE!!
        return learning_rate

    def get_bn_decay(self):
        bn_momentum = tf.train.exponential_decay(
            self.BN_INIT_DECAY,
            self.batch * self.BATCH_SIZE,
            self.BN_DECAY_DECAY_STEP,
            self.BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_decay = tf.minimum(self.BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    ##### Define DGCNN network for ShapeNet
    def DGCNN_SemiSup(self, batch_size, point_num=2048):
        '''
        DGCNN for incomplete labels as supervision
        :param output_dim:
        :param LearningRate:
        :return:
        '''
        
        ##### Define Network Inputs
        self.X_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, point_num, 3], name='InputPts')  # B*N*3
        self.Y_ph = tf.placeholder(dtype=tf.int32, shape=[batch_size, point_num, 50], name='PartGT')  # B*N*50
        self.Mask_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, point_num], name='Mask')  # B*N
        self.Is_Training_ph = tf.placeholder(dtype=tf.bool, shape=(), name='IsTraining')
        self.Label_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, 16],name='ShapeGT') # B*1*K
        self.knn = tf.placeholder(dtype=tf.int32, shape=[batch_size, point_num, 20], name='knk')  # B*N*K

        


        ##
        self.batch = tf.Variable(0, trainable=False)
        bn_decay = self.get_bn_decay()
        learning_rate = self.get_learning_rate()

        ##### Define DGCNN network
        # self.Z, self.dis_feature = network.get_model_adv_edges(self.X_ph, self.knn, self.Y_ph, self.Label_ph, self.Is_Training_ph, 16, 50, \
        #           batch_size, point_num, 0, bn_decay)
        self.Z, self.dis_feature = network.get_model(self.X_ph, self.Y_ph, self.Label_ph, self.Is_Training_ph, 16, 50, \
                  batch_size, point_num, 0, bn_decay)
        #self.dis_mat 



        # self.Z_exp = tf.exp(self.Z)
        self.Z_prob = tf.nn.softmax(self.Z, axis=-1)

        ## Segmentation Branch
        self.loss_seg_bn = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_ph, logits=self.Z)  # B*N
        self.loss_seg = tf.reduce_sum(self.Mask_ph * self.loss_seg_bn) / tf.reduce_sum(self.Mask_ph)

        ## Final Loss
        self.loss = self.loss_seg

        ##### Define Optimizer
        self.solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,global_step=self.batch)  # initialize solver
        self.saver = tf.train.Saver(max_to_keep=2)
        config = tf.ConfigProto(allow_soft_placement=False)
        config.gpu_options.allow_growth = bool(True)  # Use how much GPU memory
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())





        ##adversarial
        #self.adversarial_noised_X_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, point_num, 3], name='adv_X')
        #self.adv_Y_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, point_num, 50], name='adv_Y')
        #self.adversarial_gradient = tf.placeholder(dtype=tf.float32, shape=[batch_size, point_num, 50], name='adv_gradient')
        #self.Z_adv = self.Z*self.seg_feed
        #self.adv_Y_ph = (tf.tile(tf.reduce_max(self.Y_ph, axis=1, keepdims=True), (1, point_num, 1))).float()

        #print('self.adv_Y_ph',self.adv_Y_ph.dtype)
        #print('self.Z',self.Z.dtype)
        #self.Z_adv = self.Z*self.adv_Y_ph
        
        #self.adv_gradient = tf.GradientTape.gradient(self.Z_adv, self.X_ph)
            # seg_feed = np.max(seg_Onehot_feed,axis=1)  #b*1*50
            # seg_feed = np.tile(seg_feed[:, np.newaxis, :],(1, len(seg_Onehot_feed[0,:]), 1))
            # print('seg_feed',seg_feed.shape,seg_feed.dtype)
            # Z_mb_adv = Z_mb*seg_feed

        self.adv_gradient = tf.gradients(self.Z, self.X_ph)
        #print('self.X_ph',self.X_ph.dtype,self.X_ph.shape)
        #print('self.Z_adv',self.Z_adv.dtype,self.Z_adv.shape)
        #print('self.adv_gradient',self.adv_gradient)
        


        return True

    def TrainOneEpoch(self, Loader, file_idx_list,data_idx_list,pts_idx_list=None):
        '''
        Function to train one epoch
        :param Loader: Object to load training data
        :param samp_idx_list:  A list indicating the labelled points  B*N
        :return:
        '''
        batch_cnt = 1
        data_cnt = 0
        avg_loss = 0.
        avg_acc = 0.
        abc = 0
        while True:

            #### get next batch
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_TrainSet(shuffle_flag = True)
            
            # if abc > 3:
            #     break
            if not SuccessFlag or batch_cnt > np.inf:
                break

            if mb_size < self.BATCH_SIZE:
                continue

            #### Prepare Incomplete Labelled Training Data
            if pts_idx_list is None:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)   # B*N
            else:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)   # B*N
                
                for b_i in range(mb_size):
                    batch_samp_idx = \
                        np.where((file_idx_list==file_idx[b_i]) & (data_idx_list==data_idx[b_i]))[0]    # find the data sample from given file and data no.
                    #print('batch_samp_idx:',batch_samp_idx)
                    #print('file_idx_list:',file_idx_list,file_idx_list.shape,np.max(file_idx_list))
                    #print('data_idx_list:',data_idx_list,data_idx_list.shape)
                    
                    try: Mask_bin[b_i, pts_idx_list[batch_samp_idx][0]] = 1
                    except:
                        continue
                
            #### Convert Shape Category Label to Onehot Encoding
            label_onehot = Tool.OnehotEncode(label[:,0],Loader.NUM_CATEGORIES)

            #### Remove labels for unlabelled points (to verify)
            seg_onehot_feed = Tool.OnehotEncode(seg,50)

            #### Train One Iteration
            _, loss_mb, Z_prob_mb, dis_feature, batch_no = \
                self.sess.run([self.solver, self.loss, self.Z_prob, self.dis_feature, self.batch],
                              feed_dict={self.X_ph: data,
                                         self.Y_ph: seg_onehot_feed,
                                         self.Is_Training_ph: True,
                                         self.Mask_ph: Mask_bin,
                                         self.Label_ph:label_onehot})

            ## Calculate loss and correct rate
            # timer.tic()
            avg_loss = (avg_loss * data_cnt + loss_mb*mb_size) / (data_cnt + mb_size)
            pred = []
            for b_i in range(mb_size):
                shape_label = label[b_i][0]
                iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                Z_prob_b[:,iou_oids] += 1
                pred.append(np.argmax(Z_prob_b,axis=-1))
            pred = np.stack(pred)
            avg_acc = (avg_acc * data_cnt + np.mean(pred==seg)*mb_size) / (data_cnt + mb_size)

            data_cnt += mb_size

            print(
                '\rBatch {:d} TrainedSamp {:d}  Avg Loss {:.4f} Avg Acc {:.2f}%'.format(
                    batch_cnt, data_cnt, avg_loss, 100*avg_acc),
                end='')
            batch_cnt += 1
            abc = abc + 1 

        # return avg_loss, perdata_miou, pershape_miou
        
        return avg_loss, avg_acc

    def TrainOneEpoch_advedges(self, Loader, file_idx_list,index_knn,data_idx_list,pts_idx_list=None):
        '''
        Function to train one epoch
        :param Loader: Object to load training data
        :param samp_idx_list:  A list indicating the labelled points  B*N
        :return:
        '''
        batch_cnt = 1
        data_cnt = 0
        avg_loss = 0.
        avg_acc = 0.
        abc = 0
        while True:

            #### get next batch
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_TrainSet(shuffle_flag = True)
            #print('abc:',abc)
            # if abc > 3:
            #     break
            if not SuccessFlag or batch_cnt > np.inf:
                break

            if mb_size < self.BATCH_SIZE:
                continue

            #### Prepare Incomplete Labelled Training Data
            if pts_idx_list is None:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)   # B*N
            else:
                Mask_bin = np.zeros(shape=[mb_size, data.shape[1]], dtype=np.float32)   # B*N
                
                for b_i in range(mb_size):
                    batch_samp_idx = \
                        np.where((file_idx_list==file_idx[b_i]) & (data_idx_list==data_idx[b_i]))[0]    # find the data sample from given file and data no.
                    #print('batch_samp_idx',batch_samp_idx)
                    #print('pts_idx_list[batch_samp_idx][0]',pts_idx_list[batch_samp_idx][0].shape,pts_idx_list[batch_samp_idx][0])
                    
                    try: Mask_bin[b_i, pts_idx_list[batch_samp_idx][0]] = 1
                    except:
                        continue

            #### Convert Shape Category Label to Onehot Encoding
            label_onehot = Tool.OnehotEncode(label[:,0],Loader.NUM_CATEGORIES)

            #### Remove labels for unlabelled points (to verify)
            seg_onehot_feed = Tool.OnehotEncode(seg,50)

            #### Train One Iteration
            _, loss_mb, Z_prob_mb, dis_feature, batch_no = \
                self.sess.run([self.solver, self.loss, self.Z_prob, self.dis_feature, self.batch],
                              feed_dict={self.X_ph: data,
                                         self.Y_ph: seg_onehot_feed,
                                         self.Is_Training_ph: True,
                                         self.Mask_ph: Mask_bin,
                                         self.Label_ph:label_onehot,
                                         self.knn: index_knn})

            ## Calculate loss and correct rate
            # timer.tic()
            avg_loss = (avg_loss * data_cnt + loss_mb*mb_size) / (data_cnt + mb_size)
            pred = []
            for b_i in range(mb_size):
                shape_label = label[b_i][0]
                iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                Z_prob_b[:,iou_oids] += 1
                pred.append(np.argmax(Z_prob_b,axis=-1))
            pred = np.stack(pred)
            avg_acc = (avg_acc * data_cnt + np.mean(pred==seg)*mb_size) / (data_cnt + mb_size)

            data_cnt += mb_size

            print(
                '\rBatch {:d} TrainedSamp {:d}  Avg Loss {:.4f} Avg Acc {:.2f}%'.format(
                    batch_cnt, data_cnt, avg_loss, 100*avg_acc),
                end='')
            batch_cnt += 1
            #abc = abc + 1 

        # return avg_loss, perdata_miou, pershape_miou
        
        return avg_loss, avg_acc






    ### Evaluation on validation set function
    def EvalOneEpoch(self, Loader, Eval, TV_FLAG='val'):
        batch_cnt = 1
        samp_cnt = 0
        data_cnt = 0
        shape_cnt = np.zeros(shape=[Loader.NUM_CATEGORIES])
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_acc = 0.
        perdata_miou = 0.
        pershape_miou = np.zeros_like(shape_cnt)
        #n_time = 0
        DIS_FEATURE = np.zeros([12137,2048,1,64])
        LOSS_SEG_BN = np.zeros([12137,2048])
        START_FEATURE = 0
        END_FEATURE = 6
        if TV_FLAG == 'train': Loader.ResetLoader_TrainSet()
        while True:

            ## get next batch
            if TV_FLAG == 'val':
                SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_ValSet()
            else:
                SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_TrainSet(shuffle_flag = True)

            if not SuccessFlag:
                break

            if mb_size < self.BATCH_SIZE:
                data_feed = np.concatenate([data, np.tile(data[np.newaxis, 0, ...], [self.BATCH_SIZE - mb_size, 1, 1])],
                                           axis=0)
                seg_feed = np.concatenate([seg, np.tile(seg[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                seg_Onehot_feed = Tool.OnehotEncode(seg_feed, 50)
                label_feed = np.concatenate([label, np.tile(label[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                label_Onehot_feed = Tool.OnehotEncode(label_feed[:, 0], Loader.NUM_CATEGORIES)

            else:
                data_feed = data
                seg_Onehot_feed = Tool.OnehotEncode(seg, 50)
                label_Onehot_feed = Tool.OnehotEncode(label[:, 0], Loader.NUM_CATEGORIES)

            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])

            #### Test One Iteration
            loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb = \
                self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z],
                              feed_dict={self.X_ph: data_feed,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed})

            Z_prob_mb = Z_prob_mb[0:mb_size, ...]
            #print('Z_prob_mb:',Z_prob_mb.shape)

            if TV_FLAG == 'val':
            #### Evaluate Metrics
                for b_i in range(mb_size):
                    ## Prediction
                    shape_label = label[b_i][0]
                    iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                    Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                    #print('Z_prob_b---',Z_prob_b.shape)  #(2048,50)
                    
                    Z_prob_b[:, iou_oids] += 1
                    pred = np.argmax(Z_prob_b, axis=-1)
                    ## IoU
                    avg_iou = Eval.EvalIoU(pred, seg[b_i], iou_oids)
                    perdata_miou = (perdata_miou * data_cnt + avg_iou) / (data_cnt + 1)
                    tmp = (pershape_miou[shape_label] * shape_cnt[shape_label] + avg_iou) / (shape_cnt[shape_label] + 1)
                    pershape_miou[shape_label] = tmp
                    ## Accuracy
                    avg_acc = (avg_acc * data_cnt + np.mean(pred == seg[b_i])) / (data_cnt + 1)
                    ## Loss
                    avg_loss = (avg_loss * data_cnt + loss_mb) / (data_cnt + 1)

                    data_cnt += 1
                    shape_cnt[shape_label] += 1

                print('\rEvaluatedSamp {:d}  Avg Loss {:.4f}  Avg Acc {:.3f}%  Avg IoU {:.3f}%'.format(
                    data_cnt, avg_loss, 100 * avg_acc, 100 * np.mean(pershape_miou)), end='')
            else:
                #n_time += 1
                #if n_time%100 == 0: print('n_time',n_time)
                DIS_FEATURE[START_FEATURE:END_FEATURE, ...] = dis_feature[0:(END_FEATURE-START_FEATURE)]
                LOSS_SEG_BN[START_FEATURE:END_FEATURE, ...] = loss_seg_bn[0:(END_FEATURE-START_FEATURE)]
                START_FEATURE += 6
                if END_FEATURE == 12132:
                    END_FEATURE +=5
                else:
                    END_FEATURE +=6
                    
        if TV_FLAG == 'val':
            return avg_loss, avg_acc, perdata_miou, pershape_miou
        else:
            DIS_FEATURE = np.squeeze(DIS_FEATURE, axis=2)
            LOSS_SEG_BN = LOSS_SEG_BN[:, : ,np.newaxis]
            return DIS_FEATURE, LOSS_SEG_BN

        ### Evaluation on validation set function







    ### Evaluation on validation set function  #old_version
    def EvalAdversarialOneEquery(self, Loader, Eval, num_query, pts_idx_list, sort, TV_FLAG='val', max_iter=50):

        temp_idx_list = pts_idx_list[np.argsort(sort)]
        index_pts_idx_list = 0

        batch_cnt = 1
        samp_cnt = 0
        data_cnt = 0
        shape_cnt = np.zeros(shape=[Loader.NUM_CATEGORIES])
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_acc = 0.
        perdata_miou = 0.
        pershape_miou = np.zeros_like(shape_cnt)
        #n_time = 0
        DIS_FEATURE = np.zeros([12137,2048,1,64])
        whole_adv_noise_dis = np.zeros([12137,2048])
        LOSS_SEG_BN = np.zeros([12137,2048])
        START_FEATURE = 0
        END_FEATURE = 1
        if TV_FLAG == 'train': Loader.ResetLoader_TrainSet()
        num_gary = 0
        while True:
            if num_gary%10==0:
                print('num_gary:',num_gary); 
            num_gary+=1
            ## get next batch

            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_TrainSet()

            if not SuccessFlag:
                break

            if mb_size < self.BATCH_SIZE:
                data_feed = np.concatenate([data, np.tile(data[np.newaxis, 0, ...], [self.BATCH_SIZE - mb_size, 1, 1])],
                                           axis=0)
                seg_feed = np.concatenate([seg, np.tile(seg[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                seg_Onehot_feed = Tool.OnehotEncode(seg_feed, 50)
                label_feed = np.concatenate([label, np.tile(label[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                label_Onehot_feed = Tool.OnehotEncode(label_feed[:, 0], Loader.NUM_CATEGORIES)

            else:
                data_feed = data
                seg_Onehot_feed = Tool.OnehotEncode(seg, 50)
                label_Onehot_feed = Tool.OnehotEncode(label[:, 0], Loader.NUM_CATEGORIES)


            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])

            adv_noise = np.zeros(data_feed.shape, dtype=np.float32)  #bn3
            adv_noise_dis = np.ones((1,2048),dtype=np.float32)*np.inf
            
            whole_Y_ph = (np.tile((np.max(seg_Onehot_feed, axis=1))[:,np.newaxis,:], (1, 2048, 1))).astype(np.float32)  #bn50

            #### Test One Iteration
            loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb, adv_gradient = \
                self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z, self.adv_gradient],
                              feed_dict={self.X_ph: data_feed + adv_noise,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed})

            Z_prob_mb = Z_prob_mb[0:mb_size, ...];Z_mb = Z_mb[0:mb_size, ...];Z_adv = Z_mb

            grad_tr = ((np.array(adv_gradient)).reshape((mb_size,2048,3)))[0:mb_size, ...]
            Z_adv = Z_adv + (whole_Y_ph-1)*(np.max(Z_adv)-np.min(Z_adv))
            Ori_predict = np.argmax(Z_adv[0],axis=-1);   #n1
            S_gt = np.argmax(seg_Onehot_feed,axis=2)  #
            Z_o_pred = np.max(Z_adv,axis=-1)  #bn1

            seg_adv_sort = np.unique(S_gt[0])
            op_while = 1

            value = np.ones((2048,50),dtype=np.float32)*np.inf
            grad = np.zeros((2048,50,3),dtype=np.float32)

            iter = 0

            while (iter < max_iter) and op_while:
                for partcalss in seg_adv_sort:

                    adv_Y_ph = np.zeros(seg_Onehot_feed.shape, dtype=np.int32)
                    adv_Y_ph[0,:,partcalss] = adv_Y_ph[0,:,partcalss] + 1
                    loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb, adv_gradient = \
                        self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z, self.adv_gradient],
                                    feed_dict={self.X_ph: data_feed + adv_noise,
                                                self.Y_ph: adv_Y_ph,
                                                self.Is_Training_ph: False,
                                                self.Mask_ph: Mask_bin_feed,
                                                self.Label_ph: label_Onehot_feed})                

                    Z_prob_mb = Z_prob_mb[0:mb_size, ...];Z_mb = Z_mb[0:mb_size, ...];Z_adv = Z_mb
                    grad_adv = ((np.array(adv_gradient)).reshape((mb_size,2048,3)))[0:mb_size, ...]
                    grad[:,partcalss,:] = grad_adv[0] - grad_tr[0]; out = Z_adv[0,:,partcalss] - Z_o_pred[0]

                    Z_adv = Z_adv + (whole_Y_ph-1)*(np.max(Z_adv)-np.min(Z_adv))
                    Adv_predict = np.argmax(Z_adv[0],axis=-1)
                                
                    np.seterr(divide='ignore',invalid='ignore')
                    value[:,partcalss] = np.abs(out)/(np.linalg.norm(grad[:,partcalss,:],axis=-1) + 0.00000001)


                value = value + seg_Onehot_feed[0].astype(np.float32)*(np.max(value)-np.min(value))
                adv_sort_value = np.argmin(value,axis=-1)  #2048
                temp_value = value.reshape(-1); temp_grap = grad.reshape(-1,3)
                temp_sort = adv_sort_value + np.arange(2048) * 50


                ri = (temp_value[temp_sort]/np.linalg.norm(temp_grap[temp_sort],axis=-1)).reshape([-1,1]) * temp_grap[temp_sort]
                adv_noise += ri


                Adv_predict[temp_idx_list[index_pts_idx_list]] = Ori_predict[temp_idx_list[index_pts_idx_list]]
                if np.sum((Adv_predict!=Ori_predict)*1)>=num_query:
                    op_while = 0
                adv_noise_dis[0] = np.linalg.norm(adv_noise[0],axis=-1) 
                adv_noise_dis[0] = adv_noise_dis[0] + (Ori_predict==Adv_predict)*1*(np.max(adv_noise_dis)-np.min(adv_noise_dis))

                iter += 1


            whole_adv_noise_dis[START_FEATURE:END_FEATURE, ...] = adv_noise_dis[0:(END_FEATURE-START_FEATURE)]
            START_FEATURE += 1; END_FEATURE += 1

            index_pts_idx_list += 1

        return whole_adv_noise_dis


        ### Evaluation on validation set function




    ### Evaluation on validation set function
    def EvalAdversarialEquery(self, Loader, Eval, num_query, pts_idx_list, TV_FLAG='train', max_iter=50):
        
        nt_pts_idx_list = pts_idx_list
        index_pts_idx_list = 0

        batch_cnt = 1
        samp_cnt = 0
        data_cnt = 0
        shape_cnt = np.zeros(shape=[Loader.NUM_CATEGORIES])
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_acc = 0.
        perdata_miou = 0.
        pershape_miou = np.zeros_like(shape_cnt)
        #n_time = 0
        DIS_FEATURE = np.zeros([12137,2048,1,64])
        whole_adv_noise_dis = np.zeros([12137,2048])
        LOSS_SEG_BN = np.zeros([12137,2048])
        START_FEATURE = 0
        END_FEATURE = 6
        if TV_FLAG == 'train': Loader.ResetLoader_TrainSet()
        num_gary = 0
        while True:
            if num_gary%10==0:
                print('num_gary:',num_gary)
            num_gary+=1

            ## get next batch

            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_TrainSet()

            if not SuccessFlag:
                break

            if mb_size < self.BATCH_SIZE:
                data_feed = np.concatenate([data, np.tile(data[np.newaxis, 0, ...], [self.BATCH_SIZE - mb_size, 1, 1])],
                                           axis=0)
                seg_feed = np.concatenate([seg, np.tile(seg[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                seg_Onehot_feed = Tool.OnehotEncode(seg_feed, 50)
                label_feed = np.concatenate([label, np.tile(label[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                label_Onehot_feed = Tool.OnehotEncode(label_feed[:, 0], Loader.NUM_CATEGORIES)

            else:
                data_feed = data
                seg_Onehot_feed = Tool.OnehotEncode(seg, 50)
                label_Onehot_feed = Tool.OnehotEncode(label[:, 0], Loader.NUM_CATEGORIES)


            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])

            adv_noise = np.zeros(data_feed.shape, dtype=np.float32)  #bn3
            adv_noise_dis = np.ones((6,2048),dtype=np.float32)*np.inf

            whole_Y_ph = (np.tile((np.max(seg_Onehot_feed, axis=1))[:,np.newaxis,:], (1, 2048, 1))).astype(np.float32)  #bn50
            #print('adv_Y_ph',adv_Y_ph.shape,adv_Y_ph.dtype)
            #### Test One Iteration
            loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb, adv_gradient = \
                self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z, self.adv_gradient],
                              feed_dict={self.X_ph: data_feed + adv_noise,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed})

            Z_prob_mb = Z_prob_mb[0:6, ...];Z_mb = Z_mb[0:6, ...];Z_adv = Z_mb

            grad_tr = ((np.array(adv_gradient)).reshape(6,2048,3))[0:6, ...]
            Z_adv = Z_adv + (whole_Y_ph-1)*(np.max(Z_adv)-np.min(Z_adv))
            S_predict = np.argmax(Z_adv,axis=2); Adv_predict = S_predict  #bn1
            S_gt = np.argmax(seg_Onehot_feed,axis=2)  #
            Z_o_pred = np.max(Z_adv,axis=2)  #bn1


            num_iter = 0
            op_while = np.zeros(6,dtype=np.int32)
            FL_while = True
            while (num_iter < max_iter) and FL_while:
                num_iter += 1

                print('num_iter',num_iter)
                #op_while = np.zeros(6,dtype=np.int32)
                op_update = np.zeros(6,dtype=np.int32)
                num_partcalss = np.zeros(6,dtype=np.int32)
                adv_class = np.zeros(6,dtype=np.int32)
                adv_Y_ph = seg_Onehot_feed   #bn50
                for i in range(6): num_partcalss[i] = len(np.unique(S_gt[i]))
                max_num_partcalss = np.max(num_partcalss)
                value = np.ones((6,2048,max_num_partcalss),dtype=np.float32)*np.inf
                grad = np.zeros((6,2048,max_num_partcalss,3),dtype=np.float32)
                adv_sort_value = np.zeros((6,2048),dtype=np.int32)

                for th_num_partcalss in range(max_num_partcalss):

                    for i in range(6):
                        if (op_while[i]==0) & (th_num_partcalss<num_partcalss[i]):
                            seg_adv_sort = np.unique(S_gt[i])  #num_partcalss

                            #check_time3 = time.time()
                            adv_class[i] = seg_adv_sort[th_num_partcalss]
                            adv_Y_ph[i] = 0; adv_Y_ph[i,:,adv_class[i]] += 1

                        else: op_update[i]=1



                    loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb, adv_gradient = \
                        self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z, self.adv_gradient],
                                    feed_dict={self.X_ph: data_feed + adv_noise,
                                                self.Y_ph: seg_Onehot_feed,
                                                self.Is_Training_ph: False,
                                                self.Mask_ph: Mask_bin_feed,
                                                self.Label_ph: label_Onehot_feed})
                    Z_prob_mb = Z_prob_mb[0:6, ...]; Z_mb = Z_mb[0:6, ...]; Z_adv = Z_mb   #bn50
                    grad_ped = ((np.array(adv_gradient)).reshape(6,2048,3))[0:6, ...]; Z_o_adv = Z_adv

                    Z_adv = Z_adv + (whole_Y_ph-1)*(np.max(Z_adv)-np.min(Z_adv))
                    #value = np.zeros((6,2048,max_num_partcalss),dtype=np.float32)
                    #grad = np.zeros((6,2048,max_num_partcalss,3),dtype=np.float32)
                    out = np.zeros((6,2048),dtype=np.float32)
                    

                    
                    for i in range(6):
                        if op_update[i]==0:
                            grad[i,:,th_num_partcalss,:] = grad_ped[i] - grad_tr[i]; out[i] = Z_o_adv[i,:,adv_class[i]] - Z_o_pred[i]
                            Adv_predict[i] = np.argmax(Z_adv[i],axis=-1)
                            
                            np.seterr(divide='ignore',invalid='ignore')
                            value[i,:,th_num_partcalss] = np.abs(out[i])/np.linalg.norm(grad[i,:,th_num_partcalss,:],axis=-1)
                        if th_num_partcalss>=num_partcalss[i]:
                            value[i,:,th_num_partcalss] += np.inf


                #update adv_noise
                ri = np.zeros((6,2048,3),dtype=np.float32)
                for i in range(6):
                    
                    adv_sort_value[i] = np.argmin(value[i],axis=-1)  ##bn

                    temp_sort = adv_sort_value[i] + np.arange(2048) * max_num_partcalss  #n
                    temp_value = value[i].reshape(-1); temp_grap = grad[i].reshape(-1,3)
                    #print('temp_sort',temp_sort.shape,'temp_value',temp_value.shape,'temp_grap',temp_grap.shape)
                    
                    ri[i] = (temp_value[temp_sort]/np.linalg.norm(temp_grap[temp_sort],axis=-1)).reshape([-1,1])* temp_grap[temp_sort]
                    adv_noise[i] += ri[i]
                    

                for i in range(6):
                    for n in range(2048):
                        num_idx_list = i+6*index_pts_idx_list
                        if num_idx_list>= 12137:
                            num_idx_list = 12133
                        if Adv_predict[i,n] in nt_pts_idx_list[num_idx_list]:
                            adv_noise[i,n] = adv_noise[i,n] + np.inf


                    if np.sum((S_predict[i]!=Adv_predict[i])*1)>=num_query:
                        op_while[i] = 1
                if np.sum(op_while)>=6:
                    FL_while = False




            adv_noise_dis = np.linalg.norm(adv_noise,axis=-1)   #bn
            adv_noise_dis = adv_noise_dis + (S_predict==Adv_predict)*np.max((np.max(adv_noise_dis,axis=-1)-np.min(adv_noise_dis,axis=-1)))   #bn



            whole_adv_noise_dis[START_FEATURE:END_FEATURE, ...] = adv_noise_dis[0:(END_FEATURE-START_FEATURE)]
            START_FEATURE += 6
            if END_FEATURE == 12132:
                END_FEATURE +=5
            else:
                END_FEATURE +=6
            index_pts_idx_list += 1
            if index_pts_idx_list>10:
                break

        return whole_adv_noise_dis[sort]





    ### Evaluation on validation set function
    def EvalAdvEdgesEquery(self, Loader, Eval, num_query, pts_idx_list, index_knn, TV_FLAG='train', max_iter=50):
        
        adv_edges_idx = index_knn[:,:,:5]  #bn5


        nt_pts_idx_list = pts_idx_list
        index_pts_idx_list = 0

        batch_cnt = 1
        samp_cnt = 0
        data_cnt = 0
        shape_cnt = np.zeros(shape=[Loader.NUM_CATEGORIES])
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_acc = 0.
        perdata_miou = 0.
        pershape_miou = np.zeros_like(shape_cnt)
        #n_time = 0
        DIS_FEATURE = np.zeros([12137,2048,1,64])
        whole_adv_noise_dis = np.zeros([12137,2048])
        LOSS_SEG_BN = np.zeros([12137,2048])
        START_FEATURE = 0
        END_FEATURE = 6
        if TV_FLAG == 'train': Loader.ResetLoader_TrainSet()
        num_gary = 0
        while True:
            print('num_gary:',num_gary); num_gary+=1

            ## get next batch

            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_TrainSet()

            if not SuccessFlag:
                break

            if mb_size < self.BATCH_SIZE:
                data_feed = np.concatenate([data, np.tile(data[np.newaxis, 0, ...], [self.BATCH_SIZE - mb_size, 1, 1])],
                                           axis=0)
                seg_feed = np.concatenate([seg, np.tile(seg[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                seg_Onehot_feed = Tool.OnehotEncode(seg_feed, 50)
                label_feed = np.concatenate([label, np.tile(label[np.newaxis, 0], [self.BATCH_SIZE - mb_size, 1])], axis=0)
                label_Onehot_feed = Tool.OnehotEncode(label_feed[:, 0], Loader.NUM_CATEGORIES)

            else:
                data_feed = data
                seg_Onehot_feed = Tool.OnehotEncode(seg, 50)
                label_Onehot_feed = Tool.OnehotEncode(label[:, 0], Loader.NUM_CATEGORIES)

            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])


            adv_edges_idx = adv_edges_idx + ((np.arange(6)*2048).repeat(2048*5)).reshape((6,2048,5))  #bn5
            ori_data_edges = (data_feed.reshape((-1,3)))[adv_edges_idx] #b n 5 3
            adv_noise_edges = ori_data_edges


            

            whole_Y_ph = (np.tile((np.max(seg_Onehot_feed, axis=1))[:,np.newaxis,:], (1, 2048, 1))).astype(np.float32)  #bn50
            #print('adv_Y_ph',adv_Y_ph.shape,adv_Y_ph.dtype)
            #### Test One Iteration
            loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb, adv_gradient, Z_adv = \
                self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z, self.adv_gradient, self.Z_adv],
                              feed_dict={self.X_ph: data_feed,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed,
                                         self.adv_Y_ph: whole_Y_ph,
                                         self.ori_data_edges: ori_data_edges,
                                         self.adv_noise_edges:adv_noise_edges})

            Z_prob_mb = Z_prob_mb[0:6, ...];Z_mb = Z_mb[0:6, ...];Z_adv = Z_adv[0:6, ...]
            #Z_o_tr = Z_adv  #bn50


            grad_tr = ((np.array(adv_gradient)).reshape(6,2048,3))[0:6, ...]

            S_predict = np.argmax(Z_adv,axis=2); Adv_predict = S_predict   #bn1
            S_gt = np.argmax(seg_Onehot_feed,axis=2)  #
            Z_o_pred = np.max(Z_adv,axis=2)  #bn1


            num_iter = 0
            op_while = np.zeros(6,dtype=np.int32)
            FL_while = True
            while num_iter < max_iter and FL_while:
                num_iter += 1


                #op_while = np.zeros(6,dtype=np.int32)
                op_update = np.zeros(6,dtype=np.int32)
                num_partcalss = np.zeros(6,dtype=np.int32)
                adv_class = np.zeros(6,dtype=np.int32)
                adv_Y_ph = seg_Onehot_feed   #bn50
                for i in range(6): num_partcalss[i] = len(np.unique(S_gt[i]))
                max_num_partcalss = np.max(num_partcalss)
                value = np.zeros((6,2048,max_num_partcalss),dtype=np.float32)
                grad = np.zeros((6,2048,max_num_partcalss,3),dtype=np.float32)
                adv_sort_value = np.zeros((6,2048),dtype=np.int32)

                for th_num_partcalss in range(max_num_partcalss):

                    for i in range(6):
                        if op_while[i]==0:
                            if th_num_partcalss<num_partcalss[i]:
                                seg_adv_sort = np.unique(S_gt[i])  #num_partcalss

                                #check_time3 = time.time()
                                adv_class[i] = seg_adv_sort[th_num_partcalss]
                                adv_Y_ph[i] = 0; adv_Y_ph[i,:,adv_class[i]] += 1

                            else: op_update[i]=1
                        else: op_update[i]=1



                    loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb, adv_gradient, Z_adv = \
                        self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z, self.adv_gradient, self.Z_adv],
                                    feed_dict={self.X_ph: data_feed + adv_noise,
                                                self.Y_ph: seg_Onehot_feed,
                                                self.Is_Training_ph: False,
                                                self.Mask_ph: Mask_bin_feed,
                                                self.Label_ph: label_Onehot_feed,
                                                self.adv_Y_ph: adv_Y_ph})
                    Z_prob_mb = Z_prob_mb[0:6, ...]; Z_mb = Z_mb[0:6, ...]; Z_adv = Z_adv[0:6, ...]   #bn50
                    grad_ped = ((np.array(adv_gradient)).reshape(6,2048,3))[0:6, ...]; Z_o_adv = Z_adv

                        
                    #value = np.zeros((6,2048,max_num_partcalss),dtype=np.float32)
                    #grad = np.zeros((6,2048,max_num_partcalss,3),dtype=np.float32)
                    out = np.zeros((6,2048),dtype=np.float32)
                    

                    
                    for i in range(6):
                        if op_update[i]==0:
                            grad[i,:,th_num_partcalss,:] = grad_ped[i] - grad_tr[i]; out[i] = Z_o_adv[i,:,adv_class[i]] - Z_o_pred[i]
                            Adv_predict[i] = np.argmax(Z_adv[i],axis=-1)
                            
                            np.seterr(divide='ignore',invalid='ignore')
                            value[i,:,th_num_partcalss] = np.abs(out[i])/np.linalg.norm(grad[i,:,th_num_partcalss,:],axis=-1)
                        if th_num_partcalss>=num_partcalss[i]:
                            value[i,:,th_num_partcalss] += np.inf


                #update adv_noise
                ri = np.zeros((6,2048,3),dtype=np.float32)
                for i in range(6):
                    
                    adv_sort_value[i] = np.argmin(value[i],axis=-1)  ##bn

                    temp_sort = adv_sort_value[i] + np.arange(2048) * max_num_partcalss  #n
                    temp_value = value[i].reshape(-1); temp_grap = grad[i].reshape(-1,3)
                    #print('temp_sort',temp_sort.shape,'temp_value',temp_value.shape,'temp_grap',temp_grap.shape)
                    
                    ri[i] = (temp_value[temp_sort]/np.linalg.norm(temp_grap[temp_sort],axis=-1)).reshape([-1,1])* temp_grap[temp_sort]
                    adv_noise[i] += ri[i]
                    

                for i in range(6):
                    for n in range(2048):
                        num_idx_list = i+6*index_pts_idx_list
                        if num_idx_list>= 12137:
                            num_idx_list = 12133
                        if Adv_predict[i,n] in nt_pts_idx_list[num_idx_list]:
                            Adv_predict[i,n] = S_predict[i,n]


                    if np.sum((S_predict[i]!=Adv_predict[i])*1)>=num_query:
                        op_while[i] = 1
                if np.sum(op_while)>=6:
                    FL_while = False

                #check_time4 = time.time()
                #print('\n query43 {:.4f}-s\n'.format(check_time4-check_time3))
                #print('num_iter---',num_iter)


            adv_noise_dis = np.linalg.norm(adv_noise,axis=-1)   #bn
            adv_noise_dis = adv_noise_dis + (S_predict==Adv_predict)*np.max((np.max(adv_noise_dis,axis=-1)-np.min(adv_noise_dis,axis=-1)))   #bn
            #print(np.max(adv_noise_dis[i]))

            #check_time2 = time.time()
            #print('\n query21 {:.4f}-s\n'.format(check_time2-check_time1))


            whole_adv_noise_dis[START_FEATURE:END_FEATURE, ...] = adv_noise_dis[0:(END_FEATURE-START_FEATURE)]
            START_FEATURE += 6
            if END_FEATURE == 12132:
                END_FEATURE +=5
            else:
                END_FEATURE +=6
            index_pts_idx_list += 1

        return whole_adv_noise_dis






    ### Evaluation on test set function
    def Test(self, Loader, Eval):
        batch_cnt = 1
        samp_cnt = 0
        data_cnt = 0
        shape_cnt = np.zeros(shape=[Loader.NUM_CATEGORIES])
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_acc = 0.
        perdata_miou = 0.
        pershape_miou = np.zeros_like(shape_cnt)

        while True:

            ## get next batch
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextSamp_TestSet()
            #print('data[0,1]',data[0,1])
            
            if not SuccessFlag:
                break

            #### Resample for fixed number of points
            pts_idx = np.arange(0, data.shape[1])
            add_resamp_idx = np.random.choice(pts_idx, 3000 - len(pts_idx), True)
            resamp_idx = np.concatenate([pts_idx, add_resamp_idx])

            data_feed = data[:, resamp_idx, :]


            seg_Onehot_feed = Tool.OnehotEncode(seg[:, resamp_idx], 50)
            label_Onehot_feed = Tool.OnehotEncode(label[:,0], Loader.NUM_CATEGORIES)
            # label_Onehot_feed = Tool.OnehotEncode(np.zeros([data_feed.shape[0]], dtype=int), Loader.NUM_CATEGORIES)
            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])

            #### Test One Iteration
            loss_mb, Z_prob_mb = \
                self.sess.run([self.loss, self.Z_prob],
                              feed_dict={self.X_ph: data_feed,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed})

            Z_prob_mb = Z_prob_mb[:, pts_idx, :]
            
            #### Evaluate Metrics
            for b_i in range(mb_size):
                ## Prediction
                shape_label = label[b_i][0]
                iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                Z_prob_b[:, iou_oids] += 1
                pred = np.argmax(Z_prob_b, axis=-1)
                ## IoU
                # timer.tic()
                avg_iou = Eval.EvalIoU(pred, seg[b_i], iou_oids)
                # timer.toc()
                perdata_miou = (perdata_miou * data_cnt + avg_iou) / (data_cnt + 1)
                tmp = (pershape_miou[shape_label] * shape_cnt[shape_label] + avg_iou) / (shape_cnt[shape_label] + 1)
                pershape_miou[shape_label] = tmp
                ## Accuracy
                avg_acc = (avg_acc * data_cnt + np.mean(pred == seg[b_i])) / (data_cnt + 1)

                ## Loss
                avg_loss = (avg_loss * data_cnt + loss_mb) / (data_cnt + 1)
                # timer.tic()
                # iou, intersect, union = Tool.IoU_detail(pred[np.newaxis,:], seg[b_i][np.newaxis,:], Loader.NUM_PART_CATS)
                # np.mean(iou[0,iou_oids])
                # timer.toc()

                data_cnt += 1
                shape_cnt[shape_label] += 1

            print(
                '\rEvaluatedSamp {:d}  Avg Loss {:.4f}  Avg Acc {:.3f}%  Avg PerData IoU {:.3f}%  Avg PerCat IoU {:.3f}%'.format(
                    data_cnt, avg_loss, 100 * avg_acc, 100 * np.mean(perdata_miou), 100 * np.mean(pershape_miou)),
                end='')

            # string = '\nEval PerShape IoU:'
            # for iou in pershape_miou:
            #     string += ' {:.2f}%'.format(100 * iou)
            # print(string)

            batch_cnt += 1

        return avg_loss, avg_acc, perdata_miou, pershape_miou

    ########## Saving Checkpoint function
    def SaveCheckPoint(self, save_filepath, best_filename, eval_avg_correct_rate):

        save_filepath = os.path.abspath(save_filepath)

        self.saver.save(self.sess, save_filepath)

        filename = save_filepath.split('/')[-1]
        path = os.path.join(*save_filepath.split('/')[0:-1])
        path = '/' + path

        ## Save a copy of best performing model on validation set
        if self.bestValCorrect < np.mean(eval_avg_correct_rate):
            self.bestValCorrect = np.mean(eval_avg_correct_rate)

            src_filepath = os.path.join(path,
                                        '{}.data-00000-of-00001'.format(
                                            filename))
            # trg_filepath = os.path.join(CHECKPOINT_PATH,'Checkpoint_CAM_L2Reg_PartDropout_bestOnValid_epoch-{:d}.data-00000-of-00001'.format(epoch))
            trg_filepath = os.path.join(path,
                                        '{}.data-00000-of-00001'.format(
                                            best_filename))
            command = 'cp {:s} {:s}'.format(src_filepath, trg_filepath)
            os.system(command)

            src_filepath = os.path.join(path,
                                        '{}.index'.format(filename))
            # trg_filepath = os.path.join(CHECKPOINT_PATH,'Checkpoint_CAM_L2Reg_PartDropout_bestOnValid_epoch-{:d}.index'.format(epoch))
            trg_filepath = os.path.join(path,
                                        '{}.index'.format(best_filename))
            command = 'cp {:s} {:s}'.format(src_filepath, trg_filepath)
            os.system(command)

            src_filepath = os.path.join(path,
                                        '{}.meta'.format(filename))
            # trg_filepath = os.path.join(CHECKPOINT_PATH,'Checkpoint_CAM_L2Reg_PartDropout_bestOnValid_epoch-{:d}.meta'.format(epoch))
            trg_filepath = os.path.join(path,
                                        '{}.meta'.format(best_filename))
            command = 'cp {:s} {:s}'.format(src_filepath, trg_filepath)
            os.system(command)

    ########## Restore Checkpoint function
    def RestoreCheckPoint(self, filepath):

        self.saver.restore(self.sess, filepath)



    def SaveOneCheck(self, save_filepath, round_filename):

        save_filepath = os.path.abspath(save_filepath)

        self.saver.save(self.sess, save_filepath)

        filename = save_filepath.split('/')[-1]
        path = os.path.join(*save_filepath.split('/')[0:-1])
        path = '/' + path

        ## Save a copy of best performing model on validation set
        if True:
            

            src_filepath = os.path.join(path,
                                        '{}.data-00000-of-00001'.format(
                                            filename))
            # trg_filepath = os.path.join(CHECKPOINT_PATH,'Checkpoint_CAM_L2Reg_PartDropout_bestOnValid_epoch-{:d}.data-00000-of-00001'.format(epoch))
            trg_filepath = os.path.join(path,
                                        '{}.data-00000-of-00001'.format(
                                            round_filename))
            command = 'cp {:s} {:s}'.format(src_filepath, trg_filepath)
            os.system(command)

            src_filepath = os.path.join(path,
                                        '{}.index'.format(filename))
            # trg_filepath = os.path.join(CHECKPOINT_PATH,'Checkpoint_CAM_L2Reg_PartDropout_bestOnValid_epoch-{:d}.index'.format(epoch))
            trg_filepath = os.path.join(path,
                                        '{}.index'.format(round_filename))
            command = 'cp {:s} {:s}'.format(src_filepath, trg_filepath)
            os.system(command)

            src_filepath = os.path.join(path,
                                        '{}.meta'.format(filename))
            # trg_filepath = os.path.join(CHECKPOINT_PATH,'Checkpoint_CAM_L2Reg_PartDropout_bestOnValid_epoch-{:d}.meta'.format(epoch))
            trg_filepath = os.path.join(path,
                                        '{}.meta'.format(round_filename))
            command = 'cp {:s} {:s}'.format(src_filepath, trg_filepath)
            os.system(command)