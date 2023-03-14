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


    def reset_global_step(self):
        #self.sess.run(tf.global_variables_initializer())
        op = tf.assign(self.batch, 0)
        self.sess.run([op])
        #self.batch = tf.Variable(0, trainable=False)

    def set_global_step(self, num_step):
        #self.sess.run(tf.global_variables_initializer())
        op = tf.assign(self.batch, num_step)
        self.sess.run([op])
        #self.batch = tf.Variable(0, trainable=False)


    def global_initialization(self):
        self.sess.run(tf.global_variables_initializer())


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

        ##
        self.batch = tf.Variable(0, trainable=False)
        bn_decay = self.get_bn_decay()
        self.temp_bn_decay = bn_decay
        learning_rate = self.get_learning_rate()


        ##### Define DGCNN network
        self.Z, self.dis_feature = network.get_model(self.X_ph, self.Y_ph, self.Label_ph, self.Is_Training_ph, 16, 50, \
                  batch_size, point_num, 0, bn_decay)


        # self.Z_exp = tf.exp(self.Z)
        self.Z_prob = tf.nn.softmax(self.Z, axis=-1)

        ## Segmentation Branch
        self.loss_seg_bn = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_ph, logits=self.Z)  # B*N
        self.loss_seg = tf.reduce_sum(self.Mask_ph * self.loss_seg_bn) / (tf.reduce_sum(self.Mask_ph) + 1e-7)


        ## Final Loss
        self.loss = self.loss_seg

        ##### Define Optimizer
        self.solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,global_step=self.batch)  # initialize solver
        self.saver = tf.train.Saver(max_to_keep=2)
        config = tf.ConfigProto(allow_soft_placement=False)
        config.gpu_options.allow_growth = bool(True)  # Use how much GPU memory
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.adv_gradient = tf.gradients(self.loss, self.X_ph)

        return True




    def TrainOneEpoch(self, Loader,wl_pts_list=None):
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

        while True:

            #### get next batch
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx, sp_list = Loader.NextBatch_TrainSet(shuffle_flag = True)

            batch_pts = wl_pts_list[data_idx]

            if not SuccessFlag or batch_cnt > np.inf:
                break

            if mb_size < self.BATCH_SIZE:
                continue

            Mask_bin = batch_pts


            #### Convert Shape Category Label to Onehot Encoding
            label_onehot = Tool.OnehotEncode(label[:,0],Loader.NUM_CATEGORIES)

            #### Remove labels for unlabelled points (to verify)
            seg_onehot_feed = Tool.OnehotEncode(seg,50)


            #### Train One Iteration
            if np.sum(Mask_bin)>0:
                _, loss_mb, Z_prob_mb, batch_no = \
                    self.sess.run([self.solver, self.loss, self.Z_prob, self.batch],
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

        return avg_loss, avg_acc


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
        DIS_FEATURE = np.zeros((12137,2048,1,64), np.float32)
        LOSS_Z = np.zeros((12137,2048), np.float32)
        Z_PB_bn = np.zeros((12137,2048), np.float32)
        Z_prob_wl = np.zeros((12137,2048,50), np.float32)
        prd_seg = np.zeros((12137,2048), np.float32)
        START_FEATURE = 0
        NUM_EACH = self.BATCH_SIZE
        END_FEATURE = NUM_EACH
        if TV_FLAG == 'train': Loader.ResetLoader_TrainSet()
        while True:

            ## get next batch
            if TV_FLAG == 'val':
                SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextBatch_ValSet()
            else:
                SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx, sp_list = Loader.NextBatch_TrainSet(shuffle_flag = True)

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
            
            log_Z = - Z_prob_mb * np.log(Z_prob_mb + 1e-7)  #bn50  
            Z_entropy = np.sum(log_Z, axis = -1)   #bn  ##entropy


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
                    prd_seg[data_idx[b_i]] = pred
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
                DIS_FEATURE[data_idx] = dis_feature[0:(END_FEATURE-START_FEATURE)]
                Z_PB_bn[data_idx] = Z_entropy[0:(END_FEATURE-START_FEATURE)]
                LOSS_Z[data_idx] = loss_seg_bn[0:(END_FEATURE-START_FEATURE)]
                Z_prob_wl[data_idx] = Z_prob_mb
                START_FEATURE += NUM_EACH
                # if END_FEATURE == 12132:
                #     END_FEATURE = END_FEATURE + NUM_EACH - 1
                # else:
                #     END_FEATURE += NUM_EACH
                END_FEATURE += NUM_EACH
                print('END_FEATURE', END_FEATURE)
                # START_FEATURE += NUM_EACH
                # END_FEATURE += NUM_EACH
                    
        if TV_FLAG == 'val':
            return avg_loss, avg_acc, perdata_miou, pershape_miou
        else:
            DIS_FEATURE = np.squeeze(DIS_FEATURE, axis=2)
            #LOSS_SEG_BN = LOSS_SEG_BN[:, : ,np.newaxis]
            return DIS_FEATURE, Z_PB_bn, Z_prob_wl

        ### Evaluation on validation set function





    ### Evaluation on test set function
    def Test_traindata(self, Loader, Eval):
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
            #SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextSamp_TestSet()
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx, sp_seg_multi, click_seg_multi, click_seg_multi_TF, sp_list, sp_wl_list = Loader.NextBatch_TrainSet()
            #print('data[0,1]',data[0,1])
            
            if not SuccessFlag:
                break

            #### Resample for fixed number of points
            pts_idx = np.arange(0, data.shape[1])
            #add_resamp_idx = np.random.choice(pts_idx, 3000 - len(pts_idx), True)
            #resamp_idx = np.concatenate([pts_idx, add_resamp_idx])

            data_feed = data[:]


            seg_Onehot_feed = Tool.OnehotEncode(seg[:], 50)
            label_Onehot_feed = Tool.OnehotEncode(label[:,0], Loader.NUM_CATEGORIES)
            # label_Onehot_feed = Tool.OnehotEncode(np.zeros([data_feed.shape[0]], dtype=int), Loader.NUM_CATEGORIES)
            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])






            #### Test One Iteration
            num_sp_label = what
            YM_ph = np.zeros((Mask_bin_feed.shape[0], num_sp_label, 50), np.float32)  ##bmk
            Mask_multi_sp = np.zeros((Mask_bin_feed.shape[0], Mask_bin_feed.shape[1], num_sp_label), np.float32)  #bnm

            loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb = \
                self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z],
                              feed_dict={self.X_ph: data_feed,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed,
                                         self.Y_multi_ph: YM_ph,
                                         self.Mask_multi_sp_ph: Mask_multi_sp})



            Z_prob_mb = Z_prob_mb[:, pts_idx, :]
            
            #### Evaluate Metrics
            for b_i in range(mb_size):
                ## Prediction
                shape_label = label[b_i][0]
                iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                Z_prob_b[:, iou_oids] += 1
                pred = np.argmax(Z_prob_b, axis=-1)

                pred = pred[click_seg_multi_TF[b_i]==1]
                seg_multi = seg[b_i][click_seg_multi_TF[b_i]==1]
                ## IoU
                # timer.tic()
                #avg_iou = Eval.EvalIoU(pred, seg[b_i], iou_oids)
                if len(pred)>0:
                    
                    avg_iou = Eval.EvalIoU(pred, seg_multi, iou_oids)
                    # timer.toc()
                    perdata_miou = (perdata_miou * data_cnt + avg_iou) / (data_cnt + 1)
                    tmp = (pershape_miou[shape_label] * shape_cnt[shape_label] + avg_iou) / (shape_cnt[shape_label] + 1)
                    pershape_miou[shape_label] = tmp
                    ## Accuracy
                    avg_acc = (avg_acc * data_cnt + np.mean(pred == seg_multi)) / (data_cnt + 1)

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




    ### Evaluation on test set function
    def Test(self, Loader, Eval):
        batch_cnt = 1
        samp_cnt = 0
        data_cnt = 0
        shape_cnt = np.zeros(shape=[Loader.NUM_CATEGORIES])
        #print('Loader.NUM_CATEGORIES:',Loader.NUM_CATEGORIES)  16
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_acc = 0.
        perdata_miou = 0.
        pershape_miou = np.zeros_like(shape_cnt)

        min_num = 3000
        test_data_seg = np.zeros((12137, 3000), np.int64)
        test_data = np.zeros((12137, 3000, 3), np.int64)
        while True:

            ## get next batch
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextSamp_TestSet()
            #SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx, sp_seg_multi, click_seg_multi, click_seg_multi_TF, sp_list, sp_wl_list = Loader.NextBatch_TrainSet()
            #print('data[0,1]',data[0,1])
            
            if not SuccessFlag:
                break

            #### Resample for fixed number of points
            pts_idx = np.arange(0, data.shape[1])

            if min_num > len(pts_idx):
                min_num = len(pts_idx)

            add_resamp_idx = np.random.choice(pts_idx, 3000 - len(pts_idx), True)
            resamp_idx = np.concatenate([pts_idx, add_resamp_idx])

            data_feed = data[:, resamp_idx, :]
            

            seg_Onehot_feed = Tool.OnehotEncode(seg[:, resamp_idx], 50)
            label_Onehot_feed = Tool.OnehotEncode(label[:,0], Loader.NUM_CATEGORIES)
            # label_Onehot_feed = Tool.OnehotEncode(np.zeros([data_feed.shape[0]], dtype=int), Loader.NUM_CATEGORIES)
            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])






            #### Test One Iteration

            loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb = \
                self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z],
                              feed_dict={self.X_ph: data_feed,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed})


            Z_prob_mb = Z_prob_mb[:, pts_idx, :]
            # print('Z_prob_mb', Z_prob_mb.shape)
            # print('resamp_idx', resamp_idx.shape)
            #### Evaluate Metrics
            for b_i in range(mb_size):
                ## Prediction
                shape_label = label[b_i][0]
                iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                Z_prob_b[:, iou_oids] += 1
                pred = np.argmax(Z_prob_b, axis=-1)
                test_data_seg[data_cnt] = pred[resamp_idx]
                # test_data_seg[data_cnt] = seg[b_i][resamp_idx]
                test_data[data_cnt] = data_feed
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

        # print('min_num', min_num)
        print('test_data_seg', test_data_seg.shape)
        np.save('Superpoints/vis0907/pts.npy', test_data_seg)
        np.save('Superpoints/vis0907/dataxyz.npy', test_data)
        return avg_loss, avg_acc, perdata_miou, pershape_miou, shape_cnt




    ### Evaluation on test set function
    def Test_vis(self, Loader, Eval):
        batch_cnt = 1
        samp_cnt = 0
        data_cnt = 0
        shape_cnt = np.zeros(shape=[Loader.NUM_CATEGORIES])
        #print('Loader.NUM_CATEGORIES:',Loader.NUM_CATEGORIES)  16
        avg_loss = 0.
        avg_correct_rate = 0.
        avg_acc = 0.
        perdata_miou = 0.
        pershape_miou = np.zeros_like(shape_cnt)

        min_num = 2048
        test_data_seg = np.zeros((2874, 2048), np.int64)
        # test_data = np.zeros((12137, 2048, 3), np.int64)

        import h5py
        def load_h5_data_label_seg(h5_filename):
            f = h5py.File(h5_filename,'r')
            data = f['data'][:]
            label = f['label'][:]
            seg = f['pid'][:]
            num_data = data.shape[0]
            data_idx = np.arange(0,num_data)

            return (data, label, seg, num_data, data_idx)

        def getDataFiles(list_filename):
            return [line.rstrip() for line in open(list_filename)]
        h5_base_path = '/data3/lab-shixian/project/ActivePointCloud/Dataset/ShapeNet/hdf5_data'
        test_file_list = getDataFiles(os.path.join(h5_base_path, 'test_hdf5_file_list.txt'))

        test_data = []
        test_labels = []
        test_seg = []
        num_test = 0
        test_data_idx = []
        for cur_test_filename in test_file_list:
            print('cur_test_filename',cur_test_filename)
            cur_test_data, cur_test_labels, cur_test_seg, cur_num_test, cur_test_data_idx = load_h5_data_label_seg(os.path.join(h5_base_path,cur_test_filename))

            test_data.append(cur_test_data)
            test_labels.append(cur_test_labels)
            test_seg.append(cur_test_seg)
            test_data_idx.append(cur_test_data_idx+num_test)
            num_test += cur_num_test

            whole_test_data = np.concatenate(test_data)
            whole_test_labels = np.concatenate(test_labels)
            whole_test_seg = np.concatenate(test_seg)
            whole_test_data_idx = np.concatenate(test_data_idx)
            whole_num_test = num_test

        print('len(test_data)------',whole_test_data.shape)
        print('len(test_labels)------',whole_test_labels.shape)
        print('len(test_seg)------',whole_test_seg.shape)
        print('len(test_data_idx)------',whole_test_data_idx.shape)
        print('num_test------',whole_num_test)




        while True:

            ## get next batch
            SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx = Loader.NextSamp_TestSet()
            #SuccessFlag, data, label, seg, weak_seg_onehot, mb_size, file_idx, data_idx, sp_seg_multi, click_seg_multi, click_seg_multi_TF, sp_list, sp_wl_list = Loader.NextBatch_TrainSet()
            #print('data[0,1]',data[0,1])
            # print('data', data.shape)
            # print('label', label.shape)
            # print('seg', seg.shape)         
            if data_cnt == 2874:
                break
            data = whole_test_data[data_cnt].reshape(mb_size, 2048, 3)
            label = whole_test_labels[data_cnt].reshape(mb_size, 1)
            seg = whole_test_seg[data_cnt].reshape(mb_size, 2048)
            # print('data', data.shape)
            # print('label', label.shape)
            # print('seg', seg.shape)

            if not SuccessFlag:
                break

            #### Resample for fixed number of points
            pts_idx = np.arange(0, data.shape[1])

            # if min_num > len(pts_idx):
            #     min_num = len(pts_idx)

            # add_resamp_idx = np.random.choice(pts_idx, 3000 - len(pts_idx), True)
            # resamp_idx = np.concatenate([pts_idx, add_resamp_idx])
            # data_feed = data[:, resamp_idx, :]
            data_feed = data
            

            seg_Onehot_feed = Tool.OnehotEncode(seg[:, :], 50)
            label_Onehot_feed = Tool.OnehotEncode(label[:,0], Loader.NUM_CATEGORIES)
            # label_Onehot_feed = Tool.OnehotEncode(np.zeros([data_feed.shape[0]], dtype=int), Loader.NUM_CATEGORIES)
            Mask_bin_feed = np.ones([data_feed.shape[0], data_feed.shape[1]])






            #### Test One Iteration

            loss_mb, loss_seg_bn, Z_prob_mb, dis_feature, Z_mb = \
                self.sess.run([self.loss, self.loss_seg_bn, self.Z_prob, self.dis_feature, self.Z],
                              feed_dict={self.X_ph: data_feed,
                                         self.Y_ph: seg_Onehot_feed,
                                         self.Is_Training_ph: False,
                                         self.Mask_ph: Mask_bin_feed,
                                         self.Label_ph: label_Onehot_feed})


            Z_prob_mb = Z_prob_mb[:, pts_idx, :]
            # print('Z_prob_mb', Z_prob_mb.shape)
            # print('resamp_idx', resamp_idx.shape)
            #### Evaluate Metrics
            for b_i in range(mb_size):
                ## Prediction
                shape_label = label[b_i][0]
                iou_oids = Loader.object2setofoid[Loader.objcats[shape_label]]
                Z_prob_b = copy.deepcopy(Z_prob_mb[b_i])
                Z_prob_b[:, iou_oids] += 1
                pred = np.argmax(Z_prob_b, axis=-1)
                # print('data_cnt', data_cnt)
                test_data_seg[data_cnt] = pred
                # test_data_seg[data_cnt] = seg[b_i][resamp_idx]
                # test_data[data_cnt] = data_feed
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

        # print('min_num', min_num)
        print('test_data_seg', test_data_seg.shape)
        np.save('Superpoints/vis0907/gt.npy', test_data_seg)
        # np.save('Superpoints/vis0907/dataxyz.npy', test_data)
        return avg_loss, avg_acc, perdata_miou, pershape_miou, shape_cnt





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
