import numpy as np
import tensorflow as tf
import tf_util
import time



def coreset_query(num_query, dis_feature, pts_idx_list):
    q_idxs = 1
    return q_idxs

def kcenter_greedy(num_query, dis_feature, pts_idx_list):
    # lb_flag = np.zeros(len(dis_feature), dtype=bool)
    # for i in range(len(pts_idx_list)):
    #     lb_flag[pts_idx_list[i]] = True

    # dist_mat = np.matmul(dis_feature, dis_feature.transpose())
    # sq = np.array(dist_mat.diagonal()).reshape(len(dist_mat), 1)
    # dist_mat *= -2
    # dist_mat += sq
    # dist_mat += sq.transpose()
    # dist_mat = np.sqrt(dist_mat)
    
    dist_mat, lb_flag = distmat_lbflag(dis_feature, pts_idx_list)
    
    temp_mat = (dist_mat[0])[~(lb_flag[0]), :][:, (lb_flag[0])]
    mat = np.zeros([len(dist_mat),len(temp_mat),len(temp_mat[0,:])])
    q_idxs = np.zeros([len(dis_feature),len(pts_idx_list[0,:]) + num_query], dtype=int)
    for i in range(len(dist_mat)):
        mat[i] = (dist_mat[i])[~(lb_flag[i]), :][:, (lb_flag[i])]
    #mat = dist_mat[~lb_flag, :][:, lb_flag]
    for i in range(num_query):
        mat_min = mat.min(axis=2)
        q_idx_ = mat_min.argmax(axis=1)   
        q_idx = np.zeros(len(dis_feature),dtype=int) #(b,1)
        temp_mat = mat
        mat = np.zeros([len(temp_mat),len(temp_mat[0,:])-1,len(temp_mat[0,0,:])+1])
        for j in range(len(dis_feature)):
            q_idx[j] = np.arange(len(dis_feature[0,:]))[~(lb_flag[j])][q_idx_[j]]
            lb_flag[j,q_idx[j]] = True
            temp_mat2 = np.delete(temp_mat[j], q_idx_[j], 0)
            mat[j] = np.append(temp_mat2, (dist_mat[i])[~(lb_flag[j]), q_idx_[j]][:, None], axis=1)
    for i in range(len(dis_feature)):
        q_idxs[i] = np.arange(len(dis_feature[0,:]))[lb_flag[i]]

    # mat = dist_mat[~lb_flag, :][:, lb_flag]
    # for i in range(num_query):
    #     mat_min = mat.min(axis=1)
    #     q_idx_ = mat_min.argmax()
    #     q_idx = np.arange(len(dis_feature))[~lb_flag][q_idx_]
    #     lb_flag[q_idx] = True
    #     mat = np.delete(mat, q_idx_, 0)
    #     mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
    # q_idxs = np.arange(len(dis_feature))[lb_flag]
    return q_idxs



def kcenter_greedy_sp20(num_query, dis_feature, pts_idx_list, sp_wl_list, super_points_list, num_sp):
    
    dist_mat, lb_flag = distmat_lbflag_one(dis_feature, pts_idx_list)
    
    mat = dist_mat[:, lb_flag]   #n*k
    for i in range(num_query):
        if (lb_flag!=True).all():
            q_idx_sp = np.random.randint(0, high=20)
        else:
            dis_sp = np.zeros(num_sp, np.float32)  #k
            mat_min = np.min(mat, axis=-1)  #n
            for label_sp in range(num_sp):
                if sp_wl_list[label_sp] == 1:
                    dis_sp[label_sp] = 0
                else:
                    temp_spl = np.zeros(2048,np.float32)
                    temp_spl[super_points_list==label_sp] = 1.0   
                    dis_sp[label_sp] = np.sum(mat_min * temp_spl)/np.sum(temp_spl)
            q_idx_sp = dis_sp.argmax()
        sp_wl_list[q_idx_sp] = 1

        lb_flag[super_points_list==q_idx_sp] = True
        mat = dist_mat[:, lb_flag]
        #mat = np.append(mat, dist_mat[:, 2][:, None], axis=1)

    return sp_wl_list


def kcenter_greedy_sp20_2(num_query, dis_feature, pts_idx_list, sp_wl_list, super_points_list, num_sp):
    
    num_ins = dis_feature.shape[0]
    num_poi = dis_feature.shape[1]
    whl_poi = num_ins*num_poi 
    feature_line = np.reshape(dis_feature,(whl_poi,-1))  #wn*k





    for i_query in range(num_query):
        if i_query%100==0:
            print(i_query)

        if i_query==0:

            feature_labeled = []
            for num_i in range(num_ins):
                for num_j in range(num_poi):
                    if pts_idx_list[num_i, num_j]<2048:
                        feature_labeled.append(dis_feature[num_i, pts_idx_list[num_i, num_j]])
            feature_labeled = np.array(feature_labeled)  # num_labeled * k


            # min_mat = np.zeros(whl_poi, np.float32)
            # for i_mat in range(whl_poi):
            #     print('i_query', i_query, 'i_mat', i_mat)
            #     cross_mat_one = np.expand_dims(np.reshape(feature_line[i_mat], (1,-1)), axis=1) - np.expand_dims(feature_labeled, axis=0)
            #     dis_mat_one = np.linalg.norm(cross_mat_one, axis=-1, ord=2) #1 * num_labeled
            #     min_mat[i_mat] = np.min(dis_mat_one)

            # min_mat = np.reshape(min_mat, (num_ins, num_poi))  #num_ins*num_poi


        else:
            feature_labeled = list(feature_labeled)
            lq_ins = np.int32(q_idx_sp/num_sp)
            lq_sp = np.int32(q_idx_sp%num_sp)
            for lq_j in range(num_poi):
                if super_points_list[lq_ins, lq_j] == lq_sp:
                    feature_labeled.append(dis_feature[lq_ins, lq_j])
            feature_labeled = np.array(feature_labeled) 

        cross_mat = np.expand_dims(feature_line, axis=1) - np.expand_dims(feature_labeled, axis=0)
        dis_mat = np.linalg.norm(cross_mat, axis=-1, ord=2) # wn * num_labeled

        min_mat = np.min(dis_mat, axis=-1)  # wn

        dis_sp = np.zeros((num_ins, num_sp), np.float32)
        for num_i in range(num_ins):
            for label_sp in range(num_sp):
                if sp_wl_list[num_i, label_sp] == 1:
                    dis_sp = 0
                else:
                    temp_spl = np.zeros(2048,np.float32)
                    temp_spl[super_points_list[num_i]==label_sp] = 1.0  
                    dis_sp[num_i, label_sp] = np.sum(min_mat * temp_spl)/np.sum(temp_spl)
        dis_sp = np.reshape(dis_sp, (whl_poi, -1)) 
        sp_wl_list = np.reshape(sp_wl_list, (whl_poi, -1)) 
        q_idx_sp = dis_sp.argmax()
        sp_wl_list[q_idx_sp] = 1
        sp_wl_list = np.reshape(sp_wl_list, (num_ins, num_poi)) 



    return sp_wl_list


def kcenter_greedy_sp20_3(num_query, dis_feature, pts_idx_list, sp_wl_list, super_points_list, num_sp):
    num_ins = dis_feature.shape[0]
    num_poi = dis_feature.shape[1]
    feature_dim = dis_feature.shape[2]
    whl_sp = num_ins*num_sp 


    sp_feature = np.zeros((num_ins, num_sp, feature_dim), np.float32)
    for num_i in range(num_ins):
        for label_sp in range(num_sp):
            temp_spl = np.zeros((2048,1),np.float32)
            temp_spl[super_points_list[num_i]==label_sp] = 1.0  
            sp_feature[num_i, label_sp] = np.sum(dis_feature[num_i] * temp_spl, axis=0) / np.sum(temp_spl)
        if num_i%1000==0:
            print('num_i', num_i)
    sp_feature_line = np.reshape(sp_feature,(whl_sp,-1)) 


    for i_query in range(num_query):
        if i_query%100==0:
            print('i_query', i_query)

        if i_query==0:

            sp_labeled = []
            for num_i in range(num_ins):
                for label_sp in range(num_sp):
                    if sp_wl_list[num_i, label_sp] == 1:
                        sp_labeled.append(sp_feature[num_i, label_sp])
            sp_labeled = np.array(sp_labeled)  # num_labeled * k



        else:
            sp_labeled = list(sp_labeled)

            lq_ins = np.int32(q_idx_sp/num_sp)
            lq_sp = np.int32(q_idx_sp%num_sp)

            sp_labeled.append(sp_feature[lq_ins, lq_sp])
            sp_labeled = np.array(sp_labeled) 


        min_mat = np.zeros(whl_sp, np.float32)
        for i_ins in range(num_ins):
            if i_ins%1000==0:
                print('i_ins', i_ins)
            cross_one = np.expand_dims(sp_feature_line[i_ins*num_sp:(i_ins+1)*num_sp], axis=1) - np.expand_dims(sp_labeled, axis=0)
            dis_one = np.linalg.norm(cross_one, axis=-1, ord=2) # wn * num_labeled_sp
            min_one = np.min(dis_one, axis=-1)  # wn
            min_mat[i_ins*num_sp:(i_ins+1)*num_sp] = min_one




        sp_wl_list = np.reshape(sp_wl_list, (whl_sp, -1)) 
        q_idx_sp = min_mat.argmax()
        sp_wl_list[q_idx_sp] = 1
        sp_wl_list = np.reshape(sp_wl_list, (num_ins, num_sp)) 



    return sp_wl_list


def kcenter_greedy_sp(num_query, dis_feature, pts_idx_list, sp_idx_list, super_points_list):
    #12137,2048
    # lb_flag = np.zeros(len(dis_feature), dtype=bool)
    # for i in range(len(pts_idx_list)):
    #     lb_flag[pts_idx_list[i]] = True

    # dist_mat = np.matmul(dis_feature, dis_feature.transpose())
    # sq = np.array(dist_mat.diagonal()).reshape(len(dist_mat), 1)
    # dist_mat *= -2
    # dist_mat += sq
    # dist_mat += sq.transpose()
    # dist_mat = np.sqrt(dist_mat)
    

    new_sp_idx_list = []
    dist_mat, lb_flag = distmat_lbflag_one(dis_feature, pts_idx_list)
    #mat = dist_mat[~lb_flag, :][:, lb_flag]
    mat = dist_mat[:, lb_flag]
    num_sp_label = 100

    for i in range(num_query):
        dis_sp = np.zeros(num_sp_label, np.float32)
        mat_min = mat.min(axis=1)
        for label_sp in range(num_sp_label):
            if label_sp in sp_idx_list:
                dis_sp[label_sp] = 0
            elif label_sp in new_sp_idx_list:
                dis_sp[label_sp] = 0
            else:
                temp_spl = np.zeros(2048,np.float32)
                temp_label = np.ones(2048,int)*label_sp
                temp_spl[temp_label==super_points_list] = 1.0   
                dis_sp[label_sp] = np.sum(mat_min * temp_spl)/np.sum(temp_spl)
        q_idx_sp = dis_sp.argmax()
        new_sp_idx_list.append(q_idx_sp)

        temp_lb_flag = np.ones(2048,int)*q_idx_sp
        lb_flag[temp_lb_flag == super_points_list] = True

        mat = dist_mat[:, lb_flag]
        #mat = np.append(mat, dist_mat[:, 2][:, None], axis=1)

    new_sp_idx_list = np.array(new_sp_idx_list)
    new_sp_idx_list = np.append(sp_idx_list, new_sp_idx_list, axis = 0)



    return new_sp_idx_list


def kcenter_greedy_uncertainty(num_query, dis_feature, pts_idx_list, loss_seg_bn, l_uncer):
    dist_mat, lb_flag = distmat_lbflag(dis_feature, pts_idx_list)

    dist_mat = dist_mat + l_uncer * loss_seg_bn
    temp_mat = (dist_mat[0])[~(lb_flag[0]), :][:, (lb_flag[0])]
    mat = np.zeros([len(dist_mat),len(temp_mat),len(temp_mat[0,:])])
    q_idxs = np.zeros([len(dis_feature),len(pts_idx_list[0,:]) + num_query], dtype=int)
    for i in range(len(dist_mat)):
        mat[i] = (dist_mat[i])[~(lb_flag[i]), :][:, (lb_flag[i])]
    #mat = dist_mat[~lb_flag, :][:, lb_flag]
    for i in range(num_query):
        mat_min = mat.min(axis=2)
        q_idx_ = mat_min.argmax(axis=1)   
        q_idx = np.zeros(len(dis_feature),dtype=int) #(b,1)
        temp_mat = mat
        mat = np.zeros([len(temp_mat),len(temp_mat[0,:])-1,len(temp_mat[0,0,:])+1])
        for j in range(len(dis_feature)):
            q_idx[j] = np.arange(len(dis_feature[0,:]))[~(lb_flag[j])][q_idx_[j]]
            lb_flag[j,q_idx[j]] = True
            temp_mat2 = np.delete(temp_mat[j], q_idx_[j], 0)
            mat[j] = np.append(temp_mat2, (dist_mat[i])[~(lb_flag[j]), q_idx_[j]][:, None], axis=1)
            #mat = np.delete(mat, q_idx_, 0)
            #mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
    for i in range(len(dis_feature)):
        q_idxs[i] = np.arange(len(dis_feature[0,:]))[lb_flag[i]]
    return q_idxs


def distmat_lbflag(dis_feature, pts_idx_list):
    lb_flag = np.zeros([len(dis_feature),len(dis_feature[0,:])], dtype=bool) 
    for i in range(len(pts_idx_list)):
        for j in range(len(pts_idx_list[0,:])):
            lb_flag[i, pts_idx_list[i,j]] = True

    dist_mat = np.matmul(dis_feature, dis_feature.transpose(0,2,1))
    sq = np.array((dist_mat.transpose(1,2,0)).diagonal()).reshape(len(dist_mat),len(dist_mat[0,:]), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose(0,2,1)
    dist_mat = np.sqrt(dist_mat) 
    return dist_mat, lb_flag

def distmat_lbflag_one(dis_feature, pts_idx_list):
    lb_flag = np.zeros(dis_feature.shape[0], dtype=bool) 

    if np.max(pts_idx_list)<2048:
        lb_flag[pts_idx_list] = True

    dist_mat = np.matmul(dis_feature, dis_feature.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(dist_mat.shape[0], 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)

    return dist_mat, lb_flag

def knn_2d(dist_mat, k):
    """Get KNN based on the pairwise distance.
    Args:
        pairwise distance: (batch_size, num_points, num_points)
        k: int

    Returns:
        nearest neighbors: (batch_size, num_points, k)
    """    
    nn_idx = dist_mat.argsort()[:,:,:-1][:,:,0:k]
    return nn_idx

def uncertainty_query(num_query, dis_feature, pts_idx_list, loss_seg_bn, l_uncer):
    #dist_mat, lb_flag = distmat_lbflag(dis_feature, pts_idx_list)
    lb_flag = np.zeros([len(dis_feature),len(dis_feature[0,:])], dtype=bool) 
    #1000,2048
    for i in range(len(pts_idx_list)):
        for j in range(len(pts_idx_list[0,:])):
            lb_flag[i, pts_idx_list[i,j]] = True
            
    dist_mat = np.zeros((len(dis_feature),len(dis_feature[0,:]),len(dis_feature[0,:])), dtype=float)  ##(12137,2048,2048)
    dist_mat = dist_mat + l_uncer * loss_seg_bn

    temp_mat = (dist_mat[0])[~(lb_flag[0]), :][:, (lb_flag[0])]
    mat = np.zeros([len(dist_mat),len(temp_mat),len(temp_mat[0,:])])
    q_idxs = np.zeros([len(dis_feature),len(pts_idx_list[0,:]) + num_query], dtype=int)
    for i in range(len(dist_mat)):
        mat[i] = (dist_mat[i])[~(lb_flag[i]), :][:, (lb_flag[i])]
    #mat = dist_mat[~lb_flag, :][:, lb_flag]
    for i in range(num_query):
        mat_min = mat.min(axis=2)
        q_idx_ = mat_min.argmax(axis=1)   
        q_idx = np.zeros(len(dis_feature),dtype=int) #(b,1)
        temp_mat = mat
        mat = np.zeros([len(temp_mat),len(temp_mat[0,:])-1,len(temp_mat[0,0,:])+1])
        for j in range(len(dis_feature)):
            q_idx[j] = np.arange(len(dis_feature[0,:]))[~(lb_flag[j])][q_idx_[j]]
            lb_flag[j,q_idx[j]] = True
            temp_mat2 = np.delete(temp_mat[j], q_idx_[j], 0)
            mat[j] = np.append(temp_mat2, (dist_mat[i])[~(lb_flag[j]), q_idx_[j]][:, None], axis=1)
            #mat = np.delete(mat, q_idx_, 0)
            #mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
    for i in range(len(dis_feature)):
        q_idxs[i] = np.arange(len(dis_feature[0,:]))[lb_flag[i]]
    return q_idxs

def uncertainty_query_sp(num_query, dis_feature, pts_idx_list, sp_idx_list, super_points_list, dis_uncertainty, l_uncer):
    #dist_mat, lb_flag = distmat_lbflag(dis_feature, pts_idx_list)

    dist_mat = dis_uncertainty   #1n
    num_sp_label = 100

    # temp_spl = np.ones(2048,np.float32)
    # for label_sp in sp_idx_list:
    #     temp_label_sp = np.ones(2048,int)*label_sp
    #     temp_spl[temp_label_sp==super_points_list] = 0
    # dist_mat = dist_mat * temp_spl


    dis_sp = np.zeros(num_sp_label, np.float32)
    for label_sp in range(num_sp_label):
        if label_sp in sp_idx_list:
            dis_sp[label_sp] = 0
        else:
            temp_spl = np.zeros(2048,np.float32)
            temp_label = np.ones(2048,int)*label_sp
            temp_spl[temp_label==super_points_list] = 1.0   
            #aaaa = dist_mat * temp_spl
            #np.savetxt('/data2/lab-shixian/project/ActivePointCloud/aaaa.xyz',aaaa,fmt='%.6f')
            #print('np.sum(dist_mat * temp_spl):',np.sum(dist_mat * temp_spl))
            #print('np.sum(temp_spl):',np.sum(temp_spl))
            dis_sp[label_sp] = np.sum(dist_mat * temp_spl)/np.sum(temp_spl)


    q_idx_sp = np.argsort(dis_sp)[::-1][:num_query]
    new_sp_idx_list = q_idx_sp
    new_sp_idx_list = np.append(sp_idx_list,new_sp_idx_list, axis = 0)


    return new_sp_idx_list

def cal_dis(self, x):
    nx = torch.unsqueeze(x, 0)
    nx.requires_grad_()
    eta = torch.zeros(nx.shape)

    out, e1 = self.clf(nx+eta)
    n_class = out.shape[1]
    py = out.max(1)[1].item()
    ny = out.max(1)[1].item()

    i_iter = 0

    print('out.shpe',out.shape)
    print('out',out)
    print('ny',out.max(1))
    aaa
    while py == ny and i_iter < self.max_iter:
        out[0, py].backward(retain_graph=True)
        grad_np = nx.grad.data.clone()
        value_l = np.inf
        ri = None

        for i in range(n_class):
            if i == py:
                    continue

            nx.grad.data.zero_()
            out[0, i].backward(retain_graph=True)
            grad_i = nx.grad.data.clone()

            wi = grad_i - grad_np
            fi = out[0, i] - out[0, py]
            value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

            if value_i < value_l:
                ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi

        eta += ri.clone()
        nx.grad.data.zero_()
        out, e1 = self.clf(nx+eta)
        py = out.max(1)[1].item()
        i_iter += 1

    return (eta*eta).sum()

def call_dis(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)

def farth_points_sample(pts, K):
    n, w = pts.shape
    farthest_pts_idx = np.zeros(K, dtype=np.int64)
    farthest_pts = np.zeros((K, w), dtype=np.float32)
    farthest_pts_idx[0] = np.random.randint(len(pts))
    farthest_pts[0] = pts[farthest_pts_idx[0]]
    distances = call_dis(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts_idx[i] = np.argmax(distances)
        farthest_pts[i] = pts[farthest_pts_idx[i]]
        distances = np.minimum(distances, call_dis(farthest_pts[i], pts))
    
    return farthest_pts_idx, farthest_pts

def cal_shapescore(num_sample, super_points_list, sp_wl_list, ETP_bn, dis_feature, l_uncer):
    ##shapes_scores
    shape_feas = np.zeros([num_sample,64], np.float64) ### shape level feature
    shape_dscore = np.zeros([num_sample], np.float64) ### shape diversity score
    shape_escore = np.zeros([num_sample], np.float64) ### shape entropy score
    ed_shape_feas = []

    for i_ins in range(num_sample):
        etp_one = ETP_bn[i_ins]
        fea_one = dis_feature[i_ins]  
        sp_one = super_points_list[i_ins]
        shape_feas[i_ins] = np.mean(fea_one, axis=0)   
        shape_escore[i_ins] = np.mean(etp_one)
        if sum(sp_wl_list[i_ins])>0:  
            ed_shape_feas.append(shape_feas[i_ins])  ### annotated sample

    ####shapes_uncertainty_scores
    shape_escore = shape_escore / np.max(shape_escore)
    ####shapes_diversity_scores
    ed_shape_feas = np.expand_dims(np.array(ed_shape_feas), axis=0)   ### 1 k 64
    tp_scores = ((np.expand_dims(np.array(shape_feas), axis=1) - ed_shape_feas)**2).sum(axis=-1)  ### num_sample k
    shape_dscore[:] = np.min(tp_scores, axis=-1)
    shape_dscore = shape_dscore / np.max(shape_dscore)
    ### total shape score
    shape_scores = l_uncer * shape_escore  + (1.0 - l_uncer) * shape_dscore 

    return shape_scores

def diversity_uncertainty_shape(num_sample, num_sp, super_points_list, sp_wl_list, num_query, dis_feature, ETP_bn, l_uncer, l_shape):
    #### superpoint feature and entropy
    entropy_list = np.zeros([num_sample,num_sp], np.float64)  #### superpoint entropy
    feature_list = np.zeros([num_sample,num_sp,64], np.float64)  #### superpoint feature
    for i_ins in range(num_sample):
        etp_one = ETP_bn[i_ins]
        fea_one = dis_feature[i_ins]
        sp_one = super_points_list[i_ins]
        for i_sp in range(num_sp):
            etp_sp = etp_one[sp_one==i_sp]
            fea_sp = fea_one[sp_one==i_sp]
            entropy_list[i_ins, i_sp] = np.mean(etp_sp)
            feature_list[i_ins, i_sp] = np.mean(fea_sp, axis=0)

    #### entropy scores
    entropy_list = entropy_list / np.max(entropy_list)

    ##shapes_scores
    shape_scores = cal_shapescore(num_sample, super_points_list, sp_wl_list, ETP_bn, dis_feature, l_uncer)


    #### Farthest point sampling shortens calculation time for diversity
    K = 10
    far_idx = np.zeros([num_sample,K], np.int64)
    far_feature = np.zeros([num_sample,K,64], np.float64) #### N*K*64
    for i in range(num_sample):
        far_idx[i], far_feature[i] = farth_points_sample(feature_list[i], K)
    
    far_idx = far_idx + np.tile(np.arange(num_sample), (num_sp, 1)).T * num_sp
    far_idx = far_idx.reshape(-1) #### NK
    far_feature = far_feature.reshape(-1, 64) #### NK*64
    ### far_entropy
    far_entropy = entropy_list.reshape(-1)[far_idx]
    ### far_shape_score
    far_shape_score = np.tile(shape_scores, (K, 1)).T.reshape(-1)


    #### annotated points feature
    fea_ed_lst = []
    for i_ins in range(num_sample):
        for i_sp in range(num_sp):
            if sp_wl_list[i_ins, i_sp]==1:
                fea_ed_lst.append(feature_list[i_ins, i_sp])
    fea_ed_wl = np.array(fea_ed_lst)  ### m*64

    #### diversity distance
    dist_mat = np.min(((np.expand_dims(far_feature, axis=1) - np.expand_dims(fea_ed_wl, axis=0)) ** 2).sum(axis=-1), axis=-1)

    #### query
    active_idx = []
    for i in range(num_query):
        one_idx = np.argmax(dist_mat + far_entropy + far_shape_score)
        active_idx.append(far_idx[one_idx])
        dist_mat = np.minimum(dist_mat, call_dis(far_feature[one_idx], far_feature))
    
    #### active annotating
    sp_wl_list[active_idx] = 1
    return sp_wl_list, active_idx, entropy_list, shape_scores






