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

def kcenter_greedy_diversity(num_query, dis_feature, pts_idx_list, l_div):

    dist_mat, lb_flag = dismat_lbflag(dis_feature, pts_idx_list)

    #knn
    # k=20
    # nn_idx = knn_2d(dist_mat, k)  ##(2048,20)

    # vertices = dis_feature[nn_idx]   ##(2048,20,64)
    # edges = vertices - np.tile(dis_feature[:,np.newaxis,:],(1,k,1)) ##(2048,20,64)
    # average_edges = np.average(edges,axis=1)
    # #print('average_edges',average_edges.shape)
    # dist_edges = np.matmul(average_edges, average_edges.transpose())
    # dist_edges = np.array(dist_edges.diagonal()).reshape(len(dist_edges), 1)

    # dist_mat = np.sqrt(dist_edges) + dist_mat
    # #print('vertices:',vertices.shape,'dist_edges:',dist_edges.shape)
    k=20
    vertices, edges, dist_edges = graph(dist_mat, dis_feature, k)

    dist_mat = dist_mat + l_div * dist_edges + l_uncer * loss_seg_bn
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

def kcenter_graph(num_query, dis_feature, pts_idx_list, loss_seg_bn, l_div, l_uncer):
    check_time1 = time.time()
    dist_mat, lb_flag = distmat_lbflag(dis_feature, pts_idx_list)  ###distance mat
    check_time2 = time.time()
    #print('dist_mat:',dist_mat.shape,'lb_flag',lb_flag.shape,'time',check_time2-check_time1)
    #      (100,2048,2048)            (100,2048)
    k=20
    vertices, edges, dist_edges = graph(dist_mat, dis_feature, k)   ###graph
    #dist_edges = graph(dist_mat, dis_feature, k)
    check_time3 = time.time()
    #print('vertices:',vertices.shape,'edges',edges.shape,'dist_edges',dist_edges.shape,'time',check_time3-check_time2)
    #     (100, 2048, 20, 64)    (100, 2048, 20, 64)     (100, 2048, 1)
    # print('average ','dist_mat:', np.average(dist_mat),'dist_edges:', np.average(dist_edges), 'loss_seg_bn:', np.average(loss_seg_bn),)
    # print('max ','dist_mat:', np.max(dist_mat),'dist_edges:', np.max(dist_edges), 'loss_seg_bn:', np.max(loss_seg_bn),)
    # print('min ','dist_mat:', np.min(dist_mat),'dist_edges:', np.min(dist_edges), 'loss_seg_bn:', np.min(loss_seg_bn),)
    # aaa
    dist_mat = dist_mat + l_div * dist_edges + l_uncer * loss_seg_bn
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
    #q_idxs = np.arange(len(dis_feature))[lb_flag]
    check_time4 = time.time()
    print('\n distmat {:.4f}-s; graph {:.4f}-s; query {:.4f}-s\n'.format(check_time2-check_time1,check_time3-check_time2,check_time4-check_time3))
    
    return q_idxs

def distmat_lbflag(dis_feature, pts_idx_list):
    lb_flag = np.zeros([len(dis_feature),len(dis_feature[0,:])], dtype=bool) 
    #1000,2048
    for i in range(len(pts_idx_list)):
        for j in range(len(pts_idx_list[0,:])):
            lb_flag[i, pts_idx_list[i,j]] = True

    dist_mat = np.matmul(dis_feature, dis_feature.transpose(0,2,1))
    sq = np.array((dist_mat.transpose(1,2,0)).diagonal()).reshape(len(dist_mat),len(dist_mat[0,:]), 1)
    #print('sq:',sq.shape)  #(100,2048,1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose(0,2,1)
    dist_mat = np.sqrt(dist_mat) ##(12137,2048,2048)
    return dist_mat, lb_flag

def graph(dist_mat, dis_feature, k=20):
    k=20   #          (b,2048,64)
    nn_idx = knn_2d(dist_mat, k)  ##(b,2048,20)
    #print('nn_idx:',nn_idx.shape)
    ###
    wrong
    ###
    vertices = (np.reshape(dis_feature,(-1,len(dis_feature[0,0,:]))))[nn_idx]   ##(b,2048,20,64)
    edges = vertices - np.tile(dis_feature[:,:,np.newaxis,:],(1,1,k,1)) ##(b,2048,20,64)
    average_edges = np.average(edges,axis=2)
    #(b,2048,64)
    dist_edges = np.matmul(average_edges, average_edges.transpose(0,2,1))
    dist_edges = np.array((dist_edges.transpose(1,2,0)).diagonal()).reshape(len(dist_edges),len(dist_edges[0,:]), 1)
    return vertices, edges, np.sqrt(dist_edges)
    #return np.sqrt(dist_edges)

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

    #wrong
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





