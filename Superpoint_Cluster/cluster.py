from sklearn.cluster import SpectralClustering 
import open3d as o3d
import numpy as np
import random

color_map = {0:[0.,0.,0.],
         1:[100.,85.,144.],
         2:[255.,0.,0.],
         3:[0.,255.,0.],
         4:[0.,0.,255.],
         5:[255.,255.,0.],
         6:[255.,0.,255.],
         7:[0.,255.,255.],
         8:[214.,39.,50.],
         9:[197.,176.,213.],
         10:[148.,103.,189.],
         11:[182.,84.,63.],
         12:[23.,190.,207.],
         13:[247.,182.,210.],
         14:[66.,188.,102.],
         15:[219.,219.,141.],
         16:[140.,57.,197.],
         17:[202.,185.,52.],
         18:[51.,176.,203.],
         19:[200.,54.,131.],
         20:[92.,193.,61.],
         21:[78.,71.,183.],
         22:[172.,114.,82.],
         23:[255.,127.,14.],
         24:[91.,163.,138.],
         25:[153.,98.,156.],
         26:[140.,153.,101.],
         27:[158.,218.,229.]
         }

def compute_eign(pc):
    covariance = np.cov(pc.T)
    eignvalue, eignvector = np.linalg.eig(covariance)
    eignvalue = np.real(eignvalue)
    return eignvalue, eignvector

def kmeans(pc, k):
    num_ps = len(pc)
    tem_idx = np.array(random.sample(range(0,num_ps),int(k)))
    cluster_cen = pc[tem_idx]
    new_cluster_cen = pc[tem_idx]
    cluster_idx = np.zeros(num_ps, np.int64)
    circu = True
    while circu:
        for i_ps in range(num_ps):
            dis_ips = ((pc[i_ps] - cluster_cen) ** 2).sum(axis=1)
            cluster_idx[i_ps] = np.argmin(dis_ips)

        for i_cen in range(k):
            cluster_idata = pc[cluster_idx==i_cen]
            new_cluster_cen[i_cen] = np.mean(cluster_idata, axis=0)

        if (cluster_cen!=new_cluster_cen).any():
            cluster_cen = new_cluster_cen
        else:
            circu = False
    return cluster_cen, cluster_idx

#### Load pc sample
points = np.load('airplane.npy')

# hyperparameters
num_clusters = 100 # the number of clusters. you can try different ones.
num_points = 2048

####eignvalues
eignvalues, eignvector = compute_eign(points)
ev_st = np.sort(eignvalues, axis=-1)  

#### geometry feature :  linerity & planarity & scattering
geo_feature = np.zeros((num_points, 3), np.float32)
geo_feature[0] = (ev_st[2]-ev_st[1])/(ev_st[2] + 1e-7)
geo_feature[1] = (ev_st[1]-ev_st[0])/(ev_st[2] + 1e-7)
geo_feature[2] = ev_st[0]/(ev_st[2] + 1e-7)

### geometry parameter
geo_lambda = 0.2
fpoints = np.concatenate((points, geo_lambda*geo_feature), axis=-1)


##### apply spectral clustering
clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=40, assign_labels='kmeans').fit(fpoints)
superpoints_label = clustering.labels_

### visualize
pc = o3d.geometry.PointCloud()
points_convert = o3d.utility.Vector3dVector(fpoints[:, :3])
pc.points = points_convert

pc_colos = np.zeros((2048,3), np.float32)
for num_p in range(len(pc_colos)):
    pc_colos[num_p] = np.array(color_map[superpoints_label[num_p]%len(color_map)])/255.0

colors_convert = o3d.utility.Vector3dVector(pc_colos)
pc.colors = colors_convert

o3d.io.write_point_cloud('spectral_cluster.ply', pc)


##### apply kmeans clustering
_, kmeans_label = kmeans(fpoints[:, :3], num_clusters)

### visualize
pc = o3d.geometry.PointCloud()
points_convert = o3d.utility.Vector3dVector(fpoints[:, :3])
pc.points = points_convert

pc_colos = np.zeros((2048,3), np.float32)
for num_p in range(len(pc_colos)):
    pc_colos[num_p] = np.array(color_map[kmeans_label[num_p]%len(color_map)])/255.0

colors_convert = o3d.utility.Vector3dVector(pc_colos)
pc.colors = colors_convert

o3d.io.write_point_cloud('kmeans_cluster.ply', pc)

