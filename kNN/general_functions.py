import numpy as np
import matplotlib.pyplot as plt

# Euclidean Distance
# 欧式距离
def euc_dist(x, y):
    if isinstance(x, np.ndarray) == False or isinstance(y, np.ndarray) == False:
        x = np.array(x)
        y = np.array(y)
    if (x.shape != y.shape) :
        return -1
    return np.linalg.norm(x-y)

#Manhattan Distance
#曼哈顿距离
def mht_dist(x, y):
    if isinstance(x, np.ndarray) == False or isinstance(y, np.ndarray) == False:
        x = np.array(x)
        y = np.array(y)
    if (x.shape != y.shape) :
        return -1
    return np.linalg.norm(x-y, ord=1)

# Chebyshev Distance
# 切比雪夫距离
def chbshv_dist(x, y):
    if isinstance(x, np.ndarray) == False or isinstance(y, np.ndarray) == False:
        x = np.array(x)
        y = np.array(y)
    if (x.shape != y.shape) :
        return -1
    return np.linalg.norm(x-y, ord=np.inf)

# cosine
# 夹角余弦
def cos_ang(x, y):
    if isinstance(x, np.ndarray) == False or isinstance(y, np.ndarray) == False:
        x = np.array(x)
        y = np.array(y)
    if (x.shape != y.shape) :
        return -1
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

# 找到最近的距离
def find_cluster_dist(point, centroids):
    min_dist = np.inf
    index = 0
    tmp_dist = 0.0
    for i in range(centroids.shape[0]):
        tmp_dist = euc_dist(point, centroids[i])
        if tmp_dist < min_dist:
            min_dist = tmp_dist
            index = i
    return min_dist

# 计算归属
def find_cluster_index(point, centroids):
    min_dist = np.inf
    index = 0
    tmp_dist = 0.0
    for i in range(centroids.shape[0]):
        tmp_dist = euc_dist(point, centroids[i])
        if tmp_dist < min_dist:
            min_dist = tmp_dist
            index = i
    return index

# 计算质心
def cal_centroid(cluster):
    if cluster == []:
        print("waring")
        return np.array([0.0,0.0])
    return np.mean(cluster, axis=0)

# 显示带标签的样本
def show_clusters_with_label(dataset, labels):
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(dataset.shape[0]):
        plt.plot(dataset[i,0], dataset[i,1],mark[int(labels[i])])
    plt.show()