import sys

from general_functions import *

# random
def init_centroids(dataset, k):
    data_size, dim = dataset.shape
    centriods = np.zeros((k,dim))
    for i in range(k):
        index = int(np.random.uniform(0, data_size))
        centriods[i] = dataset[index]
    return centriods

def k_means(dataset, k):
    centroids = init_centroids(dataset, k)
    print(centroids)
    
    
    changed = True
    
    previous = np.zeros(len(dataset), int)
    while changed:
        print(changed)
        clusters = []
        for i in range(k):
            clusters.append([])
        changed = False
        for i in range(dataset.shape[0]):
            current_data = dataset[i]
            ci = find_cluster_index(current_data, centroids)
            if previous[i] != ci:
                changed = True
                previous[i] = ci
            clusters[ci].append(current_data)

        centroids = np.array(list(map(cal_centroid, clusters)))
        print(centroids)
    
    return centroids, clusters, previous

def show_clusters(centroids, clusters):
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(len(clusters)):
        for point in clusters[i]:
            plt.plot(point[0], point[1], mark[i])
    
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	# draw the centroids
    for i in range(len(clusters)):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)
    plt.show()

