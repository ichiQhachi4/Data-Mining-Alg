from k_means import *

dataset_w_l = np.loadtxt("./Datasets/LabeledPoint.txt")
dataset = dataset_w_l[:,0:2]
labels = dataset_w_l[:,2]
centroids, clusters, assignments = k_means(dataset, 4)
show_clusters(centroids, clusters)
show_clusters_with_label(dataset, labels)
