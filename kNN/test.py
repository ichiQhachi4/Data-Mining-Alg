from knn import *
from sampling import *
import pandas as pd

datasetwl = pd.read_csv(".\Datasets\dataset_data_mining_course.csv",header=None)
datasetwl = np.array(datasetwl)
dataset = datasetwl[:,0:2]
labels = datasetwl[:, 2]
seeds = [[5.0,5.0],[-10.0,-10.0]]
seedlabels = [0,1]
# 样本本身的分布
show_clusters_with_label(dataset, labels)
# 随机采样
random_samples, random_sampleslabels = simple_random_sampling(dataset, labels)
show_clusters_with_label(random_samples, random_sampleslabels)
knn(10, dataset, random_samples, random_sampleslabels, labels=labels)
# 新的采样方法（采样过程较慢）
samples, sampleslabels = new_sampling(dataset, seeds, labels, seedlabels)
show_clusters_with_label(samples, sampleslabels)
knn(5, dataset, samples, sampleslabels, labels=labels)