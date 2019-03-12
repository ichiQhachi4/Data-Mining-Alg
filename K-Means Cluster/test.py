from k_means import *
import pandas as pd


dataset = pd.read_csv(".\Datasets\dataset_data_mining_course.csv",header=None)
dataset = np.array(dataset)
print(dataset.shape)
show_clusters_with_label(dataset[:,0:2], dataset[:,2])