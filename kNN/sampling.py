from general_functions import *


def simple_random_sampling(dataset, labels, samplesize=0.1):
    # samplesize可以是比例或者个数
    if samplesize >= 1:
        size = int(samplesize)
    else:
        size = int(dataset.shape[0]*samplesize)
    
    samples = []
    sampleslabels = []
    for i in range(size):
        index = int(np.random.uniform(0, dataset.shape[0]))
        samples.append(dataset[index])
        sampleslabels.append(labels[index])
    return np.array(samples), np.array(sampleslabels)

def new_sampling(dataset, seeds, labels, seedlabels, samplesize=0.1):
    # 新的采样方式
    # samplesize可以是比例或者个数
    samples = list(seeds)
    sampleslabels = list(seedlabels)
    if samplesize >= 1:
        size = int(samplesize)
    else:
        size = int(dataset.shape[0]*samplesize)
    
    for i in range(size):
        sample, label = find_another_sample(dataset, samples, labels)
        samples.append(sample)
        sampleslabels.append(label)

    return np.array(samples), np.array(sampleslabels)

def find_another_sample(dataset, samples, labels):
    max_index = 0
    max_dist = 0
    for i in range(dataset.shape[0]):
        dist = find_cluster_dist(np.array(dataset[i]), np.array(samples))
        if dist > max_dist:
            max_dist = dist
            max_index = i
    return dataset[max_index], labels[max_index]
