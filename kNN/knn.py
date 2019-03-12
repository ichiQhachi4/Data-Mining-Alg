from general_functions import *
from collections import Counter

def knn(k, dataset, samples, sampleslabels, labels=None):
    result = []
    i = 0
    for data in dataset:
        distances = calculate_distances(data, samples,sampleslabels)
        sorted_distances = sort_distances(distances)
        label = get_classification(k, sorted_distances)
        result.append(label)
        i += 1
        if i > 300:
            pass
    result = np.array(result)
    datasize =  dataset.shape[0]
    correct = 0
    if labels.any():
        for i in range(datasize):
            if int(labels[i]) == int(result[i]):
                correct += 1
        print("Acuracy: ", correct/datasize)
    show_clusters_with_label(dataset, result)



def calculate_distances(point, samples, sampleslabels,type="euc"):
    
    distances = []
    if type == "euc":
        for i in range(samples.shape[0]):
            distances.append(np.array([euc_dist(point, samples[i]), int(sampleslabels[i])]))
    else:
        pass #todo
    return np.array(distances)

def sort_distances(distances):
    size = distances.shape[0]
    for i in range(1,size):
        for j in range(size-i):
            if distances[j][0] > distances[j+1][0]:
                distances[[j,j+1],:] = distances[[j+1,j],:]
    return distances


def get_classification(k, sorted_distances):
    labels = sorted_distances[0:k, 1]
    belongs = []
    for i in range(k):
        belongs.append(0)
    for i in range(k):
        index = int(labels[i])
        belongs[index] = belongs[index] + 1
    max = 0
    index = 0
    for i in range(k):
        if belongs[i] > max:
            index = i
            max = belongs[i]
    return index