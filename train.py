import sys
sys.path.append('./lib/')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from mlpy import dtw_std as dtw
from sklearn.svm import LinearSVC 
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ts_cluster
import parse

tool = ts_cluster.ts_cluster(10)

def CreateTestingData(data, slidewindow, cut_size, num, dictionary):
    testing = []
    start = 0
    for i in xrange(num):
        testing.append(data[start:start+cut_size])
        start += slidewindow
    print len(testing)
    return testing
    
def training(data, cut_size=50, ratio = 0.5):
    training_set = []
    testing_set = []
    training_label = []
    testing_label = []
    dictionary = []
    testing_feature = []
    labels = []
    correct = 0

    # Create dictionary and testing data
    for i in xrange(len(data)):
        print 'Data records: ' + str(len(data[i]))
        training_now, testing_now = Slide_Cut_Rand(FuzzyDirection(data[i]), cut_size)
        sample_size = int(len(training_now) * ratio) 
        print sample_size
        #sample_idx, test_idx = Sampling(tmp, sample_size)
        dictionary.extend(Sampling(training_now, sample_size))
        # Get training and testing set
        training_label.extend([i]*len(training_now))
        training_set.append(training_now)
        testing_label.extend([i]*len(testing_now))
        testing_set.append(testing_now)
        #labels.extend([i]*len(tmp))
        #testing.append(tmp)

    print np.array(dictionary).shape
    feature = np.array([[0.0] * len(dictionary)])

    # Create Features
    for i in xrange(len(data)):
        print 'Create Feature ' + str(i)
        #feature.append(CreateDTWFeature(dictionary, testing[i]))
        f = CreateDTWFeature(dictionary, training_set[i])
        feature = np.insert(feature, len(feature), f, axis=0)
        #feature.extend(CreateDTWFeature(dictionary, testing[i]))
        #model_feature.extend(CreateDTWFeature(dictionary, testing[i]))

    #return feature, labels

    feature = np.delete(feature, 0, axis=0)

    
    for i in xrange(len(data)):
        testing_feature.extend(CreateDTWFeature(dictionary, testing_set[i])) 

    #Testing
    svm = LinearSVC()
    svm.fit(feature, training_label)

    for i in range(len(testing_feature)): 
        if svm.predict(testing_feature[i]) == testing_label[i]:
            correct += 1

    return feature, labels, float(correct) / len(testing_feature)
    #return feature, float(correct) / len(testing_feature)
    #return feature
   

def _Ploting(data):
    colors = ['r', 'g', 'b', 'm']
    labels = ['label_1', 'label_2', 'label_3', 'label_4']
    fig = plt.figure()

    for d, i in zip(data, xrange(len(data))):
        plt.plot(d[np.random.randint(len(d))], c=colors[i], label=labels[i])

    plt.xlabel('1st_component')
    plt.ylabel('2nd_component')

    plt.legend()
    plt.show()

def Ploting3D(data, labels, n_dimension=3):
    pca = PCA(n_components = n_dimension)
    colors = ['r', 'g', 'b', 'm']
    labels_text = ['label_1', 'label_2', 'label_3', 'label_4']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    tmp = pca.fit_transform(data)
    print tmp.shape

    map(lambda f, label: ax.scatter(f[0], f[1], f[2], label=labels_text[label], c=colors[label], marker='o', s= 70), tmp, labels)

    print map(lambda f, label: (f[0], f[1], f[2]), tmp, labels)

    ax.set_xlabel('1st_component')
    ax.set_ylabel('2nd_component')
    ax.set_zlabel('3rd_component')

    plt.show()

def Ploting2D(data, n_dimension=2):
    pca = PCA(n_components = n_dimension)
    colors = ['r', 'g', 'b', 'm']
    labels = ['label_1', 'label_2', 'label_3', 'label_4']
    fig = plt.figure()

    idx = [0, len(data[0])]
    combined = np.array(data[0])

    # Combined all data
    for i in xrange(1, len(data)):
        combined = np.insert(combined, len(combined), data[i], axis=0)
        idx.append(idx[i]+len(data[i]))

    combined = pca.fit_transform(combined)

    for i in xrange(len(data)):
        plt.plot(combined[idx[i]:idx[i+1], 0], combined[idx[i]:idx[i+1], 1], colors[i]+'o', markersize=8, label=labels[i])

    plt.xlabel('1st_component')
    plt.ylabel('2nd_component')

    plt.legend(numpoints=1)
    plt.show()

def FuzzyDirection(data):
    pca = PCA(n_components=1)
    tmp = np.array([data['Axis1'], data['Axis2'], data['Axis3']])

    return pca.fit_transform(tmp.T).T

def CreateDTWFeature(sample_data, test_data):
    features = lambda sample_data, test_data: map(lambda ts_test: map(lambda ts_sample: dtw(ts_test, ts_sample), sample_data), test_data)

    f = features(sample_data, test_data)
    np.array(f).shape
    return f
# Create the BasedDTWDistance features
'''
def CreateDTWFeature(data, sample_idx, test_idx):
    features = lambda data, sample_idx, test_idx: map(lambda ts_test: map(lambda ts_sample: tool.DTWDistance(ts_test, ts_sample), data[sample_idx]), data[test_idx])

    f1 = features(np.array(data[0]), sample_idx, test_idx)
    f2 = features(np.array(data[1]), sample_idx, test_idx)
    f3 = features(np.array(data[2]), sample_idx, test_idx)

    return [f1, f2, f3]
'''
# return index of n-sized samples as dictionary
def Sampling(data, n):
    sample_idx = []

    all_idx = xrange(len(data))

    labels, centers = Kmeans(data, n)
    '''
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(data, kmeans.labels_)
    for mid in kmeans.cluster_centers_:
        sample_idx.append(knn.kneighbors(mid)[0][0])
    '''
    #test_idx = list(set(all_idx) - set(sample_idx))
    '''
    sample_idx =  np.random.choice(all_idx, n, replace=False)
    test_idx = list(set(all_idx) - set(sample_idx))
    '''
    #return np.array(sample_idx), np.array(test_idx)

    return centers


def Kmeans(data, sample_size):
    kmeans = KMeans(sample_size, max_iter=1000)

    kmeans.fit(data)

    return kmeans.labels_, kmeans.cluster_centers_

def Slide_Cut_Rand(data, size):
    chunks = []
    start = 0
    max_idx = len(data[0]) - size
    all_idx = np.random.choice(xrange(max_idx), int(0.8*max_idx), replace=False)
    train_idx = np.random.choice(all_idx, int(0.8*len(all_idx)), replace=False)
    test_idx = list(set(all_idx) - set(train_idx))
    print len(all_idx), len(train_idx), len(test_idx)

    training_set = []
    testing_set = []
    for idx in train_idx:
        training_set.append(data[0][idx:idx+size])
    for idx in test_idx:
        testing_set.append(data[0][idx:idx+size])
    return np.array(training_set), np.array(testing_set)

def Slide_Cut(data, size, slidewindow, num):
    chunks = []
    start = 0
    for idx in xrange(num) :
        chunks.append(data[0][start: start+size])
        start += slidewindow

    return np.array(chunks)

def Cut(data, n):
    chunks = []

    for idx in xrange(0, len(data[0]), n):
        chunks.append(data[0][idx: idx+n])


    return np.array(chunks[:-1])

# Return n-sized chucks
'''
def Cut(data, n):
    axis3_chunks = []

    for idx in xrange(0, len(data), n):
        axis1_chunks.append(list(data['Axis1'][idx:idx+n]))
        axis2_chunks.append(list(data['Axis2'][idx:idx+n]))
        axis3_chunks.append(list(data['Axis3'][idx:idx+n]))

    return axis1_chunks, axis2_chunks, axis3_chunks
'''
def Load(filenames):
    data = []
    for name in filenames:
        print 'Loading ' + name
        data.append(pd.read_csv(name))

    return data

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print 'python <python file> [filenames]'
    filenames = eval(sys.argv[1])

    data = Load(filenames)
    print len(data)
