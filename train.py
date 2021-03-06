import sys
sys.path.append('./lib/')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ts_cluster
import parse

tool = ts_cluster.ts_cluster(10)

def training(data, cut_size=150, sample_size=20):
    dictionary = []
    testing = []
    feature = []

    # Create dictionary and testing data
    for i in xrange(len(data)):
        print 'Data records: ' + str(len(data[i]))
        tmp = Cut(FuzzyDirection(data[i]), cut_size)[:-1]
        sample_idx, test_idx = Sampling(tmp, sample_size)
        dictionary.extend(tmp[sample_idx])
        testing.append(tmp[test_idx])

    # Create Features
    for i in xrange(len(data)):
        print 'Create Feature ' + str(i)
        feature.append(CreateDTWFeature(dictionary, testing[i]))

    return feature

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

def Ploting3D(data, n_dimension=3):
    pca = PCA(n_components = n_dimension)
    colors = ['r', 'g', 'b', 'm']
    labels = ['label_1', 'label_2', 'label_3', 'label_4']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    idx = [0, len(data[0])]
    combined = np.array(data[0])

    # Combined all data
    for i in xrange(1, len(data)):
        combined = np.insert(combined, len(combined), data[i], axis=0)
        idx.append(idx[i]+len(data[i]))

    combined = pca.fit_transform(combined)

    for i in xrange(len(data)):
        ax.scatter(combined[idx[i]:idx[i+1], 0], combined[idx[i]:idx[i+1], 1], combined[idx[i]:idx[i+1], 2], c=colors[i], marker='o', s=70)


    ax.set_xlabel('1st_component')
    ax.set_ylabel('2nd_component')
    ax.set_zlabel('3rd_component')

    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-60, 50)
    ax.set_zlim3d(-60, 50)

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
    features = lambda sample_data, test_data: map(lambda ts_test: map(lambda ts_sample: tool.DTWDistance(ts_test, ts_sample), sample_data), test_data)

    f = features(sample_data, test_data)

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
    all_idx = xrange(len(data))
    sample_idx =  np.random.choice(all_idx, n, replace=False)
    test_idx = list(set(all_idx) - set(sample_idx))
    return np.array(sample_idx), np.array(test_idx)

def Cut(data, n):
    chunks = []

    for idx in xrange(0, len(data[0]), n):
        chunks.append(data[0][idx: idx+n])

    return np.array(chunks)

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
