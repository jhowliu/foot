import sys
sys.path.append('./lib/')
from Envelope import envelope 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
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

def training(data, cut_size=150, slide_size=100, sample_ratio=0.8):
    dictionary = {'data': [], 'label': []}
    training   = {'data': [], 'label': []}
    testing    = {'data': [], 'label': []}

    num_std = 1.5

    print 'Size of sliding is ' + str(slide_size)
    print 'Size of cutting is ' + str(cut_size)
    # Create dictionary and testing data
    for i in xrange(len(data)):
        print 'Data records: ' + str(len(data[i]))
        tmp = Slide_Cut(FuzzyDirection(data[i])[0], cut_size, slide_size)
        print 'Length of data after sliding is ' + str(tmp.shape)
        sample_size = int(len(tmp) * sample_ratio)

        sample_idx, testing_idx = Sampling(tmp, sample_size)
        #_, centers = Kmeans(tmp[sample_idx, :], int(sample_size/2))
        #training['data'].extend(centers)
        #training['label'].extend([i]*int(sample_size/2))
        training['data'].extend(tmp[sample_idx])
        training['label'].extend([i]*len(sample_idx))

        testing['data'].extend(tmp[testing_idx, :])
        testing['label'].extend([i]*len(testing_idx))


    print 'Size of Codebook is ' + str(np.array(training['data']).shape)


    # Create Features of training
    print 'Create Features'
    training_feature = envelope(training['label'], training['data'], training['data'], num_std)
    testing_feature  = envelope(training['label'], training['data'], testing['data'], num_std)

    print np.array(training_feature).shape
    print np.array(testing_feature).shape

    # Prediction
    print 'Trainning'
    # one against one
    svm = LinearSVC()
    svm.fit(training_feature, training['label'])
    print 'Predicting'
    acc = svm.score(testing_feature, testing['label'])

    return training_feature, training['label'], acc

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

# return index of n-sized samples as dictionary
def Sampling(data, sample_size):
    all_idx = xrange(len(data))
    sample_idx =  np.random.choice(all_idx, sample_size, replace=False)
    testing_idx = list(set(all_idx) - set(sample_idx))

    return np.array(sample_idx), np.array(testing_idx)

def Kmeans(data, sample_size):
    kmeans = KMeans(sample_size, max_iter=1000)

    kmeans.fit(data)

    return kmeans.labels_, kmeans.cluster_centers_

# Cut timeseries by Slide Window
def Slide_Cut(data, cut_size, slide_size):
    max_bound = len(data) - cut_size
    indicies = xrange(0, max_bound, slide_size)

    chunks = map(lambda idx: data[idx:idx+cut_size], indicies)

    return np.array(chunks)

def Slide_Cut_Rand(data, size, slidewindow, num):
    start = 0
    chunks = []

    for idx in xrange(num):
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
    if len(sys.argv) < 5:
        print 'python <python file> [filename1] [filename2] [filename3] [filename4]'
    name = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    

    data = Load(name)
    acc = []
    label = []
    f = []
    cut_s = []
    for cut_size in range(50,200,50):
        x,y,z = training(data,cut_size)
        f.append(x)
        label.append(y)
        acc.append(z)
        cut_s.append(cut_size)
    print acc
