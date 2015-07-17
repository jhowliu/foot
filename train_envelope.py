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

def Preprocessing(train_data, test_data, cut_size=100, slide_size=100, sample_ratio=0.8, num_std=1):
    trainning = {'data': [], 'label': []}
    testing   = {'data': [], 'label': []}

    features = []

    pca_models = []

    idx = [0]
    # Concate the trainning data
    raw_data = np.zeros((1, 3))
    for d, i in zip(test_data, xrange(len(test_data))):
        idx.append(idx[i] + len(d))
        raw_data = np.insert(raw_data, len(raw_data), d, axis=0)

    raw_data = np.delete(raw_data[:, :3], 0, axis=0)

    # Create the trainning data
    for i in xrange(len(train_data)):
        print 'Number of records: ' + str(len(train_data[i]))
        tmp, model = FuzzyDirection(train_data[i])
        pca_models.append(model)
        #tmp = Alignment_Slide_Cut(tmp[0], cut_size, 1)
        tmp = Slide_Cut(tmp[0], cut_size, slide_size)

        trainning['data'].extend(tmp)
        trainning['label'].extend([i]*len(tmp))

    for i in xrange(len(test_data)):
        print 'Number of records: ' + str(len(test_data[i]))
        tmp, model = FuzzyDirection(test_data[i])
        tmp = Slide_Cut(tmp[0], cut_size, slide_size)

        testing['data'].extend(tmp)

    print 'Create Features'
    trainning['label'] = np.array(trainning['label'])
    trainning['data'] = np.array(trainning['data'])
    features = np.zeros((len(testing['data']), 1))

    print features.shape

    for i in xrange(len(train_data)):
        # Create data with sliding window
        tmp = pca_models[i].transform(raw_data).T
        chunks = np.zeros((1, cut_size))
        for ix in xrange(len(idx[:-1])):
            chunks = np.insert(chunks, len(chunks), Slide_Cut(tmp[0][idx[ix]:idx[ix+1]], cut_size, slide_size), axis=0)
        chunks = np.delete(chunks, 0, axis=0)
        print chunks.shape

        f = envelope(trainning['label'][trainning['label'] == i], trainning['data'][trainning['label']==i], chunks, num_std)
        features = np.insert(features, features.shape[1], f.T, axis=1)

    features = np.delete(features, 0, axis=1)

    print features.shape

    return features, trainning['label']

def Training(trainning_features, labels):
    print trainning_features.shape
    model = LinearSVC()
    model.fit(trainning_features, labels)

    return model

def Evaluation(model, testing_features, testing_labels):
    acc = model.score(testing_features, testing_labels)
    return acc

def Predicting(model, testing_features):
    predicted_label = model.predict(testing_features)
    return predicted_label

def Ploting3D(data, labels, n_dimension=3):
    pca = PCA(n_components = n_dimension)
    colors = ['r', 'g', 'b', 'm', 'k']
    #labels_text = ['label_0', 'label_1', 'label_2', 'label_3', 'label_4']
    labels_text = ['Jhow', 'Terry', 'Tsai', 'Terry']
    labels = np.array(labels)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    tmp = pca.fit_transform(data)
    print tmp.shape

    map(lambda i: ax.scatter(tmp[labels==i, 0], tmp[labels==i, 1], tmp[labels==i, 2], label=labels_text[i], c=colors[i], marker='o', s=55), xrange(len(np.unique(labels))))

    ax.set_xlabel('1st_component')
    ax.set_ylabel('2nd_component')
    ax.set_zlabel('3rd_component')

    plt.legend(scatterpoints=1, ncol=3)
    plt.show()

def FuzzyDirection(data):
    pca = PCA(n_components=1)
    tmp = np.array([data['Axis1'], data['Axis2'], data['Axis3']])
    pca.fit(tmp.T)

    return pca.transform(tmp.T).T, pca

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

# Alignment Cutting
def Alignment_Slide_Cut(data, cut_size, slide_size):
    max_bound = len(data) - cut_size
    bound = np.mean(data) + 1.7*np.std(data)

    # Get the position of peak which is lower than cut_size
    peak_idx = np.where(data > bound)[0][np.where(np.where(data > bound)[0] < max_bound)[0]]
    print peak_idx.shape
    indicies = [peak_idx[i] for i in xrange(0, peak_idx.shape[0], slide_size)]

    chunks = map(lambda idx: data[idx:idx+cut_size], indicies)

    return np.array(chunks[:-1])

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
