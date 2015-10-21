import sys
#sys.path.append('/Users/jhow/Study/Objective-c/udp/foot/lib/')
sys.path.append('/home/jhowliu/Work/foot/lib')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from mlpy import dtw_std as dtw

from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ts_cluster

from Vectorization import Vectorize

tool = ts_cluster.ts_cluster(10)

def CreateTestingData(data, slidewindow, cut_size, num, dictionary):
    testing = []
    start = 0
    for i in xrange(num):
        testing.append(data[start:start+cut_size])
        start += slidewindow
    print len(testing)
    return testing

def build_pca(data):
    pca = PCA(n_components=1)
    tmp = np.array([data['Axis1'].tolist(), data['Axis2'].tolist(), data['Axis3'].tolist()]).T
    print tmp
    return pca

def Train_Preprocessing(train_data, cut_size=150, slide_size=100, sample_ratio=0.8):
    training = {'data': {'Axis1':[], 'Axis2':[], 'Axis3':[]}, 'label': []}
    # First in
    flag = True
    for axis in ["Axis1", "Axis2", "Axis3"]:
        for i in xrange(len(train_data)):
            print 'Number of records: ' + str(len(train_data[i]))
            _, tmp = splitSteps(train_data[i], train_data[i]['Axis1'], axis)
            if flag:
                training['label'].extend([i]*len(tmp))
            training['data'][axis].extend(tmp)
        flag = False

    training_features = np.array([[0.0] * len(training['data']['Axis1'])]).T

    for axis in ["Axis1", "Axis2", "Axis3"]:
        training_features = np.insert(training_features, training_features.shape[1], Vectorize(np.array(training['data'][axis])),axis=1)
    training_features = np.delete(training_features, 0, axis=1)

    print np.array(training_features).shape

    print 'Finishing train features'

    return training_features, training['label']

def Test_Preprocessing(test_data):
    #print test_data
    now_data = np.array([test_data['Axis1'], test_data['Axis2'], test_data['Axis3']]).T

    data = []
    for axis in ["Axis1", "Axis2", "Axis3"]:
        _, tmp = splitSteps(test_data, test_data['Axis1'], axis)
        data.append(np.array(tmp))

    testing_features = np.array([[0.0] * len(data[0])]).T

    for x in data:
        testing_features = np.insert(testing_features, testing_features.shape[1], Vectorize(x), axis=1)

    testing_features = np.delete(testing_features, 0, axis=1)

    print testing_features.shape

    return np.array(testing_features)

def Multi_Test_Preprocessing(test_data, dictionary, pca_model, cut_size, slide_size):
    #print test_data
    testing_features = []
    testing_labels = []
    for i in range(len(test_data)):
        _, tmp = splitSteps(test_data[i], test_data[i]['Axis1'])
        testing_features.extend(CreateDTWFeature(dictionary, tmp))
        testing_labels.extend([i] * len(tmp))

    return testing_features, testing_labels

def Predicting(model, scaler, test_data):
    print "Predicting"
    testing_features = Test_Preprocessing(test_data)

    try:
        testing_features = scaler.transform(testing_features)
        predicted_label = model.predict(testing_features)
    except:
        return 0

    print "start voting."
    voting = np.zeros(2)
    for i in set(predicted_label):
        voting[i] = np.sum(predicted_label == i)

    result = voting.tolist().index(voting.max())
    print 'Max: ', voting, set(predicted_label), predicted_label, "Result:", result
    if result == 0:
        result = -1
    #if voting.max() <= (len(voting)/2):
    #    result = -1
    #print "voting: ", voting
    return result

def FindBestClf(features, labels, master_no):
    now_labels = map(lambda x: 1 if x == master_no else 0, labels)
    tuned_params = {"kernel":["rbf"], "gamma": [10**x for x in xrange(-5, 1)], "C":[10**x for x in xrange(-5, 1)]}

    grid_search = GridSearchCV(SVC(class_weight="auto"), tuned_params, cv=5, verbose=1, scoring='f1', n_jobs=4)
    result = grid_search.fit(features, now_labels)

    print "BinCount: " + str(np.bincount(now_labels))
    print "Best F1-Score: " + str(result.best_score_)
    print "Best Parameters: " + str(result.best_params_)

    return result.best_estimator_

def Normalizing(features):
    scaler = StandardScaler()
    return scaler.fit(features)

def UnderSampling(features, labels, master_no):
    features = np.array(features)
    labels = np.array(labels)
    master_num = len(features[labels==master_no])

    other_labels = filter(lambda x: x != master_no, np.unique(labels))
    sampling_features = features[labels==master_no]
    sampling_labels   = list(labels[labels==master_no])

    for target in other_labels:
        idx = np.random.choice(xrange(0, len(features[labels==target])), int(master_num/len(other_labels)), replace=False)
        tmp = features[labels==target][idx]
        sampling_features = np.insert(sampling_features, sampling_features.shape[0], tmp, axis=0)
        sampling_labels.extend([target] * int(master_num/len(other_labels)))

    return sampling_features, sampling_labels

def Training(trainning_features, labels, master_no, C = 1.0):
    print trainning_features.shape
    now_labels = map(lambda x: 1 if x == master_no else 0, labels)
    model = LinearSVC(class_weight='auto')
    model.fit(trainning_features, now_labels)
    return model

def Evaluation(model, testing_features, testing_labels):
    acc = model.score(testing_features, testing_labels)
    print acc
    return acc

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

    #pca = PCA(n_components = n_dimension)
    mds = MDS(n_components = n_dimension)
    colors = ['r', 'g', 'b', 'm', 'k']
    #labels_text = ['label_0', 'label_1', 'label_2', 'label_3', 'label_4']
    labels_text = ['Jhow', 'Terry', 'Tsai', 'Terry']
    labels = np.array(labels)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #tmp = pca.fit_transform(data)
    tmp = mds.fit_transform(np.array(data))
    print tmp.shape

    map(lambda i: ax.scatter(tmp[labels==i, 0], tmp[labels==i, 1], tmp[labels==i, 2], label=labels_text[i], c=colors[i], marker='o', s=55), xrange(len(np.unique(labels))))

    ax.set_xlabel('1st_component')
    ax.set_ylabel('2nd_component')
    ax.set_zlabel('3rd_component')

    plt.legend(scatterpoints=1, ncol=3)
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
    f = lambda sample_data, test_data: map(lambda ts_test: map(lambda ts_sample: dtw(ts_test, ts_sample), sample_data), test_data)

    features = f(sample_data, test_data)

    return features

# return index of n-sized samples as dictionary
def Sampling(data, sample_size):
    all_idx = xrange(len(data))
    sample_idx =  np.random.choice(all_idx, sample_size, replace=False)
    testing_idx = list(set(all_idx) - set(sample_idx))

    return np.array(sample_idx), np.array(testing_idx)

def Kmeans(data, sample_size):
    kmeans = KMeans(sample_size, max_iter=1000)

    kmeans.fit(data)
    

    return kmeans.cluster_centers_

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
    for idx in xrange(0, len(data), n):
        chunks.append(data[idx: idx+n])


    return np.array(chunks[:-1])

def splitSteps(df, ts, axis, threshold=0.5):
    idx   = ts.loc[ts > threshold]

    diff = np.array(idx.index[1:]) - np.array(idx.index[:-1])
    #print diff
    tmp = idx[[x & y for x, y in zip(diff != 1, diff > 10)]][:-1]
    tmp = tmp[1:]

    return tmp, map(lambda x:df.loc[xrange(tmp.index[x]-10, tmp.index[x]+20)][axis].reset_index(drop=True), xrange(0, len(tmp)))


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
    name = []
    for i in range(1,len(sys.argv)):
        name.append(sys.argv[i])


    data = Load(name)
    acc = []
    label = []
    f = []
    cut_s = []
    out = open('dtw_terry_slide_50.csv', 'w')
    for cut_size in range(10,110,10):
        training_feature, testing_feature, labels = Preprocessing(data[:-1], [data[-1]], cut_size=cut_size, slide_size=50)
        model = Training(np.array(training_feature), labels)
        out.write(str(cut_size) + ',' + str(Evaluation(model, testing_feature, [1]*len(testing_feature))) + '\n')
        tmp = np.insert(training_feature, len(training_feature), testing_feature, axis=0)
        tmp_labels = labels
        tmp_labels.extend([3] * len(testing_feature))
        Ploting3D(tmp, tmp_labels)
    print acc
