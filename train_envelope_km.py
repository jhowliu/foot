import sys
sys.path.append('./lib/')
from Envelope import envelope
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
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

def build_pca(data):
    pca = PCA(n_components=1)
    tmp = np.array([data['Axis1'], data['Axis2'], data['Axis3']]).T
    pca.fit(tmp)
    return pca

# Return the trainning fe
def Preprocessing(train_data, test_data, cut_size=150, slide_size=100, sample_ratio=0.8, num_std=1.5):
    trainning = {'data': [], 'label': []}
    testing   = {'data': [], 'label': []}

    trainning_features = np.zeros([])
    testing_features   = np.array([])
    dictionary = []
    train_label = []
    pca_model = []
    env_sample_size = []
    totalsize = []
    pca_data = pd.DataFrame(columns=['Axis1', 'Axis2', 'Axis3', 'Axis4', 'Axis5', 'Axis6', 'Time', 'Label'])
    for i in xrange(len(train_data)):
        pca_data = pca_data.append(train_data[i])
    
    for i in xrange(len(train_data)):
        pca_model.append(build_pca(train_data[i]))
        
    # Fuzzy the direction and create the trainning data
    for i in xrange(len(train_data)):
        print 'Number of records: ' + str(len(train_data[i]))
        now_data = np.array([train_data[i]['Axis1'], train_data[i]['Axis2'], train_data[i]['Axis3']]).T
        transform_data = pca_model[i].transform(now_data)
        transform_data.resize(len(transform_data)) 
        tmp = Alignment_Slide_Cut(transform_data, cut_size, slide_size)
        print "now_data:", len(now_data), "len: ",len(tmp)
        totalsize.append(len(tmp))
        sample_size = int(len(tmp) * sample_ratio)
        #print sample_size
        env_sample_size.append(sample_size)
        trainning['data'].extend(tmp)
        trainning['label'].extend([i]*len(tmp))
        dictionary.extend(Kmeans(tmp, sample_size))
        train_label.extend([i] * sample_size)
    #print "dic: ", len(dictionary)
    #print env_sample_size
    '''
    # Fuzzy the direction and create the testing data
    for i in xrange(len(test_data)):
        print 'Number of records: ' + str(len(test_data[i]))
        now_data = np.array([test_data[i]['Axis1'], test_data[i]['Axis2'], test_data[i]['Axis3']]).T
        transform_data = pca_model.transform(now_data)
        transform_data.resize(len(transform_data))
        tmp = Slide_Cut(transform_data, cut_size, slide_size)
        test_slide_size = len(tmp)
    '''
    
    test_slide_size = ((len(test_data[0]) - cut_size) / slide_size) + 1
    
    trainning_features = np.zeros([sum(totalsize), cut_size * len(train_data)])
    
    testing_features   = np.zeros([test_slide_size, cut_size * len(train_data)])
    print testing_features.shape
    print 'Create Features'
    #trainning_features = CreateDTWFeature(dictionary, trainning['data'])
    #testing_features = CreateDTWFeature(dictionary, testing['data'])
    train_label = np.array(train_label)
    dictionary = np.array(dictionary)
    start = 0
    for i in xrange(len(train_data)):
        whole_tmp = []
        for j in xrange(len(train_data)):
            now_data = np.array([train_data[j]['Axis1'], train_data[j]['Axis2'], train_data[j]['Axis3']]).T
            transform_data = pca_model[i].transform(now_data)
            transform_data.resize(len(transform_data)) 
            each_model_tmp = Alignment_Slide_Cut(transform_data, cut_size, slide_size)
            print "now_data:", len(now_data), "len: ",len(each_model_tmp) 
            whole_tmp.extend(each_model_tmp)
        #tmp_env = envelope(train_label[train_label==i], dictionary[train_label == i], tmp, num_std)
 
        #print tmp_env.shape, start+totalsize[i]
        
        print trainning_features[:,start:start+cut_size].shape, envelope(train_label[train_label==i], dictionary[train_label == i], whole_tmp, num_std).shape
        trainning_features[:,start:start+cut_size] = envelope(train_label[train_label==i], dictionary[train_label == i], whole_tmp, num_std)
        now_data = np.array([test_data[0]['Axis1'], test_data[0]['Axis2'], test_data[0]['Axis3']]).T
        transform_data = pca_model[i].transform(now_data)
        transform_data.resize(len(transform_data))
        tmp = Slide_Cut(transform_data, cut_size, slide_size)
        #print 'shit',tmp.shape, 'test', testing_features.shape
        test = envelope(train_label[train_label==i], dictionary[train_label == i], tmp, num_std)
        
        testing_features[:,start:start+cut_size] = envelope(train_label[train_label==i], dictionary[train_label == i], tmp, num_std)
        start += cut_size

    return trainning_features, testing_features, trainning['label']

def Run(data, cut_size=150, slide_size=100, sample_ratio=0.8, num_std= 1.5):
    training = {'data': [], 'label': []}
    testing  = {'data': [], 'label': []}


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
    model = Trainning(training_feature, training['label']);

    print 'Predicting'
    acc = Evaluation(testing_feature, testing['label'])

    return training_feature, training['label'], acc

def Training(trainning_features, labels):
    print trainning_features.shape
    model = LinearSVC()
    model.fit(trainning_features, labels)

    return model

def Evaluation(model, testing_features, testing_labels):
    acc = model.score(testing_features, testing_labels)
    print acc
    return acc

def Predicting(model, testing_features):
    predicted_label = model.predict(testing_features)
    return predicted_label

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

# Alignment Cutting
def Alignment_Slide_Cut(data, cut_size, slide_size):
  
    max_bound = len(data) - cut_size
    bound = np.mean(data) + np.std(data)

    # Get the position of peak which is lower than cut_size
    peak_idx = np.where(data > bound)[0][np.where(np.where(data > bound)[0] < max_bound)[0]]
    indicies = [peak_idx[i] for i in xrange(0, peak_idx.shape[0], slide_size)]

    chunks = map(lambda idx: data[idx:idx+cut_size], indicies)
    print "(data, after_cut)", len(data), len(np.array(chunks[:-1]))
    return np.array(chunks[:-1])

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
    name = []
    for i in range(1, len(sys.argv)):
        name.append(sys.argv[i])
    
    data = Load(name)
    acc = []
    label = []
    f = []
    cut_s = []
    out = open('env_km_terry_slide_10.csv', 'w')
    for cut_size in range(10,100,10):
        print cut_size
        trainning_features, testing_features, labels = Preprocessing(data[:-1],  [data[-1]], cut_size=cut_size, slide_size=50, num_std = 1.0)
        model = Training(np.array(trainning_features), labels)
        out.write(str(cut_size) + ',' + str(Evaluation(model, testing_features, [1]*len(testing_features))) + '\n')
        #tmp = np.insert(trainning_features, len(trainning_features), testing_features, axis=0)
        #tmp_labels = labels
        #tmp_labels.extend([3] * len(testing_features))
        #Ploting3D(tmp, tmp_labels)
    print acc
