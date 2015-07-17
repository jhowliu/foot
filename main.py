import train_envelope as train
import numpy as np

def Run(data, iter=10):
    result = {}
    for x in range(5, 16, 1):
        tmp = []
        for cnt in xrange(iter):
            trainning_features, labels = train.Preprocessing(data[:-1], data[:-1], num_std=x/10.0, slide_size=10, cut_size=50)
            testing_features, _ = train.Preprocessing(data[:-1], [data[-1]], num_std=x/10.0, slide_size=10, cut_size=50)
            model = train.Training(trainning_features, labels)
            acc = train.Evaluation(model, testing_features, [1]*len(testing_features))
            tmp.append(acc)

        result[x/10.0] = np.mean(tmp)

    return result

def Run_Cut(data, num_std=1, iter=1):
    result = {}

    for x in range(10, 110, 10):
        tmp = []
        for cnt in xrange(iter):
            print cnt
            trainning_features, labels = train.Preprocessing(data[:-1], data[:-1], num_std=num_std, slide_size=50, cut_size=x)
            testing_features, _ = train.Preprocessing(data[:-1], [data[-1]], num_std=num_std, slide_size=50, cut_size=x)
            model = train.Training(trainning_features, labels)
            acc = train.Evaluation(model, testing_features, [1]*len(testing_features))
            tmp.append(acc)

        result[x] = np.mean(tmp)

    return result
