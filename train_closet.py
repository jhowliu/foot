import sys
sys.path.append('./lib/')
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import ts_cluster

tool = ts_cluster.ts_cluster()
# Each guy have 6-axis data
def FuzzyDirection(data):
    pca = PCA(n_components = 1)
    result = [] 
    for idx in xrange(len(data)):
        # Acceleration
        acc = map(lambda x, y, z: pca.fit_transform(np.array([x, y, z]).T).T, data[idx][0], data[idx][1], data[idx][2])
        # G-Sensor
        g   = map(lambda x, y, z: pca.fit_transform(np.array([x, y, z]).T), data[idx][3], data[idx][4], data[idx][5])

        result.append([acc, g])

    return result

def CreateDTWFeatures(sample_data, test_data):
    features = lambda sample_data, test_data: map(lambda ts_test: map(lambda ts_sample: tool.DTWDistance(ts_test, ts_sample), sample_data), test_data)

    # Acceleration
