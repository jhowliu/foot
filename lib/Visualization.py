import matplotlib.pyplot as plt
import numpy as np

# After Sliding cut
def draw_envelope_area(train_data, test_data, labels, num_std=1):
    colors = ['r', 'g', 'b', 'm']
    plt.figure()
    print train_data.shape
    x = xrange(train_data.shape[1])

    _mean = np.mean(train_data, axis=0)
    _std  = num_std*np.std(train_data, axis=0)

    plt.fill_between(x, _mean-_std, _mean+_std, facecolor='black', alpha=0.2)

    for i in xrange(len(np.unique(labels))):
        _mean = np.mean(test_data[labels==i], axis=0)

        #plt.plot(x, test_data[labels==i][50], color=colors[i])
        plt.plot(x, _mean, color=colors[i])
    plt.axis([0, 100, -3, 5])
    plt.xlabel('Sequence number')

    plt.show()
