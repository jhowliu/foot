import train_envelope as model
import numpy as np

def train(data, iter=10):
    result = {}
    for x in range(5, 16, 1):
        tmp = []
        for cnt in xrange(iter):
            _, _, acc = model.training(data, slide_size=50, num_std=x/10.0)
            tmp.append(acc)

        result[x/10.0] = np.mean(tmp)

    return result

