import matplotlib.pyplot as plt
import scipy
import numpy as np


def movingAverage(a, n=1):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def movingMedian(x, N):
    idx = np.arange(N) + np.arange(len(x) - N + 1)[:, None]
    b = [row[row > 0] for row in x[idx]]
    #return np.array(map(np.median, b))
    return np.array([np.median(c) for c in b])  # This also works


if __name__ == "__main__":

    step = [21, 55, 111]
    h = 0.05
    y = np.array([np.sqrt(k * h) + np.random.normal() for k in range(200)])

    plt.figure()
    plt.plot(y, label='source')

    for i in step:
        res = movingAverage(y, i)
        plt.plot(range(int(i / 2), len(res) + int(i / 2)), res, label='x_mean_{i}')

    plt.legend(loc="upper left")
    plt.show()

    ##median

    plt.figure()
    plt.plot(y, label='source')
    i=0
    for i in step:
        res = movingMedian(y, i)
        plt.plot(range(int(i / 2), len(res) + int(i / 2)), res, label='x_median_{i}')

    plt.legend(loc="upper left")
    plt.show()

## 3 and 4 tasks remaining
