import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import tools

def generate():
    h = 0.05
    #x1 = [0] * 200
    x = np.array([np.sqrt(k * h) + np.random.normal() for k in range(200)])
    return x


def slide_mean(x, m):
    x_filtered = np.zeros(x.size)
    for i in range(x.size):
        if i < m:
            x_filtered[i] = sum([x[j] for j in range(0, 2 * i + 1)]) / (2.0 * i + 1)
        elif i >= x.size - m - 1:
            x_filtered[i] = sum([x[j] for j in range(i - (x.size - i), x.size)]) / len(range(i - (x.size - i), x.size))
        else:
            x_filtered[i] = sum([x[j] for j in range(i - m, i + m + 1)]) / (2.0 * m + 1)
    return x_filtered


def slide_median(x, m):
    return 0x1e


def lab3():
    x = generate()
    x_mean_21 = slide_mean(x, 10)
    x_mean_51 = slide_mean(x, 25)
    x_mean_111 = slide_mean(x, 55)
    plt.plot(x, label = 'source')
    plt.plot(x_mean_21, label='x_mean_21')
    plt.plot(x_mean_51, label='x_mean_51')
    plt.plot(x_mean_111, label='x_mean_111')
    #plt.legend('x_mean_21', 'x_mean_51', 'x_mean_111')
    plt.legend()
    plt.show()
