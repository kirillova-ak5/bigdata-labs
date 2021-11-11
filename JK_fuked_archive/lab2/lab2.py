import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


def upper_quartile (data):
   return np.percentile(data, 75)
def lower_quartile (data):
   return np.percentile(data, 25)

def laplassian_distr(x):
    if x>=0:
        return 1- 0.5 * np.exp(-x)
    else:
        return 0.5 * np.exp(x)

def lab2():
    coshi = np.random.standard_cauchy(100)
    coshi.sort()
    print('cauchy mean ')
    print(coshi.mean())
    print('cauchy median')
    
    print(np.median(coshi))
    print('cauchy extreme halfsum')
    print((coshi[0] + coshi[coshi.size - 1]) / 2.0)
    print('cauchy quartiles halfsum')
    print((upper_quartile(coshi) + lower_quartile(coshi)) / 2.0)

    uni = np.random.uniform(0, 1, 100)
    lap = np.random.laplace(0, 1, 100)
    uniform_f_to_plot = [1, 1]


    plt.hist(uni, bins=30, density=True, stacked=True)
    plt.plot(uniform_f_to_plot)
    plt.show(block=False)

    plt.figure()
    lap.sort()
    samp_dist = np.array([i/100 for i in range(1,101)])
    theor_dist = np.array([laplassian_distr(x) for x in lap])
    plt.plot(lap, samp_dist, ds='steps-post')
    plt.plot(lap, theor_dist)
    plt.show()

lab2()

#unsused

def laplassian_inv(y):
    if y>=0.5:
        return -np.log(2*(1-y))
    else:
        return np.log(2*y)