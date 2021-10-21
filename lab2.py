import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import tools


def laplassian_distr(x):
    if x <= 0:
        return 0.5 * np.exp(x)
    return 1 - 0.5 * np.exp(-x)

def lab2():
    coshi = np.random.standard_cauchy(100)
    coshi.sort()
    print('cauchy mean ')
    print(coshi.mean())
    print('cauchy median')
    print(tools.median(coshi))
    print('cauchy extreme halfsum')
    print((coshi[0] + coshi[coshi.size - 1]) / 2.0)
    print('cauchy quartiles halfsum')
    print((tools.upper_quartile(coshi) + tools.lower_quartile(coshi)) / 2.0)

    ravnom = np.random.uniform(0, 1, 100)
    laplas = np.random.laplace(0, 1, 100)
    ravnom_true = [1, 1]


    plt.hist(ravnom, bins=30, density=True, stacked=True)
    plt.plot(ravnom_true)
    plt.show()

    plt.figure()
    laplas_distr, laplas_distr_x = tools.func_distribution(laplas)
    laplas_theory = np.array([laplassian_distr(x) for x in laplas_distr_x])
    plt.plot(laplas_distr_x, laplas_distr)
    plt.plot(laplas_distr_x, laplas_theory)

    #sc.stats.laplas.pdf()
    plt.show()
