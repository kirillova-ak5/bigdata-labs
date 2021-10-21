import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


def func_distribution(a):
    a.sort()
    distr = np.zeros(a.size)
    distr_x = np.zeros(a.size)
    for i in range(a.size):
        distr_x[i] = a.min() + ((a.max() - a.min()) * i / a.size)
        distr[i] = (a < distr_x[i]).sum() / a.size
    return distr, distr_x


def median(a):
    a.sort()
    if a.size % 2:
        return (a[(a.size - 1) / 2 - 1] + a[(a.size + 1) / 2] - 1) / 2.0
    return a[int(a.size / 2) - 1]


def upper_quartile(a):
    a.sort()

    if a.size % 4:
        return a[int(a.size / 4)]
    return a[int(a.size / 4) - 1]


def lower_quartile(a):
    a.sort()

    if a.size % 4:
        return a[a.size - int(a.size / 4) + 1]
    return a[a.size - int(a.size / 4)]