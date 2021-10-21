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


def median(b: np.ndarray):
    a = b.copy()
    a.sort()
    if a.size % 2:
        return a[int(a.size / 2)]
    return (a[int((a.size) / 2)] + a[int((a.size) / 2 - 1)]) / 2.0


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


def rot_point_arr(a):
    f = lambda a, b, c: 1 if (b > a and b > c) or (b < a and b < c) else 0
    rot = np.array([f(a[i], a[i + 1], a[i + 2]) for i in range(a.size - 2)])
    return rot