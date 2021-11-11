import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


def my_beautiful_norm(a: np.ndarray, w: np.ndarray):
    if np.sum(w) != 1:
        w = np.divide(w, np.sum(w))
    return np.dot(a, w)


def lab1_task1():
    a = np.arange(-10.0, 6.0, 1.0)
    b = np.arange(-5.0, 10.1, 1.0)
    w = np.ones(16)
    for i in range(w.size):
        w[i] += np.random.normal()
    print("weights is ", w)
    print("A is ", a, " norm is ", my_beautiful_norm(a, w))
    print("B is ", b, " norm is ", my_beautiful_norm(b, w))


def input_vec(n: int):
    vec = input().split(" ")
    v = np.ones(n)
    for i in range(0, n):
        v[i] = vec[i]
    return v


def min_max_sum(v: np.ndarray):
    m1 = v.min()
    m2 = v.max()
    m3 = v.sum()
    return m1, m2, m3


def lab1_task2():
    v1 = input_vec(5)
    print(min_max_sum(v1))


def fak(n: int):
    rez = 1
    for j in range(1, n):
        rez *= j
    return rez


# T201t is for long arithmetic

import lab2
import lab3
import lab4

if __name__ == '__main__':
    #lab1_task1()
    #print(fak(100000))
    #lab2.lab2()
    #lab3.lab3()
    lab4.lab4()

