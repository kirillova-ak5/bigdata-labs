import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from tabulate import tabulate
   

def turn_point_arr(a):
    f = lambda a, b, c: 1 if (b > a and b > c) or (b < a and b < c) else 0
    rot = np.array([f(a[i], a[i + 1], a[i + 2]) for i in range(a.size - 2)])
    return rot

def ord_corr_arr(a):
    corr = np.zeros(len(a))
    for i in range(len(a)):
        corr[i]=sum([(1 if a[j] > a[i] else 0) for j in range(i+1,len(a))])
    return corr

def test_tp(x, show_table=True):
    sz = len(x)
    first = turn_point_arr(x).sum()
    second = 2/3*(sz-2)
    third = np.sqrt((16*sz-29)/90)
    if show_table:
        print(tabulate([
            ['turning points count']+[first],
            ['expected']+[second],
            ['with sigma']+[third]
            ]))
    return [first, second, third]

def test_tau(x, show_table=True):
    sz = len(x)
    first = ord_corr_arr(x).sum()*4/(sz*(sz-1))-1
    second = 0
    third = np.sqrt((4*sz+10)/(9*sz*(sz-1)))
    if show_table:
        print(tabulate([
            ['turning points count']+[first],
            ['expected']+[second],
            ['with sigma']+[third]
            ]))
    return [first, second, third]

def test(x, test = 'all', show_table=True):
    
    if (test == 'tp'):
        return test_tp(x, show_table)
    if (test == 'tau'):
        return test_tau(x, show_table)
    if (test == 'all'):
        results = [[]]
        results.append(['turning points']+test_tp(x, False))
        results.append(['Kendall tau']+test_tau(x, False))
        if show_table:
            print(tabulate([['test','res','expected for random','sigma']]+results))
        return results