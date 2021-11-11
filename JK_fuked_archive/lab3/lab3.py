#Ответы на вопросы из предыдущей лекции:
#1: порядка 100 монет
#2: <(Т-6)*10^(T-7)/10^T=(T-6)/1000, T~1000
#3: например, сравним много выборок на равенство средних


#import numpy as np
#import scipy.stats as sc
#
#samples = [[] for i in range(0,10002)]
#for i in range(0,10000):
#    samples[i] = np.random.normal(loc = i/2,size=80)
    
#s_m=0

#for i in range(0,10000):
#    for j in range(-2,2):
#        a, s = sc.ttest_ind(samples[i],samples[i+j])
#        if j!=0 and s>s_m:
#            s_m,i_m,j_m = s,i,i+j
#print(s_m, ": ",i_m/2," = ",j_m/2,"", sep='')    


#после 3 запусков: 0.9989956882621136: 1192.0 = 1192.5 ?!
#


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from tabulate import tabulate



def vals(noize, size = 201):
    h=0.05
    x = np.random.normal(size=size) if noize else np.zeros(size)
    x+=np.array([np.sqrt(i*h) for i in range(size)])
    return x


def slide_mean(x: np.array, m):
    res = np.zeros(x.size)
    
    if m*2>=x.size:
        return res

    res[0] = x[0]
    for i in range(1, m+1): 
        res[i] = np.sum(x[0:(2*i+1)])/(2*i+1)
    for i in range(m+1, x.size-m): 
        res[i] = np.sum(x[i-m:i+m+1])/(2*m+1)
    for i in range(1, m+1):  
        res[x.size-i] = np.sum(x[x.size-2*i-1:x.size])/(2*i+1)
    res[x.size-1] = x[x.size-1]
    return res


def slide_median(x: np.array, m):
    res = np.zeros(x.size)
    
    if m*2>=x.size:
        return res

    res[0] = np.median([x[0],x[1],3*x[1]-2*x[2]])
    for i in range(1, m+1): 
        res[i] = np.median(x[0:(2*i+1)])
    for i in range(m+1, x.size-m): 
        res[i] = np.median(x[i-m:i+m+1])
    for i in range(1, m+1):  
        res[x.size-i] = np.median(x[x.size-2*i-1:x.size])
    res[x.size-1] = np.median([x[x.size-1],x[x.size-2],3*x[x.size-2]-2*x[x.size-3]])

    return res


def turn_point_arr(a):
    f = lambda a, b, c: 1 if (b > a and b > c) or (b < a and b < c) else 0
    rot = np.array([f(a[i], a[i + 1], a[i + 2]) for i in range(a.size - 2)])
    return rot

def ord_corr_arr(a):
    corr = np.zeros(len(a))
    for i in range(len(a)):
        corr[i]=sum([(1 if a[j] > a[i] else 0) for j in range(i+1,len(a))])
    return corr

def run_slide(x, x_model, m : list, mode):
    
    resides = [[] for i in range(len(m))]
    
    plt.figure()
    plt.plot(x, marker='.', ls='', label = 'data')
    plt.title(mode)
    plt.plot(x_model, label = 'model', ls='--', lw=1.5)

    for i in range(len(m)):
        if mode == 'mean':
            result = slide_mean(x, m[i])
        if mode == 'median':
            result = slide_median(x, m[i])
        resides[i] = result-x_model
        plt.plot(result, label='m='+str(m[i]), lw=0.5)
           
    plt.legend()
    plt.show(block=True)
    return resides
   

def lab3():
    sz = 201
    x = vals(True, size=sz)
    x_model = vals(False, size=sz)
    
    m = [10,25,55]
    resides_mean = run_slide(x,x_model, m,'mean')
    resides_median = run_slide(x,x_model, m,'median')


    
    turning_points = [ 
        ['m']+m, 
        ['mean turning points']+([turn_point_arr(resides_mean[i]).sum() for i in range(len(m))]), 
        ['median turning points']+([turn_point_arr(resides_median[i]).sum() for i in range(len(m))]), 
        ['expected']+([2/3*(sz-2) for i in range(len(m))]), 
        ['sigma is']+([np.sqrt((16*sz-29)/90) for i in range(len(m))]), 
        ]
    correlation = [ 
        ['m']+m, 
        ['mean tau']+([ord_corr_arr(resides_mean[i]).sum()*4/(sz*(sz-1))-1 for i in range(len(m))]), 
        ['median tau']+([ord_corr_arr(resides_median[i]).sum()*4/(sz*(sz-1))-1 for i in range(len(m))]), 
        ['sigma is']+([np.sqrt((4*sz+10)/(9*sz*(sz-1))) for i in range(len(m))]) 
        ]

    print(tabulate(turning_points))
    print(tabulate(correlation))

lab3()