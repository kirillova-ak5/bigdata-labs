import numpy as np
import matplotlib.pyplot as plt


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


def slide_expmean(x: np.array, alpha):
    res = np.zeros(x.size)

    res[0] = x[0]
    for i in range(1, x.size): 
        res[i] = alpha*x[i]+(1-alpha)*res[i-1]

    return res


def run_slide(x, x_model, m : list, mode):
    
    result=[[] for i in range(len(m))]
    resides = [[] for i in range(len(m))]
    
    plt.figure()
    plt.plot(x, marker='.', ls='', label = 'data')
    plt.title(mode)
    plt.plot(x_model, label = 'model', ls='--', lw=1.5)

    for i in range(len(m)):
        if mode == 'mean':
            result[i] = slide_mean(x, m[i])
        if mode == 'median':
            result[i] = slide_median(x, m[i])
        if mode == 'expmean':
            result[i] = slide_expmean(x, m[i])
        resides[i] = result[i]-x_model
        plt.plot(result[i], label='par = '+str(m[i]), lw=0.5)
           
    plt.legend()
    plt.show(block=True)
    return resides,result