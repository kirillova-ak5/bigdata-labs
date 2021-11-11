import numpy as np
import my.slide
import my.random_test
import scipy.fft as fourier
import scipy.stats as st
import matplotlib.pyplot as plt
from tabulate import tabulate





def vals(noize, size = 201, h=0.1):
    x = np.random.normal(size=size) if noize else np.zeros(size)
    x+=np.array([0.5*np.sin(i*h) for i in range(size)])
    return x

def lab4():
    sz = 201
    x = vals(True, size=sz)
    x_pure = vals(False, size=sz)
    alpha=[0.01,0.05,0.1,0.3]
    resides, results = my.slide.run_slide(x, x_pure, alpha, "expmean")

    plt.figure()
    plt.title("fourier after trending")
    plt.vlines(0.1/2/np.pi,0,50,ls='dashed', label='origin frequency')
    q = [i/(sz-1) for i in range(sz)]
    for i in range(len(alpha)):
        plt.plot(q,abs(fourier.fft(results[i])), label='alpha = '+str(alpha[i]), lw=0.5)

    plt.legend()
    plt.show(block=True)   
    for i in range(len(alpha)):
        print('\n\n'+"alpha =", alpha[i])
        my.random_test.test(resides[i])
        avg=(np.sum(resides[i])/sz)
        sig=st.tstd(resides[i])
        print(tabulate([
            ['average is ' + str(avg)+'+-'+str(sig)],
            ['0 is '+('not ' if abs(avg)>abs(sig) else '')+ 'in'],
            ['is normal with p = ' + str(np.sqrt(st.normaltest(resides[i])[1]))]
                        ]))
    
    plt.figure()
    plt.title("resides")
    plt.plot(x-x_pure,ls='dashed', label='true reside')
    for i in range(len(alpha)):
        plt.plot(resides[i], label='alpha = '+str(alpha[i]), lw=0.5)
    plt.legend()
    plt.show(block=True)   

    plt.figure()
    plt.title("resides kde")
    r=[0.05*i for i in range(-100,101)]
    plt.plot(r,st.norm.pdf(r), lw=2, label='norm')
    plt.plot(r,st.gaussian_kde(x-x_pure).evaluate(r), ls='dashed', label='true reside')
    for i in range(len(alpha)):
        plt.plot(r,st.gaussian_kde(resides[i]).evaluate(r), label='alpha = '+str(alpha[i]), lw=0.5)
    plt.legend()
    plt.show(block=True)   



lab4()