import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import tools

def generate():
    h = 0.1
    x = np.array([0.5 * np.sin(k * h) + np.random.normal() for k in range(200)])
    return x


def model():
    h = 0.1
    x = np.array([0.5 * np.sin(k * h) for k in range(200)])
    return x


def exp_mean(x, a):
    y = np.zeros(x.size)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def lab4():
    x = generate()
    x_m = model()
    x_01 = exp_mean(x, 0.01)
    x_05 = exp_mean(x, 0.05)
    x_1 = exp_mean(x, 0.1)
    x_3 = exp_mean(x, 0.3)

    plt.plot(x, label='source')
    plt.plot(x_m, label='model')
    plt.plot(x_01, label='a = 0.01')
    plt.plot(x_05, label='a = 0.05')
    plt.plot(x_1, label='a = 0.1')
    plt.plot(x_3, label='a = 0.3')

    plt.legend()
    plt.show()

    plt.figure()

    f = np.fft.fft(x)
    ff = f.real * f.real + f.imag * f.imag
    plt.plot([(i + 1) / x.size * 2 * np.pi for i in range(x.size)], ff)
    #plt.plot(ff)
    freq = (ff.argmax() + 1.0) / x.size * 2.0 * np.pi
    freq_i = ff.argmax()
    plt.show()
    print(freq)
    print(freq_i)
    print(ff[180:])

    r_01 = x - x_01
    r_05 = x - x_05
    r_1 = x - x_1
    r_3 = x - x_3

    r_est = 2.0 / 3.0 * (x.size - 2)
    r_disp = (16 * x.size - 29) / 90.0
    print('num of rotation points estimated: ', r_est)
    print('dispersion of rotation points: ', r_disp)

    print('exp mean, a = 0.01')
    print('mean reminder ', r_01.mean())

    print('num of rotation points r_mean_21: %i', tools.rot_point_arr(r_01).sum())
    if tools.rot_point_arr(r_01).sum() < r_est + r_disp and tools.rot_point_arr(r_01).sum() > r_est - r_disp:
        print('rot point random')
    elif tools.rot_point_arr(r_01).sum() > r_est + r_disp:
        print('rot point oscillating')
    elif tools.rot_point_arr(r_01).sum() < r_est - r_disp:
        print('rot point korrelation')


