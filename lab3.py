import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import tools

def generate():
    h = 0.05
    #x1 = [0] * 200
    x = np.array([np.sqrt(k * h) + np.random.normal() for k in range(200)])
    return x


def model():
    h = 0.05
    x = np.array([np.sqrt(k * h) for k in range(200)])
    return x


def slide_mean(x, m):
    x_filtered = np.zeros(x.size)
    for i in range(x.size):
        if i < m:
            x_filtered[i] = sum([x[j] for j in range(0, 2 * i + 2)]) / (2.0 * i + 1)
        elif i >= x.size - m - 1:
            x_filtered[i] = sum([x[j] for j in range(i - (x.size - i), x.size)]) / len(range(i - (x.size - i), x.size))
        else:
            x_filtered[i] = sum([x[j] for j in range(i - m, i + m + 1)]) / (2.0 * m + 1)
    return x_filtered


def slide_median(x: np.ndarray, m):
    x_filtered = np.zeros(x.size)
    for i in range(x.size):
        if i < m:
            x_filtered[i] = tools.median(x[0 : 2 * i + 1])
        elif i >= x.size - m - 1:
            x_filtered[i] = tools.median(x[i - (x.size - i) : x.size])
        else:
            x_filtered[i] = tools.median(x[i - m : i + m + 1])
    return x_filtered


def lab3():
    x = generate()
    x_model = model()
    x_mean_21 = slide_mean(x, 10)
    x_mean_51 = slide_mean(x, 25)
    x_mean_111 = slide_mean(x, 55)
    plt.plot(x, label = 'source')
    plt.plot(x_model, label = 'model')
    plt.plot(x_mean_21, label='x_mean_21')
    plt.plot(x_mean_51, label='x_mean_51')
    plt.plot(x_mean_111, label='x_mean_111')
    plt.legend()
    plt.show()

    plt.figure()

    plt.plot(x, label = 'source')
    plt.plot(x_model, label = 'model')
    x_med_21 = slide_median(x, 10)
    x_med_51 = slide_median(x, 25)
    x_med_111 = slide_median(x, 55)
    plt.plot(x_med_21, label='x_med_21')
    plt.plot(x_med_51, label='x_med_51')
    plt.plot(x_med_111, label='x_med_111')

    plt.legend()
    plt.show()

    r_mean_21 = x - x_mean_21
    print('num of rotation points r_mean_21: %i', tools.rot_point_arr(r_mean_21).sum())
    r_mean_51 = x - x_mean_51
    print('num of rotation points r_mean_51: %i', tools.rot_point_arr(r_mean_51).sum())
    r_mean_111 = x - x_mean_111
    print('num of rotation points r_mean_111: %i', tools.rot_point_arr(r_mean_111).sum())
    r_med_21 = x - x_med_21
    print('num of rotation points r_med_21: %i', tools.rot_point_arr(r_med_21).sum())
    r_med_51 = x - x_med_51
    print('num of rotation points r_med_51: %i', tools.rot_point_arr(r_med_51).sum())
    r_med_111 = x - x_med_111
    print('num of rotation points r_med_111: %i', tools.rot_point_arr(r_med_111).sum())
