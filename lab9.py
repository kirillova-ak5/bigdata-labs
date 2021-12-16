import numpy as np
import scipy.stats as st
import scipy as scp
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import tools


def model():
    arr = np.array([[-2.0, -7.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 9.0]])
    #arr.reshape()
    return arr


def noised(err):
    arr = model()
    for a in arr:
        a[1] += np.random.normal(0, err)
    return arr


def modelx():
    return [-2.0, -1.0, 0.0, 1.0, 2.0]


def modely():
    return [-7.0, 0.0, 1.0, 2.0, 9.0]


def noisedy(err):
    my = modely()
    for i in range(len(my)):
        my[i] += np.random.normal(0, err)
    return my

def lab9():
    #print(noised(0.3))

    factors = []
    modx = modelx()
    for i in range(0, 12):
        factors.append([m ** i for m in modx])
    #print(factors)

    samples = []
    for i in range(len(modx)):
        samples.append([factor[i] for factor in factors])

    y_0 = modely()
    y_1 = noisedy(0.1)
    y_2 = noisedy(0.2)
    y_3 = noisedy(0.3)

    rezx = [(x / 100.0) for x in range(-201, 202)]

# y_0
    clf1 = lm.Lasso(alpha=1.0)
    clf1.fit(samples, y_0)
    rezy1 = [sum([clf1.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy1, label='a = 1.0')

    clf2 = lm.Lasso(alpha=0.1)
    clf2.fit(samples, y_0)
    rezy2 = [sum([clf2.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy2, label='a = 0.1')

    clf3 = lm.Lasso(alpha=0.01)
    clf3.fit(samples, y_0)
    rezy3 = [sum([clf3.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy3, label='a = 0.01')

    plt.scatter(modx, y_0)
    plt.legend()
    plt.title('Lasso, y not noised')
    plt.figure()

# y_1
    clf1 = lm.Lasso(alpha=1.0)
    clf1.fit(samples, y_1)
    rezy1 = [sum([clf1.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy1, label='a = 1.0')

    clf2 = lm.Lasso(alpha=0.1)
    clf2.fit(samples, y_1)
    rezy2 = [sum([clf2.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy2, label='a = 0.1')

    clf3 = lm.Lasso(alpha=0.01)
    clf3.fit(samples, y_1)
    rezy3 = [sum([clf3.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy3, label='a = 0.01')

    plt.scatter(modx, y_1)
    plt.legend()
    plt.title('Lasso, y noised 0.1')
    plt.figure()
# y_2
    clf1 = lm.Lasso(alpha=1.0)
    clf1.fit(samples, y_2)
    rezy1 = [sum([clf1.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy1, label='a = 1.0')

    clf2 = lm.Lasso(alpha=0.1)
    clf2.fit(samples, y_2)
    rezy2 = [sum([clf2.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy2, label='a = 0.1')

    clf3 = lm.Lasso(alpha=0.01)
    clf3.fit(samples, y_2)
    rezy3 = [sum([clf3.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy3, label='a = 0.01')

    plt.scatter(modx, y_2)
    plt.legend()
    plt.title('Lasso, y noised 0.2')
    plt.figure()
# y_3
    clf1 = lm.Lasso(alpha=1.0)
    clf1.fit(samples, y_3)
    rezy1 = [sum([clf1.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy1, label='a = 1.0')

    clf2 = lm.Lasso(alpha=0.1)
    clf2.fit(samples, y_3)
    rezy2 = [sum([clf2.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy2, label='a = 0.1')

    clf3 = lm.Lasso(alpha=0.01)
    clf3.fit(samples, y_3)
    rezy3 = [sum([clf3.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy3, label='a = 0.01')

    plt.scatter(modx, y_3)
    plt.legend()
    plt.title('Lasso, y noised 0.3')
    plt.figure()

#############
    # Ridge
#############

    # y_0
    clf1 = lm.Ridge(alpha=1.0)
    clf1.fit(samples, y_0)
    rezy1 = [sum([clf1.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy1, label='a = 1.0')

    clf2 = lm.Ridge(alpha=0.1)
    clf2.fit(samples, y_0)
    rezy2 = [sum([clf2.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy2, label='a = 0.1')

    clf3 = lm.Ridge(alpha=0.01)
    clf3.fit(samples, y_0)
    rezy3 = [sum([clf3.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy3, label='a = 0.01')

    plt.scatter(modx, y_0)
    plt.legend()
    plt.title('Ridge, y not noised')
    plt.figure()

    # y_1
    clf1 = lm.Ridge(alpha=1.0)
    clf1.fit(samples, y_1)
    rezy1 = [sum([clf1.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy1, label='a = 1.0')

    clf2 = lm.Ridge(alpha=0.1)
    clf2.fit(samples, y_1)
    rezy2 = [sum([clf2.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy2, label='a = 0.1')

    clf3 = lm.Ridge(alpha=0.01)
    clf3.fit(samples, y_1)
    rezy3 = [sum([clf3.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy3, label='a = 0.01')

    plt.scatter(modx, y_1)
    plt.legend()
    plt.title('Ridge, y noised 0.1')
    plt.figure()
    # y_2
    clf1 = lm.Ridge(alpha=1.0)
    clf1.fit(samples, y_2)
    rezy1 = [sum([clf1.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy1, label='a = 1.0')

    clf2 = lm.Ridge(alpha=0.1)
    clf2.fit(samples, y_2)
    rezy2 = [sum([clf2.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy2, label='a = 0.1')

    clf3 = lm.Ridge(alpha=0.01)
    clf3.fit(samples, y_2)
    rezy3 = [sum([clf3.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy3, label='a = 0.01')

    plt.scatter(modx, y_2)
    plt.legend()
    plt.title('Ridge, y noised 0.2')
    plt.figure()
    # y_3
    clf1 = lm.Ridge(alpha=1.0)
    clf1.fit(samples, y_3)
    rezy1 = [sum([clf1.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy1, label='a = 1.0')

    clf2 = lm.Ridge(alpha=0.1)
    clf2.fit(samples, y_3)
    rezy2 = [sum([clf2.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy2, label='a = 0.1')

    clf3 = lm.Ridge(alpha=0.01)
    clf3.fit(samples, y_3)
    rezy3 = [sum([clf3.coef_[i] * (x / 100.0) ** i for i in range(0, 12)]) for x in range(-201, 202)]
    plt.plot(rezx, rezy3, label='a = 0.01')

    plt.scatter(modx, y_3)
    plt.legend()
    plt.title('Ridge, y noised 0.3')

    plt.show()

