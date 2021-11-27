import matplotlib.pyplot as plt
import scipy
import numpy as np


# semantically false for anomaly
def threeSigmRule(num, mean, err):
    if abs(num - mean) > 3 * err:
        return False

    return True


def lab6():
    mass = np.array([np.random.normal() for k in range(200)])
    mass[195] = 5
    mass[196] = -4
    mass[197] = 3.3
    mass[198] = 2.99
    mass[199] = -3

    mass = np.sort(mass)

    math_mean = mass.mean()
    math_err = mass.std()

    print("mean value: ", math_mean)
    print("standart deviation: ", math_err)

    for i in range(3):
        print("Income: ", mass[i], "\n Result: ", threeSigmRule(mass[i], math_mean, math_err))
        print("Income: ", mass[199 - i], "\n Result: ", threeSigmRule(mass[199 - i], math_mean, math_err))

    plt.boxplot(mass)
    plt.show()
