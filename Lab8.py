import math

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

def pearsonr(x, y):
    # Assume len(x) == len(y)
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(xi*xi for xi in x)
    sum_y_sq = sum(yi*yi for yi in y)
    psum = sum(xi*yi for xi, yi in zip(x, y))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den

def getRss(X,Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    res = 0.0

    for i in range(len(X)):
        res += (X[i] - mean_x) * (Y[i] - mean_y)
    return res

if __name__ == "__main__":
    num = 10
    data = pd.read_csv('weight-height.csv').head(num)   #load first num elems
    data['Height'] *= 2.54
    data['Weight'] /= 2.205         #to kilo and meters

    x = np.array(data['Height']).reshape((-1, 1))
    y = np.array(data['Weight'])

    model = LinearRegression().fit(x,y)

    print('y = ', model.intercept_, ' + ', model.coef_[0], '* x')
    x_new = np.linspace(x.min(), x.max(), num)
    y_pred = model.intercept_ + model.coef_[0] * x_new
    plt.title('Linear regression')
    plt.scatter(x,y, color='red', label='Actual data')
    plt.plot(x_new, y_pred, c='blue', label='Regression Line')
    plt.show()

    rss = getRss(x_new, y_pred)
    print('RSS = ', rss)
    print('RSE = ', rss/(x_new.size - 2))
    print('mu^2 =  ', pearsonr(x_new, y_pred))



    plt.figure()
    plt.title('Multiplicative regression')
    plt.scatter(x, y, color='red', label='Actual data')

    log_x = x
    log_y = y
    for i in range (len(x)):
        log_x[i] = (math.log(x[i]))
        log_y[i] = (math.log(y[i]))
        x_new[i] = math.log(x_new[i])

    model = LinearRegression().fit(log_x, log_y)
    print('y = ', model.intercept_, ' * ', model.coef_[0], '^ x')

    y_pred = model.intercept_ + x_new * model.coef_[0]

    rss = math.exp(getRss(x_new, y_pred))
    print('RSS = ', rss)
    print('RSE = ', rss/(x_new.size - 2))
    print('mu^2 =  ', pearsonr(x_new, y_pred))

    for i in range (len(x)):
        y_pred[i] = math.exp(y_pred[i])
        x_new[i] = math.exp(x_new[i])

    plt.plot(x_new, y_pred, c='blue', label='Regression Line')
    plt.show()

'''
import math

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

def pearsonr(x, y):
    # Assume len(x) == len(y)
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(xi*xi for xi in x)
    sum_y_sq = sum(yi*yi for yi in y)
    psum = sum(xi*yi for xi, yi in zip(x, y))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0: return 0
    return num / den

def getRss(X,Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    res = 0.0

    for i in range(len(X)):
        res += (X[i] - mean_x) * (Y[i] - mean_y)
    return res

if __name__ == "__main__":
    num = 10
    data = pd.read_csv('weight-height.csv').head(num)   #load first num elems
    data['Height'] *= 2.54
    data['Weight'] /= 2.205         #to kilo and meters

    x = np.array(data['Height']).reshape((-1, 1))
    y = np.array(data['Weight'])

    model = LinearRegression().fit(x,y)

    print('y = ', model.intercept_, ' + ', model.coef_[0], '* x')
    x_new = np.linspace(x.min(), x.max(), num)
    y_pred = model.intercept_ + model.coef_[0] * x_new
    plt.title('Linear regression')
    plt.scatter(x,y, color='red', label='Actual data')
    plt.plot(x_new, y_pred, c='blue', label='Regression Line')
    plt.show()

    rss = getRss(x_new, y_pred)
    print('RSS = ', rss)
    print('RSE = ', rss/(x_new.size - 2))
    print('mu^2 =  ', pearsonr(x_new, y_pred))



    plt.figure()
    plt.title('Multiplicative regression')
    plt.scatter(x, y, color='red', label='Actual data')

    log_x = x
    log_y = y
    for i in range (len(x)):
        log_x[i] = (math.log(x[i]))
        log_y[i] = (math.log(y[i]))
        x_new[i] = math.log(x_new[i])

    model = LinearRegression().fit(log_x, log_y)

    y_pred = model.intercept_ + x_new * model.coef_[0]

    rss = getRss(x_new, y_pred)
    print('RSS = ', rss)
    print('RSE = ', rss/(x_new.size - 2))
    print('mu^2 =  ', pearsonr(x_new, y_pred))

    for i in range (len(x)):
        y_pred[i] = math.exp(y_pred[i])
        x_new[i] = math.exp(x_new[i])

    plt.plot(x_new, y_pred, c='blue', label='Regression Line')
    plt.show()


'''