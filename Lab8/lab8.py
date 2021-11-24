import math

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import lxml

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

def getDataWeightHeight(num):
    data = pd.read_csv('weight-height.csv').head(num)   #load first num elems
    data['Height'] *= 2.54
    data['Weight'] /= 2.205         #to kilo and meters

    x = np.array(data['Height']).reshape((-1, 1))
    y = np.array(data['Weight'])
    return x,y

def prepareDataTemp():
    years, data = pd.read_html('./AverageMSKTemp.htm')
    data = data.drop(list(range(82))+[243])
    years = years.drop(list(range(82))+[243])
    data.loc[:,:] = data.loc[:,:].astype('float32')
    years.loc[:, :] = years.loc[:, :].astype('float32')
    y, x = data.mean(axis=1).values, years.values

    x = x.reshape(-1)
    y = y.reshape(-1)
    new_data = {'Year': x, 'Temperature': y}

    df = pd.DataFrame(new_data, columns= ['Year', 'Temperature'])
    df.to_csv (r'./Temp.csv', index = False, header=True)
    return



def getDataAverageTemp(num):
    data = pd.read_csv('Temp.csv').head(num)   #load first num elems

    x = np.array(data['Year']).reshape((-1, 1))
    y = np.array(data['Temperature'])
    return x,y


if __name__ == "__main__":
    num = 161
    #prepareDataTemp()                  #uses for parsing ,htm file to csv
    #x,y = getDataWeightHeight(num)     #data with weight and height
    x,y = getDataAverageTemp(num)       #data with years and temp

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

    x = x.astype(float)
    log_x = x
    log_y = y
    for i in range (len(x)):
        log_x[i] = float(math.log(float(x[i])))
        log_y[i] = math.log(y[i])
        x_new[i] = math.log(x_new[i])

    model = LinearRegression().fit(log_x, log_y)
    print('y = ', model.intercept_, ' * x ^', model.coef_[0], )

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

