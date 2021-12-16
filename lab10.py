import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def generate(mean_x: float, mean_y: float, var_x: float, var_y: float, size: int) -> list:
    x = norm.rvs(loc=mean_x, scale=var_x, size=size)
    y = norm.rvs(loc=mean_y, scale=var_y, size=size)
    return [[x_i, y_i] for x_i, y_i in zip(x, y)]


def k_mean(data: list, title: str) -> [list, list]:
    model = KMeans(n_clusters=3)
    model.fit(data)
    centers = model.cluster_centers_
    labels = model.labels_
    draw(data, labels, centers, title)
    return labels, centers


def k_neighbors(train_x: list, train_y: list, test_x: list, test_y: list, centres: list, title: str) -> None:
    errors = []
    k, max_k, min_error = 0, 20, 0
    for i in range(1, max_k):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        mean = np.mean(predicted != test_y)
        if mean < min_error or min_error == 0:
            k = i
            min_error = mean
        errors.append(mean)

    plt.title("K-neighbors dependence")
    plt.plot(np.arange(1, max_k), errors)
    plt.show()
    print(f"K = {k}, error in k-neighbors = {min_error}")

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    draw(test_x, predicted, centres, title)


def bayes(train_x: list, train_y: list, test_x: list, test_y: list, centers: list, title: str) -> None:
    model = GaussianNB()
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    draw(test_x, predicted, centers, title)
    print(f"Error in Bayes = {np.mean(predicted != test_y)}")


def draw(data: list, labels: list, centers: list, title: str) -> None:
    plt.title(title)
    colors = ['c', 'm', 'y']
    for index in range(len(data)):
        plt.scatter(data[index][0], data[index][1], c=colors[labels[index]])
    for center in centers:
        plt.scatter(center[0], center[1], marker='^', c='k')
    plt.show()


def lab10():
    size = 200
    data_1 = generate(3, 3, 1.5, 1.5, size)
    data_2 = generate(9, 2, 1, 1, size)
    data_3 = generate(9, 6, 1, 1, size)
    data = data_1 + data_2 + data_3
    labels, centers = k_mean(data, "K-means clustering")
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.4)
    k_neighbors(train_x, train_y, test_x, test_y, centers, "K-neighbors classifier")
    bayes(train_x, train_y, test_x, test_y, centers, "Naive Bayesian classifier")
