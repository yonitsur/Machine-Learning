import numpy as np
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import pandas as pd

def hoeffding():
    N = 200000
    n = 20
    X = np.random.binomial(size=(N,n), n=1, p=0.5)
    em = np.abs(np.mean(X, axis=1)-0.5) 
    eps = np.linspace(0,1, 50)
    empirical_prob = np.array([np.sum(em>eps[i])/N for i in range(50)])
    plt.plot(eps, empirical_prob, "b", label="empirical")
    plt.plot(eps, 2*np.exp(-2*n*(eps**2)), "g", label="hoeffding")
    plt.show()

def kNN(data, lables, image, k):
  dist = np.array([np.linalg.norm(data[i] - image) for i in range(len(data))])
  k_nearest_neighbors = np.argsort(dist)[:k]
  k_labels = np.take(lables, k_nearest_neighbors).astype(int)
  return np.bincount(k_labels).argmax()

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
idx = np.random.RandomState(0).choice(70000, 11000)
train_data = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

n=1000
test_results = np.zeros(1000)
k = np.arange(1, 101)
k_results = np.zeros(100)
for i in range(100):
    for j in range(len(test_results)):
        test_results[j] = kNN(train_data[:n], train_labels[:n], test[j], k[i])
    k_results[i] = np.linalg.norm(test_results == test_labels.astype(int), 1)/len(test_results)

plt.plot(k, k_results)
plt.show()

test_results = np.zeros(1000)
N = np.arange(100, 5001, 100)
N_results = np.zeros(50)
for i in range(50):
    for j in range(len(test_results)):
        test_results[j] = kNN(train_data[:N[i]], train_labels[:N[i]], test[j], 1)
    N_results[i] = np.linalg.norm(test_results == test_labels.astype(int), 1)/len(test_results)

plt.plot(N, N_results)
plt.show()