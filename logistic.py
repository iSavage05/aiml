import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_regression(X, y, num_iterations=200, learning_rate=0.001):
    weights = np.zeros(X.shape[1])
    for _ in range(num_iterations):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient_val = np.dot(X.T, (h - y)) / y.shape[0]
        weights -= learning_rate * gradient_val
    return weights

iris = load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=9)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

weights = logistic_regression(X_train_std, y_train)

y_pred = sigmoid(np.dot(X_test_std, weights)) > 0.5

print(f'Accuracy: {np.mean(y_pred == y_test):.4f}')
