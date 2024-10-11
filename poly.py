# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Bike_Sharing.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 0)


# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_feature = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly_feature.fit_transform(X_train[:, 3:])
X_test_poly = poly_feature.fit_transform(X_test[:, 3:])
# Concatenate original matrix with polynomial feature matrix
X_new = np.concatenate((X_train[:, :3], X_poly), axis=1)
X_test_new = np.concatenate((X_test[:, :3], X_test_poly), axis=1)

regressor = LinearRegression()
regressor.fit(X_new, y_train)
regressor.fit(X_test, y_test)

# Finding the optimal degree using p-value and y-train
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
sc = StandardScaler()
y_train = sc.fit_transform(y_train.reshape(len(y_train), 1)).flatten()
y_test = sc.fit_transform(y_test.reshape(len(y_test), 1)).flatten()

poly_feature = PolynomialFeatures(degree=2) #include bias is true
regressor = sm.OLS(endog = y_train, exog = X_new).fit()
regressor.summary()


# Visualizing the Polynomial Regression results (for higher resolution and
# smoother curve) - Training Set
plt.scatter(X_new[:,3:4], y_train, color = 'red')
plt.title('Polynomial Regression (Training Set)')
plt.xlabel('Temperature')
plt.ylabel('Bike Count')
plt.show()

# Visualizing the Polynomial Regression results (for higher resolution and
# smoother curve) - Test Set
plt.scatter(X_test_new[:,3:4], y_test, color = 'red')
plt.title('Polynomial Regression (Test Set)')
plt.xlabel('Temperature')
plt.ylabel('Bike Count')
plt.show()

