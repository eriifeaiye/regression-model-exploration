# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Group_1_file.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 0)

# Training the Random Forest Regression model on the training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 850)
regressor.fit(X_train[:,3:4], y_train)


# Visualizing the Random Forest Regression results (for higher resolution and
# smoother curve)
X_grid = np.arange(min(X_train[:,3:4]), max(X_train[:,3:4]) + 0.1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train[:,3:4], y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression (Training Set)')
plt.xlabel('Temperature')
plt.ylabel('Bike Count')
plt.show()