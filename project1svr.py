# Project 1
# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Bike_Sharing.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X_train[:, 3:])


# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') 
regressor.fit(X_scaled, y_train)
regressor.fit(X_train[:, 3:4], y_train)

# Visualizing the SVR test set results
plt.scatter(X_train[:, 3:4], y_train, color = 'red')
# =============================================================================
# y_pred_scaled = regressor.predict(X_scaled)
# y_pred = sc.inverse_transform(pred_scaled.reshape(len(y_pred_scaled), 1))
# plt.plot(X, y_pred, color = 'blue')
# =============================================================================
plt.title('SVR (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Bike count')
plt.show()


# Visualizing the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X_test[:, 3:4]), max(X_test[:, 3:4]) + 0.1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1)) # reshape to get the matrix 
plt.scatter(X_test[:, 3:4], y_test, color = 'red')
# =============================================================================
# plt.plot(X_grid, y_grid_pred, color = 'blue')
# =============================================================================
plt.title('SVR (Test set)')
plt.xlabel('Temperature')
plt.ylabel('Bike count')
plt.show()




