'''
This file demonstrates and compares the use of two hyperparameter tuning techniques: **Grid Search** and **RandomizedSearchCV**. These methods are employed to optimize the performance of machine learning models by searching for the best combination of hyperparameters.

    ### Key Concepts Covered:
    1. **Grid Search**: This technique performs an exhaustive search through a manually specified parameter grid. It tries every possible combination of hyperparameters, making it thorough but computationally expensive.
    
    2. **RandomizedSearchCV**: In contrast, RandomizedSearchCV performs a random search over the specified hyperparameters. It is more efficient than Grid Search when there are many hyperparameters to tune, as it evaluates a limited number of random combinations, making it faster.

    ### Goal:
    In this file, we use both **Grid Search** and **RandomizedSearchCV** to tune the **Support Vector Regressor (SVR)** model on the **diabetes dataset** from `sklearn.datasets`. The performance of each search method is evaluated using **Mean Squared Error (MSE)** on the test set, and the results are compared.

    By the end of this file, you will be able to understand the strengths and trade-offs between these two methods of hyperparameter tuning and make informed decisions when applying them in your own machine learning projects.

'''

# Importing necessary libraries
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and prepare the dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# Initialize the SVM model
model = SVR()

# Create a parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

# Create a parameter distribution for RandomizedSearchCV
param_dist = {
    'C': np.logspace(-3, 3, 7),
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

# GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# RandomizedSearchCV
randomized_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, random_state=42)
randomized_search.fit(X_train, y_train)

# Print the best results from GridSearchCV
print(f"Best parameters from GridSearchCV: {grid_search.best_params_}")
print(f"Best score from GridSearchCV: {grid_search.best_score_}")

# Print the best results from RandomizedSearchCV
print(f"\nBest parameters from RandomizedSearchCV: {randomized_search.best_params_}")
print(f"Best score from RandomizedSearchCV: {randomized_search.best_score_}")

# Evaluate on the test set
y_pred_grid = grid_search.predict(X_test)
y_pred_random = randomized_search.predict(X_test)

mse_grid = mean_squared_error(y_test, y_pred_grid)
mse_random = mean_squared_error(y_test, y_pred_random)

print(f"\nMSE of GridSearchCV on test set: {mse_grid:.4f}")
print(f"MSE of RandomizedSearchCV on test set: {mse_random:.4f}")

