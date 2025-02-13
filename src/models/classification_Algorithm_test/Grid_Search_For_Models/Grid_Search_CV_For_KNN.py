from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# Check the structure and data types of the features
X.info()

# Check the skewness of the features and plot their distributions
X_skewness = X.skew()
sns.histplot(X)
plt.title("Feature Distributions")
plt.show()

# Check the skewness of the target variable and plot its distribution
sns.histplot(y, kde=True)
plt.title("Target Variable Distribution")
plt.show()

# Create the KNN model with default parameters
model = KNeighborsRegressor()

# This scatterplot helps visualize the relationship between the true target values (y_train) and itself.
# It creates a line of identity, meaning the points will lie along a straight diagonal line since x and y are the same.
sns.scatterplot(x=y_train, y=y_train)
sns.scatterplot(x=y_train, y=model.fit(X_train, y_train).predict(X_train), label=f'Model (train score: {model.score(X_train, y_train).round(2)})')

# Predict and calculate performance (Mean Squared Error) for the model without Grid Search
y_pred = model.predict(X_test)
mse_no_grid_search_without_grid_search = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (without Grid Search): {mse_no_grid_search_without_grid_search:.4f}")

# Define the parameter grid for Grid Search
p_grid = {
    'n_neighbors': np.arange(1, 31),  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weighting function for prediction
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm for computing nearest neighbors
    'leaf_size': np.arange(10, 50, 10),  # Leaf size for ball_tree/kd_tree algorithms
}

# Create a GridSearchCV object for KNN with cross-validation
grid = GridSearchCV(KNeighborsRegressor(), param_grid=p_grid, cv=5)

# Fit the GridSearchCV object on the training data
grid.fit(X_train, y_train)

# Display the grid search results
pd.DataFrame(grid.cv_results_).iloc[:, 5:]

# Output the best estimator, best parameters, and best score
print("Best Estimator:", grid.best_estimator_)
print("Best Parameters:", grid.best_params_)
print("Best Score (CV score):", grid.best_score_)

# Plotting predictions using the best KNN model from Grid Search
sns.scatterplot(x=y_train, y=y_train)
sns.scatterplot(x=y_train, y=model.predict(X_train), label=f'Model without Grid Search (train score: {model.score(X_train, y_train).round(2)})')
sns.scatterplot(x=y_train, y=grid.predict(X_train), label=f'Model with Grid Search (train score: {grid.score(X_train, y_train).round(2)})')

# Predict and calculate performance (Mean Squared Error) for the model with Grid Search
y_pred_grid_search = grid.predict(X_test)
mse_with_grid_search = mean_squared_error(y_test, y_pred_grid_search)
print(f"Mean Squared Error (with Grid Search): {mse_with_grid_search:.4f}")
