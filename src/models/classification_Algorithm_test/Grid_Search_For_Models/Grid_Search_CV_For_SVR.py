'''
This file is intended to practice implementing Grid Search Cross Validation (Grid Search CV) to optimize model parameters.

Dataset used: load_diabetes from sklearn.datasets.

The model applied in this file could be any machine learning model, such as linear regression or another model, and Grid Search CV will be used to find the best hyperparameters (e.g., alpha in Ridge model or other parameters).

Main steps in the file:
1. Load and prepare the dataset from `load_diabetes`.
2. Build the machine learning model (e.g., linear regression, Ridge, etc.).
3. Apply Grid Search CV to find the optimal parameters for the model.
4. Evaluate the model with the best parameters using metrics like R-squared and MSE.
5. Plot scatter plots to compare the actual values and predicted values from the model.

Metrics like R-squared and Mean Squared Error (MSE) will help assess the model’s performance, providing a better understanding of its accuracy and how well it fits the data.

'''

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# If dont use the as_frame=True, the data will be numpy array
X , y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# Check data of the dataset
X.info()
# Check the skewness of the features
X.skew()
sns.histplot(X, kde=True)
# Check the skewness of the target variable
sns.histplot(y, kde=True)
plt.show()

# Create the model

model = SVR(C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse_no_grid_search_without_grid_search = mean_squared_error(y_test, y_pred)

# This line plots a scatterplot where the x-axis is y_train and the y-axis is also y_train.
# This essentially creates a line of identity, meaning the points will lie along a straight diagonal line (since x and y are the same).
# This plot helps visualize the relationship between the true target values (y_train) and itself.
sns.scatterplot(x=y_train, y=y_train)
sns.scatterplot(x=y_train, y=model.predict(X_train), label = f'model without the grid search CV {model.score(X_train, y_train).round(2)} with Mean Squared Error {mse_no_grid_search_without_grid_search}')


# Create the Grid Search CV
# Reason for using the Grid Search CV: To find the best hyperparameters for the model

# SVR(def __init__(
    #     self,
    #     *,
    #     kernel="rbf",
    #     degree=3,
    #     gamma="scale",
    #     coef0=0.0,
    #     tol=1e-3,
    #     C=1.0,
    #     epsilon=0.1,
    #     shrinking=True,
    #     cache_size=200,
    #     verbose=False,
    #     max_iter=-1,
    # ):
# This is the parameter grid that we will use to search for the best hyperparameters for the SVR model.

SVR()
p_grid = {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'degree': np.arange(1, 10),
            'C': np.logspace(-3, 3, 7),
          }

grid = GridSearchCV(model, param_grid=p_grid, cv=5)

grid.fit(X_train, y_train)

pd.DataFrame(grid.cv_results_).iloc[:, 5:]

grid.best_estimator_

grid.best_params_

grid.best_score_

y_pred_grid_search = grid.predict(X_test)
mse_no_grid_search_with_grid_search = mean_squared_error(y_test, y_pred_grid_search)

sns.scatterplot(x=y_train, y=y_train, label = 'Line of Identity')
sns.scatterplot(x=y_train, y=model.predict(X_train), label = f'model without Grid Search {model.score(X_train, y_train).round(2)} with Mean Squared Error {mse_no_grid_search_without_grid_search}')
sns.scatterplot(x=y_train, y=grid.predict(X_train), label = f'model with Grid Search {grid.score(X_train, y_train).round(2)} with Mean Squared Error {mse_no_grid_search_with_grid_search} ')