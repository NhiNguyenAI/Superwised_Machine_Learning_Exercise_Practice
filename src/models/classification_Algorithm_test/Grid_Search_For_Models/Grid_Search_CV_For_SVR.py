'''
This file is intended to practice implementing the Grid Search CV.
The Dataset: load_diabetes of the sklearn.datasets

'''

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# This line plots a scatterplot where the x-axis is y_train and the y-axis is also y_train.
# This essentially creates a line of identity, meaning the points will lie along a straight diagonal line (since x and y are the same). 
# This plot helps visualize the relationship between the true target values (y_train) and itself.
sns.scatterplot(x=y_train, y=y_train)
sns.scatterplot(x=y_train, y=model.predict(X_train), label = f'model_1 {model.score(X_train, y_train).round(2)}')

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

sns.scatterplot(x=y_train, y=y_train)
sns.scatterplot(x=y_train, y=model.predict(X_train), label = f'model_1 {model.score(X_train, y_train).round(2)}')
sns.scatterplot(x=y_train, y=grid.predict(X_train), label = f'model_1 {grid.score(X_train, y_train).round(2)}')