'''
    This file is designed to explore and implement the concepts of Overfitting and Underfitting in Machine Learning models, using three key techniques to optimize and evaluate model performance: Ridge Regression (L2 Regularization), and Lasso Regression (L1 Regularization).


        1. **Ridge Regression (L2 Regularization)**: Ridge Regression applies L2 regularization, which adds a penalty term to the loss function proportional to the square of the magnitude of the coefficients. This helps to prevent overfitting by constraining the model's complexity and reducing the impact of less important features. It is especially useful when there is multicollinearity among the features or when the number of features is large.

        2. **Lasso Regression (L1 Regularization)**: Lasso Regression uses L1 regularization, which adds a penalty term proportional to the absolute value of the coefficients. Lasso is particularly effective for feature selection as it tends to shrink the coefficients of less important features to zero, effectively eliminating them from the model. It helps both with overfitting and underfitting by reducing the model’s complexity and improving generalization.

    In this file, we implement and compare the performance of a Support Vector Regressor (SVR) model, first without any hyperparameter tuning and then with the application of Grid Search for hyperparameter optimization. We also explore the use of Ridge and Lasso regressions to address overfitting and underfitting.

    The goal is to evaluate how well each approach mitigates overfitting and underfitting issues, using metrics such as Mean Squared Error (MSE) and model performance scores (R-squared) on both the training and test datasets. By applying these techniques, we aim to develop a deeper understanding of how to optimize machine learning models for better generalization and accuracy.

    This file serves as a practical exercise for implementing these methods and gaining insight into the importance of model tuning and regularization in the machine learning workflow.
'''


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
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

# Create the model using Support Vector Regression (SVR)
model = SVR(C=1)
model.fit(X_train, y_train)

# Evaluate the model without Grid Search on both training and test data
y_pred = model.predict(X_test)
train_score_no_grid_search = model.score(X_train, y_train)
test_score_no_grid_search = model.score(X_test, y_test)
mse_train_no_grid_search = mean_squared_error(y_train, model.predict(X_train))
mse_test_no_grid_search = mean_squared_error(y_test, y_pred)


# Regularization: Ridge (L2) and Lasso (L1) models to address overfitting/underfitting

# Ridge Regression (L2 Regularization)
ridge_model = Ridge(alpha=1.0)  # Alpha is the regularization strength
ridge_model.fit(X_train, y_train)

# Lasso Regression (L1 Regularization)
lasso_model = Lasso(alpha=0.1)  # Alpha is the regularization strength
lasso_model.fit(X_train, y_train)

# Print Ridge and Lasso Results
print(f"\nRidge Model (L2 Regularization):")
print(f"Ridge Train Score: {ridge_model.score(X_train, y_train):.2f}, Test Score: {ridge_model.score(X_test, y_test):.2f}")
print(f"Ridge MSE (Train): {mean_squared_error(y_train, ridge_model.predict(X_train)):.2f}, MSE (Test): {mean_squared_error(y_test, ridge_model.predict(X_test)):.2f}")

print(f"\nLasso Model (L1 Regularization):")
print(f"Lasso Train Score: {lasso_model.score(X_train, y_train):.2f}, Test Score: {lasso_model.score(X_test, y_test):.2f}")
print(f"Lasso MSE (Train): {mean_squared_error(y_train, lasso_model.predict(X_train)):.2f}, MSE (Test): {mean_squared_error(y_test, lasso_model.predict(X_test)):.2f}")

# Plot the results for visualization
plt.figure(figsize=(12, 6))

# Plot for the model without Grid Search
sns.scatterplot(x=y_train, y=model.predict(X_train), label=f'Model without Grid Search (Train Score: {train_score_no_grid_search:.2f}, MSE (Train): {mse_train_no_grid_search:.2f})')
sns.scatterplot(x=y_train, y=y_train, label='Line of Identity')

# Plot for the model with Grid Search
sns.scatterplot(x=y_train, y=grid.predict(X_train), label=f'Model with Grid Search (Train Score: {train_score_with_grid_search:.2f}, MSE (Train): {mse_train_with_grid_search:.2f})')

# Plot for Ridge (L2) and Lasso (L1)
sns.scatterplot(x=y_train, y=ridge_model.predict(X_train), label=f'Ridge Model (Train Score: {ridge_model.score(X_train, y_train):.2f})')
sns.scatterplot(x=y_train, y=lasso_model.predict(X_train), label=f'Lasso Model (Train Score: {lasso_model.score(X_train, y_train):.2f})')

plt.legend()
plt.show()
