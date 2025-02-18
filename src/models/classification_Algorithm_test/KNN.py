'''
This file is intended to practice implementing the K-Nearest Neighbors algorithm. 
This Project for predicting diabetes using the K-Nearest Neighbors algorithm.
Dataset: diabetes.csv

we use numpy, pandas, and sklearn libraries with the following functions:
    - train_test_split: to split the dataset into training and testing sets.
    - StandardScaler: to scale the features.
    - confusion_matrix: to evaluate the model.
    - accuracy_score: to evaluate the model.
    - f1_score: to evaluate the model.
    - KNeighborsClassifier: to build the K-Nearest Neighbors model.
    - GridSearchCV: to find the best parameters for the model.
    - Pipeline: to build a pipeline for the model.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import math
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import itertools

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------------------------
df = pd.read_csv("../../../data/interim/diabetes.csv")
df.info()

# --------------------------------------------------------------------------------------------------
# Replace zero values with the mean of respective columns
# --------------------------------------------------------------------------------------------------
columns_without_zero_values = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in columns_without_zero_values:
    df[col] = df[col].replace(0, np.nan)
    mean = int(df[col].mean(skipna=True))
    df[col] = df[col].replace(np.nan, mean)

# --------------------------------------------------------------------------------------------------
# Split the data into training and testing sets
# --------------------------------------------------------------------------------------------------
df_train = df.copy()
X = df_train.drop("Outcome", axis=1)
y = df_train["Outcome"]

# Take 20% of dataset for the test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#--------------------------------------------------------------------------------------------------
# Feature Scaling
# --------------------------------------------------------------------------------------------------
sc_X = StandardScaler()

# Fit the scaler to the training data and apply the transformation
X_train = sc_X.fit_transform(X_train)

# Apply the same transformation to the test data
X_test = sc_X.transform(X_test)

# --------------------------------------------------------------------------------------------------
# Create the K-Nearest Neighbors model
# --------------------------------------------------------------------------------------------------

def k_nearest_neighbors(train_X, train_y, test_X, n_neighbors=5, gridsearch=True, print_model_details=False):
    """
    This function builds a K-Nearest Neighbors model and returns the predictions.
    :param train_X: The features of the training set.
    :param train_y: The labels of the training set.
    :param test_X: The features of the testing set.
    :param n_neighbors: The number of neighbors to consider.
    :param gridsearch: Whether to use GridSearchCV to find the best parameters.
    :param print_model_details: Whether to print the model details.
    :return: The predictions.
    """
    
    if gridsearch:
        tuned_parameters = [{"n_neighbors": [1, 2, 5, 10]}]
        knn = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring="accuracy")
    else:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model
    knn.fit(train_X, train_y)

    if gridsearch and print_model_details:
        print(knn.best_params_)

    if gridsearch:
        knn = knn.best_estimator_

    # Apply the model
    pred_prob_training_y = knn.predict_proba(train_X)
    pred_prob_test_y = knn.predict_proba(test_X)
    pred_training_y = knn.predict(train_X)
    pred_test_y = knn.predict(test_X)

    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

# --------------------------------------------------------------------------------------------------
# Feature Selection
# --------------------------------------------------------------------------------------------------
normal_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Ensure `set_features` is a list, not a set (for column order consistency)
set_features = normal_features

# Select the relevant features
selected_train_X = X_train[:, [X.columns.get_loc(col) for col in set_features]]
selected_test_X = X_test[:, [X.columns.get_loc(col) for col in set_features]]

# --------------------------------------------------------------------------------------------------
# Train the K-Nearest Neighbors model with the k_neaeest_neighbors function
# --------------------------------------------------------------------------------------------------
score_df = pd.DataFrame()
print("\tTraining KNN")
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = k_nearest_neighbors(selected_train_X, y_train, selected_test_X, gridsearch=True)

# Evaluate the model's performance on the test data
performance_test_knn = accuracy_score(y_test, class_test_y)

# Save results to a dataframe
models = ["KNN"]
new_scores = pd.DataFrame(
    {
        "model": models,
        "accuracy": [performance_test_knn],
    }
)

score_df = pd.concat([score_df, new_scores])
accuracy = accuracy_score(y_test, class_test_y)

# --------------------------------------------------------------------------------------------------
# Train the K-Nearest Neighbors model with
# --------------------------------------------------------------------------------------------------

# n_neighbors = sqrt(len(y_test)) rounded and subtracted by 1 as per your logic
n_neighbors = int(math.sqrt(len(y_test))) - 1  # n_neighbors should be an integer

# Train the KNN classifier
classifier = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='euclidean')
classifier.fit(selected_train_X, y_train)

# Make predictions
class_test_y = classifier.predict(selected_test_X)
class_test_prob_y = classifier.predict_proba(selected_test_X)

# Create confusion matrix using actual test labels (y_test) and predicted labels (class_test_y)
cm = confusion_matrix(y_test, class_test_y)

# Plotting the confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# Define class labels (binary in this case, 0 and 1)
classes = np.unique(y_test)

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Threshold for text color: white for higher values, black for lower values
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
