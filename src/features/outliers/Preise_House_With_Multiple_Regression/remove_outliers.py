import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy
from Outlier_function import PlotBinaryOutliers as pbo
from Outlier_function import MarkOutlierIqr as iqr
from Outlier_function import MarkOutlierSchavenet as schauvenet
from Outlier_function import PlotBinaryOutliers as pbo
from Outlier_function import LocalOutlierFactor as lof

# ----------------------------------------------------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------------------------------------------------

df = pd.read_pickle("../../../../data/interim/data_process.pkl")
df.info()
# ----------------------------------------------------------------------------------------------------------------------
# 2 Adjust plot settings
# ----------------------------------------------------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-deep")
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100

# ----------------------------------------------------------------------------------------------------------------------
# 3 definite the name of columns
# columns_outliers is now included with index -> list()
# ----------------------------------------------------------------------------------------------------------------------
columns_outliers = list(df.columns[2:9])

# ----------------------------------------------------------------------------------------------------------------------
# 4.1 IQR for alll columns
# Boxplot
# look over all columns in plot_binary_outliers
# ----------------------------------------------------------------------------------------------------------------------

# Test single column: housing_median_age
fix, ax = plt.subplots(1,3,figsize=(20, 20))

df[columns_outliers[:3]+ ["ocean_proximity"]].boxplot(by="ocean_proximity", ax = ax)

# All column in one boxplot

# Create a 3x3 grid of subplots
fix, ax = plt.subplots(3, 3, figsize=(20, 20))

# Plot the first subset (columns 0, 1, 2) in the first row
for i, col in enumerate(columns_outliers[:3]):
    df[[col, "ocean_proximity"]].boxplot(by="ocean_proximity", ax=ax[0, i])

# Plot the second subset (columns 3, 4, 5) in the second row
for i, col in enumerate(columns_outliers[3:6]):
    df[[col, "ocean_proximity"]].boxplot(by="ocean_proximity", ax=ax[1, i])

# Plot the third subset (columns 6, 7, 8) in the third row
for i, col in enumerate(columns_outliers[6:9]):
    df[[col, "ocean_proximity"]].boxplot(by="ocean_proximity", ax=ax[2, i])

# Save the entire 3x3 grid plot
save_path = "../../../../reports/figures/Preis_House_With_Multiple_Regression/iqr/3x3_boxplots.png"
plt.savefig(save_path)



# Test single column: housing_median_age
fix, ax = plt.subplots(1,1,figsize=(20, 20))

df[[columns_outliers[0], "ocean_proximity"]].boxplot(by="ocean_proximity", ax = ax)
save_path = "../../../../reports/figures/Preis_House_With_Multiple_Regression/iqr/housing_median_age.png"
plt.savefig(save_path)


# Binary Outlier IQR: all columns Boxplot IQR
for col in columns_outliers:
    mark_outliers_iqr_df = iqr.mark_outliers_iqr(df, col)  
    save_path = f"../../../../reports/figures/Preis_House_With_Multiple_Regression/iqr/{col}_binary_plot.png"
    pbo.plot_binary_outliers(dataset=mark_outliers_iqr_df, col = col, outlier_col= col +"_outlier", reset_index=True, save_path=save_path)

# ----------------------------------------------------------------------------------------------------------------------
# 6 Outlier of sensor with Schauvenet
# ----------------------------------------------------------------------------------------------------------------------
df_schauvenet = df.copy()
df_schauvenet = df_schauvenet.reset_index(drop=True) 
for col in columns_outliers:
    mark_outliers_schauvenet_df = schauvenet.mark_outliers_chauvenet(df_schauvenet, col)
    save_path = f"../../../../reports/figures/Preis_House_With_Multiple_Regression/schauvenet/{col}_binary_plot.png"
    pbo.plot_binary_outliers(dataset=mark_outliers_schauvenet_df, col = col, outlier_col= col +"_outlier", reset_index=True, save_path=save_path)


