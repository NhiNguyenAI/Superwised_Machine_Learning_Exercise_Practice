import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from Outlier_function import PlotBinaryOutliers as pbo
from Outlier_function import MarkOutlierIqr as iqr
from Outlier_function import MarkOutlierSchavenet as schauvenet
from Outlier_function import PlotBinaryOutliers as pbo
from Outlier_function import LocalOutlierFactor as lof

# ----------------------------------------------------------------------------------------------------------------------
# 1. Load data
# Note: forgot in the frist step at make_data_set change the ocean_proximity to object
# delete index column to use canvenet outliers
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
columns_outliers = list(df.columns[:9])

# ----------------------------------------------------------------------------------------------------------------------
# 4.1 IQR for alll columns: Boxplot
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

# ----------------------------------------------------------------------------------------------------------------------
# 4.2 IQR for alll columns: Binary plot
# ----------------------------------------------------------------------------------------------------------------------

df_iqr = df.copy()
for col in columns_outliers:
    mark_outliers_iqr_df = iqr.mark_outliers_iqr(df_iqr, col)  
    save_path = f"../../../../reports/figures/Preis_House_With_Multiple_Regression/iqr/{col}_binary_plot.png"
    pbo.plot_binary_outliers(dataset=mark_outliers_iqr_df, col = col, outlier_col= col +"_outlier", reset_index=True, save_path=save_path)

# ----------------------------------------------------------------------------------------------------------------------
# 6 Outlier of sensor with Schauvenet
# ----------------------------------------------------------------------------------------------------------------------

df_schauvenet = df.copy()
#df_schauvenet = df_schauvenet.reset_index(drop=True) 
for col in columns_outliers:
    mark_outliers_schauvenet_df = schauvenet.mark_outliers_chauvenet(df_schauvenet, col)
    #save_path = f"../../../../reports/figures/Preis_House_With_Multiple_Regression/schauvenet/{col}_binary_plot.png"
    pbo.plot_binary_outliers(dataset=mark_outliers_schauvenet_df, col = col, outlier_col= col +"_outlier", reset_index=True)

# ----------------------------------------------------------------------------------------------------------------------
# 7 Outlier of sensor with local outlier factor
# ----------------------------------------------------------------------------------------------------------------------

df_lof = df.copy()
#df_lof = df_lof.reset_index(drop=True)
mark_outliers_lof_df, outliers, X_score = lof.mark_outliers_lof(df_lof, columns_outliers)
for col in columns_outliers:
    save_path = f"../../../../reports/figures/Preis_House_With_Multiple_Regression/lof/{col}_binary_plot.png"
    pbo.plot_binary_outliers(dataset=mark_outliers_lof_df, col = col, outlier_col= "outlier_lof", reset_index=True, save_path=save_path)

# ----------------------------------------------------------------------------------------------------------------------
# 8 Choose method and deal with outliers
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------
# 8.1 Test on single column
# --------------------------------------------------------------

df_schauvenet_test = df.copy()
df_schauvenet_test = df_schauvenet_test.reset_index(drop= True)

df_schauvenet_test =schauvenet.mark_outliers_chauvenet(df_schauvenet_test, col = "median_income")
# show the value at outlier point
df_schauvenet_test[df_schauvenet_test["median_income_outlier"]]

## set outliers value to NAN
                                    
df_schauvenet_test.loc[df_schauvenet_test["median_income_outlier"], "median_income"]= np.nan
df_schauvenet_test.info()
len(df_schauvenet_test[df_schauvenet_test["median_income_outlier"]])

# --------------------------------------------------------------
# 8.2 Test on single column with each ocean_proximity
# --------------------------------------------------------------

# Array of the ocean_proximity column
label = df["ocean_proximity"].unique()
near_bay_label = label[0]
col = "median_income"

df_schauvenet_test = df.copy()
df_schauvenet_test.info()

df_schauvenet_test = df_schauvenet_test[df_schauvenet_test["ocean_proximity"]== near_bay_label].reset_index(drop= True)
df_near_bay = schauvenet.mark_outliers_chauvenet(df_schauvenet_test, col = col)

df_near_bay.loc[df_near_bay["median_income_outlier"], "median_income"]= np.nan
len(df_near_bay[df_near_bay["median_income_outlier"]])

df_near_bay.info()

# --------------------------------------------------------------
# 8.2 Create new dataframe for all column with method canvenet
# --------------------------------------------------------------

outliers_removed_df = df.copy()

for col in columns_outliers:
    make_outliers_dataset = schauvenet.mark_outliers_chauvenet(df, col)

    # Replace values marked as outliers with NaN
    make_outliers_dataset.loc[make_outliers_dataset[col + "_outlier"], col] = np.nan

    # update the column in the original dataframe
    outliers_removed_df[col] = make_outliers_dataset[col]

    n_outliers = len(make_outliers_dataset) - len(make_outliers_dataset[col].dropna())

    print(f"Removed {n_outliers} from {col}")

outliers_removed_df.info()

# ----------------------------------------------------------------------------------------------------------------------
# 9 Check again with original dataframe
# ----------------------------------------------------------------------------------------------------------------------
sns.scatterplot(y='median_house_value',x="total_bedrooms",data=outliers_removed_df)
sns.scatterplot(y='median_house_value',x="total_bedrooms",data=df)

sns.scatterplot(y='median_house_value',x="households",data=outliers_removed_df)
sns.scatterplot(y='median_house_value',x="households",data=df)

sns.scatterplot(x='median_house_value',y="housing_median_age",data=outliers_removed_df)
sns.scatterplot(x='median_house_value',y="housing_median_age",data=df)

sns.scatterplot(y='median_house_value',x="population",data=outliers_removed_df)
sns.scatterplot(y='median_house_value',x="population",data=df)

sns.scatterplot(y='median_house_value',x="median_income",data=outliers_removed_df)
sns.scatterplot(y='median_house_value',x="median_income",data=df)

# ----------------------------------------------------------------------------------------------------------------------
# 10 covert the column ocean_proximity to numeric by labelencoder
# ----------------------------------------------------------------------------------------------------------------------
# df_test = df.copy()
# l = LabelEncoder()
# df_test['ocean_proximity'] = l.fit_transform(df_test['ocean_proximity'])
# df_test['ocean_proximity'].unique()

outliers_removed_df.to_pickle("../../../../data/interim/outliers_removed_schauvenets.pkl")