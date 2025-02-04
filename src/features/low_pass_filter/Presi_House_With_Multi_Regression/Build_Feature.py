import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data_Transformation.DataTransformation import PrincipalComponentAnalysis
from Data_Transformation.DataTransformation import PrincipalComponentAnalysis
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler

# ----------------------------------------------------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------------------------------------------------
df = pd.read_pickle("../../../../data/interim/outliers_removed_schauvenets.pkl")
predictor_columns = list(df.columns[:8])

# ----------------------------------------------------------------------------------------------------------------------
# 2. Adjust plot settings
# ----------------------------------------------------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-deep")
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100

# ----------------------------------------------------------------------------------------------------------------------
# 3. Dealing with missing values (imputation)
# ----------------------------------------------------------------------------------------------------------------------
df.info()  # Info, how many data are missing
subset = df[df["ocean_proximity"]== "INLAND"]
subset["total_bedrooms"].plot()

# Interpolation for missing values
for col in predictor_columns:
    df[col] = df[col].interpolate()
df.info()

# ----------------------------------------------------------------------------------------------------------------------
# 4. Feature Engineering: Add custom features
# ----------------------------------------------------------------------------------------------------------------------

## 1. Price per room
df['price_per_room'] = df['median_house_value'] / df['total_rooms']

## 2. Price per household
df['price_per_household'] = df['median_house_value'] / df['households']

## 3. Population density
df['population_density'] = df['population'] / df['total_rooms']

## 4. Age-related feature (New vs Old Houses)
df['is_new_house'] = df['housing_median_age'] < 20  # 20 years or less could be considered "new"

## 5. Log transformation of skewed features (Median House Value)
df['log_median_house_value'] = np.log1p(df['median_house_value'])

## 6. Log transformation of skewed features (Median Income)
df['log_median_income'] = np.log1p(df['median_income'])

## 7. Interaction term (Rooms * Households)
df['rooms_per_household'] = df['total_rooms'] / df['households']

# ## 8. One-Hot Encoding for categorical variable 'ocean_proximity'
# df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)


# ----------------------------------------------------------------------------------------------------------------------
# 5. Scaling: Apply StandardScaler or MinMaxScaler
# ----------------------------------------------------------------------------------------------------------------------

scaler = StandardScaler()
# Apply the scaler to the selected columns (e.g., the numerical features you want to scale)
scaled_features = scaler.fit_transform(df[predictor_columns])

# Convert the scaled features back to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=predictor_columns)

# Replace original columns with scaled features
df[predictor_columns] = scaled_df

# Alternatively, you can use MinMaxScaler (scaling between 0 and 1):
# min_max_scaler = MinMaxScaler()
# scaled_features_min_max = min_max_scaler.fit_transform(df[poly_columns])
# scaled_df_min_max = pd.DataFrame(scaled_features_min_max, columns=poly_columns)
# df[poly_columns] = scaled_df_min_max

# ----------------------------------------------------------------------------------------------------------------------
# 6. Principal Component Analysis (PCA)
# ----------------------------------------------------------------------------------------------------------------------
# Copy the original 'ocean_proximity' column before any transformation
df_pca = df.copy()

# Ensure the 'ocean_proximity' column is available for One-Hot Encoding
# If you removed the 'ocean_proximity' column earlier, add it back like this:
df_pca['ocean_proximity'] = df['ocean_proximity']

# Apply One-Hot Encoding for 'ocean_proximity' column
df_pca = pd.get_dummies(df_pca, columns=['ocean_proximity'], drop_first=True)

# Apply PCA with the desired number of components
PCA = PrincipalComponentAnalysis()
df_pca = PCA.apply_pca(df_pca, predictor_columns, 5)

# Example: Take a look at a subset
subset = df_pca[df_pca["ocean_proximity_INLAND"] == 1]  # Checking the one-hot encoded column
subset[["pca_1", "pca_2", "pca_3", "pca_4", "pca_5"]].plot()


