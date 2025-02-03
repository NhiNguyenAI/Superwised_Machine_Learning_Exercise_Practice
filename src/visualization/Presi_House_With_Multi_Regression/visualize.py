import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
import seaborn as sns
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------
# Adjust plot settings
# ----------------------------------------------------------------------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams['figure.figsize'] = (20,5)
mpl.rcParams['figure.dpi'] = 100

# ----------------------------------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------------------------------
df = pd.read_pickle("../../../data/interim/data_process.pkl")
df.info()

# ----------------------------------------------------------------------------------------------------------------------------
# Plot single columns
# ----------------------------------------------------------------------------------------------------------------------------
set_df = df[df["ocean_proximity"] == "NEAR BAY"]
fig, ax = plt.subplots()
set_df[["total_rooms"]].plot(ax=ax)  # Pass the axis object
ax.set_ylabel("Index")
ax.set_xlabel("Samples")
plt.legend(["total_rooms", "total_bedrooms","median_house_value"])  # Explicitly set legend labels
plt.show()

# ----------------------------------------------------------------------------------------------------------------------------
# Plot columns ocean_proximity
#----------------------------------------------------------------------------------------------------------------------------

df_housing = df.copy()

# Explore the column ocean_proximity
ocean_values = df_housing["ocean_proximity"].value_counts()

plt.figure(figsize=(10,6))
sns.countplot(x = "ocean_proximity",data=df_housing,order=ocean_values.index)

# ----------------------------------------------------------------------------------------------------------------------------
# Plot  history columns ocean_proximity
# ----------------------------------------------------------------------------------------------------------------------------

# showing the percenotge
for i in range(ocean_values.shape[0]):
    count = ocean_values[i] 
    strt='{:0.2f}%'.format(100*count / df_housing.shape[0]) 
    plt.text(i, count+100, strt, ha='center', color='black', fontsize=14)

# Histogram
df_housing.hist(bins=25,figsize=(20,10))

# check scatter plot between median_income and median_house_value
plt.scatter(df_housing["median_income"],df_housing["median_house_value"], alpha=0.1,color="g")

# ----------------------------------------------------------------------------------------------------------------------------
# Plot median_house_value with any value between range (0-100k)
# ----------------------------------------------------------------------------------------------------------------------------

## In the foLLowing example -- any value between range (0-100k) will be the same category,I name it (0-100k) 

house_value_bins = pd.cut(x=df_housing["median_house_value"],
                          bins=(-np.inf, 100000, 200000, 300000, 400000, 500000, np.inf),
                                labels=('-inf to 100k', '100k to 200k', '300k to 400k', '400k to 500k', '500k to 600k', '600k to inf') )
## countpLot for the above chunks 
plt.figure(figsize=(15,6)) 
sns.countplot(x=house_value_bins) 
plt.title('CountPlot of House Value Bins in Dataset', fontsize=14, c='k') 
plt.xlabel('House Value Bins', fontsize=14, c='k') 
plt.ylabel('counts', fontsize=14,c='k') 
plt.show() 