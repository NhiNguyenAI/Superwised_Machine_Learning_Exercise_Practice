import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
data_file = pd.read_csv("../../../data/raw/housing.csv")

data_file.info()

dataset = data_file.dropna()
dataset.info()


# --------------------------------------------------------------
# Export the Processed Dataset
# --------------------------------------------------------------
pickel_file = dataset.to_pickle("../../../data/interim/data_process.pkl")
