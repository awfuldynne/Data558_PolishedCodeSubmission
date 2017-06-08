from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_auto_data():
    AutoData = namedtuple("AutoData", ["Target", "Data", "Columns"])
    auto_data_url = "http://www-bcf.usc.edu/~gareth/ISL/Auto.csv"
    auto_data = pd.read_table(auto_data_url, sep=",", na_values="?")

    # Subset data to cars with 6 and 8 cylinders
    indexes = auto_data[auto_data.cylinders.isin([6,8])].index.tolist()
    auto_data = auto_data.loc[indexes, :]

    # Convert 6 to -1 and 8 to +1
    target = auto_data.cylinders
    target = target.replace(6, -1).replace(8, 1)

    # Subset DataFrame to features
    auto_data = auto_data.drop(["cylinders", "name", "horsepower"], axis=1)
    scaler = StandardScaler()
    data_set = scaler.fit_transform(auto_data)

    return AutoData(np.array(target), np.array(data_set), list(auto_data.columns))
