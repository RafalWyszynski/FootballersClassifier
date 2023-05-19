import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Wczytanie i wyświetlenie tabeli

stats_df = pd.read_csv(r"PlayersStats.csv", encoding='Windows-1250', index_col=0)
print(stats_df)

# Sprawdzenie ogólnych informacji o tabeli
stats_df.info()
stats_df.shape

# Obróbka danych
stats_df = stats_df.drop(["Team", "Nation", "Age"], axis=1)
stats_df = stats_df.dropna()
stats_df["Min"] = stats_df["Min"].str.replace(',', '').astype(float)
rezerwowi = stats_df["Min"] < 500
stats_df = stats_df.loc[~rezerwowi]
stats_df["Pos"] = stats_df["Pos"].str.split(',').str[0]
print(stats_df.head())


