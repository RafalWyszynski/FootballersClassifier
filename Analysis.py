import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(36)

# Wczytanie i wyświetlenie tabeli
stats_df = pd.read_csv(r"PlayersStats.csv", encoding='Windows-1250', index_col=1, sep=';')

# Sprawdzenie ogólnych informacji o tabeli
stats_df.info()

# Obróbka danych
rezerwowi = stats_df["Min"] < 300
stats_df = stats_df.loc[~rezerwowi]
stats_df = stats_df.dropna()
stats_df["Pos"] = stats_df["Pos"].str.slice(stop=2)
print(stats_df.head())
print(stats_df.info())
print(stats_df.columns)
stats_df['CrdY'] = stats_df['CrdY'].multiply(10)
stats_df['CrdY'] = stats_df['CrdY'].round(0).astype(int)
stats_df['Goals'] = stats_df['Goals'].multiply(30)
stats_df['Goals'] = stats_df['Goals'].round(0).astype(int)
stats_df['Assists'] = stats_df['Assists'].multiply(20)
stats_df['Assists'] = stats_df['Assists'].round(0).astype(int)
podsumowanie = stats_df.describe()
podsumowanie.to_excel('podsumowanie.xlsx', index=True)
stats_df = stats_df.drop(['Pos',"Squad", "Comp", "Born", "Rk", "Nation", "Age", "MP", "Starts", "Min",
                          "90s", "PaswRight", "PaswLeft", "PaswOther", "GcaFld", "CrdY", "CrdR",
                          "2CrdY"], axis=1)
correlation_matrix = stats_df.corr()
correlation_matrix.to_excel('korelacje.xlsx', index=True)
print(correlation_matrix)