# Wstępna analiza danych: średnia, mediana, minimum, maksimum, odchylenie standardowe, skośność
# boxploty, top 10 piłkarzy - histogram
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie i wyświetlenie tabeli
stats_df = pd.read_csv(r"PlayersStats.csv", encoding='Windows-1250', index_col=1, sep=';')

# Sprawdzenie ogólnych informacji o tabeli
stats_df.info()

# Obróbka danych
rezerwowi = stats_df["Min"] < 300
stats_df = stats_df.loc[~rezerwowi]
stats_df = stats_df.drop(["Squad", "Comp", "Born", "Rk", "Nation", "Age", "MP", "Starts", "Min",
                          "90s", "PaswRight", "PaswLeft", "PaswOther", "GcaFld", "CrdY", "CrdR",
                          "2CrdY"], axis=1)
stats_df = stats_df.dropna()
stats_df["Pos"] = stats_df["Pos"].str.slice(stop=2)
print(stats_df.head())
print(stats_df.info())


# Tworzenie histogramów dla kolumn numerycznych
numeric_columns = stats_df.select_dtypes(include=[np.number]).columns

# Wczytanie nazw kolumn z pliku "names.txt"
with open("names.txt", "r") as file:
    column_names = dict(line.strip().split(";") for line in file)

# Tworzenie folderu do przechowywania obrazów, jeśli nie istnieje
if not os.path.exists("histogram_images"):
    os.makedirs("histogram_images")

for column in numeric_columns:
    full_name = column_names.get(column, column)
    plt.hist(stats_df[column], bins=10)
    plt.title(f"{full_name}")
    plt.xlabel(full_name)
    plt.ylabel("Frequency")
    plt.savefig(f"histogram_images/{column}_histogram.png")
    plt.close()