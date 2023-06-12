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
napastnicy = stats_df['Pos'] == 'FW'
pomocnicy = stats_df['Pos'] == 'MF'
obroncy = stats_df['Pos'] == 'DF'
bramkarze = stats_df['Pos'] == 'GK'
napastnicy_df = stats_df.loc[napastnicy]
stats_df['CrdY'] = stats_df['CrdY'].multiply(10)
stats_df['CrdY'] = stats_df['CrdY'].round(0).astype(int)

data = [stats_df.loc[napastnicy, 'CrdY'],
        stats_df.loc[pomocnicy, 'CrdY'],
        stats_df.loc[obroncy, 'CrdY'],
        stats_df.loc[bramkarze, 'CrdY']]

labels = ['Napastnicy', 'Pomocnicy', 'Obroncy', 'Bramkarze']

# Tworzenie wykresu z boxplotami na jednej osi
boxplot = plt.boxplot(data, labels=labels, patch_artist=True)

# Ustawienie kolorów pudełek
colors = ['skyblue', 'lightgreen', 'lightyellow', 'lightpink']

for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

# Ustawienie etykiety osi y
plt.ylabel('Liczba żółtych kartek')

# Ustawienie tytułu wykresu
plt.title('Rozkład żółtych kartek według pozycji')

# Wyświetlenie wykresu
plt.show()