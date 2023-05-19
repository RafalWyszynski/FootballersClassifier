import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Wczytanie i wyświetlenie tabeli

stats_df = pd.read_csv(r"PlayersStats.csv", encoding='Windows-1250', index_col=1, sep=';')

# Sprawdzenie ogólnych informacji o tabeli
stats_df.info()

# Obróbka danych
rezerwowi = stats_df["Min"] < 500
stats_df = stats_df.loc[~rezerwowi]
stats_df = stats_df.drop(["Squad", "Comp", "Born", "Rk", "Nation", "Age", "MP", "Starts", "Min", "90s"], axis=1)
stats_df = stats_df.dropna()
stats_df["Pos"] = stats_df["Pos"].str.slice(stop=2)
print(stats_df.head())

#Inicjalizacja modelu
knn = KNeighborsClassifier()

# Tworzenie Foldów do walidacji krzyżowej
kf = KFold(n_splits=6, shuffle=True)

# Tworzenie siatki przeszukiwań
parameters = {"n_neighbors": list(range(1,20)), "weights": ["uniform", "distance"]}
stats_cv = GridSearchCV(knn, parameters, scoring="accuracy", cv=kf)

#Podział tabeli na cechy i etykiety
X = stats_df.drop("Pos", axis=1).values
y = stats_df["Pos"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#Skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#Tworzenie modelu
stats_cv.fit(X_train_scaled, y_train)
print(stats_cv.best_params_, stats_cv.best_score_)
predictions = stats_cv.score(X_test_scaled, y_test)
print("Predictions: {}".format(predictions))