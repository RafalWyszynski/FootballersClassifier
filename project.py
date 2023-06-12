import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Wczytanie i wyświetlenie tabeli
stats_df = pd.read_csv(r"PlayersStats.csv", encoding='Windows-1250', index_col=1, sep=';')
sztuczne_df = pd.read_csv(r"Sztuczne.csv", encoding='Windows-1250', index_col=0, sep=';')

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

#Inicjalizacja modelu
knn = KNeighborsClassifier()

# Tworzenie Foldów do walidacji krzyżowej
kf = KFold(n_splits=6, shuffle=True)

# Tworzenie siatki przeszukiwań
parameters = {"n_neighbors": list(range(1, 50)),
              "weights": ["uniform", "distance"],
              "metric": ["euclidean", "manhattan", "chebyshev"]}
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

#Wyświetlanie najlepszych parametrów i wyniku
print("Best Parameters: {}\nBest Score: {}".format(stats_cv.best_params_, stats_cv.best_score_))
print("Predicted values score: {}".format(stats_cv.score(X_test_scaled, y_test)))
predictions = stats_cv.predict(X_test_scaled)

#Tworzenie DataFrame'u porównującego wartości przewidziane z rzeczywistymi
comparison_df = pd.DataFrame({"Predicitons": predictions, "True Values": y_test})
print(comparison_df.iloc[0:10, :])

# Wykres boxplota dla wartości dokładności w GridSearchCV
cv_results = stats_cv.cv_results_
accuracy_values = cv_results['mean_test_score']

#Sztuczne dane:
X_szt = sztuczne_df.drop("Pos", axis=1).values
y_szt = sztuczne_df["Pos"].values
sztuczne = stats_cv.predict(X_szt)
comparison_df2 = pd.DataFrame({"Predicitons": sztuczne, "True Values": y_szt})
print(comparison_df2.iloc[0:10, :])


#Rysowanie boxplota
plt.boxplot(accuracy_values, patch_artist=True)
plt.title("Accuracy Values - Cross-validation")
plt.xlabel("Parameter Combinations")
plt.ylabel("Accuracy")
plt.show()