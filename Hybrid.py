import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(36)

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

# Podział tabeli na cechy i etykiety
X = stats_df.drop("Pos", axis=1).values
y = stats_df["Pos"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Inicjalizacja modeli
rf = RandomForestClassifier()
nb = GaussianNB()
knn = KNeighborsClassifier()

# Tworzenie Foldów do walidacji krzyżowej
kf = KFold(n_splits=6, shuffle=True)

# Tworzenie siatek przeszukiwań dla poszczególnych modeli
rf_parameters = {"n_estimators": [16, 32, 64, 128],
                 "max_depth": [2, 4, 8, 12, 13, 14],
                 "min_samples_split": [2, 4, 8, 16]}
rf_cv = GridSearchCV(rf, rf_parameters, scoring="accuracy", cv=kf)

nb_parameters = {'var_smoothing': np.logspace(0,-9, num=10)}
nb_cv = GridSearchCV(nb, nb_parameters, scoring="accuracy", cv=kf)

knn_parameters = {"n_neighbors": list(range(1, 50)),
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean", "manhattan", "chebyshev"]}
knn_cv = GridSearchCV(knn, knn_parameters, scoring="accuracy", cv=kf)

# Trenowanie modeli
rf_cv.fit(X_train_scaled, y_train)
nb_cv.fit(X_train_scaled, y_train)
knn_cv.fit(X_train_scaled, y_train)

# Wyświetlanie najlepszych parametrów i wyników dla poszczególnych modeli
print("Random Forest:")
print("Best Parameters: {}\nBest Score: {}".format(rf_cv.best_params_, rf_cv.best_score_))
print("Predicted values score: {}".format(rf_cv.score(X_test_scaled, y_test)))
rf_predictions = rf_cv.predict(X_test_scaled)

print("Gaussian Naive Bayes:")
print("Best Parameters: {}\nBest Score: {}".format(nb_cv.best_params_, nb_cv.best_score_))
print("Predicted values score: {}".format(nb_cv.score(X_test_scaled, y_test)))
nb_predictions = nb_cv.predict(X_test_scaled)

print("K-Nearest Neighbors:")
print("Best Parameters: {}\nBest Score: {}".format(knn_cv.best_params_, knn_cv.best_score_))
print("Predicted values score: {}".format(knn_cv.score(X_test_scaled, y_test)))
knn_predictions = knn_cv.predict(X_test_scaled)

# Tworzenie DataFrame'u porównującego wartości przewidziane z rzeczywistymi dla poszczególnych modeli
comparison_df = pd.DataFrame({"Random Forest": rf_predictions,
                              "Gaussian Naive Bayes": nb_predictions,
                              "K-Nearest Neighbors": knn_predictions,
                              "True Values": y_test})
print(comparison_df.iloc[0:10, :])

# Wykres boxplota dla wartości dokładności w GridSearchCV dla poszczególnych modeli
rf_cv_results = rf_cv.cv_results_
rf_accuracy_values = rf_cv_results['mean_test_score']

nb_cv_results = nb_cv.cv_results_
nb_accuracy_values = nb_cv_results['mean_test_score']

knn_cv_results = knn_cv.cv_results_
knn_accuracy_values = knn_cv_results['mean_test_score']

plt.boxplot([rf_accuracy_values, nb_accuracy_values, knn_accuracy_values], patch_artist=True)
plt.title("Accuracy Values - Cross-validation")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks([1, 2, 3], ["Random Forest", "Gaussian Naive Bayes", "K-Nearest Neighbors"])
plt.show()


# Obliczanie dokładności dla każdego modelu
rf_accuracy = rf_cv.score(X_test_scaled, y_test)
nb_accuracy = nb_cv.score(X_test_scaled, y_test)
knn_accuracy = knn_cv.score(X_test_scaled, y_test)

# Obliczanie wag dla średniej ważonej
total_accuracy = rf_accuracy + nb_accuracy + knn_accuracy
rf_weight = rf_accuracy / total_accuracy
nb_weight = nb_accuracy / total_accuracy
knn_weight = knn_accuracy / total_accuracy

# Obliczanie średniej ważonej
hybrid_accuracy = (rf_weight * rf_accuracy) + (nb_weight * nb_accuracy) + (knn_weight * knn_accuracy)

# Wyświetlanie dokładności dla powstałego modelu hybrydowego
print("Hybrid Model Accuracy: ", hybrid_accuracy)

# Tworzenie wykresu
model_labels = ["Random Forest", "Gaussian Naive Bayes", "K-Nearest Neighbors", "Hybrid Model"]
accuracy_values = [rf_accuracy_values, nb_accuracy_values, knn_accuracy_values, [hybrid_accuracy]]

plt.boxplot(accuracy_values, patch_artist=True, labels=model_labels, boxprops=dict())
plt.title("Accuracy Values - Cross-validation")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()