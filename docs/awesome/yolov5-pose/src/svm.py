import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Data
data_fall = pd.read_csv("data/fall.csv")
data_nofall = pd.read_csv("data/nofall.csv")

# Data Preprocessing
data_fall["label"] = 1
data_nofall["label"] = 0
data = pd.concat([data_fall, data_nofall], ignore_index=True)
data = data.dropna()
data = data.sample(frac=1).reset_index(drop=True)
data = data.astype("float64")

# Split data
X = data.drop("label", axis=1)
y = data["label"]
X_train = X[: int(len(X) * 0.8)]
X_test = X[int(len(X) * 0.8) :]
y_train = y[: int(len(y) * 0.8)]
y_test = y[int(len(y) * 0.8) :]
print(X, y)

# SVM
svm = SVC(kernel="rbf", C=1000, gamma=0.001)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
plt.show()

# Grid Search
param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf", "poly", "sigmoid", "linear"],
}
grid = GridSearchCV(SVC(), param_grid, refit=True)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))

# Plot
sns.heatmap(confusion_matrix(y_test, grid_predictions), annot=True, cmap="Blues")
plt.show()

# Save model
joblib.dump(grid, "model/svm.pkl")
