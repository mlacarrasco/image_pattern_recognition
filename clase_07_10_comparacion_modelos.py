from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import make_scorer, zero_one_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from sklearn.preprocessing import label_binarize


# Multiclase (manual, válido para cualquier K)
def square_loss_multiclase(y_true, proba, classes):
    Y = label_binarize(y_true, classes=classes)
    if Y.shape[1] == 1:                     # caso binario
        Y = np.hstack([1 - Y, Y])
    return np.mean(np.sum((proba - Y) ** 2, axis=1))


X, y = load_iris(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clases = np.unique(y)

modelos = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42)
}

zo_scorer = make_scorer(zero_one_loss, greater_is_better=False)

for nombre, modelo in modelos.items():
    # 0-1 loss por fold (sklearn la devuelve negada, por eso el signo menos)
    zo = -cross_val_score(modelo, X, y, cv=cv, scoring=zo_scorer)

    # Square loss con probabilidades fuera de muestra
    proba = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")
    sq = square_loss_multiclase(y, proba, clases)

    print(f"{nombre:13s}  0-1 loss: {zo.mean():.3f} ± {zo.std():.3f}   "
          f"square loss: {sq:.3f}")