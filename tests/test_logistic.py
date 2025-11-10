import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ai_starter.logistic import LogisticRegressionScratch

def test_logistic_sanity():
    X, y = make_classification(n_samples=600, n_features=12, n_informative=8, n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegressionScratch(lr=0.1, epochs=800, l2=1e-3, random_state=0)
    model.fit(X_train_s, y_train)
    acc = model.score(X_test_s, y_test)
    assert acc >= 0.80
