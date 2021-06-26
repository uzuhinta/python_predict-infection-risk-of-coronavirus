# Import the needed libraries
from matplotlib import style
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, matthews_corrcoef
from sklearn.model_selection import KFold
np.random.seed(42)
style.use('fivethirtyeight')


def prepare_data(factor):
    X = pd.read_csv(
        "../FeatureExtraction/GGAP/GGAP_data.csv", index_col=0)
    y = pd.read_csv("./target_data.csv", index_col=0)
    reps = [factor if val == 1 else 1 for val in y.target]
    X = X.loc[np.repeat(X.index.values, reps)]
    y = y.loc[np.repeat(y.index.values, reps)]
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = prepare_data(2)
    kf = KFold(n_splits=10, random_state=123, shuffle=True)
    print(X.shape, y.shape)
    # print(X.head(10))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # print(X_train.shape, X_test.shape,
    #       y_train.shape, y_test.values.ravel().shape)

    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(type(X_train))
        # print(X_train.shape, y_train.shape)
        # print(X_test.shape, y_test.shape)
        # Set the random state for reproducibility
        fit_rf = RandomForestClassifier(n_estimators=500)
        fit_rf.fit(X_train, y_train.ravel())
        # joblib.dump(fit_rf, "./random_forest.joblib")
        y_pred = fit_rf.predict(X_test)

        print("ACC:", accuracy_score(y_test, y_pred))
        print("recall_score:", recall_score(y_test, y_pred))
        print("MCC:", matthews_corrcoef(y_test, y_pred))
