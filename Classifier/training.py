# Import the needed libraries
from matplotlib import style
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
np.random.seed(42)
style.use('fivethirtyeight')

if __name__ == "__main__":
    X = pd.read_csv(
        "../FeatureExtraction/GGAP/GGAP_data.csv", index_col=0, header=1)
    y = pd.read_csv("./target_data.csv", index_col=0, header=1)
    # shuffle_indexes = np.arange(X.shape[0])
    # np.random.shuffle(shuffle_indexes)
    # image_data = X[shuffle_indexes]
    # image_labels = y[shuffle_indexes]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(X_train.shape, X_test.shape,
          y_train.shape, y_test.values.ravel().shape)
    # Set the random state for reproducibility
    fit_rf = RandomForestClassifier())
    # fit_rf.set_params(n_estimators=400,
    #                   bootstrap=True,
    #                   warm_start=False,
    #                   oob_score=False)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    fit_rf.fit(X_train, y_train.values.ravel())
    joblib.dump(fit_rf, "./random_forest.joblib")
    y_pred=fit_rf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test.values.ravel(), y_pred))
