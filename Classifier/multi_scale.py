import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler


def prepare_data(factor):
    X = pd.read_csv(
        "../FeatureExtraction/GGAP/GGAP_data.csv", index_col=0)
    y = pd.read_csv("./target_data.csv", index_col=0)
    # reps = [factor if val == 1 else 1 for val in y.target]
    # X = X.loc[np.repeat(X.index.values, reps)]
    # y = y.loc[np.repeat(y.index.values, reps)]
    return X, y


if __name__ == "__main__":
    # =================PCA================
    # X, y = prepare_data(1)
    # print(X.shape, y.shape)

    # # get positive
    # X = np.array(X)[:507, :]
    # print(type(X), X.shape)
    # # apply the MDS procedure to get a 2-dimensional dataset
    # mds = MDS(2, random_state=0)
    # X_2d = mds.fit_transform(X)

    # print((X_2d.shape))
    # np.save("2d_data_positive.npy", X_2d)

    # ==================PLOT==============
    x = np.arange(0, 1, 0.05)
    coords = np.load("2d_data_positive.npy")
    print(coords.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(-0.2, 0.2, 0.01))
    ax.set_yticks(np.arange(-0.2, 0.3, 0.01))
    plt.scatter(coords[:, 0], coords[:, 1], s=0.5)
    plt.grid()
    plt.show()
