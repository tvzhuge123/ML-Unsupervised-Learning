from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, homogeneity_score, completeness_score, silhouette_score

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse as sps
from scipy.linalg import pinv
from sklearn.metrics.pairwise import pairwise_distances

def standardize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return(scaler.transform(data))


def evaluate(X, y_true, y_pred):
    return(homogeneity_score(y_true, y_pred),
           completeness_score(y_true, y_pred),
           silhouette_score(X, y_pred, metric='euclidean'))


def ClusterPlot(df_homogeneity, df_completeness, df_silhouette, xlabel):
    fig = plt.figure(figsize=(10,10))

    ax_homogeneity = fig.add_subplot(3, 1, 1)
    ax_homogeneity.set_ylabel("Homogeneity Score")

    ax_completeness = fig.add_subplot(3, 1, 2)
    ax_completeness.set_ylabel("Completeness Score")

    ax_silhouette = fig.add_subplot(3, 1, 3)
    ax_silhouette.set_ylabel("Silhouette Score")
    ax_silhouette.set_xlabel(xlabel)

    df_homogeneity.plot(grid=True, marker='o', xticks=df_homogeneity.index,
                         title='Homogeneity Score vs ' + xlabel, ax=ax_homogeneity)
    df_completeness.plot(grid=True, marker='o', xticks=df_completeness.index,
                         title='Completeness Score vs ' + xlabel, ax=ax_completeness)
    df_silhouette.plot(grid=True, marker='o', xticks=df_silhouette.index,
                       title='Silhouette Score vs ' + xlabel, ax=ax_silhouette)

    plt.show()
    return


def kmeans(data, n_clusters, random_state=0):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    return(kmeans.fit_predict(data))


def em(data, n_components):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', tol=0.001, reg_covar=1e-06,
                          max_iter=10000, n_init=1, init_params='kmeans')
    return(gmm.fit_predict(data))


def pca(data, thr_var=0.9, plot=True):
    pca = PCA().fit(data)
    ratios = pca.explained_variance_ratio_

    if plot:
        plt.plot(range(1, len(ratios)+1), pca.singular_values_, marker='o')
        plt.xticks(range(1, len(ratios)+1))
        plt.xlabel('Component No.')
        plt.ylabel('Eigenvalue')
        plt.title('PCA Eigenvalue')
        plt.grid(True)
        plt.show()

        plt.plot(range(1, len(ratios)+1), ratios, marker='o')
        plt.xticks(range(1, len(ratios)+1))
        plt.xlabel('Component No.')
        plt.ylabel('Variance Explained')
        plt.title('PCA Variance Explained')
        plt.grid(True)
        plt.show()

    for n in range(len(ratios)):
        if(sum(ratios[:n+1]) >= thr_var):
            break
    pca = PCA(n_components=n+1).fit(data)

    return(pca.transform(data), pca.explained_variance_ratio_)


def ica(data, plot=True):
    if plot:
        dims = range(1, data.shape[1]+1)
        kurt = []
        for dim in dims:
            ica = FastICA(n_components=dim, max_iter=10000, random_state=1)
            tmp = ica.fit_transform(data)
            tmp = pd.DataFrame(tmp)
            tmp = tmp.kurt(axis=0)
            kurt.append(tmp.abs().mean())

        plt.figure()
        plt.title("ICA Kurtosis")
        plt.xticks(dims)
        plt.xlabel("Independent Components")
        plt.ylabel("Avg Kurtosis Across IC")
        plt.plot(dims, kurt, marker='o')
        plt.grid(False)
        plt.show()

    ica = FastICA(n_components=data.shape[1], max_iter=10000, random_state=1).fit(data)

    return(ica.transform(data))


def rp(data, n_components, random_state=0):
    rp = SparseRandomProjection(random_state=random_state, n_components=n_components)
    df_transform = rp.fit_transform(data)
    tmp1 = pairwiseDistCorr(df_transform, data)
    tmp2 = reconstructionError(rp, data)

    return(df_transform, tmp1, tmp2)


def pairwiseDistCorr(X1, X2):
    assert X1.shape[0] == X2.shape[0]

    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)


def RPPlot(df_corr, df_error):
    fig = plt.figure(figsize=(10,10))

    ax_corr = fig.add_subplot(2, 1, 1)
    ax_corr.set_ylabel("Reconstruction Correlation")

    ax_error = fig.add_subplot(2, 1, 2)
    ax_error.set_ylabel("Reconstruction Error")
    ax_error.set_xlabel('Dimension')

    df_corr.plot(grid=True, marker='o', xticks=df_corr.index,
                         title='Reconstruction Correlation vs Dimension', ax=ax_corr)
    df_error.plot(grid=True, marker='o', xticks=df_error.index,
                       title='Reconstruction Error vs Dimension', ax=ax_error)

    plt.show()
    return


def rf(data_x, data_y, n_estimators=100, random_state=0):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=2, random_state=random_state)
    clf.fit(data_x, data_y)
    pred = clf.predict(data_x)
    return(data_x[:,:10], clf.feature_importances_, accuracy_score(data_y, pred))


def RFPlot(df_imp, df_acc):
    fig = plt.figure(figsize=(10,10))

    ax_imp = fig.add_subplot(2, 1, 1)
    ax_imp.set_ylabel("Feature Importance")

    ax_acc = fig.add_subplot(2, 1, 2)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel('n_estimators')

    df_imp.plot(grid=True, marker='o', xticks=df_imp.index,
                         title='Feature Importance vs n_estimators', ax=ax_imp)
    df_acc.plot(grid=True, marker='o', xticks=df_acc.index,
                       title='Accuracy vs n_estimators', ax=ax_acc)

    plt.show()
    return


def neuralNetwork(data):
    # train with various training size
    train_size = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    train_scores, test_scores = [], []
    for ts in train_size:
        tmp1, tmp2 = [], []
        for state in [0,1,2,3]:
            # split train and test set
            X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=1-ts, random_state=state)

            # train and evaluate model
            clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,
                                hidden_layer_sizes=(100, 5), random_state=1, max_iter=10000)
            res_train, res_test = evaluation(clf, X_train, X_test, y_train, y_test)
            tmp1.append(res_train)
            tmp2.append(res_test)

        train_scores.append(sum(tmp1)/len(tmp1))
        test_scores.append(sum(tmp2)/len(tmp2))

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Training size %")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Training size")
    ax.plot(train_size, train_scores, marker='o', label="train")
    ax.plot(train_size, test_scores, marker='o', label="test")
    ax.legend()
    plt.show()

    print(max(test_scores))


def evaluation(clf, X_train, X_test, y_train, y_test):
    # evaluate model by cross validation
    res_train = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    res_test = cross_val_score(clf, X_test, y_test, cv=2, scoring='accuracy')

    return sum(res_train)/len(res_train), sum(res_test)/len(res_test)