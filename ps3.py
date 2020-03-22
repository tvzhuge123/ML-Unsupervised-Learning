import util
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

range_random_states = range(1,21)
range_kmeans_k = range(2, 11)
range_em_n = range(2, 11)
range_rf_n = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


alogs_clustering = ['Kmeans', 'EM']
algos_dr = ['PCA', 'ICA', 'Randomized Projections', 'Random Forest']

def part_1(df, scale=True):
    # standardization
    if scale:
        df_scaled = util.standardize(df[0])
    else:
        df_scaled = df[0]

    # Kmeans
    df_score = pd.DataFrame(columns=['homogeneity', 'completeness', 'silhouette'])
    for k in range_kmeans_k:
        df_tmp = pd.DataFrame(columns=['homogeneity', 'completeness', 'silhouette'])
        for state in range_random_states:
            df_pred = util.kmeans(df_scaled, n_clusters=k, random_state=state)
            df_tmp.loc[len(df_tmp)] = util.evaluate(df_scaled, df[1], df_pred)

        df_score.loc[len(df_score)] = df_tmp.mean(axis=0)
        print(k)

    df_score.set_index(pd.Index(range_kmeans_k), inplace=True)
    util.ClusterPlot(df_score.iloc[:,0], df_score.iloc[:,1], df_score.iloc[:,2], 'n_clusters')

    # EM
    df_score = pd.DataFrame(columns=['homogeneity', 'completeness', 'silhouette'])
    for n in range_em_n:
        df_tmp = pd.DataFrame(columns=['homogeneity', 'completeness', 'silhouette'])
        for state in range_random_states:
            df_pred = util.em(df_scaled, n_components=n)
            df_tmp.loc[len(df_tmp)] = util.evaluate(df_scaled, df[1], df_pred)

        df_score.loc[len(df_score)] = df_tmp.mean(axis=0)
        print(n)

    df_score.set_index(pd.Index(range_kmeans_k), inplace=True)
    util.ClusterPlot(df_score.iloc[:,0], df_score.iloc[:,1], df_score.iloc[:,2], 'n_components')

    return


def part_2(df):
    # standardization
    df_scaled = util.standardize(df[0])

    # PCA
    data_transform, explained_ratio = util.pca(df_scaled, 0.9)
    print(len(explained_ratio), sum(explained_ratio))

    # ICA
    data_transform = util.ica(df_scaled)

    # Randomized Projections
    df_score = pd.DataFrame(columns=['recon_corr', 'recon_error'])
    for k in range(1, df_scaled.shape[1]+1):
        df_tmp = pd.DataFrame(columns=['recon_corr', 'recon_error'])
        for state in range_random_states:
            tmp = util.rp(df_scaled, n_components=k, random_state=state)
            df_tmp.loc[len(df_tmp)] = (tmp[1], tmp[2])

        df_score.loc[len(df_score)] = df_tmp.mean(axis=0)
        print(k)

    df_score.set_index(pd.Index(range(1, df_scaled.shape[1]+1)), inplace=True)
    util.RPPlot(df_score.iloc[:, 0], df_score.iloc[:, 1])

    # Random Forest
    df_importance = pd.DataFrame(columns=list(range(1, df_scaled.shape[1]+1)))
    df_accuracy = pd.DataFrame(columns=['accuracy'])
    for n in range_rf_n:
        df_importance_tmp = pd.DataFrame(columns=list(range(1, df_scaled.shape[1]+1)))
        df_accuracy_tmp = pd.DataFrame(columns=['accuracy'])
        for state in range_random_states:
            tmp = util.rf(df_scaled, df[1], random_state=state, n_estimators=n)
            df_importance_tmp.loc[len(df_importance_tmp)] = tmp[1]
            df_accuracy_tmp.loc[len(df_importance_tmp)] = tmp[2]

        df_importance.loc[len(df_importance)] = df_importance_tmp.mean(axis=0)
        df_accuracy.loc[len(df_accuracy)] = df_accuracy_tmp.mean(axis=0)
        print(n)

    df_importance.set_index(pd.Index(range_rf_n), inplace=True)
    df_accuracy.set_index(pd.Index(range_rf_n), inplace=True)
    util.RFPlot(df_importance, df_accuracy)

    return

def part_3(df, n_labels):
    # standardization
    df_scaled = util.standardize(df[0])

    # PCA
    data_transform = util.pca(df_scaled, 0.9)[0]

    df_pred = util.kmeans(df_scaled, n_clusters=n_labels, random_state=1)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.kmeans(data_transform, n_clusters=n_labels, random_state=1)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.em(df_scaled, n_components=n_labels)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.em(data_transform, n_components=n_labels)
    print(util.evaluate(df_scaled, df[1], df_pred))


    # ICA
    data_transform = util.ica(df_scaled)

    df_pred = util.kmeans(df_scaled, n_clusters=n_labels, random_state=1)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.kmeans(data_transform, n_clusters=n_labels, random_state=1)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.em(df_scaled, n_components=n_labels)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.em(data_transform, n_components=n_labels)
    print(util.evaluate(df_scaled, df[1], df_pred))

    # RP
    data_transform = util.rp(df_scaled, n_components=10, random_state=0)[0]

    df_pred = util.kmeans(df_scaled, n_clusters=n_labels, random_state=1)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.kmeans(data_transform, n_clusters=n_labels, random_state=1)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.em(df_scaled, n_components=n_labels)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.em(data_transform, n_components=n_labels)
    print(util.evaluate(df_scaled, df[1], df_pred))

    # RF
    data_transform = util.rf(df_scaled, df[1], n_estimators=1000)[0]

    df_pred = util.kmeans(df_scaled, n_clusters=n_labels, random_state=1)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.kmeans(data_transform, n_clusters=n_labels, random_state=1)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.em(df_scaled, n_components=n_labels)
    print(util.evaluate(df_scaled, df[1], df_pred))

    df_pred = util.em(data_transform, n_components=n_labels)
    print(util.evaluate(df_scaled, df[1], df_pred))


def part_4(df):
    # standardization
    df_scaled = util.standardize(df[0])

    # PCA
    data_transform = util.pca(df_scaled, 0.9)[0]
    util.neuralNetwork((data_transform, df[1]))

    # ICA
    data_transform = util.ica(df_scaled)
    util.neuralNetwork((data_transform, df[1]))

    # RP
    data_transform = util.rp(df_scaled, n_components=10, random_state=0)[0]
    util.neuralNetwork((data_transform, df[1]))

    # RF
    data_transform = util.rf(df_scaled, df[1], n_estimators=1000)[0]
    util.neuralNetwork((df_scaled, df[1]))


def part_5(df, n_labels):
    # standardization
    df_scaled = util.standardize(df[0])

    # Kmeans
    df_pred = util.kmeans(df_scaled, n_clusters=n_labels, random_state=4)
    util.neuralNetwork((np.column_stack((df_scaled, df_pred.T)), df[1]))

    # EM
    df_pred = util.em(df_scaled, n_components=n_labels)
    util.neuralNetwork((np.column_stack((df_scaled, df_pred.T)), df[1]))


def main():
    data_wine = load_wine(return_X_y=True)
    data_bc = load_breast_cancer(return_X_y=True)

    part_1(data_wine)
    part_1(data_bc)

    part_2(data_wine)
    part_2(data_bc)

    part_3(data_wine, 3)
    part_3(data_bc, 2)

    part_4(data_wine)
    part_4(data_bc)

    part_5(data_wine, 3)
    part_5(data_bc, 2)

    return


if __name__ == '__main__':
    main()