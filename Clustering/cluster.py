import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


def betacv(distancia_centroide, centroides, rotulos, y):
    # distancias medias intra cluster
    distancia_media_intra = [distancia_centroide[y == n, n].mean() for n in rotulos]

    # media das distancias medias intra cluster
    media_distancia_media_intra = np.mean(distancia_media_intra)

    # desvio das distancias medias intra
    desvio_distancia_media_intra = [distancia_centroide[y == n, n].std() for n in rotulos]

    # media dos desvios intra
    desvio_medio_intra = np.mean(desvio_distancia_media_intra)

    # cv intra cluster
    cv_intra = desvio_medio_intra / media_distancia_media_intra

    # distancias inter clusters
    distancia_inter = pairwise_distances(centroides, centroides)

    # distancias medias inter cluster
    distancia_media_inter = [distancia_inter[n].mean() for n in range(len(centroides))]

    # media das distancias medias inter cluster
    media_distancia_media_inter = np.mean(distancia_media_inter)

    # desvio das distancias medias inter cluster
    desvio_distancia_inter = [distancia_inter[n].std() for n in range(len(centroides))]

    # media dos desvios
    media_desvio_inter = np.mean(desvio_distancia_inter)

    # cv inter cluster
    cv_inter = media_desvio_inter / media_distancia_media_inter

    # beta cv
    beta_cv = cv_intra / cv_inter

    return beta_cv


def betacv2(data, labels, metric='euclidean'):
    distances = pairwise_distances(data, metric=metric)
    n = labels.shape[0]
    A = np.array([intra_cluster_distance(distances[i], labels, i)
                  for i in range(n)])
    B = np.array([inter_cluster_distance(distances[i], labels, i)
                  for i in range(n)])
    a = np.sum(A)
    b = np.sum(B)
    labels_unq = np.unique(labels)
    members = np.array([member_count(labels, i) for i in labels_unq])
    N_in = np.array([i*(i-1) for i in members])
    n_in = np.sum(N_in)
    N_out = np.array([i*(n-i) for i in members])
    n_out = np.sum(N_out)
    betacv = (a/n_in)/(b/n_out)
    print('intra:', a)
    print('inter:', b)
    print('n_in :', n_in)
    print('n_out:', n_out)
    return betacv


def intra_cluster_distance(distances_row, labels, i):
    mask = labels == labels[i]
    mask[i] = False
    if not np.any(mask):
        # cluster of size 1
        return 0
    a = np.sum(distances_row[mask])
    return a


def inter_cluster_distance(distances_row, labels, i):
    mask = labels != labels[i]
    b = np.sum(distances_row[mask])
    return b


def member_count(labels, i):
    mask = labels == i
    return len(labels[mask])


data = pd.read_excel('Clustering/DBMS-Performance-Monitor-Log.xls', index_col=0, header=0).drop('ID', axis=1)

data.hist()
plt.show()

data.hist(cumulative=True, density=True, bins=50)
plt.show()

data.boxplot()
plt.show()

data_norm = pd.DataFrame(StandardScaler().fit_transform(X=data.apply(func=np.log10)), columns=data.columns)

data_cluster = data_norm.drop(['Disk_1', 'Disk_2'], axis=1)

lista_betacv = []
k_max = 18
k_min = 3
# n = 3

for n in range(k_min, k_max + 1):
    cluster = KMeans(n_clusters=n, random_state=10)

    y = cluster.fit_predict(data_cluster)

    rotulos = np.unique(y)

    centroides = cluster.cluster_centers_

    distancia_centroide = cluster.fit_transform(data_cluster)

    # beta_cv = betacv(distancia_centroide, centroides, rotulos, y)
    beta_cv = betacv2(data_cluster, y)
    lista_betacv.append(beta_cv)
    print("betacv para", n, ':', beta_cv)

plt.title("K x Beta-cv")
plt.xlabel('K')
plt.ylabel('Beta-cv')
plt.plot([i for i in range(k_min, k_max + 1)], lista_betacv, '-o')
plt.show()

cluster = KMeans(n_clusters=17, random_state=10)
y = cluster.fit_predict(data_cluster)
rotulos = np.unique(y)

for n in rotulos:
    print("Sumarização Cluster", n)
    print(data[y == n].describe())

    data[y == n].hist()
    plt.title('Histograma cluster ' + str(n))
    plt.show()

    data[y == n].hist(cumulative=True, density=True, bins=50)
    plt.title('CDF cluster ' + str(n))
    plt.show()

    data[y == n].boxplot()
    plt.title('BoxPlot cluster ' + str(n))
    plt.show()
