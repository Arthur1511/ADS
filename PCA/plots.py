import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import math

data = pd.read_excel('PCA/DBMS-Performance-Monitor-Log.xls', index_col=0, header=0).drop('ID', axis=1)

data.hist()
plt.show()

data.hist(cumulative=True, density=True, bins=50)
plt.show()

data.boxplot()
plt.show()

data_norm = pd.DataFrame(StandardScaler().fit_transform(X=data.apply(func=np.log10)), columns=data.columns)
data_norm2 = data_norm.copy()
a = data_norm['CPU'].values.copy()
data_norm['CPU'] = data_norm['Disk_1'].values.copy()
data_norm['Disk_1'] = a
# mean_vec = np.mean(data_norm)
# M = data_norm - mean_vec
C = data_norm.corr()  # / (data_norm.shape[0]-1)
autovalores, autovetores = np.linalg.eig(C)
total = sum(autovalores)
var_exp = [(i / total) * 100 for i in autovalores]
var_exp_acum = np.cumsum(var_exp)
print(var_exp)

pca = PCA(n_components=3)
pca.fit(data_norm)

pca_data = data_norm.dot(pca.components_.T)
soma_quadrados = (pca_data ** 2).sum()
soma_quadrados_total = soma_quadrados.sum()
var_explicada = soma_quadrados / soma_quadrados_total
print("Explicação da variação dos componentes:\n" + str(var_explicada))

print("Explicação dos componentes:", pca.explained_variance_ratio_)
# print("Numero de componentes a ser usado:", pca.n_components_)
# data_pca2 = pd.DataFrame(pca.transform(data_norm))

# mais_importante = [np.abs(pca.components_[i]).argmax() for i in range(pca.n_components_)]
# nomes_mais_importantes = [data.columns[mais_importante[i]] for i in range(pca.n_components_)]
# plt.plot(data_pca[nomes_mais_importantes[0]], data_pca[nomes_mais_importantes[1]], 'o')

plt.title('PC1 x PC2')
plt.plot(pca_data[0], pca_data[1], 'o')
plt.show()

plt.boxplot(data['CPU'])
plt.show()


def betacv(data_cluster, centroides, rotulos):
    # media da distancia de todos pra todos
    # media do desvio de todos pra todos
    # cv = desvio/ media

    distancia_media_intra = [pairwise_distances(data_cluster[y == n], data_cluster[y == n]).mean() for n in rotulos]

    media_distancia_media_intra = np.mean(distancia_media_intra)

    desvio_distancia_media_intra = np.std(distancia_media_intra)

    cv_intra = desvio_distancia_media_intra / media_distancia_media_intra

    distancia_inter = pairwise_distances(centroides, centroides)

    media_distancia_inter = np.mean(distancia_inter)
    desvio_distancia_inter = np.std(distancia_inter)

    cv_inter = desvio_distancia_inter / media_distancia_inter

    beta_cv = cv_intra / cv_inter

    return beta_cv


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

    beta_cv = betacv(data_cluster, centroides, rotulos)
    lista_betacv.append(beta_cv)
    print("betacv para", n, ':', beta_cv)

plt.title("K x Beta-cv")
plt.xlabel('K')
plt.ylabel('Beta-cv')
plt.plot([i for i in range(k_min, k_max + 1)], lista_betacv, '-o')
plt.show()

cluster = KMeans(n_clusters=13, random_state=10)
y = cluster.fit_predict(data_cluster)

plt.scatter(data_norm['CPU'], data_norm['Disk_1'], c=y)
plt.show()
