import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math

data = pd.read_excel('PCA/DBMS-Performance-Monitor-Log.xls', index_col=0, header=0).drop('ID', axis=1)

data.hist()
plt.show()

data.hist(cumulative=True, density=True, bins=50)
plt.show()

data.boxplot()
plt.show()

data_norm = pd.DataFrame(StandardScaler().fit_transform(X=data.apply(func=np.log10)), columns=data.columns)
# data_norm2 = data_norm.copy()
# a = data_norm['CPU'].values.copy()
# data_norm['CPU'] = data_norm['Disk_1'].values.copy()
# data_norm['Disk_1'] = a

C = data_norm.corr()  # / (data_norm.shape[0]-1)
autovalores, autovetores = np.linalg.eig(C)
total = sum(autovalores)
var_exp = [(i / total) * 100 for i in autovalores]
var_exp_acum = np.cumsum(var_exp)
print(var_exp)
pca_data = data_norm.dot(autovetores)
soma_quadrados = (pca_data ** 2).sum()
soma_quadrados_total = soma_quadrados.sum()
var_explicada = soma_quadrados / soma_quadrados_total
print("Explicação da variação dos componentes:\n" + str(var_explicada))


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
