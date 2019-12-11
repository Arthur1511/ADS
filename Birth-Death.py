import numpy as np

lambd = [5 for i in range(10)]
mi = [i * 1 / 2.5 for i in range(1, 11)]

K = 10

Po = 0.0
Pk = []

# calcula Po
multAux = 1.0
for k in range(K):
    k = k + 1
    for i in range(k):
        multAux = multAux * float(lambd[i]) / (mi[i])

    Po = Po + multAux
    multAux = 1
Po = 1 / (1 + Po)
print("P0 = ", round(Po, 2))

for k in range(K):
    k = k + 1
    multAux = 1
    for i in range(k):
        multAux = multAux * float(lambd[i]) / (mi[i])
    Pk.append(Po * multAux)

print("Pk's: ", np.round(Pk, 2))

# throuput
t = [Pk[i] * mi[i] for i in range(K)]
print("throughput: ", round(sum(t), 2))

# tamanho da fila
tm = [(i + 1) * Pk[i] for i in range(K)]
print("tamanho da fila : ", sum(tm))

# tempo de resposta
print("tempo de resposta: ", sum(tm) / sum(t))
