import numpy as np


def mva_exato(N, Z, K, Si, Vi):
    Ni = np.zeros(K)
    Ri = np.zeros(K)
    Ui = np.zeros(K)

    R = 0
    X0 = 0

    x0_list = np.empty(N+1)
    r_list = np.empty(N+1)

    for n in range(N + 1):
        Ri = (Si * (1 + Ni))

        R = sum(Ri * Vi)

        X0 = n / (R + Z)

        Ni = X0 * Vi * Ri

        Ui = X0 * Vi * Si

        x0_list.itemset(n, round(X0, 3))
        r_list.itemset(n, round(R, 3))

        print("iter=", n, "Ri=", np.round(Ri, 3), "R=", round(R, 3), "X0=", round(X0, 3), "Ni=", np.round(Ni, 3), "Ui=",
              np.round(Ui, 3))

    return x0_list, r_list


def mva_aproximado(N, Z, K, Si, Vi, e):
    Ni = np.array([float(N / K) for _ in range(K)])
    Ri = np.zeros(K)
    Ui = np.zeros(K)
    Ni_antigo = np.zeros(K)
    R = 0
    X0 = 0
    n = 0

    x0_list = np.empty(N + 1)
    r_list = np.empty(N + 1)
    np.ap

    while max(abs(Ni - Ni_antigo)) > e:
        Ri = Si * (1 + (((N - 1) / N) * Ni))

        R = sum(Ri * Vi)

        X0 = N / (R + Z)

        Ni_antigo = Ni

        Ni = X0 * Vi * Ri

        Ui = X0 * Vi * Si

        x0_list.itemset(n, round(X0, 3))
        r_list.itemset(n, round(R, 3))

        print("iter=", n, "Ri=", np.round(Ri, 3), "R=", round(R, 3), "X0=", round(X0, 3), "Ni=", np.round(Ni, 3), "Ui=",
              np.round(Ui, 3))

        n = n + 1

    return x0_list, r_list


N = 50
Z = 10
K = 3
Si = np.array([1.5, 4, 4/3])
Vi = np.array([7, 1.5, 4.5])

# K = 2
# Si = np.array([1.5, 4])
# Vi = np.array([4, 3])

# mva_aprox = mva_aproximado(N, Z, K, Si, Vi, 0.01)

# x0, r = mva_exato(N, Z, K, Si, Vi)
#
# print(x0)