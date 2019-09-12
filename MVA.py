import numpy as np


def mva_exato(N, Z, K, Si, Vi):
    Ni = np.zeros(K)
    Ri = np.zeros(K)
    Ui = np.zeros(K)

    R = 0
    X0 = 0

    for n in range(N + 1):
        Ri = (Si * (1 + Ni))

        R = sum(Ri * Vi)

        X0 = n / (R + Z)

        Ni = X0 * Vi * Ri

        Ui = X0 * Vi * Si

        print("iter=", n, "Ri=", np.round(Ri, 3), "R=", round(R, 3), "X0=", round(X0, 3), "Ni=", np.round(Ni, 3), "Ui=",
              np.round(Ui, 3))

    return X0, Ni, Ri, R, Ui


def mva_aproximado(N, Z, K, Si, Vi, e):
    Ni = np.array([float(N / K) for i in range(K)])
    Ri = np.zeros(K)
    Ui = np.zeros(K)
    Ni_antigo = np.zeros(K)
    R = 0
    X0 = 0
    n = 0

    while max(abs(Ni - Ni_antigo)) > e:
        Ri = Si * (1 + (((N - 1) / N) * Ni))

        R = sum(Ri * Vi)

        X0 = N / (R + Z)

        Ni_antigo = Ni

        Ni = X0 * Vi * Ri

        Ui = X0 * Vi * Si

        print("iter=", n, "Ri=", np.round(Ri, 3), "R=", round(R, 3), "X0=", round(X0, 3), "Ni=", np.round(Ni, 3), "Ui=",
              np.round(Ui, 3))

        n = n + 1

    return X0, Ni, Ri, R, Ui


N = 30
Z = 5
K = 3
Si = np.array([0.04, 0.03, 0.025])
Vi = np.array([25, 20, 4])

mva_aprox = mva_aproximado(N, Z, K, Si, Vi, 0.01)

mva_exa = mva_exato(N, Z, K, Si, Vi)
