import matplotlib.pyplot as plt
from MVA import mva_exato
import numpy as np

N = 40
Z = 15
K = 3
Si = np.array([0.018, 0.05, 0.03])
Vi = np.array([111, 10, 100])

X0, R = mva_exato(N, Z, K, Si, Vi)

D = Si * Vi

D_sum = sum(D)
D_max = D.max()

N_otimo = (D_sum + Z) / D_max

lim_inf_x0 = np.array([(n / ((n * D_sum) + Z)) for n in range(N)])

lim_sup_x0 = np.array([min((n / (D_sum + Z)), (1 / D_max)) for n in range(N)])

lim_inf_R = np.array([max(D_sum, ((n * D_max) - Z)) for n in range(N)])

lim_sup_R = np.array([(n * D_sum) for n in range(N)])

# plt.plot([0,int(N_otimo)], [0, (1 / (D[0] + Z))], label="Lim. Inf. Otimista")
# plt.plot([0,int(N_otimo)], [(1 / D_max), (1 / D_max)], label="Lim. Sup. Otimista")
# plt.plot([i for i in range(int(N_otimo))], [(1 / (D_sum + Z)) for _ in range(int(N_otimo))], label="Lim. Inf. Otimista")
# plt.plot([i for i in range(int(N_otimo), N)], [(1 / D_max) for _ in range(int(N_otimo), N)], label="Lim. Sup. Otimista")

plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. Pessimista")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("Sistema Original")
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. Pessimista")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.title("Sistema Original")
plt.show()

# CPU 2x mais rápida

Si = np.array([0.009, 0.05, 0.03])
Vi = np.array([111, 10, 100])

X0, R = mva_exato(N, Z, K, Si, Vi)

D = Si * Vi

D_sum = sum(D)
D_max = D.max()

N_otimo = (D_sum + Z) / D_max

lim_inf_x0 = np.array([(n / ((n * D_sum) + Z)) for n in range(N)])

lim_sup_x0 = np.array([min((n / (D_sum + Z)), (1 / D_max)) for n in range(N)])

lim_inf_R = np.array([max(D_sum, ((n * D_max) - Z)) for n in range(N)])

lim_sup_R = np.array([(n * D_sum) for n in range(N)])

plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. Pessimista")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("CPU 2x mais rápida")
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. Pessimista")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.title("CPU 2x mais rápida")
plt.show()

# Demanda balanceada

Si = np.array([0.018, 0.05, 0.03])
Vi = np.array([111, 41, 69])

X0, R = mva_exato(N, Z, K, Si, Vi)

D = Si * Vi

D_sum = sum(D)
D_max = D.max()

N_otimo = (D_sum + Z) / D_max

lim_inf_x0 = np.array([(n / ((n * D_sum) + Z)) for n in range(N)])

lim_sup_x0 = np.array([min((n / (D_sum + Z)), (1 / D_max)) for n in range(N)])

lim_inf_R = np.array([max(D_sum, ((n * D_max) - Z)) for n in range(N)])

lim_sup_R = np.array([(n * D_sum) for n in range(N)])

plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. Pessimista")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("Demanda balanceada")
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. Pessimista")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.title("Demanda balanceada")
plt.show()

# Terceiro dsico
K = 4
Si = np.array([0.018, 0.05, 0.03, 0.03])
Vi = np.array([111, 10, 50, 50])

X0, R = mva_exato(N, Z, K, Si, Vi)

D = Si * Vi

D_sum = sum(D)
D_max = D.max()

N_otimo = (D_sum + Z) / D_max

lim_inf_x0 = np.array([(n / ((n * D_sum) + Z)) for n in range(N)])

lim_sup_x0 = np.array([min((n / (D_sum + Z)), (1 / D_max)) for n in range(N)])

lim_inf_R = np.array([max(D_sum, ((n * D_max) - Z)) for n in range(N)])

lim_sup_R = np.array([(n * D_sum) for n in range(N)])

plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. Pessimista")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("Adicionar mais um disco")
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. Pessimista")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.title("Adicionar mais um disco")
plt.show()

# Todas as alterações

K = 4
Si = np.array([0.009, 0.05, 0.03, 0.03])
Vi = np.array([111, 25.4, 42.3, 42.3])

X0, R = mva_exato(N, Z, K, Si, Vi)

D = Si * Vi

D_sum = sum(D)
D_max = D.max()

N_otimo = (D_sum + Z) / D_max

lim_inf_x0 = np.array([(n / ((n * D_sum) + Z)) for n in range(N)])

lim_sup_x0 = np.array([min((n / (D_sum + Z)), (1 / D_max)) for n in range(N)])

lim_inf_R = np.array([max(D_sum, ((n * D_max) - Z)) for n in range(N)])

lim_sup_R = np.array([(n * D_sum) for n in range(N)])

plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. Pessimista")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("Todas as alterações")
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. Pessimista")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. Pessimista")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.title("Todas as alterações")
plt.show()
