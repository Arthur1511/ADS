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


plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. ")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("Sistema Original")
plt.savefig("X0_Sistema_Original", papertype='a4', dpi=300)
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. ")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.title("Sistema Original")
plt.savefig("R_Sistema_Original", dpi=300)
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

plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. ")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("CPU 2x mais rápida")
plt.savefig("X0_Troca_CPU", dpi=300)
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. ")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.title("CPU 2x mais rápida")
plt.savefig("R_Troca_CPU", dpi=300)
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

plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. ")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("Demanda balanceada")
plt.savefig("X0_Demanda_Balanceada", dpi=300)
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. ")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.title("Demanda balanceada")
plt.savefig("R_Demanda_Balanceada", dpi=300)
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

plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. ")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("Adicionar mais um disco")
plt.savefig("X0_Disco_Adicional", dpi=300)
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. ")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.savefig("R_Disco_Adicional", dpi=300)
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

plt.plot(lim_sup_x0, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_x0, color='orange', label="Lim. Inf. ")
plt.plot(X0, color='navy', label="X0")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("X0(N)")
plt.title("Todas as alterações")
plt.savefig("X0_Todas_Alteracoes", dpi=300)
plt.show()

plt.plot(lim_sup_R, color='yellow', label="Lim. Sup. ")
plt.plot(lim_inf_R, color='orange', label="Lim. Inf. ")
plt.plot(R, color='navy', label="R")
plt.legend(loc="lower right")
plt.grid()
plt.xlabel("N")
plt.ylabel("R(N)")
plt.title("Todas as alterações")
plt.savefig("R_Todas_Alteracoes", dpi=300)
plt.show()
