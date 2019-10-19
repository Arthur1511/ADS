import math


def intensidade_de_trafego(taxa_chegada, taxa_servico):
    return taxa_chegada / taxa_servico


def p0_jobs(intensidade_de_trafego):
    return 1 - intensidade_de_trafego


def pN_jobs(intensidade_de_trafego, n):
    return (1 - intensidade_de_trafego) * intensidade_de_trafego ** n


def pN_or_more_jobs(intensidade_de_trafego, n):
    return intensidade_de_trafego ** n


def numero_medio_jobs(intensidade_de_trafego):
    return intensidade_de_trafego / (1 - intensidade_de_trafego)


def var_num_medio_jobs(intensidade_de_trafego):
    return intensidade_de_trafego / ((1 - intensidade_de_trafego) ** 2)


def probabilidade_k_jobs_fila(intensidade_de_trafego, k):
    if k == 0:
        return 1 - (intensidade_de_trafego ** 2)

    else:
        return (1 - intensidade_de_trafego) * (intensidade_de_trafego ** (k + 1))


def num_medio_jobs_fila(intensidade_de_trafego, p0):
    return (intensidade_de_trafego ** 2) / p0


def tempo_medio_espera_fila(temp_resp):
    return intensidade_de_trafego * temp_resp


def var_temp_medio_espera_fila(intensidade_de_trafego, taxa_servico, p0):
    return (2 - intensidade_de_trafego) * (intensidade_de_trafego / ((taxa_servico ** 2) * (p0 ** 2)))


def tempo_medio_resposta(taxa_servico, p0):
    return (1 / taxa_servico) / p0


def cdf_tempo_resposta(tem_resp, taxa_servico, p0):
    return 1 - math.exp(-tem_resp * taxa_servico * p0)


def q_perc_temp_resp(q, taxa_servico, p0):
    return (1 / (taxa_servico * p0)) * math.log(100 / (100 - q))


def cdf_tempo_espera(tempo_espera, taxa_servico, p0, intensidade_de_trafego):
    return 1 - (intensidade_de_trafego * math.exp(-tempo_espera * taxa_servico * p0))


def q_perc_temp_espera(q, taxa_servico, p0, intensidade_de_trafego):
    return (1 / (taxa_servico * p0)) * math.log((100 * intensidade_de_trafego) / (100 - q))


taxa_chegada = 240
taxa_servico = 0.5

intens_traf = intensidade_de_trafego(taxa_chegada, taxa_servico)

print("Utilização/Intensidade de Trafego", round(intens_traf, 2))

p0 = p0_jobs(intens_traf)

temp_resp = tempo_medio_resposta(taxa_servico, p0)

print("Tempo de resposta:", round(temp_resp, 2))

print("Num de Jobs:", round(numero_medio_jobs(intens_traf), 2))

print("Prob k jobs:", round(pN_or_more_jobs(intens_traf, 11), 2))

print("q percentil tempo de resposta:", round(q_perc_temp_resp(90, taxa_servico, p0), 2))

print("q percentil tempo de espera:", round(q_perc_temp_espera(90, taxa_servico, p0, intens_traf), 2))
