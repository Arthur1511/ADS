import math


def intensidade_de_trafego(taxa_chegada, taxa_servico):
    return taxa_chegada / taxa_servico


def p0_jobs(intensidade_de_trafego):
    return 1 - intensidade_de_trafego


def pN_jobs(intensidade_de_trafego, n):
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
