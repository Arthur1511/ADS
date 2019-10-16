import math


def taxa_servico_mmm(taxa_servico, n, m):
    if n < m:
        return n * taxa_servico

    else:
        return m * taxa_servico


def intensidade_de_trafego_mmm(taxa_chegada, taxa_servico, m):
    return taxa_chegada / (taxa_servico * m)


def pN_jobs_mmm(intensidade_de_trafego_mmm, p0, n, m):
    if n < m:
        return (((m * intensidade_de_trafego_mmm) ** n) / (math.factorial(n))) * p0

    else:
        return (((intensidade_de_trafego_mmm ** n) * (m ** m)) / (math.factorial(m))) * p0


def p0_jobs(intensidade_de_trafego_mmm, n, m):
    return 1 / (1 + (
            ((m * intensidade_de_trafego_mmm) ** m) / (math.factorial(m) * (1 - intensidade_de_trafego_mmm))) + sum(
        [((m * intensidade_de_trafego_mmm) ** n) / (math.factorial(n)) for i in range(m)]))


def proba_espera(intensidade_de_trafego_mmm, p0, m):
    return (p0 * ((m * intensidade_de_trafego_mmm) ** m) / (math.factorial(m) * (1 - intensidade_de_trafego_mmm)))


def num_medio_jobs_fila(intensidade_de_trafego_mmm, proba_espera):
    return ((intensidade_de_trafego_mmm * proba_espera) / (1 - intensidade_de_trafego_mmm))


def num_medio_jobs_servico(intensidade_de_trafego_mmm, m):
    return m * intensidade_de_trafego_mmm


def num_jobs_sistema(num_jobs_fila, num_jobs_servico):
    return num_jobs_fila + num_jobs_servico


def var_num_medio_jobs_sistema(intensidade_de_trafego_mmm, proba_espera, m):
    return (m * intensidade_de_trafego_mmm + intensidade_de_trafego_mmm * proba_espera * (
            (1 - intensidade_de_trafego_mmm - (intensidade_de_trafego_mmm * proba_espera)) / (
            (1 - intensidade_de_trafego_mmm) ** 2)))


def var_num_medio_jobs_fila(intensidade_de_trafego_mmm, proba_espera):
    return ((proba_espera * intensidade_de_trafego_mmm) * (
            1 + intensidade_de_trafego_mmm + intensidade_de_trafego_mmm * proba_espera)) / (
                   (1 - intensidade_de_trafego_mmm) ** 2)


def tempo_medio_resposta(taxa_servico_mmm, intensidade_de_trafego_mmm, proba_espera, m):
    return (1 / taxa_servico_mmm) * (1 + proba_espera / (m * (1 - intensidade_de_trafego_mmm)))


def tempo_medio_espera(proba_espera, intensidade_de_trafego, taxa_de_servico, m):
    return proba_espera / (m * taxa_de_servico * (1 - intensidade_de_trafego))


def cdf_tempo_resp(taxa_de_servico, temp_resp, intensidade_de_trafego, proba_espera, m):

    if round(intensidade_de_trafego, 3) != round((m - 1) / m, 3):
        return 1 - math.exp(taxa_de_servico * temp_resp) - (
                proba_espera / (1 - m + m * intensidade_de_trafego)) * math.exp(
            -m * taxa_de_servico * (1 - intensidade_de_trafego) * temp_resp) - math.exp(-taxa_de_servico * temp_resp)
    else:
        return 1 - math.exp(taxa_de_servico * temp_resp) - proba_espera * taxa_de_servico * temp_resp * math.exp(
            -taxa_de_servico * temp_resp)


def q_percentil_tempo_espera(taxa_servico, intensidade_de_trafego, proba_espera, m, q):

    calc = (1/(m*taxa_servico*(1-intensidade_de_trafego)))*math.log((100*proba_espera)/(100-q))

    return max(0, calc)