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


def p0_jobs(intensidade_de_trafego_mmm, m):
    return 1 / (1 + (
            ((m * intensidade_de_trafego_mmm) ** m) / (math.factorial(m) * (1 - intensidade_de_trafego_mmm))) + sum(
        [((m * intensidade_de_trafego_mmm) ** n) / (math.factorial(n)) for n in range(1, m)]))


def proba_espera(intensidade_de_trafego_mmm, p0, m):
    return p0 * ((m * intensidade_de_trafego_mmm) ** m) / (math.factorial(m) * (1 - intensidade_de_trafego_mmm))


def num_medio_jobs_fila(intensidade_de_trafego_mmm, proba_espera):
    return (intensidade_de_trafego_mmm * proba_espera) / (1 - intensidade_de_trafego_mmm)


def num_medio_jobs_servico(intensidade_de_trafego_mmm, m):
    return m * intensidade_de_trafego_mmm


def num_jobs_sistema(intensidade_trafego, proba_espera, m):
    return ((intensidade_trafego * proba_espera) / (1 - intensidade_trafego)) + m * intensidade_trafego


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


def var_tempo_medio_resposta(taxa_servico_mmm, intensidade_de_trafego_mmm, proba_espera, m):
    return (1 / (taxa_servico_mmm ** 2) * (
            1 + ((proba_espera * (2 - proba_espera)) / ((m ** 2) * ((1 - intensidade_de_trafego_mmm) ** 2)))))


def tempo_medio_espera(proba_espera, intensidade_de_trafego, taxa_de_servico, m):
    return proba_espera / (m * taxa_de_servico * (1 - intensidade_de_trafego))


def var_tempo_medio_espera(proba_espera, intensidade_de_trafego, taxa_de_servico, m):
    return


def cdf_tempo_resp(taxa_de_servico, temp_resp, intensidade_de_trafego, proba_espera, m):
    if round(intensidade_de_trafego, 3) != round((m - 1) / m, 3):
        return 1 - math.exp(taxa_de_servico * temp_resp) - (
                proba_espera / (1 - m + m * intensidade_de_trafego)) * math.exp(
            -m * taxa_de_servico * (1 - intensidade_de_trafego) * temp_resp) - math.exp(-taxa_de_servico * temp_resp)
    else:
        return 1 - math.exp(taxa_de_servico * temp_resp) - proba_espera * taxa_de_servico * temp_resp * math.exp(
            -taxa_de_servico * temp_resp)


def q_percentil_tempo_espera(taxa_servico, intensidade_de_trafego, proba_espera, m, q):
    calc = (1 / (m * taxa_servico * (1 - intensidade_de_trafego))) * math.log((100 * proba_espera) / (100 - q))

    return max(0, calc)


def q_percentil_tempo_resposta(temp_resp, q):
    return temp_resp * math.log(100 / (100 - q))


taxa_chegada = 0.167 / 2

taxa_servico = 0.05

m = 6

utilizacao = intensidade_de_trafego_mmm(taxa_chegada, taxa_servico, m)

print("Utilização:", round(utilizacao, 2))

p0 = p0_jobs(utilizacao, m)

print("p0:", round(p0, 2))

p_espera = proba_espera(utilizacao, p0, m)

print("Probabilidade de Espera:", round(p_espera, 2))

n_jobs = num_jobs_sistema(utilizacao, p_espera, m)

print("E[n]", round(n_jobs, 2))

n_fila = num_medio_jobs_fila(utilizacao, p_espera)

print("E[nq]", round(n_fila, 2))

temp_resp = tempo_medio_resposta(taxa_servico, utilizacao, p_espera, m)

print("E[r]", round(temp_resp, 2))

temp_esp = tempo_medio_espera(p_espera, utilizacao, taxa_servico, m)

print("E[w]", round(temp_esp, 2))

q_perc = q_percentil_tempo_espera(taxa_servico, utilizacao, p_espera, m, 90)

print("Wq", round(q_perc, 2))

var_temp_resp = var_tempo_medio_resposta(taxa_servico, utilizacao, p_espera, m)

print("Var[r]:", round(var_temp_resp, 3))
