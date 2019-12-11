import math


def taxa_servico_n(taxa_servico, n, m):
    if n < m:
        return n * taxa_servico
    else:
        return m * taxa_servico


def intensidade_trafego(taxa_chegada, taxa_servico, m):
    return taxa_chegada / (m * taxa_servico)


def pN_jobs(intensidade_trafego, p0, n, m, b):
    if m == 1:
        if n <= b:
            if intensidade_trafego != 1:
                return ((1 - intensidade_trafego) / (1 - intensidade_trafego ** (b + 1))) * intensidade_trafego ** n
            else:
                return 1 / (b + 1)
        if n > b:
            return 0
    elif n < m:
        return (((m * intensidade_trafego) ** n) / (math.factorial(n))) * p0
    elif m <= n <= b:
        return (((m ** m) * (intensidade_trafego ** n)) / (math.factorial(m))) * p0


def p0_jobs(intensidade_trafego, m, b):
    if m == 1:
        if intensidade_trafego != 1:
            return (1 - intensidade_trafego) / (1 - intensidade_trafego ** (b + 1))
        else:
            return 1 / (b + 1)
    else:
        return 1 / (1 + ((1 - intensidade_trafego ** (b - m + 1)) * (m * intensidade_trafego) ** m) / (
                math.factorial(m) * (1 - intensidade_trafego)) + sum(
            [((m * intensidade_trafego) ** n) / math.factorial(n) for n in range(1, m)]))


def num_jobs_sistema(intensidade_trafego, p0, m, b):
    if m == 1:
        return intensidade_trafego / (1 - intensidade_trafego) - ((b - 1) * intensidade_trafego ** (b + 1)) / (
                1 - intensidade_trafego ** (b + 1))
    else:
        return sum([(n * pN_jobs(intensidade_trafego, p0, n, m, b)) for n in range(1, b+1)])


def num_jobs_fila(intensidade_trafego, p0, m, b):
    if m == 1:
        return intensidade_trafego / (1 - intensidade_trafego) - intensidade_trafego * (
                (1 + b * intensidade_trafego ** b) / (1 - intensidade_trafego ** (b + 1)))

    else:
        return sum([((n - m) * pN_jobs(intensidade_trafego, p0, n, m, b)) for n in range(m + 1, b+1)])


def taxa_chegada_efetiva(taxa_chegada, intensidade_trafego, p0, b, m):
    return taxa_chegada * (1 - pN_jobs(intensidade_trafego, p0, b, m, b))


def utilizacao(taxa_chegada_efetiva, taxa_servico, m):
    return taxa_chegada_efetiva / (m * taxa_servico)


def tempo_medio_resposta(taxa_chegada_efetiva, num_jobs_sistema):
    return num_jobs_sistema / taxa_chegada_efetiva


def tempo_medio_espera(temp_resposta, taxa_servico):
    return temp_resposta - (1 / taxa_servico)


def perda(taxa_chegada, taxa_chegada_esfetiva):
    return taxa_chegada - taxa_chegada_esfetiva


taxa_chegada = 5

taxa_servico = 1/2.5

m = 10

b = 10

intensidade_traf = intensidade_trafego(taxa_chegada, taxa_servico, m)

print("Intensidade de Tráfego:", round(intensidade_traf, 2))

p0 = p0_jobs(intensidade_traf, m, b)

print("P0:", round(p0, 2))

# for n in range(1, 5):
#     pn = pN_jobs(intensidade_traf, p0, n, m, b)
#
#     print("P%i:" % (n), round(pn, 2))


num_jobs = num_jobs_sistema(intensidade_traf, p0, m, b)

print("E[n]:", round(num_jobs, 2))

num_jobs_fila = num_jobs_fila(intensidade_traf, p0, m, b)

print("E[nq]:", round(num_jobs_fila, 2))

taxa_efetiva = taxa_chegada_efetiva(taxa_chegada, intensidade_traf, p0, b, m)

print("Taxa de Chegada Efetiva:", round(taxa_efetiva, 2))

perda = perda(taxa_chegada, taxa_efetiva)

print("Perda:", round(perda, 2))

utili = utilizacao(taxa_efetiva, taxa_servico, m)

print("Utilização:", round(utili, 2))
temp_resp = tempo_medio_resposta(taxa_efetiva, num_jobs)

print("Tempo de resposta:", round(temp_resp, 2))

# print(pN_jobs(intensidade_traf, p0, , m, b))