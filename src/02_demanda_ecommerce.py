"""
02_demanda_ecommerce.py — demanda de e-commerce por bairro (h_i).

Converte a populacao residente de cada bairro, produzida pela Etapa 2, na demanda
de encomendas passivel de retirada em pickup point:

    h_i = Pop_i x tau_i x f x alpha

    tau_i  penetracao de e-commerce do bairro i, ajustada pela renda
    f      pedidos por comprador por dia
    alpha  fracao de consumidores disposta a usar um ponto de retirada

Sao gerados 3 valores de beta x 3 cenarios de alpha = 9 combinacoes de h_i.

NOTA ANALITICA, para o Capitulo 4: f e alpha sao escalares uniformes — incidem
igualmente sobre todos os bairros. Varia-los muda o valor das funcoes objetivo,
mas NAO muda o conjunto otimo de pontos nem os indicadores percentuais. O mesmo
vale para a normalizacao de tau. O unico parametro desta etapa que altera a
solucao e o beta, porque so ele redistribui a demanda entre bairros.

A mesma conta e feita em duas unidades de demanda: o BAIRRO (87 pontos) e o
SETOR CENSITARIO habitado (964 pontos). No setor, tau_i usa a renda do proprio
setor. Com beta = 0 as duas unidades dao exatamente a mesma demanda por bairro,
o que e verificado.

Entradas:
    data/tratados/bairros.gpkg               saida da Etapa 2 (01_demanda.py)
    data/tratados/pontos_demanda_setor.gpkg  setores habitados (01_demanda.py)
    data/tratados/alocacao_setores.parquet   fracao de cada setor em cada bairro

Saidas:
    data/tratados/demanda_bairro.csv     h_i por bairro, uma coluna por (beta, alpha)
    data/tratados/demanda_setor.csv      h_i por setor, idem
    results/tabelas/coeficientes.csv   cada coeficiente com valor, fonte e natureza

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/02_demanda_ecommerce.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# =============================================================================
# PARAMETROS
# =============================================================================

RAIZ = Path(__file__).resolve().parents[1]
ARQ_BAIRROS = RAIZ / "data" / "tratados" / "bairros.gpkg"
ARQ_DEMANDA = RAIZ / "data" / "tratados" / "demanda_bairro.csv"
ARQ_PONTOS_SETOR = RAIZ / "data" / "tratados" / "pontos_demanda_setor.gpkg"
ARQ_ALOCACAO = RAIZ / "data" / "tratados" / "alocacao_setores.parquet"
ARQ_DEMANDA_SETOR = RAIZ / "data" / "tratados" / "demanda_setor.csv"
ARQ_COEFICIENTES = RAIZ / "results" / "tabelas" / "coeficientes.csv"

POP_REFERENCIA = 537_211  # Censo 2022, Florianopolis
N_BAIRROS_ESPERADO = 87

# -----------------------------------------------------------------------------
# Mercado brasileiro de e-commerce — ABIACOM (ex-ABComm)
# -----------------------------------------------------------------------------
# Fonte: ABIACOM, painel dados.abcomm.org, dados de 2025, acessado em setembro de
# 2026, complementado por envio de dados solicitado pelo formulario da propria
# pagina. Dados OBSERVADOS, nao projecao.
#
# A pagina salva em docs/ nao serve para extracao: os valores sao carregados por
# JavaScript a partir da API da entidade e o HTML gravado vem com os campos
# vazios. Os numeros abaixo vieram da pagina renderizada e do envio por e-mail.
#
# A ABIACOM declara que o recorte por estado e estimativa propria, nao observacao.
# Por isso usa-se exclusivamente o agregado nacional.
ANO_REFERENCIA = 2025
FATURAMENTO_ANUAL = 235_500_000_000  # R$ 235,5 bilhoes
TICKET_MEDIO = 536.60                # R$
PEDIDOS_ANUAIS = 438_900_000         # 438,9 milhoes
COMPRADORES_ONLINE = 94_200_000      # 94,2 milhoes

# Tolerancia do teste de coerencia faturamento / pedidos == ticket medio.
TOLERANCIA_TICKET = 0.001  # 0,1%

# -----------------------------------------------------------------------------
# Populacao brasileira
# -----------------------------------------------------------------------------
# IBGE, Censo Demografico 2022, populacao residente do Brasil.
POP_BRASIL_2022 = 203_080_756

# -----------------------------------------------------------------------------
# alpha — disposicao a usar ponto de retirada
# -----------------------------------------------------------------------------
# Tres cenarios, de dois estudos brasileiros de preferencia declarada que usam OS
# MESMOS 18 CENARIOS: Firmeza (2021) declara ter transcrito as situacoes de
# Silva (2018), o que torna os valores diretamente comparaveis entre cidades.
#
#   conservador  Firmeza (2021), Fortaleza, cenario 17, Tabela 8, p. 60:
#                mesmo custo de frete, mesmo prazo, deslocamento extra ate 2 km.
#                Piso empirico observado entre cidades brasileiras.
#   referencia   Silva (2018), Belo Horizonte, demanda minima: custo e prazo
#                equivalentes aos da entrega convencional, deslocamento extra de
#                2 a 5 km. E a REFERENCIA porque reproduz as premissas deste
#                modelo, que nao incorpora desconto de frete nem reducao de prazo.
#   otimista     Silva (2018), demanda maxima: desconto de 50% no frete e prazo
#                48 h menor.
#
# A mesma condicao experimental produz 21,2% em Fortaleza e 50,03% em Belo
# Horizonte — menos da metade. E essa dispersao entre cidades brasileiras que
# justifica tratar alpha por cenarios, e nao por valor unico.
ALPHA_CENARIOS = {
    "conservador": 0.2120,
    "referencia": 0.5003,
    "otimista": 0.9203,
}
ALPHA_PADRAO = "referencia"

# -----------------------------------------------------------------------------
# beta — elasticidade da penetracao em relacao a renda
# -----------------------------------------------------------------------------
# tau_i = tau_BR x (renda_i / renda_municipio) ** beta
#
# beta = 0 e o controle: sem ajuste de renda, h_i fica exatamente proporcional a
# Pop_i. Serve de teste de regressao do proprio modelo.
BETA_VALORES = (0.0, 0.3, 0.6)

# -----------------------------------------------------------------------------
# Normalizacao de tau
# -----------------------------------------------------------------------------
# A media de (renda_i / renda_municipio) ** beta nao e 1 (desigualdade de Jensen),
# entao, sem normalizar, beta desloca tambem o NIVEL total da demanda, e nao so o
# seu gradiente espacial.
#
# Ligado (padrao): tau_i e reescalado para que a media ponderada pela populacao
# seja exatamente tau_BR. A correcao e um escalar uniforme — nao altera a solucao
# otima nem os indicadores percentuais, so as magnitudes absolutas. Com ela, a
# demanda total fica constante entre os tres beta e as tabelas ficam comparaveis.
#
# Desligado: tau_i segue a formula literal, e a demanda total varia com beta.
NORMALIZAR_TAU = True

DIAS_DO_ANO = 365  # denominador declarado de f

# =============================================================================
# INFRAESTRUTURA
# =============================================================================


class FalhaDeSanidade(Exception):
    """Erro de verificacao: o dado nao esta como o metodo exige."""


def checar(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise FalhaDeSanidade(mensagem)


def exigir_preenchido(nome: str, valor) -> None:
    """Impede a execucao enquanto um coeficiente nao tiver valor real."""
    checar(
        valor is not None,
        f"O parametro {nome} ainda esta como None. Preencher com o valor da fonte "
        "declarada no bloco de parametros antes de rodar o script.",
    )


def titulo(texto: str) -> None:
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def subtitulo(texto: str) -> None:
    print()
    print("-" * 78)
    print(texto)
    print("-" * 78)


def numero_br(valor: float, casas: int = 0) -> str:
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def rotulo_coluna(beta: float, cenario: str) -> str:
    """Nome da coluna de h_i para um par (beta, cenario de alpha)."""
    return f"h_beta{beta:g}_{cenario}".replace(".", "")


# =============================================================================
# COEFICIENTES
# =============================================================================


def validar_parametros() -> None:
    """Confere que todo coeficiente tem valor e que os publicados sao coerentes."""
    subtitulo("1. Coeficientes declarados")

    for nome, valor in [
        ("ANO_REFERENCIA", ANO_REFERENCIA),
        ("FATURAMENTO_ANUAL", FATURAMENTO_ANUAL),
        ("TICKET_MEDIO", TICKET_MEDIO),
        ("PEDIDOS_ANUAIS", PEDIDOS_ANUAIS),
        ("COMPRADORES_ONLINE", COMPRADORES_ONLINE),
        ("POP_BRASIL_2022", POP_BRASIL_2022),
    ]:
        exigir_preenchido(nome, valor)

    for cenario, valor in ALPHA_CENARIOS.items():
        exigir_preenchido(f"ALPHA_CENARIOS['{cenario}']", valor)
        checar(
            0.0 < valor <= 1.0,
            f"alpha do cenario '{cenario}' e {valor}; deve estar em (0, 1].",
        )
    checar(
        ALPHA_PADRAO in ALPHA_CENARIOS,
        f"ALPHA_PADRAO '{ALPHA_PADRAO}' nao esta em ALPHA_CENARIOS.",
    )
    checar(
        all(b >= 0 for b in BETA_VALORES),
        "Ha beta negativo: inverteria o sentido do gradiente de renda.",
    )
    checar(0.0 in BETA_VALORES, "BETA_VALORES precisa conter 0, que e o controle.")

    # Coerencia interna do dado publicado: faturamento / pedidos == ticket medio.
    ticket_derivado = FATURAMENTO_ANUAL / PEDIDOS_ANUAIS
    desvio = abs(ticket_derivado - TICKET_MEDIO) / TICKET_MEDIO
    print(f"ano de referencia ...................... {ANO_REFERENCIA} (dado observado)")
    print(f"faturamento anual ...................... R$ {numero_br(FATURAMENTO_ANUAL)}")
    print(f"pedidos anuais ......................... {numero_br(PEDIDOS_ANUAIS)}")
    print(f"compradores online ..................... {numero_br(COMPRADORES_ONLINE)}")
    print(f"ticket medio publicado ................. R$ {numero_br(TICKET_MEDIO, 2)}")
    print(f"ticket medio derivado .................. R$ {numero_br(ticket_derivado, 2)}  (desvio {desvio * 100:.3f}%)")
    checar(
        desvio <= TOLERANCIA_TICKET,
        f"Incoerencia nos dados da ABIACOM: faturamento/pedidos = {ticket_derivado:.2f}, "
        f"ticket medio publicado = {TICKET_MEDIO:.2f}, desvio {desvio * 100:.3f}% acima "
        f"da tolerancia de {TOLERANCIA_TICKET * 100:.1f}%.",
    )
    print("coerencia faturamento/pedidos/ticket ... conferida")

    print(f"\ncenarios de alpha ...................... " + " · ".join(
        f"{c}: {v:.4f}" for c, v in ALPHA_CENARIOS.items()
    ))
    print(f"valores de beta ........................ " + " · ".join(f"{b:g}" for b in BETA_VALORES))
    print(f"normalizacao de tau .................... {'ligada' if NORMALIZAR_TAU else 'desligada'}")


def calcular_tau_br() -> float:
    """Penetracao nacional: compradores online sobre a populacao brasileira."""
    tau_br = COMPRADORES_ONLINE / POP_BRASIL_2022
    print(f"\ntau_BR = {numero_br(COMPRADORES_ONLINE)} / {numero_br(POP_BRASIL_2022)}")
    print(f"       = {tau_br:.6f}   ({tau_br * 100:.2f}% da populacao compra online)")
    checar(0 < tau_br < 1, f"tau_BR fora de (0,1): {tau_br}.")
    return tau_br


def calcular_f() -> float:
    """Pedidos por comprador por dia.

    DERIVADO, nao dado primario: a ABIACOM publica pedidos e compradores anuais;
    a taxa diaria e razao dos dois dividida pelo numero de dias do ano. O
    denominador adotado e o ano corrido (365 dias), nao dias uteis, porque o
    e-commerce opera em todos os dias da semana. Vira nota de rodape na
    monografia.
    """
    pedidos_por_comprador_ano = PEDIDOS_ANUAIS / COMPRADORES_ONLINE
    f = pedidos_por_comprador_ano / DIAS_DO_ANO
    print(f"\nf = ({numero_br(PEDIDOS_ANUAIS)} / {numero_br(COMPRADORES_ONLINE)}) / {DIAS_DO_ANO}")
    print(f"  = {pedidos_por_comprador_ano:.4f} pedidos por comprador por ano")
    print(f"  = {f:.6f} pedido por comprador por dia          [DERIVADO]")
    checar(f > 0, "f nao positivo.")
    return f


# =============================================================================
# DEMANDA POR BAIRRO
# =============================================================================


def carregar_bairros() -> pd.DataFrame:
    """Le a saida da Etapa 2 e reconfere os totais."""
    subtitulo("2. Bairros — saida da Etapa 2")

    checar(
        ARQ_BAIRROS.exists(),
        f"{ARQ_BAIRROS} nao existe. Rodar antes: python src/01_demanda.py",
    )
    bairros = gpd.read_file(ARQ_BAIRROS)
    colunas = ["CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL", "POP", "RENDA_MEDIA", "PESO_RENDA"]
    faltando = [c for c in colunas if c not in bairros.columns]
    checar(
        not faltando,
        f"Colunas ausentes em bairros.gpkg: {faltando}. Rodar novamente src/01_demanda.py.",
    )
    dados = pd.DataFrame(bairros[colunas])

    print(f"bairros lidos .......................... {len(dados)}")
    print(f"populacao total ........................ {numero_br(dados['POP'].sum())}")
    checar(
        len(dados) == N_BAIRROS_ESPERADO,
        f"Foram lidos {len(dados)} bairros, esperado {N_BAIRROS_ESPERADO}.",
    )
    checar(
        int(dados["POP"].sum()) == POP_REFERENCIA,
        f"Populacao dos bairros e {int(dados['POP'].sum())}, esperado {POP_REFERENCIA}.",
    )
    checar(
        dados["RENDA_MEDIA"].notna().all(),
        "Ha bairro sem renda media; o ajuste por renda nao pode ser calculado. "
        f"Bairros: {dados.loc[dados['RENDA_MEDIA'].isna(), 'NM_BAIRRO'].tolist()}",
    )
    checar(
        bool((dados["RENDA_MEDIA"] > 0).all()),
        "Ha bairro com renda media nao positiva; (renda_i/renda_mun)**beta divergiria.",
    )
    return dados


def carregar_setores() -> pd.DataFrame:
    """Le os setores habitados gravados pela Etapa 2 e reconfere os totais."""
    subtitulo("10. Setores censitarios — saida da Etapa 2")

    checar(
        ARQ_PONTOS_SETOR.exists(),
        f"{ARQ_PONTOS_SETOR} nao existe. Rodar antes: python src/01_demanda.py",
    )
    setores = gpd.read_file(ARQ_PONTOS_SETOR)
    colunas = ["CD_SETOR", "CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL", "POP", "RENDA_MEDIA", "PESO_RENDA"]
    faltando = [c for c in colunas if c not in setores.columns]
    checar(not faltando, f"Colunas ausentes em pontos_demanda_setor.gpkg: {faltando}.")
    dados = pd.DataFrame(setores[colunas])

    print(f"setores habitados lidos ................ {len(dados)}")
    print(f"populacao total ........................ {numero_br(dados['POP'].sum())}")
    checar(dados["CD_SETOR"].is_unique, "CD_SETOR repetido na camada de setores.")
    checar(
        int(dados["POP"].sum()) == POP_REFERENCIA,
        f"Populacao dos setores e {int(dados['POP'].sum())}, esperado {POP_REFERENCIA}.",
    )
    checar(bool((dados["POP"] > 0).all()), "Ha setor sem populacao entre os pontos de demanda.")
    checar(
        dados["RENDA_MEDIA"].notna().all() and bool((dados["RENDA_MEDIA"] > 0).all()),
        "Ha setor habitado sem renda positiva; (renda_i/renda_mun)**beta nao pode ser calculado.",
    )
    return dados


def conferir_setor_contra_bairro(demanda_setor: pd.DataFrame, demanda_bairro: pd.DataFrame) -> None:
    """Com beta = 0, a demanda dos setores rateada por bairro reproduz a do bairro.

    Sem ajuste de renda h_i e proporcional a populacao nas duas unidades, entao a
    soma por bairro tem de bater. A tolerancia absorve so o arredondamento da
    populacao do bairro para inteiro (feito na Etapa 2 sobre as fracoes de rateio).
    Com beta > 0 as unidades divergem de proposito: o setor usa a propria renda.
    """
    subtitulo("12. Setor x bairro — conferencia com beta = 0")

    alocacao = pd.read_parquet(ARQ_ALOCACAO)
    base = alocacao.merge(demanda_setor, on="CD_SETOR", how="inner", suffixes=("", "_DOMINANTE"))
    por_capita = demanda_bairro[rotulo_coluna(0.0, ALPHA_PADRAO)].sum() / POP_REFERENCIA
    for beta in BETA_VALORES:
        coluna = rotulo_coluna(beta, ALPHA_PADRAO)
        soma_bairro = (base[coluna] * base["FRACAO"]).groupby(base["CD_BAIRRO"]).sum()
        referencia = demanda_bairro.set_index("CD_BAIRRO")[coluna].reindex(soma_bairro.index)
        diferenca = (soma_bairro - referencia).abs()
        print(f"   beta = {beta:g}: maior diferenca por bairro {diferenca.max():.4f} encomendas/dia "
              f"({diferenca.max() / por_capita:.2f} hab equivalentes) · total setor "
              f"{numero_br(demanda_setor[coluna].sum(), 2)} x bairro {numero_br(demanda_bairro[coluna].sum(), 2)}")
        if beta == 0.0:
            checar(
                float(diferenca.max()) <= por_capita * 1.0,
                f"Com beta = 0 a demanda por setor, somada por bairro, diverge da do bairro em "
                f"{diferenca.max():.4f} — mais que um habitante. Conferir alocacao_setores.parquet.",
            )
    print("   beta = 0 reproduz a demanda de cada bairro ... conferido")


def renda_do_municipio(dados: pd.DataFrame) -> float:
    """Renda media do municipio, agregada exatamente como a dos bairros.

    Como renda_b = soma(renda_s w_s) / soma(w_s) e PESO_RENDA_b = soma(w_s), a
    media ponderada das rendas de bairro por PESO_RENDA reproduz termo a termo a
    media ponderada sobre os setores. Nao e a media simples dos 87 bairros.
    """
    renda_mun = float(
        (dados["RENDA_MEDIA"] * dados["PESO_RENDA"]).sum() / dados["PESO_RENDA"].sum()
    )
    print(f"renda media do municipio ............... R$ {numero_br(renda_mun, 2)}")
    print(
        f"renda por unidade de demanda ............ "
        f"min R$ {numero_br(dados['RENDA_MEDIA'].min(), 2)} · "
        f"max R$ {numero_br(dados['RENDA_MEDIA'].max(), 2)} · "
        f"razao {dados['RENDA_MEDIA'].max() / dados['RENDA_MEDIA'].min():.2f}x"
    )
    checar(renda_mun > 0, "Renda do municipio nao positiva.")
    return renda_mun


def calcular_tau(
    dados: pd.DataFrame, tau_br: float, renda_mun: float, beta: float
) -> tuple[pd.Series, dict]:
    """tau_i para um beta, com normalizacao opcional e truncamento em [0,1]."""
    razao_renda = dados["RENDA_MEDIA"] / renda_mun
    tau = tau_br * razao_renda**beta

    diagnostico = {"beta": beta, "fator_normalizacao": 1.0, "truncados": 0}

    if NORMALIZAR_TAU:
        # Media de tau ponderada pela populacao, que e o que precisa valer tau_BR.
        media_ponderada = float((tau * dados["POP"]).sum() / dados["POP"].sum())
        fator = tau_br / media_ponderada
        tau = tau * fator
        diagnostico["fator_normalizacao"] = fator

    estourou = tau > 1.0
    if estourou.any():
        diagnostico["truncados"] = int(estourou.sum())
        print(
            f"   AVISO beta={beta:g}: {int(estourou.sum())} unidade(s) de demanda com tau > 1, truncados em 1,0 — "
            + ", ".join(
                f"{n} ({v:.3f})"
                for n, v in zip(dados.loc[estourou, "NM_BAIRRO"], tau[estourou])
            )
        )
        if NORMALIZAR_TAU:
            print("   o truncamento quebra a normalizacao exata; ver o resumo abaixo.")
        tau = tau.clip(upper=1.0)

    checar(
        bool(((tau >= 0) & (tau <= 1)).all()),
        f"tau fora de [0,1] para beta={beta} mesmo apos truncamento.",
    )
    return tau, diagnostico


def montar_demanda(
    dados: pd.DataFrame, tau_br: float, f: float, renda_mun: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula h_i para as 9 combinacoes de beta e alpha."""
    subtitulo("3. Penetracao ajustada pela renda (tau) e demanda (h)")

    saida = dados.copy()
    diagnosticos = []

    for beta in BETA_VALORES:
        tau, diagnostico = calcular_tau(dados, tau_br, renda_mun, beta)
        coluna_tau = f"tau_beta{beta:g}".replace(".", "")
        saida[coluna_tau] = tau

        # Demanda per capita = tau_i * f * alpha. A razao entre o maior e o menor
        # valor nao depende de alpha, so de beta: e a medida do gradiente.
        diagnostico["tau_min"] = float(tau.min())
        diagnostico["tau_max"] = float(tau.max())
        diagnostico["razao_percapita"] = float(tau.max() / tau.min())
        diagnostico["tau_medio_ponderado"] = float((tau * dados["POP"]).sum() / dados["POP"].sum())

        for cenario, alpha in ALPHA_CENARIOS.items():
            saida[rotulo_coluna(beta, cenario)] = dados["POP"] * tau * f * alpha

        diagnostico["demanda_total_referencia"] = float(
            saida[rotulo_coluna(beta, ALPHA_PADRAO)].sum()
        )
        diagnosticos.append(diagnostico)

    return saida, pd.DataFrame(diagnosticos)


# =============================================================================
# VERIFICACOES
# =============================================================================


def verificar(demanda: pd.DataFrame, tau_br: float, f: float) -> None:
    """Verificacoes obrigatorias da etapa."""
    subtitulo("4. Verificacoes")

    checar(
        int(demanda["POP"].sum()) == POP_REFERENCIA,
        f"Populacao mudou: {int(demanda['POP'].sum())} != {POP_REFERENCIA}.",
    )
    print(f"populacao preservada ................... {numero_br(demanda['POP'].sum())}")

    for beta in BETA_VALORES:
        coluna = f"tau_beta{beta:g}".replace(".", "")
        serie = demanda[coluna]
        checar(
            bool(((serie >= 0) & (serie <= 1)).all()),
            f"tau fora de [0,1] em {coluna}.",
        )
    print(f"tau em [0,1] nos {len(BETA_VALORES)} betas ................. conferido")

    # Teste de regressao do proprio modelo: com beta = 0 nao ha ajuste de renda,
    # logo h_i tem de ser exatamente proporcional a Pop_i.
    for cenario, alpha in ALPHA_CENARIOS.items():
        coluna = rotulo_coluna(0.0, cenario)
        per_capita = demanda[coluna] / demanda["POP"]
        esperado = tau_br * f * alpha
        checar(
            bool(np.allclose(per_capita, esperado, rtol=1e-12, atol=0.0)),
            f"Com beta=0, h_i nao ficou proporcional a Pop_i no cenario '{cenario}': "
            f"per capita varia de {per_capita.min():.10f} a {per_capita.max():.10f}, "
            f"esperado {esperado:.10f} constante.",
        )
    print("beta=0 proporcional a populacao ........ conferido nos 3 cenarios de alpha")

    for coluna in [c for c in demanda.columns if c.startswith("h_")]:
        checar(
            bool((demanda[coluna] >= 0).all()),
            f"Demanda negativa em {coluna}.",
        )
    print("nenhuma demanda negativa ............... conferido")


def resumir(demanda: pd.DataFrame, diagnosticos: pd.DataFrame, tau_br: float) -> None:
    """Resumo para conferencia manual."""
    subtitulo("5. Efeito do beta sobre o gradiente de renda")

    print(f"   {'beta':>5} {'tau min':>9} {'tau max':>9} {'razao max/min':>14} "
          f"{'tau medio pond.':>16} {'fator norm.':>12} {'trunc.':>7}")
    for linha in diagnosticos.itertuples():
        print(
            f"   {linha.beta:>5g} {linha.tau_min:>9.4f} {linha.tau_max:>9.4f} "
            f"{linha.razao_percapita:>13.2f}x {linha.tau_medio_ponderado:>16.6f} "
            f"{linha.fator_normalizacao:>12.4f} {linha.truncados:>7}"
        )
    print(f"\n   tau_BR de referencia: {tau_br:.6f}")
    if NORMALIZAR_TAU:
        print("   com a normalizacao ligada, tau medio ponderado deve igualar tau_BR")
        print("   e a demanda total deve ser identica nos tres betas.")
    else:
        print("   com a normalizacao desligada, tau medio ponderado se afasta de tau_BR")
        print("   e a demanda total varia entre os betas.")

    subtitulo("6. Demanda diaria total por combinacao (encomendas/dia)")
    print(f"   {'beta':>5}" + "".join(f"{c:>16}" for c in ALPHA_CENARIOS))
    for beta in BETA_VALORES:
        linha = f"   {beta:>5g}"
        for cenario in ALPHA_CENARIOS:
            linha += f"{numero_br(demanda[rotulo_coluna(beta, cenario)].sum(), 0):>16}"
        print(linha)
    print(f"\n   alpha altera so a ESCALA da demanda: multiplica todos os h_i pela mesma")
    print(f"   constante, logo nao muda o conjunto otimo de pontos nem as coberturas")
    print(f"   percentuais. Quem redistribui demanda entre bairros e o beta.")

    subtitulo("7. Demanda por regiao funcional (cenario de referencia)")
    ordem = ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]
    print(f"   {'regiao':<14} {'populacao':>11} {'pop %':>7}" + "".join(
        f"{'h beta=' + f'{b:g}':>13}" for b in BETA_VALORES
    ) + f"{'part. b=0.6':>13}")
    agregado = demanda.groupby("REGIAO_FUNCIONAL").agg(
        POP=("POP", "sum"),
        **{
            rotulo_coluna(b, ALPHA_PADRAO): (rotulo_coluna(b, ALPHA_PADRAO), "sum")
            for b in BETA_VALORES
        },
    ).reindex(ordem)
    total_maior_beta = agregado[rotulo_coluna(max(BETA_VALORES), ALPHA_PADRAO)].sum()
    for regiao, linha in agregado.iterrows():
        texto = f"   {regiao:<14} {numero_br(linha['POP']):>11} {100 * linha['POP'] / agregado['POP'].sum():>6.1f}%"
        for beta in BETA_VALORES:
            texto += f"{numero_br(linha[rotulo_coluna(beta, ALPHA_PADRAO)], 0):>13}"
        texto += f"{100 * linha[rotulo_coluna(max(BETA_VALORES), ALPHA_PADRAO)] / total_maior_beta:>12.1f}%"
        print(texto)

    subtitulo("8. Bairros nos extremos do ajuste por renda")
    beta_maior = max(BETA_VALORES)
    coluna_tau = f"tau_beta{beta_maior:g}".replace(".", "")
    extremos = demanda.sort_values(coluna_tau, ascending=False)
    print(f"   maiores tau (beta={beta_maior:g}):")
    for linha in extremos.head(5).itertuples():
        print(
            f"      {linha.NM_BAIRRO[:26]:<26} {linha.REGIAO_FUNCIONAL:<12} "
            f"R$ {numero_br(linha.RENDA_MEDIA, 2):>10}   tau {getattr(linha, coluna_tau):.4f}"
        )
    print(f"   menores tau (beta={beta_maior:g}):")
    for linha in extremos.tail(5).itertuples():
        print(
            f"      {linha.NM_BAIRRO[:26]:<26} {linha.REGIAO_FUNCIONAL:<12} "
            f"R$ {numero_br(linha.RENDA_MEDIA, 2):>10}   tau {getattr(linha, coluna_tau):.4f}"
        )


# =============================================================================
# SAIDAS
# =============================================================================


def tabela_de_coeficientes(tau_br: float, f: float, renda_mun: float) -> pd.DataFrame:
    """Quadro de coeficientes com valor, fonte, ano e natureza do dado."""
    fonte_abiacom = (
        "ABIACOM (ex-ABComm), painel dados.abcomm.org, acesso em set/2026, "
        "complementado por envio de dados solicitado no formulario da pagina"
    )
    linhas = [
        {
            "coeficiente": "faturamento_anual_BR",
            "valor": FATURAMENTO_ANUAL,
            "unidade": "R$/ano",
            "natureza": "observado",
            "ano": ANO_REFERENCIA,
            "fonte": fonte_abiacom,
            "sensibilidade": "nao usado no calculo de h; serve a coerencia do ticket",
        },
        {
            "coeficiente": "pedidos_anuais_BR",
            "valor": PEDIDOS_ANUAIS,
            "unidade": "pedidos/ano",
            "natureza": "observado",
            "ano": ANO_REFERENCIA,
            "fonte": fonte_abiacom,
            "sensibilidade": "entra em f",
        },
        {
            "coeficiente": "compradores_online_BR",
            "valor": COMPRADORES_ONLINE,
            "unidade": "pessoas",
            "natureza": "observado",
            "ano": ANO_REFERENCIA,
            "fonte": fonte_abiacom,
            "sensibilidade": "entra em tau_BR e em f",
        },
        {
            "coeficiente": "ticket_medio_BR",
            "valor": TICKET_MEDIO,
            "unidade": "R$/pedido",
            "natureza": "observado",
            "ano": ANO_REFERENCIA,
            "fonte": fonte_abiacom,
            "sensibilidade": "usado so na verificacao de coerencia",
        },
        {
            "coeficiente": "populacao_BR",
            "valor": POP_BRASIL_2022,
            "unidade": "pessoas",
            "natureza": "observado",
            "ano": 2022,
            "fonte": "IBGE, Censo Demografico 2022, populacao residente do Brasil",
            "sensibilidade": "denominador de tau_BR",
        },
        {
            "coeficiente": "populacao_Florianopolis",
            "valor": POP_REFERENCIA,
            "unidade": "pessoas",
            "natureza": "observado",
            "ano": 2022,
            "fonte": "IBGE, Censo Demografico 2022, agregados por setor censitario",
            "sensibilidade": "base de h_i",
        },
        {
            "coeficiente": "tau_BR",
            "valor": round(tau_br, 6),
            "unidade": "fracao da populacao",
            "natureza": "derivado",
            "ano": ANO_REFERENCIA,
            "fonte": "compradores_online_BR / populacao_BR",
            "sensibilidade": "ancora do nivel de penetracao",
        },
        {
            "coeficiente": "f",
            "valor": round(f, 8),
            "unidade": "pedidos/comprador/dia",
            "natureza": "derivado",
            "ano": ANO_REFERENCIA,
            "fonte": (
                f"(pedidos_anuais_BR / compradores_online_BR) / {DIAS_DO_ANO}; "
                "denominador em ano corrido, nao dias uteis"
            ),
            "sensibilidade": "escalar uniforme: nao altera o conjunto otimo",
        },
        {
            "coeficiente": "renda_media_Florianopolis",
            "valor": round(renda_mun, 2),
            "unidade": "R$/mes",
            "natureza": "derivado",
            "ano": 2022,
            "fonte": (
                "IBGE, Censo 2022, V06004 ponderado por V06001; agregacao propria "
                "(src/01_demanda.py)"
            ),
            "sensibilidade": "denominador do ajuste por renda",
        },
    ]
    for cenario, alpha in ALPHA_CENARIOS.items():
        origem = {
            "conservador": "Firmeza (2021), Fortaleza, cenario 17, Tabela 8, p. 60",
            "referencia": "Silva (2018), Belo Horizonte, demanda minima",
            "otimista": "Silva (2018), Belo Horizonte, demanda maxima",
        }[cenario]
        condicao = {
            "conservador": "mesmo frete, mesmo prazo, deslocamento extra ate 2 km",
            "referencia": "frete e prazo equivalentes, deslocamento extra de 2 a 5 km",
            "otimista": "desconto de 50% no frete e prazo 48 h menor",
        }[cenario]
        linhas.append(
            {
                "coeficiente": f"alpha_{cenario}",
                "valor": alpha,
                "unidade": "fracao dos consumidores",
                "natureza": "observado (preferencia declarada)",
                "ano": 2021 if cenario == "conservador" else 2018,
                "fonte": f"{origem}; {condicao}",
                "sensibilidade": "escalar uniforme: nao altera o conjunto otimo",
            }
        )
    for beta in BETA_VALORES:
        linhas.append(
            {
                "coeficiente": f"beta_{beta:g}",
                "valor": beta,
                "unidade": "elasticidade",
                "natureza": "parametro de sensibilidade",
                "ano": "",
                "fonte": (
                    "sem estimativa local; 0 e o controle sem ajuste de renda"
                    if beta == 0
                    else "sem estimativa local; valor testado por sensibilidade"
                ),
                "sensibilidade": "UNICO parametro desta etapa que altera a solucao otima",
            }
        )
    return pd.DataFrame(linhas)


def salvar(
    demanda: pd.DataFrame,
    coeficientes: pd.DataFrame | None,
    arquivo: Path = ARQ_DEMANDA,
    identificacao: tuple[str, ...] = ("CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL"),
) -> None:
    subtitulo(f"Gravacao — {arquivo.name}")
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    ARQ_COEFICIENTES.parent.mkdir(parents=True, exist_ok=True)

    ordem_colunas = (
        list(identificacao) + ["POP", "RENDA_MEDIA"]
        + [f"tau_beta{b:g}".replace(".", "") for b in BETA_VALORES]
        + [rotulo_coluna(b, c) for b in BETA_VALORES for c in ALPHA_CENARIOS]
    )
    # Arredondamento de apresentacao: a precisao de ponto flutuante nao carrega
    # informacao aqui e polui a tabela que vai para a monografia.
    tabela = demanda[ordem_colunas].copy()
    tabela["RENDA_MEDIA"] = tabela["RENDA_MEDIA"].round(2)
    for coluna in tabela.columns:
        if coluna.startswith("tau_"):
            tabela[coluna] = tabela[coluna].round(6)
        elif coluna.startswith("h_"):
            tabela[coluna] = tabela[coluna].round(4)
    tabela.to_csv(arquivo, index=False, sep=";", decimal=",")
    gravados = [arquivo]
    if coeficientes is not None:
        coeficientes.to_csv(ARQ_COEFICIENTES, index=False, sep=";", decimal=",")
        gravados.append(ARQ_COEFICIENTES)

    for caminho in gravados:
        print(f"   {caminho.relative_to(RAIZ).as_posix():<45} {caminho.stat().st_size / 1024:>8.1f} KB")
    print(f"   colunas de demanda gravadas ......... {len(BETA_VALORES) * len(ALPHA_CENARIOS)} combinacoes")


# =============================================================================
# EXECUCAO
# =============================================================================


def main() -> int:
    inicio = time.perf_counter()

    titulo(
        "02_demanda_ecommerce.py — demanda de e-commerce por bairro\n"
        f"h_i = Pop_i x tau_i x f x alpha   ·   "
        f"betas: {', '.join(f'{b:g}' for b in BETA_VALORES)}   ·   "
        f"alphas: {', '.join(ALPHA_CENARIOS)}   ·   "
        f"tau {'normalizado' if NORMALIZAR_TAU else 'sem normalizacao'}"
    )

    validar_parametros()
    tau_br = calcular_tau_br()
    f = calcular_f()

    dados = carregar_bairros()
    renda_mun = renda_do_municipio(dados)

    demanda, diagnosticos = montar_demanda(dados, tau_br, f, renda_mun)
    verificar(demanda, tau_br, f)
    resumir(demanda, diagnosticos, tau_br)

    coeficientes = tabela_de_coeficientes(tau_br, f, renda_mun)
    salvar(demanda, coeficientes)

    # -------------------------------------------------------------------------
    # Mesma conta por setor censitario. tau_BR, f e alpha sao os mesmos; so a
    # renda passa a ser a do setor.
    # -------------------------------------------------------------------------
    setores = carregar_setores()
    renda_mun_setor = renda_do_municipio(setores)
    # A renda do bairro e a media dos setores ponderada por PESO_RENDA, entao a do
    # municipio tem de sair identica pelas duas unidades.
    checar(
        abs(renda_mun_setor - renda_mun) < 0.01,
        f"Renda do municipio diverge entre unidades: setor R$ {renda_mun_setor:.2f} x "
        f"bairro R$ {renda_mun:.2f}.",
    )
    subtitulo("11. Demanda por setor")
    demanda_setor, diagnosticos_setor = montar_demanda(setores, tau_br, f, renda_mun)
    verificar(demanda_setor, tau_br, f)
    print(f"\n   {'beta':>5} {'tau min':>9} {'tau max':>9} {'razao max/min':>14} {'trunc.':>7} {'total (ref.)':>14}")
    for linha in diagnosticos_setor.itertuples():
        print(f"   {linha.beta:>5g} {linha.tau_min:>9.4f} {linha.tau_max:>9.4f} "
              f"{linha.razao_percapita:>13.2f}x {linha.truncados:>7} "
              f"{numero_br(linha.demanda_total_referencia, 2):>14}")
    conferir_setor_contra_bairro(demanda_setor, demanda)
    salvar(
        demanda_setor, None, ARQ_DEMANDA_SETOR,
        ("CD_SETOR", "CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL"),
    )

    total = demanda[rotulo_coluna(BETA_VALORES[0], ALPHA_PADRAO)].sum()
    titulo(
        f"Concluido em {time.perf_counter() - inicio:.1f} s  ·  "
        f"{len(demanda)} bairros e {len(demanda_setor)} setores  ·  "
        f"{numero_br(total, 0)} encomendas/dia no cenario de referencia (beta=0)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FalhaDeSanidade as erro:
        print()
        print("!" * 78)
        print("VERIFICACAO DE SANIDADE FALHOU")
        print("!" * 78)
        print(erro)
        sys.exit(1)
