"""
06_modelo.py — modelo de localizacao, resolucao lexicografica em dois estagios.

Implementa exatamente a formulacao de docs/formulacao.md:

    Estagio 1:  Max Z1 = SOMA_i SOMA_j h_i y_ij                        (9)
    Estagio 2:  Min Z2 = SOMA_i SOMA_j h_i d^a_ij y_ij                 (10)
                sujeito a SOMA_i SOMA_j h_i y_ij >= 0.999999 * Z1*     (11)

    (2')  SOMA_j y_ij <= 1        para todo i
    (3)   y_ij <= x_j             para todo i, j
    (4)   SOMA_j x_j = p
    (5)   x_j binaria
    (6)   y_ij em [0,1]
    (7)   y_ij = 0 se d_ij > r    -- distancia FISICA, nunca a generalizada

Pontos que nao podem ser alterados, sob pena de regressao:
    - (2') e <= 1, NAO = 1. Igualdade inviabiliza o modelo quando p e r sao
      restritivos, que e justamente o caso de C4.
    - (7) usa d_ij fisica. A atratividade entra so em d^a_ij = d_ij / a_j, no
      Estagio 2.
    - a tolerancia 0.999999 e proposital: igualdade estrita pode inviabilizar o
      Estagio 2 por arredondamento do solver.
    - y_ij e CONTINUA em [0,1]. Com x_j binaria a solucao otima ja sai inteira, e
      resolve muito mais rapido.

Cenario misto (C6): o raio e a metrica de cada ponto de demanda dependem da sua
regiao funcional — 800 m a pe em Centro/Sede e Continente, 1.500 m de carro no
Norte, Leste e Sul. So muda o conjunto de pares elegiveis e a matriz de onde vem
cada d_ij; a formulacao e a mesma.

Unidade de demanda (I): o SETOR censitario habitado (964 pontos, padrao) ou o
BAIRRO (87 pontos, para comparacao). A formulacao e identica; so muda o conjunto I e a matriz d_ij. Com
setores, a cobertura por bairro e por regiao e obtida somando a demanda coberta
de cada setor pela fracao dele em cada bairro (alocacao_setores.parquet), de
modo que as tabelas das duas unidades sejam diretamente comparaveis.

Entradas:
    data/tratados/demanda_bairro.csv        h_i por bairro (Etapa 3)
    data/tratados/demanda_setor.csv         h_i por setor (Etapa 3), com --unidade setor
    data/tratados/alocacao_setores.parquet  fracao de cada setor em cada bairro
    data/tratados/candidatos_com_aj.gpkg    a_j por candidato (Etapa 5)
    data/tratados/dist_rede_{walk,drive}[_setor].parquet  e dist_euclid[_setor].parquet (Etapa 6)

Saidas:
    results/logs/execucoes.csv            uma linha por execucao
    results/nao_usados/tabelas/solucao_<cenario>_<variante>_<metrica>[_setor].csv   (teste)

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/06_modelo.py                     # cenario de teste (C2), por setor
    python src/06_modelo.py --unidade bairro    # idem, por bairro (fora do escopo)
"""

from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pulp

# =============================================================================
# PARAMETROS
# =============================================================================

RAIZ = Path(__file__).resolve().parents[1]
DIR_TRATADOS = RAIZ / "data" / "tratados"
DIR_LOGS = RAIZ / "results" / "logs"   # redefinido em main() conforme a unidade
# Resultados que NAO entram na monografia vao para results/nao_usados/. A rodada
# por bairro saiu do escopo quando o setor censitario virou a unidade de demanda
# (docs/decisoes_etapa10.md): ela continua reproduzivel, mas grava ali, para nao
# se misturar com o que vai para o texto.
DIR_RESULTADOS = {"setor": RAIZ / "results", "bairro": RAIZ / "results" / "nao_usados"}
# A solucao do cenario de teste deste script e conferencia, nao resultado do texto:
# vai sempre para nao_usados. Os resultados da monografia saem de 07_cenarios.py.
DIR_SOLUCAO_TESTE = RAIZ / "results" / "nao_usados" / "tabelas"

ARQ_DEMANDA_BAIRRO = DIR_TRATADOS / "demanda_bairro.csv"
ARQ_ALOCACAO = DIR_TRATADOS / "alocacao_setores.parquet"
ARQ_CANDIDATOS = DIR_TRATADOS / "candidatos_com_aj.gpkg"
ARQ_LOG = DIR_LOGS / "execucoes.csv"      # idem

CRS_TRABALHO = 31982
N_CANDIDATOS_ESPERADO = 856

# -----------------------------------------------------------------------------
# Unidade de demanda
# -----------------------------------------------------------------------------
# 'setor'   um ponto por setor censitario habitado — configuracao PRINCIPAL. O
#           ponto do bairro ficava longe da populacao em muitos bairros; ver
#           docs/decisoes_etapa10.md.
# 'bairro'  87 pontos, um por bairro — a configuracao original, para comparacao.
UNIDADES = {
    "bairro": {
        "demanda": DIR_TRATADOS / "demanda_bairro.csv", "chave": "CD_BAIRRO",
        "sufixo": "", "n_esperado": 87,
    },
    "setor": {
        "demanda": DIR_TRATADOS / "demanda_setor.csv", "chave": "CD_SETOR",
        "sufixo": "_setor", "n_esperado": 964,
    },
}
UNIDADE_PADRAO = "setor"

# -----------------------------------------------------------------------------
# Cenarios e metricas (identicos aos da Etapa 6)
# -----------------------------------------------------------------------------
CENARIOS = {
    "C1": {"p": 10, "r": 800},
    "C2": {"p": 20, "r": 800},
    "C3": {"p": 40, "r": 800},
    "C4": {"p": 20, "r": 500},
    "C5": {"p": 20, "r": 1500},
    # C6 — cenario MISTO: o raio e a metrica dependem da regiao funcional do ponto
    # de demanda i. Nas regioes densas (Centro/Sede e Continente) o acesso e a pe,
    # 800 m na rede caminhavel; nas demais, de carro, 1.500 m na rede viaria. As
    # matrizes sao as mesmas de C2 e C5: nenhum grafo e recalculado. Para cada i,
    # os pares elegiveis E a distancia d_ij da funcao objetivo vem da metrica da
    # sua regiao. O resto da formulacao nao muda.
    "C6": {
        "p": 20, "r": "800/1500",
        "raio_por_regiao": {
            "Centro/Sede": ("walk", 800),
            "Continente": ("walk", 800),
            "Norte": ("drive", 1500),
            "Leste": ("drive", 1500),
            "Sul": ("drive", 1500),
        },
    },
}
METRICA_POR_RAIO = {500: "walk", 800: "walk", 1500: "drive"}
METRICA_MISTA = "misto"

NOME_MATRIZ = {"walk": "dist_rede_walk", "drive": "dist_rede_drive", "euclid": "dist_euclid"}


def caminho_matriz(metrica: str, unidade: str) -> Path:
    return DIR_TRATADOS / f"{NOME_MATRIZ[metrica]}{UNIDADES[unidade]['sufixo']}.parquet"


def eh_misto(cenario: str) -> bool:
    return "raio_por_regiao" in CENARIOS[cenario]


def metrica_de_rede(cenario: str) -> str:
    """Metrica de rede do cenario: a do raio, ou 'misto' no cenario por regiao."""
    if eh_misto(cenario):
        return METRICA_MISTA
    return METRICA_POR_RAIO[CENARIOS[cenario]["r"]]

# -----------------------------------------------------------------------------
# Coeficientes de demanda (Etapa 3)
# -----------------------------------------------------------------------------
BETA_PADRAO = 0.3
ALPHA_PADRAO = "referencia"

# -----------------------------------------------------------------------------
# Variantes de ponderacao
# -----------------------------------------------------------------------------
# 'aj_estimado' usa o indice da Etapa 5; 'aj_unitario' faz a_j = 1 para todo j,
# equivalente a formulacao classica das p-medianas. A cobertura do Estagio 1 e
# identica nas duas por construcao, o que permite atribuir integralmente ao
# indice as diferencas observadas.
VARIANTES = ("aj_estimado", "aj_unitario")
VARIANTE_PADRAO = "aj_estimado"

# -----------------------------------------------------------------------------
# Resolucao
# -----------------------------------------------------------------------------
TOLERANCIA_ESTAGIO2 = 0.999999   # proposital; ver docstring
SOLVER_PADRAO = "CBC"
TEMPO_LIMITE_S = 600

# Gap relativo exigido do solver. Com ZERO, o status "Optimal" significa
# otimalidade COMPROVADA, e nao "dentro da tolerancia padrao" — o HiGHS assume
# 0,01% por padrao e encerraria com gap de 0,00142% no Estagio 2, que seria
# reportado como otimo sem o ser em sentido estrito. A Secao 3.2.5 promete gap
# nulo, entao a exigencia e feita explicitamente.
GAP_RELATIVO_EXIGIDO = 0.0

# Cenario de teste
CENARIO_TESTE = "C2"

# =============================================================================
# INFRAESTRUTURA
# =============================================================================


class FalhaDeSanidade(Exception):
    """Erro de verificacao: o dado nao esta como o metodo exige."""


def checar(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise FalhaDeSanidade(mensagem)


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


def coluna_demanda(beta: float, alpha: str) -> str:
    return f"h_beta{beta:g}_{alpha}".replace(".", "")


def obter_solver(nome: str):
    """Devolve o solver do PuLP e a sua versao."""
    if nome.upper() == "CBC":
        solver = pulp.PULP_CBC_CMD(
            msg=False, timeLimit=TEMPO_LIMITE_S, gapRel=GAP_RELATIVO_EXIGIDO
        )
        versao = f"CBC via PuLP {pulp.__version__}"
    elif nome.upper() == "HIGHS":
        import highspy
        checar(
            pulp.HiGHS(msg=False).available(),
            "HiGHS nao esta disponivel no ambiente. Instalar highspy.",
        )
        # output_flag=False silencia o log que o highspy escreve direto no stdout,
        # que o msg=False do PuLP nao alcanca.
        solver = pulp.HiGHS(
            msg=False, timeLimit=TEMPO_LIMITE_S, gapRel=GAP_RELATIVO_EXIGIDO,
            output_flag=False, log_to_console=False,
        )
        versao = f"HiGHS {highspy.Highs().version()} via PuLP {pulp.__version__}"
    else:
        raise FalhaDeSanidade(f"Solver desconhecido: {nome}.")
    return solver, versao


# =============================================================================
# DADOS
# =============================================================================


def carregar_dados(
    beta: float, alpha: str, unidade: str = UNIDADE_PADRAO
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Demanda da unidade escolhida e candidatos com a_j."""
    cfg = UNIDADES[unidade]
    arquivo = cfg["demanda"]
    checar(arquivo.exists(), f"{arquivo} nao existe. Rodar src/02_demanda_ecommerce.py")
    checar(ARQ_CANDIDATOS.exists(), f"{ARQ_CANDIDATOS} nao existe. Rodar src/04_atratividade.py")

    # Os codigos precisam ser TEXTO: no parquet das matrizes eles sao string, e um
    # merge entre int64 e string falha em silencio, produzindo NaN.
    demanda = pd.read_csv(arquivo, sep=";", decimal=",", dtype={"CD_BAIRRO": str, "CD_SETOR": str})
    coluna = coluna_demanda(beta, alpha)
    checar(
        coluna in demanda.columns,
        f"Coluna de demanda '{coluna}' ausente. Disponiveis: "
        f"{[c for c in demanda.columns if c.startswith('h_')]}",
    )
    demanda = demanda.rename(columns={coluna: "H"})
    checar(
        len(demanda) == cfg["n_esperado"],
        f"{len(demanda)} pontos de demanda ({unidade}), esperado {cfg['n_esperado']}.",
    )
    checar(demanda[cfg["chave"]].is_unique, f"{cfg['chave']} repetido em {arquivo.name}.")
    checar(bool((demanda["H"] > 0).all()), f"Ha {unidade} com demanda nao positiva.")

    candidatos = gpd.read_file(ARQ_CANDIDATOS).to_crs(CRS_TRABALHO)
    checar(
        len(candidatos) == N_CANDIDATOS_ESPERADO,
        f"{len(candidatos)} candidatos, esperado {N_CANDIDATOS_ESPERADO}.",
    )
    checar(
        bool(((candidatos["AJ"] >= 0.2) & (candidatos["AJ"] <= 1.0)).all()),
        "Ha a_j fora de [0,2; 1] — d_ij/a_j divergiria.",
    )
    return demanda, candidatos


def carregar_matriz(metrica: str, n_candidatos: int, unidade: str = UNIDADE_PADRAO) -> pd.DataFrame:
    caminho = caminho_matriz(metrica, unidade)
    opcao = "" if unidade == "bairro" else f" --unidade {unidade}"
    checar(caminho.exists(), f"{caminho} nao existe. Rodar src/05_distancias.py{opcao}")
    matriz = pd.read_parquet(caminho)
    chave = UNIDADES[unidade]["chave"]
    matriz[chave] = matriz[chave].astype(str)
    checar(
        int(matriz["ID_CANDIDATO"].max()) == n_candidatos - 1,
        f"A matriz {metrica} indexa ate {int(matriz['ID_CANDIDATO'].max())}, mas ha "
        f"{n_candidatos} candidatos. Reprocessar src/05_distancias.py.",
    )
    return matriz


def pares_elegiveis(
    cenario: str, metrica: str, demanda: pd.DataFrame, n_candidatos: int, unidade: str
) -> pd.DataFrame:
    """Pares (i, j) com d_ij <= r, ja podados — a poda vem ANTES das variaveis.

    No cenario misto, cada ponto de demanda leva os pares da metrica e do raio da
    SUA regiao funcional; a coluna MODO registra de qual matriz veio cada par.
    """
    cfg = CENARIOS[cenario]
    chave = UNIDADES[unidade]["chave"]
    if metrica != METRICA_MISTA:
        matriz = carregar_matriz(metrica, n_candidatos, unidade)
        pares = matriz[matriz["DIST_M"] <= cfg["r"]].copy()
        pares["MODO"] = metrica
        return pares

    checar(eh_misto(cenario), f"A metrica '{METRICA_MISTA}' so se aplica a cenario misto.")
    regiao_de = demanda.set_index(chave)["REGIAO_FUNCIONAL"]
    faltando = set(regiao_de) - set(cfg["raio_por_regiao"])
    checar(not faltando, f"Regiao sem raio definido no cenario {cenario}: {sorted(faltando)}.")
    partes = []
    matrizes: dict[str, pd.DataFrame] = {}
    for regiao, (modo, raio) in cfg["raio_por_regiao"].items():
        if modo not in matrizes:
            matrizes[modo] = carregar_matriz(modo, n_candidatos, unidade)
        matriz = matrizes[modo]
        pontos_da_regiao = set(regiao_de.index[regiao_de == regiao])
        parte = matriz[matriz[chave].isin(pontos_da_regiao) & (matriz["DIST_M"] <= raio)].copy()
        parte["MODO"] = modo
        partes.append(parte)
    pares = pd.concat(partes, ignore_index=True)
    # Cada par (i, j) vem de uma unica matriz: um ponto de demanda pertence a uma
    # unica regiao. Duplicata aqui seria regiao repetida ou chave inconsistente.
    checar(
        not pares.duplicated([chave, "ID_CANDIDATO"]).any(),
        f"{cenario}: par (i, j) repetido entre as metricas do cenario misto.",
    )
    return pares


def cobertura_por_bairro(por_unidade: pd.DataFrame, unidade: str) -> pd.DataFrame:
    """Leva a cobertura de cada ponto de demanda para o bairro.

    Por bairro, a tabela e a propria. Por setor, soma H e demanda coberta de cada
    setor multiplicadas pela fracao dele em cada bairro — a mesma fracao que a
    Etapa 2 usou para ratear a populacao dos setores divididos. Assim a cobertura
    de um bairro deixa de ser 0 ou 100% e passa a medir QUANTO dele esta coberto.
    """
    nomes = pd.read_csv(ARQ_DEMANDA_BAIRRO, sep=";", decimal=",", dtype={"CD_BAIRRO": str})
    nomes = nomes[["CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL"]]

    if unidade == "bairro":
        tabela = por_unidade[["CD_BAIRRO", "H", "COBERTA"]]
    else:
        chave = UNIDADES[unidade]["chave"]
        alocacao = pd.read_parquet(ARQ_ALOCACAO)[["CD_SETOR", "CD_BAIRRO", "FRACAO"]]
        alocacao = alocacao.astype({"CD_SETOR": str, "CD_BAIRRO": str})
        base = por_unidade.merge(alocacao, left_on=chave, right_on="CD_SETOR", how="left")
        checar(base["FRACAO"].notna().all(), "Ponto de demanda sem linha em alocacao_setores.parquet.")
        base["H"] = base["H"] * base["FRACAO"]
        base["COBERTA"] = base["COBERTA"] * base["FRACAO"]
        tabela = base.groupby("CD_BAIRRO", as_index=False)[["H", "COBERTA"]].sum()

    por_bairro = nomes.merge(tabela, on="CD_BAIRRO", how="left")
    checar(por_bairro["H"].notna().all(), "Bairro sem demanda apos a agregacao.")
    por_bairro["cobertura_pct"] = 100 * por_bairro["COBERTA"] / por_bairro["H"]
    return por_bairro[["CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL", "H", "COBERTA", "cobertura_pct"]]


# =============================================================================
# MODELO
# =============================================================================


def resolver(
    cenario: str,
    variante: str = VARIANTE_PADRAO,
    metrica: str | None = None,
    beta: float = BETA_PADRAO,
    alpha: str = ALPHA_PADRAO,
    solver_nome: str = SOLVER_PADRAO,
    verboso: bool = True,
    unidade: str = UNIDADE_PADRAO,
    aj_externo: pd.Series | None = None,
    z1_reaproveitado: float | None = None,
) -> dict:
    """Resolve um cenario em dois estagios e devolve resultado e indicadores.

    `aj_externo` substitui o a_j do arquivo de candidatos (variante aj_estimado),
    para a analise de sensibilidade do indice. `z1_reaproveitado` pula o Estagio 1
    e usa o Z1* informado: como o indice so entra no Estagio 2, o Z1* de um cenario
    nao depende do a_j, e resolve-lo de novo a cada variacao do indice seria
    redundante. Quem passa o valor e responsavel por te-lo obtido do mesmo
    cenario, unidade, beta e alpha.
    """
    cfg = CENARIOS[cenario]
    p, r = cfg["p"], cfg["r"]
    metrica = metrica or metrica_de_rede(cenario)
    checar(variante in VARIANTES, f"Variante desconhecida: {variante}.")
    checar(unidade in UNIDADES, f"Unidade de demanda desconhecida: {unidade}.")
    checar(
        (metrica == METRICA_MISTA) == eh_misto(cenario),
        f"{cenario}: o cenario misto usa so a metrica '{METRICA_MISTA}', e ela so vale para ele.",
    )
    checar(aj_externo is None or variante == "aj_estimado",
           "a_j externo so faz sentido na variante aj_estimado.")
    chave = UNIDADES[unidade]["chave"]

    demanda, candidatos = carregar_dados(beta, alpha, unidade)

    if verboso:
        subtitulo(f"{cenario}: p={p}, r={r} m, metrica={metrica}, variante={variante}, unidade={unidade}")
        print(f"demanda: beta={beta:g}, alpha={alpha} "
              f"({numero_br(demanda['H'].sum(), 1)} encomendas/dia)")

    # ------------------------------------------------------------------
    # Poda por r ANTES de criar as variaveis, usando a distancia FISICA.
    # ------------------------------------------------------------------
    pares = pares_elegiveis(cenario, metrica, demanda, len(candidatos), unidade)
    n_pares_matriz = len(demanda) * len(candidatos)
    checar(len(pares) > 0, f"Nenhum par sobrevive a r = {r} m na metrica {metrica}.")

    h = demanda.set_index(chave)["H"]
    if aj_externo is not None:
        checar(aj_externo.index.equals(candidatos.index), "a_j externo nao indexa os candidatos.")
        checar(bool(((aj_externo >= 0.2 - 1e-9) & (aj_externo <= 1.0 + 1e-9)).all()),
               "a_j externo fora de [0,2; 1] — d_ij/a_j divergiria.")
        aj = aj_externo
    elif variante == "aj_estimado":
        aj = candidatos["AJ"]
    else:
        aj = pd.Series(1.0, index=candidatos.index)
    pares["H"] = pares[chave].map(h)
    pares["AJ"] = pares["ID_CANDIDATO"].map(aj)
    # Blindagem contra juncao silenciosamente vazia por incompatibilidade de tipo
    # entre o CSV da demanda e o parquet das distancias.
    checar(
        pares["H"].notna().all(),
        f"{int(pares['H'].isna().sum())} pares ficaram sem demanda apos a juncao. "
        f"Provavel incompatibilidade de tipo em {chave} entre a tabela de demanda "
        "e a matriz de distancias.",
    )
    checar(pares["AJ"].notna().all(), "Ha par sem a_j apos a juncao.")
    # Distancia generalizada: a atratividade entra SO aqui, no Estagio 2.
    pares["DIST_GEN"] = pares["DIST_M"] / pares["AJ"]

    demanda_elegivel = sorted(set(pares[chave]))
    candidatos_elegiveis = sorted(set(pares["ID_CANDIDATO"]))

    # Teto de Z1*: ponto de demanda sem candidato ao alcance nao e coberto por p algum.
    teto_demanda = float(h.reindex(demanda_elegivel).sum())
    demanda_total = float(h.sum())
    checar(
        teto_demanda > 0,
        "O teto de Z1* saiu zero: nenhum ponto de demanda elegivel casou com a tabela "
        f"de demanda. Conferir o tipo de {chave} nas duas fontes.",
    )

    if verboso:
        print(f"pares apos a poda ...................... {numero_br(len(pares))} de "
              f"{numero_br(n_pares_matriz)} ({100 * len(pares) / n_pares_matriz:.2f}%)")
        print(f"pontos de demanda com candidato ........ {len(demanda_elegivel)} de {len(demanda)}")
        print(f"candidatos elegiveis ................... {len(candidatos_elegiveis)} de {len(candidatos)}")
        print(f"teto de Z1* ............................ {numero_br(teto_demanda, 1)} "
              f"({100 * teto_demanda / demanda_total:.1f}% da demanda do municipio)")

    checar(
        p <= len(candidatos_elegiveis),
        f"p = {p} excede os {len(candidatos_elegiveis)} candidatos elegiveis a r = {r} m.",
    )

    solver, versao_solver = obter_solver(solver_nome)
    chaves = list(zip(pares[chave], pares["ID_CANDIDATO"]))
    peso = dict(zip(chaves, pares["H"]))
    dist_gen = dict(zip(chaves, pares["DIST_GEN"]))
    por_ponto: dict[str, list] = {}
    for (i, j) in chaves:
        por_ponto.setdefault(i, []).append(j)

    def montar(nome: str):
        """Constroi as variaveis e as restricoes comuns aos dois estagios."""
        modelo = pulp.LpProblem(nome, pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", candidatos_elegiveis, cat=pulp.LpBinary)
        # (6) y contínua em [0,1]: com x binaria a solucao otima ja sai inteira.
        y = pulp.LpVariable.dicts("y", chaves, lowBound=0, upBound=1, cat=pulp.LpContinuous)
        # (2') SOMA_j y_ij <= 1 — cobertura PARCIAL permitida. Nunca = 1.
        for i, js in por_ponto.items():
            modelo += pulp.lpSum(y[(i, j)] for j in js) <= 1, f"c2_{i}"
        # (3) y_ij <= x_j
        for (i, j) in chaves:
            modelo += y[(i, j)] <= x[j], f"c3_{i}_{j}"
        # (4) SOMA_j x_j = p
        modelo += pulp.lpSum(x[j] for j in candidatos_elegiveis) == p, "c4_p"
        return modelo, x, y

    cobertura = pulp.lpSum  # alias legivel

    # ------------------------------------------------------------------
    # Estagio 1 — maximizar a demanda coberta
    # ------------------------------------------------------------------
    inicio1 = time.perf_counter()
    if z1_reaproveitado is None:
        m1, x1, y1 = montar(f"{cenario}_estagio1")
        m1 += -cobertura(peso[k] * y1[k] for k in chaves)   # Max via minimizacao do negativo
        status1 = m1.solve(solver)
        checar(
            pulp.LpStatus[status1] == "Optimal",
            f"Estagio 1 nao resolveu ate otimalidade: {pulp.LpStatus[status1]}.",
        )
        z1 = float(sum(peso[k] * y1[k].value() for k in chaves))
        n_restricoes_e1 = len(m1.constraints)
        status1_txt = pulp.LpStatus[status1]
    else:
        z1 = float(z1_reaproveitado)
        n_restricoes_e1 = None
        status1_txt = "reaproveitado"
    tempo1 = time.perf_counter() - inicio1

    # ------------------------------------------------------------------
    # Estagio 2 — minimizar a distancia generalizada ponderada
    # ------------------------------------------------------------------
    inicio2 = time.perf_counter()
    m2, x2, y2 = montar(f"{cenario}_estagio2")
    m2 += cobertura(peso[k] * dist_gen[k] * y2[k] for k in chaves)
    # (11) preserva a cobertura otima, com a tolerancia proposital.
    m2 += cobertura(peso[k] * y2[k] for k in chaves) >= TOLERANCIA_ESTAGIO2 * z1, "c11"
    status2 = m2.solve(solver)
    tempo2 = time.perf_counter() - inicio2
    checar(
        pulp.LpStatus[status2] == "Optimal",
        f"Estagio 2 nao resolveu ate otimalidade: {pulp.LpStatus[status2]}.",
    )
    z2 = float(pulp.value(m2.objective))
    z1_estagio2 = float(sum(peso[k] * y2[k].value() for k in chaves))

    # ------------------------------------------------------------------
    # Verificacoes da solucao
    # ------------------------------------------------------------------
    abertos = [j for j in candidatos_elegiveis if x2[j].value() > 0.5]
    checar(
        len(abertos) == p,
        f"Foram abertos {len(abertos)} pontos, mas p = {p}.",
    )
    valores_y = np.array([y2[k].value() for k in chaves])
    fracionarios = int(np.sum((valores_y > 1e-6) & (valores_y < 1 - 1e-6)))
    # Com x binaria, a solucao do Estagio 1 sai inteira. No Estagio 2 pode restar
    # fracao, e ela e consequencia DIRETA da tolerancia proposital de (11): o
    # modelo abre mao de ate (1 - 0.999999) da cobertura otima para reduzir a
    # distancia, e o faz encolhendo um unico y. A folga usada tem de caber na
    # tolerancia; se exceder, ha erro de formulacao.
    folga_usada = z1 - z1_estagio2
    folga_permitida = (1 - TOLERANCIA_ESTAGIO2) * z1
    # A comparacao usa a tolerancia de viabilidade do solver (1e-6 absoluto): o
    # CBC satisfaz (11) ate a sua propria precisao, e exigir igualdade exata em
    # ponto flutuante faria a verificacao disparar sempre que a folga fosse usada
    # por inteiro, que e o caso normal.
    checar(
        folga_usada <= folga_permitida + 1e-6,
        f"O Estagio 2 abriu mao de {folga_usada:.9f} de cobertura, acima da folga "
        f"de {folga_permitida:.9f} que a tolerancia de {TOLERANCIA_ESTAGIO2} permite.",
    )
    checar(
        z1_estagio2 >= TOLERANCIA_ESTAGIO2 * z1 - 1e-6,
        f"O Estagio 2 degradou a cobertura: {z1_estagio2:.4f} < {TOLERANCIA_ESTAGIO2 * z1:.4f}.",
    )
    checar(
        z1 <= teto_demanda + 1e-6,
        f"Z1* = {z1:.4f} excede o teto de {teto_demanda:.4f}, o que e impossivel.",
    )

    # ------------------------------------------------------------------
    # Indicadores
    # ------------------------------------------------------------------
    atribuicao = pares.copy()
    atribuicao["Y"] = [y2[k].value() for k in chaves]
    atribuicao = atribuicao[atribuicao["Y"] > 1e-6].copy()
    atribuicao["DEMANDA_COBERTA"] = atribuicao["H"] * atribuicao["Y"]

    # Distancia media ponderada, calculada SO sobre a demanda coberta.
    dist_media = float(
        (atribuicao["DIST_M"] * atribuicao["DEMANDA_COBERTA"]).sum()
        / atribuicao["DEMANDA_COBERTA"].sum()
    )
    dist_media_gen = float(
        (atribuicao["DIST_GEN"] * atribuicao["DEMANDA_COBERTA"]).sum()
        / atribuicao["DEMANDA_COBERTA"].sum()
    )
    # No cenario misto, a mesma media separada por modo de acesso: somar metros a
    # pe e de carro numa media so esconde a diferenca entre os dois regimes.
    dist_por_modo = {}
    for modo, sub in atribuicao.groupby("MODO"):
        dist_por_modo[modo] = {
            "dist_media_ponderada_m": float(
                (sub["DIST_M"] * sub["DEMANDA_COBERTA"]).sum() / sub["DEMANDA_COBERTA"].sum()
            ),
            "demanda_coberta": float(sub["DEMANDA_COBERTA"].sum()),
        }
    pares_por_modo = pares.groupby("MODO").size().to_dict()

    # Cobertura por ponto de demanda, depois por BAIRRO e por regiao. Quem tem a
    # atribuicao y_ij e o modelo, entao as tabelas nascem aqui — o script de mapas
    # apenas le, nunca reotimiza.
    coberta_por_ponto = atribuicao.groupby(chave)["DEMANDA_COBERTA"].sum()
    por_unidade = pd.DataFrame(
        {chave: h.index, "H": h.to_numpy(), "COBERTA": coberta_por_ponto.reindex(h.index).fillna(0.0).to_numpy()}
    )
    por_unidade["cobertura_pct"] = 100 * por_unidade["COBERTA"] / por_unidade["H"]
    por_bairro = cobertura_por_bairro(por_unidade, unidade)
    checar(
        abs(por_bairro["H"].sum() - demanda_total) < 1e-6 * max(demanda_total, 1.0),
        f"A demanda agregada por bairro ({por_bairro['H'].sum():.6f}) difere da total "
        f"({demanda_total:.6f}) — fracoes de alocacao inconsistentes.",
    )
    checar(
        abs(por_bairro["COBERTA"].sum() - z1_estagio2) < 1e-6 * max(z1_estagio2, 1.0),
        "A demanda coberta agregada por bairro difere de Z1 do Estagio 2.",
    )
    por_regiao = (
        por_bairro.groupby("REGIAO_FUNCIONAL")
        .agg(demanda=("H", "sum"), coberta=("COBERTA", "sum"))
    )
    por_regiao.index.name = "REGIAO"
    por_regiao["cobertura_pct"] = 100 * por_regiao["coberta"] / por_regiao["demanda"]

    # Pontos abertos que nao cobrem demanda alguma: medida direta do retorno
    # decrescente de p. Quando p excede o que a estrutura de pares elegiveis
    # comporta, a restricao (4) obriga a abrir pontos sem funcao.
    carga = atribuicao.groupby("ID_CANDIDATO")["DEMANDA_COBERTA"].sum()
    sem_demanda = [j for j in abertos if float(carga.get(j, 0.0)) <= 1e-9]

    # a_j de referencia para os indicadores de atratividade: o do arquivo, ou o
    # externo, quando a execucao e de sensibilidade do indice.
    aj_todos = aj_externo if aj_externo is not None else candidatos["AJ"]
    resultado = {
        "cenario": cenario, "p": p, "r_m": r, "metrica": metrica, "variante": variante,
        "beta": beta, "alpha": alpha,
        "unidade_demanda": unidade,
        "n_demanda": len(demanda), "n_candidatos": len(candidatos),
        "n_demanda_elegiveis": len(demanda_elegivel),
        "n_candidatos_elegiveis": len(candidatos_elegiveis),
        "n_pares_podados": len(pares),
        "n_binarias": len(candidatos_elegiveis),
        "n_continuas": len(chaves),
        "n_restricoes_e1": n_restricoes_e1, "n_restricoes_e2": len(m2.constraints),
        "Z1": z1, "Z2": z2, "Z1_no_estagio2": z1_estagio2,
        "demanda_total": demanda_total, "teto_Z1": teto_demanda,
        "cobertura_pct_municipio": 100 * z1 / demanda_total,
        "cobertura_pct_teto": 100 * z1 / teto_demanda,
        "dist_media_ponderada_m": dist_media,
        "dist_media_generalizada_m": dist_media_gen,
        "n_pontos_sem_demanda": len(sem_demanda),
        "aj_medio_abertos": float(aj_todos.reindex(abertos).mean()),
        "aj_medio_elegiveis": float(aj_todos.reindex(candidatos_elegiveis).mean()),
        "aj_medio_conjunto": float(aj_todos.mean()),
        "y_fracionarios": fracionarios,
        "folga_cobertura_usada": folga_usada,
        "folga_cobertura_permitida": folga_permitida,
        "status_e1": status1_txt, "status_e2": pulp.LpStatus[status2],
        "gap_e1": 0.0, "gap_e2": 0.0,
        "gap_relativo_exigido": GAP_RELATIVO_EXIGIDO,
        "criterio_gap": (
            f"gapRel exigido = {GAP_RELATIVO_EXIGIDO}; status Optimal nos dois "
            "estagios significa otimalidade comprovada"
        ),
        "tempo_e1_s": tempo1, "tempo_e2_s": tempo2, "tempo_total_s": tempo1 + tempo2,
        "solver": versao_solver,
        "python": platform.python_version(),
        "executado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "_abertos": abertos, "_atribuicao": atribuicao, "_por_regiao": por_regiao,
        "_candidatos": candidatos, "_elegiveis": candidatos_elegiveis,
        "_por_bairro": por_bairro,
        "_por_unidade": por_unidade,
        "_sem_demanda": sem_demanda,
        "_dist_por_modo": dist_por_modo,
        "_pares_por_modo": pares_por_modo,
        "_aj": aj_todos,
    }
    # So o cenario misto ganha colunas por modo no log: nos demais haveria um modo
    # so, igual a media geral.
    if metrica == METRICA_MISTA:
        for modo in ("walk", "drive"):
            resultado[f"n_pares_{modo}"] = int(pares_por_modo.get(modo, 0))
            info = dist_por_modo.get(modo)
            resultado[f"dist_media_ponderada_{modo}_m"] = info["dist_media_ponderada_m"] if info else None
            resultado[f"demanda_coberta_{modo}"] = info["demanda_coberta"] if info else 0.0
    return resultado


# =============================================================================
# RELATORIO E LOG
# =============================================================================


def relatar(resultado: dict) -> None:
    subtitulo("Resultado")
    r = resultado
    print(f"Z1* (demanda coberta) .................. {numero_br(r['Z1'], 2)} encomendas/dia")
    print(f"Z2  (distancia generalizada ponderada) . {numero_br(r['Z2'], 0)}")
    print(f"cobertura sobre a demanda do municipio . {r['cobertura_pct_municipio']:.2f}%")
    print(f"cobertura sobre o TETO de r={r['r_m']} m ..... {r['cobertura_pct_teto']:.2f}%")
    print(f"   teto: {numero_br(r['teto_Z1'], 2)} de {numero_br(r['demanda_total'], 2)} "
          f"({100 * r['teto_Z1'] / r['demanda_total']:.1f}%)")
    print(f"distancia media ponderada (coberta) .... {r['dist_media_ponderada_m']:.0f} m")
    print(f"distancia generalizada media ........... {r['dist_media_generalizada_m']:.0f} m")
    print(f"pontos abertos ......................... {len(r['_abertos'])}")

    print(f"\nporte da instancia:")
    print(f"   pontos de demanda ({r['unidade_demanda']}) ........... {r['n_demanda']} "
          f"({r['n_demanda_elegiveis']} elegiveis)")
    print(f"   candidatos .......................... {r['n_candidatos']} "
          f"({r['n_candidatos_elegiveis']} elegiveis)")
    print(f"   pares apos a poda ................... {numero_br(r['n_pares_podados'])}")
    print(f"   variaveis binarias .................. {r['n_binarias']}")
    print(f"   variaveis continuas ................. {numero_br(r['n_continuas'])}")
    print(f"   restricoes .......................... E1 {numero_br(r['n_restricoes_e1'])} · "
          f"E2 {numero_br(r['n_restricoes_e2'])}")
    print(f"   y fracionarios na solucao ........... {r['y_fracionarios']}  "
          f"(folga usada {r['folga_cobertura_usada']:.6f} de "
          f"{r['folga_cobertura_permitida']:.6f} permitida pela tolerancia)")
    print(f"   status .............................. E1 {r['status_e1']} · E2 {r['status_e2']}")
    print(f"   tempo ............................... E1 {r['tempo_e1_s']:.2f} s · "
          f"E2 {r['tempo_e2_s']:.2f} s · total {r['tempo_total_s']:.2f} s")
    print(f"   solver .............................. {r['solver']}")

    print(f"\ncobertura por regiao funcional:")
    print(f"   {'regiao':<14} {'demanda':>12} {'coberta':>12} {'%':>8}")
    for regiao, linha in r["_por_regiao"].iterrows():
        print(f"   {regiao:<14} {numero_br(linha['demanda'], 1):>12} "
              f"{numero_br(linha['coberta'], 1):>12} {linha['cobertura_pct']:>7.1f}%")

    candidatos = r["_candidatos"]
    abertos = candidatos.loc[r["_abertos"]].copy()
    carga = r["_atribuicao"].groupby("ID_CANDIDATO")["DEMANDA_COBERTA"].sum()
    abertos["DEMANDA"] = carga.reindex(abertos.index).fillna(0.0)
    abertos["N_PONTOS"] = r["_atribuicao"].groupby("ID_CANDIDATO").size().reindex(abertos.index).fillna(0).astype(int)

    print(f"\nos {len(abertos)} pontos selecionados:")
    print(f"   {'#':>3} {'nome':<30} {'tipo':<16} {'bairro':<20} {'a_j':>6} "
          f"{'demanda':>9} {'pontos':>8}")
    for n, linha in enumerate(abertos.sort_values("DEMANDA", ascending=False).itertuples(), 1):
        nome = (linha.NOME or "(sem nome)")[:30]
        print(f"   {n:>3} {nome:<30} {linha.TIPO:<16} {linha.NM_BAIRRO[:20]:<20} "
              f"{linha.AJ:>6.3f} {numero_br(linha.DEMANDA, 1):>9} {linha.N_PONTOS:>8}")

    print(f"\n   por tipo: " + " · ".join(
        f"{k}: {v}" for k, v in abertos["TIPO"].value_counts().items()
    ))
    # A comparacao correta e contra os ELEGIVEIS do cenario, nao contra o conjunto
    # inteiro: o Estagio 1 so pode escolher entre os que estao ao alcance de r.
    print(f"   a_j medio: abertos {r['aj_medio_abertos']:.4f} · "
          f"elegiveis {r['aj_medio_elegiveis']:.4f} · conjunto {r['aj_medio_conjunto']:.4f}")
    if r["n_pontos_sem_demanda"]:
        print(f"   pontos abertos sem demanda alguma ... {r['n_pontos_sem_demanda']}")


def registrar(resultado: dict) -> None:
    """Acrescenta uma linha ao log de execucoes.

    O log e relido e regravado, em vez de anexado as cegas: se o conjunto de
    colunas mudou entre versoes do script, o anexo desalinharia as colunas sem
    erro nenhum.
    """
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    linha = {k: v for k, v in resultado.items() if not k.startswith("_")}
    tabela = pd.DataFrame([linha])
    if ARQ_LOG.exists():
        anterior = pd.read_csv(ARQ_LOG, sep=";", decimal=",", dtype=str)
        tabela = pd.concat([anterior, tabela.astype(str)], ignore_index=True)
    tabela.to_csv(ARQ_LOG, index=False, sep=";", decimal=",")
    print(f"\nregistrado em {ARQ_LOG.relative_to(RAIZ).as_posix()}")


def salvar_solucao(resultado: dict) -> None:
    DIR_SOLUCAO_TESTE.mkdir(parents=True, exist_ok=True)
    r = resultado
    candidatos = r["_candidatos"]
    abertos = candidatos.loc[r["_abertos"]].copy()
    carga = r["_atribuicao"].groupby("ID_CANDIDATO")["DEMANDA_COBERTA"].sum()
    abertos["DEMANDA_ATENDIDA"] = carga.reindex(abertos.index).fillna(0.0).round(3)
    colunas = ["OSM_ID", "FONTE", "TIPO", "NOME", "NM_BAIRRO", "REGIAO_FUNCIONAL",
               "AJ", "DEMANDA_ATENDIDA"]
    sufixo = UNIDADES[r["unidade_demanda"]]["sufixo"]
    nome = f"solucao_{r['cenario']}_{r['variante']}_{r['metrica']}{sufixo}.csv"
    caminho = DIR_SOLUCAO_TESTE / nome
    pd.DataFrame(abertos[[c for c in colunas if c in abertos.columns]]).to_csv(
        caminho, index=False, sep=";", decimal=","
    )
    print(f"solucao gravada em {caminho.relative_to(RAIZ).as_posix()}")


# =============================================================================
# EXECUCAO
# =============================================================================


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cenario de teste do modelo em dois estagios.")
    parser.add_argument("--unidade", choices=sorted(UNIDADES), default=UNIDADE_PADRAO,
                        help="unidade de demanda: setor censitario (padrao) ou bairro")
    unidade = parser.parse_args().unidade
    global DIR_LOGS, ARQ_LOG
    DIR_LOGS = DIR_RESULTADOS[unidade] / "logs"
    ARQ_LOG = DIR_LOGS / "execucoes.csv"

    inicio = time.perf_counter()
    cfg = CENARIOS[CENARIO_TESTE]
    titulo(
        f"06_modelo.py — cenario de teste {CENARIO_TESTE}\n"
        f"p={cfg['p']}, r={cfg['r']} m, metrica={metrica_de_rede(CENARIO_TESTE)}, "
        f"variante={VARIANTE_PADRAO}, beta={BETA_PADRAO:g}, alpha={ALPHA_PADRAO}, unidade={unidade}"
    )

    resultado = resolver(CENARIO_TESTE, unidade=unidade)
    relatar(resultado)
    registrar(resultado)
    salvar_solucao(resultado)

    titulo(f"Concluido em {time.perf_counter() - inicio:.1f} s")
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
