"""
07_cenarios.py — execucao dos cenarios e consolidacao dos indicadores.

Matriz principal: 5 cenarios x 2 ponderacoes x 2 metricas = 20 execucoes, cada
uma em dois estagios, com alpha de referencia e beta = 0,3; mais o cenario misto
C6 (rede caminhavel ou viaria conforme a regiao), so na rede: 2 execucoes.

Acrescentam-se:
    - 9 execucoes em C2 (3 beta x 3 alpha), para comprovar empiricamente que
      alpha, sendo escalar uniforme, nao altera o conjunto otimo;
    - cross-check CBC x HiGHS em C2 e C5;
    - os cenarios de rede com beta = 0, para separar o efeito do beta do gradiente
      de renda da cobertura (C2 com beta = 0 ja vem da sensibilidade).

Este script ORQUESTRA: a formulacao vive em src/06_modelo.py e e reusada por
importacao. Reimplementa-la aqui abriria espaco para divergencia entre o cenario
rodado isoladamente e o mesmo cenario rodado na matriz.

A unidade de demanda e escolhida na linha de comando; o padrao e o setor
censitario. Na rodada por setor as tabelas e as solucoes levam o sufixo _setor e
as tabelas por bairro e por regiao sao agregadas a partir dos setores. A rodada por
bairro (--unidade bairro, sem sufixo) saiu do escopo e grava em results/nao_usados/,
com log proprio. Cada rodada substitui so as linhas da sua unidade no log.

Saidas (com --unidade setor, sufixo _setor nos nomes; com --unidade bairro, tabelas e
log vao para results/nao_usados/):
    results/logs/execucoes.csv
    results/tabelas/indicadores_cenarios.csv
    results/tabelas/cobertura_por_regiao_cenarios.csv
    results/tabelas/cobertura_por_bairro_cenarios.csv
    results/tabelas/cobertura_por_setor_cenarios_setor.csv   (so por setor)
    results/tabelas/composicao_por_tipo_cenarios.csv
    results/tabelas/efeito_atratividade.csv
    results/tabelas/sensibilidade_alpha_beta.csv
    results/tabelas/cross_check_solvers.csv
    results/tabelas/equidade_renda.csv
    data/tratados/solucoes_cenarios.gpkg

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/07_cenarios.py                     # setores censitarios (padrao)
    python src/07_cenarios.py --unidade bairro    # bairros (fora do escopo; vai para nao_usados)
    python src/07_cenarios.py --saida v36         # rodada de revisao, em results/v36/
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_TRATADOS = RAIZ / "data" / "tratados"
# Os tres abaixo sao redefinidos em main() conforme a unidade de demanda.
DIR_TABELAS = RAIZ / "results" / "tabelas"
DIR_LOGS = RAIZ / "results" / "logs"
ARQ_LOG = DIR_LOGS / "execucoes.csv"
# Resultados que NAO entram na monografia vao para results/nao_usados/. A rodada
# por bairro saiu do escopo quando o setor censitario virou a unidade de demanda
# (docs/decisoes_etapa10.md): ela continua reproduzivel, mas grava ali, para nao
# se misturar com o que vai para o texto.
DIR_RESULTADOS = {"setor": RAIZ / "results", "bairro": RAIZ / "results" / "nao_usados"}
ARQ_SOLUCOES = DIR_TRATADOS / "solucoes_cenarios.gpkg"   # recebe o sufixo da unidade


def _carregar_modelo():
    """Importa src/06_modelo.py, cujo nome comeca com digito e nao e importavel."""
    caminho = Path(__file__).with_name("06_modelo.py")
    spec = importlib.util.spec_from_file_location("modelo", caminho)
    modelo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modelo)
    return modelo


M = _carregar_modelo()

# =============================================================================
# PARAMETROS
# =============================================================================

# Matriz principal: alpha e beta FIXOS. Como alpha e escalar uniforme, variar os
# outros valores nao altera o conjunto otimo — o que e comprovado adiante.
ALPHA_PRINCIPAL = "referencia"
BETA_PRINCIPAL = 0.3

# Sensibilidade: as nove combinacoes, num unico cenario.
CENARIO_SENSIBILIDADE = "C2"
BETAS = (0.0, 0.3, 0.6)
ALPHAS = ("conservador", "referencia", "otimista")

# Cross-check de solver. C5 alem de C2: e o cenario de maior porte e
# o unico com metrica drive, entao exercita caminho diferente. C6, o cenario
# misto, exercita a montagem de pares a partir de duas matrizes.
CENARIOS_CROSS_CHECK = ("C2", "C5", "C6")
SOLVERES = ("CBC", "HIGHS")

# Equidade por renda. Com beta > 0 a demanda ja pondera pela renda, entao um
# gradiente de renda na cobertura pode vir do PESO dado aos setores ricos ou da
# GEOGRAFIA do comercio. Com beta = 0 a demanda e proporcional a populacao: o que
# sobrar de gradiente nao vem da ponderacao. Por isso os cenarios de rede rodam
# tambem com beta = 0.
BETA_SEM_RENDA = 0.0
N_FAIXAS_RENDA = 5   # quintis POPULACIONAIS: cada faixa reune ~20% dos habitantes

# O log passa a refletir sempre a ultima rodada completa DE CADA UNIDADE. Com
# append, reexecucoes acumulariam linhas duplicadas e a banca nao saberia qual e a
# rodada valida; apagando tudo, a rodada por setor apagaria a por bairro.
LIMPAR_LOG = True

ORDEM_REGIOES = ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]

# =============================================================================
# EXECUCAO DA MATRIZ
# =============================================================================


def metricas_do_cenario(cenario: str) -> list[str]:
    """As duas metricas de cada cenario: euclidiana e a rede pertinente ao raio.

    O cenario misto (C6) tem uma so: a mistura de rede caminhavel e viaria e
    definida pela rede, e nao ha versao euclidiana dela.
    """
    if M.eh_misto(cenario):
        return [M.METRICA_MISTA]
    return ["euclid", M.metrica_de_rede(cenario)]


def rodar_matriz_principal(unidade: str) -> list[dict]:
    n = sum(len(M.VARIANTES) * len(metricas_do_cenario(c)) for c in M.CENARIOS)
    M.subtitulo(f"1. Matriz principal — {n} execucoes")
    resultados = []
    print(f"   {'cenario':<8} {'variante':<14} {'metrica':<8} {'pares':>7} "
          f"{'Z1':>10} {'% mun':>7} {'% teto':>7} {'dist m':>8} {'tempo':>7}")
    for cenario in M.CENARIOS:
        for variante in M.VARIANTES:
            for metrica in metricas_do_cenario(cenario):
                r = M.resolver(
                    cenario, variante=variante, metrica=metrica,
                    beta=BETA_PRINCIPAL, alpha=ALPHA_PRINCIPAL, verboso=False,
                    unidade=unidade,
                )
                r["grupo"] = "matriz_principal"
                resultados.append(r)
                print(f"   {cenario:<8} {variante:<14} {metrica:<8} "
                      f"{r['n_pares_podados']:>7} {r['Z1']:>10.1f} "
                      f"{r['cobertura_pct_municipio']:>6.1f}% {r['cobertura_pct_teto']:>6.1f}% "
                      f"{r['dist_media_ponderada_m']:>8.0f} {r['tempo_total_s']:>6.2f}s")
    return resultados


def rodar_sensibilidade(unidade: str) -> list[dict]:
    M.subtitulo(f"2. Sensibilidade alpha x beta — 9 execucoes em {CENARIO_SENSIBILIDADE}")
    metrica = M.metrica_de_rede(CENARIO_SENSIBILIDADE)
    resultados = []
    for beta in BETAS:
        for alpha in ALPHAS:
            r = M.resolver(
                CENARIO_SENSIBILIDADE, variante="aj_estimado", metrica=metrica,
                beta=beta, alpha=alpha, verboso=False, unidade=unidade,
            )
            r["grupo"] = "sensibilidade_alpha_beta"
            resultados.append(r)
    return resultados


def rodar_beta_zero(unidade: str) -> list[dict]:
    M.subtitulo("3b. Cenarios de rede com beta = 0 — para a analise de equidade")
    resultados = []
    for cenario in M.CENARIOS:
        if cenario == CENARIO_SENSIBILIDADE:
            continue   # ja rodado na sensibilidade alpha x beta
        metrica = M.metrica_de_rede(cenario)
        r = M.resolver(
            cenario, variante="aj_estimado", metrica=metrica,
            beta=BETA_SEM_RENDA, alpha=ALPHA_PRINCIPAL, verboso=False, unidade=unidade,
        )
        r["grupo"] = "equidade_beta0"
        resultados.append(r)
        print(f"   {cenario}: cobertura {r['cobertura_pct_municipio']:.1f}% com beta = 0")
    return resultados


def rodar_cross_check(unidade: str) -> list[dict]:
    M.subtitulo("3. Cross-check CBC x HiGHS")
    resultados = []
    for cenario in CENARIOS_CROSS_CHECK:
        metrica = M.metrica_de_rede(cenario)
        for solver in SOLVERES:
            r = M.resolver(
                cenario, variante="aj_estimado", metrica=metrica,
                beta=BETA_PRINCIPAL, alpha=ALPHA_PRINCIPAL,
                solver_nome=solver, verboso=False, unidade=unidade,
            )
            r["grupo"] = f"cross_check_{cenario}"
            resultados.append(r)
    return resultados


# =============================================================================
# CONSOLIDACAO
# =============================================================================


def tabela_indicadores(resultados: list[dict]) -> pd.DataFrame:
    colunas = [
        "unidade_demanda", "grupo", "cenario", "p", "r_m", "metrica", "variante", "beta", "alpha",
        "n_demanda", "n_pares_podados", "n_demanda_elegiveis", "n_candidatos_elegiveis",
        "Z1", "Z2", "demanda_total", "teto_Z1",
        "cobertura_pct_municipio", "cobertura_pct_teto",
        "dist_media_ponderada_m", "dist_media_generalizada_m",
        "n_pontos_sem_demanda", "aj_medio_abertos", "aj_medio_elegiveis",
        "aj_medio_conjunto", "y_fracionarios",
        "n_binarias", "n_continuas", "n_restricoes_e1", "n_restricoes_e2",
        "status_e1", "status_e2", "gap_relativo_exigido",
        "folga_cobertura_usada", "folga_cobertura_permitida",
        "tempo_total_s", "solver",
        # So o cenario misto preenche as colunas por modo de acesso.
        "n_pares_walk", "n_pares_drive",
        "dist_media_ponderada_walk_m", "dist_media_ponderada_drive_m",
        "demanda_coberta_walk", "demanda_coberta_drive",
    ]
    linhas = [{c: r.get(c) for c in colunas} for r in resultados]
    return pd.DataFrame(linhas)


def tabela_por_regiao(resultados: list[dict]) -> pd.DataFrame:
    linhas = []
    for r in resultados:
        if r["grupo"] != "matriz_principal":
            continue
        for regiao, linha in r["_por_regiao"].iterrows():
            linhas.append(
                {
                    "cenario": r["cenario"], "p": r["p"], "r_m": r["r_m"],
                    "variante": r["variante"], "metrica": r["metrica"],
                    "regiao_funcional": regiao,
                    "demanda": round(float(linha["demanda"]), 3),
                    "coberta": round(float(linha["coberta"]), 3),
                    "cobertura_pct": round(float(linha["cobertura_pct"]), 2),
                }
            )
    return pd.DataFrame(linhas)


def tabela_por_bairro(resultados: list[dict]) -> pd.DataFrame:
    """Cobertura por bairro em cada execucao — insumo dos mapas da Etapa 9."""
    partes = []
    for r in resultados:
        if r["grupo"] != "matriz_principal":
            continue
        parte = r["_por_bairro"].copy()
        parte.insert(0, "cenario", r["cenario"])
        parte.insert(1, "p", r["p"])
        parte.insert(2, "r_m", r["r_m"])
        parte.insert(3, "variante", r["variante"])
        parte.insert(4, "metrica", r["metrica"])
        partes.append(parte)
    tabela = pd.concat(partes, ignore_index=True)
    for coluna in ["H", "COBERTA", "cobertura_pct"]:
        tabela[coluna] = tabela[coluna].round(4)
    return tabela


def tabela_por_setor(resultados: list[dict]) -> pd.DataFrame:
    """Cobertura por setor em cada execucao — so existe na rodada por setor."""
    partes = []
    for r in resultados:
        if r["grupo"] != "matriz_principal":
            continue
        parte = r["_por_unidade"].copy()
        parte.insert(0, "cenario", r["cenario"])
        parte.insert(1, "variante", r["variante"])
        parte.insert(2, "metrica", r["metrica"])
        partes.append(parte)
    tabela = pd.concat(partes, ignore_index=True)
    for coluna in ["H", "COBERTA", "cobertura_pct"]:
        tabela[coluna] = tabela[coluna].round(4)
    return tabela


def tabela_por_tipo(resultados: list[dict]) -> pd.DataFrame:
    linhas = []
    for r in resultados:
        if r["grupo"] != "matriz_principal":
            continue
        abertos = r["_candidatos"].loc[r["_abertos"]]
        for tipo, n in abertos["TIPO"].value_counts().items():
            linhas.append(
                {
                    "cenario": r["cenario"], "p": r["p"], "r_m": r["r_m"],
                    "variante": r["variante"], "metrica": r["metrica"],
                    "tipo": tipo, "pontos": int(n),
                    "pct_dos_abertos": round(100 * int(n) / r["p"], 1),
                }
            )
    return pd.DataFrame(linhas)


def tabela_efeito_atratividade(resultados: list[dict]) -> pd.DataFrame:
    """Pares elegiveis x pontos que mudam de local entre a_j = 1 e a_j estimado."""
    M.subtitulo("4. Efeito da atratividade — pares elegiveis x pontos que mudam")

    por_chave = {
        (r["cenario"], r["metrica"], r["variante"]): r
        for r in resultados if r["grupo"] == "matriz_principal"
    }
    linhas = []
    for cenario in M.CENARIOS:
        for metrica in metricas_do_cenario(cenario):
            a = por_chave[(cenario, metrica, "aj_estimado")]
            u = por_chave[(cenario, metrica, "aj_unitario")]
            conj_a, conj_u = set(a["_abertos"]), set(u["_abertos"])
            mudam = len(conj_a - conj_u)
            linhas.append(
                {
                    "cenario": cenario, "p": a["p"], "r_m": a["r_m"], "metrica": metrica,
                    "pares_elegiveis": a["n_pares_podados"],
                    "candidatos_elegiveis": a["n_candidatos_elegiveis"],
                    "pontos_que_mudam": mudam,
                    "fracao_de_p": round(mudam / a["p"], 4),
                    "Z1_igual": abs(a["Z1"] - u["Z1"]) < 1e-6,
                    "Z2_aj_estimado": round(a["Z2"], 2),
                    "Z2_aj_unitario": round(u["Z2"], 2),
                    "dist_aj_estimado_m": round(a["dist_media_ponderada_m"], 1),
                    "dist_aj_unitario_m": round(u["dist_media_ponderada_m"], 1),
                    "aj_medio_estimado": round(a["aj_medio_abertos"], 4),
                    "aj_medio_unitario": round(u["aj_medio_abertos"], 4),
                }
            )
    tabela = pd.DataFrame(linhas)

    for metrica_rotulo, sub in tabela.groupby("metrica"):
        print(f"\n   === metrica {metrica_rotulo} ===")
        print(f"   {'cenario':<8} {'p':>4} {'r':>6} {'pares':>7} {'mudam':>7} "
              f"{'% de p':>8} {'dist a_j=1':>11} {'dist a_j est':>13}")
        for linha in sub.sort_values("pares_elegiveis").itertuples():
            print(f"   {linha.cenario:<8} {linha.p:>4} {linha.r_m:>5}m "
                  f"{linha.pares_elegiveis:>7} {linha.pontos_que_mudam:>7} "
                  f"{100 * linha.fracao_de_p:>7.1f}% {linha.dist_aj_unitario_m:>10.0f} m "
                  f"{linha.dist_aj_estimado_m:>11.0f} m")

    print("\n   Duas series separam os efeitos, que os cenarios confundem:")
    print("   C1-C2-C3: pares fixos, p varia  -> isola o efeito de p")
    print("   C2-C4-C5: p fixo, pares variam  -> isola o efeito dos pares elegiveis")
    checar_z1 = tabela["Z1_igual"].all()
    print(f"\n   Z1 identico entre as duas ponderacoes em todas as execucoes: "
          f"{'sim' if checar_z1 else 'NAO'}")
    M.checar(
        checar_z1,
        "Z1 difere entre a_j = 1 e a_j estimado. Por construcao do modelo em dois "
        "estagios a cobertura tem de ser identica — se difere, a atratividade "
        "vazou para o Estagio 1.",
    )
    return tabela


def tabela_sensibilidade(resultados: list[dict]) -> pd.DataFrame:
    """Comprova que alpha nao altera o conjunto otimo, e que beta altera."""
    M.subtitulo("5. Sensibilidade alpha x beta — o conjunto otimo muda?")

    sens = [r for r in resultados if r["grupo"] == "sensibilidade_alpha_beta"]
    linhas = []
    for r in sens:
        linhas.append(
            {
                "cenario": r["cenario"], "beta": r["beta"], "alpha": r["alpha"],
                "demanda_total": round(r["demanda_total"], 3),
                "Z1": round(r["Z1"], 4), "Z2": round(r["Z2"], 2),
                "cobertura_pct_municipio": round(r["cobertura_pct_municipio"], 3),
                "dist_media_ponderada_m": round(r["dist_media_ponderada_m"], 1),
                "conjunto": ",".join(str(j) for j in sorted(r["_abertos"])),
            }
        )
    tabela = pd.DataFrame(linhas)

    print(f"   {'beta':>5} {'alpha':<14} {'demanda':>10} {'Z1':>10} "
          f"{'cobertura':>10} {'dist m':>8}")
    for linha in tabela.itertuples():
        print(f"   {linha.beta:>5g} {linha.alpha:<14} {linha.demanda_total:>10.1f} "
              f"{linha.Z1:>10.1f} {linha.cobertura_pct_municipio:>9.2f}% "
              f"{linha.dist_media_ponderada_m:>8.0f}")

    print(f"\n   conjunto otimo por beta:")
    todos_iguais = True
    for beta, sub in tabela.groupby("beta"):
        conjuntos = set(sub["conjunto"])
        igual = len(conjuntos) == 1
        todos_iguais &= igual
        print(f"      beta = {beta:g}: os {len(sub)} valores de alpha dao "
              f"{'O MESMO conjunto' if igual else f'{len(conjuntos)} conjuntos DISTINTOS'}")
    tabela["alpha_invariante"] = todos_iguais

    conjuntos_por_beta = tabela.groupby("beta")["conjunto"].first()
    distintos = conjuntos_por_beta.nunique()
    print(f"   entre os betas: {distintos} conjunto(s) distinto(s) de {len(conjuntos_por_beta)}")
    for b1 in conjuntos_por_beta.index:
        for b2 in conjuntos_por_beta.index:
            if b1 < b2:
                a, b = set(conjuntos_por_beta[b1].split(",")), set(conjuntos_por_beta[b2].split(","))
                print(f"      beta {b1:g} x beta {b2:g}: {len(a - b)} ponto(s) mudam de local")
    return tabela


def tabela_cross_check(resultados: list[dict]) -> pd.DataFrame:
    M.subtitulo("6. Cross-check CBC x HiGHS")
    linhas = []
    for cenario in CENARIOS_CROSS_CHECK:
        por_solver = {
            r["solver"].split()[0]: r
            for r in resultados if r["grupo"] == f"cross_check_{cenario}"
        }
        cbc, highs = por_solver["CBC"], por_solver["HiGHS"]
        conj_cbc, conj_highs = set(cbc["_abertos"]), set(highs["_abertos"])
        linhas.append(
            {
                "cenario": cenario,
                "Z1_CBC": round(cbc["Z1"], 6), "Z1_HiGHS": round(highs["Z1"], 6),
                "dif_Z1": abs(cbc["Z1"] - highs["Z1"]),
                "Z2_CBC": round(cbc["Z2"], 6), "Z2_HiGHS": round(highs["Z2"], 6),
                "dif_Z2_relativa": abs(cbc["Z2"] - highs["Z2"]) / max(abs(cbc["Z2"]), 1e-9),
                "pontos_identicos": len(conj_cbc & conj_highs),
                "p": cbc["p"],
                "tempo_CBC_s": round(cbc["tempo_total_s"], 3),
                "tempo_HiGHS_s": round(highs["tempo_total_s"], 3),
                "solver_CBC": cbc["solver"], "solver_HiGHS": highs["solver"],
            }
        )
        print(f"   {cenario}: Z1 difere em {abs(cbc['Z1'] - highs['Z1']):.2e} · "
              f"Z2 em {abs(cbc['Z2'] - highs['Z2']):.2e} · "
              f"{len(conj_cbc & conj_highs)} de {cbc['p']} pontos identicos · "
              f"CBC {cbc['tempo_total_s']:.2f}s / HiGHS {highs['tempo_total_s']:.2f}s")
        M.checar(
            abs(cbc["Z1"] - highs["Z1"]) < 1e-6,
            f"{cenario}: Z1 diverge entre CBC e HiGHS.",
        )
    return pd.DataFrame(linhas)


def faixas_de_renda(unidade: str) -> pd.DataFrame:
    """Faixa de renda de cada ponto de demanda, em quintis POPULACIONAIS.

    Ordena os pontos pela renda media e corta a populacao acumulada em partes
    iguais. Quintil por numero de setores misturaria setores de 200 e de 2.000
    habitantes; por populacao, cada faixa tem o mesmo peso demografico.
    """
    cfg = M.UNIDADES[unidade]
    demanda = pd.read_csv(cfg["demanda"], sep=";", decimal=",", dtype={"CD_BAIRRO": str, "CD_SETOR": str})
    base = demanda[[cfg["chave"], "POP", "RENDA_MEDIA"]].sort_values("RENDA_MEDIA").reset_index(drop=True)
    acumulada = base["POP"].cumsum() - base["POP"] / 2
    base["FAIXA_RENDA"] = (acumulada / base["POP"].sum() * N_FAIXAS_RENDA).astype(int).clip(0, N_FAIXAS_RENDA - 1) + 1
    return base


def tabela_equidade(resultados: list[dict], unidade: str) -> pd.DataFrame:
    """Populacao coberta por faixa de renda, em cada execucao de rede com a_j estimado."""
    M.subtitulo("7b. Equidade — cobertura da populacao por faixa de renda")

    chave = M.UNIDADES[unidade]["chave"]
    faixas = faixas_de_renda(unidade)
    linhas = []
    for r in resultados:
        if r["metrica"] == "euclid" or r["variante"] != "aj_estimado" or r["grupo"].startswith("cross_check"):
            continue
        cobertura = r["_por_unidade"][[chave, "cobertura_pct"]]
        base = faixas.merge(cobertura, on=chave, how="left")
        M.checar(base["cobertura_pct"].notna().all(), "Ponto de demanda sem cobertura na analise de equidade.")
        base["POP_COBERTA"] = base["POP"] * base["cobertura_pct"] / 100
        for faixa, sub in base.groupby("FAIXA_RENDA"):
            linhas.append(
                {
                    "grupo": r["grupo"], "cenario": r["cenario"], "metrica": r["metrica"],
                    "beta": r["beta"], "alpha": r["alpha"], "faixa_renda": int(faixa),
                    "renda_min": round(float(sub["RENDA_MEDIA"].min()), 2),
                    "renda_max": round(float(sub["RENDA_MEDIA"].max()), 2),
                    "pontos": len(sub), "pop": int(sub["POP"].sum()),
                    "pop_coberta": round(float(sub["POP_COBERTA"].sum()), 1),
                    "cobertura_pop_pct": round(100 * float(sub["POP_COBERTA"].sum()) / float(sub["POP"].sum()), 2),
                }
            )
    tabela = pd.DataFrame(linhas)

    # Resumo: faixa mais pobre (1) x mais rica (5), com beta = 0,3 e beta = 0.
    alvo = tabela[tabela["alpha"] == ALPHA_PRINCIPAL]
    print(f"   {'cenario':<8} {'beta':>5} {'faixa 1':>9} {'faixa 5':>9} {'razao 5/1':>10}")
    for (cenario, beta), sub in alvo.groupby(["cenario", "beta"]):
        if beta not in (BETA_SEM_RENDA, BETA_PRINCIPAL):
            continue
        # A matriz principal e a sensibilidade repetem C2 com beta = 0,3: basta uma.
        sub = sub.drop_duplicates("faixa_renda")
        f1 = float(sub.loc[sub["faixa_renda"] == 1, "cobertura_pop_pct"].iloc[0])
        f5 = float(sub.loc[sub["faixa_renda"] == N_FAIXAS_RENDA, "cobertura_pop_pct"].iloc[0])
        razao = f5 / f1 if f1 > 0 else float("inf")
        print(f"   {cenario:<8} {beta:>5g} {f1:>8.1f}% {f5:>8.1f}% {razao:>9.2f}x")
    print("\n   Com beta = 0 a demanda e proporcional a populacao: o gradiente que resta")
    print("   entre as faixas vem da localizacao do comercio e da densidade, nao do peso")
    print("   dado a renda. A diferenca para beta = 0,3 e o que o beta acrescenta.")
    return tabela


def analisar_aj(resultados: list[dict]) -> None:
    """O a_j dos abertos contra o dos ELEGIVEIS, que e a comparacao pertinente."""
    M.subtitulo("7. Atratividade dos pontos abertos")

    print(f"   {'cenario':<8} {'variante':<14} {'metrica':<8} {'abertos':>9} "
          f"{'elegiveis':>10} {'conjunto':>9} {'abertos-eleg':>13}")
    abaixo = 0
    for r in resultados:
        if r["grupo"] != "matriz_principal":
            continue
        delta = r["aj_medio_abertos"] - r["aj_medio_elegiveis"]
        if r["variante"] == "aj_estimado" and delta < 0:
            abaixo += 1
        print(f"   {r['cenario']:<8} {r['variante']:<14} {r['metrica']:<8} "
              f"{r['aj_medio_abertos']:>9.4f} {r['aj_medio_elegiveis']:>10.4f} "
              f"{r['aj_medio_conjunto']:>9.4f} {delta:>+13.4f}")
    n_estimado = sum(1 for r in resultados if r["grupo"] == "matriz_principal" and r["variante"] == "aj_estimado")
    print(f"\n   execucoes com a_j estimado cujos abertos ficam ABAIXO dos elegiveis: "
          f"{abaixo} de {n_estimado}")
    print("   O Estagio 1 ignora a atratividade e seleciona por cobertura; o Estagio 2")
    print("   so desempata entre solucoes de cobertura identica. Por isso o NIVEL do a_j")
    print("   e ditado pela estrutura de cobertura, nao pelo indice.")

    # A comparacao que isola o efeito do indice nao e contra os elegiveis, e sim
    # entre as DUAS PONDERACOES do mesmo cenario e metrica: e a unica em que todo
    # o resto permanece constante.
    print("\n   efeito isolado do indice — mesmo cenario e metrica, a_j estimado x a_j = 1:")
    print(f"   {'cenario':<8} {'metrica':<8} {'a_j = 1':>9} {'a_j est':>9} {'ganho':>9}")
    por_chave = {
        (r["cenario"], r["metrica"], r["variante"]): r
        for r in resultados if r["grupo"] == "matriz_principal"
    }
    ganhos = []
    for (cenario, metrica, variante), r in sorted(por_chave.items()):
        if variante != "aj_estimado":
            continue
        u = por_chave[(cenario, metrica, "aj_unitario")]
        ganho = r["aj_medio_abertos"] - u["aj_medio_abertos"]
        ganhos.append(ganho)
        print(f"   {cenario:<8} {metrica:<8} {u['aj_medio_abertos']:>9.4f} "
              f"{r['aj_medio_abertos']:>9.4f} {ganho:>+9.4f}")
    positivos = sum(1 for g in ganhos if g > 1e-9)
    print(f"\n   ganho positivo em {positivos} de {len(ganhos)} pares de execucoes "
          f"(media {sum(ganhos) / len(ganhos):+.4f})")
    print("   O indice ELEVA a atratividade media dos pontos escolhidos em relacao a")
    print("   variante sem ponderacao. Que o nivel absoluto siga abaixo da media dos")
    print("   elegiveis mede o quanto a cobertura restringe a escolha, nao falha do indice.")


def salvar_solucoes(resultados: list[dict], sufixo: str) -> None:
    M.subtitulo("10. Gravacao das solucoes")
    partes = []
    for r in resultados:
        if r["grupo"] != "matriz_principal":
            continue
        abertos = r["_candidatos"].loc[r["_abertos"]].copy()
        carga = r["_atribuicao"].groupby("ID_CANDIDATO")["DEMANDA_COBERTA"].sum()
        abertos["DEMANDA_ATENDIDA"] = carga.reindex(abertos.index).fillna(0.0).round(4)
        abertos["CENARIO"] = r["cenario"]
        abertos["P"] = r["p"]
        abertos["R_M"] = r["r_m"]
        abertos["VARIANTE"] = r["variante"]
        abertos["METRICA"] = r["metrica"]
        abertos["EXECUCAO"] = f"{r['cenario']}_{r['variante']}_{r['metrica']}"
        partes.append(abertos)

    solucoes = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=partes[0].crs)
    colunas = [
        "EXECUCAO", "CENARIO", "P", "R_M", "VARIANTE", "METRICA",
        "OSM_ID", "FONTE", "TIPO", "NOME", "NM_BAIRRO", "REGIAO_FUNCIONAL",
        "AJ", "DEMANDA_ATENDIDA", "geometry",
    ]
    solucoes = solucoes[[c for c in colunas if c in solucoes.columns]]
    for coluna in solucoes.columns:
        if solucoes[coluna].dtype == "string":
            solucoes[coluna] = solucoes[coluna].astype(object).where(
                solucoes[coluna].notna(), None
            )
    arquivo = ARQ_SOLUCOES.with_name(f"{ARQ_SOLUCOES.stem}{sufixo}.gpkg")
    arquivo.unlink(missing_ok=True)
    solucoes.to_file(arquivo, layer="solucoes", driver="GPKG")
    print(f"   {arquivo.relative_to(RAIZ).as_posix():<48} "
          f"{len(solucoes)} pontos de {solucoes['EXECUCAO'].nunique()} execucoes")


def gravar(nome: str, tabela: pd.DataFrame) -> None:
    """Grava a tabela em results/tabelas; o nome ja traz o sufixo da unidade."""
    caminho = DIR_TABELAS / f"{nome}.csv"
    tabela.to_csv(caminho, index=False, sep=";", decimal=",")
    print(f"   {caminho.relative_to(RAIZ).as_posix():<48} {len(tabela):>4} linhas")


def registrar_log(resultados: list[dict], unidade: str) -> None:
    """Grava a rodada no log, substituindo so as linhas da mesma unidade.

    Linhas de um log anterior a existencia da coluna unidade_demanda sao da rodada
    por bairro, que era a unica.
    """
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    novas = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in resultados])
    mantidas = pd.DataFrame()
    if ARQ_LOG.exists():
        anterior = pd.read_csv(ARQ_LOG, sep=";", decimal=",")
        if "unidade_demanda" not in anterior.columns:
            anterior["unidade_demanda"] = "bairro"
        mantidas = anterior[anterior["unidade_demanda"] != unidade] if LIMPAR_LOG else anterior
        # Coluna que a versao atual do modelo nao grava mais (ex.: n_bairros_elegiveis,
        # renomeada para n_demanda_elegiveis) sobraria vazia no log: fica so o que a
        # rodada atual tambem grava.
        mantidas = mantidas[[c for c in mantidas.columns if c in novas.columns]]
    log = pd.concat([mantidas, novas], ignore_index=True)
    colunas = ["unidade_demanda"] + [c for c in log.columns if c != "unidade_demanda"]
    log[colunas].to_csv(ARQ_LOG, index=False, sep=";", decimal=",")
    print(f"   {ARQ_LOG.relative_to(RAIZ).as_posix():<48} {len(novas):>4} execucoes desta rodada "
          f"({len(mantidas)} de outra unidade mantidas)")


def resumo_por_regiao(resultados: list[dict]) -> None:
    """Cobertura por regiao nas dez execucoes de rede, lado a lado."""
    M.subtitulo("8. Cobertura por regiao funcional — todas as execucoes de rede")

    principais = [
        r for r in resultados
        if r["grupo"] == "matriz_principal" and r["metrica"] != "euclid"
    ]
    print(f"   {'regiao':<14}" + "".join(
        f"{r['cenario'] + '/' + ('est' if r['variante'] == 'aj_estimado' else 'un'):>11}"
        for r in principais
    ))
    for regiao in ORDEM_REGIOES:
        texto = f"   {regiao:<14}"
        for r in principais:
            valor = r["_por_regiao"]["cobertura_pct"].get(regiao, float("nan"))
            texto += f"{valor:>10.1f}%"
        print(texto)

    print("\n   minimo e maximo por regiao, entre as dez execucoes de rede:")
    for regiao in ORDEM_REGIOES:
        valores = [r["_por_regiao"]["cobertura_pct"].get(regiao, float("nan"))
                   for r in principais]
        print(f"      {regiao:<14} min {min(valores):>5.1f}%  max {max(valores):>5.1f}%")


# =============================================================================
# EXECUCAO
# =============================================================================


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Matriz de cenarios do modelo em dois estagios.")
    parser.add_argument("--unidade", choices=sorted(M.UNIDADES), default=M.UNIDADE_PADRAO,
                        help="unidade de demanda: setor censitario (padrao) ou bairro")
    parser.add_argument("--saida", default=None,
                        help="versao da rodada (ex.: v36): le os candidatos de data/tratados/<saida>/ "
                             "e grava em results/<saida>/ e data/tratados/<saida>/, sem tocar "
                             "nos resultados vigentes")
    argumentos = parser.parse_args()
    unidade = argumentos.unidade
    sufixo = M.UNIDADES[unidade]["sufixo"]
    global DIR_TABELAS, DIR_LOGS, ARQ_LOG, ARQ_SOLUCOES
    base = DIR_RESULTADOS[unidade]
    if argumentos.saida:
        # results/ -> results/<saida>/ ; results/nao_usados -> results/<saida>/nao_usados
        base = RAIZ / "results" / argumentos.saida / base.relative_to(RAIZ / "results")
        ARQ_SOLUCOES = DIR_TRATADOS / argumentos.saida / ARQ_SOLUCOES.name
        ARQ_SOLUCOES.parent.mkdir(parents=True, exist_ok=True)
        M.ARQ_CANDIDATOS = DIR_TRATADOS / argumentos.saida / M.ARQ_CANDIDATOS.name
        M.checar(M.ARQ_CANDIDATOS.exists(),
                 f"{M.ARQ_CANDIDATOS} nao existe. Rodar antes: python src/04_atratividade.py "
                 f"--saida {argumentos.saida}")
    DIR_TABELAS = base / "tabelas"
    DIR_LOGS = base / "logs"
    ARQ_LOG = DIR_LOGS / "execucoes.csv"

    n_principal = sum(len(M.VARIANTES) * len(metricas_do_cenario(c)) for c in M.CENARIOS)
    inicio = time.perf_counter()
    M.titulo(
        f"07_cenarios.py — matriz de cenarios · unidade de demanda: {unidade}\n"
        f"{len(M.CENARIOS)} cenarios x 2 ponderacoes x metricas = {n_principal} execucoes "
        f"na matriz principal\n"
        f"alpha = {ALPHA_PRINCIPAL} · beta = {BETA_PRINCIPAL:g} · "
        f"gapRel exigido = {M.GAP_RELATIVO_EXIGIDO}\n"
        f"candidatos: {M.ARQ_CANDIDATOS.relative_to(RAIZ).as_posix()} · "
        f"saidas: {DIR_TABELAS.parent.relative_to(RAIZ).as_posix()}/"
    )
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)

    principais = rodar_matriz_principal(unidade)
    sensibilidade = rodar_sensibilidade(unidade)
    beta_zero = rodar_beta_zero(unidade)
    cross = rodar_cross_check(unidade)
    todos = principais + sensibilidade + beta_zero + cross

    efeito = tabela_efeito_atratividade(principais)
    sens = tabela_sensibilidade(sensibilidade)
    cross_tab = tabela_cross_check(cross)
    analisar_aj(principais)
    equidade = tabela_equidade(principais + sensibilidade + beta_zero, unidade)
    resumo_por_regiao(principais)

    M.subtitulo("9. Gravacao das tabelas")
    gravar(f"indicadores_cenarios{sufixo}", tabela_indicadores(todos))
    gravar(f"cobertura_por_regiao_cenarios{sufixo}", tabela_por_regiao(principais))
    gravar(f"composicao_por_tipo_cenarios{sufixo}", tabela_por_tipo(principais))
    gravar(f"cobertura_por_bairro_cenarios{sufixo}", tabela_por_bairro(principais))
    if unidade != "bairro":
        gravar(f"cobertura_por_{unidade}_cenarios{sufixo}", tabela_por_setor(principais))
    gravar(f"efeito_atratividade{sufixo}", efeito)
    gravar(f"sensibilidade_alpha_beta{sufixo}", sens)
    gravar(f"cross_check_solvers{sufixo}", cross_tab)
    gravar(f"equidade_renda{sufixo}", equidade)
    registrar_log(todos, unidade)
    salvar_solucoes(principais, sufixo)

    # Tempo de parede da rodada inteira — o numero que o texto cita ("N execucoes
    # em M minutos"). A soma dos tempos de solver, menor, fica ao lado.
    tempo_total = time.perf_counter() - inicio
    gravar(f"resumo_rodada{sufixo}", pd.DataFrame([{
        "execucoes": len(todos), "matriz_principal": len(principais),
        "sensibilidade_alpha_beta": len(sensibilidade), "equidade_beta0": len(beta_zero),
        "cross_check": len(cross),
        "tempo_parede_s": round(tempo_total, 1),
        "tempo_solver_s": round(sum(r["tempo_total_s"] for r in todos), 1),
        "todas_otimas": all(r["status_e1"] == "Optimal" and r["status_e2"] == "Optimal" for r in todos),
        "candidatos": M.ARQ_CANDIDATOS.relative_to(RAIZ).as_posix(),
    }]))

    M.titulo(
        f"Concluido em {tempo_total:.1f} s · "
        f"{len(todos)} execucoes ({len(principais)} na matriz principal)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except M.FalhaDeSanidade as erro:
        print()
        print("!" * 78)
        print("VERIFICACAO DE SANIDADE FALHOU")
        print("!" * 78)
        print(erro)
        sys.exit(1)
