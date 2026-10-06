"""
09_curva_p.py — curva de cobertura em funcao de p (so o Estagio 1).

Justifica os valores de p dos cenarios (10, 20 e 40) com o proprio modelo: resolve
o Estagio 1 — maxima demanda coberta — para p de 5 a 60, em passos de 5, e mostra
como a cobertura cresce e quanto cada ponto adicional acrescenta.

Nao ha Estagio 2 aqui. A pergunta e quanto cobrir com p pontos, e isso o Estagio 1
responde sozinho; a atratividade entra so no Estagio 2 e nao altera Z1*.

A instancia NAO e refeita: demanda, candidatos, matrizes de distancia, poda por r,
restricoes (2'), (3), (4) e solver vem de src/06_modelo.py, por importacao, como em
07_cenarios.py. Beta = 0,3 e alpha de referencia, os mesmos da matriz principal.

Solver: CBC com gap relativo ZERO e o limite de tempo de 06_modelo.py. Um p so e
registrado como otimo se o CBC comprovar a otimalidade. O PuLP devolve status
"Optimal" tambem quando o CBC para no limite de tempo com uma solucao apenas viavel;
por isso confere-se tambem o sol_status. Se algum p nao fechar, ele sai marcado no
CSV e no resumo — nao ha heuristica implementada no projeto para substitui-lo.

Validacao: para os pares (p, r) que sao cenarios (C1 a C5), o Z1 obtido aqui tem de
reproduzir o Z1 do log da matriz principal. E o que garante que a curva e a mesma
formulacao dos cenarios, e nao uma parecida.

Estas execucoes NAO entram em results/logs/execucoes.csv: sao so o Estagio 1, e o
log registra execucoes lexicograficas completas (o texto cita 37). Os dados de
execucao de cada p — solver, gap, status, binarias, restricoes, tempo — ficam no
proprio curva_p.csv.

Saidas:
    results/tabelas/curva_p.csv
    results/figuras/curva_p.png          r = 800 m, rede caminhavel
    results/figuras/curva_p_raios.png    500 m caminhavel, 800 m caminhavel, 1.500 m viaria

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/09_curva_p.py                 # os tres raios
    python src/09_curva_p.py --raios 800     # so 800 m
    python src/09_curva_p.py --so-figuras    # so redesenha, a partir de curva_p.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

matplotlib.use("Agg")


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

RAIZ = Path(__file__).resolve().parents[1]
DIR_TABELAS = RAIZ / "results" / "tabelas"
DIR_FIGURAS = RAIZ / "results" / "figuras"
ARQ_INDICADORES = DIR_TABELAS / "indicadores_cenarios_setor.csv"

UNIDADE = "setor"
VALORES_P = list(range(5, 61, 5))
RAIOS = [800, 500, 1500]               # 800 m primeiro: e a curva principal
METRICA_POR_RAIO = M.METRICA_POR_RAIO  # a pe ate 800 m, viaria em 1.500 m

# Cenarios que caem sobre as curvas: destacados nas figuras e usados na validacao.
CENARIOS_NA_CURVA = {
    (cfg["r"], cfg["p"]): nome for nome, cfg in M.CENARIOS.items()
}

TOLERANCIA_VALIDACAO_Z1 = 1e-3   # encomendas/dia; o log grava Z1 com 4 casas
DPI = 300

ROTULO_METRICA = {"walk": "rede caminhável", "drive": "rede viária"}
COR_RAIO = {500: "#6a3d9a", 800: "#1f4e79", 1500: "#e69138"}
COR_DESTAQUE = "#c0392b"


def numero_br(valor: float, casas: int = 1) -> str:
    return M.numero_br(valor, casas)


# =============================================================================
# ESTAGIO 1
# =============================================================================


def preparar_instancia(r: int) -> dict:
    """Pares elegiveis a r, demanda e teto — a mesma poda de 06_modelo.resolver()."""
    metrica = METRICA_POR_RAIO[r]
    demanda, candidatos = M.carregar_dados(M.BETA_PADRAO, M.ALPHA_PADRAO, UNIDADE)
    matriz = M.carregar_matriz(metrica, len(candidatos), UNIDADE)
    chave = M.UNIDADES[UNIDADE]["chave"]

    # Poda por r ANTES de criar as variaveis, pela distancia FISICA — restricao (7).
    pares = matriz[matriz["DIST_M"] <= r].copy()
    h = demanda.set_index(chave)["H"]
    pares["H"] = pares[chave].map(h)
    M.checar(pares["H"].notna().all(), "Par sem demanda apos a juncao: conferir o tipo de CD_SETOR.")

    demanda_elegivel = sorted(set(pares[chave]))
    candidatos_elegiveis = sorted(set(pares["ID_CANDIDATO"]))
    chaves = list(zip(pares[chave], pares["ID_CANDIDATO"]))
    por_ponto: dict[str, list] = {}
    for i, j in chaves:
        por_ponto.setdefault(i, []).append(j)

    return {
        "r": r, "metrica": metrica, "chaves": chaves, "por_ponto": por_ponto,
        "peso": dict(zip(chaves, pares["H"])),
        "candidatos_elegiveis": candidatos_elegiveis,
        "demanda_total": float(h.sum()),
        # Setor sem candidato ao alcance nao e coberto por p algum.
        "teto": float(h.reindex(demanda_elegivel).sum()),
    }


def resolver_estagio1(inst: dict, p: int) -> dict:
    """Max SOMA h_i y_ij sob (2'), (3), (4) — o Estagio 1 de 06_modelo.py, com p livre."""
    M.checar(p <= len(inst["candidatos_elegiveis"]),
             f"p = {p} excede os {len(inst['candidatos_elegiveis'])} candidatos elegiveis a r = {inst['r']} m.")
    chaves, peso = inst["chaves"], inst["peso"]

    inicio = time.perf_counter()
    modelo = pulp.LpProblem(f"curva_p_r{inst['r']}_p{p}", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", inst["candidatos_elegiveis"], cat=pulp.LpBinary)
    # (6) y continua em [0,1]: com x binaria a solucao otima ja sai inteira.
    y = pulp.LpVariable.dicts("y", chaves, lowBound=0, upBound=1, cat=pulp.LpContinuous)
    # (2') SOMA_j y_ij <= 1 — cobertura parcial permitida.
    for i, js in inst["por_ponto"].items():
        modelo += pulp.lpSum(y[(i, j)] for j in js) <= 1, f"c2_{i}"
    # (3) y_ij <= x_j
    for i, j in chaves:
        modelo += y[(i, j)] <= x[j], f"c3_{i}_{j}"
    # (4) SOMA_j x_j = p
    modelo += pulp.lpSum(x[j] for j in inst["candidatos_elegiveis"]) == p, "c4_p"
    # Estagio 1: maximizar a demanda coberta (minimizacao do negativo, como em 06).
    modelo += -pulp.lpSum(peso[k] * y[k] for k in chaves)

    solver, versao = M.obter_solver("CBC")
    status = modelo.solve(solver)
    tempo = time.perf_counter() - inicio

    # "Optimal" do PuLP nao basta: com parada por tempo o CBC devolve uma solucao
    # viavel e o PuLP ainda diz Optimal. So sol_status = 1 e otimo comprovado.
    comprovado = (pulp.LpStatus[status] == "Optimal"
                  and modelo.sol_status == pulp.LpSolutionOptimal)
    z1 = float(sum(peso[k] * y[k].value() for k in chaves))
    abertos = [j for j in inst["candidatos_elegiveis"] if x[j].value() > 0.5]
    M.checar(len(abertos) == p, f"r = {inst['r']}, p = {p}: {len(abertos)} pontos abertos.")

    return {
        "raio_m": inst["r"], "metrica": inst["metrica"], "p": p,
        "Z1": z1, "demanda_total": inst["demanda_total"], "teto_Z1": inst["teto"],
        "cobertura_municipio_pct": 100 * z1 / inst["demanda_total"],
        "cobertura_teto_pct": 100 * z1 / inst["teto"],
        "metodo": ("CBC, otimo comprovado (gap relativo 0)" if comprovado
                   else f"CBC SEM prova de otimalidade ({pulp.LpStatus[status]}, sol_status {modelo.sol_status})"),
        "otimo_comprovado": comprovado,
        "tempo_s": tempo,
        "n_binarias": len(x), "n_continuas": len(y), "n_restricoes": len(modelo.constraints),
        "gap_relativo_exigido": M.GAP_RELATIVO_EXIGIDO, "limite_tempo_s": M.TEMPO_LIMITE_S,
        "solver": versao,
    }


def curva(r: int) -> pd.DataFrame:
    M.subtitulo(f"r = {r} m, {ROTULO_METRICA[METRICA_POR_RAIO[r]]}")
    inst = preparar_instancia(r)
    print(f"   pares elegiveis {M.numero_br(len(inst['chaves']))} · candidatos elegiveis "
          f"{len(inst['candidatos_elegiveis'])} · teto {numero_br(100 * inst['teto'] / inst['demanda_total'])}% da demanda")
    linhas = []
    for p in VALORES_P:
        linha = resolver_estagio1(inst, p)
        linhas.append(linha)
        marca = "" if linha["otimo_comprovado"] else "   <-- SEM PROVA DE OTIMALIDADE"
        print(f"   p = {p:>2}  cobertura {numero_br(linha['cobertura_municipio_pct'], 2):>6}%  "
              f"teto {numero_br(linha['cobertura_teto_pct'], 2):>6}%  {linha['tempo_s']:6.1f} s{marca}")
    tabela = pd.DataFrame(linhas)
    # Ganho marginal: pontos percentuais por ponto adicionado, no passo de 5. Para
    # p = 5 a referencia e p = 0, cobertura nula.
    anterior = tabela["cobertura_municipio_pct"].shift(1, fill_value=0.0)
    passo = tabela["p"].diff().fillna(tabela["p"])
    tabela["ganho_marginal_pp_por_ponto"] = (tabela["cobertura_municipio_pct"] - anterior) / passo
    return tabela


# =============================================================================
# VERIFICACOES
# =============================================================================


def verificar(tabela: pd.DataFrame) -> None:
    M.subtitulo("Verificacoes")
    # 1. Os cenarios que caem na curva reproduzem o Z1 do log da matriz principal.
    ind = pd.read_csv(ARQ_INDICADORES, sep=";", decimal=",")
    ind = ind[(ind["grupo"] == "matriz_principal") & (ind["variante"] == "aj_estimado")]
    for (r, p), nome in sorted(CENARIOS_NA_CURVA.items(), key=lambda kv: kv[1]):
        linha = tabela[(tabela["raio_m"] == r) & (tabela["p"] == p)]
        if linha.empty:
            continue
        z1_log = float(ind[(ind["cenario"] == nome) & (ind["metrica"] == METRICA_POR_RAIO[r])]["Z1"].iloc[0])
        z1_aqui = float(linha["Z1"].iloc[0])
        M.checar(abs(z1_aqui - z1_log) <= TOLERANCIA_VALIDACAO_Z1,
                 f"{nome}: Z1 = {z1_aqui:.4f} aqui, {z1_log:.4f} no log. A curva nao e a formulacao dos cenarios.")
        print(f"   {nome} (p = {p}, r = {r} m): Z1 {z1_aqui:.4f} = log {z1_log:.4f}")
    # 2. Mais pontos nunca cobrem menos: Z1 nao decresce em p.
    for r, sub in tabela.groupby("raio_m"):
        M.checar(bool((sub.sort_values("p")["Z1"].diff().dropna() >= -1e-6).all()),
                 f"r = {r} m: Z1 decresce em algum p.")
        M.checar(bool((sub["Z1"] <= sub["teto_Z1"] + 1e-6).all()), f"r = {r} m: Z1 acima do teto.")
    print("   Z1 nao decresce em p e nunca passa do teto, nos raios rodados")


# =============================================================================
# FIGURAS
# =============================================================================


def _virgula(valor: float, casas: int = 1) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def figura_800(tabela: pd.DataFrame) -> None:
    """Cobertura (em cima) e ganho marginal (embaixo) para r = 800 m caminhavel.

    Dois paineis com o mesmo eixo de p, e nao um grafico de dois eixos: cobertura e
    ganho marginal tem escalas diferentes.
    """
    sub = tabela[tabela["raio_m"] == 800].sort_values("p")
    teto = 100 * sub["teto_Z1"].iloc[0] / sub["demanda_total"].iloc[0]
    destaque = sub[sub["p"].isin([p for (r, p) in CENARIOS_NA_CURVA if r == 800])]

    figura, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax1.plot(sub["p"], sub["cobertura_municipio_pct"], marker="o", markersize=5,
             color=COR_RAIO[800], linewidth=2, label="Cobertura da demanda municipal")
    ax1.scatter(destaque["p"], destaque["cobertura_municipio_pct"], s=110, color=COR_DESTAQUE,
                edgecolor="black", linewidth=0.8, zorder=5, label="Cenários C1, C2 e C3")
    for _, linha in destaque.iterrows():
        nome = CENARIOS_NA_CURVA[(800, int(linha["p"]))]
        ax1.annotate(f"{nome} · {_virgula(linha['cobertura_municipio_pct'])}%",
                     (linha["p"], linha["cobertura_municipio_pct"]),
                     textcoords="offset points", xytext=(8, -14), ha="left", fontsize=8.5)
    ax1.axhline(teto, color="#555555", linestyle="--", linewidth=1)
    ax1.annotate(f"teto do raio: {_virgula(teto)}%", (sub["p"].min(), teto),
                 textcoords="offset points", xytext=(0, 4), fontsize=8, color="#444444")
    ax1.set_ylabel("Cobertura da demanda municipal (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower right", frameon=False, fontsize=8.5)

    cores = [COR_DESTAQUE if p in set(destaque["p"]) else COR_RAIO[800] for p in sub["p"]]
    ax2.bar(sub["p"], sub["ganho_marginal_pp_por_ponto"], width=3.2, color=cores, edgecolor="white")
    ax2.set_ylabel("Ganho por ponto\nadicionado (p.p.)")
    ax2.set_xlabel("Pontos abertos (p)")
    ax2.set_xticks(VALORES_P)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)
    _salvar(figura, "curva_p.png")


def figura_raios(tabela: pd.DataFrame) -> None:
    """As tres curvas sobrepostas, com o teto de cada raio e os cinco cenarios."""
    figura, ax = plt.subplots(figsize=(8.5, 5.6))
    for r in sorted(RAIOS):
        sub = tabela[tabela["raio_m"] == r].sort_values("p")
        if sub.empty:
            continue
        # Separador de milhar trocado SO no numero: a virgula do rotulo fica.
        raio_br = f"{r:,}".replace(",", ".")
        rotulo = f"r = {raio_br} m, {ROTULO_METRICA[METRICA_POR_RAIO[r]]}"
        ax.plot(sub["p"], sub["cobertura_municipio_pct"], marker="o", markersize=4.5,
                color=COR_RAIO[r], linewidth=2, label=rotulo)
        teto = 100 * sub["teto_Z1"].iloc[0] / sub["demanda_total"].iloc[0]
        ax.axhline(teto, color=COR_RAIO[r], linestyle="--", linewidth=0.9, alpha=0.8)
        # Rotulo do teto a esquerda: a direita, a curva de 1.500 m cruza o teto de 800 m.
        ax.annotate(f"teto {_virgula(teto)}%", (VALORES_P[0], teto), textcoords="offset points",
                    xytext=(0, 3), ha="left", fontsize=7.5, color=COR_RAIO[r])
        for (rr, p), nome in CENARIOS_NA_CURVA.items():
            if rr != r:
                continue
            linha = sub[sub["p"] == p]
            if linha.empty:
                continue
            valor = float(linha["cobertura_municipio_pct"].iloc[0])
            ax.scatter([p], [valor], s=90, color=COR_RAIO[r], edgecolor="black", linewidth=0.9, zorder=5)
            ax.annotate(nome, (p, valor), textcoords="offset points", xytext=(6, -12), fontsize=8.5)
    ax.set_xlabel("Pontos abertos (p)")
    ax.set_ylabel("Cobertura da demanda municipal (%)")
    ax.set_xticks(VALORES_P)
    ax.set_ylim(0, 105)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8.5)
    _salvar(figura, "curva_p_raios.png")


def _salvar(figura, arquivo: str) -> None:
    # Sem titulo dentro da imagem: titulo e fonte entram no Word (ABNT).
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.savefig(DIR_FIGURAS / arquivo, dpi=DPI, bbox_inches="tight")
    plt.close(figura)
    print(f"   {arquivo}")


# =============================================================================
# EXECUCAO
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Curva de cobertura em funcao de p (Estagio 1).")
    parser.add_argument("--raios", default=",".join(map(str, RAIOS)),
                        help="raios a rodar, separados por virgula (padrao: 800,500,1500)")
    parser.add_argument("--so-figuras", action="store_true",
                        help="redesenha as figuras a partir de curva_p.csv, sem resolver de novo")
    argumentos = parser.parse_args()
    if argumentos.so_figuras:
        arquivo = DIR_TABELAS / "curva_p.csv"
        M.checar(arquivo.exists(), f"{arquivo} nao existe. Rodar sem --so-figuras.")
        tabela = pd.read_csv(arquivo, sep=";", decimal=",")
        M.subtitulo("Figuras (a partir de curva_p.csv)")
        if 800 in set(tabela["raio_m"]):
            figura_800(tabela)
        if tabela["raio_m"].nunique() > 1:
            figura_raios(tabela)
        return 0
    raios = [int(r) for r in argumentos.raios.split(",")]
    M.checar(all(r in METRICA_POR_RAIO for r in raios), f"Raio sem metrica definida: {raios}.")

    inicio = time.perf_counter()
    M.titulo(f"09_curva_p.py — Estagio 1 para p = {VALORES_P[0]} a {VALORES_P[-1]} · "
             f"beta = {M.BETA_PADRAO:g}, alpha {M.ALPHA_PADRAO} · CBC, gap 0, limite {M.TEMPO_LIMITE_S} s")

    tabela = pd.concat([curva(r) for r in raios], ignore_index=True)
    verificar(tabela)

    colunas = [
        "raio_m", "metrica", "p", "cobertura_municipio_pct", "cobertura_teto_pct",
        "ganho_marginal_pp_por_ponto", "metodo", "tempo_s", "Z1", "demanda_total", "teto_Z1",
        "otimo_comprovado", "n_binarias", "n_continuas", "n_restricoes",
        "gap_relativo_exigido", "limite_tempo_s", "solver",
    ]
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)
    tabela[colunas].to_csv(DIR_TABELAS / "curva_p.csv", sep=";", decimal=",", index=False)

    M.subtitulo("Figuras")
    if 800 in raios:
        figura_800(tabela)
    if len(raios) > 1:
        figura_raios(tabela)

    M.titulo("Resumo")
    for r, sub in tabela.groupby("raio_m", sort=False):
        print(f"r = {r} m ({ROTULO_METRICA[METRICA_POR_RAIO[r]]})")
        for _, l in sub.iterrows():
            print(f"   p = {int(l['p']):>2}  {numero_br(l['cobertura_municipio_pct'], 2):>6}% do municipio  "
                  f"{numero_br(l['cobertura_teto_pct'], 2):>6}% do teto  ganho {numero_br(l['ganho_marginal_pp_por_ponto'], 2):>5} p.p./ponto  "
                  f"{l['tempo_s']:6.1f} s")
    sem_prova = tabela[~tabela["otimo_comprovado"]]
    print()
    if sem_prova.empty:
        print(f"Todos os {len(tabela)} valores de p com otimo comprovado pelo CBC (gap 0).")
    else:
        print("!" * 78)
        print(f"{len(sem_prova)} execucao(oes) SEM prova de otimalidade:")
        print(sem_prova[["raio_m", "p", "metodo", "tempo_s"]].to_string(index=False))
        print("!" * 78)
    print(f"\nConcluido em {time.perf_counter() - inicio:.1f} s")
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
