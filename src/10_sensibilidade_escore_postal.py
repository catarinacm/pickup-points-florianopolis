"""
10_sensibilidade_escore_postal.py — sensibilidade ao escore de tipo das agencias postais.

O escore de tipo das agencias postais e o unico que nao vem de Firmeza (2021): e
derivado da razao agencia/supermercado de Guarino Neto e Vidal Vieira (2023),
pesquisa de outra cidade. Este script mede o quanto a solucao depende dele.

Varia o escore em sete valores, sob as duas ordenacoes de pesos do indice — (A),
adotada, com o tipo em ultimo, e (B), com o tipo em primeiro — nos cenarios C2,
C4, C5 e C6. Sao 7 x 2 x 4 = 56 combinacoes.

Como o indice so entra no Estagio 2, o Z1* de cada cenario NAO depende do escore.
O Estagio 1 e resolvido uma vez por cenario, no caso-base, e o Z1* e reaproveitado
nas demais combinacoes, que resolvem so o Estagio 2. O Z1* do caso-base e
conferido contra o da rodada de 07_cenarios.py.

Caso-base: escore derivado (0,8323) e ordenacao (A). Para cada combinacao:
    - agencias postais selecionadas (quantas e quais);
    - pontos que mudam em relacao ao caso-base e indice de Jaccard;
    - a_j medio dos selecionados;
    - variacao da distancia media ponderada.

A formulacao e a de src/06_modelo.py e os escores e pesos sao os de
src/04_atratividade.py, ambos reusados por importacao.

Entradas:
    data/tratados/candidatos_com_aj.gpkg                 escores por atributo (Etapa 5)
    results/tabelas/indicadores_cenarios_setor.csv     Z1* de referencia (07)

Saidas (com --saida v36, em results/v36/):
    sensibilidade_escore_postal.csv           uma linha por combinacao
    quadro_sensibilidade_escore_postal.md     quadro para o texto

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/10_sensibilidade_escore_postal.py
    python src/10_sensibilidade_escore_postal.py --saida v36
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_TRATADOS = RAIZ / "data" / "tratados"


def _carregar(nome_arquivo: str, apelido: str):
    """Importa um script de src/ cujo nome comeca com digito e nao e importavel."""
    caminho = Path(__file__).with_name(nome_arquivo)
    spec = importlib.util.spec_from_file_location(apelido, caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


M = _carregar("06_modelo.py", "modelo")
A = _carregar("04_atratividade.py", "atratividade")

# =============================================================================
# PARAMETROS
# =============================================================================

# Do piso ao teto dos escores, passando pelo valor arbitrado ate a v35 (0,45) e
# pelo derivado (0,8323), que e calculado em 04_atratividade.py, nao digitado.
ESCORES = (0.20, 0.30, 0.45, 0.60, 0.70, A.ESCORE_POSTAL_DERIVADO, 1.00)
ORDENS = (A.ORDEM_PRINCIPAL, A.ORDEM_SENSIBILIDADE)
CENARIOS = ("C2", "C4", "C5", "C6")

ESCORE_BASE = A.ESCORE_POSTAL_DERIVADO
ORDEM_BASE = A.ORDEM_PRINCIPAL

TIPO_POSTAL = "agencia_postal"
UNIDADE = "setor"
BETA = M.BETA_PADRAO
ALPHA = M.ALPHA_PADRAO

COLUNA_ESCORE = {
    "seguranca": "S_SEGURANCA", "acessibilidade": "S_ACESSIBILIDADE",
    "disponibilidade": "S_DISPONIBILIDADE", "tipo": "S_TIPO",
}
ROTULO_ORDEM = {"A": "adotada (tipo em último)", "B": "alternativa (tipo em primeiro)"}

# =============================================================================
# INDICE
# =============================================================================


def indice(candidatos: gpd.GeoDataFrame, escore_postal: float, ordem: str) -> pd.Series:
    """a_j de todos os candidatos com o escore postal e a ordenacao informados.

    Parte dos escores por atributo ja gravados pela Etapa 5; so o escore de tipo
    das agencias postais e trocado. Os pesos sao os ROC de 04_atratividade.py.
    """
    pesos = A.pesos_da_ordem(ordem)
    escores = {atributo: candidatos[coluna].to_numpy(dtype=float).copy()
               for atributo, coluna in COLUNA_ESCORE.items()}
    escores["tipo"][(candidatos["TIPO"] == TIPO_POSTAL).to_numpy()] = escore_postal
    aj = np.zeros(len(candidatos))
    for atributo, peso in pesos.items():
        aj += peso * escores[atributo]
    return pd.Series(aj, index=candidatos.index)


def conferir_indice(candidatos: gpd.GeoDataFrame) -> None:
    """O indice recalculado aqui tem de reproduzir o que a Etapa 5 gravou."""
    escore_do_arquivo = float(candidatos.loc[candidatos["TIPO"] == TIPO_POSTAL, "S_TIPO"].iloc[0])
    for ordem, coluna in [(A.ORDEM_PRINCIPAL, "AJ"), (A.ORDEM_SENSIBILIDADE, f"AJ_ORDEM_{A.ORDEM_SENSIBILIDADE}")]:
        recalculado = indice(candidatos, escore_do_arquivo, ordem)
        diferenca = float((recalculado - candidatos[coluna]).abs().max())
        # Os escores sao gravados com 6 casas; a soma ponderada herda esse arredondamento.
        M.checar(diferenca < 5e-6,
                 f"O a_j recalculado na ordem ({ordem}) difere do gravado em ate {diferenca:.2e}.")
    print(f"   indice recalculado reproduz o do arquivo nas duas ordens (escore postal "
          f"gravado: {escore_do_arquivo:.4f})")
    M.checar(
        abs(escore_do_arquivo - ESCORE_BASE) < 1e-6,
        f"O arquivo de candidatos foi gerado com escore postal {escore_do_arquivo:.4f}, mas o "
        f"caso-base desta analise e {ESCORE_BASE:.4f}. Rodar 04_atratividade.py antes.",
    )


# =============================================================================
# EXECUCAO
# =============================================================================


def z1_de_referencia(dir_tabelas: Path, cenario: str) -> float:
    """Z1* do cenario na rodada de 07_cenarios.py, para a conferencia cruzada."""
    arquivo = dir_tabelas / "indicadores_cenarios_setor.csv"
    M.checar(arquivo.exists(), f"{arquivo} nao existe. Rodar antes src/07_cenarios.py.")
    ind = pd.read_csv(arquivo, sep=";", decimal=",")
    linha = ind[
        (ind["cenario"] == cenario) & (ind["grupo"] == "matriz_principal")
        & (ind["variante"] == "aj_estimado") & (ind["metrica"] == M.metrica_de_rede(cenario))
    ]
    M.checar(len(linha) == 1, f"{cenario}: Z1 de referencia nao encontrado em {arquivo.name}.")
    return float(linha["Z1"].iloc[0])


def descrever_postais(candidatos: gpd.GeoDataFrame, abertos: list[int]) -> tuple[int, str]:
    sub = candidatos.loc[abertos]
    sub = sub[sub["TIPO"] == TIPO_POSTAL]
    nomes = [f"{(linha.NOME or '(sem nome)')} [{linha.NM_BAIRRO}]" for linha in sub.itertuples()]
    return len(sub), " | ".join(sorted(nomes))


def rodar(candidatos: gpd.GeoDataFrame, dir_tabelas: Path) -> pd.DataFrame:
    linhas = []
    for cenario in CENARIOS:
        metrica = M.metrica_de_rede(cenario)
        M.subtitulo(f"{cenario} — metrica {metrica}")

        # Caso-base: os dois estagios. E daqui que sai o Z1* reaproveitado.
        base = M.resolver(
            cenario, variante="aj_estimado", metrica=metrica, beta=BETA, alpha=ALPHA,
            verboso=False, unidade=UNIDADE, aj_externo=indice(candidatos, ESCORE_BASE, ORDEM_BASE),
        )
        z1 = base["Z1"]
        z1_ref = z1_de_referencia(dir_tabelas, cenario)
        M.checar(abs(z1 - z1_ref) < 1e-6,
                 f"{cenario}: Z1* do caso-base ({z1:.6f}) difere do da rodada de cenarios ({z1_ref:.6f}).")
        conjunto_base = set(base["_abertos"])
        print(f"   Z1* = {z1:.4f}, identico ao da rodada de cenarios (diferenca "
              f"{abs(z1 - z1_ref):.1e}); reaproveitado nas demais combinacoes")
        print(f"   {'ordem':<6} {'escore':>7} {'postais':>8} {'mudam':>6} {'Jaccard':>8} "
              f"{'a_j medio':>10} {'dist m':>8} {'delta m':>8} {'tempo':>7}")

        for ordem in ORDENS:
            for escore in ESCORES:
                eh_base = ordem == ORDEM_BASE and np.isclose(escore, ESCORE_BASE)
                if eh_base:
                    r = base
                else:
                    r = M.resolver(
                        cenario, variante="aj_estimado", metrica=metrica, beta=BETA, alpha=ALPHA,
                        verboso=False, unidade=UNIDADE,
                        aj_externo=indice(candidatos, escore, ordem), z1_reaproveitado=z1,
                    )
                conjunto = set(r["_abertos"])
                n_postais, quais = descrever_postais(candidatos, r["_abertos"])
                mudam = len(conjunto - conjunto_base)
                jaccard = len(conjunto & conjunto_base) / len(conjunto | conjunto_base)
                linhas.append({
                    "cenario": cenario, "metrica": metrica, "p": r["p"],
                    "ordem_pesos": ordem, "escore_postal": round(float(escore), 4),
                    "caso_base": bool(eh_base),
                    "n_agencias_postais": n_postais, "agencias_postais": quais,
                    "pontos_que_mudam": mudam, "jaccard": round(jaccard, 4),
                    "aj_medio_selecionados": round(r["aj_medio_abertos"], 4),
                    "dist_media_ponderada_m": round(r["dist_media_ponderada_m"], 2),
                    "delta_dist_media_m": round(r["dist_media_ponderada_m"] - base["dist_media_ponderada_m"], 2),
                    "Z1": round(z1, 6), "Z1_no_estagio2": round(r["Z1_no_estagio2"], 6),
                    "Z2": round(r["Z2"], 4),
                    "estagio1": r["status_e1"], "status_e2": r["status_e2"],
                    "folga_cobertura_usada": r["folga_cobertura_usada"],
                    "folga_cobertura_permitida": r["folga_cobertura_permitida"],
                    "gap_relativo_exigido": r["gap_relativo_exigido"],
                    "tempo_s": round(r["tempo_total_s"], 2), "solver": r["solver"],
                    "conjunto": ",".join(str(j) for j in sorted(conjunto)),
                })
                print(f"   {ordem:<6} {escore:>7.4f} {n_postais:>8} {mudam:>6} {jaccard:>8.3f} "
                      f"{r['aj_medio_abertos']:>10.4f} {r['dist_media_ponderada_m']:>8.1f} "
                      f"{linhas[-1]['delta_dist_media_m']:>+8.1f} {r['tempo_total_s']:>6.1f}s"
                      f"{'  <- caso-base' if eh_base else ''}")
    return pd.DataFrame(linhas)


def verificar(tabela: pd.DataFrame) -> None:
    M.subtitulo("Verificacoes")
    M.checar(len(tabela) == len(CENARIOS) * len(ORDENS) * len(ESCORES),
             f"{len(tabela)} combinacoes, esperado {len(CENARIOS) * len(ORDENS) * len(ESCORES)}.")
    M.checar(bool((tabela["status_e2"] == "Optimal").all()), "Ha Estagio 2 sem otimalidade comprovada.")
    M.checar(int(tabela["caso_base"].sum()) == len(CENARIOS), "Caso-base ausente em algum cenario.")
    M.checar(bool((tabela["folga_cobertura_usada"] <= tabela["folga_cobertura_permitida"] + 1e-6).all()),
             "Ha combinacao que usou mais folga de cobertura do que a tolerancia permite.")
    base = tabela[tabela["caso_base"]]
    M.checar(bool((base["pontos_que_mudam"] == 0).all()) and bool((base["jaccard"] == 1.0).all()),
             "O caso-base difere de si mesmo.")
    print(f"   {len(tabela)} combinacoes · Estagio 2 otimo em todas (gapRel = "
          f"{M.GAP_RELATIVO_EXIGIDO}) · folga da tolerancia {M.TOLERANCIA_ESTAGIO2} respeitada")
    print(f"   Z1* identico ao da rodada de cenarios nos {len(CENARIOS)} cenarios")


def numero_br(valor: float, casas: int) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def quadro_markdown(tabela: pd.DataFrame) -> str:
    """Quadro para o texto: uma linha por escore, colunas por cenario e ordenacao."""
    linhas = [
        "# Sensibilidade ao escore de tipo das agências postais",
        "",
        f"Caso-base: escore {numero_br(ESCORE_BASE, 4)} e ordenação adotada (tipo em último). "
        f"p = 20 nos quatro cenários; β = {numero_br(BETA, 1)}, α de referência. "
        "Z1* idêntico em todas as combinações de cada cenário (só o segundo estágio é resolvido de novo).",
        "",
    ]
    blocos = [
        ("Agências postais selecionadas / pontos que mudam em relação ao caso-base",
         lambda l: f"{int(l.n_agencias_postais)} / {int(l.pontos_que_mudam)}"),
        ("Índice de Jaccard em relação ao caso-base",
         lambda l: numero_br(l.jaccard, 2)),
        ("Índice de atratividade médio dos pontos selecionados",
         lambda l: numero_br(l.aj_medio_selecionados, 3)),
        ("Variação da distância média ponderada em relação ao caso-base (m)",
         lambda l: ("0" if abs(l.delta_dist_media_m) < 0.05 else f"{l.delta_dist_media_m:+.1f}".replace(".", ","))),
    ]
    for titulo_bloco, formato in blocos:
        linhas.append(f"## {titulo_bloco}")
        linhas.append("")
        cabecalho = "| Escore |" + "".join(f" {c} adotada | {c} alternativa |" for c in CENARIOS)
        linhas.append(cabecalho)
        linhas.append("|---|" + "---|---|" * len(CENARIOS))
        for escore in ESCORES:
            texto = f"| {numero_br(escore, 4 if np.isclose(escore, ESCORE_BASE) else 2)} |"
            for cenario in CENARIOS:
                for ordem in ORDENS:
                    l = tabela[(tabela["cenario"] == cenario) & (tabela["ordem_pesos"] == ordem)
                               & np.isclose(tabela["escore_postal"], round(float(escore), 4))].iloc[0]
                    texto += f" {formato(l)} |"
            linhas.append(texto)
        linhas.append("")

    linhas.append("## Agências postais selecionadas, por combinação")
    linhas.append("")
    com_postal = tabela[tabela["n_agencias_postais"] > 0]
    if com_postal.empty:
        linhas.append("Nenhuma agência postal é selecionada em nenhuma das combinações.")
    else:
        linhas.append("| Cenário | Ordenação | Escore | Agências |")
        linhas.append("|---|---|---|---|")
        for l in com_postal.itertuples():
            linhas.append(f"| {l.cenario} | {ROTULO_ORDEM[l.ordem_pesos]} | "
                          f"{numero_br(l.escore_postal, 4)} | {l.agencias_postais} |")
    linhas.append("")
    return "\n".join(linhas)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sensibilidade ao escore de tipo das agencias postais.")
    parser.add_argument("--saida", default=None,
                        help="versao da rodada (ex.: v36): le de data/tratados/<saida>/ e "
                             "results/<saida>/tabelas/ e grava em results/<saida>/")
    saida = parser.parse_args().saida
    if saida:
        M.ARQ_CANDIDATOS = DIR_TRATADOS / saida / M.ARQ_CANDIDATOS.name
        dir_tabelas = RAIZ / "results" / saida / "tabelas"
        dir_saida = RAIZ / "results" / saida
    else:
        dir_tabelas = RAIZ / "results" / "tabelas"
        dir_saida = dir_tabelas

    inicio = time.perf_counter()
    M.titulo(
        "10_sensibilidade_escore_postal.py — escore de tipo das agencias postais\n"
        f"escores: {', '.join(f'{e:.4f}' for e in ESCORES)}\n"
        f"ordens: {', '.join(ORDENS)} · cenarios: {', '.join(CENARIOS)} · "
        f"caso-base: escore {ESCORE_BASE:.4f}, ordem ({ORDEM_BASE})\n"
        f"candidatos: {M.ARQ_CANDIDATOS.relative_to(RAIZ).as_posix()}"
    )
    M.checar(M.ARQ_CANDIDATOS.exists(), f"{M.ARQ_CANDIDATOS} nao existe. Rodar src/04_atratividade.py.")
    candidatos = gpd.read_file(M.ARQ_CANDIDATOS)
    M.checar(len(candidatos) == M.N_CANDIDATOS_ESPERADO, f"{len(candidatos)} candidatos.")
    print(f"   agencias postais entre os candidatos: {int((candidatos['TIPO'] == TIPO_POSTAL).sum())}")
    conferir_indice(candidatos)

    tabela = rodar(candidatos, dir_tabelas)
    verificar(tabela)

    M.subtitulo("Gravacao")
    dir_saida.mkdir(parents=True, exist_ok=True)
    arq_csv = dir_saida / "sensibilidade_escore_postal.csv"
    arq_md = dir_saida / "quadro_sensibilidade_escore_postal.md"
    tabela.to_csv(arq_csv, index=False, sep=";", decimal=",")
    arq_md.write_text(quadro_markdown(tabela), encoding="utf-8")
    for caminho in (arq_csv, arq_md):
        print(f"   {caminho.relative_to(RAIZ).as_posix()}")

    resolucoes_e2 = int((~tabela["caso_base"]).sum())
    M.titulo(
        f"Concluido em {time.perf_counter() - inicio:.1f} s · {len(tabela)} combinacoes "
        f"({len(CENARIOS)} casos-base em dois estagios + {resolucoes_e2} so do Estagio 2)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (M.FalhaDeSanidade, A.FalhaDeSanidade) as erro:
        print()
        print("!" * 78)
        print("VERIFICACAO DE SANIDADE FALHOU")
        print("!" * 78)
        print(erro)
        sys.exit(1)
