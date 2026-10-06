"""
05_distancias.py — matrizes de distancia entre pontos de demanda e candidatos.

Tres matrizes completas de pontos de demanda x 856 candidatos, para a unidade de
demanda escolhida — 964 setores censitarios habitados (padrao) ou 87 bairros:

    euclidiana   referencia, em EPSG:31982
    rede walk    para os cenarios de r = 500 m e r = 800 m
    rede drive   para o cenario de r = 1500 m

As matrizes sao COMPLETAS: nao ha poda por raio aqui. A poda entra na Etapa 7,
antes de construir as variaveis do modelo.

Pares inalcancaveis recebem infinito, nunca um numero grande arbitrario.

Entradas:
    data/tratados/pontos_demanda_setor.gpkg  964 pontos de setor (Etapa 2)
    data/tratados/pontos_demanda.gpkg        87 pontos de bairro (Etapa 2), com --unidade bairro
    data/tratados/candidatos_com_aj.gpkg     856 candidatos (Etapa 5)
    data/brutos/ibge/SC_setores_CD2022.gpkg  massas d'agua, para a auditoria
    OpenStreetMap via OSMnx                   grafos walk e drive

Saidas (com --unidade setor, todos os nomes ganham o sufixo _setor; com --unidade
bairro, as tabelas vao para results/nao_usados/tabelas/):
    data/tratados/dist_euclid.parquet
    data/tratados/dist_rede_walk.parquet
    data/tratados/dist_rede_drive.parquet
    results/tabelas/comparacao_distancias.csv
    results/tabelas/elegibilidade_por_raio.csv
    results/tabelas/travessias_grafos.csv

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/05_distancias.py                     # setores censitarios (padrao)
    python src/05_distancias.py --unidade bairro    # bairros (fora do escopo; tabelas em nao_usados)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString

# =============================================================================
# PARAMETROS
# =============================================================================

RAIZ = Path(__file__).resolve().parents[1]
DIR_TRATADOS = RAIZ / "data" / "tratados"
DIR_TABELAS = RAIZ / "results" / "tabelas"   # redefinido em main() conforme a unidade
# Resultados que NAO entram na monografia vao para results/nao_usados/. A rodada
# por bairro saiu do escopo quando o setor censitario virou a unidade de demanda
# (docs/decisoes_etapa10.md): ela continua reproduzivel, mas grava ali, para nao
# se misturar com o que vai para o texto.
DIR_RESULTADOS = {"setor": RAIZ / "results", "bairro": RAIZ / "results" / "nao_usados"}

ARQ_CANDIDATOS = DIR_TRATADOS / "candidatos_com_aj.gpkg"
ARQ_BAIRROS = DIR_TRATADOS / "bairros.gpkg"
ARQ_SETORES = RAIZ / "data" / "brutos" / "ibge" / "SC_setores_CD2022.gpkg"

CD_MUN = "4205407"
CRS_TRABALHO = 31982
CD_SIT_MASSA_DAGUA = "9"
POP_REFERENCIA = 537_211
N_CANDIDATOS_ESPERADO = 856

# -----------------------------------------------------------------------------
# Unidade de demanda
# -----------------------------------------------------------------------------
# 'setor'   um ponto por setor censitario habitado — a configuracao PRINCIPAL,
#           adotada depois que a auditoria mostrou que o ponto do bairro fica longe
#           da populacao em muitos bairros (docs/decisoes_etapa10.md).
# 'bairro'  87 pontos, um por bairro — a configuracao original, mantida para
#           comparacao.
# A chave e a coluna que identifica o ponto de demanda nas matrizes.
UNIDADES = {
    "bairro": {"pontos": DIR_TRATADOS / "pontos_demanda.gpkg", "chave": "CD_BAIRRO", "sufixo": "", "n_esperado": 87},
    "setor": {"pontos": DIR_TRATADOS / "pontos_demanda_setor.gpkg", "chave": "CD_SETOR", "sufixo": "_setor", "n_esperado": 964},
}

# -----------------------------------------------------------------------------
# Cenarios (registrados em docs/decisoes_etapa6.md)
# -----------------------------------------------------------------------------
# C1-C3 medem o retorno decrescente de p; C4 e C5, o efeito do raio.
CENARIOS = {
    "C1": {"p": 10, "r": 800},
    "C2": {"p": 20, "r": 800},
    "C3": {"p": 40, "r": 800},
    "C4": {"p": 20, "r": 500},
    "C5": {"p": 20, "r": 1500},
}
# Metrica de rede associada a cada raio: a pe ate 800 m, motorizada em 1500 m.
METRICA_POR_RAIO = {500: "walk", 800: "walk", 1500: "drive"}

# -----------------------------------------------------------------------------
# Grafos
# -----------------------------------------------------------------------------
MODOS = ("walk", "drive")
FORCAR_DOWNLOAD_GRAFOS = False
ARQ_GRAFO = {modo: DIR_TRATADOS / f"_cache_grafo_{modo}.graphml" for modo in MODOS}
ARQ_METADADOS_GRAFOS = DIR_TRATADOS / "_cache_grafos_metadados.json"

# -----------------------------------------------------------------------------
# Travessias Ilha <-> Continente
# -----------------------------------------------------------------------------
# Sao TRES pontes, nao duas. A Ponte Hercilio Luz e aberta a automoveis de passeio
# em dias uteis, com fluxo continuo a partir das 5h, e fechada a carros de sexta a
# noite ate segunda as 5h, quando fica exclusiva de pedestres e ciclistas. Motos e
# caminhoes sao proibidos em qualquer dia.
#
# Verificado no OSM: a via traz highway=secondary, motorcar=yes e
# motorcar:conditional="no @ (Sa,Su)". O filtro drive do OSMnx nao interpreta tags
# conditional, entao a ponte ENTRA no grafo drive — que e o regime de dia util, o
# relevante para retirada de encomendas.
#
# No grafo walk, Colombo Salles e Pedro Ivo ficam de fora por serem motorway; a
# travessia a pe se da por uma passarela etiquetada bridge=yes SEM NOME no OSM.
# Por isso a verificacao por nome NAO BASTA e existe a auditoria geometrica.
PONTES_ESPERADAS_DRIVE = [
    "Ponte Hercílio Luz",
    "Ponte Governador Colombo Machado Salles",
    "Ponte Governador Pedro Ivo Campos",
]
# Comprimento minimo de intersecao com a massa d'agua para uma aresta contar como
# travessia. Abaixo disso e imprecisao de digitalizacao da linha de costa.
MIN_TRAVESSIA_M = 20.0

# A mascara d'agua vem dos setores censitarios do IBGE, cuja linha de costa tem
# resolucao de setor, nao de levantamento topografico. Vias e trilhas litoraneas
# encostam nela e aparecem como "cruzando agua" por algumas dezenas de metros sem
# serem ponte — verificado: Rodovia Gilson da Costa Xavier, Rua da Croa, Trilha
# pelas Pedras. Isso e ruido da mascara, nao travessia inexistente.
#
# O que NAO pode passar e uma travessia sem ponte que ligue a Ilha ao Continente:
# essa criaria um atalho inexistente e falsearia toda a matriz. Por isso a falha
# dura e reservada a esse caso, e o resto e reportado nominalmente para conferencia.
FALHAR_SO_EM_TRAVESSIA_ILHA_CONTINENTE = True

# Bairros de referencia para o teste nominal de travessia.
BAIRRO_ILHA_TESTE = "Centro"
BAIRRO_CONTINENTE_TESTE = "Estreito"

# Distancia acima da qual a ancoragem de um ponto ao no mais proximo do grafo
# vira sinal de area sem via mapeada.
LIMITE_ANCORAGEM_M = 300.0

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


def texto_de(valor) -> str:
    """Normaliza atributos do OSM que podem vir como lista."""
    if isinstance(valor, list):
        return " | ".join(str(x) for x in valor)
    return "" if valor is None else str(valor)


# =============================================================================
# DADOS
# =============================================================================


def carregar_pontos(unidade: dict) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, object]:
    subtitulo("1. Pontos de demanda e candidatos")

    for caminho, etapa in [(unidade["pontos"], "01_demanda.py"), (ARQ_CANDIDATOS, "04_atratividade.py")]:
        checar(caminho.exists(), f"{caminho} nao existe. Rodar antes: python src/{etapa}")

    demanda = gpd.read_file(unidade["pontos"]).to_crs(CRS_TRABALHO)
    candidatos = gpd.read_file(ARQ_CANDIDATOS).to_crs(CRS_TRABALHO)
    checar(
        len(demanda) == unidade["n_esperado"],
        f"{len(demanda)} pontos de demanda, esperado {unidade['n_esperado']}.",
    )
    checar(demanda[unidade["chave"]].is_unique, f"{unidade['chave']} repetido nos pontos de demanda.")
    checar(
        int(demanda["POP"].sum()) == POP_REFERENCIA,
        f"Os pontos de demanda somam {int(demanda['POP'].sum())} habitantes, esperado {POP_REFERENCIA}.",
    )
    checar(
        len(candidatos) == N_CANDIDATOS_ESPERADO,
        f"{len(candidatos)} candidatos, esperado {N_CANDIDATOS_ESPERADO}.",
    )

    setores = gpd.read_file(ARQ_SETORES, where=f"CD_MUN = '{CD_MUN}'")
    setores = setores[["CD_SETOR", "CD_SIT", "geometry"]].dissolve(
        by="CD_SETOR", as_index=False, aggfunc="first"
    ).to_crs(CRS_TRABALHO)
    agua = setores[setores["CD_SIT"] == CD_SIT_MASSA_DAGUA].geometry.make_valid().union_all()

    # Ilha ou continente, para a verificacao das travessias.
    demanda["INSULAR"] = demanda["REGIAO_FUNCIONAL"] != "Continente"
    candidatos["INSULAR"] = candidatos["REGIAO_FUNCIONAL"] != "Continente"

    print(f"pontos de demanda ...................... {len(demanda)}")
    print(f"candidatos ............................. {len(candidatos)}")
    print(f"pares a calcular por matriz ............ {numero_br(len(demanda) * len(candidatos))}")
    print(f"bairros insulares ...................... {int(demanda['INSULAR'].sum())} · "
          f"continentais {int((~demanda['INSULAR']).sum())}")
    print(f"candidatos insulares ................... {int(candidatos['INSULAR'].sum())} · "
          f"continentais {int((~candidatos['INSULAR']).sum())}")
    return demanda, candidatos, agua


def carregar_grafo(modo: str, territorio) -> nx.MultiDiGraph:
    """Grafo viario do modo, do cache ou do OSM."""
    if ARQ_GRAFO[modo].exists() and not FORCAR_DOWNLOAD_GRAFOS:
        grafo = ox.load_graphml(ARQ_GRAFO[modo])
        print(f"   grafo {modo}: lido do cache — "
              f"{grafo.number_of_nodes()} nos, {grafo.number_of_edges()} arestas")
        return grafo

    poly = gpd.GeoSeries([territorio], crs=CRS_TRABALHO).to_crs(4326).iloc[0]
    print(f"   grafo {modo}: baixando do OSM...")
    grafo = ox.graph_from_polygon(poly, network_type=modo, simplify=True)
    ox.save_graphml(grafo, ARQ_GRAFO[modo])
    print(f"   grafo {modo}: {grafo.number_of_nodes()} nos, "
          f"{grafo.number_of_edges()} arestas — cache gravado")
    return grafo


# =============================================================================
# VERIFICACAO DOS GRAFOS
# =============================================================================


def margem_de(pontos: gpd.GeoSeries, bairros: gpd.GeoDataFrame) -> list[bool]:
    """True se o ponto esta na Ilha, False se no Continente."""
    juncao = gpd.sjoin_nearest(
        gpd.GeoDataFrame(geometry=pontos, crs=CRS_TRABALHO),
        bairros[["REGIAO_FUNCIONAL", "geometry"]],
        how="left",
    )
    juncao = juncao[~juncao.index.duplicated(keep="first")]
    return (juncao["REGIAO_FUNCIONAL"] != "Continente").tolist()


def auditar_travessias(
    grafo: nx.MultiDiGraph, modo: str, agua, bairros: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Confere as travessias sobre a agua, por nome E por geometria.

    A verificacao por nome nao basta: no grafo walk a travessia se da por uma
    passarela SEM NOME no OSM, e no drive o nome 'Hercilio Luz' tambem casa com a
    Avenida Hercilio Luz, que fica no Centro e nao cruza nada.

    A verificacao decisiva e geometrica: toda aresta que cruza a massa d'agua por
    mais de MIN_TRAVESSIA_M precisa estar etiquetada bridge. Uma que nao esteja e
    travessia inexistente — o analogo, aqui, das massas d'agua coladas a bairros
    na Etapa 2.
    """
    arestas = ox.graph_to_gdfs(grafo, nodes=False).to_crs(CRS_TRABALHO)

    # 1) Verificacao nominal, so no drive, onde as tres pontes sao esperadas.
    if modo == "drive":
        print(f"   travessias esperadas, por nome:")
        nomes = arestas["name"].map(texto_de)
        for ponte in PONTES_ESPERADAS_DRIVE:
            casou = arestas[nomes.str.contains(ponte, case=False, na=False, regex=False)]
            checar(
                len(casou) > 0,
                f"A {ponte} nao esta no grafo {modo}. Sem ela a matriz de distancias "
                "Ilha-Continente sai errada.",
            )
            tags = casou.iloc[0]
            print(f"      {ponte:<42} {len(casou):>3} aresta(s) · "
                  f"{casou['length'].sum():>7.0f} m · highway={texto_de(tags.get('highway'))}")

    # 2) Verificacao geometrica: quem realmente cruza a agua.
    candidatas = arestas[arestas.intersects(agua)].copy()
    candidatas["SOBRE_AGUA_M"] = candidatas.geometry.intersection(agua).length
    travessias = candidatas[candidatas["SOBRE_AGUA_M"] >= MIN_TRAVESSIA_M].copy()
    if "bridge" in travessias.columns:
        travessias["E_PONTE"] = (
            travessias["bridge"].map(texto_de).str.lower().isin(["yes", "viaduct", "true", "1"])
        )
    else:
        travessias["E_PONTE"] = False

    # Cada travessia liga dois pontos: se estao em margens opostas, ela e uma
    # ligacao Ilha-Continente e precisa obrigatoriamente ser ponte.
    inicios = gpd.GeoSeries(
        [LineString(g.coords).interpolate(0, normalized=True) for g in travessias.geometry],
        crs=CRS_TRABALHO, index=travessias.index,
    )
    fins = gpd.GeoSeries(
        [LineString(g.coords).interpolate(1, normalized=True) for g in travessias.geometry],
        crs=CRS_TRABALHO, index=travessias.index,
    )
    travessias["LIGA_MARGENS"] = [
        a != b for a, b in zip(margem_de(inicios, bairros), margem_de(fins, bairros))
    ]

    print(f"   arestas que cruzam massa d'agua (> {MIN_TRAVESSIA_M:.0f} m): {len(travessias)}")
    print(f"      etiquetadas como ponte .............. {int(travessias['E_PONTE'].sum())}")
    print(f"      ligando Ilha e Continente ........... {int(travessias['LIGA_MARGENS'].sum())}")

    graves = travessias[travessias["LIGA_MARGENS"] & ~travessias["E_PONTE"]]
    checar(
        len(graves) == 0,
        f"Grafo {modo}: {len(graves)} aresta(s) ligam Ilha e Continente SEM tag de "
        "ponte — travessia inexistente, que criaria atalho sobre a baia.",
    )
    print(f"   nenhuma travessia Ilha-Continente sem ponte ... conferido")

    leves = travessias[~travessias["LIGA_MARGENS"] & ~travessias["E_PONTE"]]
    if len(leves) > 0:
        resumo_leves = (
            leves.assign(NOME=leves["name"].map(texto_de).replace("", "(sem nome)"))
            .groupby("NOME", as_index=False)
            .agg(arestas=("length", "size"), sobre_agua_m=("SOBRE_AGUA_M", "max"))
            .sort_values("sobre_agua_m", ascending=False)
        )
        print(f"   cruzam agua sem ser ponte, na MESMA margem: {len(leves)} aresta(s)")
        print(f"      atribuido a resolucao da mascara d'agua (setores do IBGE):")
        for linha in resumo_leves.itertuples():
            print(f"      {linha.NOME[:48]:<48} {linha.sobre_agua_m:>6.0f} m")

    resumo = (
        travessias.assign(NOME=travessias["name"].map(texto_de).replace("", "(sem nome)"))
        .groupby("NOME", as_index=False)
        .agg(arestas=("length", "size"), extensao_m=("length", "sum"),
             sobre_agua_m=("SOBRE_AGUA_M", "sum"))
        .sort_values("sobre_agua_m", ascending=False)
    )
    for linha in resumo.itertuples():
        print(f"      {linha.NOME[:46]:<46} {linha.sobre_agua_m:>7.0f} m sobre agua")
    resumo.insert(0, "modo", modo)
    return resumo


def auditar_componentes(grafo: nx.MultiDiGraph, modo: str) -> set:
    """Componentes do grafo e escolha da maior; devolve os nos que ficam."""
    if modo == "drive":
        componentes = sorted(nx.strongly_connected_components(grafo), key=len, reverse=True)
        criterio = "fortemente conexas"
    else:
        componentes = sorted(nx.connected_components(nx.Graph(grafo)), key=len, reverse=True)
        criterio = "conexas"

    maior = componentes[0]
    fora = grafo.number_of_nodes() - len(maior)
    print(f"   componentes {criterio}: {len(componentes)} · maior {len(maior)} nos "
          f"({100 * len(maior) / grafo.number_of_nodes():.1f}%) · fora {fora}")
    if len(componentes) > 1:
        tamanhos = [len(c) for c in componentes[1:6]]
        print(f"      demais componentes (5 maiores): {tamanhos}")
    return maior


def ancorar(
    pontos: gpd.GeoDataFrame, grafo: nx.MultiDiGraph, maior: set, rotulo: str, coluna_nome: str
) -> tuple[np.ndarray, np.ndarray]:
    """Associa cada ponto ao no mais proximo DA COMPONENTE PRINCIPAL do grafo.

    Ancorar no no mais proximo sem restricao pode cair numa componente isolada
    — um trecho de via sem saida desconectado da malha —, e entao o ponto fica a
    distancia infinita de todos os candidatos. Verificado com setores: um setor de
    Recanto dos Acores ancorava assim no grafo drive. A perna de ancoragem ate o
    no da componente principal e somada a distancia, como qualquer outra.
    """
    p4326 = pontos.to_crs(4326)
    nos_livres = np.asarray(ox.distance.nearest_nodes(grafo, p4326.geometry.x, p4326.geometry.y))
    nos = np.asarray(ox.distance.nearest_nodes(grafo.subgraph(maior), p4326.geometry.x, p4326.geometry.y))

    gdf_nos = ox.graph_to_gdfs(grafo, edges=False).to_crs(CRS_TRABALHO)
    coords = gdf_nos.loc[nos, ["geometry"]].reset_index(drop=True)
    distancia = gpd.GeoSeries(pontos.geometry.to_numpy(), crs=CRS_TRABALHO).distance(
        gpd.GeoSeries(coords.geometry.to_numpy(), crs=CRS_TRABALHO)
    ).to_numpy()

    checar(bool(np.isin(nos, list(maior)).all()), f"{rotulo}: ancoragem fora da componente principal.")
    fora = nos_livres != nos
    print(f"   {rotulo}: ancoragem mediana {np.median(distancia):.0f} m · "
          f"max {distancia.max():.0f} m")
    if fora.any():
        print(f"      no mais proximo em componente isolada: {int(fora.sum())} — reancorados na principal")
        for nome in pontos.loc[fora, coluna_nome].head(10):
            print(f"         {nome}")
    else:
        print(f"      todos na componente principal ...... conferido")

    longe = distancia > LIMITE_ANCORAGEM_M
    if longe.any():
        print(f"      ancorados a mais de {LIMITE_ANCORAGEM_M:.0f} m: {int(longe.sum())} "
              "(possivel area sem via mapeada)")
        for nome, d in zip(pontos.loc[longe, coluna_nome].head(8), distancia[longe][:8]):
            print(f"         {str(nome)[:40]:<40} {d:>7.0f} m")
    return nos, distancia


def testar_travessia_nominal(grafo: nx.MultiDiGraph, modo: str, demanda: gpd.GeoDataFrame) -> None:
    """Rota entre um bairro insular e um continental, conferindo que cruza ponte."""
    p_ilha = demanda.loc[demanda["NM_BAIRRO"] == BAIRRO_ILHA_TESTE].to_crs(4326)
    p_cont = demanda.loc[demanda["NM_BAIRRO"] == BAIRRO_CONTINENTE_TESTE].to_crs(4326)
    n1 = ox.distance.nearest_nodes(grafo, p_ilha.geometry.x.iloc[0], p_ilha.geometry.y.iloc[0])
    n2 = ox.distance.nearest_nodes(grafo, p_cont.geometry.x.iloc[0], p_cont.geometry.y.iloc[0])
    try:
        comprimento = nx.shortest_path_length(grafo, n1, n2, weight="length")
    except nx.NetworkXNoPath:
        raise FalhaDeSanidade(
            f"Grafo {modo}: nao ha caminho de {BAIRRO_ILHA_TESTE} a "
            f"{BAIRRO_CONTINENTE_TESTE}. As travessias nao estao no grafo."
        )
    print(f"   {BAIRRO_ILHA_TESTE} -> {BAIRRO_CONTINENTE_TESTE}: {numero_br(comprimento)} m")


# =============================================================================
# MATRIZES
# =============================================================================


def matriz_euclidiana(demanda: gpd.GeoDataFrame, candidatos: gpd.GeoDataFrame, chave: str) -> pd.DataFrame:
    subtitulo("2. Matriz euclidiana")
    xd = demanda.geometry.x.to_numpy()[:, None]
    yd = demanda.geometry.y.to_numpy()[:, None]
    xc = candidatos.geometry.x.to_numpy()[None, :]
    yc = candidatos.geometry.y.to_numpy()[None, :]
    distancias = np.hypot(xd - xc, yd - yc)

    matriz = pd.DataFrame(
        {
            chave: np.repeat(demanda[chave].to_numpy(), len(candidatos)),
            "ID_CANDIDATO": np.tile(candidatos.index.to_numpy(), len(demanda)),
            "DIST_M": distancias.ravel(),
        }
    )
    print(f"pares .................................. {numero_br(len(matriz))}")
    print(f"distancia .............................. min {matriz['DIST_M'].min():.0f} m · "
          f"mediana {matriz['DIST_M'].median():.0f} m · max {matriz['DIST_M'].max():.0f} m")
    return matriz


def matriz_rede(
    demanda: gpd.GeoDataFrame,
    candidatos: gpd.GeoDataFrame,
    grafo: nx.MultiDiGraph,
    nos_demanda: np.ndarray,
    nos_candidatos: np.ndarray,
    anc_demanda: np.ndarray,
    anc_candidatos: np.ndarray,
    modo: str,
    chave: str,
) -> pd.DataFrame:
    """Uma execucao de Dijkstra por no de origem, nunca par a par.

    Com bairros sao 87 arvores de caminhos minimos por grafo, contra 75.690 buscas
    que a abordagem par a par exigiria. Com setores, varios pontos ancoram no
    mesmo no do grafo: a arvore e calculada uma vez por no e reaproveitada.

    A distancia total soma as duas PERNAS DE ANCORAGEM — do ponto de demanda ate
    o no do grafo e do no ate o candidato. Sem elas, um bairro e um candidato que
    se ancorem no mesmo no ficariam a distancia ZERO mesmo separados por centenas
    de metros, o que ocorria em Carianos, onde a rede a pe e esparsa.
    """
    print(f"   calculando {len(demanda)} arvores de caminhos minimos...")
    inicio = time.perf_counter()
    linhas = []
    inalcancaveis = 0

    # Distancia de rede no-a-no ate os nos dos candidatos, por no de origem. So as
    # distancias ate os candidatos ficam em memoria, nunca a arvore inteira.
    por_no: dict = {}
    for i, (codigo, no_origem) in enumerate(zip(demanda[chave].to_numpy(), nos_demanda)):
        if no_origem not in por_no:
            alcance = nx.single_source_dijkstra_path_length(grafo, no_origem, weight="length")
            por_no[no_origem] = np.array([alcance.get(n, np.inf) for n in nos_candidatos])
        distancias = por_no[no_origem] + anc_demanda[i] + anc_candidatos
        inalcancaveis += int(np.isinf(distancias).sum())
        linhas.append(
            pd.DataFrame(
                {
                    chave: codigo,
                    "ID_CANDIDATO": candidatos.index.to_numpy(),
                    "DIST_M": distancias,
                }
            )
        )
    print(f"   nos de origem distintos ................ {len(por_no)} para {len(demanda)} pontos")

    matriz = pd.concat(linhas, ignore_index=True)
    finitos = matriz.loc[np.isfinite(matriz["DIST_M"]), "DIST_M"]
    print(f"   concluido em {time.perf_counter() - inicio:.1f} s")
    print(f"   ancoragem somada ....................... demanda mediana "
          f"{np.median(anc_demanda):.0f} m · candidatos mediana {np.median(anc_candidatos):.0f} m")
    print(f"   pares .................................. {numero_br(len(matriz))}")
    print(f"   inalcancaveis (inf) .................... {numero_br(inalcancaveis)} "
          f"({100 * inalcancaveis / len(matriz):.2f}%)")
    print(f"   distancia .............................. min {finitos.min():.0f} m · "
          f"mediana {finitos.median():.0f} m · max {finitos.max():.0f} m")
    return matriz


# =============================================================================
# PLAUSIBILIDADE
# =============================================================================


def comparar_metricas(
    euclid: pd.DataFrame,
    redes: dict[str, pd.DataFrame],
    demanda: gpd.GeoDataFrame,
    candidatos: gpd.GeoDataFrame,
    chave: str,
) -> pd.DataFrame:
    """Razao rede/euclidiana por regiao e por travessia da baia.

    Consistencia interna nao detecta grafo atravessando onde nao ha via. Razao
    proxima de 1 no Sul ou no Norte da Ilha, ou em pares Ilha-Continente, e o
    sinal de alarme.
    """
    subtitulo("5. Plausibilidade — razao rede / euclidiana")

    regiao_bairro = demanda.set_index(chave)["REGIAO_FUNCIONAL"]
    nome_bairro = demanda.set_index(chave)["NM_BAIRRO"]
    insular_bairro = demanda.set_index(chave)["INSULAR"]

    tabelas = []
    for modo, rede in redes.items():
        base = euclid.merge(
            rede, on=[chave, "ID_CANDIDATO"], suffixes=("_EUCLID", "_REDE")
        )
        base = base[np.isfinite(base["DIST_M_REDE"]) & (base["DIST_M_EUCLID"] > 0)].copy()
        base["RAZAO"] = base["DIST_M_REDE"] / base["DIST_M_EUCLID"]
        base["REGIAO"] = base[chave].map(regiao_bairro)
        base["ILHA_BAIRRO"] = base[chave].map(insular_bairro)
        base["ILHA_CAND"] = base["ID_CANDIDATO"].map(candidatos["INSULAR"])
        base["CRUZA_BAIA"] = base["ILHA_BAIRRO"] != base["ILHA_CAND"]

        print(f"\n   === {modo} ===")
        print(f"   {'regiao':<14} {'pares':>9} {'razao mediana':>14} {'p10':>7} {'p90':>7}")
        for regiao in ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]:
            sub = base[base["REGIAO"] == regiao]
            if sub.empty:
                continue
            print(f"   {regiao:<14} {numero_br(len(sub)):>9} {sub['RAZAO'].median():>14.2f} "
                  f"{sub['RAZAO'].quantile(0.1):>7.2f} {sub['RAZAO'].quantile(0.9):>7.2f}")
            tabelas.append(
                {
                    "modo": modo, "recorte": regiao, "pares": len(sub),
                    "razao_mediana": round(float(sub["RAZAO"].median()), 3),
                    "razao_p10": round(float(sub["RAZAO"].quantile(0.1)), 3),
                    "razao_p90": round(float(sub["RAZAO"].quantile(0.9)), 3),
                }
            )

        for rotulo, mascara in [("mesma margem", ~base["CRUZA_BAIA"]),
                                ("cruza a baia", base["CRUZA_BAIA"])]:
            sub = base[mascara]
            if sub.empty:
                continue
            print(f"   {rotulo:<14} {numero_br(len(sub)):>9} {sub['RAZAO'].median():>14.2f} "
                  f"{sub['RAZAO'].quantile(0.1):>7.2f} {sub['RAZAO'].quantile(0.9):>7.2f}")
            tabelas.append(
                {
                    "modo": modo, "recorte": rotulo, "pares": len(sub),
                    "razao_mediana": round(float(sub["RAZAO"].median()), 3),
                    "razao_p10": round(float(sub["RAZAO"].quantile(0.1)), 3),
                    "razao_p90": round(float(sub["RAZAO"].quantile(0.9)), 3),
                }
            )

        cruza = base[base["CRUZA_BAIA"]]
        if not cruza.empty:
            checar(
                float(cruza["RAZAO"].median()) > 1.2,
                f"Grafo {modo}: pares que cruzam a baia tem razao mediana "
                f"{cruza['RAZAO'].median():.2f}, proxima de 1. Isso indica travessia "
                "inexistente — a rota deveria ser obrigada a passar pelas pontes.",
            )
            print(f"   pares Ilha-Continente com razao > 1,2 ... conferido "
                  f"(mediana {cruza['RAZAO'].median():.2f})")

        base["NM_BAIRRO"] = base[chave].map(nome_bairro)
        base["NM_CAND"] = base["ID_CANDIDATO"].map(candidatos["NOME"].fillna("(sem nome)"))
        print(f"\n   10 maiores razoes ({modo}):")
        for linha in base.nlargest(10, "RAZAO").itertuples():
            print(f"      {linha.NM_BAIRRO[:20]:<20} -> {str(linha.NM_CAND)[:26]:<26} "
                  f"euclid {linha.DIST_M_EUCLID:>7.0f} m · rede {linha.DIST_M_REDE:>8.0f} m · "
                  f"{linha.RAZAO:>6.2f}x")
        print(f"   10 menores razoes ({modo}):")
        for linha in base.nsmallest(10, "RAZAO").itertuples():
            print(f"      {linha.NM_BAIRRO[:20]:<20} -> {str(linha.NM_CAND)[:26]:<26} "
                  f"euclid {linha.DIST_M_EUCLID:>7.0f} m · rede {linha.DIST_M_REDE:>8.0f} m · "
                  f"{linha.RAZAO:>6.2f}x")

        # Razao restrita aos pares DENTRO de cada raio de cenario. Pares de dezenas
        # de quilometros convergem para 1, porque o percurso se aproxima de uma
        # rodovia quase retilinea, e diluem o contraste entre regioes. A faixa que
        # importa para acesso a um ponto de retirada e a curta.
        raios_do_modo = [r for r, m in METRICA_POR_RAIO.items() if m == modo]
        if raios_do_modo:
            print(f"\nrazao restrita a pares curtos ({modo}), por raio de cenario:")
            cabecalho = f"   {'regiao':<14} {'todos os pares':>15}"
            for raio in sorted(raios_do_modo):
                cabecalho += f"{'<= ' + str(raio) + ' m':>16}"
            print(cabecalho)
            for regiao in ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]:
                sub = base[base["REGIAO"] == regiao]
                if sub.empty:
                    continue
                texto = f"   {regiao:<14} {sub['RAZAO'].median():>14.2f}x"
                for raio in sorted(raios_do_modo):
                    curto = sub[sub["DIST_M_REDE"] <= raio]
                    if len(curto) < 5:
                        texto += f"{'n<5':>16}"
                    else:
                        texto += f"{curto['RAZAO'].median():>13.2f}x ({len(curto)})"
                    tabelas.append(
                        {
                            "modo": modo, "recorte": f"{regiao} | rede <= {raio} m",
                            "pares": len(curto),
                            "razao_mediana": (round(float(curto["RAZAO"].median()), 3)
                                              if len(curto) else None),
                            "razao_p10": (round(float(curto["RAZAO"].quantile(0.1)), 3)
                                          if len(curto) else None),
                            "razao_p90": (round(float(curto["RAZAO"].quantile(0.9)), 3)
                                          if len(curto) else None),
                        }
                    )
                print(texto)
            print("   (entre parenteses, o numero de pares no recorte)")

        checar(
            float(base["RAZAO"].min()) >= 0.999,
            f"Grafo {modo}: ha par com distancia de rede MENOR que a euclidiana "
            f"({base['RAZAO'].min():.4f}x), o que e geometricamente impossivel.",
        )

    return pd.DataFrame(tabelas)


def antecipar_cenarios(
    euclid: pd.DataFrame, redes: dict[str, pd.DataFrame], demanda: gpd.GeoDataFrame, chave: str
) -> pd.DataFrame:
    """Quantos pares e quantos pontos de demanda sobrevivem a cada raio dos cenarios.

    O teto aqui e medido sobre a POPULACAO. O teto que o modelo registra no log e
    medido sobre a DEMANDA h_i, que pondera por renda quando beta > 0; os dois
    diferem em menos de um ponto percentual e nao devem ser misturados no texto.
    """
    subtitulo("6. Elegibilidade por raio — antecipacao dos cenarios")

    populacao = demanda.set_index(chave)["POP"]
    total_pop = float(populacao.sum())

    linhas = []
    print(f"   {'raio':>6} {'metrica':<9} {'pares':>9} {'pontos cobriveis ':>18} "
          f"{'teto de cobertura':>19} {'candidatos uteis':>17}")
    for raio in sorted({c["r"] for c in CENARIOS.values()}):
        modo = METRICA_POR_RAIO[raio]
        for rotulo, matriz in [(modo, redes[modo]), ("euclid", euclid)]:
            elegiveis = matriz[matriz["DIST_M"] <= raio]
            cobriveis = set(elegiveis[chave])
            # Teto de cobertura: como cada unidade de demanda e representada por UM
            # ponto, a que nao tem candidato ao alcance nao e coberta por p algum.
            # Este e o limite superior de Z1*, dado r — independe de p.
            teto = 100 * float(populacao.reindex(cobriveis).sum()) / total_pop
            candidatos_uteis = elegiveis["ID_CANDIDATO"].nunique()
            print(f"   {raio:>5}m {rotulo:<9} {numero_br(len(elegiveis)):>9} "
                  f"{len(cobriveis):>10} de {len(demanda):<4} {teto:>18.1f}% "
                  f"{candidatos_uteis:>17}")
            linhas.append(
                {
                    "raio_m": raio, "metrica": rotulo, "pares_elegiveis": len(elegiveis),
                    "pct_pares": round(100 * len(elegiveis) / len(matriz), 3),
                    "pontos_cobriveis": len(cobriveis),
                    "pontos_sem_candidato": len(demanda) - len(cobriveis),
                    "teto_cobertura_pct": round(teto, 2),
                    "candidatos_uteis": candidatos_uteis,
                    "candidatos_sem_bairro": matriz["ID_CANDIDATO"].nunique() - candidatos_uteis,
                }
            )

    print(f"\n   cenarios:")
    for nome, cfg in CENARIOS.items():
        modo = METRICA_POR_RAIO[cfg["r"]]
        elegiveis = redes[modo][redes[modo]["DIST_M"] <= cfg["r"]]
        cobriveis = set(elegiveis[chave])
        teto = 100 * float(populacao.reindex(list(cobriveis)).sum()) / total_pop
        print(f"      {nome}: p={cfg['p']:>2}, r={cfg['r']:>4} m ({modo}) — "
              f"{len(cobriveis):>2} pontos cobriveis, teto de Z1* = {teto:>5.1f}% da populacao")
    print("\n   O teto e imposto pelo RAIO, nao por p: uma unidade cujo ponto representativo")
    print("   nao tem candidato ao alcance nao e coberto por nenhum valor de p. Aumentar p")
    print("   so aproveita a folga que existe ABAIXO do teto.")

    return pd.DataFrame(linhas)


# =============================================================================
# SAIDAS
# =============================================================================


def salvar(
    euclid: pd.DataFrame,
    redes: dict[str, pd.DataFrame],
    candidatos: gpd.GeoDataFrame,
    comparacao: pd.DataFrame,
    elegibilidade: pd.DataFrame,
    travessias: pd.DataFrame,
    sufixo: str,
) -> None:
    subtitulo("7. Gravacao das saidas")
    DIR_TRATADOS.mkdir(parents=True, exist_ok=True)
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)

    arquivos = []
    for nome, matriz in [("dist_euclid", euclid)] + [
        (f"dist_rede_{modo}", redes[modo]) for modo in redes
    ]:
        caminho = DIR_TRATADOS / f"{nome}{sufixo}.parquet"
        matriz.to_parquet(caminho, index=False)
        arquivos.append(caminho)

    for nome, tabela in [
        ("comparacao_distancias", comparacao),
        ("elegibilidade_por_raio", elegibilidade),
        ("travessias_grafos", travessias),
    ]:
        caminho = DIR_TABELAS / f"{nome}{sufixo}.csv"
        tabela.to_csv(caminho, index=False, sep=";", decimal=",")
        arquivos.append(caminho)

    for caminho in arquivos:
        print(f"   {caminho.relative_to(RAIZ).as_posix():<48} {caminho.stat().st_size / 1024:>9.1f} KB")

    ARQ_METADADOS_GRAFOS.write_text(
        json.dumps(
            {
                "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "osmnx": ox.__version__,
                "networkx": nx.__version__,
                "cenarios": CENARIOS,
                "metrica_por_raio": {str(k): v for k, v in METRICA_POR_RAIO.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# =============================================================================
# EXECUCAO
# =============================================================================


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Matrizes de distancia demanda x candidatos.")
    parser.add_argument("--unidade", choices=sorted(UNIDADES), default="setor",
                        help="unidade de demanda: setor censitario (padrao) ou bairro")
    nome_unidade = parser.parse_args().unidade
    unidade = UNIDADES[nome_unidade]
    chave = unidade["chave"]
    global DIR_TABELAS
    DIR_TABELAS = DIR_RESULTADOS[nome_unidade] / "tabelas"

    inicio = time.perf_counter()
    titulo(
        "05_distancias.py — matrizes de distancia\n"
        f"unidade de demanda: {nome_unidade} · {unidade['n_esperado']} x {N_CANDIDATOS_ESPERADO} pares · "
        f"metricas: euclidiana, rede walk, rede drive\n"
        f"matrizes COMPLETAS: a poda por raio entra na Etapa 7"
    )

    demanda, candidatos, agua = carregar_pontos(unidade)
    bairros = gpd.read_file(ARQ_BAIRROS).to_crs(CRS_TRABALHO)
    territorio = bairros.union_all()

    euclid = matriz_euclidiana(demanda, candidatos, chave)

    redes: dict[str, pd.DataFrame] = {}
    travessias = []
    for modo in MODOS:
        subtitulo(f"3. Grafo {modo} — verificacao previa")
        grafo = carregar_grafo(modo, territorio)
        travessias.append(auditar_travessias(grafo, modo, agua, bairros))
        maior = auditar_componentes(grafo, modo)
        testar_travessia_nominal(grafo, modo, demanda)
        nos_demanda, anc_demanda = ancorar(demanda, grafo, maior, "pontos de demanda", "NM_BAIRRO")
        nos_candidatos, anc_candidatos = ancorar(candidatos, grafo, maior, "candidatos", "NOME")

        subtitulo(f"4. Matriz de rede — {modo}")
        redes[modo] = matriz_rede(
            demanda, candidatos, grafo, nos_demanda, nos_candidatos,
            anc_demanda, anc_candidatos, modo, chave,
        )

    comparacao = comparar_metricas(euclid, redes, demanda, candidatos, chave)
    elegibilidade = antecipar_cenarios(euclid, redes, demanda, chave)
    salvar(
        euclid, redes, candidatos, comparacao, elegibilidade,
        pd.concat(travessias, ignore_index=True), unidade["sufixo"],
    )

    titulo(f"Concluido em {time.perf_counter() - inicio:.1f} s · 3 matrizes de "
           f"{numero_br(len(euclid))} pares")
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
