"""
04_atratividade.py — indice de atratividade a_j dos candidatos.

Constroi o indice que pondera a distancia generalizada do Estagio 2 do modelo:

    a_j = SOMA_k w_k s_kj                                    (Equacao 8)
    d^a_ij = d_ij / a_j

Quatro atributos, com pesos por Rank-Order Centroid na ordem de importancia de
Baia, Silva e Filenga (2025). Escores normalizados em [0,2; 1], garantindo
a_j >= 0,2 por construcao: sem piso, d_ij / a_j diverge.

Entradas:
    data/tratados/candidatos.gpkg                 saida da Etapa 4
    data/tratados/bairros.gpkg                    territorio
    data/brutos/cnefe/4205407_FLORIANOPOLIS.csv   densidade comercial
    OpenStreetMap via OSMnx                        paradas, vias, rede caminhavel

Saidas:
    data/tratados/candidatos_com_aj.gpkg          candidatos com a_j e escores
    results/tabelas/quadro_atratividade.csv     atributos, pesos e fontes
    results/tabelas/sensibilidade_ordem_pesos.csv   leitura (A) contra (B)

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/04_atratividade.py
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
from scipy import stats
from shapely.geometry import box

# =============================================================================
# PARAMETROS
# =============================================================================

RAIZ = Path(__file__).resolve().parents[1]
DIR_TRATADOS = RAIZ / "data" / "tratados"
DIR_TABELAS = RAIZ / "results" / "tabelas"

ARQ_CANDIDATOS = DIR_TRATADOS / "candidatos.gpkg"
ARQ_BAIRROS = DIR_TRATADOS / "bairros.gpkg"
ARQ_CNEFE = RAIZ / "data" / "brutos" / "cnefe" / "4205407_FLORIANOPOLIS.csv"
ARQ_SAIDA = DIR_TRATADOS / "candidatos_com_aj.gpkg"

CRS_TRABALHO = 31982
N_CANDIDATOS_ESPERADO = 856

# -----------------------------------------------------------------------------
# Pesos Rank-Order Centroid
# -----------------------------------------------------------------------------
# Baia, Silva e Filenga (2025) ordenam TRES atributos — seguranca, acessibilidade
# e disponibilidade — e a Secao 3.2.4 define os pesos "segundo a ordem de
# importancia relatada" por esses autores. O tipo de estabelecimento vem de
# Firmeza (2021), fora dessa hierarquia, e por isso ocupa a quarta posicao.
#
# 'A' e a leitura literal da Secao 3.2.4 e e a principal.
# 'B' inverte a hierarquia colocando o tipo em primeiro; roda como sensibilidade,
#     porque o formulacao.md listava os atributos nessa ordem antes da correcao.
# Os pesos ROC sao calculados pela formula, nao digitados: os valores publicados
# na monografia — 0,5208 / 0,2708 / 0,1458 / 0,0625 — sao o arredondamento em
# quatro casas e somam 0,9999, nao 1. Usar os arredondados faria a media
# ponderada perder 0,01% do total e quebraria a verificacao de que os pesos
# somam 1. O script usa os exatos e confere que arredondam para os publicados.
#
#     w_k = (1/K) * SOMA_{i=k..K} (1/i)      Rank-Order Centroid, Barron & Barrett
PESOS_ROC_PUBLICADOS = (0.5208, 0.2708, 0.1458, 0.0625)


def _pesos_roc(k: int) -> tuple[float, ...]:
    return tuple(sum(1.0 / i for i in range(pos, k + 1)) / k for pos in range(1, k + 1))


PESOS_ROC = _pesos_roc(4)
ORDENS_ATRIBUTOS = {
    "A": ["seguranca", "acessibilidade", "disponibilidade", "tipo"],
    "B": ["tipo", "seguranca", "acessibilidade", "disponibilidade"],
}
ORDEM_PRINCIPAL = "A"
ORDEM_SENSIBILIDADE = "B"

# Piso obrigatorio dos escores. Sem ele, d_ij / a_j diverge quando a_j -> 0.
PISO_ESCORE = 0.2

# -----------------------------------------------------------------------------
# Atributo TIPO — Firmeza (2021), Figura 23, p. 62
# -----------------------------------------------------------------------------
# Pergunta 3.1, resposta multipla, percentual de citacao sobre o total de
# observacoes: Shoppings 78% · Supermercados 63% · Farmacias 40% · Postos de
# gasolina 37% · Lojas 23% · Academias 21% · Terminais de onibus 12% ·
# Padarias 8% · Casas lotericas 2%.
#
# A hierarquia dos tipos NAO e a de Guarino Neto e Vidal Vieira (2023): o nucleo
# daquele artigo e um modelo de equacoes estruturais sobre intencao comportamental
# (monografia corrigida na v15). A estatistica descritiva da sua Tabela A2 e usada,
# a partir da v36, SO para posicionar a agencia postal — ver adiante.
# A monografia (v16, Secao 3.2.4) declara a formula de normalizacao:
#
#     s = 0,2 + 0,8 * (p / p_max)
#
# em que p e o percentual de citacao do tipo em Firmeza (2021) e p_max o do tipo
# mais citado ENTRE OS CONSIDERADOS — supermercados, com 63%. Shoppings, com 78%,
# nao entram: nao sao tipo de candidato deste modelo.
#
# Os escores sao CALCULADOS pela formula, nao digitados, para que o valor do
# arquivo e o do texto nao possam divergir.
PERCENTUAL_CITACAO_FIRMEZA = {
    "supermercado": 63.0,
    "farmacia": 40.0,
    "conveniencia": 37.0,      # por analogia com postos de gasolina
    "mercado_pequeno": 8.0,    # por analogia com padarias
}

# Agencias postais nao constam da pergunta 3.1 de Firmeza (2021), entao nao ha
# percentual de citacao DAQUELA pesquisa do qual derivar o valor. Ate a v35 o
# escore era arbitrado em 0,45. A partir da v36 ele e derivado de:
#
#     Guarino Neto, L.; Vidal Vieira, J. G. (2023). An investigation of consumer
#     intention to use pick-up point services for last-mile distribution in a
#     developing country. Journal of Retailing and Consumer Services, 74, 103425.
#     Tabela A2 (p. 13), "Preferred pick-up point location", coluna All Groups:
#     Drugstore 65% · Supermarket 62% · Post office 49% · Mall 39% · Gas station 38%.
#
# O supermercado e o tipo comum as duas pesquisas e ja e a referencia (escore 1)
# da normalizacao de Firmeza. A agencia postal e posicionada em relacao a ele:
#
#     s = 0,2 + 0,8 * (49 / 62) = 0,8323
#
# So a RAZAO agencia/supermercado vem de Guarino Neto e Vidal Vieira; a hierarquia
# dos demais tipos continua sendo a de Firmeza (2021). As duas pesquisas sao de
# cidades diferentes (Sao Paulo e Fortaleza) e ordenam farmacia e supermercado de
# modo distinto — por isso o escore e PARAMETRO DE CONFIGURACAO, alteravel na
# linha de comando (--escore-postal) e submetido a analise de sensibilidade em
# src/10_sensibilidade_escore_postal.py.
PCT_GUARINO_AGENCIA_POSTAL = 49.0
PCT_GUARINO_SUPERMERCADO = 62.0
ESCORE_POSTAL_ARBITRADO = 0.45          # valor da monografia ate a v35
ESCORE_POSTAL_DERIVADO = PISO_ESCORE + (1.0 - PISO_ESCORE) * (
    PCT_GUARINO_AGENCIA_POSTAL / PCT_GUARINO_SUPERMERCADO
)
ESCORE_AGENCIA_POSTAL = ESCORE_POSTAL_DERIVADO
ORIGEM_ESCORE_POSTAL = (
    "derivado, por razao com supermercados",
    "Guarino Neto e Vidal Vieira (2023), Tabela A2: agencia postal 49%, supermercado 62%",
)


def _escores_tipo(escore_postal: float) -> dict[str, float]:
    p_max = max(PERCENTUAL_CITACAO_FIRMEZA.values())
    derivados = {
        tipo: PISO_ESCORE + (1.0 - PISO_ESCORE) * (pct / p_max)
        for tipo, pct in PERCENTUAL_CITACAO_FIRMEZA.items()
    }
    return {**derivados, "agencia_postal": escore_postal}


ESCORE_TIPO = _escores_tipo(ESCORE_AGENCIA_POSTAL)
ANCORA_TIPO = {
    "supermercado": ("derivado", "Supermercados, 63% de citacao"),
    "farmacia": ("derivado", "Farmacias, 40% de citacao"),
    "conveniencia": ("derivado, por analogia", "Postos de gasolina, 37% de citacao"),
    "agencia_postal": ORIGEM_ESCORE_POSTAL,
    "mercado_pequeno": ("derivado, por analogia", "Padarias, 8% de citacao"),
}


def definir_escore_postal(valor: float) -> None:
    """Troca o escore das agencias postais (linha de comando ou sensibilidade)."""
    global ESCORE_AGENCIA_POSTAL, ESCORE_TIPO
    checar(PISO_ESCORE <= valor <= 1.0, f"Escore postal {valor} fora de [{PISO_ESCORE}; 1].")
    ESCORE_AGENCIA_POSTAL = float(valor)
    ESCORE_TIPO = _escores_tipo(ESCORE_AGENCIA_POSTAL)
    if np.isclose(valor, ESCORE_POSTAL_DERIVADO):
        ANCORA_TIPO["agencia_postal"] = ORIGEM_ESCORE_POSTAL
    elif np.isclose(valor, ESCORE_POSTAL_ARBITRADO):
        ANCORA_TIPO["agencia_postal"] = ("ARBITRADO", "valor da monografia ate a v35, sem ancoragem empirica")
    else:
        ANCORA_TIPO["agencia_postal"] = (
            "parametro informado", f"--escore-postal {valor:.4f}, valor de sensibilidade"
        )

# -----------------------------------------------------------------------------
# Atributo DISPONIBILIDADE — valor declarado por tipo
# -----------------------------------------------------------------------------
# O atributo NAO e medido candidato a candidato: e PARAMETRIZADO POR TIPO. Das
# 856 candidaturas, apenas 84 (9,8%) tem opening_hours no OSM, e nenhuma das 496
# de fonte CNEFE tem. Imputar 90% dos valores por uma mediana global faria o
# atributo nao discriminar nada.
#
# Os quatro primeiros valores sao a MEDIANA OBSERVADA do tipo, de modo que os 84
# registros informam os parametros em vez de serem descartados. O quinto e
# hipotese isolada num unico numero, que pode ser testado. As horas semanais vem
# de src/horarios_osm.py, que documenta como trata feriados, 24/7, intervalos e
# regras condicionais.
#
# Justificativa empirica da estratificacao por tipo, calculada sobre os 84
# observados do conjunto final (docs/decisoes_etapa12.md): Kruskal-Wallis
# H = 42,41, p = 3,3 x 10^-9; pos-teste de Dunn com correcao de Holm, 4 dos 6
# pares significativos a 5%. O valor anterior (H = 42,6 sobre 85 observacoes,
# "5 de 6 pares") era do conjunto de 872 candidatos e usava Mann-Whitney par a
# par SEM correcao para comparacoes multiplas.
#
# O script RECALCULA o teste a partir de HORARIO e falha se as medianas, as
# contagens ou as constantes abaixo divergirem dos dados.
HORAS_SEMANAIS_POR_TIPO = {
    "agencia_postal": 40.0,
    "mercado_pequeno": 60.0,
    "conveniencia": 70.0,
    "supermercado": 98.0,
    "farmacia": 107.5,
}
ANCORA_HORAS = {
    "agencia_postal": ("observado", "mediana de 15 observacoes"),
    "mercado_pequeno": ("declarado", "zero observacoes; abaixo de conveniencia, "
                                     "comercio de vizinhanca com horario reduzido"),
    "conveniencia": ("observado", "mediana de 15 observacoes"),
    "supermercado": ("observado", "mediana de 34 observacoes"),
    "farmacia": ("observado", "mediana de 20 observacoes"),
}
N_OBSERVACOES_HORARIO = 84
KRUSKAL_H = 42.41                 # arredondado a 2 casas; conferido contra o recalculo
KRUSKAL_P = 3.3e-9                # idem, a 2 algarismos significativos
POS_TESTE = "Dunn (1964) com correcao de Holm-Bonferroni"
PARES_SIGNIFICATIVOS = 4          # de 6, ao nivel de 5%, apos a correcao

# -----------------------------------------------------------------------------
# Atributos espaciais — acessibilidade e seguranca
# -----------------------------------------------------------------------------
RAIO_VIZINHANCA_M = 400.0     # raio de analise de entorno (passo a passo)
LIMITE_PARADA_M = 800.0       # distancia a parada acima da qual o escore zera
LIMITE_VIA_M = 400.0          # idem, para via de hierarquia superior
PERCENTIL_CAP = 95            # teto das contagens, para o Centro nao dominar

# Composicao de cada atributo espacial: pesos internos declarados, iguais por
# ausencia de base para hierarquiza-los.
PESO_ACESSIBILIDADE = {"parada_onibus": 0.5, "densidade_caminhavel": 0.5}
PESO_SEGURANCA = {"hierarquia_viaria": 0.5, "densidade_comercial": 0.5}

CLASSES_VIA_PRINCIPAL = ["motorway", "trunk", "primary", "secondary", "tertiary"]

# -----------------------------------------------------------------------------
# Cache das extracoes do OSM
# -----------------------------------------------------------------------------
FORCAR_DOWNLOAD_OSM = False
ARQ_CACHE_PARADAS = DIR_TRATADOS / "_cache_osm_paradas.parquet"
ARQ_CACHE_VIAS = DIR_TRATADOS / "_cache_osm_vias.parquet"
ARQ_CACHE_GRAFO = DIR_TRATADOS / "_cache_osm_rede_caminhavel.graphml"
ARQ_METADADOS = DIR_TRATADOS / "_cache_osm_etapa5_metadados.json"

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


def normalizar(valores: np.ndarray, piso: float = PISO_ESCORE, cap: int | None = None) -> np.ndarray:
    """Min-max reescalado para [piso, 1].

    O piso entra aqui, em cada escore de atributo: como a_j e media ponderada de
    escores todos >= piso, o proprio a_j fica >= piso por construcao.

    `cap` limita o valor maximo ao percentil indicado antes de normalizar, para
    que uma contagem muito alta numa unica area nao comprima todo o resto.
    """
    v = np.asarray(valores, dtype=float)
    if cap is not None:
        v = np.minimum(v, np.nanpercentile(v, cap))
    lo, hi = np.nanmin(v), np.nanmax(v)
    if np.isclose(hi, lo):
        return np.full_like(v, 1.0)
    return piso + (1.0 - piso) * (v - lo) / (hi - lo)


# =============================================================================
# DADOS
# =============================================================================


def carregar_candidatos() -> gpd.GeoDataFrame:
    subtitulo("1. Candidatos")
    checar(
        ARQ_CANDIDATOS.exists(),
        f"{ARQ_CANDIDATOS} nao existe. Rodar antes: python src/03_candidatos.py",
    )
    candidatos = gpd.read_file(ARQ_CANDIDATOS).to_crs(CRS_TRABALHO)
    print(f"candidatos lidos ....................... {len(candidatos)}")
    print(f"CRS .................................... EPSG:{candidatos.crs.to_epsg()}")
    checar(
        len(candidatos) == N_CANDIDATOS_ESPERADO,
        f"Foram lidos {len(candidatos)} candidatos, esperado {N_CANDIDATOS_ESPERADO}.",
    )
    faltando = set(candidatos["TIPO"]) - set(ESCORE_TIPO)
    checar(not faltando, f"Tipos sem escore declarado: {sorted(faltando)}.")
    faltando = set(candidatos["TIPO"]) - set(HORAS_SEMANAIS_POR_TIPO)
    checar(not faltando, f"Tipos sem horario declarado: {sorted(faltando)}.")
    for tipo, n in candidatos["TIPO"].value_counts().items():
        print(f"   {tipo:<18} {n:>5}")
    return candidatos


def extrair_osm(terra) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.MultiDiGraph]:
    """Paradas de onibus, vias de hierarquia superior e rede caminhavel.

    Mesmo criterio de cache da Etapa 4: o OSM muda continuamente, e sem o cache
    os numeros da monografia nao seriam reproduziveis.
    """
    subtitulo("2. Extracao do OpenStreetMap")

    tem_cache = (
        ARQ_CACHE_PARADAS.exists() and ARQ_CACHE_VIAS.exists() and ARQ_CACHE_GRAFO.exists()
    )
    if tem_cache and not FORCAR_DOWNLOAD_OSM:
        paradas = gpd.read_parquet(ARQ_CACHE_PARADAS)
        vias = gpd.read_parquet(ARQ_CACHE_VIAS)
        grafo = ox.load_graphml(ARQ_CACHE_GRAFO)
        metadados = json.loads(ARQ_METADADOS.read_text(encoding="utf-8"))
        print(f"lido do cache .......................... extracao de {metadados['extraido_em']}")
        print(f"paradas de onibus ...................... {len(paradas)}")
        print(f"vias de hierarquia superior ............ {len(vias)}")
        print(f"rede caminhavel ........................ {grafo.number_of_nodes()} nos")
        print("   (FORCAR_DOWNLOAD_OSM = True para reextrair)")
        return paradas, vias, grafo

    terra_4326 = gpd.GeoSeries([terra], crs=CRS_TRABALHO).to_crs(4326).iloc[0]
    caixa = box(*terra_4326.bounds)

    print("consultando paradas de onibus...")
    paradas = ox.features_from_polygon(caixa, {"highway": "bus_stop"}).reset_index()
    paradas = gpd.GeoDataFrame(paradas[["geometry"]], geometry="geometry", crs=paradas.crs)
    print(f"   {len(paradas)} paradas")

    print("consultando vias de hierarquia superior...")
    vias = ox.features_from_polygon(caixa, {"highway": CLASSES_VIA_PRINCIPAL}).reset_index()
    vias = vias[vias.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    vias = gpd.GeoDataFrame(vias[["highway", "geometry"]], geometry="geometry", crs=vias.crs)
    print(f"   {len(vias)} vias")

    print("baixando a rede caminhavel (pode levar cerca de um minuto)...")
    grafo = ox.graph_from_polygon(terra_4326, network_type="walk", simplify=True)
    print(f"   {grafo.number_of_nodes()} nos, {grafo.number_of_edges()} arestas")

    DIR_TRATADOS.mkdir(parents=True, exist_ok=True)
    paradas.to_parquet(ARQ_CACHE_PARADAS, index=False)
    vias.to_parquet(ARQ_CACHE_VIAS, index=False)
    ox.save_graphml(grafo, ARQ_CACHE_GRAFO)
    ARQ_METADADOS.write_text(
        json.dumps(
            {
                "extraido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "osmnx": ox.__version__,
                "paradas": int(len(paradas)),
                "vias": int(len(vias)),
                "nos_rede_caminhavel": int(grafo.number_of_nodes()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("cache gravado.")
    return paradas, vias, grafo


def carregar_estabelecimentos_cnefe() -> gpd.GeoDataFrame:
    """Todos os estabelecimentos do CNEFE (especie 6), proxy de densidade comercial."""
    cnefe = pd.read_csv(
        ARQ_CNEFE, sep=";", encoding="latin-1", dtype=str,
        usecols=["COD_ESPECIE", "LATITUDE", "LONGITUDE"],
    )
    e6 = cnefe[cnefe["COD_ESPECIE"] == "6"]
    return gpd.GeoDataFrame(
        e6[[]],
        geometry=gpd.points_from_xy(e6["LONGITUDE"].astype(float), e6["LATITUDE"].astype(float)),
        crs=4326,
    ).to_crs(CRS_TRABALHO)


# =============================================================================
# ESCORES
# =============================================================================


def contar_no_raio(
    candidatos: gpd.GeoDataFrame, pontos: gpd.GeoDataFrame, raio: float
) -> np.ndarray:
    """Quantos `pontos` caem a menos de `raio` de cada candidato."""
    vizinhanca = gpd.GeoDataFrame(
        {"ID": range(len(candidatos))}, geometry=candidatos.geometry.buffer(raio), crs=candidatos.crs
    )
    juncao = gpd.sjoin(pontos[["geometry"]], vizinhanca, how="inner", predicate="within")
    contagem = juncao.groupby("ID").size()
    return contagem.reindex(range(len(candidatos))).fillna(0).to_numpy(dtype=float)


def escore_tipo(candidatos: gpd.GeoDataFrame) -> np.ndarray:
    subtitulo("3. Escore de TIPO — Firmeza (2021); agencia postal por Guarino Neto e Vidal Vieira (2023)")
    escores = candidatos["TIPO"].map(ESCORE_TIPO).to_numpy(dtype=float)
    print(f"   {'tipo':<18} {'escore':>7} {'n':>6}  origem")
    for tipo in sorted(ESCORE_TIPO, key=lambda t: -ESCORE_TIPO[t]):
        n = int((candidatos["TIPO"] == tipo).sum())
        origem, ancora = ANCORA_TIPO[tipo]
        print(f"   {tipo:<18} {ESCORE_TIPO[tipo]:>7.4f} {n:>6}  {origem}: {ancora}")
    checar(
        bool(((escores >= PISO_ESCORE) & (escores <= 1.0)).all()),
        f"Escore de tipo fora de [{PISO_ESCORE}; 1].",
    )
    return escores


def verificar_disponibilidade(candidatos: gpd.GeoDataFrame) -> None:
    """Recalcula medianas e Kruskal-Wallis a partir de HORARIO e confere as constantes.

    As horas por tipo sao parametros digitados; esta verificacao garante que eles
    continuam sendo a mediana observada e que o teste citado no texto e o que os
    dados dao. Se o conjunto de candidatos mudar, o script para aqui.
    """
    import horarios_osm

    horarios_osm.conferir_casos()
    obs = candidatos[candidatos["HORARIO"].notna()].copy()
    checar(len(obs) == N_OBSERVACOES_HORARIO,
           f"{len(obs)} candidatos com opening_hours, esperado {N_OBSERVACOES_HORARIO}.")
    obs["HORAS"] = obs["HORARIO"].map(horarios_osm.horas_semanais)
    por_tipo = {t: sub["HORAS"].tolist() for t, sub in obs.groupby("TIPO")}
    for tipo, horas in por_tipo.items():
        origem, ancora = ANCORA_HORAS[tipo]
        checar(origem == "observado", f"{tipo} tem observacoes, mas esta declarado como '{origem}'.")
        checar(float(np.median(horas)) == HORAS_SEMANAIS_POR_TIPO[tipo],
               f"{tipo}: mediana observada {np.median(horas)} h, parametro {HORAS_SEMANAIS_POR_TIPO[tipo]} h.")
        checar(f"mediana de {len(horas)} observacoes" == ancora,
               f"{tipo}: {len(horas)} observacoes, mas a origem diz '{ancora}'.")
    teste = horarios_osm.analisar(por_tipo, "Kruskal-Wallis sobre as observacoes de opening_hours")
    checar(abs(teste["H"] - KRUSKAL_H) < 0.005, f"H recalculado {teste['H']:.4f}, constante {KRUSKAL_H}.")
    checar(abs(teste["p"] - KRUSKAL_P) / KRUSKAL_P < 0.05, f"p recalculado {teste['p']:.3e}, constante {KRUSKAL_P:g}.")
    significativos = sum(par["dunn_holm_p"] < 0.05 for par in teste["pares"])
    checar(significativos == PARES_SIGNIFICATIVOS,
           f"{significativos} pares significativos pelo pos-teste, constante {PARES_SIGNIFICATIVOS}.")


def escore_disponibilidade(candidatos: gpd.GeoDataFrame) -> np.ndarray:
    subtitulo("4. Escore de DISPONIBILIDADE — valor declarado por tipo")
    verificar_disponibilidade(candidatos)
    horas = candidatos["TIPO"].map(HORAS_SEMANAIS_POR_TIPO).to_numpy(dtype=float)
    escores = normalizar(horas)

    print(f"\n   justificativa da estratificacao: Kruskal-Wallis H = {KRUSKAL_H:.2f}, "
          f"p = {KRUSKAL_P:.1e}, sobre {N_OBSERVACOES_HORARIO} observacoes; "
          f"{POS_TESTE}: {PARES_SIGNIFICATIVOS} de 6 pares significativos")
    print(f"\n   {'tipo':<18} {'h/semana':>9} {'escore':>7} {'n':>6}  origem")
    for tipo in sorted(HORAS_SEMANAIS_POR_TIPO, key=lambda t: HORAS_SEMANAIS_POR_TIPO[t]):
        n = int((candidatos["TIPO"] == tipo).sum())
        origem, ancora = ANCORA_HORAS[tipo]
        valor = HORAS_SEMANAIS_POR_TIPO[tipo]
        escore = float(normalizar(np.array(list(HORAS_SEMANAIS_POR_TIPO.values())))[
            list(HORAS_SEMANAIS_POR_TIPO).index(tipo)
        ])
        print(f"   {tipo:<18} {valor:>9.1f} {escore:>7.3f} {n:>6}  {origem}: {ancora}")
    print("\n   O atributo NAO e medido candidato a candidato: e parametrizado por tipo.")
    return escores


def escore_acessibilidade(
    candidatos: gpd.GeoDataFrame, paradas: gpd.GeoDataFrame, grafo: nx.MultiDiGraph
) -> tuple[np.ndarray, pd.DataFrame]:
    """Proximidade a parada de onibus e densidade caminhavel em 400 m."""
    subtitulo("5. Escore de ACESSIBILIDADE")

    paradas = paradas.to_crs(CRS_TRABALHO)
    vizinho = gpd.sjoin_nearest(
        candidatos[["geometry"]], paradas[["geometry"]], how="left", distance_col="DIST_M"
    )
    vizinho = vizinho[~vizinho.index.duplicated(keep="first")]
    dist_parada = vizinho["DIST_M"].to_numpy(dtype=float)
    # Escore decrescente com a distancia, zerado acima do limite declarado.
    bruto_parada = 1.0 - np.minimum(dist_parada, LIMITE_PARADA_M) / LIMITE_PARADA_M

    # Densidade caminhavel: cruzamentos da rede a pe (nos de grau >= 3) em 400 m.
    nao_direcionado = ox.convert.to_undirected(grafo)
    nos = ox.graph_to_gdfs(grafo, edges=False).to_crs(CRS_TRABALHO)
    graus = dict(nao_direcionado.degree())
    nos["GRAU"] = nos.index.map(graus)
    cruzamentos = nos[nos["GRAU"] >= 3]
    n_cruzamentos = contar_no_raio(candidatos, cruzamentos, RAIO_VIZINHANCA_M)

    s_parada = normalizar(bruto_parada)
    s_densidade = normalizar(n_cruzamentos, cap=PERCENTIL_CAP)
    escores = (
        PESO_ACESSIBILIDADE["parada_onibus"] * s_parada
        + PESO_ACESSIBILIDADE["densidade_caminhavel"] * s_densidade
    )

    print(f"paradas de onibus ...................... {len(paradas)}")
    print(f"cruzamentos da rede caminhavel ......... {len(cruzamentos)} (de {len(nos)} nos)")
    print(f"distancia a parada ..................... mediana {np.median(dist_parada):.0f} m · "
          f"max {dist_parada.max():.0f} m")
    print(f"candidatos a mais de {LIMITE_PARADA_M:.0f} m de parada .. {int((dist_parada > LIMITE_PARADA_M).sum())}")
    print(f"cruzamentos em {RAIO_VIZINHANCA_M:.0f} m ................. mediana {np.median(n_cruzamentos):.0f} · "
          f"max {n_cruzamentos.max():.0f}")
    print(f"composicao ............................. " + " + ".join(
        f"{v:.0%} {k}" for k, v in PESO_ACESSIBILIDADE.items()
    ))
    detalhe = pd.DataFrame(
        {"DIST_PARADA_M": dist_parada, "N_CRUZAMENTOS_400M": n_cruzamentos,
         "S_PARADA": s_parada, "S_DENS_CAMINHAVEL": s_densidade}
    )
    return escores, detalhe


def escore_seguranca(
    candidatos: gpd.GeoDataFrame, vias: gpd.GeoDataFrame, estabelecimentos: gpd.GeoDataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Proxy por hierarquia viaria e densidade comercial no entorno.

    Nao existe base publica georreferenciada de ocorrencias em nivel de endereco
    para o municipio. A proxy assume que via de hierarquia superior e entorno
    comercialmente denso implicam maior movimento e vigilancia natural — hipotese
    plausivel, mas NAO verificada, e declarada como limitacao.
    """
    subtitulo("6. Escore de SEGURANCA (proxy)")

    vias = vias.to_crs(CRS_TRABALHO)
    vizinho = gpd.sjoin_nearest(
        candidatos[["geometry"]], vias[["geometry"]], how="left", distance_col="DIST_M"
    )
    vizinho = vizinho[~vizinho.index.duplicated(keep="first")]
    dist_via = vizinho["DIST_M"].to_numpy(dtype=float)
    bruto_via = 1.0 - np.minimum(dist_via, LIMITE_VIA_M) / LIMITE_VIA_M

    n_estab = contar_no_raio(candidatos, estabelecimentos, RAIO_VIZINHANCA_M)

    s_via = normalizar(bruto_via)
    s_comercio = normalizar(n_estab, cap=PERCENTIL_CAP)
    escores = (
        PESO_SEGURANCA["hierarquia_viaria"] * s_via
        + PESO_SEGURANCA["densidade_comercial"] * s_comercio
    )

    print(f"vias de hierarquia superior ............ {len(vias)}")
    print(f"estabelecimentos do CNEFE (especie 6) .. {numero_br(len(estabelecimentos))}")
    print(f"distancia a via principal .............. mediana {np.median(dist_via):.0f} m · "
          f"max {dist_via.max():.0f} m")
    print(f"candidatos a mais de {LIMITE_VIA_M:.0f} m de via ...... {int((dist_via > LIMITE_VIA_M).sum())}")
    print(f"estabelecimentos em {RAIO_VIZINHANCA_M:.0f} m ............ mediana {np.median(n_estab):.0f} · "
          f"max {n_estab.max():.0f}")
    print(f"composicao ............................. " + " + ".join(
        f"{v:.0%} {k}" for k, v in PESO_SEGURANCA.items()
    ))
    print("\n   LIMITACAO: proxy sem base publica de ocorrencias. Sob a ordem (A), este")
    print("   e o atributo de MAIOR peso do indice — a fragilidade esta na medicao,")
    print("   nao na ponderacao, que reflete o que Baia, Silva e Filenga (2025) mediram.")
    detalhe = pd.DataFrame(
        {"DIST_VIA_M": dist_via, "N_ESTAB_400M": n_estab,
         "S_HIERARQUIA_VIARIA": s_via, "S_DENS_COMERCIAL": s_comercio}
    )
    return escores, detalhe


# =============================================================================
# INDICE
# =============================================================================


def pesos_da_ordem(ordem: str) -> dict[str, float]:
    """Associa cada atributo ao seu peso ROC, conforme a ordem escolhida."""
    checar(ordem in ORDENS_ATRIBUTOS, f"Ordem de pesos desconhecida: {ordem}.")
    return dict(zip(ORDENS_ATRIBUTOS[ordem], PESOS_ROC))


def calcular_aj(escores: dict[str, np.ndarray], ordem: str) -> np.ndarray:
    pesos = pesos_da_ordem(ordem)
    checar(
        np.isclose(sum(pesos.values()), 1.0),
        f"Os pesos somam {sum(pesos.values())}, deveriam somar 1.",
    )
    aj = np.zeros(len(next(iter(escores.values()))))
    for atributo, peso in pesos.items():
        aj += peso * escores[atributo]
    return aj


def montar_indice(escores: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    subtitulo("7. Indice de atratividade")

    aj_a = calcular_aj(escores, ORDEM_PRINCIPAL)
    aj_b = calcular_aj(escores, ORDEM_SENSIBILIDADE)

    for rotulo, ordem, aj in [
        (f"({ORDEM_PRINCIPAL}) principal", ORDEM_PRINCIPAL, aj_a),
        (f"({ORDEM_SENSIBILIDADE}) sensibilidade", ORDEM_SENSIBILIDADE, aj_b),
    ]:
        pesos = pesos_da_ordem(ordem)
        print(f"\n   {rotulo}: " + " · ".join(f"{k} {v:.4f}" for k, v in pesos.items()))
        print(f"      a_j: min {aj.min():.4f} · mediana {np.median(aj):.4f} · "
              f"max {aj.max():.4f} · dp {aj.std():.4f}")
        print(f"      d/a maximo: {1 / aj.min():.2f}x a distancia fisica")
        # Peso EFETIVO do tipo: o quanto o atributo consegue mover o a_j. E o peso
        # ROC vezes a amplitude dos escores de tipo; a diferenca entre a agencia
        # postal e o supermercado, multiplicada pelo peso, e o que separa os dois
        # tipos no indice, com os demais atributos iguais.
        amplitude = max(ESCORE_TIPO.values()) - min(ESCORE_TIPO.values())
        print(f"      peso efetivo do tipo: peso ROC {pesos['tipo']:.4f} x amplitude dos "
              f"escores {amplitude:.4f} = {pesos['tipo'] * amplitude:.4f} de a_j")
        print(f"      agencia postal x supermercado: {pesos['tipo']:.4f} x "
              f"({ESCORE_TIPO['agencia_postal']:.4f} - {ESCORE_TIPO['supermercado']:.4f}) = "
              f"{pesos['tipo'] * (ESCORE_TIPO['agencia_postal'] - ESCORE_TIPO['supermercado']):+.4f} de a_j")

    return aj_a, aj_b


def sensibilidade_ordem(
    candidatos: gpd.GeoDataFrame, aj_a: np.ndarray, aj_b: np.ndarray
) -> pd.DataFrame:
    """Quantos candidatos mudam de posicao no ranking entre as leituras (A) e (B)."""
    subtitulo("8. Sensibilidade a ordem dos pesos")

    rank_a = pd.Series(aj_a).rank(ascending=False, method="first").astype(int)
    rank_b = pd.Series(aj_b).rank(ascending=False, method="first").astype(int)
    delta = (rank_a - rank_b).abs()

    rho, pval = stats.spearmanr(aj_a, aj_b)
    print(f"correlacao de Spearman entre a_j (A) e (B) .. {rho:.4f}  (p = {pval:.3g})")
    print(f"candidatos que mudam de posicao ............. {int((delta > 0).sum())} de {len(delta)} "
          f"({100 * (delta > 0).mean():.1f}%)")
    for limite in (10, 50, 100, 200):
        print(f"   mudam mais de {limite:>3} posicoes ................ {int((delta > limite).sum())}")
    print(f"deslocamento mediano ........................ {delta.median():.0f} posicoes")
    print(f"deslocamento maximo ......................... {delta.max():.0f} posicoes")

    print("\n   Top 10 por a_j em cada leitura:")
    print(f"   {'#':>3}  {'(A) principal':<44}  {'(B) sensibilidade':<44}")
    for i in range(10):
        ia = int(rank_a[rank_a == i + 1].index[0])
        ib = int(rank_b[rank_b == i + 1].index[0])
        a_txt = f"{(candidatos['NOME'].iloc[ia] or '(sem nome)')[:26]:<26} {candidatos['TIPO'].iloc[ia][:13]:<13} {aj_a[ia]:.3f}"
        b_txt = f"{(candidatos['NOME'].iloc[ib] or '(sem nome)')[:26]:<26} {candidatos['TIPO'].iloc[ib][:13]:<13} {aj_b[ib]:.3f}"
        print(f"   {i + 1:>3}  {a_txt:<44}  {b_txt:<44}")

    print("\n   por tipo, a_j medio em cada leitura:")
    comparacao = pd.DataFrame(
        {"TIPO": candidatos["TIPO"].to_numpy(), "AJ_A": aj_a, "AJ_B": aj_b}
    )
    resumo = comparacao.groupby("TIPO").agg(
        n=("AJ_A", "size"), aj_A=("AJ_A", "mean"), aj_B=("AJ_B", "mean"),
        aj_A_min=("AJ_A", "min"), aj_A_mediana=("AJ_A", "median"), aj_A_max=("AJ_A", "max"),
        aj_B_min=("AJ_B", "min"), aj_B_mediana=("AJ_B", "median"), aj_B_max=("AJ_B", "max"),
    )
    resumo["diferenca"] = resumo["aj_B"] - resumo["aj_A"]
    # Linha do conjunto: o Quadro 12 traz minimo, mediana, maximo e media gerais.
    resumo.loc["(todos)"] = {
        "n": len(comparacao), "aj_A": aj_a.mean(), "aj_B": aj_b.mean(),
        "aj_A_min": aj_a.min(), "aj_A_mediana": np.median(aj_a), "aj_A_max": aj_a.max(),
        "aj_B_min": aj_b.min(), "aj_B_mediana": np.median(aj_b), "aj_B_max": aj_b.max(),
        "diferenca": aj_b.mean() - aj_a.mean(),
    }
    print(f"   {'tipo':<18} {'n':>6} {'a_j (A)':>9} {'a_j (B)':>9} {'B - A':>9}")
    for tipo, linha in resumo.sort_values("aj_A", ascending=False).iterrows():
        print(f"   {tipo:<18} {int(linha['n']):>6} {linha['aj_A']:>9.4f} "
              f"{linha['aj_B']:>9.4f} {linha['diferenca']:>+9.4f}")

    return resumo.reset_index().round(4)


def verificar(escores: dict[str, np.ndarray], aj_a: np.ndarray, aj_b: np.ndarray) -> None:
    subtitulo("9. Verificacoes")

    for atributo, valores in escores.items():
        checar(
            bool(np.isfinite(valores).all()),
            f"Escore de {atributo} tem valor nao finito.",
        )
        checar(
            bool(((valores >= PISO_ESCORE - 1e-9) & (valores <= 1.0 + 1e-9)).all()),
            f"Escore de {atributo} fora de [{PISO_ESCORE}; 1]: "
            f"min {valores.min():.4f}, max {valores.max():.4f}.",
        )
    print(f"escores dos 4 atributos em [{PISO_ESCORE}; 1] ..... conferido")

    for rotulo, aj in [(ORDEM_PRINCIPAL, aj_a), (ORDEM_SENSIBILIDADE, aj_b)]:
        checar(
            bool(((aj >= PISO_ESCORE - 1e-9) & (aj <= 1.0 + 1e-9)).all()),
            f"a_j da ordem ({rotulo}) fora de [{PISO_ESCORE}; 1]: "
            f"min {aj.min():.4f}, max {aj.max():.4f}.",
        )
        checar(bool((aj > 0).all()), f"a_j nulo ou negativo na ordem ({rotulo}).")
    print(f"a_j em [{PISO_ESCORE}; 1] nas duas ordens ......... conferido")
    print(f"a_j > 0, d/a finito ....................... conferido")

    checar(
        all(
            np.isclose(round(exato, 4), publicado)
            for exato, publicado in zip(PESOS_ROC, PESOS_ROC_PUBLICADOS)
        ),
        f"Os pesos ROC calculados {tuple(round(p, 4) for p in PESOS_ROC)} nao "
        f"correspondem aos publicados na monografia {PESOS_ROC_PUBLICADOS}.",
    )
    print(f"pesos ROC batem com os publicados .......... conferido "
          f"({', '.join(f'{p:.4f}' for p in PESOS_ROC)})")

    for ordem in ORDENS_ATRIBUTOS:
        pesos = pesos_da_ordem(ordem)
        checar(
            np.isclose(sum(pesos.values()), 1.0),
            f"Pesos da ordem ({ordem}) nao somam 1.",
        )
        checar(
            set(pesos) == set(escores),
            f"Ordem ({ordem}) nao cobre exatamente os atributos calculados.",
        )
    print("pesos somam 1 e cobrem os 4 atributos ..... conferido")


# =============================================================================
# SAIDAS
# =============================================================================


def quadro_de_atratividade(candidatos: gpd.GeoDataFrame) -> pd.DataFrame:
    """Quadro de atributos, escores, pesos e fontes."""
    pesos_a = pesos_da_ordem(ORDEM_PRINCIPAL)
    pesos_b = pesos_da_ordem(ORDEM_SENSIBILIDADE)
    linhas = []

    for tipo in sorted(ESCORE_TIPO, key=lambda t: -ESCORE_TIPO[t]):
        origem, ancora = ANCORA_TIPO[tipo]
        linhas.append(
            {
                "atributo": "tipo",
                "categoria": tipo,
                "valor_bruto": (f"{numero_br(PERCENTUAL_CITACAO_FIRMEZA[tipo], 0)}% de citacao"
                                if tipo in PERCENTUAL_CITACAO_FIRMEZA else "sem percentual"),
                "escore": round(ESCORE_TIPO[tipo], 4),
                "n_candidatos": int((candidatos["TIPO"] == tipo).sum()),
                "natureza": origem,
                "peso_A": round(pesos_a["tipo"], 4),
                "peso_B": round(pesos_b["tipo"], 4),
                # A agencia postal nao vem de Firmeza (2021): a fonte e a da origem
                # declarada do escore, para o quadro nao atribuir o valor a quem nao o mediu.
                "fonte": (ancora if tipo == "agencia_postal"
                          else f"Firmeza (2021), Figura 23, p. 62 — {ancora}"),
            }
        )

    horas = np.array(list(HORAS_SEMANAIS_POR_TIPO.values()))
    escores_h = normalizar(horas)
    for i, tipo in enumerate(HORAS_SEMANAIS_POR_TIPO):
        origem, ancora = ANCORA_HORAS[tipo]
        linhas.append(
            {
                "atributo": "disponibilidade",
                "categoria": tipo,
                "valor_bruto": f"{numero_br(HORAS_SEMANAIS_POR_TIPO[tipo], 1)} h/semana",
                "escore": round(float(escores_h[i]), 4),
                "n_candidatos": int((candidatos["TIPO"] == tipo).sum()),
                "natureza": origem,
                "peso_A": round(pesos_a["disponibilidade"], 4),
                "peso_B": round(pesos_b["disponibilidade"], 4),
                "fonte": f"OSM opening_hours, {ancora}; estratificacao validada por "
                         f"Kruskal-Wallis H={KRUSKAL_H}, p={KRUSKAL_P:.1e} "
                         f"({N_OBSERVACOES_HORARIO} obs.; {POS_TESTE}, "
                         f"{PARES_SIGNIFICATIVOS} de 6 pares)",
            }
        )

    for atributo, composicao, fonte in [
        ("acessibilidade", PESO_ACESSIBILIDADE,
         f"OSM highway=bus_stop (limite {LIMITE_PARADA_M:.0f} m) e densidade de "
         f"cruzamentos da rede caminhavel em {RAIO_VIZINHANCA_M:.0f} m"),
        ("seguranca", PESO_SEGURANCA,
         f"proxy: distancia a via {'/'.join(CLASSES_VIA_PRINCIPAL)} (limite "
         f"{LIMITE_VIA_M:.0f} m) e densidade de estabelecimentos do CNEFE em "
         f"{RAIO_VIZINHANCA_M:.0f} m — SEM base publica de ocorrencias"),
    ]:
        for componente, peso_interno in composicao.items():
            linhas.append(
                {
                    "atributo": atributo,
                    "categoria": componente,
                    "valor_bruto": f"peso interno {peso_interno:.0%}",
                    "escore": np.nan,
                    "n_candidatos": len(candidatos),
                    "natureza": "calculado" if atributo == "acessibilidade" else "proxy declarada",
                    "peso_A": round(pesos_a[atributo], 4),
                    "peso_B": round(pesos_b[atributo], 4),
                    "fonte": fonte,
                }
            )

    return pd.DataFrame(linhas)


def salvar(
    candidatos: gpd.GeoDataFrame,
    escores: dict[str, np.ndarray],
    aj_a: np.ndarray,
    aj_b: np.ndarray,
    detalhes: pd.DataFrame,
    sensibilidade: pd.DataFrame,
) -> None:
    subtitulo("10. Gravacao das saidas")
    ARQ_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)

    saida = candidatos.copy()
    for atributo, valores in escores.items():
        saida[f"S_{atributo.upper()}"] = np.round(valores, 6)
    for coluna in detalhes.columns:
        saida[coluna] = np.round(detalhes[coluna].to_numpy(), 4)
    saida["AJ"] = np.round(aj_a, 6)                      # principal, usado no modelo
    saida[f"AJ_ORDEM_{ORDEM_SENSIBILIDADE}"] = np.round(aj_b, 6)
    saida["ORDEM_PESOS"] = ORDEM_PRINCIPAL

    for coluna in saida.columns:
        if saida[coluna].dtype == "string":
            saida[coluna] = saida[coluna].astype(object).where(saida[coluna].notna(), None)

    ARQ_SAIDA.unlink(missing_ok=True)
    saida.to_file(ARQ_SAIDA, layer="candidatos_com_aj", driver="GPKG")

    arq_quadro = DIR_TABELAS / "quadro_atratividade.csv"
    quadro_de_atratividade(candidatos).to_csv(arq_quadro, index=False, sep=";", decimal=",")

    arq_sens = DIR_TABELAS / "sensibilidade_ordem_pesos.csv"
    sensibilidade.to_csv(arq_sens, index=False, sep=";", decimal=",")

    for caminho in (ARQ_SAIDA, arq_quadro, arq_sens):
        print(f"   {caminho.relative_to(RAIZ).as_posix():<48} {caminho.stat().st_size / 1024:>8.1f} KB")


def resumir(candidatos: gpd.GeoDataFrame, aj: np.ndarray) -> None:
    subtitulo("11. Resumo do indice (ordem principal)")

    dados = candidatos.copy()
    dados["AJ"] = aj

    print(f"   {'regiao':<14} {'n':>6} {'a_j medio':>11} {'a_j min':>9} {'a_j max':>9}")
    for regiao in ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]:
        sub = dados[dados["REGIAO_FUNCIONAL"] == regiao]
        if sub.empty:
            continue
        print(f"   {regiao:<14} {len(sub):>6} {sub['AJ'].mean():>11.4f} "
              f"{sub['AJ'].min():>9.4f} {sub['AJ'].max():>9.4f}")

    print(f"\n   10 candidatos de maior a_j:")
    for linha in dados.nlargest(10, "AJ").itertuples():
        print(f"      {(linha.NOME or '(sem nome)')[:30]:<30} {linha.TIPO:<16} "
              f"{linha.NM_BAIRRO[:18]:<18} {linha.AJ:.4f}")
    print(f"\n   10 candidatos de menor a_j:")
    for linha in dados.nsmallest(10, "AJ").itertuples():
        print(f"      {(linha.NOME or '(sem nome)')[:30]:<30} {linha.TIPO:<16} "
              f"{linha.NM_BAIRRO[:18]:<18} {linha.AJ:.4f}")


# =============================================================================
# EXECUCAO
# =============================================================================


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Indice de atratividade a_j dos candidatos.")
    parser.add_argument("--saida", default=None,
                        help="versao da rodada (ex.: v36): grava em data/tratados/<saida>/ e "
                             "results/<saida>/tabelas/, sem tocar nos arquivos vigentes")
    parser.add_argument("--escore-postal", type=float, default=None,
                        help=f"escore de tipo das agencias postais (padrao {ESCORE_AGENCIA_POSTAL:.4f})")
    argumentos = parser.parse_args()
    global ARQ_SAIDA, DIR_TABELAS
    if argumentos.saida:
        ARQ_SAIDA = DIR_TRATADOS / argumentos.saida / ARQ_SAIDA.name
        DIR_TABELAS = RAIZ / "results" / argumentos.saida / "tabelas"
    if argumentos.escore_postal is not None:
        definir_escore_postal(argumentos.escore_postal)

    inicio = time.perf_counter()

    pesos = pesos_da_ordem(ORDEM_PRINCIPAL)
    titulo(
        "04_atratividade.py — indice de atratividade a_j\n"
        f"ordem principal ({ORDEM_PRINCIPAL}): "
        + " · ".join(f"{k} {v:.4f}" for k, v in pesos.items())
        + f"\npiso dos escores: {PISO_ESCORE} · escore da agencia postal: "
        + f"{ESCORE_AGENCIA_POSTAL:.4f} ({ANCORA_TIPO['agencia_postal'][0]})"
        + f"\nsaidas: {ARQ_SAIDA.relative_to(RAIZ).as_posix()} · "
        + f"{DIR_TABELAS.relative_to(RAIZ).as_posix()}/"
    )

    candidatos = carregar_candidatos()
    bairros = gpd.read_file(ARQ_BAIRROS).to_crs(CRS_TRABALHO)
    paradas, vias, grafo = extrair_osm(bairros.union_all())
    estabelecimentos = carregar_estabelecimentos_cnefe()

    escores = {}
    escores["tipo"] = escore_tipo(candidatos)
    escores["disponibilidade"] = escore_disponibilidade(candidatos)
    escores["acessibilidade"], det_acess = escore_acessibilidade(candidatos, paradas, grafo)
    escores["seguranca"], det_seg = escore_seguranca(candidatos, vias, estabelecimentos)
    detalhes = pd.concat([det_acess, det_seg], axis=1)

    aj_a, aj_b = montar_indice(escores)
    sensibilidade = sensibilidade_ordem(candidatos, aj_a, aj_b)
    verificar(escores, aj_a, aj_b)

    salvar(candidatos, escores, aj_a, aj_b, detalhes, sensibilidade)
    resumir(candidatos, aj_a)

    titulo(
        f"Concluido em {time.perf_counter() - inicio:.1f} s  ·  "
        f"{len(candidatos)} candidatos  ·  "
        f"a_j de {aj_a.min():.3f} a {aj_a.max():.3f}"
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
