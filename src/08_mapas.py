"""
08_mapas.py — mapas e graficos da monografia.

Uma unica funcao de plotagem parametrizada, `mapa()`, reaproveitada por todas as
figuras cartograficas. Os graficos estatisticos tem funcoes proprias, por nao
compartilharem nenhum elemento cartografico.

Padrao ABNT adotado:
    - SEM ax.set_title(). A ABNT pede titulo acima e fonte abaixo, FORA da area
      do mapa. As figuras saem limpas e o titulo entra na inserção no Word.
    - legenda sempre fora da area de dados;
    - escala grafica por ScaleBar(1, units="m") — em EPSG:31982 a unidade ja e
      metro, entao sai correta sem conversao;
    - seta norte desenhada a mao, porque o matplotlib nao tem uma pronta;
    - savefig(dpi=300, bbox_inches="tight"), PNG.

Florianopolis tem bounding box muito mais alto que largo. O figsize e calculado a
partir da proporcao REAL do recorte, com ax.set_aspect("equal"); sem isso sobra
faixa branca lateral e o mapa encolhe na pagina.

O recorte e a area de TERRA (413,4 km2), com as massas d'agua ja excluidas em
bairros.gpkg.

Este script apenas LE: nunca reotimiza nem recalcula indicadores.

Entradas:
    data/tratados/bairros.gpkg
    data/tratados/demanda_bairro.csv
    data/tratados/candidatos_com_aj.gpkg
    data/tratados/solucoes_cenarios.gpkg
    data/brutos/ibge/SC_setores_CD2022.gpkg          massas d'agua, verificacao
    results/tabelas/*.csv                          indicadores da Etapa 8

Com --unidade setor, le as tabelas e solucoes de sufixo _setor e grava as
figuras de resultado com o mesmo sufixo. Os mapas de cenario passam a colorir os
SETORES (coberto ou nao), que e a resolucao em que o modelo decide, e ganham um
recorte ampliado da area urbana central. As figuras de diagnostico (fig01 a fig03)
independem da unidade e saem so na rodada por setor, que e a do texto.

Saidas:
    results/figuras/*.png                 rodada por setor
    results/nao_usados/figuras/*.png      rodada por bairro e figuras de apoio

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/08_mapas.py                     # setores censitarios (padrao)
    python src/08_mapas.py --unidade bairro    # bairros (fora do escopo; vai para nao_usados)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import box

matplotlib.use("Agg")

# =============================================================================
# PARAMETROS
# =============================================================================

RAIZ = Path(__file__).resolve().parents[1]
DIR_TRATADOS = RAIZ / "data" / "tratados"
# Redefinidos em main() conforme a unidade de demanda.
DIR_TABELAS = RAIZ / "results" / "tabelas"
DIR_FIGURAS = RAIZ / "results" / "figuras"
# Resultados que NAO entram na monografia vao para results/nao_usados/. A rodada
# por bairro saiu do escopo quando o setor censitario virou a unidade de demanda
# (docs/decisoes_etapa10.md): ela continua reproduzivel, mas grava ali, para nao
# se misturar com o que vai para o texto.
DIR_RESULTADOS = {"setor": RAIZ / "results", "bairro": RAIZ / "results" / "nao_usados"}
# A figura de ponderacao na metrica euclidiana e material de apoio (Etapa 9): o texto
# usa a de rede. Sai sempre em nao_usados.
DIR_FIGURAS_NAO_USADAS = RAIZ / "results" / "nao_usados" / "figuras"

ARQ_BAIRROS = DIR_TRATADOS / "bairros.gpkg"
ARQ_DEMANDA = DIR_TRATADOS / "demanda_bairro.csv"
ARQ_DEMANDA_SETOR = DIR_TRATADOS / "demanda_setor.csv"
ARQ_ALOCACAO = DIR_TRATADOS / "alocacao_setores.parquet"
ARQ_CANDIDATOS = DIR_TRATADOS / "candidatos_com_aj.gpkg"
ARQ_SOLUCOES = DIR_TRATADOS / "solucoes_cenarios.gpkg"
ARQ_SETORES = RAIZ / "data" / "brutos" / "ibge" / "SC_setores_CD2022.gpkg"
ARQ_SETORES_DEMANDA = DIR_TRATADOS / "setores.gpkg"

# Unidade de demanda da rodada: define o sufixo das tabelas lidas e das figuras.
UNIDADES = {"bairro": "", "setor": "_setor"}

CD_MUN = "4205407"
CRS_TRABALHO = 31982
CD_SIT_MASSA_DAGUA = "9"

POP_REFERENCIA = 537_211
COLUNA_DEMANDA = "h_beta03_referencia"   # beta = 0,3, alpha de referencia

# -----------------------------------------------------------------------------
# Aparencia
# -----------------------------------------------------------------------------
DPI = 300
ALTURA_POL = 9.0          # altura util do mapa, em polegadas
LARGURA_LEGENDA_POL = 2.2  # folga a direita, para a barra de cores
MARGEM_BBOX = 0.03        # folga alem do bbox da terra, em fracao da maior dimensao

CMAP_DENSIDADE = "YlOrRd"
CMAP_DEMANDA = "YlGnBu"
CMAP_COBERTURA = "RdYlGn"

# Escala das figuras de cenario FIXA em 0 a 100%. Normalizar pelo maximo
# observado distorceria a comparacao entre cenarios, e o ponto da figura e
# justamente mostrar o Sul vazio nos cenarios compactos e recuperado em C5.
ESCALA_COBERTURA = (0.0, 100.0)

# Recorte ampliado dos mapas de cenario por setor. Na escala do municipio os
# setores urbanos cobertos ficam pequenos demais para ler. O recorte e FIXO — a
# caixa dos bairros destas regioes, com folga — para que os cinco cenarios sejam
# comparaveis entre si; um recorte ajustado a cada solucao mudaria a escala de
# figura para figura.
REGIOES_RECORTE_AMPLIADO = ["Centro/Sede", "Continente"]
FOLGA_RECORTE_M = 600.0

CORES_TIPO = {
    "supermercado": "#1f4e79",
    "farmacia": "#c0392b",
    "conveniencia": "#e69138",
    "agencia_postal": "#6a3d9a",
    "mercado_pequeno": "#2e7d32",
}
ROTULO_TIPO = {
    "supermercado": "Supermercado",
    "farmacia": "Farmácia",
    "conveniencia": "Conveniência",
    "agencia_postal": "Agência postal",
    "mercado_pequeno": "Varejo de pequeno porte",
}
ORDEM_REGIOES = ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]

# Numeracao das figuras = ordem em que entram no Capitulo 4: diagnostico (01-03),
# mapas de resultado por cenario (04-06), ponderacao (07) e graficos (08-11). Nas
# series de cinco mapas, a letra identifica o cenario, sempre na mesma ordem.
LETRA_CENARIO = {"C1": "a", "C2": "b", "C3": "c", "C4": "d", "C5": "e", "C6": "f"}

# -----------------------------------------------------------------------------
# Figuras de pagina inteira: localizacoes otimas, uma por cenario (fig04a a fig04e)
# -----------------------------------------------------------------------------
# O painel com os cinco cenarios lado a lado ficou pequeno demais na pagina. Cada
# cenario ganha uma figura do tamanho da area util de uma pagina A4 (16 x 22 cm),
# com o mapa do municipio na altura toda e, sobre o oceano a leste da ilha, a
# legenda (em cima) e um recorte ampliado de Centro/Sede + Continente (embaixo).
# O arquivo sai com o tamanho exato da pagina, SEM bbox_inches="tight": o corte
# automatico mudaria as dimensoes de cenario para cenario.
CM = 1 / 2.54
PAGINA_CM = (16.0, 22.0)
ARQUIVOS_PAGINA = {cenario: f"fig04{letra}_{cenario}.png" for cenario, letra in LETRA_CENARIO.items()}
# O cenario misto entrou na revisao (v36), com nome proprio de arquivo.
ARQUIVOS_PAGINA["C6"] = "fig_mapa_C6.png"
# Folga vertical alem da terra, em metros. A largura da pagina e maior que a do
# municipio na mesma escala; a sobra fica a leste, sobre o oceano.
FOLGA_PAGINA_M = 800.0
MARGEM_QUADRO_CM = 0.25       # afastamento do recorte ampliado ate a borda da pagina
AFASTAMENTO_TERRA_M = 400.0   # o recorte ampliado nao pode encostar na costa

# Cobertura do setor em duas classes. O modelo aloca o setor inteiro ou nada: nos
# cinco cenarios, so 3 dos 964 setores ficam fora de {0%, 100%}, com 99,9x% por
# tolerancia numerica do solver. O limiar de 50% separa as classes sem ambiguidade.
LIMIAR_COBERTO_PCT = 50.0
COR_COBERTO = "#6baed6"       # azul medio
COR_NAO_COBERTO = "#fdd9a8"   # laranja claro — par azul x laranja resiste a daltonismo
COR_SEM_POPULACAO = "#d9d9d9"

# Tipo do estabelecimento com cor E forma: a forma garante a leitura em impressao
# em tons de cinza e para leitores daltonicos. As cores sao as da fig03.
MARCADOR_TIPO = {
    "supermercado": "s",
    "farmacia": "P",
    "conveniencia": "^",
    "agencia_postal": "D",
    "mercado_pequeno": "o",
}
TAMANHO_PONTO_PAGINA = 34     # mapa do municipio (pt²)
TAMANHO_PONTO_RECORTE = 70    # recorte ampliado (pt²)

# Rotulos das regioes funcionais. Parte do ponto representativo interno da regiao
# e aplica um deslocamento, em metros, quando o rotulo cairia sobre os pontos ou
# fora do mapa. Os deslocamentos sao so de legibilidade: nao mudam dado nenhum.
# No mapa principal, Centro/Sede e Continente saem para fora do quadro do recorte,
# onde os pontos se amontoam: Centro/Sede ao norte do quadro (Saco Grande, ainda
# dentro da regiao) e Continente ao sul, sobre a Baia Sul.
DESLOCAMENTO_ROTULO_M = {
    "Centro/Sede": (1_170.0, 6_770.0),
    "Continente": (900.0, -6_000.0),
    "Norte": (0.0, 0.0),
    "Leste": (0.0, 0.0),
    "Sul": (0.0, 0.0),
}
# No recorte ampliado so aparecem as duas regioes que ele cobre, com rotulo sobre
# a agua: Continente na Baia Sul, Centro/Sede na Baia Norte.
DESLOCAMENTO_ROTULO_RECORTE_M = {
    "Centro/Sede": (-1_630.0, 4_470.0),
    "Continente": (130.0, -3_300.0),
}
# Seta norte e escala no canto noroeste, sobre a Baia Norte, em fracao do eixo.
POSICAO_SETA_PAGINA = (0.07, 0.975)
POSICAO_ESCALA_PAGINA = (0.01, 0.87)
# Os cinco cenarios dos graficos e do recorte ampliado FIXO. O cenario misto (C6)
# ganha os mesmos mapas quando existe na rodada lida, mas nao entra nos graficos
# nem redefine o recorte: as figuras de C1 a C5 tem de continuar identicas.
ORDEM_CENARIOS = ["C1", "C2", "C3", "C4", "C5"]
CENARIO_MISTO = "C6"
RAIO_MISTO = "800/1500"   # 800 m a pe em Centro/Sede e Continente; 1.500 m de carro no resto

# Metrica de rede de cada cenario. As figuras de cenario usam a rede, que e a
# metrica dos resultados; a euclidiana e referencia de comparacao.
METRICA_REDE = {"C1": "walk", "C2": "walk", "C3": "walk", "C4": "walk", "C5": "drive", "C6": "misto"}

# -----------------------------------------------------------------------------
# Figura do artigo: densidade demografica e regioes funcionais
# -----------------------------------------------------------------------------
ARQUIVO_ARTIGO = "fig_artigo_regioes_densidade.png"
ARQ_GRAFO_DRIVE = DIR_TRATADOS / "_cache_grafo_drive.graphml"
# As tres travessias Ilha <-> Continente, pelos nomes do OSM no grafo viario —
# os mesmos conferidos em 05_distancias.py.
PONTES = [
    "Ponte Hercílio Luz",
    "Ponte Governador Colombo Machado Salles",
    "Ponte Governador Pedro Ivo Campos",
]
COR_PONTE = "#0b3d91"
# O Continente fica na borda oeste do municipio. A pagina e deslocada para oeste
# para que ele, seu rotulo e a indicacao das pontes nao encostem na margem.
FOLGA_OESTE_ARTIGO_M = 4_500.0
# Rotulos das regioes na figura do artigo, deslocados do ponto representativo para
# nao cobrir os bairros mais densos (deslocamento em metros; so legibilidade).
DESLOCAMENTO_ROTULO_ARTIGO_M = {
    "Centro/Sede": (2_200.0, 900.0),
    "Continente": (-300.0, 4_600.0),
    "Norte": (0.0, 0.0),
    "Leste": (0.0, 0.0),
    "Sul": (0.0, 0.0),
}

# -----------------------------------------------------------------------------
# Mapa base
# -----------------------------------------------------------------------------
# Depende de rede. Falhando, a figura e gerada SEM fundo e o nome dela entra no
# resumo final — aviso no meio do log rolaria para fora da tela e a figura
# poderia ir para o Word sem base sem que ninguem percebesse.
USAR_BASEMAP = True

# ATENCAO — o CartoDB.Positron, previsto no CLAUDE.md, passou a EXIGIR CHAVE DE
# API. O add_basemap NAO levanta excecao: os tiles chegam como marca d'agua
# "API KEY REQUIRED" e a figura sai plausivel e errada. O OpenStreetMap.Mapnik
# devolve 403 "Access blocked" por politica de uso de servidores voluntarios.
#
# Fontes tentadas em ordem; a primeira que passar na validacao e usada.
FONTES_BASEMAP = [
    ("Esri.WorldGrayCanvas", ctx.providers.Esri.WorldGrayCanvas),
    ("CartoDB.Positron", ctx.providers.CartoDB.Positron),
    ("CartoDB.Voyager", ctx.providers.CartoDB.Voyager),
    ("OpenStreetMap.Mapnik", ctx.providers.OpenStreetMap.Mapnik),
]

# Um tile de area urbana tem centenas de cores; marca d'agua de erro tem dezenas.
# Medido: Esri 1.063 cores, CartoDB 67, OSM 55.
MIN_CORES_TILE = 200
# Tiles de lugares diferentes precisam ser diferentes. Uma marca d'agua e a mesma
# imagem em toda parte, entao esta e a verificacao decisiva.
MIN_DIFERENCA_ENTRE_TILES = 4.0

FONTE_BASEMAP = None          # definida em escolher_fonte_basemap()
NOME_FONTE_BASEMAP = "nenhuma"
FONTES_REJEITADAS: list[str] = []

FIGURAS_SEM_BASE: list[str] = []
FIGURAS_GERADAS: list[tuple[str, str, str]] = []   # arquivo, dimensoes, pasta

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


def verificar_total(serie: pd.Series, esperado: float, rotulo: str, tol: float = 0.5) -> None:
    """Confere que a coluna plotada e a certa, comparando a soma com a fonte.

    Um coropletico da coluna errada passa em todos os testes de CRS e de
    geometria: continua bonito e continua errado. Esta e a unica verificacao que
    pega essa classe de erro.
    """
    obtido = float(serie.sum())
    checar(
        abs(obtido - esperado) <= tol,
        f"{rotulo}: a soma dos valores plotados e {obtido:,.2f}, mas a fonte diz "
        f"{esperado:,.2f}. Provavel coluna errada no coropletico.",
    )
    print(f"   soma conferida ({rotulo}): {numero_br(obtido, 2)}")


# =============================================================================
# DADOS
# =============================================================================


def carregar(unidade: str) -> dict:
    subtitulo("1. Dados")
    sufixo = UNIDADES[unidade]

    bairros = gpd.read_file(ARQ_BAIRROS).to_crs(CRS_TRABALHO)
    candidatos = gpd.read_file(ARQ_CANDIDATOS).to_crs(CRS_TRABALHO)
    arquivo_solucoes = ARQ_SOLUCOES.with_name(f"{ARQ_SOLUCOES.stem}{sufixo}.gpkg")
    checar(arquivo_solucoes.exists(), f"{arquivo_solucoes} nao existe. Rodar src/07_cenarios.py --unidade {unidade}")
    solucoes = gpd.read_file(arquivo_solucoes).to_crs(CRS_TRABALHO)
    demanda = pd.read_csv(ARQ_DEMANDA, sep=";", decimal=",", dtype={"CD_BAIRRO": str})
    bairros["CD_BAIRRO"] = bairros["CD_BAIRRO"].astype(str)

    setores = gpd.read_file(ARQ_SETORES, where=f"CD_MUN = '{CD_MUN}'")
    setores = setores[["CD_SETOR", "CD_SIT", "geometry"]].dissolve(
        by="CD_SETOR", as_index=False, aggfunc="first"
    ).to_crs(CRS_TRABALHO)
    agua = setores[setores["CD_SIT"] == CD_SIT_MASSA_DAGUA].geometry.make_valid().union_all()

    tabelas = {}
    nomes = [
        "cobertura_por_bairro_cenarios", "cobertura_por_regiao_cenarios",
        "indicadores_cenarios", "efeito_atratividade",
    ] + (["cobertura_por_setor_cenarios"] if unidade == "setor" else [])
    for nome in nomes:
        caminho = DIR_TABELAS / f"{nome}{sufixo}.csv"
        checar(caminho.exists(), f"{caminho} nao existe. Rodar src/07_cenarios.py --unidade {unidade}")
        # CD_SETOR tem 15 digitos: lido como numero, perde a identidade no merge.
        tabelas[nome] = pd.read_csv(caminho, sep=";", decimal=",", dtype={"CD_SETOR": str, "CD_BAIRRO": str})
        # O cenario misto grava o raio como texto ("800/1500"), o que faz a coluna
        # inteira ser lida como texto. Volta a ser numero; o do misto fica vazio.
        if "r_m" in tabelas[nome].columns:
            tabelas[nome]["r_m"] = pd.to_numeric(tabelas[nome]["r_m"], errors="coerce")

    setores = None
    if unidade == "setor":
        setores = gpd.read_file(ARQ_SETORES_DEMANDA).to_crs(CRS_TRABALHO)
        setores["CD_SETOR"] = setores["CD_SETOR"].astype(str)

    if unidade == "setor":
        # A demanda do modelo e a dos SETORES, com a renda de cada setor. O mapa por
        # bairro soma os setores pela fracao de cada um no bairro — a mesma regra das
        # tabelas de cobertura —, para que a figura mostre a demanda que o modelo usa.
        setor = pd.read_csv(ARQ_DEMANDA_SETOR, sep=";", decimal=",", dtype={"CD_SETOR": str})
        alocacao = pd.read_parquet(ARQ_ALOCACAO).astype({"CD_SETOR": str, "CD_BAIRRO": str})
        base = alocacao.merge(setor[["CD_SETOR", COLUNA_DEMANDA]], on="CD_SETOR", how="inner")
        demanda = (
            (base[COLUNA_DEMANDA] * base["FRACAO"]).groupby(base["CD_BAIRRO"]).sum()
            .rename(COLUNA_DEMANDA).reset_index()
        )
        checar(
            abs(demanda[COLUNA_DEMANDA].sum() - setor[COLUNA_DEMANDA].sum()) < 1e-6,
            "A demanda dos setores agregada por bairro nao reproduz o total.",
        )
    bairros = bairros.merge(
        demanda[["CD_BAIRRO", COLUNA_DEMANDA]].rename(columns={COLUNA_DEMANDA: "H"}),
        on="CD_BAIRRO", how="left",
    )
    checar(bairros["H"].notna().all(), "Bairro sem demanda apos a juncao.")

    print(f"bairros ................................ {len(bairros)}")
    print(f"candidatos ............................. {len(candidatos)}")
    print(f"solucoes (pontos abertos) .............. {len(solucoes)} de "
          f"{solucoes['EXECUCAO'].nunique()} execucoes")
    print(f"area de terra .......................... {numero_br(bairros.area.sum() / 1e6, 1)} km2")
    print(f"CRS .................................... EPSG:{bairros.crs.to_epsg()}")
    for camada, nome in [(bairros, "bairros"), (candidatos, "candidatos"), (solucoes, "solucoes")]:
        checar(
            camada.crs.to_epsg() == CRS_TRABALHO,
            f"Camada {nome} fora do EPSG:{CRS_TRABALHO}.",
        )
    return {
        "unidade": unidade, "sufixo": sufixo, "setores": setores,
        "bairros": bairros, "candidatos": candidatos, "solucoes": solucoes,
        "agua": agua, **tabelas,
    }


# =============================================================================
# ELEMENTOS CARTOGRAFICOS
# =============================================================================


def figsize_do_recorte(limites: tuple[float, float, float, float]) -> tuple[float, float]:
    """figsize pela proporcao REAL do recorte, com folga para a legenda."""
    xmin, ymin, xmax, ymax = limites
    razao = (xmax - xmin) / (ymax - ymin)
    return (ALTURA_POL * razao + LARGURA_LEGENDA_POL, ALTURA_POL)


def limites_com_margem(bairros: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bairros.total_bounds
    folga = MARGEM_BBOX * max(xmax - xmin, ymax - ymin)
    return xmin - folga, ymin - folga, xmax + folga, ymax + folga


def desenhar_seta_norte(ax, posicao: tuple[float, float] = (0.94, 0.97)) -> None:
    """Seta norte: nao existe pronta no matplotlib. `posicao` e a ponta da seta."""
    x, y = posicao
    ax.annotate(
        "N",
        xy=(x, y), xytext=(x, y - 0.09),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=13, fontweight="bold",
        arrowprops=dict(facecolor="black", edgecolor="black", width=3.5, headwidth=11),
    )


def adicionar_escala(
    ax, posicao: str | tuple[float, float] = "lower right", comprimento: float = 0.3
) -> None:
    """ScaleBar(1, units='m'): em EPSG:31982 a unidade ja e metro.

    `posicao` e um canto ('lower right', ...) ou um ponto (x, y) em fracao do
    eixo, onde fica o canto superior esquerdo da barra.
    """
    if isinstance(posicao, tuple):
        ancoragem = dict(location="upper left", bbox_to_anchor=posicao, bbox_transform=ax.transAxes)
    else:
        # 'lower right' por padrao: o canto inferior esquerdo e ocupado pela
        # atribuicao obrigatoria do provedor de tiles, e as duas se sobrepunham.
        ancoragem = dict(location=posicao)
    ax.add_artist(
        ScaleBar(1, units="m", box_alpha=0.8, border_pad=0.6,
                 length_fraction=comprimento, **ancoragem)
    )


def _tile_de_teste(fonte, centro_x: float, centro_y: float, lado: float = 4000.0):
    """Busca um tile de teste em Web Mercator e devolve o array RGB."""
    caixa = (centro_x - lado, centro_y - lado, centro_x + lado, centro_y + lado)
    imagem, _ = ctx.bounds2img(*caixa, source=fonte, zoom=13)
    return np.asarray(imagem)[:, :, :3].astype(float)


def escolher_fonte_basemap(bairros: gpd.GeoDataFrame) -> None:
    """Testa as fontes e adota a primeira que devolva mapa DE VERDADE.

    O add_basemap nao levanta excecao quando o servidor responde com marca
    d'agua de erro: a figura sai com fundo plausivel e errado. Duas verificacoes
    pegam isso — a contagem de cores do tile e, sobretudo, a comparacao entre
    tiles de lugares diferentes, porque uma marca d'agua e a mesma imagem em
    toda parte.
    """
    global FONTE_BASEMAP, NOME_FONTE_BASEMAP
    subtitulo("2. Validacao do mapa base")

    if not USAR_BASEMAP:
        print("   USAR_BASEMAP = False — figuras serao geradas sem fundo.")
        return

    mercator = bairros.to_crs(3857)
    x0, y0, x1, y1 = mercator.total_bounds
    centro = ((x0 + x1) / 2, (y0 + y1) / 2)
    norte = ((x0 + x1) / 2, y1 - 6000)

    for nome, fonte in FONTES_BASEMAP:
        try:
            a = _tile_de_teste(fonte, *centro)
            b = _tile_de_teste(fonte, *norte)
        except Exception as erro:
            FONTES_REJEITADAS.append(f"{nome}: {type(erro).__name__}")
            print(f"   {nome:<24} FALHOU ({type(erro).__name__})")
            continue

        cores = len(np.unique(a.reshape(-1, 3), axis=0))
        # Os dois recortes podem sair com tamanhos diferentes, porque a caixa se
        # ajusta a grade de tiles. Compara-se a regiao comum.
        linhas = min(a.shape[0], b.shape[0])
        colunas = min(a.shape[1], b.shape[1])
        diferenca = float(np.abs(a[:linhas, :colunas] - b[:linhas, :colunas]).mean())
        if cores < MIN_CORES_TILE:
            FONTES_REJEITADAS.append(f"{nome}: apenas {cores} cores no tile (marca d'agua?)")
            print(f"   {nome:<24} REJEITADA — {cores} cores, abaixo de {MIN_CORES_TILE}")
            continue
        if diferenca < MIN_DIFERENCA_ENTRE_TILES:
            FONTES_REJEITADAS.append(f"{nome}: tiles de lugares diferentes sao iguais")
            print(f"   {nome:<24} REJEITADA — tiles distintos diferem so {diferenca:.2f}")
            continue

        FONTE_BASEMAP, NOME_FONTE_BASEMAP = fonte, nome
        print(f"   {nome:<24} ADOTADA — {cores} cores, diferenca entre tiles {diferenca:.1f}")
        return

    print("   NENHUMA fonte passou. As figuras sairao sem fundo cartografico.")


def adicionar_basemap(ax, nome_figura: str) -> bool:
    if not USAR_BASEMAP or FONTE_BASEMAP is None:
        if USAR_BASEMAP:
            FIGURAS_SEM_BASE.append(f"{nome_figura} (nenhuma fonte valida)")
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx.add_basemap(ax, crs=CRS_TRABALHO, source=FONTE_BASEMAP, attribution_size=6)
        return True
    except Exception as erro:  # rede indisponivel, tile fora do ar, etc.
        FIGURAS_SEM_BASE.append(f"{nome_figura} ({type(erro).__name__})")
        return False


def registrar_figura(caminho: Path) -> None:
    imagem = mpimg.imread(caminho)
    altura, largura = imagem.shape[0], imagem.shape[1]
    FIGURAS_GERADAS.append(
        (caminho.name, f"{largura}x{altura} px · {largura / altura:.2f}", caminho.parent.relative_to(RAIZ).as_posix())
    )


# =============================================================================
# A FUNCAO DE PLOTAGEM
# =============================================================================


def mapa(
    arquivo: str,
    bairros: gpd.GeoDataFrame,
    coluna_cor: str | None = None,
    cmap: str = "viridis",
    escala: tuple[float, float] | None = None,
    rotulo_cor: str = "",
    pontos: list[dict] | None = None,
    posicao_legenda: str = "abaixo",
    cor_fundo_bairros: str = "#f2f2f2",
    largura_borda: float = 0.35,
    contorno: gpd.GeoDataFrame | None = None,
    limites: tuple[float, float, float, float] | None = None,
    ax=None,
    categorias: dict | None = None,
    largura_contorno: float = 0.45,
    rotulos: list[dict] | None = None,
    posicao_seta: tuple[float, float] | None = (0.94, 0.97),
    posicao_escala: str | tuple[float, float] = "lower right",
    comprimento_escala: float = 0.3,
    barra_cores: bool = True,
    cor_contorno: str = "#3c3c3c",
) -> list:
    """Mapa parametrizado, usado por todas as figuras cartograficas.

    `bairros` e sempre a base — os poligonos de bairro ou, na rodada por setor,
    os de setor; `coluna_cor` a torna coropletica. `pontos` e uma lista de camadas,
    cada uma um dicionario com dados, cor, marcador, tamanho e rotulo. Isso cobre
    coropletico puro, pontos puros e a combinacao dos dois sem ramificacao especial.

    Poligono sem valor na coluna de cor (setor sem populacao) sai em cinza claro,
    nunca em branco, para nao parecer buraco no territorio. `contorno` desenha por
    cima os limites de outra camada, como os bairros sobre os setores. `limites`
    fixa um recorte (xmin, ymin, xmax, ymax); sem ele, o mapa mostra todo o
    territorio. O figsize segue a proporcao do recorte nos dois casos.

    `categorias` troca a escala continua por classes discretas: dicionario
    {valor da coluna: cor}, sem barra de cores. `rotulos` escreve textos no mapa
    (dicionarios com texto, x, y e, opcionalmente, tamanho).

    Com `ax`, o mapa e desenhado num eixo ja existente — o caso das figuras de
    pagina inteira, com mapa principal e recorte ampliado na mesma imagem. Nesse
    modo a funcao NAO monta legenda nem salva: devolve os manipuladores de
    legenda e deixa a composicao da pagina para quem chamou.
    """
    limites = limites if limites is not None else limites_com_margem(bairros)
    desenha_em_eixo_existente = ax is not None
    if desenha_em_eixo_existente:
        figura = ax.figure
    else:
        figura, ax = plt.subplots(figsize=figsize_do_recorte(limites))

    if coluna_cor is None:
        bairros.plot(ax=ax, color=cor_fundo_bairros, edgecolor="#8c8c8c", linewidth=0.4)
    elif categorias is not None:
        # Classes discretas: cada poligono recebe a cor da sua classe. Poligono
        # sem classe (setor sem populacao) sai em cinza, como no caso continuo.
        cores = bairros[coluna_cor].map(categorias).fillna("#d9d9d9")
        bairros.plot(ax=ax, color=cores, edgecolor="#6e6e6e", linewidth=largura_borda)
    else:
        vmin, vmax = escala if escala else (bairros[coluna_cor].min(), bairros[coluna_cor].max())
        bairros.plot(
            ax=ax, column=coluna_cor, cmap=cmap, vmin=vmin, vmax=vmax,
            edgecolor="#6e6e6e", linewidth=largura_borda,
            missing_kwds={"color": "#d9d9d9", "edgecolor": "#6e6e6e", "linewidth": largura_borda},
        )
        # barra_cores=False: quem chamou desenha a barra onde quiser (figura de
        # pagina inteira, em que a barra vai sobre o oceano e nao ao lado do mapa).
        if barra_cores:
            norma = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            barra = figura.colorbar(
                matplotlib.cm.ScalarMappable(norm=norma, cmap=cmap),
                ax=ax, fraction=0.035, pad=0.02, shrink=0.72,
            )
            barra.set_label(rotulo_cor, fontsize=9)
            barra.ax.tick_params(labelsize=8)

    if contorno is not None:
        contorno.boundary.plot(ax=ax, color=cor_contorno, linewidth=largura_contorno, zorder=3)

    manipuladores = []
    for camada in pontos or []:
        dados = camada["data"]
        if len(dados) == 0:
            continue
        dados.plot(
            ax=ax, color=camada.get("cor", "black"), marker=camada.get("marcador", "o"),
            markersize=camada.get("tamanho", 18), edgecolor=camada.get("borda", "white"),
            linewidth=camada.get("largura_borda", 0.5), zorder=camada.get("zorder", 5),
            alpha=camada.get("alpha", 1.0),
        )
        manipuladores.append(
            Line2D(
                [0], [0], marker=camada.get("marcador", "o"), linestyle="none",
                markerfacecolor=camada.get("cor", "black"),
                markeredgecolor=camada.get("borda", "white"),
                markersize=np.sqrt(camada.get("tamanho", 18)) * 1.4,
                label=camada["rotulo"],
            )
        )

    for rotulo in rotulos or []:
        # Halo branco: o texto fica legivel sobre qualquer cor de setor.
        ax.text(
            rotulo["x"], rotulo["y"], rotulo["texto"], ha="center", va="center",
            fontsize=rotulo.get("tamanho", 9), fontweight="bold", color="#1a1a1a", zorder=7,
            path_effects=[patheffects.withStroke(linewidth=2.6, foreground="white")],
        )

    ax.set_xlim(limites[0], limites[2])
    ax.set_ylim(limites[1], limites[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    # Sem ax.set_title(): titulo e fonte entram fora da figura, no Word.

    com_base = adicionar_basemap(ax, arquivo)
    adicionar_escala(ax, posicao_escala, comprimento_escala)
    if posicao_seta is not None:
        desenhar_seta_norte(ax, posicao_seta)

    if desenha_em_eixo_existente:
        print(f"   {arquivo:<44} {'com base' if com_base else 'SEM BASE'}")
        return manipuladores

    if manipuladores:
        if posicao_legenda == "abaixo":
            ax.legend(
                handles=manipuladores, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                ncol=min(3, len(manipuladores)), frameon=False, fontsize=9,
            )
        else:
            ax.legend(
                handles=manipuladores, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                frameon=False, fontsize=9,
            )

    destino = DIR_FIGURAS_NAO_USADAS if "_euclid" in arquivo else DIR_FIGURAS
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / arquivo
    figura.savefig(caminho, dpi=DPI, bbox_inches="tight")
    plt.close(figura)
    registrar_figura(caminho)
    print(f"   {arquivo:<44} {'com base' if com_base else 'SEM BASE'}")
    return manipuladores


# =============================================================================
# VERIFICACAO ESPACIAL
# =============================================================================


def verificar_pontos_em_terra(dados: dict) -> None:
    """Nenhum ponto plotado pode cair sobre massa d'agua.

    Licao das Etapas 2 e 6: figura plausivel pode estar errada, e nem CRS nem
    geometria detectam ponto no lugar errado.
    """
    subtitulo("3. Verificacao espacial das camadas de pontos")

    agua = dados["agua"]
    terra = dados["bairros"].union_all()
    for nome, camada in [("candidatos", dados["candidatos"]), ("solucoes", dados["solucoes"])]:
        na_agua = camada.geometry.within(agua)
        checar(
            not bool(na_agua.any()),
            f"{int(na_agua.sum())} ponto(s) de '{nome}' caem sobre massa d'agua: "
            f"{camada.loc[na_agua, 'NOME'].head(5).tolist()}",
        )
        fora = ~camada.geometry.within(terra)
        print(f"   {nome:<12} {len(camada):>5} pontos · sobre agua 0 · "
              f"fora da terra {int(fora.sum())}")

    limites = limites_com_margem(dados["bairros"])
    xmin, ymin, xmax, ymax = dados["bairros"].total_bounds
    checar(
        limites[0] <= xmin and limites[1] <= ymin and limites[2] >= xmax and limites[3] >= ymax,
        "Os limites do mapa nao contem a area de terra.",
    )
    razao = (xmax - xmin) / (ymax - ymin)
    print(f"   recorte: {numero_br((xmax - xmin) / 1000, 1)} km de largura x "
          f"{numero_br((ymax - ymin) / 1000, 1)} km de altura · razao {razao:.2f}")
    print(f"   figsize derivado: {figsize_do_recorte(limites)[0]:.1f} x "
          f"{figsize_do_recorte(limites)[1]:.1f} polegadas")


# =============================================================================
# FIGURAS DE DIAGNOSTICO
# =============================================================================


def figuras_diagnostico(dados: dict) -> None:
    subtitulo("4. Figuras de diagnostico")
    bairros = dados["bairros"]

    verificar_total(bairros["POP"], POP_REFERENCIA, "populacao")
    esperado_dens = bairros["POP"] / bairros["AREA_KM2"]
    checar(
        bool(np.allclose(bairros["DENS_HAB_KM2"], esperado_dens, rtol=1e-6)),
        "A coluna DENS_HAB_KM2 nao corresponde a POP / AREA_KM2.",
    )
    mapa(
        "fig01_densidade_demografica.png", bairros,
        coluna_cor="DENS_HAB_KM2", cmap=CMAP_DENSIDADE,
        rotulo_cor="Densidade demográfica (hab/km²)",
    )

    verificar_total(bairros["H"], float(bairros["H"].sum()), "demanda", tol=1e-6)
    print(f"   demanda total plotada: {numero_br(bairros['H'].sum(), 1)} encomendas/dia")
    mapa(
        "fig02_demanda_potencial.png", bairros,
        coluna_cor="H", cmap=CMAP_DEMANDA,
        rotulo_cor="Demanda potencial (encomendas/dia)",
    )

    candidatos = dados["candidatos"]
    camadas = []
    for tipo in CORES_TIPO:
        sub = candidatos[candidatos["TIPO"] == tipo]
        camadas.append(
            {
                "data": sub, "cor": CORES_TIPO[tipo], "tamanho": 14,
                "largura_borda": 0.3,
                "rotulo": f"{ROTULO_TIPO[tipo]} ({len(sub)})",
            }
        )
    checar(
        sum(len(c["data"]) for c in camadas) == len(candidatos),
        "As camadas por tipo nao somam o total de candidatos.",
    )
    mapa(
        "fig03_candidatos_por_tipo.png", bairros, pontos=camadas,
        posicao_legenda="abaixo",
    )


# =============================================================================
# FIGURAS DE RESULTADO
# =============================================================================


def dados_do_cenario(
    dados: dict, cenario: str, base_mapa: gpd.GeoDataFrame, chave: str
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, int, int]:
    """Cobertura por unidade e pontos abertos de um cenario, ja conferidos.

    Devolve a base do mapa com a coluna `cobertura_pct`, os pontos abertos, p e r.
    A soma da demanda coberta tem de reproduzir o Z1 do log, e o numero de pontos
    abertos tem de ser p: e o que garante que a figura mostra a solucao registrada.
    """
    por_setor = chave == "CD_SETOR"
    metrica = METRICA_REDE[cenario]
    cobertura = dados["cobertura_por_setor_cenarios" if por_setor else "cobertura_por_bairro_cenarios"]
    cobertura[chave] = cobertura[chave].astype(str)
    indicadores = dados["indicadores_cenarios"]
    solucoes = dados["solucoes"]

    sub = cobertura[
        (cobertura["cenario"] == cenario)
        & (cobertura["metrica"] == metrica)
        & (cobertura["variante"] == "aj_estimado")
    ]
    n_esperado = int((base_mapa["POP"] > 0).sum()) if por_setor else len(dados["bairros"])
    checar(len(sub) == n_esperado, f"{cenario}: cobertura por unidade incompleta ({len(sub)} de {n_esperado}).")

    # A soma da demanda coberta tem de reproduzir o Z1 registrado no log.
    z1 = float(
        indicadores[
            (indicadores["cenario"] == cenario)
            & (indicadores["metrica"] == metrica)
            & (indicadores["variante"] == "aj_estimado")
            & (indicadores["grupo"] == "matriz_principal")
        ]["Z1"].iloc[0]
    )
    verificar_total(sub["COBERTA"], z1, f"demanda coberta em {cenario}", tol=0.01)

    base = base_mapa.merge(sub[[chave, "cobertura_pct"]], on=chave, how="left")
    abertos = solucoes[
        (solucoes["CENARIO"] == cenario)
        & (solucoes["METRICA"] == metrica)
        & (solucoes["VARIANTE"] == "aj_estimado")
    ]
    p = int(indicadores[(indicadores["cenario"] == cenario)]["p"].iloc[0])
    r = RAIO_MISTO if cenario == CENARIO_MISTO else int(indicadores[(indicadores["cenario"] == cenario)]["r_m"].iloc[0])
    checar(len(abertos) == p, f"{cenario}: {len(abertos)} pontos abertos, esperado {p}.")
    return base, abertos, p, r


def cenarios_a_mapear(dados: dict) -> list[str]:
    """C1 a C5 e, se a rodada lida o tiver, o cenario misto."""
    tem_misto = bool((dados["solucoes"]["CENARIO"] == CENARIO_MISTO).any())
    return ORDEM_CENARIOS + ([CENARIO_MISTO] if tem_misto else [])


def figuras_cenarios(dados: dict) -> None:
    subtitulo("5b. Figuras de cenario — escala fixa em 0 a 100%")

    por_setor = dados["unidade"] == "setor"
    bairros = dados["bairros"]
    # Na rodada por setor o mapa colore os SETORES: e a resolucao em que o modelo
    # decide. Colorir o bairro pela media esconderia onde, dentro dele, esta a
    # parte coberta. Os limites de bairro vao por cima, como referencia.
    base_mapa = dados["setores"] if por_setor else bairros
    chave = "CD_SETOR" if por_setor else "CD_BAIRRO"
    sufixo = dados["sufixo"]

    area_central = bairros[bairros["REGIAO_FUNCIONAL"].isin(REGIOES_RECORTE_AMPLIADO)]
    xmin, ymin, xmax, ymax = area_central.total_bounds
    recorte = (xmin - FOLGA_RECORTE_M, ymin - FOLGA_RECORTE_M, xmax + FOLGA_RECORTE_M, ymax + FOLGA_RECORTE_M)
    if por_setor:
        print(f"   recorte ampliado: {', '.join(REGIOES_RECORTE_AMPLIADO)} · "
              f"{numero_br((recorte[2] - recorte[0]) / 1000, 1)} x {numero_br((recorte[3] - recorte[1]) / 1000, 1)} km")

    for cenario in cenarios_a_mapear(dados):
        base, abertos, p, r = dados_do_cenario(dados, cenario, base_mapa, chave)

        mapa(
            f"fig05{LETRA_CENARIO[cenario]}_cobertura_{cenario}{sufixo}.png", base,
            coluna_cor="cobertura_pct", cmap=CMAP_COBERTURA, escala=ESCALA_COBERTURA,
            rotulo_cor=("Cobertura da demanda do setor (%)" if por_setor
                        else "Cobertura da demanda do bairro (%)"),
            largura_borda=0.08 if por_setor else 0.35,
            contorno=bairros if por_setor else None,
            pontos=[
                {
                    "data": abertos, "cor": "#111111", "marcador": "*",
                    "tamanho": 140, "borda": "white", "largura_borda": 0.7,
                    "rotulo": f"Ponto de retirada (p = {p}, r = {r} m, {METRICA_REDE[cenario]})",
                }
            ],
        )

        if por_setor:
            dentro = abertos.geometry.within(
                gpd.GeoSeries.from_xy([recorte[0], recorte[2]], [recorte[1], recorte[3]], crs=CRS_TRABALHO)
                .union_all().envelope
            )
            mapa(
                f"fig06{LETRA_CENARIO[cenario]}_cobertura_ampliada_{cenario}{sufixo}.png", base,
                coluna_cor="cobertura_pct", cmap=CMAP_COBERTURA, escala=ESCALA_COBERTURA,
                rotulo_cor="Cobertura da demanda do setor (%)",
                largura_borda=0.15, contorno=bairros, limites=recorte,
                pontos=[
                    {
                        "data": abertos, "cor": "#111111", "marcador": "*",
                        "tamanho": 170, "borda": "white", "largura_borda": 0.8,
                        "rotulo": (f"Ponto de retirada — {int(dentro.sum())} de {p} no recorte "
                                   f"(r = {r} m, {METRICA_REDE[cenario]})"),
                    }
                ],
            )


def limites_da_pagina(bairros: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Extensao do mapa principal com a proporcao exata da pagina (16 x 22 cm).

    A altura da terra, com folga, ocupa a altura toda; a largura sai da proporcao
    da pagina. Como o municipio e mais estreito que a pagina, a sobra fica a leste,
    sobre o oceano — e ali que entram a legenda e o recorte ampliado.
    """
    xmin, ymin, xmax, ymax = bairros.total_bounds
    y0, y1 = ymin - FOLGA_PAGINA_M, ymax + FOLGA_PAGINA_M
    metros_por_cm = (y1 - y0) / PAGINA_CM[1]
    x0 = xmin - FOLGA_PAGINA_M
    x1 = x0 + PAGINA_CM[0] * metros_por_cm
    checar(x1 >= xmax + FOLGA_PAGINA_M, "A pagina nao comporta a largura do municipio.")
    return x0, y0, x1, y1


def recorte_ampliado_da_pagina(
    bairros: gpd.GeoDataFrame, abertos_todos: gpd.GeoDataFrame
) -> tuple[float, float, float, float]:
    """Recorte ampliado FIXO, o mesmo nos cinco cenarios.

    Cobre o Continente inteiro e todos os pontos abertos em Centro/Sede ou
    Continente em QUALQUER dos cinco cenarios, com folga. A caixa inteira dos
    bairros dessas regioes (usada na fig06, ampliada) tem 16 x 16 km e se estende
    ate areas sem ponto nenhum; na pagina, isso encolheria o recorte a quase a
    escala do mapa principal.
    """
    continente = bairros[bairros["REGIAO_FUNCIONAL"] == "Continente"].total_bounds
    pontos = abertos_todos[abertos_todos["REGIAO_FUNCIONAL"].isin(REGIOES_RECORTE_AMPLIADO)]
    xs = np.concatenate([[continente[0], continente[2]], pontos.geometry.x.to_numpy()])
    ys = np.concatenate([[continente[1], continente[3]], pontos.geometry.y.to_numpy()])
    return (xs.min() - FOLGA_RECORTE_M, ys.min() - FOLGA_RECORTE_M,
            xs.max() + FOLGA_RECORTE_M, ys.max() + FOLGA_RECORTE_M)


def posicionar_quadro_recorte(
    limites_pagina: tuple[float, float, float, float],
    recorte: tuple[float, float, float, float],
    terra,
) -> tuple[float, float, float, float]:
    """Maior quadro possivel no canto inferior direito que NAO cubra terra.

    O quadro do recorte ampliado fica sobre o oceano a sudeste da ilha. Parte da
    largura maxima da pagina e reduz em passos de 0,5 mm ate o quadro (convertido
    para coordenadas do mapa principal) ficar a mais de AFASTAMENTO_TERRA_M da
    costa. Devolve (esquerda, base, largura, altura) em centimetros.
    """
    x0, y0, _, y1 = limites_pagina
    metros_por_cm = (y1 - y0) / PAGINA_CM[1]
    razao = (recorte[2] - recorte[0]) / (recorte[3] - recorte[1])
    terra_afastada = terra.buffer(AFASTAMENTO_TERRA_M)

    largura = PAGINA_CM[0] - 2 * MARGEM_QUADRO_CM
    while largura > 3.0:
        altura = largura / razao
        direita = PAGINA_CM[0] - MARGEM_QUADRO_CM
        esquerda = direita - largura
        base = MARGEM_QUADRO_CM
        quadro = box(
            x0 + esquerda * metros_por_cm, y0 + base * metros_por_cm,
            x0 + direita * metros_por_cm, y0 + (base + altura) * metros_por_cm,
        )
        if not quadro.intersects(terra_afastada):
            return esquerda, base, largura, altura
        largura -= 0.05
    raise FalhaDeSanidade("Nao ha espaco sobre o oceano para o recorte ampliado.")


def rotulos_de_regiao(bairros: gpd.GeoDataFrame, deslocamentos: dict, tamanho: float) -> list[dict]:
    """Rotulo de cada regiao funcional no ponto representativo interno, deslocado."""
    regioes = bairros.dissolve(by="REGIAO_FUNCIONAL")
    rotulos = []
    for regiao, (dx, dy) in deslocamentos.items():
        ponto = regioes.loc[regiao, "geometry"].representative_point()
        rotulos.append({"texto": regiao.upper(), "x": ponto.x + dx, "y": ponto.y + dy, "tamanho": tamanho})
    return rotulos


def legenda_da_pagina() -> list:
    """Legenda FIXA, identica nos cinco cenarios — inclusive os tipos sem ponto aberto.

    Uma legenda que mudasse de cenario para cenario atrapalharia a comparacao, que e
    o objetivo das cinco figuras. As linhas sem simbolo sao cabecalhos de grupo.
    """
    def cabecalho(texto):
        return Line2D([], [], linestyle="none", label=texto)

    itens = [
        cabecalho("Setores censitários"),
        Patch(facecolor=COR_COBERTO, edgecolor="#6e6e6e", linewidth=0.4, label="Coberto"),
        Patch(facecolor=COR_NAO_COBERTO, edgecolor="#6e6e6e", linewidth=0.4, label="Não coberto"),
        Patch(facecolor=COR_SEM_POPULACAO, edgecolor="#6e6e6e", linewidth=0.4, label="Sem população"),
        Line2D([0, 1], [0, 1], color="#3c3c3c", linewidth=0.6, label="Limite de bairro"),
        Patch(facecolor="none", edgecolor="black", linewidth=1.0, label="Área do recorte ampliado"),
        cabecalho("Ponto de retirada aberto"),
    ]
    for tipo, marcador in MARCADOR_TIPO.items():
        itens.append(
            Line2D([0], [0], marker=marcador, linestyle="none", markersize=6.5,
                   markerfacecolor=CORES_TIPO[tipo], markeredgecolor="black",
                   markeredgewidth=0.5, label=ROTULO_TIPO[tipo])
        )
    return itens


def camadas_por_tipo(abertos: gpd.GeoDataFrame, tamanho: float) -> list[dict]:
    """Uma camada de pontos por tipo de estabelecimento, com cor e forma proprias."""
    camadas = []
    for tipo, marcador in MARCADOR_TIPO.items():
        camadas.append({
            "data": abertos[abertos["TIPO"] == tipo], "cor": CORES_TIPO[tipo],
            "marcador": marcador, "tamanho": tamanho, "borda": "black",
            "largura_borda": 0.5, "rotulo": ROTULO_TIPO[tipo], "zorder": 6,
        })
    checar(
        sum(len(c["data"]) for c in camadas) == len(abertos),
        "Ha ponto aberto de tipo fora dos cinco previstos.",
    )
    return camadas


def caixa_em_metros(ax, artista) -> object:
    """Caixa de um elemento desenhado (legenda, eixo) em coordenadas do mapa."""
    janela = artista.get_window_extent()
    (xa, ya), (xb, yb) = ax.transData.inverted().transform([[janela.x0, janela.y0], [janela.x1, janela.y1]])
    return box(xa, ya, xb, yb)


def figuras_pagina_cenarios(dados: dict) -> None:
    """Localizacoes otimas, uma figura por cenario, do tamanho de uma pagina A4.

    Mesma extensao, mesmo recorte ampliado, mesma paleta e mesma legenda nos cinco
    cenarios, para que a comparacao entre figuras seja direta. Os setores saem em
    duas classes, coberto e nao coberto — o modelo aloca o setor inteiro ou nada.
    """
    subtitulo("5a. Localizacoes otimas em pagina inteira (16 x 22 cm) — fig04a a fig04e")

    bairros = dados["bairros"]
    setores = dados["setores"]
    terra = bairros.union_all()

    limites_pagina = limites_da_pagina(bairros)
    metros_por_cm = (limites_pagina[3] - limites_pagina[1]) / PAGINA_CM[1]

    solucoes = dados["solucoes"]
    abertos_todos = pd.concat([
        solucoes[(solucoes["CENARIO"] == c) & (solucoes["METRICA"] == METRICA_REDE[c])
                 & (solucoes["VARIANTE"] == "aj_estimado")]
        for c in ORDEM_CENARIOS
    ])
    recorte = recorte_ampliado_da_pagina(bairros, abertos_todos)
    esquerda, base_cm, largura, altura = posicionar_quadro_recorte(limites_pagina, recorte, terra)
    escala_recorte = (recorte[2] - recorte[0]) / largura

    print(f"   pagina: {PAGINA_CM[0]:.0f} x {PAGINA_CM[1]:.0f} cm · mapa principal 1:"
          f"{numero_br(metros_por_cm * 100, 0)}")
    print(f"   recorte ampliado: {numero_br((recorte[2] - recorte[0]) / 1000, 1)} x "
          f"{numero_br((recorte[3] - recorte[1]) / 1000, 1)} km · quadro de "
          f"{numero_br(largura, 1)} x {numero_br(altura, 1)} cm · 1:{numero_br(escala_recorte * 100, 0)} "
          f"({numero_br(metros_por_cm / escala_recorte, 1)}x o mapa principal)")

    rotulos_principal = rotulos_de_regiao(bairros, DESLOCAMENTO_ROTULO_M, tamanho=8.5)
    rotulos_recorte = rotulos_de_regiao(bairros, DESLOCAMENTO_ROTULO_RECORTE_M, tamanho=8.5)
    categorias = {"coberto": COR_COBERTO, "nao_coberto": COR_NAO_COBERTO}
    caixa_recorte = box(*recorte)

    print(f"\n   {'arquivo':<16} {'p':>3} {'r (m)':>6} {'rede':>6} {'setores cobertos':>17} "
          f"{'demanda coberta':>16} {'pontos no recorte':>18}")
    for cenario in cenarios_a_mapear(dados):
        base, abertos, p, r = dados_do_cenario(dados, cenario, setores, "CD_SETOR")
        if cenario == CENARIO_MISTO:
            # O recorte e o dos cinco cenarios originais. Ponto do cenario misto em
            # Centro/Sede ou Continente fora dele ficaria sem ampliacao.
            centrais = abertos[abertos["REGIAO_FUNCIONAL"].isin(REGIOES_RECORTE_AMPLIADO)]
            fora = int((~centrais.geometry.within(caixa_recorte)).sum())
            checar(fora == 0, f"{cenario}: {fora} ponto(s) de Centro/Sede ou Continente fora do recorte ampliado fixo.")

        # Duas classes. Conferir antes que nao ha cobertura parcial de verdade: se
        # houvesse, o mapa binario esconderia informacao.
        pct = base["cobertura_pct"]
        parcial = pct.notna() & (pct > 0.1) & (pct < 99.9)
        checar(not bool(parcial.any()), f"{cenario}: {int(parcial.sum())} setor(es) com cobertura parcial.")
        base["CLASSE"] = np.where(pct.isna(), None, np.where(pct >= LIMIAR_COBERTO_PCT, "coberto", "nao_coberto"))
        checar(
            int((base["CLASSE"].notna()).sum()) == int((base["POP"] > 0).sum()),
            f"{cenario}: setor habitado sem classe de cobertura.",
        )

        arquivo = ARQUIVOS_PAGINA[cenario]
        # figsize a partir do numero INTEIRO de pixels: o matplotlib trunca, e
        # 16 cm a 300 dpi (1889,8 px) sairia com 1889.
        pixels = (round(PAGINA_CM[0] * CM * DPI), round(PAGINA_CM[1] * CM * DPI))
        figura = plt.figure(figsize=(pixels[0] / DPI, pixels[1] / DPI))

        # Mapa principal: a pagina inteira.
        ax = figura.add_axes([0, 0, 1, 1])
        mapa(
            arquivo, base, coluna_cor="CLASSE", categorias=categorias,
            largura_borda=0.05, contorno=bairros, largura_contorno=0.3,
            limites=limites_pagina, ax=ax,
            pontos=camadas_por_tipo(abertos, TAMANHO_PONTO_PAGINA),
            rotulos=rotulos_principal,
            posicao_seta=POSICAO_SETA_PAGINA, posicao_escala=POSICAO_ESCALA_PAGINA,
            comprimento_escala=0.12,
        )
        ax.add_patch(Rectangle(
            (recorte[0], recorte[1]), recorte[2] - recorte[0], recorte[3] - recorte[1],
            fill=False, edgecolor="black", linewidth=1.0, zorder=8,
        ))

        # Recorte ampliado, sobre o oceano a sudeste.
        ax_recorte = figura.add_axes([
            esquerda / PAGINA_CM[0], base_cm / PAGINA_CM[1],
            largura / PAGINA_CM[0], altura / PAGINA_CM[1],
        ])
        mapa(
            f"{arquivo} (recorte)", base, coluna_cor="CLASSE", categorias=categorias,
            largura_borda=0.1, contorno=bairros, largura_contorno=0.45,
            limites=recorte, ax=ax_recorte,
            pontos=camadas_por_tipo(abertos, TAMANHO_PONTO_RECORTE),
            rotulos=rotulos_recorte, posicao_seta=None,
            posicao_escala="lower right", comprimento_escala=0.25,
        )
        # Moldura do recorte: o eixo volta a ser visivel, sem marcas nem numeros.
        ax_recorte.set_axis_on()
        ax_recorte.set_xticks([])
        ax_recorte.set_yticks([])
        for borda in ax_recorte.spines.values():
            borda.set_edgecolor("black")
            borda.set_linewidth(1.0)

        legenda = ax.legend(
            handles=legenda_da_pagina(), loc="upper right",
            bbox_to_anchor=(1 - MARGEM_QUADRO_CM / PAGINA_CM[0], 1 - MARGEM_QUADRO_CM / PAGINA_CM[1]),
            frameon=True, framealpha=0.95, edgecolor="#8c8c8c", fancybox=False,
            fontsize=8, borderpad=0.8, labelspacing=0.55, handlelength=1.6,
        )
        legenda.set_zorder(10)
        for texto, item in zip(legenda.get_texts(), legenda.legend_handles):
            if item.get_label() in ("Setores censitários", "Ponto de retirada aberto"):
                texto.set_fontweight("bold")

        # Legenda e recorte ficam sobre o oceano: nenhum dos dois pode cobrir
        # terra do municipio, nem um ao outro.
        figura.canvas.draw()
        caixa_legenda = caixa_em_metros(ax, legenda)
        caixa_quadro = caixa_em_metros(ax, ax_recorte)
        checar(not caixa_legenda.intersects(terra), f"{arquivo}: a legenda cobre terra do municipio.")
        checar(not caixa_quadro.intersects(terra), f"{arquivo}: o recorte ampliado cobre terra do municipio.")
        checar(not caixa_legenda.intersects(caixa_quadro), f"{arquivo}: legenda e recorte se sobrepoem.")

        DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
        caminho = DIR_FIGURAS / arquivo
        # Sem bbox_inches="tight": a imagem tem de sair com o tamanho exato da pagina.
        figura.savefig(caminho, dpi=DPI)
        plt.close(figura)
        registrar_figura(caminho)
        imagem = mpimg.imread(caminho)
        esperado = (pixels[1], pixels[0])
        checar(imagem.shape[:2] == esperado,
               f"{arquivo}: {imagem.shape[1]}x{imagem.shape[0]} px, esperado {esperado[1]}x{esperado[0]}.")

        no_recorte = int(abertos.geometry.within(caixa_recorte).sum())
        cobertos = int((base["CLASSE"] == "coberto").sum())
        indicadores = dados["indicadores_cenarios"]
        cobertura_pct = float(indicadores[
            (indicadores["cenario"] == cenario) & (indicadores["metrica"] == METRICA_REDE[cenario])
            & (indicadores["variante"] == "aj_estimado") & (indicadores["grupo"] == "matriz_principal")
        ]["cobertura_pct_municipio"].iloc[0])
        print(f"   {arquivo:<16} {p:>3} {r:>6} {METRICA_REDE[cenario]:>6} "
              f"{cobertos:>10} de {int((base['POP'] > 0).sum())} {numero_br(cobertura_pct, 1):>15}% "
              f"{no_recorte:>11} de {p}")


def carregar_pontes() -> gpd.GeoDataFrame:
    """As tres travessias Ilha <-> Continente, do grafo viario em cache.

    Sao lidas do mesmo grafo que gerou a matriz de distancias, pelos nomes ja
    conferidos em 05_distancias.py. Nada e baixado do OSM aqui.
    """
    import osmnx as ox

    checar(ARQ_GRAFO_DRIVE.exists(), f"{ARQ_GRAFO_DRIVE} nao existe. Rodar src/05_distancias.py.")
    arestas = ox.graph_to_gdfs(ox.load_graphml(ARQ_GRAFO_DRIVE), nodes=False)
    nomes = arestas["name"].astype(str)
    partes = []
    for ponte in PONTES:
        sub = arestas[nomes.str.contains(ponte, case=False, regex=False)]
        checar(len(sub) > 0, f"A {ponte} nao esta no grafo viario em cache.")
        partes.append(sub[["geometry"]].assign(PONTE=ponte))
    return gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=arestas.crs).to_crs(CRS_TRABALHO)


def figura_artigo(dados: dict) -> None:
    """Densidade demografica por bairro com as cinco regioes funcionais — figura do artigo.

    Mesmo dado da fig01 (densidade por bairro), no formato de pagina das figuras de
    cenario: 16 cm de largura, municipio na altura toda, legenda e barra de cores
    sobre o oceano a leste. Os limites das regioes funcionais vao por cima, mais
    grossos, com o nome de cada regiao; as tres pontes sao desenhadas e indicadas.
    Sem titulo dentro da imagem.
    """
    subtitulo("4b. Figura do artigo — densidade demografica e regioes funcionais")

    bairros = dados["bairros"]
    terra = bairros.union_all()
    verificar_total(bairros["POP"], POP_REFERENCIA, "populacao")
    checar(
        bool(np.allclose(bairros["DENS_HAB_KM2"], bairros["POP"] / bairros["AREA_KM2"], rtol=1e-6)),
        "A coluna DENS_HAB_KM2 nao corresponde a POP / AREA_KM2.",
    )
    regioes = bairros[["REGIAO_FUNCIONAL", "POP", "geometry"]].dissolve(by="REGIAO_FUNCIONAL", aggfunc="sum")
    checar(sorted(regioes.index) == sorted(ORDEM_REGIOES), "As regioes funcionais nao sao as cinco esperadas.")
    pontes = carregar_pontes()

    x0, y0, x1, y1 = limites_da_pagina(bairros)
    desloca = FOLGA_OESTE_ARTIGO_M - FOLGA_PAGINA_M
    limites_pagina = (x0 - desloca, y0, x1 - desloca, y1)
    checar(limites_pagina[2] >= bairros.total_bounds[2] + FOLGA_PAGINA_M,
           "O deslocamento para oeste tirou parte do municipio da pagina.")
    pixels = (round(PAGINA_CM[0] * CM * DPI), round(PAGINA_CM[1] * CM * DPI))
    figura = plt.figure(figsize=(pixels[0] / DPI, pixels[1] / DPI))
    ax = figura.add_axes([0, 0, 1, 1])

    vmin, vmax = float(bairros["DENS_HAB_KM2"].min()), float(bairros["DENS_HAB_KM2"].max())
    mapa(
        ARQUIVO_ARTIGO, bairros, coluna_cor="DENS_HAB_KM2", cmap=CMAP_DENSIDADE,
        escala=(vmin, vmax), barra_cores=False, largura_borda=0.25,
        contorno=regioes, largura_contorno=1.6, cor_contorno="#111111",
        limites=limites_pagina, ax=ax,
        rotulos=rotulos_de_regiao(bairros, DESLOCAMENTO_ROTULO_ARTIGO_M, tamanho=9.5),
        posicao_seta=POSICAO_SETA_PAGINA, posicao_escala=POSICAO_ESCALA_PAGINA,
        comprimento_escala=0.12,
    )

    # Pontes: linha sobre o estreito e uma unica indicacao, porque as tres ficam a
    # menos de 700 m umas das outras e na escala do municipio se confundem. Os
    # nomes vao na legenda; no mapa, so um rotulo curto na Baia Sul.
    pontes.plot(ax=ax, color=COR_PONTE, linewidth=1.8, zorder=6)
    x0, y0, x1, y1 = pontes.total_bounds
    ax.annotate(
        "Pontes", xy=((x0 + x1) / 2, y0), xytext=((x0 + x1) / 2 + 300.0, y0 - 4_200.0),
        ha="center", va="top", fontsize=8.5, fontweight="bold", color=COR_PONTE, zorder=7,
        arrowprops=dict(arrowstyle="-", color=COR_PONTE, linewidth=0.9),
        path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")],
    )

    # Legenda e barra de cores sobre o oceano, a sudeste da ilha: a legenda logo
    # acima da barra. No canto superior direito ela cobriria ilhas do municipio.
    legenda = ax.legend(
        handles=[
            Line2D([0, 1], [0, 1], color="#111111", linewidth=1.6, label="Limite de região funcional"),
            Line2D([0, 1], [0, 1], color="#6e6e6e", linewidth=0.5, label="Limite de bairro"),
            Line2D([0, 1], [0, 1], color=COR_PONTE, linewidth=1.8, label="Pontes Hercílio Luz, Colombo\nSalles e Pedro Ivo"),
        ],
        loc="lower right",
        bbox_to_anchor=(1 - MARGEM_QUADRO_CM / PAGINA_CM[0], 0.385),
        frameon=True, framealpha=0.95, edgecolor="#8c8c8c", fancybox=False,
        fontsize=8.5, borderpad=0.8, labelspacing=0.6, handlelength=1.8,
    )
    legenda.set_zorder(10)

    # Barra de cores num eixo proprio, no oceano a sudeste — mesma posicao do
    # recorte ampliado nas figuras de cenario.
    ax_barra = figura.add_axes([0.80, 0.045, 0.035, 0.30])
    barra = figura.colorbar(
        matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=CMAP_DENSIDADE),
        cax=ax_barra,
    )
    barra.set_label("Densidade demográfica (hab/km²)", fontsize=9)
    barra.ax.tick_params(labelsize=8.5)
    barra.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda valor, _: numero_br(valor, 0))
    )

    # Nada pode cobrir terra do municipio: legenda, barra de cores e seu rotulo.
    figura.canvas.draw()
    caixa_legenda = caixa_em_metros(ax, legenda)
    janela = barra.ax.get_tightbbox(figura.canvas.get_renderer())
    (xa, ya), (xb, yb) = ax.transData.inverted().transform([[janela.x0, janela.y0], [janela.x1, janela.y1]])
    caixa_barra = box(xa, ya, xb, yb)
    checar(not caixa_legenda.intersects(terra), f"{ARQUIVO_ARTIGO}: a legenda cobre terra do municipio.")
    checar(not caixa_barra.intersects(terra), f"{ARQUIVO_ARTIGO}: a barra de cores cobre terra do municipio.")
    checar(not caixa_legenda.intersects(caixa_barra), f"{ARQUIVO_ARTIGO}: legenda e barra de cores se sobrepoem.")

    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = DIR_FIGURAS / ARQUIVO_ARTIGO
    figura.savefig(caminho, dpi=DPI)   # tamanho exato da pagina, sem bbox_inches="tight"
    plt.close(figura)
    registrar_figura(caminho)

    print(f"   {PAGINA_CM[0]:.0f} x {PAGINA_CM[1]:.0f} cm a {DPI} dpi · {len(bairros)} bairros · "
          f"densidade de {numero_br(vmin, 0)} a {numero_br(vmax, 0)} hab/km2")
    for regiao in ORDEM_REGIOES:
        print(f"      {regiao:<14} {numero_br(regioes.loc[regiao, 'POP'], 0):>9} hab")
    print(f"   pontes desenhadas: {', '.join(PONTES)}")


def figuras_ponderacao(dados: dict) -> None:
    subtitulo("6. Comparacao das ponderacoes em C5")

    bairros = dados["bairros"]
    solucoes = dados["solucoes"]

    # A metrica de REDE e a que entra no texto; a euclidiana fica como material
    # de apoio. Mostrar o caso de maior efeito na metrica que nao sustenta as
    # conclusoes seria escolher o numero mais favoravel.
    for metrica, papel in [("drive", "resultado"), ("euclid", "apoio")]:
        estimado = solucoes[
            (solucoes["CENARIO"] == "C5") & (solucoes["METRICA"] == metrica)
            & (solucoes["VARIANTE"] == "aj_estimado")
        ]
        unitario = solucoes[
            (solucoes["CENARIO"] == "C5") & (solucoes["METRICA"] == metrica)
            & (solucoes["VARIANTE"] == "aj_unitario")
        ]
        chave_est = set(estimado["OSM_ID"].astype(str) + "|" + estimado["NOME"].astype(str))
        chave_uni = set(unitario["OSM_ID"].astype(str) + "|" + unitario["NOME"].astype(str))
        comuns = chave_est & chave_uni

        def marcar(camada):
            return (camada["OSM_ID"].astype(str) + "|" + camada["NOME"].astype(str)).isin(comuns)

        n_mudam = len(chave_est - comuns)
        print(f"   C5/{metrica} ({papel}): {n_mudam} ponto(s) mudam de local")

        mapa(
            f"fig07_ponderacao_C5_{metrica}{dados['sufixo']}.png", bairros,
            pontos=[
                {
                    "data": estimado[marcar(estimado)], "cor": "#9e9e9e", "marcador": "o",
                    "tamanho": 60, "rotulo": f"Escolhido pelas duas variantes ({len(comuns)})",
                    "zorder": 4,
                },
                {
                    "data": unitario[~marcar(unitario)], "cor": "#1f4e79", "marcador": "s",
                    "tamanho": 110, "rotulo": f"Só com a_j = 1 ({n_mudam})", "zorder": 6,
                },
                {
                    "data": estimado[~marcar(estimado)], "cor": "#c0392b", "marcador": "D",
                    "tamanho": 110, "rotulo": f"Só com a_j estimado ({n_mudam})", "zorder": 6,
                },
            ],
        )


# =============================================================================
# GRAFICOS
# =============================================================================


def _salvar_grafico(figura, arquivo: str) -> None:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = DIR_FIGURAS / arquivo
    figura.savefig(caminho, dpi=DPI, bbox_inches="tight")
    plt.close(figura)
    registrar_figura(caminho)
    print(f"   {arquivo:<44} grafico")


def grafico_cobertura_regiao(dados: dict) -> None:
    subtitulo("7. Graficos")

    tabela = dados["cobertura_por_regiao_cenarios"]
    tabela = tabela[(tabela["variante"] == "aj_estimado") & (tabela["metrica"] != "euclid")]
    pivo = tabela.pivot_table(
        index="regiao_funcional", columns="cenario", values="cobertura_pct"
    ).reindex(ORDEM_REGIOES)[ORDEM_CENARIOS]

    figura, ax = plt.subplots(figsize=(10, 5.2))
    largura = 0.16
    posicoes = np.arange(len(ORDEM_REGIOES))
    cores = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(ORDEM_CENARIOS)))
    for i, cenario in enumerate(ORDEM_CENARIOS):
        ax.bar(
            posicoes + (i - 2) * largura, pivo[cenario], largura,
            label=cenario, color=cores[i], edgecolor="white", linewidth=0.5,
        )
    ax.set_xticks(posicoes)
    ax.set_xticklabels(ORDEM_REGIOES)
    ax.set_ylabel("Cobertura da demanda (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    _salvar_grafico(figura, f"fig08_cobertura_por_regiao{dados['sufixo']}.png")


def grafico_tradeoff(dados: dict) -> None:
    ind = dados["indicadores_cenarios"]
    ind = ind[
        (ind["grupo"] == "matriz_principal") & (ind["variante"] == "aj_estimado")
        & (ind["metrica"] != "euclid")
    ].set_index("cenario").reindex(ORDEM_CENARIOS)

    figura, ax = plt.subplots(figsize=(8.5, 5.4))
    dispersao = ax.scatter(
        ind["p"], ind["cobertura_pct_municipio"],
        s=ind["dist_media_ponderada_m"] * 1.1, c=ind["r_m"],
        cmap="plasma", alpha=0.85, edgecolor="black", linewidth=0.8,
    )
    # Rotulo a DIREITA da bolha, numa linha so. Acima dela, os rotulos de C2, C4 e
    # C5 — todos com p = 20 — se sobrepunham e o de C4 caia sobre a bolha de C2.
    for cenario, linha in ind.iterrows():
        raio_pt = np.sqrt(linha["dist_media_ponderada_m"] * 1.1 / np.pi)
        ax.annotate(
            f"{cenario} · {linha['dist_media_ponderada_m']:.0f} m",
            (linha["p"], linha["cobertura_pct_municipio"]),
            textcoords="offset points", xytext=(raio_pt + 5, 0),
            ha="left", va="center", fontsize=8.5,
        )
    ax.set_xlabel("Pontos abertos (p)")
    ax.set_ylabel("Cobertura da demanda municipal (%)")
    ax.set_xlim(0, 52)
    # Faixa derivada dos dados: fixa em 35-95%, a rodada por setor ficaria fora.
    baixo = max(0.0, 5 * np.floor((ind["cobertura_pct_municipio"].min() - 8) / 5))
    alto = min(100.0, 5 * np.ceil((ind["cobertura_pct_municipio"].max() + 8) / 5))
    ax.set_ylim(baixo, alto)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    barra = figura.colorbar(dispersao, ax=ax, fraction=0.04, pad=0.02)
    barra.set_label("Raio de cobertura r (m)")
    ax.scatter([], [], s=200 * 1.1, c="lightgray", edgecolor="black",
               label="área ∝ distância média ponderada")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    _salvar_grafico(figura, f"fig09_tradeoff{dados['sufixo']}.png")


def graficos_efeito_atratividade(dados: dict) -> None:
    """Duas figuras SEPARADAS: juntar as series reintroduziria o confundimento.

    Os cenarios variam p e r ao mesmo tempo. C1-C2-C3 tem pares elegiveis fixos e
    p variavel; C2-C4-C5 tem p fixo e pares variaveis. Num grafico so, nao haveria
    como saber qual dos dois efeitos produz a variacao.
    """
    efeito = dados["efeito_atratividade"]
    efeito = efeito[efeito["metrica"] != "euclid"].set_index("cenario")

    serie_p = efeito.reindex(["C1", "C2", "C3"])
    figura, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(serie_p["p"], serie_p["pontos_que_mudam"], marker="o", color="#1f4e79", linewidth=2)
    for cenario, linha in serie_p.iterrows():
        ax.annotate(f"{cenario}\n({linha['pares_elegiveis']:.0f} pares)",
                    (linha["p"], linha["pontos_que_mudam"]),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8.5)
    ax.set_xlabel(f"Pontos abertos (p) — pares elegíveis fixos em {serie_p['pares_elegiveis'].iloc[0]:.0f}")
    ax.set_ylabel("Pontos que mudam de local")
    ax.set_ylim(-0.5, max(7, serie_p["pontos_que_mudam"].max() + 2))
    ax.margins(x=0.15)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _salvar_grafico(figura, f"fig10a_efeito_de_p{dados['sufixo']}.png")

    serie_pares = efeito.reindex(["C4", "C2", "C5"]).sort_values("pares_elegiveis")
    figura, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(serie_pares["pares_elegiveis"], serie_pares["pontos_que_mudam"],
            marker="s", color="#c0392b", linewidth=2)
    for cenario, linha in serie_pares.iterrows():
        ax.annotate(f"{cenario}\n(r = {linha['r_m']:.0f} m)",
                    (linha["pares_elegiveis"], linha["pontos_que_mudam"]),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8.5)
    ax.set_xlabel("Pares elegíveis — p fixo em 20")
    ax.set_ylabel("Pontos que mudam de local")
    ax.set_ylim(-0.5, max(7, serie_pares["pontos_que_mudam"].max() + 2))
    # Folga lateral: sem ela os rotulos de C4 e C5 encostavam nas bordas do eixo.
    ax.margins(x=0.15)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _salvar_grafico(figura, f"fig10b_efeito_dos_pares{dados['sufixo']}.png")


def grafico_cobertura_vs_teto(dados: dict) -> None:
    ind = dados["indicadores_cenarios"]
    ind = ind[
        (ind["grupo"] == "matriz_principal") & (ind["variante"] == "aj_estimado")
        & (ind["metrica"] != "euclid")
    ].set_index("cenario").reindex(ORDEM_CENARIOS)

    figura, ax = plt.subplots(figsize=(8.5, 5.0))
    posicoes = np.arange(len(ORDEM_CENARIOS))
    largura = 0.36
    ax.bar(posicoes - largura / 2, ind["cobertura_pct_municipio"], largura,
           label="% da demanda municipal", color="#4c72b0", edgecolor="white")
    ax.bar(posicoes + largura / 2, ind["cobertura_pct_teto"], largura,
           label="% do teto atingível no raio", color="#dd8452", edgecolor="white")
    # O teto vai ACIMA do par de barras. Na base, o texto ficava cortado pela
    # divisa entre as duas barras.
    for i, cenario in enumerate(ORDEM_CENARIOS):
        teto = 100 * ind.loc[cenario, "teto_Z1"] / ind.loc[cenario, "demanda_total"]
        topo = max(ind.loc[cenario, "cobertura_pct_municipio"], ind.loc[cenario, "cobertura_pct_teto"])
        ax.annotate(f"teto {teto:.1f}%".replace(".", ","), (i, topo + 1.5),
                    ha="center", va="bottom", fontsize=8, color="#444444")
    ax.set_xticks(posicoes)
    ax.set_xticklabels(
        [f"{c}\np={ind.loc[c, 'p']:.0f}, r={ind.loc[c, 'r_m']:.0f} m" for c in ORDEM_CENARIOS]
    )
    ax.set_ylabel("Cobertura (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    _salvar_grafico(figura, f"fig11_cobertura_municipio_vs_teto{dados['sufixo']}.png")


# =============================================================================
# RESUMO
# =============================================================================


def resumo_final() -> None:
    titulo("Figuras geradas")
    print(f"   {'arquivo':<46} {'dimensoes':<22} {'pasta'}")
    for nome, dimensoes, pasta in FIGURAS_GERADAS:
        print(f"   {nome:<46} {dimensoes:<22} {pasta}")
    por_pasta = pd.Series([pasta for _, _, pasta in FIGURAS_GERADAS]).value_counts()
    print(f"\n   total: {len(FIGURAS_GERADAS)} figuras, {DPI} dpi — "
          + " · ".join(f"{n} em {pasta}/" for pasta, n in por_pasta.items()))

    print()
    print(f"   mapa base adotado: {NOME_FONTE_BASEMAP}")
    for rejeitada in FONTES_REJEITADAS:
        print(f"      rejeitada — {rejeitada}")

    print()
    if FIGURAS_SEM_BASE:
        print("!" * 78)
        print(f"ATENCAO: {len(FIGURAS_SEM_BASE)} figura(s) foram geradas SEM MAPA BASE.")
        print("Os dados estao corretos, mas o fundo cartografico nao carregou.")
        print("NAO inserir no Word sem regerar com rede disponivel:")
        for nome in FIGURAS_SEM_BASE:
            print(f"   - {nome}")
        print("!" * 78)
    else:
        print("   mapa base carregado em todas as figuras cartograficas.")


# =============================================================================
# EXECUCAO
# =============================================================================


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Mapas e graficos da monografia.")
    parser.add_argument("--unidade", choices=sorted(UNIDADES), default="setor",
                        help="unidade de demanda: setor censitario (padrao) ou bairro")
    parser.add_argument("--figuras", choices=["todas", "paginas", "artigo"], default="todas",
                        help="'paginas' gera so as figuras de pagina inteira (fig04a a fig04e e, "
                             "se houver, a do cenario misto); 'artigo', so a figura do artigo")
    parser.add_argument("--saida", default=None,
                        help="versao da rodada (ex.: v36): le de data/tratados/<saida>/ e "
                             "results/<saida>/tabelas/ e grava em results/<saida>/figuras/")
    argumentos = parser.parse_args()
    unidade = argumentos.unidade
    checar(
        argumentos.figuras == "todas" or unidade == "setor",
        "As figuras de pagina inteira e a do artigo existem so na rodada por setor.",
    )
    global DIR_TABELAS, DIR_FIGURAS, DIR_FIGURAS_NAO_USADAS, ARQ_CANDIDATOS, ARQ_SOLUCOES
    base = DIR_RESULTADOS[unidade]
    if argumentos.saida:
        raiz_resultados = RAIZ / "results"
        base = raiz_resultados / argumentos.saida / base.relative_to(raiz_resultados)
        DIR_FIGURAS_NAO_USADAS = raiz_resultados / argumentos.saida / "nao_usados" / "figuras"
        ARQ_CANDIDATOS = DIR_TRATADOS / argumentos.saida / ARQ_CANDIDATOS.name
        ARQ_SOLUCOES = DIR_TRATADOS / argumentos.saida / ARQ_SOLUCOES.name
    DIR_TABELAS = base / "tabelas"
    DIR_FIGURAS = base / "figuras"

    inicio = time.perf_counter()
    titulo(
        f"08_mapas.py — mapas e graficos · unidade de demanda: {unidade}\n"
        f"EPSG:{CRS_TRABALHO} · {DPI} dpi · sem ax.set_title(), por exigencia da ABNT"
    )

    dados = carregar(unidade)
    escolher_fonte_basemap(dados["bairros"])
    verificar_pontos_em_terra(dados)
    if argumentos.figuras in ("paginas", "artigo"):
        if argumentos.figuras == "paginas":
            figuras_pagina_cenarios(dados)
        else:
            figura_artigo(dados)
        resumo_final()
        print(f"\nConcluido em {time.perf_counter() - inicio:.1f} s")
        return 0

    if unidade == "setor":
        figuras_diagnostico(dados)
        figura_artigo(dados)
        figuras_pagina_cenarios(dados)
    figuras_cenarios(dados)
    figuras_ponderacao(dados)
    grafico_cobertura_regiao(dados)
    grafico_tradeoff(dados)
    graficos_efeito_atratividade(dados)
    grafico_cobertura_vs_teto(dados)

    resumo_final()
    print(f"\nConcluido em {time.perf_counter() - inicio:.1f} s")
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
