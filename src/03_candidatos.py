"""
03_candidatos.py — candidatos a ponto de retirada em Florianopolis/SC.

Extrai os estabelecimentos elegiveis a pickup point, trata a geometria, deduplica,
associa cada candidato ao seu bairro e valida o conjunto contra o CNEFE 2022.

Cinco tipos (Secoes 3.2.2 e 3.2.4): supermercados, conveniencias, farmacias,
agencias postais e varejo alimentar de pequeno porte. Os quatro primeiros vem do
OpenStreetMap; o quinto vem do CNEFE, unica base com cobertura desse segmento. A
coluna FONTE distingue as duas origens.

Entradas:
    data/tratados/bairros.gpkg              territorio e bairros (Etapa 2)
    data/brutos/ibge/SC_setores_CD2022.gpkg massas d'agua (CD_SIT = 9)
    data/brutos/cnefe/4205407_FLORIANOPOLIS.csv   validacao cruzada
    OpenStreetMap via OSMnx                  na primeira execucao; depois, cache

Saidas:
    data/tratados/candidatos.gpkg                  candidatos tratados
    data/tratados/_cache_osm_candidatos.parquet    extracao bruta do OSM
    data/tratados/_cache_osm_metadados.json        data e porte da extracao
    results/tabelas/validacao_osm_cnefe.csv      taxas de correspondencia
    results/tabelas/validacao_osm_cnefe_por_regiao.csv   idem, por regiao
    results/tabelas/candidatos_por_bairro.csv    contagem por bairro e tipo
    results/tabelas/candidatos_com_sem_mercado_pequeno.csv   as duas rodadas
    results/tabelas/candidatos_descartados.csv   descartes da verificacao espacial
    results/tabelas/candidatos_fundidos_nome_aproximado.csv   fusoes por nome aproximado

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/03_candidatos.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import box

# =============================================================================
# PARAMETROS
# =============================================================================

RAIZ = Path(__file__).resolve().parents[1]
DIR_TRATADOS = RAIZ / "data" / "tratados"
DIR_TABELAS = RAIZ / "results" / "tabelas"

ARQ_BAIRROS = DIR_TRATADOS / "bairros.gpkg"
ARQ_SETORES = RAIZ / "data" / "brutos" / "ibge" / "SC_setores_CD2022.gpkg"
ARQ_CNEFE = RAIZ / "data" / "brutos" / "cnefe" / "4205407_FLORIANOPOLIS.csv"
ARQ_CANDIDATOS = DIR_TRATADOS / "candidatos.gpkg"

CD_MUN = "4205407"
CRS_TRABALHO = 31982
CD_SIT_MASSA_DAGUA = "9"

# -----------------------------------------------------------------------------
# Extracao do OpenStreetMap
# -----------------------------------------------------------------------------
# O cache e o artefato reprodutivel da etapa: sem ele o resultado muda a cada
# execucao, porque o OSM e editado continuamente, e a banca nao consegue
# reproduzir os numeros da monografia. Fica em data/tratados/ e nao em
# data/brutos/, que e somente leitura por convencao do projeto — mesmo criterio
# ja adotado para o recorte da planilha de renda na Etapa 2.
ARQ_CACHE_OSM = DIR_TRATADOS / "_cache_osm_candidatos.parquet"
ARQ_METADADOS_OSM = DIR_TRATADOS / "_cache_osm_metadados.json"
FORCAR_DOWNLOAD_OSM = False

# Tags consultadas e o tipo canonico de cada uma.
TAGS_OSM = {"shop": ["supermarket", "convenience"], "amenity": ["pharmacy", "post_office"]}
TIPO_POR_TAG = {
    ("shop", "supermarket"): "supermercado",
    ("shop", "convenience"): "conveniencia",
    ("amenity", "pharmacy"): "farmacia",
    ("amenity", "post_office"): "agencia_postal",
}
TIPOS_OSM = ["supermercado", "conveniencia", "farmacia", "agencia_postal"]
ATRIBUTOS_OSM = ["name", "brand", "operator", "opening_hours"]

# -----------------------------------------------------------------------------
# Quinto tipo: varejo alimentar de pequeno porte
# -----------------------------------------------------------------------------
# Mercearias, minimercados e armazens entram em J como TIPO PROPRIO, nunca
# fundidos a supermercado, para que a Etapa 5 possa lhes atribuir escore de
# atratividade distinto (Secoes 3.2.2 e 3.2.4 da monografia).
#
# A FONTE e o CNEFE, nao o OSM. Verificado na extracao: o OSM traz apenas 55
# estabelecimentos sob shop=grocery/general/greengrocer/deli/farm/wholesale/
# variety_store em Florianopolis, e sao hortifrutis, sacoloes e lojas de
# variedades — nao as mercearias e armazens que o CNEFE registra. O cadastro
# censitario e a unica base com cobertura desse segmento.
#
# Consequencia metodologica declarada em docs/decisoes_etapa4.md: os candidatos
# passam a vir de duas fontes, e a coluna FONTE distingue cada registro.
INCLUIR_MERCADO_PEQUENO = True
TIPO_MERCADO_PEQUENO = "mercado_pequeno"

TIPOS = TIPOS_OSM + ([TIPO_MERCADO_PEQUENO] if INCLUIR_MERCADO_PEQUENO else [])

# -----------------------------------------------------------------------------
# Remocoes nominais
# -----------------------------------------------------------------------------
# Estabelecimentos etiquetados no OSM com uma tag que nao corresponde a sua
# funcao. So entram aqui casos identificados NOMINALMENTE, com motivo declarado,
# nunca filtros genericos: remover por regra ampla arriscaria excluir candidatos
# legitimos.
#
# AZUL CARGO e TAM CARGO sao terminais de carga aerea no aeroporto, etiquetados
# amenity=post_office. Ficam a 7,5 km do registro de agencia postal mais proximo
# do CNEFE e nao sao ponto de retirada para o consumidor sob nenhuma leitura.
REMOVER_POR_NOME = {
    ("agencia_postal", "AZUL CARGO"): "terminal de carga aerea, nao agencia postal",
    ("agencia_postal", "TAM CARGO"): "terminal de carga aerea, nao agencia postal",
}

# -----------------------------------------------------------------------------
# Verificacao espacial
# -----------------------------------------------------------------------------
# Licao da Etapa 2: a verificacao que faltava nao era sobre o ponto, era sobre o
# poligono. O analogo aqui e o candidato com coordenada plausivel em local errado.
#
# Candidato fora do territorio mas a ate TOLERANCIA_BORDA_M da costa e mantido:
# em orla, o ponto do OSM cai frequentemente alguns metros na agua por impresisao
# de digitalizacao. Alem disso, ou sobre massa d'agua, e descartado e listado
# nominalmente.
TOLERANCIA_BORDA_M = 50.0

# A consulta ao Overpass usa a caixa envolvente do municipio, que inevitavelmente
# cobre Sao Jose, Palhoca, Biguacu e outros vizinhos. Feicoes a mais de
# LIMITE_OUTRO_MUNICIPIO_M da terra de Florianopolis sao de outra cidade: sao
# filtragem esperada, nao anomalia, e entram no resumo por contagem. Entre a
# tolerancia de borda e esse limite, o candidato e listado NOMINALMENTE, porque
# ali mora o caso que interessa: coordenada plausivel em local errado.
LIMITE_OUTRO_MUNICIPIO_M = 500.0

# A tolerancia de borda foi pensada para a ORLA, mas tambem deixava passar ponto a
# poucos metros alem de uma divisa TERRESTRE. Verificado: o supermercado
# IMPERATRIZ, atribuido a Capoeiras, fica 38 m dentro de Sao Jose (bairro
# Campinas). Por isso, antes da tolerancia, todo ponto que cai em setor censitario
# de terra de OUTRO municipio e descartado, a qualquer distancia.
FOLGA_BUSCA_VIZINHOS_M = 2000.0   # recorte da malha de SC em volta do municipio

# -----------------------------------------------------------------------------
# Deduplicacao
# -----------------------------------------------------------------------------
# Dois registros so colapsam se estiverem a menos de RAIO_DEDUP_M E tiverem o
# mesmo tipo E nome normalizado equivalente. Sem a condicao de nome, uma farmacia
# e um supermercado vizinhos num mesmo centro comercial virariam um candidato so.
# Filiais distintas da mesma rede a algumas centenas de metros permanecem
# separadas — sao pontos de retirada diferentes.
RAIO_DEDUP_M = 25.0

# -----------------------------------------------------------------------------
# Deduplicacao por nome aproximado (auditoria de setembro de 2026)
# -----------------------------------------------------------------------------
# A regra acima exige nome IDENTICO, e deixava passar o mesmo estabelecimento
# grafado de dois jeitos ("MINIMERCADO VASSORINHA" x "MINI MERCADO VASSOURINHA") ou
# registrado nas duas fontes ("SUPERMERCADO 5 MENINAS" no OSM x "MERCADO 05
# MENINAS" no CNEFE, a 29 m). Colapsam agora dois registros que atendam as TRES
# condicoes:
#   - distancia ate RAIO_DEDUP_APROX_M;
#   - mesmo GRUPO de tipo: varejo alimentar com varejo alimentar, farmacia com
#     farmacia. Farmacia e supermercado vizinhos nunca se fundem, mesmo com nome
#     parecido ("FARMACIA ANGELONI" x "ANGELONI", a 42 m, sao dois negocios);
#   - similaridade do nome, depois de retirar as palavras genericas, de pelo menos
#     SIMILARIDADE_MINIMA (razao de SequenceMatcher).
# Os limites vem da distribuicao observada: os 13 pares acima deles estao todos a
# ate 72 m e similaridade >= 0,84; o proximo par ja esta a 106 m com similaridade
# 0,74, e depois vem filiais de rede (PRECO POPULAR, a 109 m). Lista completa em
# results/tabelas/candidatos_fundidos_nome_aproximado.csv.
RAIO_DEDUP_APROX_M = 75.0
SIMILARIDADE_MINIMA = 0.80
TAMANHO_MINIMO_NOME = 3   # nome que, sem as palavras genericas, fica menor nao identifica ninguem
GRUPO_DE_TIPO = {
    "supermercado": "varejo_alimentar",
    "conveniencia": "varejo_alimentar",
    "mercado_pequeno": "varejo_alimentar",
    "farmacia": "farmacia",
    "agencia_postal": "agencia_postal",
}
PALAVRAS_GENERICAS = {
    "MERCADO", "MERCADOS", "MERCADINHO", "MINI", "MINIMERCADO", "MERCEARIA", "SUPERMERCADO",
    "SUPER", "ARMAZEM", "EMPORIO", "BAR", "CONVENIENCIA", "FARMACIA", "DROGARIA", "LOJA",
    "COMERCIO", "LTDA", "ME", "EPP", "NOSSA", "SENHORA", "SRA", "SR",
    "E", "DA", "DO", "DE", "DOS", "DAS", "S",
}
# Qual registro fica quando dois se fundem: o do OSM (nome comercial e tipo
# etiquetado) antes do CNEFE (descricao livre); depois o de mais atributos; depois
# o de maior formato de loja.
ORDEM_TIPO_NA_FUSAO = {"supermercado": 0, "conveniencia": 1, "mercado_pequeno": 2, "farmacia": 0, "agencia_postal": 0}

# -----------------------------------------------------------------------------
# Pareamento OSM <-> CNEFE
# -----------------------------------------------------------------------------
# Tres raios: o principal alimenta a tabela da Secao 3.2.2, os outros dois
# mostram que a taxa de correspondencia nao depende do raio escolhido. As
# coordenadas do CNEFE sao do endereco e as do OSM podem ser do centro do
# edificio, o que justifica um raio mais folgado que o de deduplicacao.
RAIOS_PAREAMENTO_M = (25.0, 50.0, 100.0)
RAIO_PAREAMENTO_PRINCIPAL = 50.0

# -----------------------------------------------------------------------------
# Consistencia setorial
# -----------------------------------------------------------------------------
# Desligada por padrao. Ver a ressalva metodologica em docs/decisoes_etapa4.md:
# ABRAS e CFF publicam por UF, nao por municipio, e sob definicoes que nao
# correspondem nem ao OSM nem ao CNEFE. Comparar contagem municipal com total
# estadual sob definicoes incompativeis produz divergencia inconclusiva.
#
# Excecao: as agencias dos Correios. O localizador de agencias da o numero exato
# para Florianopolis, e fonte primaria e a definicao e inequivoca.
#
# Ligar exige preencher os valores; o script bloqueia se algum estiver como None.
# Cada referencia e validada se estiver preenchida; as que seguem None sao
# reportadas como pendentes. O bloqueio vale para o inverso: nao se reporta
# validacao setorial de um tipo sem o numero da fonte.
VALIDAR_CONSISTENCIA_SETORIAL = True
N_SUPERMERCADOS_ABRAS = None    # ABRAS, Ranking — publicado por UF, nao por municipio
N_FARMACIAS_CFF = None          # CFF — publicado por UF
# Correios (2026), localizador de agencias, acesso em 09/09/2026: 16 agencias em
# Florianopolis, contando proprias e franqueadas, excluidas caixas de coleta e
# pontos sem atendimento de encomendas. Incluir franqueadas e deliberado: nem o
# amenity=post_office do OSM nem o registro do CNEFE distinguem os dois tipos, e
# ambas cumprem funcao equivalente de ponto de retirada para o consumidor.
N_AGENCIAS_CORREIOS = 16

# -----------------------------------------------------------------------------
# Dicionario de classificacao do CNEFE
# -----------------------------------------------------------------------------
# DSC_ESTABELECIMENTO e texto livre em latin-1. A classificacao normaliza acentos
# e pontuacao, aplica primeiro as exclusoes e depois classifica em ordem de
# prioridade — a primeira classe que casar vence.
#
# 'mercado_pequeno' e uma classe PROPRIA, separada de 'supermercado'. O CNEFE
# registra centenas de mercearias, minimercados, armazens e "mercados" de bairro
# que nao sao supermercados no sentido do shop=supermarket do OSM. Fundi-los
# distorceria as duas pontas da comparacao.
EXCLUSOES_CNEFE = {
    "deposito_ou_galpao": r"^DEPOSITO\b|\bGALPAO\b|\bMERCADORIAS?\b|\bESTOQUE\b",
    "imovel_vago": r"\bVAGO\b|\bVAGA\b|\bVAGAS\b|DESATIVAD|\bFECHADO\b",
    "mercado_publico": r"MERCADO PUBLICO|MERCADO MUNICIPAL",
    "outro_sentido_de_mercado": r"IMOBILIARI|MERCADO FINANCEIR|MERCADO DE TRABALHO",
}
DICIONARIO_CNEFE = {
    # ordem de prioridade
    "farmacia": r"\bFARMACIAS?\b|\bDROGARIAS?\b|\bDROGASIL\b|\bDROGA RAIA\b|\bPANVEL\b",
    "agencia_postal": r"\bCORREIOS?\b|\bECT\b|EMPRESA BRASILEIRA DE CORREIOS",
    "conveniencia": r"CONVENIENCIA",
    "supermercado": r"\bSUPERMERCAD\w*|\bHIPERMERCAD\w*|\bSUPER MERCAD\w*",
    "mercado_pequeno": r"\bMERCEARIA\w*|\bMINIMERCAD\w*|\bMINI MERCAD\w*|\bMERCADINHO\w*|"
                       r"\bMERCADOS?\b|\bARMAZEM\b|\bEMPORIO\b",
}

# Classes do CNEFE aceitas como correspondencia de cada tipo do OSM. A separacao
# supermercado/conveniencia do OSM nao tem equivalente exato no CNEFE, entao a
# compatibilidade e declarada explicitamente em vez de suposta.
COMPATIBILIDADE = {
    "supermercado": {"supermercado", "mercado_pequeno"},
    "conveniencia": {"conveniencia", "mercado_pequeno", "supermercado"},
    "farmacia": {"farmacia"},
    "agencia_postal": {"agencia_postal"},
}

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


def normalizar_texto(valor) -> str:
    """Maiusculas, sem acento, sem pontuacao, espacos colapsados."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    texto = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    texto = re.sub(r"[^A-Z0-9 ]", " ", texto.upper())
    return re.sub(r"\s+", " ", texto).strip()


# =============================================================================
# TERRITORIO
# =============================================================================


def carregar_territorio() -> tuple[gpd.GeoDataFrame, object, object, object]:
    """Bairros, poligono de terra do municipio, massas d'agua e terra dos vizinhos."""
    subtitulo("1. Territorio de referencia")

    checar(
        ARQ_BAIRROS.exists(),
        f"{ARQ_BAIRROS} nao existe. Rodar antes: python src/01_demanda.py",
    )
    bairros = gpd.read_file(ARQ_BAIRROS).to_crs(CRS_TRABALHO)
    terra = bairros.union_all()

    setores = gpd.read_file(ARQ_SETORES, where=f"CD_MUN = '{CD_MUN}'")
    setores = setores[["CD_SETOR", "CD_SIT", "geometry"]].dissolve(
        by="CD_SETOR", as_index=False, aggfunc="first"
    ).to_crs(CRS_TRABALHO)
    agua_gdf = setores[setores["CD_SIT"] == CD_SIT_MASSA_DAGUA]
    agua = agua_gdf.geometry.make_valid().union_all()

    # Terra dos municipios vizinhos: setores de SC que nao sao de Florianopolis nem
    # massa d'agua, num recorte em volta do municipio.
    caixa = gpd.GeoSeries([terra.envelope.buffer(FOLGA_BUSCA_VIZINHOS_M)], crs=CRS_TRABALHO)
    em_volta = gpd.read_file(ARQ_SETORES, bbox=tuple(caixa.to_crs(4674).total_bounds))
    em_volta = em_volta[(em_volta["CD_MUN"] != CD_MUN) & (em_volta["CD_SIT"] != CD_SIT_MASSA_DAGUA)]
    vizinhos = em_volta.to_crs(CRS_TRABALHO).geometry.make_valid().union_all()

    print(f"bairros ................................ {len(bairros)}")
    print(f"area de terra .......................... {numero_br(terra.area / 1e6, 1)} km2")
    print(f"setores de massa d'agua ................ {len(agua_gdf)} ({numero_br(agua.area / 1e6, 1)} km2)")
    print(f"municipios vizinhos no recorte ......... {em_volta['NM_MUN'].nunique()} ({', '.join(sorted(em_volta['NM_MUN'].unique()))})")
    print(f"CRS de trabalho ........................ EPSG:{CRS_TRABALHO}")
    return bairros, terra, agua, vizinhos


# =============================================================================
# EXTRACAO DO OSM
# =============================================================================


def extrair_osm(terra) -> gpd.GeoDataFrame:
    """Le o cache do OSM ou baixa a extracao e grava o cache.

    A consulta usa a CAIXA ENVOLVENTE do municipio, nao o poligono dissolvido dos
    87 bairros: a geometria da Ilha tem milhares de vertices e o Overpass recusa
    ou degrada consultas com poligonos muito complexos. O recorte fino ao
    territorio e feito depois, localmente, na verificacao espacial.
    """
    subtitulo("2. Extracao do OpenStreetMap")

    if ARQ_CACHE_OSM.exists() and not FORCAR_DOWNLOAD_OSM:
        bruto = gpd.read_parquet(ARQ_CACHE_OSM)
        metadados = json.loads(ARQ_METADADOS_OSM.read_text(encoding="utf-8"))
        print(f"lido do cache .......................... {ARQ_CACHE_OSM.name}")
        print(f"data da extracao ....................... {metadados['extraido_em']}")
        print(f"osmnx da extracao ...................... {metadados['osmnx']}")
        print(f"feicoes ................................ {len(bruto)}")
        print("   (FORCAR_DOWNLOAD_OSM = True para reextrair)")
        return bruto

    caixa_4326 = box(*gpd.GeoSeries([terra], crs=CRS_TRABALHO).to_crs(4326).total_bounds)
    print(f"consultando o Overpass na caixa envolvente do municipio...")
    bruto = ox.features_from_polygon(caixa_4326, TAGS_OSM)
    print(f"feicoes retornadas ..................... {len(bruto)}")

    bruto = bruto.reset_index()
    colunas = [c for c in ["element", "id", "shop", "amenity", *ATRIBUTOS_OSM] if c in bruto.columns]
    bruto = gpd.GeoDataFrame(bruto[colunas + ["geometry"]], geometry="geometry", crs=bruto.crs)

    DIR_TRATADOS.mkdir(parents=True, exist_ok=True)
    bruto.to_parquet(ARQ_CACHE_OSM, index=False)
    ARQ_METADADOS_OSM.write_text(
        json.dumps(
            {
                "extraido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "osmnx": ox.__version__,
                "tags": TAGS_OSM,
                "feicoes": int(len(bruto)),
                "caixa_4326": list(caixa_4326.bounds),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"cache gravado .......................... {ARQ_CACHE_OSM.name}")
    return bruto


def tratar_osm(bruto: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Tipa, reprojeta e converte poligonos em ponto representativo interno."""
    subtitulo("3. Tratamento das feicoes do OSM")

    dados = bruto.copy()
    for coluna in ["shop", "amenity", *ATRIBUTOS_OSM]:
        if coluna not in dados.columns:
            dados[coluna] = None

    # A comparacao sobre StringDtype devolve BooleanDtype com pd.NA nas linhas em
    # que a tag esta ausente, e Series.mask trata esse NA como "substituir". Sem o
    # fillna(False), a ultima tag do laco sobrescreveria todas as feicoes que nao
    # tem aquela chave. Por isso a condicao vira array booleano puro.
    dados["TIPO"] = pd.NA
    for (chave, valor), nome in TIPO_POR_TAG.items():
        marca = (dados[chave].astype("string") == valor).fillna(False).to_numpy(dtype=bool)
        dados.loc[marca, "TIPO"] = nome
        print(f"   {chave}={valor:<12} -> {nome:<16} {int(marca.sum()):>5}")

    sem_tipo = int(dados["TIPO"].isna().sum())
    if sem_tipo:
        print(f"feicoes sem tipo reconhecido (descartadas) {sem_tipo}")
    dados = dados[dados["TIPO"].notna()].copy()

    geometrias = dados.geometry.geom_type.value_counts().to_dict()
    print(f"tipos de geometria na extracao ......... {geometrias}")

    dados = dados.to_crs(CRS_TRABALHO)
    dados["geometry"] = dados.geometry.make_valid()
    # Estabelecimentos mapeados como edificio viram o seu ponto representativo
    # interno; nunca o centroide, que pode cair fora de um poligono em L ou U.
    nao_pontual = dados.geometry.geom_type != "Point"
    if nao_pontual.any():
        print(f"feicoes em area convertidas a ponto ..... {int(nao_pontual.sum())}")
        dados.loc[nao_pontual, "geometry"] = dados.loc[nao_pontual].geometry.representative_point()
    checar(
        bool((dados.geometry.geom_type == "Point").all()),
        "Restou feicao nao pontual apos a conversao.",
    )
    checar(
        dados.crs.to_epsg() == CRS_TRABALHO,
        f"Candidatos fora do EPSG:{CRS_TRABALHO}.",
    )

    dados["NOME"] = dados["name"].map(normalizar_texto)
    dados["MARCA"] = dados["brand"].map(normalizar_texto)
    dados["OPERADOR"] = dados["operator"].map(normalizar_texto)
    dados["HORARIO"] = dados["opening_hours"].astype("string")

    print(f"feicoes tipadas ........................ {len(dados)}")
    print("por tipo:")
    for nome, n in dados["TIPO"].value_counts().items():
        print(f"   {nome:<18} {n:>5}")
    print(f"com nome ............................... {int((dados['NOME'] != '').sum())} "
          f"({100 * (dados['NOME'] != '').mean():.1f}%)")
    print(f"com opening_hours ...................... {int(dados['HORARIO'].notna().sum())} "
          f"({100 * dados['HORARIO'].notna().mean():.1f}%)  [alimenta a Etapa 5]")

    if REMOVER_POR_NOME:
        chave = list(zip(dados["TIPO"], dados["NOME"]))
        marca = np.array([c in REMOVER_POR_NOME for c in chave])
        if marca.any():
            print(f"\nremocoes nominais ...................... {int(marca.sum())}")
            for tipo, nome in sorted({c for c, m in zip(chave, marca) if m}):
                print(f"   {nome:<24} {tipo:<16} {REMOVER_POR_NOME[(tipo, nome)]}")
            dados = dados[~marca].copy()
            print(f"feicoes restantes ...................... {len(dados)}")
    return dados


# =============================================================================
# VERIFICACAO ESPACIAL
# =============================================================================


def verificar_localizacao(
    dados: gpd.GeoDataFrame, terra, agua, vizinhos
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Descarta candidato sobre massa d'agua ou fora do territorio do municipio.

    Verificacao exigida pela licao da Etapa 2: consistencia interna nao detecta
    coordenada plausivel em local errado.
    """
    subtitulo("4. Verificacao espacial")

    dados = dados.copy()
    dados["EM_TERRA"] = dados.geometry.within(terra)
    dados["EM_AGUA"] = dados.geometry.within(agua)
    dados["DIST_TERRA_M"] = dados.geometry.distance(terra)
    dados["EM_VIZINHO"] = dados.geometry.within(vizinhos)

    fora = ~dados["EM_TERRA"]
    sobre_agua = dados["EM_AGUA"]
    # Dentro de setor de outro municipio: descartado a qualquer distancia da divisa.
    em_vizinho = fora & (~sobre_agua) & dados["EM_VIZINHO"]
    fora_tolerado = fora & (~sobre_agua) & (~em_vizinho) & (dados["DIST_TERRA_M"] <= TOLERANCIA_BORDA_M)
    suspeito = (
        fora & (~sobre_agua) & (~em_vizinho)
        & (dados["DIST_TERRA_M"] > TOLERANCIA_BORDA_M)
        & (dados["DIST_TERRA_M"] <= LIMITE_OUTRO_MUNICIPIO_M)
    )
    outro_municipio = fora & (~sobre_agua) & (em_vizinho | (dados["DIST_TERRA_M"] > LIMITE_OUTRO_MUNICIPIO_M))
    vizinho_perto = em_vizinho & (dados["DIST_TERRA_M"] <= LIMITE_OUTRO_MUNICIPIO_M)

    print(f"candidatos avaliados ................... {len(dados)}")
    print(f"dentro do territorio ................... {int(dados['EM_TERRA'].sum())}")
    print(f"sobre massa d'agua ..................... {int(sobre_agua.sum())}   [descartados]")
    print(f"fora, ate {TOLERANCIA_BORDA_M:.0f} m da costa ............... {int(fora_tolerado.sum())}   [mantidos]")
    print(f"fora, {TOLERANCIA_BORDA_M:.0f}-{LIMITE_OUTRO_MUNICIPIO_M:.0f} m ......................... {int(suspeito.sum())}   [descartados, suspeitos]")
    print(f"fora, alem de {LIMITE_OUTRO_MUNICIPIO_M:.0f} m ou em setor vizinho .. {int(outro_municipio.sum())}   [outro municipio]")
    if vizinho_perto.any():
        print("   em setor de outro municipio a menos de "
              f"{LIMITE_OUTRO_MUNICIPIO_M:.0f} m da divisa, nominalmente:")
        for linha in dados[vizinho_perto].sort_values("DIST_TERRA_M").itertuples():
            print(f"      {(linha.NOME or '(sem nome)')[:36]:<36} {linha.TIPO:<16} {linha.DIST_TERRA_M:>6.0f} m alem da divisa")

    descartados = dados[sobre_agua | suspeito | outro_municipio].copy()
    descartados["MOTIVO"] = np.select(
        [
            descartados["EM_AGUA"].to_numpy(dtype=bool),
            suspeito.reindex(descartados.index).to_numpy(dtype=bool),
        ],
        ["sobre massa d'agua", "fora da terra, distancia suspeita"],
        default="outro municipio na caixa envolvente",
    )

    nominais = descartados[descartados["MOTIVO"] != "outro municipio na caixa envolvente"]
    if len(nominais) > 0:
        print("\ncandidatos descartados por anomalia, nominalmente:")
        for linha in nominais.sort_values("DIST_TERRA_M", ascending=False).itertuples():
            nome = linha.NOME or "(sem nome)"
            print(
                f"   {nome[:38]:<38} {linha.TIPO:<16} {linha.MOTIVO:<36} "
                f"{linha.DIST_TERRA_M:>8.0f} m da terra"
            )
    else:
        print("\nnenhum candidato descartado por anomalia de coordenada.")

    if int(outro_municipio.sum()) > 0:
        print(f"\ndescartados por estarem em outro municipio: {int(outro_municipio.sum())}")
        print("   (filtragem esperada da caixa envolvente; lista completa no CSV de descartados)")
    if len(fora_tolerado[fora_tolerado]) > 0:
        print(f"\ncandidatos mantidos por tolerancia de borda (imprecisao de orla):")
        for linha in dados[fora_tolerado].sort_values("DIST_TERRA_M", ascending=False).itertuples():
            nome = linha.NOME or "(sem nome)"
            print(f"   {nome[:38]:<38} {linha.TIPO:<16} {linha.DIST_TERRA_M:>8.1f} m da terra")

    mantidos = dados[~(sobre_agua | suspeito | outro_municipio)].copy()
    checar(
        int((mantidos.geometry.within(agua)).sum()) == 0,
        "Restou candidato sobre massa d'agua apos o descarte.",
    )
    print(f"\ncandidatos mantidos .................... {len(mantidos)}")
    return mantidos.drop(columns=["EM_TERRA", "EM_AGUA", "EM_VIZINHO"]), descartados


# =============================================================================
# DEDUPLICACAO
# =============================================================================


def deduplicar(dados: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Colapsa registros do mesmo estabelecimento mapeados mais de uma vez.

    Criterio: mesmo tipo, nome normalizado equivalente e distancia menor que
    RAIO_DEDUP_M. Dois pontos a distancia d tem buffers de raio d/2 que se
    intersectam se e somente se d <= RAIO_DEDUP_M, o que reduz o problema a uma
    juncao espacial e a componentes conexas.
    """
    subtitulo("5. Deduplicacao")

    dados = dados.reset_index(drop=True).copy()
    dados["ID_TMP"] = dados.index

    buffers = gpd.GeoDataFrame(
        dados[["ID_TMP", "TIPO", "NOME"]],
        geometry=dados.geometry.buffer(RAIO_DEDUP_M / 2.0),
        crs=dados.crs,
    )
    pares = gpd.sjoin(buffers, buffers, how="inner", predicate="intersects")
    pares = pares[
        (pares["ID_TMP_left"] < pares["ID_TMP_right"])
        & (pares["TIPO_left"] == pares["TIPO_right"])
        & (pares["NOME_left"] == pares["NOME_right"])
    ]

    grafo = nx.Graph()
    grafo.add_nodes_from(dados["ID_TMP"])
    grafo.add_edges_from(zip(pares["ID_TMP_left"], pares["ID_TMP_right"]))
    grupos = {no: i for i, comp in enumerate(nx.connected_components(grafo)) for no in comp}
    dados["GRUPO"] = dados["ID_TMP"].map(grupos)

    # Dentro de cada grupo fica o registro com mais atributos preenchidos; empate
    # resolvido pelo menor id do OSM, para que o resultado seja deterministico.
    dados["N_ATRIBUTOS"] = (
        (dados["NOME"] != "").astype(int)
        + (dados["MARCA"] != "").astype(int)
        + (dados["OPERADOR"] != "").astype(int)
        + dados["HORARIO"].notna().astype(int)
    )
    ordenado = dados.sort_values(["GRUPO", "N_ATRIBUTOS", "id"], ascending=[True, False, True])
    unicos = ordenado.drop_duplicates("GRUPO").copy()
    unicos["N_FUNDIDOS"] = unicos["GRUPO"].map(dados["GRUPO"].value_counts())

    removidos = len(dados) - len(unicos)
    sem_nome = int(((unicos["N_FUNDIDOS"] > 1) & (unicos["NOME"] == "")).sum())
    print(f"raio de deduplicacao ................... {RAIO_DEDUP_M:.0f} m + nome normalizado")
    print(f"registros antes ........................ {len(dados)}")
    print(f"registros fundidos ..................... {removidos}")
    print(f"candidatos unicos ...................... {len(unicos)}")
    print(f"grupos fundidos sem nome ............... {sem_nome}  (auditar se elevado)")

    maiores = unicos[unicos["N_FUNDIDOS"] > 1].sort_values("N_FUNDIDOS", ascending=False)
    if len(maiores) > 0:
        print("\nmaiores fusoes:")
        for linha in maiores.head(8).itertuples():
            print(f"   {(linha.NOME or '(sem nome)')[:40]:<40} {linha.TIPO:<16} {linha.N_FUNDIDOS} registros")

    return unicos.drop(columns=["ID_TMP", "GRUPO", "N_ATRIBUTOS"]).reset_index(drop=True)


def nome_essencial(nome) -> str:
    """Nome sem acento, pontuacao, palavras genericas e zeros a esquerda.

    "BAR E MINI MERCADO NOSSA SRA APARECIDA" e "... NOSSA SENHORA APARECIDA" viram
    ambos "APARECIDA"; "SUPERMERCADO 5 MENINAS" e "MERCADO 05 MENINAS", "5MENINAS".
    """
    palavras = []
    for palavra in normalizar_texto(nome).split():
        if palavra.isdigit():
            palavra = palavra.lstrip("0") or "0"
        if palavra not in PALAVRAS_GENERICAS:
            palavras.append(palavra)
    return "".join(palavras)


def deduplicar_nome_aproximado(dados: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Funde o mesmo estabelecimento grafado de formas diferentes ou vindo das duas fontes.

    Roda sobre o conjunto JA UNIDO (OSM + CNEFE), depois das deduplicacoes exatas.
    Criterios e limites no bloco de parametros. Devolve os candidatos e a lista das
    fusoes, para auditoria nominal.
    """
    subtitulo("9b. Deduplicacao por nome aproximado")

    dados = dados.reset_index(drop=True).copy()
    essencial = dados["NOME"].map(nome_essencial)
    xy = np.column_stack([dados.geometry.x.to_numpy(), dados.geometry.y.to_numpy()])

    arestas, linhas = [], []
    vizinhos_por_ponto = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=dados.crs).sindex
    for i in range(len(dados)):
        candidatos_i = vizinhos_por_ponto.query(dados.geometry.iloc[i].buffer(RAIO_DEDUP_APROX_M))
        for j in candidatos_i:
            if j <= i:
                continue
            if GRUPO_DE_TIPO[dados.at[i, "TIPO"]] != GRUPO_DE_TIPO[dados.at[j, "TIPO"]]:
                continue
            a, b = essencial.iat[i], essencial.iat[j]
            if min(len(a), len(b)) < TAMANHO_MINIMO_NOME:
                continue
            distancia = float(np.hypot(*(xy[i] - xy[j])))
            similaridade = SequenceMatcher(None, a, b).ratio()
            if distancia <= RAIO_DEDUP_APROX_M and similaridade >= SIMILARIDADE_MINIMA:
                arestas.append((i, j))
                linhas.append(
                    {
                        "DIST_M": round(distancia, 1), "SIMILARIDADE": round(similaridade, 3),
                        "FONTE_1": dados.at[i, "FONTE"], "TIPO_1": dados.at[i, "TIPO"], "NOME_1": dados.at[i, "NOME"],
                        "FONTE_2": dados.at[j, "FONTE"], "TIPO_2": dados.at[j, "TIPO"], "NOME_2": dados.at[j, "NOME"],
                    }
                )

    grafo = nx.Graph()
    grafo.add_nodes_from(range(len(dados)))
    grafo.add_edges_from(arestas)
    dados["GRUPO"] = pd.Series({no: k for k, comp in enumerate(nx.connected_components(grafo)) for no in comp})

    dados["PRIORIDADE_FONTE"] = (dados["FONTE"] != "OSM").astype(int)
    dados["N_ATRIBUTOS"] = (
        (dados["NOME"].fillna("") != "").astype(int)
        + (dados["MARCA"].fillna("") != "").astype(int)
        + (dados["OPERADOR"].fillna("") != "").astype(int)
        + dados["HORARIO"].notna().astype(int)
    )
    dados["ORDEM_TIPO"] = dados["TIPO"].map(ORDEM_TIPO_NA_FUSAO)
    # O id do OSM e numerico e o do CNEFE e texto ("CNEFE-12"): ordenar pela forma
    # textual evita comparar tipos diferentes e mantem o desempate deterministico.
    dados["ID_TEXTO"] = dados["id"].astype(str)
    dados["N_FUNDIDOS"] = dados["N_FUNDIDOS"].fillna(1).astype(int)
    total_fundidos = dados.groupby("GRUPO")["N_FUNDIDOS"].sum()
    ordenado = dados.sort_values(
        ["GRUPO", "PRIORIDADE_FONTE", "N_ATRIBUTOS", "ORDEM_TIPO", "ID_TEXTO"],
        ascending=[True, True, False, True, True],
    )
    unicos = ordenado.drop_duplicates("GRUPO").copy()
    unicos["N_FUNDIDOS"] = unicos["GRUPO"].map(total_fundidos)

    fusoes = pd.DataFrame(linhas).sort_values("DIST_M") if linhas else pd.DataFrame()
    removidos = len(dados) - len(unicos)
    print(f"criterio ............................... ate {RAIO_DEDUP_APROX_M:.0f} m, mesmo grupo de tipo, "
          f"similaridade >= {SIMILARIDADE_MINIMA:.2f}")
    print(f"pares fundidos ......................... {len(fusoes)}")
    print(f"registros removidos .................... {removidos}")
    for linha in fusoes.itertuples():
        print(f"   {linha.DIST_M:>5.0f} m  {linha.SIMILARIDADE:.2f}  {linha.FONTE_1:<5} {str(linha.NOME_1)[:30]:<30} "
              f"x {linha.FONTE_2:<5} {str(linha.NOME_2)[:30]}")
    checar(
        len(unicos) + removidos == len(dados),
        "A fusao por nome aproximado perdeu ou duplicou registros.",
    )
    return (
        unicos.drop(columns=["GRUPO", "PRIORIDADE_FONTE", "N_ATRIBUTOS", "ORDEM_TIPO", "ID_TEXTO"]).reset_index(drop=True),
        fusoes,
    )


def auditar_duplicatas_residuais(dados: gpd.GeoDataFrame) -> None:
    """Lista pares de mesmo tipo e nome que sobraram acima do raio de deduplicacao.

    Serve para conferir que o raio de 25 m nao deixou duplicata obvia para tras.
    Filiais distintas da mesma rede devem aparecer aqui e PERMANECER separadas.
    """
    subtitulo("6. Auditoria de duplicatas residuais")

    # sjoin_nearest devolve so O vizinho mais proximo de cada feicao, e o vizinho
    # mais proximo de uma filial costuma ser um concorrente, nao a outra filial da
    # mesma rede. Por isso a auditoria percorre os grupos de mesmo tipo e mesmo
    # nome e calcula a menor distancia DENTRO do grupo.
    nomeados = dados[dados["NOME"] != ""].copy()
    grupos = nomeados.groupby(["TIPO", "NOME"])

    linhas = []
    for (tipo, nome), grupo in grupos:
        if len(grupo) < 2:
            continue
        xy = np.column_stack([grupo.geometry.x.to_numpy(), grupo.geometry.y.to_numpy()])
        distancias = np.hypot(
            xy[:, 0][:, None] - xy[:, 0][None, :], xy[:, 1][:, None] - xy[:, 1][None, :]
        )
        np.fill_diagonal(distancias, np.inf)
        linhas.append(
            {
                "TIPO": tipo,
                "NOME": nome,
                "UNIDADES": len(grupo),
                "MENOR_DIST_M": float(distancias.min()),
            }
        )

    repetidos = pd.DataFrame(linhas)
    print(f"nomes com mais de uma unidade .......... {len(repetidos)}")
    if len(repetidos) == 0:
        print("   nenhuma rede com mais de uma unidade no conjunto — auditoria vazia.")
        return

    print(f"unidades envolvidas .................... {int(repetidos['UNIDADES'].sum())}")
    abaixo = repetidos[repetidos["MENOR_DIST_M"] <= RAIO_DEDUP_M]
    checar(
        len(abaixo) == 0,
        "Sobrou par de mesmo tipo e nome a menos do raio de deduplicacao: "
        f"{abaixo[['NOME', 'MENOR_DIST_M']].to_dict('records')}",
    )
    print(f"nenhum par abaixo de {RAIO_DEDUP_M:.0f} m ................ conferido")

    print(f"\n   {'nome':<34} {'tipo':<16} {'unid.':>6} {'menor dist.':>12}")
    for linha in repetidos.sort_values("MENOR_DIST_M").head(12).itertuples():
        print(
            f"   {linha.NOME[:34]:<34} {linha.TIPO:<16} {linha.UNIDADES:>6} "
            f"{linha.MENOR_DIST_M:>10.0f} m"
        )
    print(f"\n   Distancias de centenas de metros sao filiais distintas da mesma rede")
    print(f"   e devem permanecer separadas: sao pontos de retirada diferentes.")


# =============================================================================
# BAIRRO E REGIAO
# =============================================================================


def associar_bairro(dados: gpd.GeoDataFrame, bairros: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Junta cada candidato ao seu bairro; usa o mais proximo na tolerancia de borda."""
    subtitulo("10. Associacao ao bairro")

    colunas = ["CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL", "geometry"]
    dentro = gpd.sjoin(dados, bairros[colunas], how="left", predicate="within").drop(
        columns="index_right"
    )
    dentro = dentro[~dentro.index.duplicated(keep="first")]

    sem_bairro = dentro["CD_BAIRRO"].isna()
    if sem_bairro.any():
        # Sao os candidatos mantidos pela tolerancia de borda: caem fora de todo
        # poligono, mas a poucos metros da costa.
        resgate = gpd.sjoin_nearest(
            dados.loc[sem_bairro.values, ["geometry"]],
            bairros[colunas],
            how="left",
            distance_col="DIST_BAIRRO_M",
        ).drop(columns="index_right")
        resgate = resgate[~resgate.index.duplicated(keep="first")]
        for coluna in ["CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL"]:
            dentro.loc[sem_bairro.values, coluna] = resgate[coluna].to_numpy()
        print(f"candidatos associados por proximidade .. {int(sem_bairro.sum())} (tolerancia de borda)")

    checar(
        dentro["CD_BAIRRO"].notna().all(),
        "Candidato sem bairro apos a juncao espacial.",
    )
    print(f"candidatos associados .................. {len(dentro)}")
    print(f"bairros com ao menos um candidato ...... {dentro['CD_BAIRRO'].nunique()} de {len(bairros)}")

    vazios = bairros[~bairros["CD_BAIRRO"].isin(dentro["CD_BAIRRO"])]
    if len(vazios) > 0:
        print(f"bairros sem nenhum candidato ........... {len(vazios)}")
        print("   " + ", ".join(sorted(vazios["NM_BAIRRO"])[:20]))
    return dentro


# =============================================================================
# VALIDACAO CRUZADA COM O CNEFE
# =============================================================================


def classificar_cnefe() -> gpd.GeoDataFrame:
    """Classifica os estabelecimentos do CNEFE (especie 6) por tipo."""
    subtitulo("7. CNEFE — classificacao dos estabelecimentos")

    cnefe = pd.read_csv(
        ARQ_CNEFE, sep=";", encoding="latin-1", dtype=str,
        usecols=["COD_ESPECIE", "DSC_ESTABELECIMENTO", "LATITUDE", "LONGITUDE"],
    )
    e6 = cnefe[cnefe["COD_ESPECIE"] == "6"].copy()
    print(f"estabelecimentos de especie 6 .......... {numero_br(len(e6))}")

    e6["DESCRICAO"] = e6["DSC_ESTABELECIMENTO"].map(normalizar_texto)

    excluido = pd.Series(False, index=e6.index)
    print("\nexclusoes aplicadas antes da classificacao:")
    for motivo, padrao in EXCLUSOES_CNEFE.items():
        marca = e6["DESCRICAO"].str.contains(padrao, regex=True, na=False) & ~excluido
        excluido |= marca
        print(f"   {motivo:<26} {int(marca.sum()):>5}")
    print(f"   {'total excluido':<26} {int(excluido.sum()):>5}")
    e6["EXCLUIDO"] = excluido

    classe = pd.Series(pd.NA, index=e6.index, dtype="object")
    for nome, padrao in DICIONARIO_CNEFE.items():
        marca = (
            e6["DESCRICAO"].str.contains(padrao, regex=True, na=False)
            & classe.isna()
            & ~excluido
        )
        classe = classe.mask(marca, nome)
    e6["CLASSE"] = classe

    print("\nclassificacao:")
    contagem = e6["CLASSE"].value_counts()
    for nome in DICIONARIO_CNEFE:
        print(f"   {nome:<18} {int(contagem.get(nome, 0)):>5}")
    print(f"   {'nao classificado':<18} {int(e6['CLASSE'].isna().sum() - excluido.sum()):>5}")

    classificados = e6[e6["CLASSE"].notna()].copy()
    pontos = gpd.GeoDataFrame(
        classificados[["CLASSE", "DESCRICAO"]],
        geometry=gpd.points_from_xy(
            classificados["LONGITUDE"].astype(float), classificados["LATITUDE"].astype(float)
        ),
        crs=4326,
    ).to_crs(CRS_TRABALHO)
    return pontos


def verificar_plausibilidade(cnefe: gpd.GeoDataFrame) -> None:
    """Farmacias tem de superar supermercados no CNEFE classificado.

    Florianopolis tem muito mais farmacia que supermercado. Se a contagem de
    supermercados superar a de farmacias, o dicionario esta capturando mercearias
    e mercados de bairro como supermercado.
    """
    subtitulo("8. Verificacao de plausibilidade da classificacao")

    contagem = cnefe["CLASSE"].value_counts()
    n_farmacia = int(contagem.get("farmacia", 0))
    n_super = int(contagem.get("supermercado", 0))
    n_pequeno = int(contagem.get("mercado_pequeno", 0))

    print(f"farmacias .............................. {n_farmacia}")
    print(f"supermercados (sentido estrito) ........ {n_super}")
    print(f"mercados pequenos (classe separada) .... {n_pequeno}")
    checar(
        n_farmacia > n_super,
        f"Implausivel: {n_super} supermercados contra {n_farmacia} farmacias. "
        "Florianopolis tem muito mais farmacia que supermercado — o dicionario "
        "ainda esta classificando mercearias e mercados de bairro como supermercado.",
    )
    print("farmacias > supermercados .............. conferido")
    print(
        f"\n   Os {n_pequeno} mercados pequenos ficam em classe propria. Se fossem somados\n"
        f"   a supermercado, o total ({n_super + n_pequeno}) superaria o de farmacias e a\n"
        f"   verificacao falharia — o que e exatamente o efeito que ela existe para pegar."
    )


def preparar_mercados_pequenos(
    cnefe: gpd.GeoDataFrame, candidatos_osm: gpd.GeoDataFrame, terra, agua, vizinhos
) -> gpd.GeoDataFrame:
    """Converte o varejo alimentar de pequeno porte do CNEFE em candidatos.

    Passa pelas mesmas verificacoes espaciais dos candidatos do OSM, e depois por
    uma deduplicacao ENTRE FONTES: um registro do CNEFE a menos de RAIO_DEDUP_M
    de um candidato do OSM e, com altissima probabilidade, o mesmo
    estabelecimento visto pelas duas bases. Aqui a deduplicacao NAO pode exigir
    nome equivalente — o CNEFE guarda uma descricao livre e o OSM, um nome
    comercial, e os dois quase nunca coincidem textualmente.
    """
    subtitulo("9. Varejo de pequeno porte do CNEFE como candidato")

    alvo = cnefe[cnefe["CLASSE"] == TIPO_MERCADO_PEQUENO].copy()
    print(f"classificados como {TIPO_MERCADO_PEQUENO} ..... {len(alvo)}")

    novos = gpd.GeoDataFrame(
        {
            "id": [f"CNEFE-{i}" for i in range(len(alvo))],
            "TIPO": TIPO_MERCADO_PEQUENO,
            "NOME": alvo["DESCRICAO"].to_numpy(),
            "MARCA": "",
            "OPERADOR": "",
            "HORARIO": pd.Series([pd.NA] * len(alvo), dtype="string").to_numpy(),
        },
        geometry=alvo.geometry.to_numpy(),
        crs=alvo.crs,
    )

    # Mesmas verificacoes espaciais aplicadas ao OSM.
    novos["EM_TERRA"] = novos.geometry.within(terra)
    novos["EM_AGUA"] = novos.geometry.within(agua)
    novos["DIST_TERRA_M"] = novos.geometry.distance(terra)
    fora_grave = (~novos["EM_TERRA"]) & (
        novos["EM_AGUA"] | (novos["DIST_TERRA_M"] > TOLERANCIA_BORDA_M)
        | novos.geometry.within(vizinhos)
    )
    if int(fora_grave.sum()) > 0:
        print(f"descartados por localizacao ............ {int(fora_grave.sum())}")
        for linha in novos[fora_grave].sort_values("DIST_TERRA_M", ascending=False).head(10).itertuples():
            motivo = "sobre massa d'agua" if linha.EM_AGUA else "fora do territorio"
            print(f"   {linha.NOME[:38]:<38} {motivo:<22} {linha.DIST_TERRA_M:>8.0f} m")
    else:
        print("descartados por localizacao ............ 0")
    novos = novos[~fora_grave].drop(columns=["EM_TERRA", "EM_AGUA", "DIST_TERRA_M"])

    # Deduplicacao interna ao CNEFE: mesma descricao a menos de RAIO_DEDUP_M.
    antes = len(novos)
    novos = deduplicar_silencioso(novos)
    print(f"fundidos dentro do proprio CNEFE ....... {antes - len(novos)}")

    # Deduplicacao entre fontes, por proximidade apenas.
    if len(candidatos_osm) > 0:
        vizinho = gpd.sjoin_nearest(
            novos[["geometry"]], candidatos_osm[["geometry"]], how="left", distance_col="DIST_M"
        )
        vizinho = vizinho[~vizinho.index.duplicated(keep="first")]
        ja_no_osm = (vizinho["DIST_M"] <= RAIO_DEDUP_M).to_numpy(dtype=bool)
        print(f"ja presentes no OSM (a ate {RAIO_DEDUP_M:.0f} m) ..... {int(ja_no_osm.sum())}")
        novos = novos[~ja_no_osm]

    novos["N_FUNDIDOS"] = novos.get("N_FUNDIDOS", 1)
    print(f"candidatos acrescentados ............... {len(novos)}")
    return novos.reset_index(drop=True)


def deduplicar_silencioso(dados: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Mesmo criterio de deduplicar(), sem o relatorio — uso interno."""
    dados = dados.reset_index(drop=True).copy()
    dados["ID_TMP"] = dados.index
    buffers = gpd.GeoDataFrame(
        dados[["ID_TMP", "TIPO", "NOME"]],
        geometry=dados.geometry.buffer(RAIO_DEDUP_M / 2.0),
        crs=dados.crs,
    )
    pares = gpd.sjoin(buffers, buffers, how="inner", predicate="intersects")
    pares = pares[
        (pares["ID_TMP_left"] < pares["ID_TMP_right"])
        & (pares["TIPO_left"] == pares["TIPO_right"])
        & (pares["NOME_left"] == pares["NOME_right"])
    ]
    grafo = nx.Graph()
    grafo.add_nodes_from(dados["ID_TMP"])
    grafo.add_edges_from(zip(pares["ID_TMP_left"], pares["ID_TMP_right"]))
    grupos = {no: i for i, comp in enumerate(nx.connected_components(grafo)) for no in comp}
    dados["GRUPO"] = dados["ID_TMP"].map(grupos)
    contagem = dados["GRUPO"].value_counts()
    unicos = dados.drop_duplicates("GRUPO").copy()
    unicos["N_FUNDIDOS"] = unicos["GRUPO"].map(contagem)
    return unicos.drop(columns=["ID_TMP", "GRUPO"]).reset_index(drop=True)


def parear_osm_cnefe(
    candidatos: gpd.GeoDataFrame, cnefe: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Taxa de correspondencia OSM -> CNEFE nos tres raios."""
    subtitulo("11. Validacao cruzada OSM x CNEFE")

    linhas = []
    for tipo in TIPOS_OSM:
        alvo = candidatos[candidatos["TIPO"] == tipo]
        classes = COMPATIBILIDADE[tipo]
        referencia = cnefe[cnefe["CLASSE"].isin(classes)]
        if len(alvo) == 0 or len(referencia) == 0:
            for raio in RAIOS_PAREAMENTO_M:
                linhas.append(
                    {
                        "tipo_osm": tipo,
                        "classes_cnefe": " | ".join(sorted(classes)),
                        "candidatos_osm": len(alvo),
                        "estabelecimentos_cnefe": len(referencia),
                        "raio_m": raio,
                        "pareados": 0,
                        "taxa_pct": np.nan,
                    }
                )
            continue

        vizinho = gpd.sjoin_nearest(
            alvo[["geometry"]], referencia[["geometry"]], how="left", distance_col="DIST_M"
        )
        vizinho = vizinho[~vizinho.index.duplicated(keep="first")]
        for raio in RAIOS_PAREAMENTO_M:
            pareados = int((vizinho["DIST_M"] <= raio).sum())
            linhas.append(
                {
                    "tipo_osm": tipo,
                    "classes_cnefe": " | ".join(sorted(classes)),
                    "candidatos_osm": len(alvo),
                    "estabelecimentos_cnefe": len(referencia),
                    "raio_m": raio,
                    "pareados": pareados,
                    "taxa_pct": round(100 * pareados / len(alvo), 1),
                }
            )

    tabela = pd.DataFrame(linhas)
    print(f"   {'tipo (OSM)':<18} {'OSM':>6} {'CNEFE':>7}" + "".join(
        f"{'r=' + f'{r:.0f}m':>12}" for r in RAIOS_PAREAMENTO_M
    ))
    for tipo in TIPOS_OSM:
        sub = tabela[tabela["tipo_osm"] == tipo]
        if sub.empty:
            continue
        texto = (
            f"   {tipo:<18} {int(sub['candidatos_osm'].iloc[0]):>6} "
            f"{int(sub['estabelecimentos_cnefe'].iloc[0]):>7}"
        )
        for raio in RAIOS_PAREAMENTO_M:
            linha = sub[sub["raio_m"] == raio].iloc[0]
            texto += f"{linha['taxa_pct']:>11.1f}%"
        print(texto)

    print(f"\n   raio principal: {RAIO_PAREAMENTO_PRINCIPAL:.0f} m")
    print(f"   amplitude da taxa entre {RAIOS_PAREAMENTO_M[0]:.0f} m e {RAIOS_PAREAMENTO_M[-1]:.0f} m, por tipo:")
    for tipo in TIPOS_OSM:
        sub = tabela[tabela["tipo_osm"] == tipo]
        if sub.empty or sub["taxa_pct"].isna().all():
            continue
        menor = float(sub["taxa_pct"].min())
        maior = float(sub["taxa_pct"].max())
        amplitude = maior - menor
        leitura = "estavel" if amplitude <= 15 else "SENSIVEL ao raio"
        print(f"      {tipo:<18} {menor:>5.1f}% a {maior:>5.1f}%   amplitude {amplitude:>5.1f} p.p.   {leitura}")
    print(
        "\n   Onde a amplitude e pequena, a correspondencia nao e artefato do raio.\n"
        "   Onde e grande, a taxa depende da tolerancia adotada e precisa ser reportada\n"
        "   com o raio explicito, nunca como numero unico."
    )
    return tabela


def parear_por_regiao(candidatos: gpd.GeoDataFrame, cnefe: gpd.GeoDataFrame) -> pd.DataFrame:
    """Taxa de correspondencia OSM -> CNEFE desagregada por regiao funcional.

    Verifica se a concordancia entre as bases e homogenea no territorio. Uma taxa
    baixa numa regiao indica cobertura desigual do OSM ali, o que e exatamente o
    vies que a Etapa 4 precisa declarar.
    """
    subtitulo("12. Correspondencia OSM x CNEFE por regiao funcional")

    ordem = ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]
    linhas = []
    for regiao in ordem:
        alvo_regiao = candidatos[candidatos["REGIAO_FUNCIONAL"] == regiao]
        for tipo in TIPOS_OSM:
            alvo = alvo_regiao[alvo_regiao["TIPO"] == tipo]
            referencia = cnefe[cnefe["CLASSE"].isin(COMPATIBILIDADE[tipo])]
            if len(alvo) == 0 or len(referencia) == 0:
                continue
            vizinho = gpd.sjoin_nearest(
                alvo[["geometry"]], referencia[["geometry"]], how="left", distance_col="DIST_M"
            )
            vizinho = vizinho[~vizinho.index.duplicated(keep="first")]
            pareados = int((vizinho["DIST_M"] <= RAIO_PAREAMENTO_PRINCIPAL).sum())
            linhas.append(
                {
                    "regiao_funcional": regiao,
                    "tipo_osm": tipo,
                    "candidatos_osm": len(alvo),
                    "pareados": pareados,
                    "taxa_pct": round(100 * pareados / len(alvo), 1),
                }
            )

    tabela = pd.DataFrame(linhas)
    print(f"   raio de {RAIO_PAREAMENTO_PRINCIPAL:.0f} m")
    print(f"   {'regiao':<14}" + "".join(f"{t[:13]:>15}" for t in TIPOS_OSM) + f"{'geral':>10}")
    for regiao in ordem:
        sub = tabela[tabela["regiao_funcional"] == regiao]
        if sub.empty:
            continue
        texto = f"   {regiao:<14}"
        for tipo in TIPOS_OSM:
            linha = sub[sub["tipo_osm"] == tipo]
            texto += f"{linha['taxa_pct'].iloc[0]:>14.1f}%" if not linha.empty else f"{'—':>15}"
        geral = 100 * sub["pareados"].sum() / sub["candidatos_osm"].sum()
        texto += f"{geral:>9.1f}%"
        print(texto)

    geral_por_regiao = tabela.groupby("regiao_funcional").apply(
        lambda g: 100 * g["pareados"].sum() / g["candidatos_osm"].sum(), include_groups=False
    )
    amplitude = float(geral_por_regiao.max() - geral_por_regiao.min())
    print(
        f"\n   amplitude entre regioes: {amplitude:.1f} p.p. "
        f"({geral_por_regiao.idxmin()} {geral_por_regiao.min():.1f}% a "
        f"{geral_por_regiao.idxmax()} {geral_por_regiao.max():.1f}%)"
    )
    return tabela


def comparar_com_e_sem(candidatos: gpd.GeoDataFrame, bairros: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compara o conjunto de candidatos com e sem o varejo de pequeno porte.

    Os candidatos do CNEFE sao acrescentados DEPOIS da deduplicacao do OSM e nao
    alteram nenhum registro do OSM, entao o conjunto sem eles e exatamente o
    subconjunto de FONTE == 'OSM'. As duas rodadas saem de uma execucao so.
    """
    subtitulo("13. Comparacao: com e sem o varejo de pequeno porte")

    if not INCLUIR_MERCADO_PEQUENO:
        print("INCLUIR_MERCADO_PEQUENO = False — rodada unica, sem comparacao.")
        return pd.DataFrame()

    sem = candidatos[candidatos["FONTE"] == "OSM"]
    com = candidatos
    ordem = ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]

    print(f"   {'indicador':<40} {'sem':>10} {'com':>10} {'variacao':>12}")
    print(f"   {'total de candidatos':<40} {len(sem):>10} {len(com):>10} "
          f"{100 * (len(com) / len(sem) - 1):>11.1f}%")

    b_sem = set(sem["CD_BAIRRO"])
    b_com = set(com["CD_BAIRRO"])
    print(f"   {'bairros com ao menos um candidato':<40} {len(b_sem):>10} {len(b_com):>10} "
          f"{len(b_com) - len(b_sem):>+12}")
    print(f"   {'bairros sem nenhum candidato':<40} {len(bairros) - len(b_sem):>10} "
          f"{len(bairros) - len(b_com):>10} {len(b_sem) - len(b_com):>+12}")

    resgatados = bairros[bairros["CD_BAIRRO"].isin(b_com - b_sem)]
    if len(resgatados) > 0:
        print(f"\n   bairros que deixam de estar vazios ({len(resgatados)}):")
        for linha in resgatados.sort_values("POP", ascending=False).itertuples():
            n = int((com["CD_BAIRRO"] == linha.CD_BAIRRO).sum())
            print(f"      {linha.NM_BAIRRO[:26]:<26} {linha.REGIAO_FUNCIONAL:<12} "
                  f"{numero_br(linha.POP):>8} hab   {n} candidato(s)")

    ainda_vazios = bairros[~bairros["CD_BAIRRO"].isin(b_com)]
    if len(ainda_vazios) > 0:
        print(f"\n   bairros ainda sem candidato ({len(ainda_vazios)}):")
        for linha in ainda_vazios.sort_values("POP", ascending=False).itertuples():
            print(f"      {linha.NM_BAIRRO[:26]:<26} {linha.REGIAO_FUNCIONAL:<12} "
                  f"{numero_br(linha.POP):>8} hab")

    print(f"\n   distribuicao por regiao funcional:")
    print(f"   {'regiao':<14} {'sem':>8} {'com':>8} {'acrescimo':>11} {'part. sem':>11} {'part. com':>11}")
    linhas = []
    for regiao in ordem:
        n_sem = int((sem["REGIAO_FUNCIONAL"] == regiao).sum())
        n_com = int((com["REGIAO_FUNCIONAL"] == regiao).sum())
        print(
            f"   {regiao:<14} {n_sem:>8} {n_com:>8} {n_com - n_sem:>+11} "
            f"{100 * n_sem / len(sem):>10.1f}% {100 * n_com / len(com):>10.1f}%"
        )
        linhas.append(
            {
                "regiao_funcional": regiao,
                "candidatos_sem": n_sem,
                "candidatos_com": n_com,
                "acrescimo": n_com - n_sem,
                "participacao_sem_pct": round(100 * n_sem / len(sem), 1),
                "participacao_com_pct": round(100 * n_com / len(com), 1),
            }
        )
    return pd.DataFrame(linhas)


def validar_consistencia_setorial(
    candidatos: gpd.GeoDataFrame, cnefe: gpd.GeoDataFrame
) -> None:
    """Confronto com totais setoriais, para as referencias que estiverem declaradas."""
    subtitulo("14. Consistencia setorial")

    referencias = [
        ("supermercado", N_SUPERMERCADOS_ABRAS, "ABRAS — publica por UF, nao por municipio"),
        ("farmacia", N_FARMACIAS_CFF, "CFF — publica por UF"),
        ("agencia_postal", N_AGENCIAS_CORREIOS,
         "Correios (2026), localizador de agencias, acesso em 09/09/2026 — municipal"),
    ]
    contagem = candidatos["TIPO"].value_counts()

    if not VALIDAR_CONSISTENCIA_SETORIAL:
        print("desligada (VALIDAR_CONSISTENCIA_SETORIAL = False).")
        return

    print(f"   {'tipo':<18} {'OSM':>6} {'CNEFE':>7} {'referencia':>12} {'OSM/ref':>9}  fonte")
    for tipo, referencia, fonte in referencias:
        n_osm = int(contagem.get(tipo, 0))
        n_cnefe = int((cnefe["CLASSE"] == tipo).sum())
        if referencia is None:
            print(f"   {tipo:<18} {n_osm:>6} {n_cnefe:>7} {'pendente':>12} {'—':>9}  {fonte}")
            continue
        checar(
            referencia > 0,
            f"Referencia setorial de {tipo} declarada como {referencia}; deve ser positiva.",
        )
        print(f"   {tipo:<18} {n_osm:>6} {n_cnefe:>7} {referencia:>12} "
              f"{n_osm / referencia:>8.2f}x  {fonte}")

    print("\nRessalva metodologica (docs/decisoes_etapa4.md): ABRAS e CFF publicam por")
    print("   UF, sob definicoes que nao correspondem nem ao OSM nem ao CNEFE. Comparar")
    print("   contagem municipal com total estadual produz divergencia inconclusiva. A")
    print("   excecao sao os Correios, cuja definicao e inequivoca e o recorte, municipal.")


def auditar_agencias_postais(
    candidatos: gpd.GeoDataFrame, cnefe: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Lista nominalmente as agencias postais do OSM e o seu par no CNEFE.

    O localizador dos Correios (16) coincide exatamente com o CNEFE (16), enquanto
    o OSM traz 19. A lista identifica quais nao tem correspondencia — candidatas a
    agencia desativada ainda mapeada ou ponto etiquetado indevidamente.
    """
    subtitulo("15. Auditoria nominal das agencias postais")

    alvo = candidatos[candidatos["TIPO"] == "agencia_postal"].copy()
    referencia = cnefe[cnefe["CLASSE"] == "agencia_postal"]
    print(f"OSM {len(alvo)} · CNEFE {len(referencia)} · Correios (2026) "
          f"{N_AGENCIAS_CORREIOS if N_AGENCIAS_CORREIOS is not None else 'pendente'}")

    vizinho = gpd.sjoin_nearest(
        alvo[["geometry"]], referencia[["DESCRICAO", "geometry"]],
        how="left", distance_col="DIST_M",
    )
    vizinho = vizinho[~vizinho.index.duplicated(keep="first")]
    alvo["DIST_CNEFE_M"] = vizinho["DIST_M"].to_numpy()
    alvo["PAR_CNEFE"] = vizinho["DESCRICAO"].to_numpy()
    alvo["TEM_PAR"] = alvo["DIST_CNEFE_M"] <= RAIO_PAREAMENTO_PRINCIPAL

    print(f"\n{'#':>3} {'nome (OSM)':<32} {'bairro':<20} {'dist. CNEFE':>12}  par")
    for i, linha in enumerate(alvo.sort_values("DIST_CNEFE_M").itertuples(), start=1):
        nome = (linha.NOME or "(sem nome)")[:32]
        marca = "sim" if linha.TEM_PAR else "NAO"
        print(f"   {i:>3} {nome:<32} {linha.NM_BAIRRO[:20]:<20} "
              f"{linha.DIST_CNEFE_M:>10.0f} m  {marca}")

    sem_par = alvo[~alvo["TEM_PAR"]]
    print(f"\nsem correspondencia no CNEFE a {RAIO_PAREAMENTO_PRINCIPAL:.0f} m: {len(sem_par)}")
    for linha in sem_par.sort_values("DIST_CNEFE_M", ascending=False).itertuples():
        print(f"      {(linha.NOME or '(sem nome)')[:32]:<32} {linha.NM_BAIRRO[:20]:<20} "
              f"{linha.DIST_CNEFE_M:>8.0f} m do mais proximo")
    print("\nCandidatas a agencia desativada ainda mapeada ou ponto etiquetado")
    print("   indevidamente. NAO foram removidas do conjunto: decisao pendente.")

    return alvo[["OSM_ID" if "OSM_ID" in alvo.columns else "id", "NOME", "NM_BAIRRO",
                 "REGIAO_FUNCIONAL", "DIST_CNEFE_M", "PAR_CNEFE", "TEM_PAR"]]


# =============================================================================
# SAIDAS
# =============================================================================


def salvar(
    candidatos: gpd.GeoDataFrame,
    validacao: pd.DataFrame,
    validacao_regiao: pd.DataFrame,
    comparacao: pd.DataFrame,
    descartados: pd.DataFrame,
    agencias: pd.DataFrame,
    fusoes: pd.DataFrame,
) -> None:
    subtitulo("16. Gravacao das saidas")
    DIR_TRATADOS.mkdir(parents=True, exist_ok=True)
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)

    colunas = [
        "id", "FONTE", "TIPO", "NOME", "MARCA", "OPERADOR", "HORARIO",
        "CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL", "N_FUNDIDOS", "geometry",
    ]
    saida = candidatos[[c for c in colunas if c in candidatos.columns]].copy()
    saida = saida.rename(columns={"id": "OSM_ID"})
    saida["OSM_ID"] = saida["OSM_ID"].astype("string")
    # O driver GPKG grava pd.NA de colunas 'string' como o TEXTO "<NA>", que na
    # leitura seguinte volta como valor preenchido e finge cobertura que nao existe.
    # Converter para object com None garante NULL de verdade no arquivo.
    for coluna in ["OSM_ID", "HORARIO", "NOME", "MARCA", "OPERADOR"]:
        if coluna in saida.columns:
            saida[coluna] = saida[coluna].astype(object).where(saida[coluna].notna(), None)

    ARQ_CANDIDATOS.unlink(missing_ok=True)
    saida.to_file(ARQ_CANDIDATOS, layer="candidatos", driver="GPKG")

    arq_validacao = DIR_TABELAS / "validacao_osm_cnefe.csv"
    validacao.to_csv(arq_validacao, index=False, sep=";", decimal=",")

    por_bairro = (
        candidatos.groupby(["REGIAO_FUNCIONAL", "NM_BAIRRO", "TIPO"])
        .size()
        .unstack("TIPO", fill_value=0)
        .reset_index()
    )
    por_bairro["total"] = por_bairro[[c for c in TIPOS if c in por_bairro.columns]].sum(axis=1)
    arq_bairro = DIR_TABELAS / "candidatos_por_bairro.csv"
    por_bairro.sort_values("total", ascending=False).to_csv(
        arq_bairro, index=False, sep=";", decimal=","
    )

    arquivos = [ARQ_CANDIDATOS, arq_validacao, arq_bairro]

    if len(validacao_regiao) > 0:
        arq_regiao = DIR_TABELAS / "validacao_osm_cnefe_por_regiao.csv"
        validacao_regiao.to_csv(arq_regiao, index=False, sep=";", decimal=",")
        arquivos.append(arq_regiao)

    arq_agencias = DIR_TABELAS / "auditoria_agencias_postais.csv"
    agencias.to_csv(arq_agencias, index=False, sep=";", decimal=",")
    arquivos.append(arq_agencias)

    if len(comparacao) > 0:
        arq_comparacao = DIR_TABELAS / "candidatos_com_sem_mercado_pequeno.csv"
        comparacao.to_csv(arq_comparacao, index=False, sep=";", decimal=",")
        arquivos.append(arq_comparacao)
    arq_fusoes = DIR_TABELAS / "candidatos_fundidos_nome_aproximado.csv"
    fusoes.to_csv(arq_fusoes, index=False, sep=";", decimal=",")
    arquivos.append(arq_fusoes)
    if len(descartados) > 0:
        arq_descartados = DIR_TABELAS / "candidatos_descartados.csv"
        descartados[["id", "TIPO", "NOME", "MOTIVO", "DIST_TERRA_M"]].to_csv(
            arq_descartados, index=False, sep=";", decimal=","
        )
        arquivos.append(arq_descartados)

    for caminho in arquivos:
        print(f"   {caminho.relative_to(RAIZ).as_posix():<48} {caminho.stat().st_size / 1024:>8.1f} KB")


def resumir(candidatos: gpd.GeoDataFrame) -> None:
    subtitulo("17. Resumo dos candidatos")

    print(f"   {'regiao':<14}" + "".join(f"{t[:13]:>15}" for t in TIPOS) + f"{'total':>9}")
    ordem = ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]
    tabela = (
        candidatos.groupby(["REGIAO_FUNCIONAL", "TIPO"]).size().unstack("TIPO", fill_value=0)
    )
    for tipo in TIPOS:
        if tipo not in tabela.columns:
            tabela[tipo] = 0
    tabela = tabela.reindex(ordem).fillna(0).astype(int)
    for regiao, linha in tabela.iterrows():
        print(
            f"   {regiao:<14}" + "".join(f"{int(linha[t]):>15}" for t in TIPOS)
            + f"{int(linha[TIPOS].sum()):>9}"
        )
    print(
        f"   {'TOTAL':<14}" + "".join(f"{int(tabela[t].sum()):>15}" for t in TIPOS)
        + f"{int(tabela[TIPOS].to_numpy().sum()):>9}"
    )

    print(f"\n   candidatos por 10 mil habitantes ....... "
          f"{10_000 * len(candidatos) / 537_211:.1f}")


# =============================================================================
# EXECUCAO
# =============================================================================


def main() -> int:
    inicio = time.perf_counter()

    titulo(
        "03_candidatos.py — candidatos a ponto de retirada\n"
        f"tipos: {', '.join(TIPOS)}  ·  "
        f"dedup {RAIO_DEDUP_M:.0f} m + nome  ·  "
        f"pareamento {RAIO_PAREAMENTO_PRINCIPAL:.0f} m"
    )

    bairros, terra, agua, vizinhos = carregar_territorio()
    bruto = extrair_osm(terra)
    dados = tratar_osm(bruto)
    dados, descartados = verificar_localizacao(dados, terra, agua, vizinhos)
    dados = deduplicar(dados)
    auditar_duplicatas_residuais(dados)
    dados["FONTE"] = "OSM"

    cnefe = classificar_cnefe()
    verificar_plausibilidade(cnefe)

    if INCLUIR_MERCADO_PEQUENO:
        extras = preparar_mercados_pequenos(cnefe, dados, terra, agua, vizinhos)
        extras["FONTE"] = "CNEFE"
        dados = gpd.GeoDataFrame(
            pd.concat([dados, extras], ignore_index=True), geometry="geometry", crs=dados.crs
        )
    else:
        subtitulo("9. Varejo de pequeno porte do CNEFE como candidato")
        print("INCLUIR_MERCADO_PEQUENO = False — nao acrescentado.")

    dados, fusoes = deduplicar_nome_aproximado(dados)
    candidatos = associar_bairro(dados, bairros)

    # A validacao cruzada usa SO os candidatos de fonte OSM: os do CNEFE se
    # parearia consigo mesmos a distancia zero e a taxa perderia sentido.
    somente_osm = candidatos[candidatos["FONTE"] == "OSM"]
    validacao = parear_osm_cnefe(somente_osm, cnefe)
    validacao_regiao = parear_por_regiao(somente_osm, cnefe)
    comparacao = comparar_com_e_sem(candidatos, bairros)
    validar_consistencia_setorial(candidatos, cnefe)
    agencias = auditar_agencias_postais(candidatos, cnefe)

    salvar(candidatos, validacao, validacao_regiao, comparacao, descartados, agencias, fusoes)
    resumir(candidatos)

    titulo(
        f"Concluido em {time.perf_counter() - inicio:.1f} s  ·  "
        f"{len(candidatos)} candidatos em {candidatos['CD_BAIRRO'].nunique()} bairros"
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
