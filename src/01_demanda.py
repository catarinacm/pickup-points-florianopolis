"""
01_demanda.py — Demanda espacial por bairro em Florianopolis/SC (IBGE 4205407).

Constroi os pontos de demanda do modelo de localizacao de pickup points a partir
dos setores censitarios do Censo 2022, agregando populacao e renda por bairro,
classificando os bairros nas cinco regioes funcionais do Decreto no 29.142/2026 e
gravando as camadas usadas por todas as etapas seguintes.

Entradas (somente leitura, em data/brutos/):
    ibge/SC_setores_CD2022.gpkg                        malha de setores (so a geometria)
    ibge/Agregados_por_setores_basico_BR.csv           populacao e domicilios por setor
    ibge/Agregados_por_setores_renda_responsavel_BR.xlsx   renda do responsavel por setor
    ibge/SC_bairros_CD2022.gpkg                        malha de bairros (referencia espacial)
    ibge/Agregados_por_bairros_basico_BR.csv           gabarito de validacao
    cnefe/4205407_FLORIANOPOLIS.csv                    enderecos, para o rateio por domicilios

Saidas:
    data/tratados/bairros.gpkg                        poligonos com populacao, renda e regiao
    data/tratados/pontos_demanda.gpkg                 ponto representativo de cada bairro
    data/tratados/alocacao_setores.parquet            rastreabilidade setor -> bairro
    data/tratados/setores.gpkg                        setores de terra com populacao, renda e bairro
    data/tratados/pontos_demanda_setor.gpkg           ponto representativo de cada setor habitado
    results/nao_usados/tabelas/representatividade_pontos_demanda.csv   domicilios perto do ponto, bairro x setor
    results/tabelas/validacao_bairros.csv           comparacao com o gabarito do IBGE
    results/tabelas/resumo_regioes.csv              populacao e renda por regiao funcional

Executar com o ambiente 'tcc' ativo:
    conda activate tcc
    python src/01_demanda.py
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
# Toda escolha metodologica deste script esta neste bloco. Nada abaixo daqui
# precisa ser editado para trocar de malha de bairros ou de regra de alocacao.

RAIZ = Path(__file__).resolve().parents[1]

DIR_BRUTOS = RAIZ / "data" / "brutos"
DIR_TRATADOS = RAIZ / "data" / "tratados"
DIR_TABELAS = RAIZ / "results" / "tabelas"
# Resultados que NAO entram na monografia vao para results/nao_usados/. A comparacao
# bairro x setor saiu do escopo quando o setor censitario virou a unidade de demanda
# (docs/decisoes_etapa10.md): a tabela de representatividade continua sendo gerada,
# mas grava ali, para nao se misturar com o que vai para o texto.
DIR_TABELAS_NAO_USADAS = RAIZ / "results" / "nao_usados" / "tabelas"

ARQ_SETORES_GPKG = DIR_BRUTOS / "ibge" / "SC_setores_CD2022.gpkg"
ARQ_BASICO_CSV = DIR_BRUTOS / "ibge" / "Agregados_por_setores_basico_BR.csv"
ARQ_RENDA_XLSX = DIR_BRUTOS / "ibge" / "Agregados_por_setores_renda_responsavel_BR.xlsx"
ARQ_BAIRROS_CSV = DIR_BRUTOS / "ibge" / "Agregados_por_bairros_basico_BR.csv"
ARQ_CNEFE_CSV = DIR_BRUTOS / "cnefe" / "4205407_FLORIANOPOLIS.csv"

# Cache do recorte de Florianopolis da planilha de renda. Ler o XLSX inteiro
# (458.772 linhas) leva alguns minutos; o recorte tem 964 linhas.
ARQ_CACHE_RENDA = DIR_TRATADOS / "_cache_renda_floripa.parquet"
USAR_CACHE_RENDA = True

CD_MUN = "4205407"
CRS_TRABALHO = 31982  # SIRGAS 2000 / UTM 22S — unidade em metros
POP_REFERENCIA = 537_211  # Censo 2022, municipio de Florianopolis
POP_GABARITO_BAIRROS = 519_463  # Agregados_por_bairros_basico_BR.csv, 87 bairros

# -----------------------------------------------------------------------------
# Malha de bairros
# -----------------------------------------------------------------------------
# 'ibge_87'    delimitacao vigente a definicao do escopo. Os setores ja trazem
#              CD_BAIRRO preenchido, exceto 109 deles.
# 'decreto_55' delimitacao do Decreto no 29.142/2026 (REPLAN, 15/07/2026). Nenhum
#              setor traz esse codigo, logo TODOS sao alocados espacialmente.
#
# Trocar de malha = trocar a linha MALHA_BAIRROS abaixo.

MALHA_BAIRROS = "ibge_87"

MALHAS = {
    "ibge_87": {
        "arquivo": DIR_BRUTOS / "ibge" / "SC_bairros_CD2022.gpkg",
        "col_codigo": "CD_BAIRRO",
        "col_nome": "NM_BAIRRO",
        "n_esperado": 87,
        # Os setores do IBGE ja carregam a atribuicao setor -> bairro pronta.
        "usar_atributo_do_setor": True,
        "fonte": "IBGE, malha de bairros do Censo 2022",
    },
    "decreto_55": {
        "arquivo": DIR_BRUTOS / "documentos" / "bairros_decreto_29142_2026.gpkg",
        "col_codigo": "CD_BAIRRO",
        "col_nome": "NM_BAIRRO",
        "n_esperado": 55,
        # Nenhum setor traz o codigo do decreto: alocacao 100% espacial.
        "usar_atributo_do_setor": False,
        "fonte": "Decreto no 29.142/2026 (REPLAN), PMF",
    },
}

# -----------------------------------------------------------------------------
# Massas d'agua (setores com CD_SIT == 9)
# -----------------------------------------------------------------------------
# A malha censitaria de Florianopolis inclui 8 setores de massa d'agua — baias
# Norte e Sul, Lagoa da Conceicao e faixa oceanica —, somando 261,9 km2 dos
# 675,3 km2 do municipio. Todos tem populacao zero e nenhum traz CD_BAIRRO.
#
# Sem excluir esses setores, a agua e anexada ao bairro vizinho e o
# representative_point() do bairro cai DENTRO DA BAIA: na primeira execucao, 6
# dos 87 pontos de demanda ficaram sobre a agua, com deslocamentos de ate 5,6 km
# (Sambaqui), 3,9 km (Ribeirao da Ilha) e 2,9 km (Abraao). Isso contaminaria a
# matriz de distancias da Etapa 6 e, por consequencia, o modelo inteiro.
#
# Excluir e seguro do ponto de vista demografico: os setores de agua nao carregam
# populacao nem domicilios, e o script confere isso antes de descarta-los.
EXCLUIR_MASSAS_DAGUA = True
CD_SIT_MASSA_DAGUA = "9"

# -----------------------------------------------------------------------------
# DECISAO 1 — os setores sem CD_BAIRRO (17.748 habitantes)
# -----------------------------------------------------------------------------
# ATENCAO, resultado apurado na execucao: na malha 'ibge_87' a interseccao
# espacial e VAZIA. A malha oficial de bairros do IBGE (176,7 km2) e exatamente a
# uniao dos 895 setores que ja trazem CD_BAIRRO; os 109 orfaos ocupam os 498,6 km2
# restantes e nao encostam nela. Por isso a opcao 'interseccao' degenera na regra
# de proximidade quando aplicada a essa malha — o script avisa quando isso ocorre.
# A opcao segue implementada porque passa a ser util na malha 'decreto_55', que e
# independente da malha de setores.
#
# 'interseccao'            recorta o setor contra a malha de bairros de referencia
#                          e rateia a populacao entre as partes resultantes.
# 'rateio_por_domicilios'  divide a populacao do setor entre bairros conforme o
#                          bairro mais proximo de cada domicilio do CNEFE contido
#                          nele — permite cobertura parcial sem depender de
#                          sobreposicao geometrica.
# 'bairro_mais_proximo'    atribui o setor inteiro ao bairro mais proximo do ponto
#                          representativo interno do setor (distancia ao poligono,
#                          que e zero quando o ponto cai dentro dele).
REGRA_SETORES_ORFAOS = "rateio_por_domicilios"

# Peso do rateio quando REGRA_SETORES_ORFAOS == 'interseccao'.
# 'domicilios'  numero de enderecos de especie 1 do CNEFE dentro de cada parte.
# 'area'        area da parte.
PESO_DO_RATEIO = "domicilios"

# O que fazer quando o setor nao tem nenhum domicilio do CNEFE para pesar o rateio.
# 'area'                 rateia por area da interseccao (so vale em 'interseccao').
# 'bairro_mais_proximo'  trata o setor inteiro pela regra de proximidade.
FALLBACK_SEM_DOMICILIO = "bairro_mais_proximo"

# -----------------------------------------------------------------------------
# DECISAO 2 — renda do responsavel ausente
# -----------------------------------------------------------------------------
# Dois casos distintos, tratados por parametros separados:
#   suprimida  = 23 setores cujas variaveis V06001..V06006 vem com 'X' (sigilo)
#   sem_linha  = 40 setores sem nenhuma linha no arquivo de renda (todos com
#                populacao zero)
#
# 'mediana_bairro'     imputa a mediana de V06004 do bairro de destino
# 'mediana_municipio'  imputa a mediana de V06004 do municipio
# 'excluir'            fica fora do numerador e do denominador da media ponderada
REGRA_RENDA_SUPRIMIDA = "mediana_bairro"
REGRA_RENDA_SEM_LINHA = "excluir"

# -----------------------------------------------------------------------------
# DECISAO 3 — setores sem endereco no CNEFE
# -----------------------------------------------------------------------------
# A correspondencia CNEFE <-> malha censitaria e ESPACIAL, nao cadastral: o CNEFE
# reflete a coleta de 2022 e a malha, a edicao de 2026, e houve renumeracao de
# setores no intervalo (79 codigos do CNEFE nao existem na malha; 180 setores da
# malha ficariam sem endereco numa juncao por codigo, contra 35 pela juncao
# espacial). Ver data/FONTES.md.
JUNCAO_CNEFE = "espacial"  # | 'codigo' (mantida so para o diagnostico comparativo)

# -----------------------------------------------------------------------------
# Regioes funcionais — Decreto no 29.142/2026 combinado com os distritos do IBGE
# -----------------------------------------------------------------------------
# Os onze distritos nao-sede mapeiam direto em Norte, Leste e Sul. O distrito sede
# 'Florianopolis' e dividido em Regiao Continental (bairros dos antigos distritos
# de Estreito e Coqueiros) e Centro/Sede.

DISTRITO_REGIAO = {
    "Cachoeira do Bom Jesus": "Norte",
    "Canasvieiras": "Norte",
    "Ingleses": "Norte",
    "Ratones": "Norte",
    "Santo Antônio de Lisboa": "Norte",
    "Lagoa da Conceição": "Leste",
    "Rio Vermelho": "Leste",
    "Barra da Lagoa": "Leste",
    "Campeche": "Sul",
    "Pântano do Sul": "Sul",
    "Ribeirão da Ilha": "Sul",
}

BAIRROS_CONTINENTE = {
    "Abraão",
    "Balneário",
    "Bom Abrigo",
    "Canto",
    "Capoeiras",
    "Coloninha",
    "Coqueiros",
    "Estreito",
    "Itaguaçu",
    "Jardim Atlântico",
    "Monte Cristo",
}

DISTRITO_SEDE = "Florianópolis"
REGIAO_SEDE = "Centro/Sede"
REGIAO_CONTINENTE = "Continente"

ORDEM_REGIOES = ["Centro/Sede", "Continente", "Norte", "Leste", "Sul"]

# Controle da classificacao, valido apenas para a malha de 87 bairros e ANTES da
# entrada dos 109 setores orfaos. Serve para detectar erro no mapeamento.
CONTROLE_REGIOES_SEM_ORFAOS = {
    "Centro/Sede": (13, 182_811),
    "Continente": (11, 93_018),
    "Norte": (27, 108_367),
    "Leste": (9, 45_541),
    "Sul": (27, 89_726),
}

# =============================================================================
# INFRAESTRUTURA DE VERIFICACAO
# =============================================================================


class FalhaDeSanidade(Exception):
    """Erro de verificacao: o dado nao esta como o metodo exige."""


def checar(condicao: bool, mensagem: str) -> None:
    """Interrompe a execucao com mensagem explicita se a verificacao falhar."""
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
    """Formata numero no padrao brasileiro (1.234,56)."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def verificar_ausencia_de_centroid() -> None:
    """Garante que o script nao usa centroid() em lugar de representative_point().

    Varios bairros de Florianopolis sao recortados pela orla e o centroide
    geometrico cai na agua, fora do proprio poligono.
    """
    marcador = "verificacao-centroid"
    proibido = "." + "centroid"
    linhas = Path(__file__).read_text(encoding="utf-8").splitlines()
    ocorrencias = [
        (n, linha.strip())
        for n, linha in enumerate(linhas, start=1)
        if proibido in linha and marcador not in linha
    ]
    checar(
        not ocorrencias,
        "Uso de centroid() detectado em 01_demanda.py, linhas "
        f"{[n for n, _ in ocorrencias]}. Usar representative_point().",
    )


def para_numero(serie: pd.Series, decimal: str = ",") -> pd.Series:
    """Converte texto do IBGE em numero.

    Os arquivos do IBGE nao sao homogeneos: os CSVs de agregados usam VIRGULA
    decimal ('0,5393102') e a planilha de renda usa PONTO ('17796.13'). O
    separador precisa ser declarado em cada chamada — ler renda como se fosse
    virgula multiplica os valores por cem.

    Sentinelas nao numericos ('X' de sigilo, '.' de nao aplicavel) viram NaN,
    NUNCA zero.
    """
    checar(decimal in (",", "."), f"Separador decimal invalido: {decimal!r}.")
    texto = serie.astype("string").str.strip()
    texto = texto.replace({"X": pd.NA, ".": pd.NA, "": pd.NA})
    if decimal == ",":
        # Nenhum dos CSVs usa separador de milhar; um ponto remanescente aqui
        # seria sinal de leitura errada e precisa aparecer, nao ser silenciado.
        suspeitas = texto.str.contains(".", regex=False, na=False)
        checar(
            not bool(suspeitas.any()),
            "Valor com ponto encontrado em campo de decimal-virgula "
            f"(ex.: {texto[suspeitas].head(3).tolist()}). Conferir o separador.",
        )
        texto = texto.str.replace(",", ".", regex=False)
    else:
        suspeitas = texto.str.contains(",", regex=False, na=False)
        checar(
            not bool(suspeitas.any()),
            "Valor com virgula encontrado em campo de decimal-ponto "
            f"(ex.: {texto[suspeitas].head(3).tolist()}). Conferir o separador.",
        )
    return pd.to_numeric(texto, errors="coerce")


def arredondar_preservando_total(valores: pd.Series, total: int) -> pd.Series:
    """Arredonda para inteiros preservando exatamente a soma (maiores restos).

    O rateio dos setores orfaos produz fracoes de habitante. A monografia reporta
    populacao inteira e a verificacao de sanidade exige soma exata igual a
    537.211, entao o arredondamento nao pode ser feito termo a termo.
    """
    piso = np.floor(valores).astype("int64")
    faltam = int(total - piso.sum())
    checar(
        faltam >= 0,
        f"Arredondamento inconsistente: piso soma {piso.sum()}, total alvo {total}.",
    )
    if faltam > 0:
        resto = (valores - piso).sort_values(ascending=False)
        piso.loc[resto.index[:faltam]] += 1
    checar(
        int(piso.sum()) == total,
        f"Arredondamento nao preservou o total: {piso.sum()} != {total}.",
    )
    return piso


# =============================================================================
# LEITURA DOS DADOS
# =============================================================================


def carregar_setores() -> tuple[gpd.GeoDataFrame, gpd.GeoSeries]:
    """Malha de setores de Florianopolis, deduplicada e reprojetada.

    O GeoPackage traz 1.027 linhas para 1.004 setores: setores multipartes
    aparecem repetidos com atributos identicos. As partes sao DISSOLVIDAS, nunca
    descartadas, para nao perder territorio. Os atributos do arquivo sao da edicao
    11/2024 e sao ignorados: so a geometria e usada.
    """
    subtitulo("1. Malha de setores censitarios")
    bruto = gpd.read_file(ARQ_SETORES_GPKG, where=f"CD_MUN = '{CD_MUN}'")
    print(f"linhas lidas do GeoPackage ............. {len(bruto)}")
    print(f"CD_SETOR distintos ..................... {bruto['CD_SETOR'].nunique()}")
    print(f"CRS de origem .......................... {bruto.crs.to_string()}")

    if len(bruto) != bruto["CD_SETOR"].nunique():
        n_multipartes = len(bruto) - bruto["CD_SETOR"].nunique()
        print(f"setores multipartes a dissolver ........ {n_multipartes} linhas excedentes")

    setores = bruto[["CD_SETOR", "CD_SIT", "v0001", "geometry"]].dissolve(
        by="CD_SETOR", as_index=False, aggfunc="first"
    )
    checar(
        len(setores) == setores["CD_SETOR"].nunique(),
        "Ainda ha CD_SETOR duplicado apos a dissolucao da malha.",
    )

    setores = setores.to_crs(CRS_TRABALHO)
    checar(
        setores.crs.to_epsg() == CRS_TRABALHO,
        f"A malha de setores nao esta em EPSG:{CRS_TRABALHO}.",
    )
    invalidas = int((~setores.geometry.is_valid).sum())
    if invalidas:
        print(f"geometrias invalidas corrigidas ........ {invalidas}")
        setores["geometry"] = setores.geometry.make_valid()

    setores["AREA_M2_SETOR"] = setores.geometry.area
    print(f"setores apos dissolucao ................ {len(setores)}")
    print(f"CRS de trabalho ........................ EPSG:{setores.crs.to_epsg()}")
    print(f"area total do municipio ................ {numero_br(setores['AREA_M2_SETOR'].sum() / 1e6, 1)} km2")

    agua = setores[setores["CD_SIT"] == CD_SIT_MASSA_DAGUA]
    print(f"setores de massa d'agua (CD_SIT=9) ..... {len(agua)}")
    print(f"   area ................................ {numero_br(agua['AREA_M2_SETOR'].sum() / 1e6, 1)} km2")
    if EXCLUIR_MASSAS_DAGUA and len(agua) > 0:
        # Descartar so e legitimo se nao houver ninguem morando ali.
        populacao_na_agua = int(para_numero(agua["v0001"], decimal=".").fillna(0).sum())
        checar(
            populacao_na_agua == 0,
            f"Os setores de massa d'agua somam {populacao_na_agua} habitantes na malha; "
            "nao podem ser descartados sem perder populacao.",
        )
        setores = setores[setores["CD_SIT"] != CD_SIT_MASSA_DAGUA].reset_index(drop=True)
        print(f"   excluidos (populacao zero) .......... sim")
        print(f"setores restantes ...................... {len(setores)}")
        print(f"area de terra do municipio ............. {numero_br(setores['AREA_M2_SETOR'].sum() / 1e6, 1)} km2")

    return setores.drop(columns=["CD_SIT", "v0001"]), agua.geometry


def carregar_basico() -> pd.DataFrame:
    """Populacao e domicilios por setor, do CSV de 2026 (nao da malha)."""
    subtitulo("2. Agregados por setor — arquivo basico")
    colunas = [
        "CD_SETOR", "SITUACAO", "CD_DIST", "NM_DIST",
        "CD_BAIRRO", "NM_BAIRRO",
        "v0001", "v0002", "v0005", "v0007",
    ]
    pedacos = []
    leitor = pd.read_csv(
        ARQ_BASICO_CSV, sep=";", encoding="latin-1", dtype=str,
        usecols=colunas + ["CD_MUN"], chunksize=200_000,
    )
    for pedaco in leitor:
        pedacos.append(pedaco[pedaco["CD_MUN"] == CD_MUN])
    basico = pd.concat(pedacos, ignore_index=True)[colunas]

    print(f"linhas de Florianopolis ................ {len(basico)}")
    print(f"CD_SETOR distintos ..................... {basico['CD_SETOR'].nunique()}")
    checar(
        len(basico) == basico["CD_SETOR"].nunique(),
        "O arquivo basico traz CD_SETOR duplicado — dissolver antes de somar.",
    )

    basico = basico.rename(
        columns={"v0001": "POP", "v0002": "DOM_TOTAL", "v0005": "MORADORES_MEDIA", "v0007": "DOM_OCUPADOS"}
    )
    for coluna in ["POP", "DOM_TOTAL", "MORADORES_MEDIA", "DOM_OCUPADOS"]:
        basico[coluna] = para_numero(basico[coluna])

    checar(
        basico["POP"].notna().all(),
        "Ha populacao nao numerica no arquivo basico — investigar antes de somar.",
    )
    total = int(basico["POP"].sum())
    print(f"populacao somada ....................... {numero_br(total)}")
    checar(
        total == POP_REFERENCIA,
        f"Populacao do arquivo basico e {total}, esperado {POP_REFERENCIA}.",
    )

    # O IBGE usa '.' como sentinela de bairro nao atribuido, nao valor nulo.
    basico["SEM_BAIRRO_IBGE"] = basico["CD_BAIRRO"].isin([".", ""]) | basico["CD_BAIRRO"].isna()
    orfaos = basico.loc[basico["SEM_BAIRRO_IBGE"]]
    print(f"setores sem CD_BAIRRO .................. {len(orfaos)}")
    print(f"populacao nesses setores ............... {numero_br(orfaos['POP'].sum())}")
    print(f"bairros distintos preenchidos .......... {basico.loc[~basico['SEM_BAIRRO_IBGE'], 'CD_BAIRRO'].nunique()}")
    print(f"setores com populacao zero ............. {(basico['POP'] == 0).sum()}")
    return basico


def carregar_renda(setores_do_municipio: set[str]) -> pd.DataFrame:
    """Renda media do responsavel por setor (V06004), ponderavel por V06001.

    Distingue os dois motivos de ausencia:
        'X'         supressao por sigilo estatistico
        sem linha   setor ausente do arquivo
    """
    subtitulo("3. Agregados por setor — renda do responsavel")

    if USAR_CACHE_RENDA and ARQ_CACHE_RENDA.exists():
        renda = pd.read_parquet(ARQ_CACHE_RENDA)
        print(f"lido do cache .......................... {ARQ_CACHE_RENDA.name}")
    else:
        print("lendo a planilha completa (458.772 linhas), pode levar alguns minutos...")
        completo = pd.read_excel(ARQ_RENDA_XLSX, sheet_name=0, dtype=str)
        renda = completo[completo["CD_SETOR"].isin(setores_do_municipio)].copy()
        ARQ_CACHE_RENDA.parent.mkdir(parents=True, exist_ok=True)
        renda.to_parquet(ARQ_CACHE_RENDA, index=False)
        print(f"cache gravado .......................... {ARQ_CACHE_RENDA.name}")

    print(f"setores de Florianopolis com linha ..... {len(renda)}")
    checar(
        len(renda) == renda["CD_SETOR"].nunique(),
        "O arquivo de renda traz CD_SETOR duplicado.",
    )

    # A supressao por sigilo marca TODAS as variaveis do setor com 'X'.
    renda["RENDA_SUPRIMIDA"] = renda["V06004"].astype("string").str.strip().eq("X")
    renda["RENDA_MEDIA"] = para_numero(renda["V06004"], decimal=".")
    renda["DOM_RESPONSAVEL"] = para_numero(renda["V06001"], decimal=".")

    n_suprimidos = int(renda["RENDA_SUPRIMIDA"].sum())
    sem_linha = setores_do_municipio - set(renda["CD_SETOR"])
    print(f"setores com renda suprimida ('X') ...... {n_suprimidos}")
    print(f"setores sem linha no arquivo ........... {len(sem_linha)}")

    validos = renda.loc[renda["RENDA_MEDIA"].notna(), "RENDA_MEDIA"]
    print(
        f"V06004 valida .......................... min R$ {numero_br(validos.min(), 2)} · "
        f"mediana R$ {numero_br(validos.median(), 2)} · max R$ {numero_br(validos.max(), 2)}"
    )
    checar(
        int(renda.loc[renda["RENDA_SUPRIMIDA"], "RENDA_MEDIA"].fillna(-1).eq(0).sum()) == 0,
        "Valor 'X' de renda foi convertido em zero — erro de leitura.",
    )
    checar(
        renda.loc[renda["RENDA_SUPRIMIDA"], "RENDA_MEDIA"].isna().all(),
        "Setor com renda suprimida ficou com valor numerico.",
    )
    # Valores de referencia conferidos na coleta (ver data/FONTES.md). Servem de
    # sentinela contra erro de separador decimal, que passaria despercebido.
    for rotulo, obtido, esperado in [
        ("minimo", float(validos.min()), 1_473.28),
        ("mediana", float(validos.median()), 5_312.65),
        ("maximo", float(validos.max()), 28_213.33),
    ]:
        checar(
            abs(obtido - esperado) < 0.01,
            f"V06004: {rotulo} lido {obtido:.2f}, referencia {esperado:.2f}. "
            "Provavel erro de separador decimal na leitura da planilha de renda.",
        )

    return renda[["CD_SETOR", "RENDA_MEDIA", "DOM_RESPONSAVEL", "RENDA_SUPRIMIDA"]]


def carregar_domicilios_cnefe() -> gpd.GeoDataFrame:
    """Enderecos de especie 1 (domicilio particular) do CNEFE, como pontos."""
    subtitulo("4. CNEFE — domicilios particulares")
    cnefe = pd.read_csv(
        ARQ_CNEFE_CSV, sep=";", encoding="latin-1", dtype=str,
        usecols=["COD_SETOR", "COD_ESPECIE", "LATITUDE", "LONGITUDE"],
    )
    print(f"enderecos no arquivo ................... {numero_br(len(cnefe))}")
    contagem = cnefe["COD_ESPECIE"].value_counts().sort_index()
    print("por especie ............................ " + " · ".join(f"{k}: {numero_br(v)}" for k, v in contagem.items()))

    domicilios = cnefe[cnefe["COD_ESPECIE"] == "1"].copy()
    checar(
        domicilios["LATITUDE"].notna().all() and domicilios["LONGITUDE"].notna().all(),
        "Ha domicilio do CNEFE sem coordenada.",
    )
    pontos = gpd.GeoDataFrame(
        domicilios[["COD_SETOR"]],
        geometry=gpd.points_from_xy(
            domicilios["LONGITUDE"].astype(float), domicilios["LATITUDE"].astype(float)
        ),
        crs=4326,
    ).to_crs(CRS_TRABALHO)
    print(f"domicilios particulares (especie 1) .... {numero_br(len(pontos))}")
    print(f"CRS apos reprojecao .................... EPSG:{pontos.crs.to_epsg()}")
    return pontos


def carregar_malha_bairros_referencia() -> gpd.GeoDataFrame:
    """Malha de bairros usada como REFERENCIA ESPACIAL da alocacao.

    Nao e a geometria final dos bairros: a malha oficial do IBGE cobre 176,7 km2,
    cerca de um quarto do municipio. Os poligonos entregues por este script sao
    construidos dissolvendo os setores pela alocacao final.
    """
    subtitulo("5. Malha de bairros de referencia")
    config = MALHAS[MALHA_BAIRROS]
    print(f"malha selecionada ...................... {MALHA_BAIRROS} ({config['fonte']})")

    if not config["arquivo"].exists():
        raise FalhaDeSanidade(
            f"A malha '{MALHA_BAIRROS}' precisa do arquivo {config['arquivo']}, que ainda\n"
            "nao foi construido. Ele NAO existe publicado e nao ha o que baixar: a\n"
            "delimitacao do Decreto no 29.142/2026 nao foi divulgada em formato\n"
            "geoespacial.\n\n"
            "Para produzi-lo, agregar os 87 bairros do IBGE pelos nomes previstos no\n"
            "decreto — por exemplo, 'Campeche Central', 'Campeche Leste', 'Campeche Norte'\n"
            "e 'Campeche Sul' dissolvidos em 'Campeche' — e gravar o resultado nesse\n"
            "caminho com as colunas CD_BAIRRO, NM_BAIRRO e CD_MUN.\n\n"
            "Tarefa opcional de fim de projeto (rodada de robustez da Etapa 8b), nao\n"
            "pre-requisito de nenhuma etapa anterior."
        )

    bruto = gpd.read_file(config["arquivo"], where=f"CD_MUN = '{CD_MUN}'")
    print(f"linhas lidas ........................... {len(bruto)}")
    print(f"bairros distintos ...................... {bruto[config['col_codigo']].nunique()}")

    bairros = bruto[[config["col_codigo"], config["col_nome"], "geometry"]].dissolve(
        by=config["col_codigo"], as_index=False, aggfunc="first"
    )
    bairros = bairros.rename(
        columns={config["col_codigo"]: "CD_BAIRRO", config["col_nome"]: "NM_BAIRRO"}
    ).to_crs(CRS_TRABALHO)
    bairros["geometry"] = bairros.geometry.make_valid()

    checar(
        len(bairros) == config["n_esperado"],
        f"A malha '{MALHA_BAIRROS}' devolveu {len(bairros)} bairros, esperado {config['n_esperado']}.",
    )
    area_km2 = bairros.geometry.area.sum() / 1e6
    print(f"bairros apos dissolucao ................ {len(bairros)}")
    print(f"area coberta pela malha ................ {numero_br(area_km2, 1)} km2")
    return bairros


# =============================================================================
# CONTAGEM DE DOMICILIOS POR SETOR
# =============================================================================


def contar_domicilios_por_setor(
    pontos: gpd.GeoDataFrame, setores: gpd.GeoDataFrame, basico: pd.DataFrame
) -> pd.DataFrame:
    """Conta domicilios do CNEFE por setor e diagnostica os setores sem endereco.

    A juncao e ESPACIAL. O codigo de setor do CNEFE tem 16 caracteres (sufixo 'P')
    e reflete a numeracao da coleta de 2022, incompativel com a malha de 2026.
    """
    subtitulo("6. Domicilios do CNEFE por setor")

    juncao = gpd.sjoin(
        pontos, setores[["CD_SETOR", "geometry"]], how="left", predicate="within"
    )
    fora = int(juncao["CD_SETOR"].isna().sum())
    print(f"criterio de juncao ..................... {JUNCAO_CNEFE}")
    print(f"domicilios fora de qualquer setor ...... {fora}")

    contagem = (
        juncao.dropna(subset=["CD_SETOR"]).groupby("CD_SETOR").size().rename("DOM_CNEFE")
    )
    # O diagnostico vale para os setores efetivamente em uso: se as massas d'agua
    # foram excluidas, elas nao contam como 'setor sem endereco'.
    em_uso = basico[basico["CD_SETOR"].isin(set(setores["CD_SETOR"]))]
    resultado = em_uso[["CD_SETOR", "POP"]].merge(
        contagem, left_on="CD_SETOR", right_index=True, how="left"
    )
    resultado["DOM_CNEFE"] = resultado["DOM_CNEFE"].fillna(0).astype("int64")

    sem_endereco = resultado[resultado["DOM_CNEFE"] == 0]
    com_populacao = sem_endereco[sem_endereco["POP"] > 0]
    print(f"setores com ao menos um domicilio ...... {(resultado['DOM_CNEFE'] > 0).sum()} de {len(resultado)}")
    print(f"setores sem endereco ................... {len(sem_endereco)}")
    print(f"   destes, com populacao > 0 ........... {len(com_populacao)} ({numero_br(com_populacao['POP'].sum())} hab)")

    # Diagnostico comparativo: o que a juncao por codigo produziria.
    codigos_cnefe = set(pontos["COD_SETOR"].str.slice(0, 15))
    codigos_malha = set(setores["CD_SETOR"])
    print(
        "comparativo (juncao por codigo) ........ "
        f"{len(codigos_cnefe - codigos_malha)} codigos do CNEFE fora da malha · "
        f"{len(codigos_malha - codigos_cnefe)} setores da malha sem endereco"
    )

    if len(com_populacao) > 0:
        print("setores com populacao e sem endereco no CNEFE:")
        for linha in com_populacao.sort_values("POP", ascending=False).itertuples():
            print(f"   {linha.CD_SETOR}  {numero_br(linha.POP):>7} hab")

    return resultado[["CD_SETOR", "DOM_CNEFE"]]


# =============================================================================
# DECISAO 1 — ALOCACAO DOS SETORES A BAIRROS
# =============================================================================


def alocar_por_proximidade(
    setores_alvo: gpd.GeoDataFrame, bairros_ref: gpd.GeoDataFrame, regra: str
) -> pd.DataFrame:
    """Atribui cada setor inteiro ao bairro mais proximo do seu ponto interno.

    A distancia e medida ao POLIGONO do bairro, nao a sua fronteira: um ponto que
    caia dentro de um bairro tem distancia zero e e atribuido a ele, o que a
    medicao ate a fronteira nao garantiria.
    """
    pontos = setores_alvo.copy()
    pontos["geometry"] = pontos.geometry.representative_point()  # verificacao-centroid

    vizinho = gpd.sjoin_nearest(
        pontos[["CD_SETOR", "geometry"]],
        bairros_ref[["CD_BAIRRO", "geometry"]],
        how="left",
        distance_col="DIST_BAIRRO_M",
    )
    # sjoin_nearest pode devolver empates; fica o primeiro, de forma determinista.
    vizinho = vizinho.sort_values(["CD_SETOR", "DIST_BAIRRO_M"]).drop_duplicates("CD_SETOR")

    alocacao = vizinho[["CD_SETOR", "CD_BAIRRO"]].copy()
    alocacao["FRACAO"] = 1.0
    alocacao["REGRA"] = regra
    return alocacao.merge(
        setores_alvo[["CD_SETOR", "geometry"]], on="CD_SETOR", how="left"
    )


def alocar_por_rateio_de_domicilios(
    setores_alvo: gpd.GeoDataFrame,
    bairros_ref: gpd.GeoDataFrame,
    pontos_cnefe: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Divide a populacao do setor conforme o bairro mais proximo de cada domicilio.

    Alternativa de cobertura parcial que nao depende de sobreposicao geometrica:
    cada endereco de especie 1 dentro do setor orfao e atribuido ao bairro mais
    proximo, e a populacao do setor e rateada pela contagem resultante. Um setor
    de borda que se estenda entre dois bairros e dividido conforme onde os
    domicilios efetivamente estao.

    A GEOMETRIA do setor vai integralmente ao bairro dominante, enquanto a
    populacao e repartida. A assimetria e deliberada: garante que a uniao dos
    poligonos finais cubra o municipio exatamente, sem vazios nem sobreposicoes.
    Setores sem nenhum domicilio caem na regra de proximidade.
    """
    dentro = gpd.sjoin(
        pontos_cnefe[["geometry"]],
        setores_alvo[["CD_SETOR", "geometry"]],
        how="inner",
        predicate="within",
    )

    resultados: list[pd.DataFrame] = []
    sem_domicilio = setores_alvo[~setores_alvo["CD_SETOR"].isin(set(dentro["CD_SETOR"]))]
    if len(sem_domicilio) > 0:
        resultados.append(
            alocar_por_proximidade(sem_domicilio, bairros_ref, "proximidade_sem_domicilio")
        )

    if len(dentro) > 0:
        vizinho = gpd.sjoin_nearest(
            dentro[["CD_SETOR", "geometry"]],
            bairros_ref[["CD_BAIRRO", "geometry"]],
            how="left",
            distance_col="DIST_M",
        )
        contagem = (
            vizinho.groupby(["CD_SETOR", "CD_BAIRRO"]).size().rename("DOM").reset_index()
        )
        contagem["FRACAO"] = contagem["DOM"] / contagem.groupby("CD_SETOR")["DOM"].transform("sum")
        contagem["REGRA"] = "rateio_por_domicilios"

        # So a linha dominante de cada setor carrega a geometria.
        dominante = contagem.sort_values("FRACAO", ascending=False).drop_duplicates("CD_SETOR")
        contagem = contagem.merge(
            dominante[["CD_SETOR", "CD_BAIRRO"]].assign(E_DOMINANTE=True),
            on=["CD_SETOR", "CD_BAIRRO"],
            how="left",
        )
        # astype(bool) e obrigatorio: apos o fillna a coluna fica com dtype object,
        # e o operador ~ sobre object devolve inteiros (-1/-2), nao booleanos.
        contagem["E_DOMINANTE"] = contagem["E_DOMINANTE"].fillna(False).astype(bool)
        contagem = contagem.merge(
            setores_alvo[["CD_SETOR", "geometry"]], on="CD_SETOR", how="left"
        )
        contagem.loc[~contagem["E_DOMINANTE"], "geometry"] = None
        resultados.append(contagem[["CD_SETOR", "CD_BAIRRO", "FRACAO", "REGRA", "geometry"]])

    return pd.concat(resultados, ignore_index=True)


def alocar_por_interseccao(
    setores_alvo: gpd.GeoDataFrame,
    bairros_ref: gpd.GeoDataFrame,
    pontos_cnefe: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Recorta cada setor contra a malha de bairros e rateia a populacao.

    O peso de cada parte e o numero de domicilios do CNEFE que caem dentro dela
    (PESO_DO_RATEIO == 'domicilios') ou a sua area. Setores que nao intersectam
    bairro nenhum — situacao comum, porque a malha oficial cobre apenas o
    perimetro urbano — caem obrigatoriamente na regra de proximidade.
    """
    partes = gpd.overlay(
        setores_alvo[["CD_SETOR", "geometry"]],
        bairros_ref[["CD_BAIRRO", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    partes = partes[partes.geometry.area > 0].copy()

    com_interseccao = set(partes["CD_SETOR"])
    sem_interseccao = setores_alvo[~setores_alvo["CD_SETOR"].isin(com_interseccao)]

    if len(com_interseccao) == 0:
        print()
        print("   AVISO: a interseccao com a malha de bairros e VAZIA.")
        print("   A malha de bairros do IBGE e a uniao dos setores que ja trazem CD_BAIRRO,")
        print("   entao os setores orfaos nao a tocam. A regra 'interseccao' degenera")
        print("   integralmente na regra de proximidade. Ver DECISAO 1 no topo do arquivo.")
        print()
    elif len(sem_interseccao) > 0:
        print(f"   setores sem interseccao com a malha ... {len(sem_interseccao)} (vao por proximidade)")

    resultados: list[pd.DataFrame] = []

    if len(sem_interseccao) > 0:
        resultados.append(
            alocar_por_proximidade(
                sem_interseccao, bairros_ref, "proximidade_sem_interseccao"
            )
        )

    if len(partes) > 0:
        partes["AREA_PARTE"] = partes.geometry.area
        contagem = gpd.sjoin(
            pontos_cnefe[["geometry"]],
            partes.reset_index(names="ID_PARTE")[["ID_PARTE", "geometry"]],
            how="inner",
            predicate="within",
        )
        partes["DOM_PARTE"] = (
            contagem.groupby("ID_PARTE").size().reindex(partes.index).fillna(0).astype("int64")
        )

        # A parte do setor que fica fora de toda a malha de bairros e reincorporada
        # ao bairro dominante, para que a uniao dos poligonos finais continue
        # cobrindo o municipio inteiro.
        sobras = gpd.overlay(
            setores_alvo[["CD_SETOR", "geometry"]],
            bairros_ref[["geometry"]].dissolve(),
            how="difference",
            keep_geom_type=True,
        )
        sobras = sobras[sobras.geometry.area > 0]

        blocos = []
        for cd_setor, grupo in partes.groupby("CD_SETOR"):
            grupo = grupo.copy()

            if PESO_DO_RATEIO == "domicilios" and grupo["DOM_PARTE"].sum() > 0:
                peso = grupo["DOM_PARTE"].astype(float)
                regra = "interseccao_domicilios"
            elif PESO_DO_RATEIO == "domicilios" and FALLBACK_SEM_DOMICILIO == "bairro_mais_proximo":
                unico = setores_alvo[setores_alvo["CD_SETOR"] == cd_setor]
                blocos.append(
                    alocar_por_proximidade(unico, bairros_ref, "proximidade_sem_domicilio")
                )
                continue
            else:
                peso = grupo["AREA_PARTE"].astype(float)
                regra = (
                    "interseccao_area"
                    if PESO_DO_RATEIO == "area"
                    else "interseccao_area_fallback"
                )

            grupo["FRACAO"] = peso / peso.sum()
            grupo["REGRA"] = regra

            sobra = sobras.loc[sobras["CD_SETOR"] == cd_setor, "geometry"]
            if len(sobra) > 0:
                dominante = grupo["FRACAO"].idxmax()
                grupo.loc[dominante, "geometry"] = grupo.loc[dominante, "geometry"].union(
                    sobra.union_all()
                )

            blocos.append(grupo[["CD_SETOR", "CD_BAIRRO", "FRACAO", "REGRA", "geometry"]])

        if blocos:
            resultados.append(pd.concat(blocos, ignore_index=True))

    return pd.concat(resultados, ignore_index=True)


def alocar_setores(
    setores: gpd.GeoDataFrame,
    basico: pd.DataFrame,
    bairros_ref: gpd.GeoDataFrame,
    pontos_cnefe: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Monta a tabela longa setor x bairro com a fracao de populacao de cada par."""
    subtitulo("7. Alocacao dos setores aos bairros")
    config = MALHAS[MALHA_BAIRROS]

    geo = setores.merge(basico, on="CD_SETOR", how="left", validate="one_to_one")
    checar(geo["POP"].notna().all(), "Setor da malha sem linha no arquivo basico.")

    if config["usar_atributo_do_setor"]:
        diretos = geo.loc[~geo["SEM_BAIRRO_IBGE"]].copy()
        a_alocar = geo.loc[geo["SEM_BAIRRO_IBGE"]].copy()
        base = pd.DataFrame(
            {
                "CD_SETOR": diretos["CD_SETOR"],
                "CD_BAIRRO": diretos["CD_BAIRRO"],
                "FRACAO": 1.0,
                "REGRA": "cd_bairro_do_ibge",
            }
        ).merge(diretos[["CD_SETOR", "geometry"]], on="CD_SETOR", how="left")
        print(f"setores com CD_BAIRRO do IBGE .......... {len(diretos)}")
    else:
        base = pd.DataFrame(columns=["CD_SETOR", "CD_BAIRRO", "FRACAO", "REGRA", "geometry"])
        a_alocar = geo.copy()
        print("a malha selecionada nao usa o CD_BAIRRO do setor: alocacao 100% espacial")

    print(f"setores a alocar espacialmente ......... {len(a_alocar)}")
    print(f"populacao envolvida .................... {numero_br(a_alocar['POP'].sum())}")
    print(f"regra aplicada ......................... {REGRA_SETORES_ORFAOS}", end="")
    if REGRA_SETORES_ORFAOS == "interseccao":
        print(f" (peso: {PESO_DO_RATEIO}, fallback: {FALLBACK_SEM_DOMICILIO})")
    else:
        print()

    if len(a_alocar) > 0:
        if REGRA_SETORES_ORFAOS == "interseccao":
            alocados = alocar_por_interseccao(
                gpd.GeoDataFrame(a_alocar, geometry="geometry", crs=setores.crs),
                bairros_ref,
                pontos_cnefe,
            )
        elif REGRA_SETORES_ORFAOS == "rateio_por_domicilios":
            alocados = alocar_por_rateio_de_domicilios(
                gpd.GeoDataFrame(a_alocar, geometry="geometry", crs=setores.crs),
                bairros_ref,
                pontos_cnefe,
            )
        elif REGRA_SETORES_ORFAOS == "bairro_mais_proximo":
            alocados = alocar_por_proximidade(
                gpd.GeoDataFrame(a_alocar, geometry="geometry", crs=setores.crs),
                bairros_ref,
                "bairro_mais_proximo",
            )
        else:
            raise FalhaDeSanidade(f"REGRA_SETORES_ORFAOS invalida: {REGRA_SETORES_ORFAOS}")
    else:
        alocados = pd.DataFrame(columns=base.columns)

    alocacao = gpd.GeoDataFrame(
        pd.concat([base, alocados], ignore_index=True), geometry="geometry", crs=setores.crs
    )

    # Verificacao: cada setor tem fracoes somando exatamente 1.
    soma = alocacao.groupby("CD_SETOR")["FRACAO"].sum()
    checar(
        len(soma) == len(setores),
        f"A alocacao cobre {len(soma)} setores, mas a malha tem {len(setores)}.",
    )
    checar(
        bool(np.allclose(soma.to_numpy(), 1.0)),
        "Ha setor cujas fracoes de rateio nao somam 1 — populacao seria perdida ou duplicada.",
    )

    print("\nsetores alocados por regra:")
    resumo = (
        alocacao.drop_duplicates("CD_SETOR")
        .merge(basico[["CD_SETOR", "POP"]], on="CD_SETOR")
        .groupby("REGRA")
        .agg(setores=("CD_SETOR", "nunique"), populacao=("POP", "sum"))
        .sort_values("setores", ascending=False)
    )
    for regra, linha in resumo.iterrows():
        print(f"   {regra:<32} {linha['setores']:>5} setores   {numero_br(linha['populacao']):>9} hab")

    partes_por_setor = alocacao.groupby("CD_SETOR").size()
    divididos = int((partes_por_setor > 1).sum())
    print(f"setores divididos entre bairros ........ {divididos}")
    return alocacao


# =============================================================================
# DECISAO 2 — IMPUTACAO DA RENDA
# =============================================================================


def imputar_renda(
    alocacao: gpd.GeoDataFrame, basico: pd.DataFrame, renda: pd.DataFrame
) -> pd.DataFrame:
    """Resolve renda ausente por setor, separando sigilo de ausencia de linha.

    Devolve, por setor: renda usada, peso na media ponderada e a origem do valor.
    O peso e V06001 (domicilios com responsavel); quando a regra e 'excluir', o
    peso vai a zero e o setor sai do numerador e do denominador.
    """
    subtitulo("8. Renda do responsavel — imputacao")

    # Bairro dominante do setor: define de qual mediana o setor se serve.
    dominante = (
        alocacao.sort_values("FRACAO", ascending=False)
        .drop_duplicates("CD_SETOR")[["CD_SETOR", "CD_BAIRRO"]]
        .rename(columns={"CD_BAIRRO": "CD_BAIRRO_DOMINANTE"})
    )

    tabela = (
        basico[["CD_SETOR", "POP", "DOM_OCUPADOS"]]
        .merge(renda, on="CD_SETOR", how="left")
        .merge(dominante, on="CD_SETOR", how="left")
    )
    tabela["RENDA_SUPRIMIDA"] = tabela["RENDA_SUPRIMIDA"].fillna(False).astype(bool)
    tabela["SEM_LINHA_RENDA"] = tabela["DOM_RESPONSAVEL"].isna() & ~tabela["RENDA_SUPRIMIDA"]

    mediana_municipio = float(tabela["RENDA_MEDIA"].median())
    mediana_por_bairro = (
        tabela.groupby("CD_BAIRRO_DOMINANTE")["RENDA_MEDIA"].median().rename("MEDIANA_BAIRRO")
    )
    tabela = tabela.merge(
        mediana_por_bairro, left_on="CD_BAIRRO_DOMINANTE", right_index=True, how="left"
    )

    def aplicar(regra: str, mascara: pd.Series, rotulo: str) -> int:
        """Aplica uma das regras de imputacao a um subconjunto de setores."""
        n = int(mascara.sum())
        if n == 0:
            return 0
        if regra == "mediana_bairro":
            valor = tabela.loc[mascara, "MEDIANA_BAIRRO"].fillna(mediana_municipio)
            tabela.loc[mascara, "RENDA_USADA"] = valor
            tabela.loc[mascara, "PESO_RENDA"] = tabela.loc[mascara, "DOM_RESPONSAVEL"].fillna(
                tabela.loc[mascara, "DOM_OCUPADOS"]
            )
            tabela.loc[mascara, "ORIGEM_RENDA"] = f"imputada_mediana_bairro_{rotulo}"
        elif regra == "mediana_municipio":
            tabela.loc[mascara, "RENDA_USADA"] = mediana_municipio
            tabela.loc[mascara, "PESO_RENDA"] = tabela.loc[mascara, "DOM_RESPONSAVEL"].fillna(
                tabela.loc[mascara, "DOM_OCUPADOS"]
            )
            tabela.loc[mascara, "ORIGEM_RENDA"] = f"imputada_mediana_municipio_{rotulo}"
        elif regra == "excluir":
            tabela.loc[mascara, "RENDA_USADA"] = np.nan
            tabela.loc[mascara, "PESO_RENDA"] = 0.0
            tabela.loc[mascara, "ORIGEM_RENDA"] = f"excluida_{rotulo}"
        else:
            raise FalhaDeSanidade(f"Regra de renda invalida: {regra}")
        return n

    tabela["RENDA_USADA"] = tabela["RENDA_MEDIA"]
    tabela["PESO_RENDA"] = tabela["DOM_RESPONSAVEL"]
    tabela["ORIGEM_RENDA"] = "observada"

    n_suprimida = aplicar(REGRA_RENDA_SUPRIMIDA, tabela["RENDA_SUPRIMIDA"], "sigilo")
    n_sem_linha = aplicar(REGRA_RENDA_SEM_LINHA, tabela["SEM_LINHA_RENDA"], "sem_linha")

    tabela["PESO_RENDA"] = tabela["PESO_RENDA"].fillna(0.0)

    print(f"regra para renda suprimida ('X') ....... {REGRA_RENDA_SUPRIMIDA}  ({n_suprimida} setores)")
    print(f"regra para setor sem linha ............. {REGRA_RENDA_SEM_LINHA}  ({n_sem_linha} setores)")
    print(f"mediana municipal de referencia ........ R$ {numero_br(mediana_municipio, 2)}")
    print(f"setores com renda observada ............ {(tabela['ORIGEM_RENDA'] == 'observada').sum()}")
    print(f"populacao em setores com renda imputada  {numero_br(tabela.loc[tabela['ORIGEM_RENDA'].str.startswith('imputada'), 'POP'].sum())}")
    print(f"populacao em setores excluidos da renda  {numero_br(tabela.loc[tabela['ORIGEM_RENDA'].str.startswith('excluida'), 'POP'].sum())}")

    checar(
        not bool((tabela["RENDA_USADA"] == 0).any()),
        "Ha setor com renda igual a zero — sentinela 'X' provavelmente lido como numero.",
    )
    return tabela[["CD_SETOR", "RENDA_USADA", "PESO_RENDA", "ORIGEM_RENDA", "RENDA_SUPRIMIDA", "SEM_LINHA_RENDA"]]


# =============================================================================
# AGREGACAO POR BAIRRO
# =============================================================================


def classificar_regiao(nm_dist: str, nm_bairro: str) -> str:
    """Regiao funcional do bairro (Decreto no 29.142/2026 + distritos do IBGE)."""
    if nm_dist == DISTRITO_SEDE:
        return REGIAO_CONTINENTE if nm_bairro in BAIRROS_CONTINENTE else REGIAO_SEDE
    regiao = DISTRITO_REGIAO.get(nm_dist)
    checar(regiao is not None, f"Distrito sem regiao funcional mapeada: '{nm_dist}'.")
    return regiao


def agregar_por_bairro(
    alocacao: gpd.GeoDataFrame,
    basico: pd.DataFrame,
    renda_setor: pd.DataFrame,
    bairros_ref: gpd.GeoDataFrame,
    area_municipio_km2: float,
) -> gpd.GeoDataFrame:
    """Agrega populacao, domicilios e renda por bairro e monta os poligonos.

    Os poligonos vem da dissolucao das geometrias alocadas, nao da malha oficial
    de bairros: a malha do IBGE cobre so o perimetro urbano e deixaria de fora
    tres quartos do municipio.
    """
    subtitulo("9. Agregacao por bairro")

    dados = (
        alocacao.merge(basico[["CD_SETOR", "POP", "DOM_OCUPADOS", "NM_DIST"]], on="CD_SETOR")
        .merge(renda_setor, on="CD_SETOR")
    )
    dados["POP_PARTE"] = dados["POP"] * dados["FRACAO"]
    dados["DOM_PARTE"] = dados["DOM_OCUPADOS"] * dados["FRACAO"]
    dados["PESO_PARTE"] = dados["PESO_RENDA"] * dados["FRACAO"]
    dados["RENDA_X_PESO"] = dados["RENDA_USADA"].fillna(0.0) * dados["PESO_PARTE"]

    checar(
        abs(dados["POP_PARTE"].sum() - POP_REFERENCIA) < 1e-6,
        f"O rateio perdeu populacao: {dados['POP_PARTE'].sum():.4f} contra {POP_REFERENCIA}.",
    )

    agregado = dados.groupby("CD_BAIRRO", as_index=False).agg(
        POP_FRAC=("POP_PARTE", "sum"),
        DOM_OCUPADOS=("DOM_PARTE", "sum"),
        PESO_RENDA=("PESO_PARTE", "sum"),
        RENDA_X_PESO=("RENDA_X_PESO", "sum"),
        N_SETORES=("CD_SETOR", "nunique"),
    )
    agregado["RENDA_MEDIA"] = np.where(
        agregado["PESO_RENDA"] > 0, agregado["RENDA_X_PESO"] / agregado["PESO_RENDA"], np.nan
    )
    agregado["POP"] = arredondar_preservando_total(agregado["POP_FRAC"], POP_REFERENCIA)

    # Poligonos: dissolucao das geometrias alocadas. Sob a regra de rateio, apenas
    # a linha do bairro dominante carrega geometria; as demais entram so com
    # populacao e sao descartadas aqui.
    com_geometria = dados[dados["geometry"].notna() & ~dados["geometry"].is_empty]
    geometrias = (
        gpd.GeoDataFrame(
            com_geometria[["CD_BAIRRO", "geometry"]], geometry="geometry", crs=alocacao.crs
        ).dissolve(by="CD_BAIRRO", as_index=False)
    )
    checar(
        len(geometrias) == dados["CD_BAIRRO"].nunique(),
        f"{dados['CD_BAIRRO'].nunique() - len(geometrias)} bairro(s) receberam populacao "
        "sem receber geometria — revisar a regra de alocacao.",
    )
    area_dissolvida = geometrias.geometry.area.sum() / 1e6
    checar(
        abs(area_dissolvida - area_municipio_km2) < 0.5,
        f"A uniao dos bairros cobre {area_dissolvida:.1f} km2, contra "
        f"{area_municipio_km2:.1f} km2 da malha de setores. Territorio perdido na alocacao.",
    )

    # Nome e distrito de cada bairro: da malha de referencia, com o distrito
    # majoritario dos setores como complemento.
    distrito_do_bairro = (
        dados.groupby(["CD_BAIRRO", "NM_DIST"])["POP_PARTE"].sum().reset_index()
        .sort_values("POP_PARTE", ascending=False)
        .drop_duplicates("CD_BAIRRO")[["CD_BAIRRO", "NM_DIST"]]
    )

    bairros = (
        geometrias.merge(agregado, on="CD_BAIRRO")
        .merge(bairros_ref[["CD_BAIRRO", "NM_BAIRRO"]], on="CD_BAIRRO", how="left")
        .merge(distrito_do_bairro, on="CD_BAIRRO", how="left")
    )
    checar(
        bairros["NM_BAIRRO"].notna().all(),
        "Bairro sem nome apos a juncao com a malha de referencia.",
    )

    bairros["REGIAO_FUNCIONAL"] = [
        classificar_regiao(d, b) for d, b in zip(bairros["NM_DIST"], bairros["NM_BAIRRO"])
    ]
    bairros["AREA_KM2"] = bairros.geometry.area / 1e6
    bairros["DENS_HAB_KM2"] = bairros["POP"] / bairros["AREA_KM2"]

    n_esperado = MALHAS[MALHA_BAIRROS]["n_esperado"]
    checar(
        len(bairros) == n_esperado,
        f"Foram agregados {len(bairros)} bairros, esperado {n_esperado}.",
    )
    checar(
        int(bairros["POP"].sum()) == POP_REFERENCIA,
        f"Populacao agregada e {int(bairros['POP'].sum())}, esperado {POP_REFERENCIA}.",
    )

    print(f"bairros agregados ...................... {len(bairros)}")
    print(f"populacao total ........................ {numero_br(bairros['POP'].sum())}  (confere com {numero_br(POP_REFERENCIA)})")
    print(f"area total ............................. {numero_br(bairros['AREA_KM2'].sum(), 1)} km2")
    print(f"bairros sem renda calculavel ........... {int(bairros['RENDA_MEDIA'].isna().sum())}")

    colunas = [
        "CD_BAIRRO", "NM_BAIRRO", "NM_DIST", "REGIAO_FUNCIONAL",
        "POP", "POP_FRAC", "DOM_OCUPADOS", "RENDA_MEDIA", "PESO_RENDA",
        "N_SETORES", "AREA_KM2", "DENS_HAB_KM2", "geometry",
    ]
    return gpd.GeoDataFrame(bairros[colunas], geometry="geometry", crs=alocacao.crs)


def parte_com_mais_domicilios(geometria, domicilios: gpd.GeoSeries):
    """Devolve a parte do poligono onde estao mais domicilios do CNEFE.

    Bairros recortados pela orla saem como MultiPolygon, e 15 dos 87 tem mais de
    uma parte. O representative_point() de um MultiPolygon pode cair numa ilhota
    ou num apendice desabitado — verificado: em 5 bairros o ponto caia fora da
    parte habitada, com deslocamento de ate 12,7 km em Lagoinha do Norte, cuja
    parte escolhida nao tinha NENHUM domicilio.

    O criterio e o mesmo ja adotado na Decisao 1: onde a populacao efetivamente
    esta, medida pelos enderecos do CNEFE. Sem domicilio em nenhuma parte, vale a
    de maior area.
    """
    if geometria.geom_type != "MultiPolygon":
        return geometria, 1, 0
    partes = list(geometria.geoms)
    # predicate="intersects": para ponto e poligono equivale a conter, e nao tem a
    # ambiguidade de direcao que "within" tem em sindex.query.
    contagens = [len(domicilios.sindex.query(parte, predicate="intersects")) for parte in partes]
    if max(contagens) == 0:
        escolhida = max(range(len(partes)), key=lambda i: partes[i].area)
    else:
        escolhida = int(np.argmax(contagens))
    return partes[escolhida], len(partes), contagens[escolhida]


def gerar_pontos_demanda(
    bairros: gpd.GeoDataFrame, massas_dagua: gpd.GeoSeries, pontos_cnefe: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Ponto representativo interno de cada bairro.

    representative_point() garante ponto DENTRO do poligono. O centroide
    geometrico cai na agua em bairros recortados pela orla.

    Em bairros multiparte, o ponto e tomado sobre a PARTE HABITADA, nao sobre o
    MultiPolygon inteiro.
    """
    subtitulo("10. Pontos de demanda")

    domicilios = pontos_cnefe.geometry
    geometrias, n_partes, n_domicilios, realocados = [], [], [], []
    for linha in bairros.itertuples():
        parte, partes, domic = parte_com_mais_domicilios(linha.geometry, domicilios)
        ingenuo = linha.geometry.representative_point()  # verificacao-centroid
        escolhido = parte.representative_point()         # verificacao-centroid
        if partes > 1 and not parte.contains(ingenuo):
            realocados.append((linha.NM_BAIRRO, linha.POP, ingenuo.distance(escolhido) / 1000))
        geometrias.append(escolhido)
        n_partes.append(partes)
        n_domicilios.append(domic)

    pontos = bairros.copy()
    pontos["geometry"] = gpd.GeoSeries(geometrias, index=bairros.index, crs=bairros.crs)
    pontos["N_PARTES"] = n_partes
    pontos["DOM_NA_PARTE"] = n_domicilios

    print(f"bairros multiparte ..................... {int((pontos['N_PARTES'] > 1).sum())} de {len(pontos)}")
    print(f"pontos realocados para a parte habitada  {len(realocados)}")
    for nome, pop, km in sorted(realocados, key=lambda t: -t[2]):
        print(f"   {nome[:28]:<28} {numero_br(pop):>8} hab   deslocamento {km:>6.2f} km")

    dentro = pontos.geometry.within(bairros.geometry)
    checar(
        bool(dentro.all()),
        "Ponto de demanda fora do proprio bairro: "
        f"{bairros.loc[~dentro, 'NM_BAIRRO'].tolist()}",
    )
    checar(
        pontos.crs.to_epsg() == CRS_TRABALHO,
        f"Pontos de demanda fora do EPSG:{CRS_TRABALHO}.",
    )
    # Nenhum ponto pode cair em baia, lagoa ou mar: seria origem impossivel para
    # a matriz de distancias da Etapa 6.
    if len(massas_dagua) > 0:
        na_agua = pontos.geometry.within(massas_dagua.union_all())
        checar(
            not bool(na_agua.any()),
            "Ponto de demanda sobre massa d'agua: "
            f"{pontos.loc[na_agua, 'NM_BAIRRO'].tolist()}. "
            "Conferir EXCLUIR_MASSAS_DAGUA no topo do script.",
        )
        print(f"nenhum ponto sobre massa d'agua ........ conferido")
    sem_domicilio = pontos[(pontos["N_PARTES"] > 1) & (pontos["DOM_NA_PARTE"] == 0)]
    if len(sem_domicilio) > 0:
        print(f"AVISO: {len(sem_domicilio)} bairro(s) multiparte sem domicilio em parte alguma; "
              "usada a de maior area: " + ", ".join(sem_domicilio["NM_BAIRRO"]))
    print(f"pontos gerados ......................... {len(pontos)}")
    print(f"todos internos ao proprio bairro ....... sim")
    print(f"CRS .................................... EPSG:{pontos.crs.to_epsg()}")
    return pontos


# =============================================================================
# DEMANDA POR SETOR CENSITARIO
# =============================================================================


def gerar_demanda_setor(
    setores: gpd.GeoDataFrame,
    basico: pd.DataFrame,
    alocacao: gpd.GeoDataFrame,
    renda_setor: pd.DataFrame,
    bairros: gpd.GeoDataFrame,
    massas_dagua: gpd.GeoSeries,
    pontos_cnefe: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Poligonos de setor e um ponto de demanda por setor habitado.

    Motivo (auditoria de setembro/2026): o ponto representativo do BAIRRO pode
    ficar longe de onde a populacao mora. Em Lagoa ele caia a 4,5 km da mediana
    dos domicilios, e em 35 bairros menos da metade dos domicilios estava a 800 m
    do ponto. Como a restricao (7) usa um ponto unico por unidade de demanda, isso
    falseia a cobertura nos dois sentidos. O setor censitario e a menor unidade
    com populacao e renda publicadas.

    A regra do ponto e a MESMA dos bairros: representative_point() sobre a parte
    com mais domicilios do CNEFE. So muda a unidade.

    Setores com populacao zero ficam na camada de poligonos, para os mapas, mas
    nao geram ponto de demanda: h_i seria zero e so aumentariam a instancia.
    """
    subtitulo("10b. Demanda por setor censitario")

    # Bairro dominante do setor (maior fracao de populacao), que da nome e regiao.
    # Os setores divididos entre bairros continuam rateados pela tabela de
    # alocacao na hora de agregar cobertura por bairro; o dominante so rotula.
    dominante = (
        alocacao.drop(columns="geometry")
        .sort_values("FRACAO", ascending=False)
        .drop_duplicates("CD_SETOR")[["CD_SETOR", "CD_BAIRRO"]]
    )
    n_bairros = alocacao.groupby("CD_SETOR").size().rename("N_BAIRROS")

    poligonos = (
        setores[["CD_SETOR", "geometry"]]
        .merge(basico[["CD_SETOR", "POP", "DOM_OCUPADOS"]], on="CD_SETOR", how="left", validate="one_to_one")
        .merge(renda_setor[["CD_SETOR", "RENDA_USADA", "PESO_RENDA", "ORIGEM_RENDA"]],
               on="CD_SETOR", how="left", validate="one_to_one")
        .merge(dominante, on="CD_SETOR", how="left", validate="one_to_one")
        .merge(n_bairros, left_on="CD_SETOR", right_index=True, how="left")
        .merge(bairros[["CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL"]], on="CD_BAIRRO", how="left")
        .rename(columns={"RENDA_USADA": "RENDA_MEDIA"})
    )
    poligonos = gpd.GeoDataFrame(poligonos, geometry="geometry", crs=setores.crs)
    poligonos["AREA_KM2"] = poligonos.geometry.area / 1e6

    checar(
        len(poligonos) == len(setores),
        f"A camada de setores tem {len(poligonos)} linhas, mas ha {len(setores)} setores de terra.",
    )
    checar(poligonos["CD_BAIRRO"].notna().all(), "Setor de terra sem bairro dominante.")
    checar(poligonos["REGIAO_FUNCIONAL"].notna().all(), "Setor sem regiao funcional.")
    checar(
        int(poligonos["POP"].sum()) == POP_REFERENCIA,
        f"Os setores de terra somam {int(poligonos['POP'].sum())} habitantes, esperado {POP_REFERENCIA}.",
    )

    habitados = poligonos[poligonos["POP"] > 0].copy()
    checar(
        habitados["RENDA_MEDIA"].notna().all() and bool((habitados["RENDA_MEDIA"] > 0).all()),
        "Ha setor habitado sem renda positiva — tau_i nao poderia ser calculado: "
        f"{habitados.loc[habitados['RENDA_MEDIA'].isna(), 'CD_SETOR'].tolist()[:10]}",
    )

    domicilios = pontos_cnefe.geometry
    geometrias, n_partes, n_domicilios = [], [], []
    for linha in habitados.itertuples():
        parte, partes, domic = parte_com_mais_domicilios(linha.geometry, domicilios)
        geometrias.append(parte.representative_point())  # verificacao-centroid
        n_partes.append(partes)
        n_domicilios.append(domic)

    pontos = habitados.copy()
    pontos["geometry"] = gpd.GeoSeries(geometrias, index=habitados.index, crs=setores.crs)
    pontos["N_PARTES"] = n_partes
    pontos["DOM_NA_PARTE"] = n_domicilios
    pontos = pontos.reset_index(drop=True)

    dentro = pontos.geometry.within(habitados.geometry.reset_index(drop=True))
    checar(bool(dentro.all()), f"{int((~dentro).sum())} ponto(s) de setor fora do proprio setor.")
    if len(massas_dagua) > 0:
        na_agua = pontos.geometry.within(massas_dagua.union_all())
        checar(not bool(na_agua.any()), f"{int(na_agua.sum())} ponto(s) de setor sobre massa d'agua.")

    print(f"setores de terra ....................... {len(poligonos)}")
    print(f"   com populacao (pontos de demanda) ... {len(pontos)}")
    print(f"   com populacao zero (so poligono) .... {len(poligonos) - len(pontos)}")
    print(f"populacao nos pontos ................... {numero_br(pontos['POP'].sum())}")
    print(f"setores multiparte habitados ........... {int((pontos['N_PARTES'] > 1).sum())}")
    print(f"setores divididos entre bairros ........ {int((pontos['N_BAIRROS'] > 1).sum())}")
    print(f"area do setor habitado ................. mediana {pontos['AREA_KM2'].median():.3f} km2 · "
          f"max {pontos['AREA_KM2'].max():.1f} km2")
    print(f"renda imputada (sigilo) ................ {int(pontos['ORIGEM_RENDA'].str.startswith('imputada').sum())} setores")
    print(f"nenhum ponto sobre massa d'agua ........ conferido")
    return poligonos, pontos


def representatividade_dos_pontos(
    bairros: gpd.GeoDataFrame,
    pontos_bairro: gpd.GeoDataFrame,
    setores: gpd.GeoDataFrame,
    pontos_setor: gpd.GeoDataFrame,
    pontos_cnefe: gpd.GeoDataFrame,
    raio_m: float = 800.0,
) -> pd.DataFrame:
    """Quanto de cada bairro mora perto do ponto que o representa, nas duas unidades.

    Para cada domicilio do CNEFE mede a distancia em LINHA RETA ate o ponto da
    unidade que o contem — o ponto do bairro, numa leitura, e o ponto do setor, na
    outra. Linha reta e limite inferior da distancia de rede, entao a fracao abaixo
    e OTIMISTA: se ja e baixa em linha reta, na rede e menor.
    """
    subtitulo("10c. Representatividade dos pontos de demanda")

    dom = pontos_cnefe[["geometry"]].copy()
    dom = gpd.sjoin(dom, bairros[["CD_BAIRRO", "geometry"]], how="inner", predicate="within").drop(columns="index_right")
    dom = gpd.sjoin(dom, setores[["CD_SETOR", "geometry"]], how="inner", predicate="within").drop(columns="index_right")

    pb = pontos_bairro.set_index("CD_BAIRRO").geometry
    ps = pontos_setor.set_index("CD_SETOR").geometry
    x, y = dom.geometry.x.to_numpy(), dom.geometry.y.to_numpy()
    dom["DIST_PONTO_BAIRRO_M"] = np.hypot(x - dom["CD_BAIRRO"].map(pb.x).to_numpy(), y - dom["CD_BAIRRO"].map(pb.y).to_numpy())
    # Domicilio em setor sem populacao no Censo nao tem ponto de setor: fica fora
    # da comparacao (sao poucos e nao pesam na demanda).
    tem_ponto = dom["CD_SETOR"].isin(ps.index)
    dom["DIST_PONTO_SETOR_M"] = np.nan
    dom.loc[tem_ponto, "DIST_PONTO_SETOR_M"] = np.hypot(
        x[tem_ponto.to_numpy()] - dom.loc[tem_ponto, "CD_SETOR"].map(ps.x).to_numpy(),
        y[tem_ponto.to_numpy()] - dom.loc[tem_ponto, "CD_SETOR"].map(ps.y).to_numpy(),
    )

    tabela = (
        dom.groupby("CD_BAIRRO")
        .agg(
            DOM_CNEFE=("DIST_PONTO_BAIRRO_M", "size"),
            PCT_DOM_ATE_RAIO_PONTO_BAIRRO=("DIST_PONTO_BAIRRO_M", lambda s: 100 * (s <= raio_m).mean()),
            PCT_DOM_ATE_RAIO_PONTO_SETOR=("DIST_PONTO_SETOR_M", lambda s: 100 * (s.dropna() <= raio_m).mean()),
            DIST_MEDIANA_PONTO_BAIRRO_M=("DIST_PONTO_BAIRRO_M", "median"),
            DIST_MEDIANA_PONTO_SETOR_M=("DIST_PONTO_SETOR_M", "median"),
        )
        .reset_index()
        .merge(bairros[["CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL", "POP"]], on="CD_BAIRRO")
        .sort_values("PCT_DOM_ATE_RAIO_PONTO_BAIRRO")
    )

    abaixo_b = tabela[tabela["PCT_DOM_ATE_RAIO_PONTO_BAIRRO"] < 50]
    abaixo_s = tabela[tabela["PCT_DOM_ATE_RAIO_PONTO_SETOR"] < 50]
    print(f"domicilios comparados .................. {numero_br(tem_ponto.sum())} de {numero_br(len(dom))}")
    print(f"a ate {raio_m:.0f} m (linha reta) do ponto:")
    print(f"   ponto do bairro ..................... {100 * (dom['DIST_PONTO_BAIRRO_M'] <= raio_m).mean():.1f}% dos domicilios")
    print(f"   ponto do setor ...................... {100 * (dom.loc[tem_ponto, 'DIST_PONTO_SETOR_M'] <= raio_m).mean():.1f}% dos domicilios")
    print(f"bairros com menos da metade dos domicilios a {raio_m:.0f} m:")
    print(f"   ponto do bairro ..................... {len(abaixo_b)} ({numero_br(abaixo_b['POP'].sum())} hab)")
    print(f"   ponto do setor ...................... {len(abaixo_s)} ({numero_br(abaixo_s['POP'].sum())} hab)")
    print("piores casos sob o ponto do bairro:")
    for linha in tabela.head(8).itertuples():
        print(f"   {linha.NM_BAIRRO[:24]:<24} {linha.PCT_DOM_ATE_RAIO_PONTO_BAIRRO:>5.1f}% com ponto do bairro · "
              f"{linha.PCT_DOM_ATE_RAIO_PONTO_SETOR:>5.1f}% com ponto do setor")
    for coluna in tabela.columns:
        if coluna.startswith(("PCT_", "DIST_")):
            tabela[coluna] = tabela[coluna].round(1)
    return tabela


# =============================================================================
# VALIDACAO CONTRA O GABARITO DO IBGE
# =============================================================================


def validar_contra_gabarito(bairros: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compara bairro a bairro com Agregados_por_bairros_basico_BR.csv.

    Os setores que ja trazem CD_BAIRRO somam exatamente os 519.463 habitantes do
    gabarito. Toda divergencia observada aqui e, portanto, efeito exclusivo da
    alocacao dos setores orfaos: e o teste isolado da Decisao 1.
    """
    subtitulo("11. Validacao contra o agregado por bairros do IBGE")

    if MALHA_BAIRROS != "ibge_87":
        print("gabarito indisponivel para esta malha — validacao pulada.")
        return pd.DataFrame()

    gabarito = pd.read_csv(ARQ_BAIRROS_CSV, sep=";", encoding="latin-1", dtype=str)
    gabarito = gabarito[gabarito["CD_MUN"] == CD_MUN][["CD_BAIRRO", "NM_BAIRRO", "v0001"]]
    gabarito["POP_IBGE"] = para_numero(gabarito["v0001"])
    checar(
        len(gabarito) == gabarito["CD_BAIRRO"].nunique(),
        "O gabarito de bairros traz CD_BAIRRO duplicado.",
    )
    checar(
        int(gabarito["POP_IBGE"].sum()) == POP_GABARITO_BAIRROS,
        f"Gabarito soma {int(gabarito['POP_IBGE'].sum())}, esperado {POP_GABARITO_BAIRROS}.",
    )

    comparacao = bairros[["CD_BAIRRO", "NM_BAIRRO", "REGIAO_FUNCIONAL", "POP"]].merge(
        gabarito[["CD_BAIRRO", "POP_IBGE"]], on="CD_BAIRRO", how="outer", indicator=True
    )
    checar(
        bool((comparacao["_merge"] == "both").all()),
        "Ha bairro presente em apenas um dos lados da comparacao.",
    )
    comparacao = comparacao.drop(columns="_merge")
    comparacao["DIFERENCA"] = comparacao["POP"] - comparacao["POP_IBGE"]
    comparacao["DIF_PCT"] = np.where(
        comparacao["POP_IBGE"] > 0, 100 * comparacao["DIFERENCA"] / comparacao["POP_IBGE"], np.nan
    )

    total_dif = int(comparacao["DIFERENCA"].sum())
    print(f"diferenca total ........................ {numero_br(total_dif)} hab")
    print(f"   (esperado: {numero_br(POP_REFERENCIA - POP_GABARITO_BAIRROS)} hab dos setores sem CD_BAIRRO)")
    checar(
        total_dif == POP_REFERENCIA - POP_GABARITO_BAIRROS,
        f"A diferenca total e {total_dif}, esperado {POP_REFERENCIA - POP_GABARITO_BAIRROS}.",
    )

    maiores = comparacao.reindex(
        comparacao["DIFERENCA"].abs().sort_values(ascending=False).index
    ).head(15)
    print("\n15 maiores divergencias:")
    print(f"   {'bairro':<28} {'regiao':<12} {'modelo':>9} {'IBGE':>9} {'dif':>9} {'dif %':>8}")
    for linha in maiores.itertuples():
        pct = "     —" if pd.isna(linha.DIF_PCT) else f"{linha.DIF_PCT:7.1f}%"
        print(
            f"   {linha.NM_BAIRRO[:28]:<28} {linha.REGIAO_FUNCIONAL:<12} "
            f"{numero_br(linha.POP):>9} {numero_br(linha.POP_IBGE):>9} "
            f"{numero_br(linha.DIFERENCA):>9} {pct:>8}"
        )

    # O Centro nao recebe setor orfao: divergencia ali indica erro de alocacao.
    centro = comparacao[comparacao["NM_BAIRRO"] == "Centro"]
    if len(centro) == 1:
        dif_centro = float(centro["DIF_PCT"].iloc[0])
        print(f"\ndivergencia no Centro .................. {dif_centro:.2f}%")
        if abs(dif_centro) > 1.0:
            print(
                "   ATENCAO: o Centro nao deveria absorver setores de borda. "
                "Divergencia acima de 1% indica erro na alocacao dos orfaos."
            )
        else:
            print("   dentro do esperado — a diferenca esta nos bairros de borda.")

    por_regiao = (
        comparacao.groupby("REGIAO_FUNCIONAL")["DIFERENCA"].sum().reindex(ORDEM_REGIOES)
    )
    print("\ndiferenca absorvida por regiao funcional:")
    for regiao, valor in por_regiao.items():
        print(f"   {regiao:<14} {numero_br(valor):>9} hab")

    return comparacao


# =============================================================================
# RESUMO E SAIDAS
# =============================================================================


def resumo_por_regiao(bairros: gpd.GeoDataFrame) -> pd.DataFrame:
    """Populacao, renda e area por regiao funcional."""
    subtitulo("12. Resumo por regiao funcional")

    bairros = bairros.copy()
    bairros["RENDA_X_DOM"] = bairros["RENDA_MEDIA"].fillna(0) * bairros["DOM_OCUPADOS"]
    bairros["DOM_COM_RENDA"] = np.where(bairros["RENDA_MEDIA"].notna(), bairros["DOM_OCUPADOS"], 0)

    resumo = bairros.groupby("REGIAO_FUNCIONAL").agg(
        BAIRROS=("CD_BAIRRO", "nunique"),
        POP=("POP", "sum"),
        DOM_OCUPADOS=("DOM_OCUPADOS", "sum"),
        AREA_KM2=("AREA_KM2", "sum"),
        RENDA_X_DOM=("RENDA_X_DOM", "sum"),
        DOM_COM_RENDA=("DOM_COM_RENDA", "sum"),
    )
    resumo["RENDA_MEDIA"] = resumo["RENDA_X_DOM"] / resumo["DOM_COM_RENDA"]
    resumo["POP_PCT"] = 100 * resumo["POP"] / resumo["POP"].sum()
    resumo["DENS_HAB_KM2"] = resumo["POP"] / resumo["AREA_KM2"]
    resumo = resumo.reindex(ORDEM_REGIOES).drop(columns=["RENDA_X_DOM", "DOM_COM_RENDA"])

    print(f"   {'regiao':<14} {'bairros':>8} {'populacao':>11} {'%':>7} {'area km2':>10} {'hab/km2':>10} {'renda media':>13}")
    for regiao, linha in resumo.iterrows():
        print(
            f"   {regiao:<14} {int(linha['BAIRROS']):>8} {numero_br(linha['POP']):>11} "
            f"{linha['POP_PCT']:>6.1f}% {numero_br(linha['AREA_KM2'], 1):>10} "
            f"{numero_br(linha['DENS_HAB_KM2'], 0):>10} R$ {numero_br(linha['RENDA_MEDIA'], 2):>10}"
        )
    print(
        f"   {'TOTAL':<14} {int(resumo['BAIRROS'].sum()):>8} {numero_br(resumo['POP'].sum()):>11} "
        f"{100.0:>6.1f}% {numero_br(resumo['AREA_KM2'].sum(), 1):>10}"
    )

    checar(
        int(resumo["POP"].sum()) == POP_REFERENCIA,
        "A soma por regiao funcional nao fecha com a populacao do municipio.",
    )
    return resumo.reset_index()


def verificar_controle_de_regioes(alocacao: gpd.GeoDataFrame, basico: pd.DataFrame) -> None:
    """Confere a classificacao regional ANTES da entrada dos setores orfaos.

    Controle independente do rateio: isola erro no mapeamento distrito -> regiao.
    """
    if MALHA_BAIRROS != "ibge_87":
        return
    subtitulo("13. Controle da classificacao regional (sem os setores orfaos)")

    diretos = alocacao[alocacao["REGRA"] == "cd_bairro_do_ibge"]
    dados = diretos.merge(basico[["CD_SETOR", "POP", "NM_DIST", "NM_BAIRRO"]], on="CD_SETOR")
    dados["REGIAO_FUNCIONAL"] = [
        classificar_regiao(d, b) for d, b in zip(dados["NM_DIST"], dados["NM_BAIRRO"])
    ]
    obtido = dados.groupby("REGIAO_FUNCIONAL").agg(
        bairros=("CD_BAIRRO", "nunique"), pop=("POP", "sum")
    )

    print(f"   {'regiao':<14} {'bairros':>8} {'esperado':>9} {'populacao':>11} {'esperado':>11}")
    for regiao in ORDEM_REGIOES:
        n_esp, pop_esp = CONTROLE_REGIOES_SEM_ORFAOS[regiao]
        n_obt = int(obtido.loc[regiao, "bairros"])
        pop_obt = int(obtido.loc[regiao, "pop"])
        print(
            f"   {regiao:<14} {n_obt:>8} {n_esp:>9} {numero_br(pop_obt):>11} {numero_br(pop_esp):>11}"
        )
        checar(
            n_obt == n_esp and pop_obt == pop_esp,
            f"Controle da regiao {regiao} falhou: {n_obt} bairros / {pop_obt} hab, "
            f"esperado {n_esp} / {pop_esp}. Revisar DISTRITO_REGIAO e BAIRROS_CONTINENTE.",
        )
    print("   controle conferido — mapeamento distrito/bairro -> regiao esta correto.")


def salvar(
    bairros: gpd.GeoDataFrame,
    pontos: gpd.GeoDataFrame,
    alocacao: gpd.GeoDataFrame,
    comparacao: pd.DataFrame,
    resumo: pd.DataFrame,
    setores_poligonos: gpd.GeoDataFrame,
    pontos_setor: gpd.GeoDataFrame,
    representatividade: pd.DataFrame,
) -> None:
    subtitulo("14. Gravacao das saidas")
    DIR_TRATADOS.mkdir(parents=True, exist_ok=True)
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)
    DIR_TABELAS_NAO_USADAS.mkdir(parents=True, exist_ok=True)

    saidas = [
        (DIR_TRATADOS / "bairros.gpkg", lambda p: bairros.to_file(p, layer="bairros", driver="GPKG")),
        (DIR_TRATADOS / "pontos_demanda.gpkg", lambda p: pontos.to_file(p, layer="pontos_demanda", driver="GPKG")),
        (
            DIR_TRATADOS / "alocacao_setores.parquet",
            lambda p: alocacao.drop(columns="geometry").to_parquet(p, index=False),
        ),
        (DIR_TRATADOS / "setores.gpkg", lambda p: setores_poligonos.to_file(p, layer="setores", driver="GPKG")),
        (
            DIR_TRATADOS / "pontos_demanda_setor.gpkg",
            lambda p: pontos_setor.to_file(p, layer="pontos_demanda_setor", driver="GPKG"),
        ),
        (DIR_TABELAS / "resumo_regioes.csv", lambda p: resumo.to_csv(p, index=False, sep=";", decimal=",")),
        (
            DIR_TABELAS_NAO_USADAS / "representatividade_pontos_demanda.csv",
            lambda p: representatividade.to_csv(p, index=False, sep=";", decimal=","),
        ),
    ]
    if len(comparacao) > 0:
        saidas.append(
            (
                DIR_TABELAS / "validacao_bairros.csv",
                lambda p: comparacao.sort_values("DIFERENCA", ascending=False).to_csv(
                    p, index=False, sep=";", decimal=","
                ),
            )
        )

    for caminho, gravar in saidas:
        caminho.unlink(missing_ok=True)
        gravar(caminho)
        print(f"   {caminho.relative_to(RAIZ).as_posix():<45} {caminho.stat().st_size / 1024:>8.0f} KB")


# =============================================================================
# EXECUCAO
# =============================================================================


def main() -> int:
    inicio = time.perf_counter()

    titulo(
        "01_demanda.py — demanda espacial por bairro\n"
        f"malha de bairros: {MALHA_BAIRROS}  ·  "
        f"orfaos: {REGRA_SETORES_ORFAOS}  ·  "
        f"renda 'X': {REGRA_RENDA_SUPRIMIDA}  ·  "
        f"renda sem linha: {REGRA_RENDA_SEM_LINHA}"
    )
    verificar_ausencia_de_centroid()

    setores, massas_dagua = carregar_setores()
    area_municipio_km2 = float(setores["AREA_M2_SETOR"].sum() / 1e6)
    basico = carregar_basico()
    renda = carregar_renda(set(basico["CD_SETOR"]))
    pontos_cnefe = carregar_domicilios_cnefe()
    bairros_ref = carregar_malha_bairros_referencia()

    contar_domicilios_por_setor(pontos_cnefe, setores, basico)

    alocacao = alocar_setores(setores, basico, bairros_ref, pontos_cnefe)
    renda_setor = imputar_renda(alocacao, basico, renda)
    bairros = agregar_por_bairro(
        alocacao, basico, renda_setor, bairros_ref, area_municipio_km2
    )
    pontos = gerar_pontos_demanda(bairros, massas_dagua, pontos_cnefe)
    setores_poligonos, pontos_setor = gerar_demanda_setor(
        setores, basico, alocacao, renda_setor, bairros, massas_dagua, pontos_cnefe
    )
    representatividade = representatividade_dos_pontos(
        bairros, pontos, setores_poligonos, pontos_setor, pontos_cnefe
    )

    comparacao = validar_contra_gabarito(bairros)
    resumo = resumo_por_regiao(bairros)
    verificar_controle_de_regioes(alocacao, basico)

    salvar(
        bairros, pontos, alocacao, comparacao, resumo,
        setores_poligonos, pontos_setor, representatividade,
    )

    titulo(
        f"Concluido em {time.perf_counter() - inicio:.1f} s  ·  "
        f"{len(bairros)} bairros  ·  {numero_br(bairros['POP'].sum())} habitantes"
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
