"""
00_baixar_dados_brutos.py — baixa os dados brutos do IBGE para data/brutos/.

Os arquivos brutos NAO sao versionados neste repositorio: sao grandes (a malha de
setores de SC tem 116 MB; os agregados nacionais, centenas de MB descompactados) e
tem fonte oficial estavel. Este script os busca no servidor de arquivos do IBGE.

Fonte: IBGE, Censo Demografico 2022 — Agregados por Setores Censitarios, malha de
setores e de bairros com atributos, e Cadastro Nacional de Enderecos para Fins
Estatisticos (CNEFE).

Datas:
    - download usado na monografia: 08/09/2026 (data dos arquivos em data/brutos/);
    - enderecos abaixo conferidos em: 05/10/2026.

O IBGE publica revisoes dos agregados com a data no nome do arquivo. Os nomes
abaixo sao os vigentes na data de conferencia; uma revisao posterior muda o nome,
e o script avisa quando o arquivo nao e encontrado.

Uso:
    python src/00_baixar_dados_brutos.py            # baixa o que faltar
    python src/00_baixar_dados_brutos.py --listar   # so mostra os enderecos
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_BRUTOS = RAIZ / "data" / "brutos"

FTP = "https://ftp.ibge.gov.br"
CENSO = f"{FTP}/Censos/Censo_Demografico_2022"
AGREGADOS = f"{CENSO}/Agregados_por_Setores_Censitarios"

# (endereco, pasta de destino em data/brutos/, descricao, usado por)
ARQUIVOS = [
    (f"{AGREGADOS}/malha_com_atributos/setores/gpkg/UF/SC/SC_setores_CD2022.gpkg",
     "ibge", "Malha de setores censitarios de SC, com atributos", "01, 03, 05, 08"),
    (f"{AGREGADOS}/malha_com_atributos/bairros/gpkg/UF/SC/SC_bairros_CD2022.gpkg",
     "ibge", "Malha de bairros de SC, com atributos", "01"),
    (f"{AGREGADOS}/Agregados_por_Setor_csv/Agregados_por_setores_basico_BR_20260520.zip",
     "ibge", "Agregados por setores — basico", "01"),
    (f"{AGREGADOS}/Agregados_por_Bairro_csv/Agregados_por_bairros_basico_BR_20260520.zip",
     "ibge", "Agregados por bairros — basico", "01"),
    (f"{CENSO}/Agregados_por_Setores_Censitarios_Rendimento_do_Responsavel/"
     "Agregados_por_setores_renda_responsavel_BR_20260508_xlsx.zip",
     "ibge", "Agregados por setores — rendimento do responsavel", "01"),
    (f"{AGREGADOS}/dicionario_de_dados_agregados_por_setores_censitarios_20260520.xlsx",
     "ibge", "Dicionario de dados dos agregados", "consulta"),
    (f"{FTP}/Cadastro_Nacional_de_Enderecos_para_Fins_Estatisticos/Censo_Demografico_2022/"
     "Arquivos_CNEFE/CSV/Municipio/42_SC/4205407_FLORIANOPOLIS.zip",
     "cnefe", "CNEFE 2022 — Florianopolis (4205407)", "01, 03, 04"),
]


def baixar(endereco: str, destino: Path) -> bool:
    """Baixa um arquivo, com barra de progresso simples. Devolve False se falhar."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    try:
        with urllib.request.urlopen(endereco, timeout=60) as resposta, open(parcial, "wb") as saida:
            total = int(resposta.headers.get("Content-Length", 0))
            lido = 0
            while True:
                bloco = resposta.read(1 << 20)
                if not bloco:
                    break
                saida.write(bloco)
                lido += len(bloco)
                if total:
                    print(f"\r      {lido / 1e6:7.1f} de {total / 1e6:.1f} MB", end="")
        print()
    except (urllib.error.URLError, TimeoutError) as erro:
        print(f"\n      FALHOU: {erro}")
        print("      O IBGE pode ter publicado uma revisao com outra data no nome. Conferir em:")
        print(f"      {endereco.rsplit('/', 1)[0]}/")
        parcial.unlink(missing_ok=True)
        return False
    parcial.replace(destino)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa os dados brutos do IBGE.")
    parser.add_argument("--listar", action="store_true", help="so mostra os enderecos, sem baixar")
    listar = parser.parse_args().listar

    falhas = 0
    for endereco, pasta, descricao, usado_por in ARQUIVOS:
        nome = endereco.rsplit("/", 1)[1]
        destino = DIR_BRUTOS / pasta / nome
        print(f"{descricao}  (scripts: {usado_por})")
        print(f"   {endereco}")
        if listar:
            continue
        if destino.exists():
            print("   ja existe — mantido")
        elif not baixar(endereco, destino):
            falhas += 1
            continue
        if destino.suffix == ".zip":
            with zipfile.ZipFile(destino) as pacote:
                faltam = [n for n in pacote.namelist() if not (destino.parent / n).exists()]
                if faltam:
                    pacote.extractall(destino.parent, members=faltam)
                print(f"   descompactado: {', '.join(pacote.namelist())}")

    if not listar:
        print()
        print("Conferido em 06/10/2026: os arquivos descompactados chegam com os nomes que")
        print("src/01_demanda.py espera (sem a data da revisao). Se uma revisao futura do IBGE")
        print("trouxer a data no nome do .csv ou do .xlsx, renomear para:")
        print("   data/brutos/ibge/Agregados_por_setores_basico_BR.csv")
        print("   data/brutos/ibge/Agregados_por_bairros_basico_BR.csv")
        print("   data/brutos/ibge/Agregados_por_setores_renda_responsavel_BR.xlsx")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
