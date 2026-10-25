"""Executa o pipeline completo, do download dos dados brutos as figuras, na ordem certa.

Uso (a partir da raiz do repositorio, com o ambiente ativado):

    python src/executar_tudo.py                  roda tudo, de 00 a 08
    python src/executar_tudo.py --a-partir-de 05 retoma a partir de um script
    python src/executar_tudo.py --sem-download   pula o 00 (dados brutos ja baixados)
    python src/executar_tudo.py --listar         so mostra a ordem, sem executar

Cada script roda num processo proprio, com o mesmo interpretador deste. Se um deles
falhar, a execucao para ali e informa qual foi; os anteriores ja gravaram suas saidas
em data/ e results/, entao basta corrigir e retomar com --a-partir-de.

O 06_modelo.py nao entra na sequencia: ele contem o modelo usado pelo 07, 09 e 10 e,
rodado sozinho, resolve um unico cenario como teste rapido.

Tempo de referencia: cerca de 1 h 20 min num computador pessoal, a maior parte no
07_cenarios.py e no 10_sensibilidade_escore_postal.py.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

DIR_SRC = Path(__file__).resolve().parent

# Ordem de execucao. O 08 (figuras) vem por ultimo porque le as saidas do 07, do 09
# e do 10.
ETAPAS = [
    ("00", "00_baixar_dados_brutos.py", "baixa os dados brutos do IBGE (cerca de 170 MB)"),
    ("01", "01_demanda.py", "setores, bairros e pontos de demanda"),
    ("02", "02_demanda_ecommerce.py", "demanda de comercio eletronico por setor"),
    ("03", "03_candidatos.py", "estabelecimentos candidatos e validacao OSM x CNEFE"),
    ("04", "04_atratividade.py", "indice de atratividade dos candidatos"),
    ("05", "05_distancias.py", "matrizes de distancia (euclidiana, a pe e viaria)"),
    ("07", "07_cenarios.py", "cenarios C1 a C6 e sensibilidades (42 execucoes)"),
    ("09", "09_curva_p.py", "curva de cobertura em funcao de p"),
    ("10", "10_sensibilidade_escore_postal.py", "sensibilidade ao escore das agencias postais"),
    ("08", "08_mapas.py", "mapas e graficos"),
]


def formatar_tempo(segundos: float) -> str:
    minutos, seg = divmod(int(round(segundos)), 60)
    horas, minutos = divmod(minutos, 60)
    return f"{horas} h {minutos:02d} min {seg:02d} s" if horas else f"{minutos} min {seg:02d} s"


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa o pipeline completo na ordem.")
    parser.add_argument("--a-partir-de", metavar="NN", help="numero do script em que comecar (ex.: 05)")
    parser.add_argument("--sem-download", action="store_true", help="pula o 00_baixar_dados_brutos.py")
    parser.add_argument("--listar", action="store_true", help="so mostra a ordem de execucao")
    args = parser.parse_args()

    etapas = list(ETAPAS)
    if args.sem_download:
        etapas = [e for e in etapas if e[0] != "00"]
    if args.a_partir_de:
        numeros = [e[0] for e in etapas]
        if args.a_partir_de not in numeros:
            parser.error(f"--a-partir-de deve ser um destes: {', '.join(numeros)}")
        etapas = etapas[numeros.index(args.a_partir_de):]

    print("Ordem de execucao:")
    for numero, arquivo, descricao in etapas:
        print(f"   {numero}  {arquivo:<38} {descricao}")
    if args.listar:
        return 0

    inicio_total = time.perf_counter()
    for numero, arquivo, descricao in etapas:
        print()
        print("=" * 78)
        print(f"{arquivo} — {descricao}")
        print("=" * 78, flush=True)
        inicio = time.perf_counter()
        resultado = subprocess.run([sys.executable, str(DIR_SRC / arquivo)])
        duracao = time.perf_counter() - inicio
        if resultado.returncode != 0:
            print(f"\nFALHOU: {arquivo} (codigo {resultado.returncode}) depois de {formatar_tempo(duracao)}.")
            print(f"Depois de corrigir, retome com: python src/executar_tudo.py --a-partir-de {numero}")
            return resultado.returncode
        print(f"\n{arquivo} concluido em {formatar_tempo(duracao)}.")

    print(f"\nPipeline completo em {formatar_tempo(time.perf_counter() - inicio_total)}.")
    print("Resultados em results/ (tabelas, figuras e logs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
