"""
horarios_osm.py — conversao de opening_hours do OSM em horas semanais.

Modulo auxiliar de src/04_atratividade.py. O atributo DISPONIBILIDADE e
parametrizado por tipo pela MEDIANA das horas semanais observadas; esta e a rotina
que converte o texto de opening_hours nessas horas. A versao original da Etapa 5
nao ficou no projeto; esta a reescreve e e validada contra os valores daquela etapa
(ver docs/decisoes_etapa12.md).

O que se mede: as horas de funcionamento de uma SEMANA-PADRAO, sem feriados.

Regras de interpretacao (a especificacao do OSM e maior; so o que aparece nos dados
de Florianopolis e tratado, e o resto FALHA em vez de ser adivinhado):

    - Regras separadas por ";" sao regras NORMAIS: uma regra posterior substitui as
      anteriores nos dias que ela seleciona ("Mo-Su 08:00-22:00; Su 09:00-13:00"
      deixa o domingo com 4 h).
    - Uma virgula seguida de um novo seletor de dia, depois de um horario, abre uma
      regra ADICIONAL, que soma em vez de substituir.
    - "24/7" vale 24 h nos sete dias (168 h).
    - Regra sem seletor de dia ("08:00-22:00") vale para os sete dias.
    - Intervalos: "HH:MM-HH:MM", varios separados por virgula (turnos). Fim menor ou
      igual ao inicio atravessa a meia-noite ("17:00-03:00" = 10 h; "07:00-00:00" =
      17 h). As horas sao contadas no dia em que o intervalo comeca, o que nao altera
      o total semanal.
    - Fim em aberto ("06:00-24:00+"): conta ate o fim declarado. O "+" diz que pode
      passar dele, mas nao quanto.
    - "off" / "closed": dia fechado.
    - FERIADOS (PH, SH): nao entram na semana-padrao. Regra cujo unico seletor e PH
      ou SH e ignorada por inteiro — inclusive "PH off", que numa versao anterior
      era aplicada aos sete dias e zerava a semana. PH junto de dias ("Mo-Su, PH")
      vale so para os dias.
    - Regras CONDICIONAIS por data ("Dec 25 off", "Jan 01 off", meses, semanas):
      excecoes de calendario, ignoradas. Nao ha nenhuma regra sazonal que mude a
      semana-padrao nos dados.
    - Alternativas "||" (fallback): so a parte anterior e lida.
    - Comentarios entre aspas sao removidos.

Executar isolado valida a conversao e refaz o teste de Kruskal-Wallis:
    conda activate tcc
    python src/horarios_osm.py
"""

from __future__ import annotations

import re

DIAS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
FERIADOS = {"PH", "SH"}
MESES = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

_ITEM_DIA = r"(?:Mo|Tu|We|Th|Fr|Sa|Su|PH|SH)(?:-(?:Mo|Tu|We|Th|Fr|Sa|Su))?"
_SELETOR = re.compile(rf"^({_ITEM_DIA}(?:\s*,\s*{_ITEM_DIA})*)(?=\s|$)\s*(.*)$")
_CONDICIONAL = re.compile(rf"^(?:\d{{4}}\b|{MESES}\b|week\b)")
_INTERVALO = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})(\+?)$")
# Virgula depois de um horario e antes de um seletor de dia: regra adicional.
_ADICIONAL = re.compile(r"(?<=\d|\+),\s*(?=(?:Mo|Tu|We|Th|Fr|Sa|Su|PH|SH)\b)")


class HorarioNaoInterpretado(ValueError):
    """Texto de opening_hours fora das regras tratadas: falha em vez de adivinhar."""


def _expandir_dias(seletor: str) -> tuple[list[str], bool]:
    """Dias da semana do seletor e se ele mencionava feriado."""
    dias: list[str] = []
    tinha_feriado = False
    for item in re.split(r"\s*,\s*", seletor.strip()):
        if item in FERIADOS:
            tinha_feriado = True
            continue
        if "-" in item:
            ini, fim = item.split("-")
            i, f = DIAS.index(ini), DIAS.index(fim)
            # Faixa que atravessa o domingo ("Fr-Mo") da a volta na semana.
            faixa = range(i, f + 1) if i <= f else list(range(i, 7)) + list(range(0, f + 1))
            dias.extend(DIAS[k] for k in faixa)
        else:
            dias.append(item)
    return list(dict.fromkeys(dias)), tinha_feriado


def _intervalos(texto: str, original: str) -> list[tuple[int, int]]:
    """Intervalos em minutos a partir de "HH:MM-HH:MM[,HH:MM-HH:MM...]"."""
    saida = []
    for parte in re.split(r"\s*,\s*", texto.strip()):
        m = _INTERVALO.match(parte)
        if not m:
            raise HorarioNaoInterpretado(f"intervalo nao reconhecido '{parte}' em '{original}'")
        h1, m1, h2, m2, _aberto = m.groups()
        inicio = int(h1) * 60 + int(m1)
        fim = int(h2) * 60 + int(m2)
        if fim <= inicio:  # atravessa a meia-noite; "00:00" como fim e 24:00
            fim += 24 * 60
        saida.append((inicio, fim))
    return saida


def _uniao(intervalos: list[tuple[int, int]]) -> int:
    """Minutos cobertos pela uniao dos intervalos (turnos sobrepostos contam uma vez)."""
    total, fim_atual = 0, None
    for ini, fim in sorted(intervalos):
        if fim_atual is None or ini > fim_atual:
            total += fim - ini
            fim_atual = fim
        elif fim > fim_atual:
            total += fim - fim_atual
            fim_atual = fim
    return total


def interpretar(texto: str) -> dict:
    """Horas por dia da semana-padrao e as regras ignoradas, para auditoria."""
    original = texto
    texto = re.sub(r'"[^"]*"', "", texto)           # comentarios
    texto = texto.split("||")[0]                     # alternativas: so a primeira
    texto = re.sub(r"\s+", " ", texto).strip()

    por_dia: dict[str, list[tuple[int, int]]] = {d: [] for d in DIAS}
    ignoradas: list[str] = []
    for normal in [r.strip() for r in texto.split(";") if r.strip()]:
        for n, regra in enumerate(_ADICIONAL.split(normal)):
            adicional = n > 0
            regra = regra.strip()
            if regra == "24/7":
                for d in DIAS:
                    por_dia[d] = [(0, 24 * 60)]
                continue
            if _CONDICIONAL.match(regra):
                ignoradas.append(f"{regra} (excecao de calendario)")
                continue
            m = _SELETOR.match(regra)
            if m:
                dias, tinha_feriado = _expandir_dias(m.group(1))
                resto = m.group(2).strip()
                if not dias:
                    ignoradas.append(f"{regra} (so feriado)")
                    continue
                if tinha_feriado:
                    ignoradas.append(f"{m.group(1)}: parte de feriado desconsiderada")
            else:
                dias, resto = list(DIAS), regra   # sem seletor: os sete dias
            if resto in ("off", "closed"):
                novos: list[tuple[int, int]] = []
            elif resto == "":
                raise HorarioNaoInterpretado(f"regra sem horario '{regra}' em '{original}'")
            else:
                novos = _intervalos(resto, original)
            for d in dias:
                por_dia[d] = (por_dia[d] + novos) if adicional else list(novos)

    horas = {d: _uniao(iv) / 60 for d, iv in por_dia.items()}
    return {"horas_por_dia": horas, "horas_semanais": sum(horas.values()), "ignoradas": ignoradas}


def horas_semanais(texto: str) -> float:
    """Horas de funcionamento numa semana-padrao, sem feriados."""
    return interpretar(texto)["horas_semanais"]


# =============================================================================
# CASOS CONFERIDOS A MAO
# =============================================================================
# Cada caso cobre uma regra da docstring; o valor esperado foi calculado a mao.
CASOS_CONFERIDOS = [
    ("24/7", 168.0),                                               # 24/7
    ("Mo-Fr 09:00-17:00; Sa,Su,PH off", 40.0),                     # dias fechados + PH
    ("Mo-Su 17:00-03:00", 70.0),                                   # atravessa meia-noite
    ("Mo-Sa 07:00-00:00; Su 07:00-21:00", 116.0),                  # 00:00 como fim
    ("Mo-Sa 08:00-12:00, 14:00-20:00; Su 08:00-12:00", 64.0),      # dois turnos
    ("08:00-22:00", 98.0),                                         # sem seletor de dia
    ("Mo-Su, PH 08:00-23:00; Dec 25 off; Jan 01 off", 105.0),      # PH junto + datas
    ("PH off", 0.0),                                               # so feriado: ignorada
    ("Mo-Fr 07:00-21:00; Sa 07:00-19:00; PH off", 82.0),           # PH off NAO zera a semana
    ("Mo-Fr 06:00-24:00+; Sa,Su 07:00-22:00+", 120.0),             # fim em aberto
    ("Mo-Su 08:00-22:00; Su 09:00-13:00", 88.0),                   # regra posterior substitui
    ("Mo-Fr 08:00-12:00, Sa 08:00-12:00", 24.0),                   # regra adicional
    ("Mo-Su 00:00-24:00", 168.0),
]


def conferir_casos() -> None:
    for texto, esperado in CASOS_CONFERIDOS:
        obtido = horas_semanais(texto)
        if abs(obtido - esperado) > 1e-9:
            raise AssertionError(f"'{texto}': {obtido} h, esperado {esperado} h")


# =============================================================================
# VALIDACAO E TESTE ESTATISTICO (execucao isolada)
# =============================================================================


def _dunn(grupos: dict[str, list[float]]) -> list[dict]:
    """Teste de Dunn (1964) par a par, com correcao de empates, sobre os postos globais."""
    import numpy as np
    from scipy.stats import norm, rankdata

    nomes = list(grupos)
    valores = np.concatenate([grupos[g] for g in nomes])
    postos = rankdata(valores)
    n = len(valores)
    _, contagens = np.unique(valores, return_counts=True)
    correcao_empates = (contagens**3 - contagens).sum() / (12 * (n - 1))
    media_posto, tamanho, ini = {}, {}, 0
    for g in nomes:
        k = len(grupos[g])
        media_posto[g], tamanho[g] = postos[ini:ini + k].mean(), k
        ini += k
    saida = []
    for a_i, a in enumerate(nomes):
        for b in nomes[a_i + 1:]:
            erro = np.sqrt((n * (n + 1) / 12 - correcao_empates) * (1 / tamanho[a] + 1 / tamanho[b]))
            z = (media_posto[a] - media_posto[b]) / erro
            saida.append({"par": (a, b), "z": z, "p": 2 * norm.sf(abs(z))})
    return saida


def _holm(pvalores: list[float]) -> list[float]:
    """Correcao de Holm-Bonferroni, com a monotonicidade garantida."""
    ordem = sorted(range(len(pvalores)), key=lambda i: pvalores[i])
    m, ajustados, maior = len(pvalores), [0.0] * len(pvalores), 0.0
    for posicao, i in enumerate(ordem):
        maior = max(maior, min(1.0, (m - posicao) * pvalores[i]))
        ajustados[i] = maior
    return ajustados


def analisar(horas_por_tipo: dict[str, list[float]], rotulo: str) -> dict:
    """Kruskal-Wallis, medianas, desvios e pos-testes par a par."""
    import numpy as np
    from scipy.stats import kruskal, mannwhitneyu

    grupos = {t: v for t, v in sorted(horas_por_tipo.items()) if v}
    h, p = kruskal(*grupos.values())
    print(f"\n== {rotulo}: {sum(len(v) for v in grupos.values())} observacoes")
    print(f"   Kruskal-Wallis H = {h:.4f}, p = {p:.3e}")
    for t, v in grupos.items():
        print(f"   {t:<16} n = {len(v):>2}  mediana {np.median(v):6.2f}  media {np.mean(v):6.2f}  "
              f"desvio-padrao {np.std(v, ddof=1):5.2f}")
    dunn = _dunn(grupos)
    holm = _holm([d["p"] for d in dunn])
    pares = []
    print(f"   {'par':<36} {'MW p':>8} {'Dunn p':>8} {'Dunn-Holm':>10}")
    for d, ph in zip(dunn, holm):
        a, b = d["par"]
        mw = mannwhitneyu(grupos[a], grupos[b], alternative="two-sided").pvalue
        pares.append({"par": f"{a} x {b}", "mw_p": mw, "dunn_p": d["p"], "dunn_holm_p": ph,
                      "mediana_a": float(np.median(grupos[a])), "mediana_b": float(np.median(grupos[b]))})
        print(f"   {a + ' x ' + b:<36} {mw:8.4f} {d['p']:8.4f} {ph:10.4f}")
    return {"H": h, "p": p, "grupos": grupos, "pares": pares}


def _main() -> int:
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd

    raiz = Path(__file__).resolve().parents[1]
    conferir_casos()
    print(f"casos conferidos a mao: {len(CASOS_CONFERIDOS)} de {len(CASOS_CONFERIDOS)} corretos")

    atual = gpd.read_file(raiz / "data" / "tratados" / "candidatos_com_aj.gpkg")
    obs_atual = atual[atual["HORARIO"].notna()][["OSM_ID", "TIPO", "NOME", "HORARIO"]].copy()

    # Cenario antigo (872 candidatos): as 84 de hoje mais os registros do OSM com
    # horario que estavam no conjunto de 872 e sairam depois. So um tinha horario: o
    # IMPERATRIZ 771704487, descartado em setembro por estar 38 m dentro de Sao Jose.
    cache = pd.read_parquet(raiz / "data" / "tratados" / "_cache_osm_candidatos.parquet")
    removido = cache[cache["id"].astype(str) == "771704487"]
    extra = pd.DataFrame({"OSM_ID": ["771704487"], "TIPO": ["supermercado"],
                          "NOME": [removido["name"].iloc[0]], "HORARIO": [removido["opening_hours"].iloc[0]]})
    obs_antigo = pd.concat([obs_atual, extra], ignore_index=True)

    resultados = {}
    for rotulo, obs in [("CENARIO ANTIGO (872 candidatos)", obs_antigo), ("CONJUNTO FINAL (856 candidatos)", obs_atual)]:
        obs["HORAS"] = obs["HORARIO"].map(horas_semanais)
        por_tipo = {t: sub["HORAS"].tolist() for t, sub in obs.groupby("TIPO")}
        resultados[rotulo] = analisar(por_tipo, rotulo)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
