# Localização de pickup points em áreas urbanas: método baseado no problema das p-medianas e aplicação ao caso de Florianópolis

*[English version below](#pickup-point-location-in-urban-areas-a-p-median-based-method-applied-to-florianópolis)*

**Autora:** Catarina Corrêa Miguel  
**Orientador:** Prof. Enzo Morosini Frazzon  
Engenharia de Produção Civil, Universidade Federal de Santa Catarina (UFSC)

Código, dados e resultados do trabalho, que localiza pickup points (pontos de retirada de encomendas) em Florianópolis por um modelo de p-medianas com restrição de cobertura e índice de atratividade dos estabelecimentos.

## Como rodar

Python 3.13. Instalar o ambiente:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --only-binary=:all: -r requirements.txt
```

Ordem de execução dos scripts de `src/`:

1. `00_baixar_dados_brutos.py` — baixa os dados brutos do IBGE (cerca de 170 MB).
2. `01` a `05` — demanda, candidatos, índice de atratividade e distâncias.
3. `07_cenarios.py`, `09_curva_p.py`, `10_sensibilidade_escore_postal.py` e `08_mapas.py` — cenários, curva de cobertura, sensibilidade e figuras.

O `06_modelo.py` contém o modelo, usado pelos demais; rodado sozinho, resolve só um cenário, como teste rápido.

**Tempo:** a rodada completa, do download às figuras, levou 1 hora e 18 minutos num computador pessoal. A maior parte é o `07_cenarios.py` (42 execuções; de 13 a 38 minutos nas rodadas medidas) e o `10_sensibilidade_escore_postal.py` (de 8 a 24 minutos).

**Solvers:** CBC 2.10.3 (distribuído com o PuLP 2.8.0) e HiGHS 1.15.1, com gap relativo zero nos dois estágios.

As extrações do OpenStreetMap usadas no trabalho (setembro de 2026) estão em `data/tratados/_cache_*`. Com elas, a rodada do zero reproduz os resultados de `results/`.

## Licença

Código sob licença MIT. Dados derivados do OpenStreetMap sob ODbL 1.0 (© colaboradores do OpenStreetMap); dados derivados do IBGE, Censo Demográfico 2022, com citação da fonte. Detalhes em `LICENSE`.

## Referências

- MIGUEL, Catarina Corrêa. *Localização de pickup points em áreas urbanas: método baseado no problema das p-medianas e aplicação ao caso de Florianópolis*. Trabalho de Conclusão de Curso (Graduação em Engenharia de Produção Civil) – Universidade Federal de Santa Catarina, Florianópolis, 2026. Orientador: Enzo Morosini Frazzon.
- MIGUEL, Catarina Corrêa; FRAZZON, Enzo Morosini; OLIVEIRA, Bruna Rigon de. *Localização de pickup points em áreas urbanas: método baseado no problema das p-medianas e aplicação a Florianópolis*. Em preparação, 2026.

---

# Pickup point location in urban areas: a p-median-based method applied to Florianópolis

**Author:** Catarina Corrêa Miguel  
**Advisor:** Prof. Enzo Morosini Frazzon  
Civil Production Engineering, Universidade Federal de Santa Catarina (UFSC), Brazil

Code, data and results of the study, which locates parcel pickup points in Florianópolis with a p-median model with a coverage constraint and a store attractiveness index.

## Running

Python 3.13. Set up the environment:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --only-binary=:all: -r requirements.txt
```

Run the scripts in `src/` in this order:

1. `00_baixar_dados_brutos.py` — downloads the raw IBGE census data (about 170 MB).
2. `01` to `05` — demand, candidate stores, attractiveness index and distances.
3. `07_cenarios.py`, `09_curva_p.py`, `10_sensibilidade_escore_postal.py` and `08_mapas.py` — scenarios, coverage curve, sensitivity analysis and figures.

`06_modelo.py` holds the model used by the others; run on its own, it solves a single scenario as a quick test.

**Time:** the full run, from download to figures, took 1 hour 18 minutes on a personal computer, most of it in `07_cenarios.py` (42 runs; 13 to 38 minutes in the measured runs) and `10_sensibilidade_escore_postal.py` (8 to 24 minutes).

**Solvers:** CBC 2.10.3 (shipped with PuLP 2.8.0) and HiGHS 1.15.1, with a zero relative gap in both stages.

The OpenStreetMap extracts used in the study (September 2026) are in `data/tratados/_cache_*`. With them, a run from scratch reproduces the results in `results/`.

## Licence

Code under the MIT licence. Data derived from OpenStreetMap under ODbL 1.0 (© OpenStreetMap contributors); data derived from IBGE, 2022 Demographic Census, with source citation. See `LICENSE`.

## References

- Miguel, C. C. (2026). *Localização de pickup points em áreas urbanas: método baseado no problema das p-medianas e aplicação ao caso de Florianópolis*. Undergraduate thesis, Civil Production Engineering, Universidade Federal de Santa Catarina. Advisor: Enzo Morosini Frazzon.
- Miguel, C. C., Frazzon, E. M., & Oliveira, B. R. de. *Localização de pickup points em áreas urbanas: método baseado no problema das p-medianas e aplicação a Florianópolis*. In preparation, 2026.
