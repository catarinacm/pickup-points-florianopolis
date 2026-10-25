# Localização de pickup points em Florianópolis

Código do Trabalho de Conclusão de Curso *Localização de pickup points em áreas urbanas: método baseado no problema das p-medianas e aplicação ao caso de Florianópolis*, de Catarina Corrêa Miguel (Engenharia de Produção Civil, UFSC, 2026; orientador: Prof. Enzo Morosini Frazzon).

## Como rodar

Python 3.13:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --only-binary=:all: -r requirements.txt
python src/executar_tudo.py
```

O `executar_tudo.py` roda os scripts de `src/` na ordem, do download dos dados do IBGE às figuras (cerca de 1 h 20 min), e grava os resultados em `results/`. As extrações do OpenStreetMap usadas no trabalho (setembro de 2026) estão em `data/tratados/`.

Code for an undergraduate thesis on locating parcel pickup points in Florianópolis, Brazil. Run `python src/executar_tudo.py` after installing `requirements.txt`.

## Licença

Código sob licença MIT; dados derivados do OpenStreetMap (ODbL 1.0) e do IBGE. Ver `LICENSE`.
