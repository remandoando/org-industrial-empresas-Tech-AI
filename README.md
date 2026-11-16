# Sustitución entre modelos de lenguaje cerrados y abiertos vía API (H2)

Código del paper *Cournot, Bertrand y poder de mercado en el mercado de modelos de lenguaje vía API*.
Prueba H2: los usuarios sustituyen más **dentro** del grupo de modelos cerrados y dentro del de
abiertos que **entre** grupos (logit anidado, σ > 0).

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # pega tu llave gratuita de OpenRouter
python tests/smoke_test.py                             # 1 min: verifica el código con datos simulados
python run_all.py                                      # descarga, construye y estima todo
```

La descarga toma de 20 a 60 minutos (límites de las APIs y del Internet Archive). Después puedes
re-estimar sin descargar: `python run_all.py --no-download`.

## Qué produce cada script

| Script | Salida | Contenido |
| --- | --- | --- |
| `src/01_download.py` | `data/raw/` | Datos crudos con fecha de consulta en el nombre |
| `src/02_build_panel.py` | `data/clean/panel.csv`, `output/tables/limpieza.csv`, `cobertura.csv` | Panel modelo-semana y registro de cada filtro |
| `src/03_descriptives.py` | Tabla 1, Figuras 1–2 | Descriptivos por nido, participación e IHH, precio vs calidad |
| `src/04_demand.py` | Tablas 2–4, Figura 3 | Logit/anidado MCO y VI; elasticidades y desvíos con IC 95%; monopolista hipotético |
| `src/05_robustness.py` | Tabla 6 | 11 especificaciones alternativas |
| `src/nl.py` | — | Instrumentos, estimador y fórmulas del logit anidado |

## Pasos manuales (documéntalos en el paper)

1. **Tabla de equivalencias.** `02_build_panel.py` escribe `data/crosswalk/crosswalk_auto.csv`
   con una propuesta de empate entre OpenRouter y LMArena. Revísala fila por fila, corrige
   `arena_name` y `open_weights` (1 = pesos públicos) y guárdala como `data/crosswalk/crosswalk.csv`.
   Vuelve a correr desde el paso 2.
2. **Precios faltantes.** Si el Internet Archive no cubre alguna semana, el código usa el precio
   actual hacia atrás y lo marca como `actual_hacia_atras` (ver `output/tables/cobertura.csv`).
   Para reemplazarlo, copia `data/manual/prices_manual_PLANTILLA.csv` como `prices_manual.csv`,
   borra la fila de ejemplo y agrega solo precios con fuente fechada.

## Fuentes y citas

- OpenRouter Data API y catálogo de modelos (CC BY 4.0). Citar: "Source: OpenRouter
  (openrouter.ai/rankings), as of {fecha}" con la fecha de `data/raw/openrouter_rankings_meta_*.json`.
- LMArena, `lmarena-ai/leaderboard-dataset` (Hugging Face).
- Internet Archive, Wayback Machine (capturas de `openrouter.ai/api/v1/models`).

## Decisiones de diseño que el texto debe explicar

- Periodo: desde el 2025-01-01, porque el dataset público de OpenRouter empieza ahí.
- Mercado potencial M_t = λ × tokens totales de la semana. Con efectos fijos de semana, λ no
  cambia α ni σ (ln s_0t queda absorbido); solo cambia elasticidades y desvíos.
- Instrumentos de diferenciación (Gandhi y Houde, 2019): número de rivales en el nido, rivales a
  menos de 50 puntos Arena (todos y del nido) y suma de diferencias cuadradas de calidad.
- La medida directa de H2 es `mismo_nido_entre_internos`: de los tokens que pierde un modelo y que
  se quedan en modelos del top 50, qué fracción se queda en su mismo nido.
- Reproducibilidad: semilla en `config.py`; versiones fijas en `requirements.txt`.

## Pruebas

- `tests/smoke_test.py` simula un mercado con α = 0.4 y σ = 0.6 y comprueba que el estimador los
  recupera. Sus tablas van a `tests/output/` y **no son resultados del paper**.
- `tests/test_build_panel.py` construye el panel con archivos falsos con el formato de las APIs.
