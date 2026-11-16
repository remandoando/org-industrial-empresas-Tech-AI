# Tres capas de competencia en el mercado de modelos de lenguaje vía API

Código y datos del working paper *Tres capas de competencia en el mercado de modelos de lenguaje vía API:
entrada, sustitución entre grupos y Bertrand entre proveedores*.

El paper estudia la competencia en tres capas con datos públicos de OpenRouter (enero de 2025 a noviembre de 2026):

| Capa | Hipótesis | Método | Script |
| --- | --- | --- | --- |
| Entre modelos | H8: la entrada no sigue al tamaño del mercado (Sutton contra competencia monopolística) | Elasticidad del número de modelos y del IHH al tamaño del mercado | `08_sutton_monopolistic.py` |
| Entre grupos | H2: los usuarios sustituyen más dentro del grupo de modelos cerrados o del de abiertos que entre grupos | Logit anidado con instrumentos de diferenciación; estudio de eventos sin precios | `04_demand.py`, `05_robustness.py`, `06_event_study.py` |
| Entre proveedores | H7: competencia entre proveedores de un mismo modelo abierto (Cournot contra Bertrand) | Precio contra número de proveedores, al estilo de Bresnahan y Reiss | `07_provider_cournot.py` |

## Cómo reproducir los resultados

Probado con Python 3.13. En macOS usa `python3`.

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # pega tus llaves (ver abajo)
python3 tests/smoke_test.py                            # 1 min: verifica el código con datos simulados
```

**Opción A: con los datos de este repositorio (recomendada).** Reproduce las cifras del paper.

```bash
python3 src/01_download.py --only arena                # LMArena no viene en el repositorio (ver "Datos")
python3 run_all.py --no-download
```

**Opción B: descargar todo de nuevo.** Toma de 20 a 60 minutos por los límites de las APIs y del Internet Archive.

```bash
python3 run_all.py
```

Con la opción B las cifras de H7 pueden cambiar: los precios por proveedor de OpenRouter cambian durante
el día. En tres descargas del 15 y 16 de noviembre de 2026, el efecto del segundo proveedor fue −41%, −36%
y −39%.

### Llaves (archivo `.env`, nunca se sube al repositorio)

| Variable | Para qué | Cómo obtenerla |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Tokens semanales por modelo (`/api/v1/datasets/rankings-daily`) | Gratis en openrouter.ai → Keys |
| `HF_TOKEN` | Dataset de LMArena en Hugging Face | Gratis en huggingface.co → Settings → Access Tokens (lectura) |

## Datos

Corrida definitiva: 16 de noviembre de 2026. La fecha de consulta va en el nombre de cada archivo de `data/raw/`.

| Fuente | Archivo | Contenido | ¿En el repositorio? |
| --- | --- | --- | --- |
| OpenRouter Data API | `openrouter_rankings_week_*.csv` | Tokens semanales del top 50 + "other" (4,947 filas, 97 semanas) | Sí |
| Catálogo de OpenRouter | `openrouter_models_*.json` | 469 modelos con precio, contexto y Hugging Face ID | Sí |
| Endpoints de OpenRouter | `openrouter_endpoint_prices_*.csv` | 1,538 filas proveedor-modelo (H7) | Sí |
| Internet Archive | `archive/` | Catálogo semanal histórico: 81 semanas, 28,024 precios | Sí |
| LMArena | `lmarena_text_*.csv` | Calificación Arena por modelo y fecha (1,279,476 filas) | **No** |

**LMArena no se incluye** porque el archivo pesa unos 190 MB y GitHub no acepta archivos de más de 100 MB.
Para obtenerlo: `python3 src/01_download.py --only arena` (requiere `HF_TOKEN`). Fuente: dataset
`lmarena-ai/leaderboard-dataset` en Hugging Face, configuración `text`, descargado el 16 de noviembre de 2026.
Cada calificación trae su fecha de publicación y el código usa la última anterior a cada semana, así que una
descarga posterior debería dar la misma calidad para el periodo del paper.

## Qué produce cada script

| Script | Salida | Contenido |
| --- | --- | --- |
| `src/01_download.py` | `data/raw/` | Datos crudos con fecha de consulta en el nombre |
| `src/02_build_panel.py` | `data/clean/panel.csv`, `output/tables/limpieza.csv`, `cobertura.csv` | Panel modelo-semana y registro de cada filtro |
| `src/crosswalk_helper.py` | `data/crosswalk/crosswalk_propuesto.csv` | Propuesta de equivalencias OpenRouter–LMArena con nivel de confianza |
| `src/03_descriptives.py` | Tabla 1, Figuras 1–2 | Descriptivos por grupo, participación e IHH, precio contra calidad |
| `src/04_demand.py` | Tablas 2–3, Figura 3 | Logit y logit anidado (MCO y VI); razones de desvío con IC 95% |
| `src/05_robustness.py` | Tabla 6 | 12 especificaciones alternativas del logit anidado |
| `src/06_event_study.py` | Tabla 7, Figura 4, `eventos_lista.csv` | Validación de H2 sin precios: entrada de modelos al top 50 |
| `src/07_provider_cournot.py` | Tabla 8, Figura 5, `mercados_proveedores.csv`, `tabla9_umbrales_entrada.md` | H7: precio contra número de proveedores; umbrales de entrada (solo si hay datos suficientes) |
| `src/08_sutton_monopolistic.py` | Tabla 11, Figuras 7–9, `serie_variedades.csv` | H8: variedades y concentración contra tamaño de mercado |
| `src/nl.py` | — | Instrumentos, estimador y fórmulas del logit anidado |

Notas sobre las salidas:

- **Tabla 3** reporta solo razones de desvío. Sus intervalos se obtienen simulando σ, porque en el logit anidado
  los desvíos no dependen del coeficiente de precio, y este no queda identificado (Tabla 2).
- **No hay Tabla 4** (monopolista hipotético): requiere un precio identificado.
- **Tabla 9 y Figura 6** (umbrales de entrada) solo se generan si cada categoría de N tiene al menos 5 mercados
  con tamaño observado. En la corrida definitiva no se cumple, y `tabla9_umbrales_entrada.md` explica por qué.
- **No hay Tablas 5 ni 10**: la numeración sigue la del paper.

## Tabla de equivalencias OpenRouter–LMArena

Los nombres de los modelos no coinciden entre fuentes. `data/crosswalk/crosswalk.csv` es la tabla usada
en el paper. Se construyó así:

1. `02_build_panel.py` escribe una propuesta automática en `data/crosswalk/crosswalk_auto.csv`.
2. `python3 src/crosswalk_helper.py` propone el nombre de Arena de cada modelo con su nivel de confianza:
   - Empata por palabras y números de versión, sin fechas, sufijos (preview, instruct), cuantización
     (fp8, nvfp4) ni prefijos de proveedor; solo acepta candidatos del mismo creador.
   - Arena lista muchos modelos de razonamiento solo por nivel de esfuerzo. Cuando no existe el nombre simple,
     elige la variante con esta prioridad: simple, medium, high, low, xhigh, thinking, reasoning, max, minimal.
     La confianza queda como `alta (variante X)`.
3. Se revisaron a mano las filas de confianza "media". Los modelos sin un equivalente verdadero se dejaron
   sin empatar; su calidad se imputa con la mediana de su grupo en esa semana (27.9% de los modelo-semana).
4. Arena usa dos formatos de nombre (`claude-opus-5.5-high` y `Claude Opus 5.5 (High)`). `02_build_panel.py`
   los combina con un nombre normalizado y promedia el rating de la misma fecha.

Para rehacerla: corre el helper, revisa `crosswalk_propuesto.csv`, guárdalo como `crosswalk.csv` y corre
`python3 run_all.py --no-download`. La consola debe decir `usando crosswalk.csv (revisado a mano)`.

## Precios faltantes

Si el Internet Archive no cubre alguna semana, el código usa el precio actual hacia atrás y lo marca como
`actual_hacia_atras` (6.4% de los precios; ver `output/tables/cobertura.csv`). Para reemplazarlo, copia
`data/manual/prices_manual_PLANTILLA.csv` como `prices_manual.csv`, borra la fila de ejemplo y agrega solo
precios con fuente fechada.

## Decisiones de diseño

- Periodo: desde el 2025-01-01, porque el dataset público de OpenRouter empieza ahí.
- Mercado potencial M_t = λ × tokens totales de la semana (λ = 1.5). Con efectos fijos de semana, λ no
  cambia α ni σ; solo cambia los desvíos.
- Instrumentos de diferenciación (Gandhi y Houde, 2019): número de rivales en el grupo, rivales a menos de
  50 puntos Arena (todos y del grupo) y suma de diferencias cuadradas de calidad.
- La medida directa de H2 es `mismo_nido_entre_internos`: de los tokens que pierde un modelo y se quedan en
  el top 50, qué fracción se queda en su mismo grupo.
- Reproducibilidad: semilla 20261115 en `config.py`; versiones fijas en `requirements.txt`.

## Fuentes y licencias

- OpenRouter Data API y catálogo de modelos (CC BY 4.0). Cita: "Source: OpenRouter (openrouter.ai/rankings),
  as of {fecha}", con la fecha de `data/raw/openrouter_rankings_meta_*.json`.
- LMArena, `lmarena-ai/leaderboard-dataset` (Hugging Face).
- Internet Archive, Wayback Machine (capturas de `openrouter.ai/api/v1/models`).

## Pruebas

Las tablas de las pruebas van a `tests/output/` y **no son resultados del paper**.

- `tests/smoke_test.py` simula un mercado con α = 0.4 y σ = 0.6 y comprueba que el estimador los recupera.
- `tests/test_build_panel.py` construye el panel con archivos falsos con el formato de las APIs.
- `tests/test_provider_cournot.py` simula mercados de Cournot y comprueba que la prueba rechaza Bertrand.
- `tests/test_sutton.py` simula entrada con elasticidad 1 y comprueba que la recupera.

## Uso de IA

Se usó Claude Opus 5.5 (Anthropic) en noviembre de 2026 para proponer la estrategia empírica, escribir y
depurar el código, construir la tabla de equivalencias e interpretar resultados. La autora revisó la tabla de
equivalencias, ejecutó el código y redactó el paper.
