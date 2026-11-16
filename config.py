"""Parámetros del proyecto. Todo lo que cambia una tabla vive aquí."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_env(path=ROOT / ".env"):
    """Carga las llaves de .env (OPENROUTER_API_KEY, HF_TOKEN) en el entorno.
    No sobrescribe variables que ya estén definidas en la terminal."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
RAW = ROOT / "data" / "raw"
CROSSWALK = ROOT / "data" / "crosswalk"
MANUAL = ROOT / "data" / "manual"
CLEAN = ROOT / "data" / "clean"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
for _p in (RAW, CROSSWALK, MANUAL, CLEAN, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20261115

# Periodo. El dataset público de OpenRouter empieza el 2025-01-01.
START_DATE = "2025-01-01"
END_DATE = None          # None = última semana completa disponible

# Precio mezclado (USD por millón de tokens) = W_PROMPT*entrada + (1-W_PROMPT)*salida
W_PROMPT = 0.75

# Mercado potencial: M_t = LAMBDA * tokens totales de la semana (top 50 + "other")
LAMBDA_BASE = 1.5
LAMBDA_GRID = [1.2, 1.5, 2.0, 3.0]

# Instrumentos de diferenciación (Gandhi y Houde, 2019)
CLOSE_QUALITY_BAND = 50   # puntos Arena: rivales "cercanos" en calidad

# Robustez
LAUNCH_WEEKS_DROP = 4     # semanas iniciales de cada modelo que se excluyen
N_BOOT = 500              # extracciones para intervalos de elasticidades y desvíos
TOP_N_HEATMAP = 15

# Estudio de eventos (06_event_study.py): entrada de modelos al top 50
EVENT_TOP_RANK = 10       # el entrante debe llegar a este lugar (o mejor) en sus primeras 4 semanas
EVENT_PRE, EVENT_POST = 4, 8   # semanas antes y después de la entrada
EVENT_EXCLUSION = 1       # descarta eventos con otro evento a <= estas semanas
EVENT_MIN_PRE_WEEKS = 3   # semanas previas mínimas para contar como modelo establecido

# Competencia entre proveedores de un mismo modelo abierto (07_provider_cournot.py)
N_CAP = 5                 # el número de proveedores se agrupa como 1, 2, ..., N_CAP o más
MARKET_SIZE_WEEKS = 8     # semanas recientes para medir el tamaño de mercado (tokens)
N_BOOT_BR = 500           # remuestreos para los umbrales de entrada

# Nombres de nidos
NEST_CLOSED, NEST_OPEN = "cerrado", "abierto"
