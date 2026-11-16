"""Prueba de 02_build_panel.py con archivos crudos FALSOS que imitan el formato de las APIs.
Escribe en tests/output/raw_fake/; no toca data/. Uso: python tests/test_build_panel.py"""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import config as C  # noqa: E402

base = ROOT / "tests" / "output" / "build_fake"
for name in ("RAW", "CROSSWALK", "MANUAL", "CLEAN", "TABLES"):
    setattr(C, name, base / name.lower())
    getattr(C, name).mkdir(parents=True, exist_ok=True)

# tokens semanales con formato de /datasets/rankings-daily?period=week
rows = []
for d in ["2025-03-03", "2025-03-10", "2025-03-17"]:
    rows += [{"date": d, "model_permaslug": "openai/gpt-x-2025-01-01", "total_tokens": "5000"},
             {"date": d, "model_permaslug": "meta-llama/llama-z-70b", "total_tokens": "3000"},
             {"date": d, "model_permaslug": "meta-llama/llama-z-70b:free", "total_tokens": "800"},
             {"date": d, "model_permaslug": "other", "total_tokens": "2000"}]
pd.DataFrame(rows).to_csv(C.RAW / "openrouter_rankings_week_2026-11-15.csv", index=False)
models = [
    {"id": "openai/gpt-x", "canonical_slug": "openai/gpt-x-2025-01-01",
     "pricing": {"prompt": "0.000002", "completion": "0.000008"}, "hugging_face_id": ""},
    {"id": "meta-llama/llama-z-70b", "canonical_slug": "meta-llama/llama-z-70b",
     "pricing": {"prompt": "0.0000002", "completion": "0.0000004"},
     "hugging_face_id": "meta-llama/Llama-Z-70B"},
]
(C.RAW / "openrouter_models_2026-11-15.json").write_text(json.dumps(models))
# una captura histórica: gpt-x más caro en marzo de 2025
pd.DataFrame([{"snapshot": "2025-03-01", "id": "openai/gpt-x", "canonical_slug": "openai/gpt-x-2025-01-01",
               "prompt": "0.000004", "completion": "0.000012"}]).to_csv(
    C.RAW / "archive_prices_2026-11-15.csv", index=False)
pd.DataFrame([
    {"model_name": "gpt-x", "rating": 1300, "leaderboard_publish_date": "2025-02-20",
     "category": "overall", "license": "Proprietary"},
    {"model_name": "llama-z-70b", "rating": 1200, "leaderboard_publish_date": "2025-02-20",
     "category": "overall", "license": "Llama"},
]).to_csv(C.RAW / "lmarena_text_2026-11-15.csv", index=False)

build = importlib.import_module("02_build_panel")
build.main()
p = pd.read_csv(C.CLEAN / "panel.csv")
print(p[["week", "model", "tokens", "week_total_tokens", "price", "price_source", "quality", "nest"]])
gx = p[p["model"].str.startswith("openai")]
assert (gx["price"].round(4) == round(1e6 * (0.75 * 4e-6 + 0.25 * 12e-6), 4)).all(), "precio histórico"
assert (p.loc[p["model"].str.startswith("meta") & ~p["is_free"], "nest"] == C.NEST_OPEN).all()
assert (p["week_total_tokens"] == 10800).all()
print("OK: el panel se construye con el formato esperado.")
