"""Corre todo el proyecto de principio a fin.

    python run_all.py              # descarga, construye, estima
    python run_all.py --no-download  # usa lo que ya está en data/raw/
"""
import importlib
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import config as C  # noqa: E402

random.seed(C.SEED)
np.random.seed(C.SEED)

if "--no-download" not in sys.argv:
    dl = importlib.import_module("01_download")
    dl.download_rankings()
    dl.download_models_and_endpoints()
    dl.download_archive_prices()
    dl.download_arena()

for step in ("02_build_panel", "03_descriptives", "04_demand", "05_robustness", "06_event_study",
             "07_provider_cournot", "08_sutton_monopolistic"):
    print(f"\n=== {step} ===")
    importlib.import_module(step).main()
print("\nListo: tablas en output/tables/, figuras en output/figures/.")
