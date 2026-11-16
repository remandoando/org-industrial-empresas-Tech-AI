"""Prueba de 07_provider_cournot.py con mercados SIMULADOS bajo Cournot (no son resultados).
Cada modelo abierto tiene demanda S; entran proveedores mientras la ganancia de Cournot cubre un
costo fijo; el precio es c + (a − c)/(N + 1). La prueba exige que el precio siga bajando después de
N = 2 (se rechaza Bertrand) y que corran los umbrales de entrada.  Uso: python tests/test_provider_cournot.py
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import config as C  # noqa: E402

OUT = ROOT / "tests" / "output"
C.TABLES, C.FIGURES = OUT / "tables", OUT / "figures"
C.N_BOOT_BR = 50
for d in (C.TABLES, C.FIGURES):
    d.mkdir(parents=True, exist_ok=True)


def simulate(n_markets=150, seed=3):
    rng = np.random.default_rng(seed)
    eps, rk, pn = [], [], []
    weeks = pd.date_range("2026-09-07", periods=C.MARKET_SIZE_WEEKS, freq="W-MON")
    for j in range(n_markets):
        slug = f"lab{j % 9}/open-{j}-{int(rng.choice([8, 32, 70, 235]))}b"
        S = np.exp(rng.normal(23, 1.2))                  # tokens por semana
        c = 0.05 * float(slug.split("-")[-1][:-1]) ** 0.6 * np.exp(rng.normal(0, 0.15))
        a = c * 6
        N = int(np.clip(np.floor(np.sqrt(S / np.exp(21.5)) * np.exp(rng.normal(0, 0.3)) - 1), 1, 9))
        P = c + (a - c) / (N + 1)
        for i in range(N):
            p_i = P * np.exp(rng.normal(0, 0.05))
            eps.append({"id": slug, "canonical_slug": slug, "hugging_face_id": slug,
                        "model_context_length": 131072, "provider_name": f"prov{i}", "tag": "x",
                        "quantization": "fp8", "service_tier": np.nan, "status": 0,
                        "prompt": p_i / 1e6, "completion": p_i / 1e6})
        for w in weeks:
            rk.append({"date": w.date().isoformat(), "model_permaslug": slug,
                       "total_tokens": str(int(S * np.exp(rng.normal(0, 0.1))))})
        pn.append({"week": weeks[-1], "base_slug": slug, "quality": 1300 + rng.normal(0, 40)})
    return pd.DataFrame(eps), pd.DataFrame(rk), pd.DataFrame(pn)


mod = importlib.import_module("07_provider_cournot")
e, r, p = simulate()
m = mod.build_markets(e, r, p)
mod.main(m)
t8 = pd.read_csv(C.TABLES / "tabla8_precios_vs_N.csv")
row = t8[(t8["precio"] == "mínimo") & (t8["especificación"] == "(1) Base")].iloc[0]
print(f"p-valor H0 Bertrand: {row['p H0 Bertrand (N≥3 = N=2)']:.4f}")
assert row["p H0 Bertrand (N≥3 = N=2)"] < 0.05, "no detecta el patrón de Cournot simulado"
assert row[f"N={C.N_CAP}+"] < row["N=2"], "el precio debería seguir bajando con más proveedores"
assert (C.TABLES / "tabla9_umbrales_entrada.csv").exists()
print("OK: 07_provider_cournot corre y detecta el patrón de Cournot simulado.")
