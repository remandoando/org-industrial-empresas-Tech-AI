"""Prueba del código con datos SIMULADOS de parámetros conocidos (alpha=0.4, sigma=0.6).

Sirve solo para verificar que el estimador recupera los parámetros verdaderos y que todos los
scripts corren. Sus salidas van a tests/output/, nunca a output/. NO son resultados del paper.

Uso: python tests/smoke_test.py
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

OUT = ROOT / "tests" / "output"
C.TABLES = OUT / "tables"
C.FIGURES = OUT / "figures"
C.N_BOOT = 100
for d in (C.TABLES, C.FIGURES):
    d.mkdir(parents=True, exist_ok=True)

import nl  # noqa: E402

ALPHA, SIGMA, LAM = 0.4, 0.6, 1.5


def simulate(n_models=90, n_weeks=80, seed=1):
    rng = np.random.default_rng(seed)
    models = pd.DataFrame({
        "model": [f"lab{j % 12}/modelo-{j}" for j in range(n_models)],
        "open": rng.random(n_models) < 0.45,
        "q0": rng.normal(1250, 60, n_models),
        "fe": rng.normal(0, 0.7, n_models),
        "cost": np.exp(rng.normal(0.3, 0.6, n_models)),
        "entry": rng.integers(0, n_weeks - 10, n_models),
    })
    # los modelos abiertos entran más tarde: el tamaño de cada nido cambia en el tiempo
    late = models["open"].to_numpy()
    models.loc[late, "entry"] = rng.integers(n_weeks // 3, n_weeks - 5, late.sum())
    models["exit"] = models["entry"] + rng.integers(10, 40, n_models)
    models.loc[models["open"], "cost"] *= 0.3
    weeks = pd.date_range("2025-01-06", periods=n_weeks, freq="W-MON")
    rows = []
    for t, wk in enumerate(weeks):
        act = models[(models["entry"] <= t) & (models["exit"] > t)].copy()
        act["quality"] = act["q0"] + rng.normal(0, 8, len(act))
        act["nest"] = np.where(act["open"], C.NEST_OPEN, C.NEST_CLOSED)
        # rivales cercanos en el mismo nido presionan el precio a la baja (relevancia de los VI)
        q = act["quality"].to_numpy()
        same = act["nest"].to_numpy()[:, None] == act["nest"].to_numpy()[None, :]
        close = ((np.abs(q[:, None] - q[None, :]) < C.CLOSE_QUALITY_BAND) & same).sum(1) - 1
        xi = rng.normal(0, 0.4, len(act))
        act["price"] = np.clip(act["cost"] * (1 + 3.0 / (1 + close)) + 0.8 * xi, 0.05, None)
        act["age_weeks"] = t - act["entry"]
        delta = (-5.5 + 0.004 * (act["quality"] - 1250) + 0.1 * np.log1p(act["age_weeks"])
                 + act["fe"] - ALPHA * act["price"] + xi + 0.15 * t / n_weeks)
        e = np.exp(delta / (1 - SIGMA))
        Dg = e.groupby(act["nest"]).transform("sum")
        Dsum = (e.groupby(act["nest"]).sum() ** (1 - SIGMA)).sum()
        s = e / Dg * Dg ** (1 - SIGMA) / (1 + Dsum)
        M = 1e12
        act["tokens"] = s * M
        act["week"] = wk
        act["week_total_tokens"] = M / LAM
        rows.append(act)
    p = pd.concat(rows, ignore_index=True)
    p["is_free"] = False
    p["quality_imputed"] = False
    p["ln_age"] = np.log1p(p["age_weeks"])
    return p[["week", "model", "tokens", "week_total_tokens", "is_free", "price", "quality",
              "quality_imputed", "nest", "age_weeks", "ln_age"]]


def main():
    panel = simulate()
    demand = importlib.import_module("04_demand")
    d = demand.base_sample(panel)
    res = nl.estimate(d)
    a, s = -res.params["price"], res.params["ln_sjg"]
    print(f"alpha estimado {a:.3f} (verdadero {ALPHA}); sigma estimado {s:.3f} (verdadero {SIGMA})")
    assert abs(a - ALPHA) < 0.1, "alpha no se recupera"
    assert abs(s - SIGMA) < 0.1, "sigma no se recupera"

    # la inversión de Berry reproduce las participaciones observadas
    last = d[d["week"] == d["week"].max()]
    assert abs(nl.group_elasticity(last, a, s, C.NEST_CLOSED, pct=0.0)) < 1e-10
    # desvíos de cada fila suman 1 con el bien externo
    gd = nl.group_diversion(last, a, s)
    assert np.allclose(gd[["desvio_mismo_nido", "desvio_otro_nido", "desvio_externo"]].sum(axis=1), 1)

    importlib.import_module("03_descriptives").main(panel)
    demand.main(panel)
    importlib.import_module("05_robustness").main(panel)
    ev = importlib.import_module("06_event_study")
    ev.main(panel)
    t7 = pd.read_csv(C.TABLES / "tabla7_eventos.csv")
    base = t7.iloc[0]
    print(f"eventos: beta post {base['β post (mismo nido)']:.3f} (debe ser < 0 con sigma = 0.6)")
    assert base["β post (mismo nido)"] < 0, "el estudio de eventos no detecta el anidamiento simulado"
    print("OK: todo corre y el estimador recupera los parámetros simulados.")


if __name__ == "__main__":
    main()
