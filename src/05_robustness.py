"""Tabla 6: robustez de H2. Cada fila re-estima el logit anidado VI y reporta sigma y el
destino de los tokens que pierde un modelo cerrado al subir su precio."""
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import nl  # noqa: E402

demand = importlib.import_module("04_demand")
SOURCE = "Fuente: elaboración propia con datos de OpenRouter (openrouter.ai/rankings) y LMArena."


def tier_nests(panel):
    """Nidos alternativos: tercil de calidad dentro de cada semana."""
    p = panel.copy()
    pct = p.groupby("week")["quality"].rank(pct=True)
    p["nest_tier"] = pd.cut(pct, [0, 1 / 3, 2 / 3, 1], labels=["baja", "media", "frontera"]).astype(str)
    return p


def is_max_variant(arena_name, base_slug):
    """True si el rating viene de una variante 'max' de Arena que no es parte del nombre del modelo
    (p. ej. 'gpt-6-luna-max' o 'GPT-6 Luna (Max)' para gpt-6-luna; no 'qwen3-max' para qwen3-max)."""
    if pd.isna(arena_name):
        return False
    a = str(arena_name).lower()
    return bool(re.search(r"(^|[-_\s(])max\)?$", a)) and "max" not in str(base_slug).lower()


def without_max_ratings(panel):
    """Trata como no observada la calidad de los modelos empatados a una variante 'max' y la imputa
    con la mediana del nido en la semana (la misma regla que 02 usa para los modelos sin empate)."""
    p = panel.copy()
    if not {"arena_name", "base_slug"} <= set(p.columns):          # panel sin tabla de equivalencias
        return p, 0
    mask = [is_max_variant(a, b) for a, b in zip(p["arena_name"], p["base_slug"])]
    mask = pd.Series(mask, index=p.index)
    p.loc[mask, "quality_imputed"] = True
    obs = p["quality"].where(~p["quality_imputed"])
    p["quality"] = obs.fillna(obs.groupby([p["week"], p["nest"]]).transform("median"))
    p["quality"] = p["quality"].fillna(obs.groupby(p["week"]).transform("median"))
    p["quality"] = p["quality"].fillna(obs.median())               # semana sin ninguna calidad observada
    return p, int(p.loc[mask, "model"].nunique())


def row(label, d, res, group="nest"):
    a, s = -res.params["price"], res.params["ln_sjg"]
    out = {"especificación": label, "σ": s, "ee(σ)": res.std_errors["ln_sjg"],
           "α": a, "ee(α)": res.std_errors["price"], "obs": int(res.nobs)}
    if group == "nest":
        ns = nl.nest_summary(d, a, s)
        out["cerrado → cerrado"] = ns.loc[C.NEST_CLOSED, "desvio_mismo_nido"]
        out["cerrado → abierto"] = ns.loc[C.NEST_CLOSED, "desvio_otro_nido"]
        out["cerrado: mismo nido entre internos"] = ns.loc[C.NEST_CLOSED, "mismo_nido_entre_internos"]
    fs = res.first_stage.diagnostics["f.stat"]
    out["F mín. primera etapa"] = float(fs.min())
    return out


def main(panel=None):
    panel = panel if panel is not None else pd.read_csv(C.CLEAN / "panel.csv", parse_dates=["week"])
    rows = []
    for lam in C.LAMBDA_GRID:
        d = demand.base_sample(panel, lam=lam)
        rows.append(row(f"Mercado potencial λ = {lam}", d, nl.estimate(d)))
    d = demand.base_sample(panel, include_free=True)
    rows.append(row("Incluye variantes gratuitas", d, nl.estimate(d)))
    d = demand.base_sample(panel, drop_launch=C.LAUNCH_WEEKS_DROP)
    rows.append(row(f"Sin las primeras {C.LAUNCH_WEEKS_DROP} semanas de cada modelo", d, nl.estimate(d)))
    d = demand.base_sample(panel, observed_quality_only=True)
    rows.append(row("Solo calidad observada (sin imputar)", d, nl.estimate(d)))
    p_nomax, n_max = without_max_ratings(panel)
    d = demand.base_sample(p_nomax)
    rows.append(row(f"Sin ratings de variantes 'max' ({n_max} modelos, calidad imputada)", d, nl.estimate(d)))
    d = demand.base_sample(panel)
    for ivs in ("solo_calidad", "solo_conteos"):
        rows.append(row(f"Instrumentos: {ivs.replace('_', ' ')}", d, nl.estimate(d, ivs=ivs)))
    d["author"] = d["model"].str.split("/").str[0].astype("category").cat.codes
    rows.append(row("Errores agrupados por creador", d, nl.estimate(d, cluster="author")))
    d = demand.base_sample(tier_nests(panel), group="nest_tier")
    rows.append(row("Nidos por tercil de calidad", d, nl.estimate(d), group="nest_tier"))

    t6 = pd.DataFrame(rows)
    t6.to_csv(C.TABLES / "tabla6_robustez.csv", index=False)
    (C.TABLES / "tabla6_robustez.md").write_text(
        "Tabla 6. Robustez del logit anidado VI\n\n" + t6.round(4).to_markdown(index=False)
        + "\n\nNota: \"cerrado → cerrado\" y \"cerrado → abierto\" son la fracción promedio de los "
          "tokens que pierde un modelo cerrado al subir su precio que se va a otro cerrado o a uno "
          "abierto. Con efectos fijos de semana, λ solo mueve ln s_0t, que queda absorbido: α y σ no "
          "cambian y solo cambian elasticidades y desvíos. En la fila de nidos por tercil σ mide "
          "sustitución dentro del tercil. En la fila sin variantes 'max', la calidad de los modelos cuyo "
          "rating de Arena viene de su variante de máximo esfuerzo se reemplaza por la mediana del nido "
          "en la semana.\n" + SOURCE + "\n")
    print(t6.round(3).to_string())


if __name__ == "__main__":
    main()
