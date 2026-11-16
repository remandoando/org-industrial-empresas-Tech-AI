"""Construye el panel modelo-semana desde data/raw/ y deja un registro de limpieza.

Salidas:
  data/clean/panel.csv              panel listo para estimar (todas las variantes, con banderas)
  data/crosswalk/crosswalk_auto.csv propuesta automática OpenRouter <-> LMArena para revisar a mano
  output/tables/limpieza.csv        cuántas observaciones elimina cada paso
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

LOG = []


def log(step, n):
    LOG.append({"paso": step, "observaciones": n})
    print(f"  {step}: {n:,}")


def latest(pattern):
    files = sorted(glob.glob(str(C.RAW / pattern)))
    if not files:
        raise FileNotFoundError(f"No hay {pattern} en data/raw/. Corre 01_download.py.")
    return files[-1]


def norm(name):
    """Normaliza nombres para empatar fuentes: sin autor, fechas, sufijos ni puntuación."""
    n = str(name).lower().split("/")[-1].split(":")[0]
    n = re.sub(r"-?20\d{2}-?\d{2}-?\d{2}", "", n)
    n = re.sub(r"-(instruct|chat|preview|latest|exp)\b", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


# ----------------------------------------------------------------------------- tokens
def load_tokens():
    r = pd.read_csv(latest("openrouter_rankings_week_*.csv"))
    r["tokens"] = pd.to_numeric(r["total_tokens"])
    r["week"] = pd.to_datetime(r["date"]).dt.to_period("W").dt.start_time
    tot = r.groupby("week")["tokens"].sum().rename("week_total_tokens")
    r = r[r["model_permaslug"] != "other"].rename(columns={"model_permaslug": "model"})
    r = r.groupby(["week", "model"], as_index=False)["tokens"].sum().merge(tot, on="week")
    r["is_free"] = r["model"].str.contains(":free")
    r["base_slug"] = r["model"].str.split(":").str[0]
    log("filas modelo-semana en el top 50 (sin 'other')", len(r))
    return r


# ----------------------------------------------------------------------------- precios
def load_prices():
    """Precio por semana: capturas del Internet Archive + catálogo actual + archivo manual.
    Gana la fuente manual, luego la captura más reciente anterior a la semana."""
    frames = []
    arch = sorted(glob.glob(str(C.RAW / "archive_prices_*.csv")))
    if arch:
        a = pd.read_csv(arch[-1], parse_dates=["snapshot"])
        frames.append(a.assign(source="archive"))
    cur = json.loads(Path(latest("openrouter_models_*.json")).read_text())
    cur_date = pd.Timestamp(re.search(r"(\d{4}-\d{2}-\d{2})", latest("openrouter_models_*.json")).group(1))
    frames.append(pd.DataFrame([{"snapshot": cur_date, "id": m["id"],
                                 "canonical_slug": m.get("canonical_slug"),
                                 "prompt": (m.get("pricing") or {}).get("prompt"),
                                 "completion": (m.get("pricing") or {}).get("completion")}
                                for m in cur]).assign(source="actual"))
    man = C.MANUAL / "prices_manual.csv"
    if man.exists():
        m = pd.read_csv(man, parse_dates=["date"])
        frames.append(pd.DataFrame({"snapshot": m["date"], "id": m["slug"], "canonical_slug": m["slug"],
                                    "prompt": m["prompt_usd_per_m"] / 1e6,
                                    "completion": m["completion_usd_per_m"] / 1e6,
                                    "source": "manual"}))
    p = pd.concat(frames, ignore_index=True)
    for c in ("prompt", "completion"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["prompt", "completion"])
    p = p[(p["prompt"] >= 0) & (p["completion"] >= 0)]
    p["price"] = 1e6 * (C.W_PROMPT * p["prompt"] + (1 - C.W_PROMPT) * p["completion"])
    # una fila por clave de empate (canonical_slug e id)
    long = pd.concat([p.assign(key=p["canonical_slug"]), p.assign(key=p["id"])])
    long = long.dropna(subset=["key"]).drop_duplicates(["key", "snapshot", "source"])
    long["prio"] = long["source"].map({"manual": 0, "archive": 1, "actual": 2})
    # merge_asof toma la última fila empatada: se ordena para que "manual" quede al final
    long = long.sort_values(["snapshot", "prio"], ascending=[True, False], kind="stable")
    return long[["key", "snapshot", "price", "source"]]


def attach_prices(panel, prices):
    """Para cada modelo-semana toma el último precio observado en o antes de esa semana.
    El precio del catálogo actual solo se usa hacia atrás si no hubo captura previa (bandera)."""
    out = []
    for key_col in ("model", "base_slug"):
        left = panel[["week", key_col]].drop_duplicates().rename(columns={key_col: "key"})
        left = left.sort_values("week").astype({"key": object})
        m = pd.merge_asof(left, prices.astype({"key": object}), left_on="week", right_on="snapshot",
                          by="key", direction="backward")
        out.append(m.rename(columns={"key": key_col}).set_index(["week", key_col]))
    a = panel.join(out[0][["price", "source"]], on=["week", "model"])
    b = panel.join(out[1][["price", "source"]], on=["week", "base_slug"])
    a["price"] = a["price"].fillna(b["price"])
    a["price_source"] = a["source"].fillna(b["source"])
    # sin precio previo: usar el actual hacia atrás y marcarlo
    cur = prices[prices["source"] == "actual"].drop_duplicates("key").set_index("key")["price"]
    miss = a["price"].isna()
    a.loc[miss, "price"] = a.loc[miss, "model"].map(cur).fillna(a.loc[miss, "base_slug"].map(cur))
    a.loc[miss & a["price"].notna(), "price_source"] = "actual_hacia_atras"
    return a.drop(columns=["source"])


# ----------------------------------------------------------------------------- calidad y apertura
def load_arena():
    f = latest("lmarena_*.csv")
    a = pd.read_csv(f)
    cols = {c.lower(): c for c in a.columns}
    pick = lambda *opts: next((cols[o] for o in opts if o in cols), None)  # noqa: E731
    name = pick("model_name", "model", "name")
    rating = pick("rating", "score", "arena_score", "elo")
    date = pick("leaderboard_publish_date", "publish_date", "date")
    cat = pick("category")
    lic = pick("license")
    if not all([name, rating, date]):
        raise KeyError(f"Revisa columnas de LMArena: {list(a.columns)}")
    if cat:
        a = a[a[cat].astype(str).str.lower().isin(["overall", "text", "full"])]
    a = a.rename(columns={name: "arena_name", rating: "quality", date: "arena_date"})
    a["arena_date"] = pd.to_datetime(a["arena_date"], errors="coerce")
    a["arena_open"] = (~a[lic].astype(str).str.lower().str.contains("propriet")) if lic else np.nan
    a["norm"] = a["arena_name"].map(norm)
    return a[["arena_name", "norm", "arena_date", "quality", "arena_open"]].dropna(subset=["arena_date"])


def build_crosswalk(panel, arena):
    """Propuesta automática por nombre normalizado. Revísala y guárdala como crosswalk.csv."""
    cur = json.loads(Path(latest("openrouter_models_*.json")).read_text())
    hf = {}
    for m in cur:
        for k in (m.get("id"), m.get("canonical_slug")):
            if k:
                hf[k] = m.get("hugging_face_id") or ""
    models = panel[["model", "base_slug"]].drop_duplicates()
    an = arena.drop_duplicates("norm").set_index("norm")["arena_name"]
    models["arena_name"] = models["base_slug"].map(norm).map(an)
    models["hugging_face_id"] = models["model"].map(hf).fillna(models["base_slug"].map(hf)).fillna("")
    models["open_weights"] = np.where(models["hugging_face_id"] != "", 1, np.nan)
    models.to_csv(C.CROSSWALK / "crosswalk_auto.csv", index=False)
    manual = C.CROSSWALK / "crosswalk.csv"
    if manual.exists():
        print("  usando crosswalk.csv (revisado a mano)")
        return pd.read_csv(manual)
    print("  AVISO: no hay crosswalk.csv; se usa la propuesta automática sin revisar")
    return models


def attach_quality_open(panel, arena, cw):
    p = panel.merge(cw[["model", "arena_name", "open_weights"]], on="model", how="left")
    p["arena_key"] = p["arena_name"].map(lambda x: norm(x) if pd.notna(x) and str(x).strip() else np.nan)
    # Arena usa dos formatos ('claude-opus-5.5-high' y 'Claude Opus 5.5 (High)'): se empata por nombre
    # normalizado y se promedian los ratings de ambos formatos en la misma fecha
    ar = (arena.groupby(["norm", "arena_date"], as_index=False)
          .agg(quality=("quality", "mean"), arena_open=("arena_open", "max"))
          .rename(columns={"norm": "arena_key"}).sort_values("arena_date"))
    q = pd.merge_asof(p.sort_values("week")[["week", "model", "arena_key"]].dropna()
                      .astype({"arena_key": object}),
                      ar.astype({"arena_key": object}),
                      left_on="week", right_on="arena_date", by="arena_key", direction="backward")
    p = p.merge(q[["week", "model", "quality", "arena_open"]], on=["week", "model"], how="left")
    p = p.drop(columns=["arena_key"])
    # apertura: crosswalk/hugging_face_id; si falta, licencia de Arena; si falta, cerrado
    p["open_weights"] = p["open_weights"].fillna(p["arena_open"].astype(float)).fillna(0).astype(int)
    p["nest"] = np.where(p["open_weights"] == 1, C.NEST_OPEN, C.NEST_CLOSED)
    p["quality_imputed"] = p["quality"].isna()
    p["quality"] = p["quality"].fillna(p.groupby(["week", "nest"])["quality"].transform("median"))
    p["quality"] = p["quality"].fillna(p.groupby("week")["quality"].transform("median"))
    return p.drop(columns=["arena_open"])


# ----------------------------------------------------------------------------- principal
def main():
    print("Construyendo panel")
    panel = load_tokens()
    panel = attach_prices(panel, load_prices())
    arena = load_arena()
    cw = build_crosswalk(panel, arena)
    panel = attach_quality_open(panel, arena, cw)

    panel["first_week"] = panel.groupby("model")["week"].transform("min")
    panel["age_weeks"] = (panel["week"] - panel["first_week"]).dt.days // 7
    panel["ln_age"] = np.log1p(panel["age_weeks"])

    # las variantes ":free" cuestan cero por definición
    panel.loc[panel["is_free"], ["price", "price_source"]] = [0.0, "gratis"]

    n0 = len(panel)
    panel = panel[panel["price"].notna()]
    log("con precio observado o reconstruido", len(panel))
    panel = panel[~((panel["price"] <= 0) & ~panel["is_free"])]
    log("sin precios cero en variantes de pago", len(panel))
    panel = panel[panel["quality"].notna()]
    log("con calidad (observada o imputada)", len(panel))
    print(f"  eliminadas en total: {n0 - len(panel):,}")

    panel.to_csv(C.CLEAN / "panel.csv", index=False)
    pd.DataFrame(LOG).to_csv(C.TABLES / "limpieza.csv", index=False)
    summary = {
        "semanas": panel["week"].nunique(), "modelos": panel["model"].nunique(),
        "pct_precio_hacia_atras": round(100 * (panel["price_source"] == "actual_hacia_atras").mean(), 1),
        "pct_calidad_imputada": round(100 * panel["quality_imputed"].mean(), 1),
    }
    print("  resumen:", summary)
    pd.Series(summary).to_csv(C.TABLES / "cobertura.csv", header=["valor"])


if __name__ == "__main__":
    main()
