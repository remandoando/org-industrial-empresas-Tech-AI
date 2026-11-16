"""H8: ¿competencia monopolística o costos hundidos endógenos (Sutton)?

Competencia monopolística con entrada libre (Krugman, 1980): el número de variedades crece en
proporción al tamaño de mercado (elasticidad ≈ 1) y la concentración cae al crecer el mercado.
Costos hundidos endógenos (Sutton, 1991): la calidad se compra con gasto fijo (entrenamiento), el
número de empresas crece poco (elasticidad < 1) y la concentración no baja de un piso.

Es un solo mercado observado en el tiempo, no un corte de muchos mercados como en Sutton: se lee
como prueba de consistencia. Para evitar regresiones espurias entre dos series con tendencia se
reportan diferencias de 4 semanas con errores HAC (Newey-West), además de niveles con tendencia.

Entradas: data/raw/archive_prices_*.csv (catálogo semanal del Internet Archive),
          data/raw/openrouter_models_*.json, data/raw/openrouter_rankings_week_*.csv, data/clean/panel.csv
Salidas:  output/tables/tabla11_sutton.csv/.md, serie_variedades.csv
          output/figures/figura7_variedades_vs_mercado.png, figura8_concentracion_vs_mercado.png,
          figura9_rango_participacion.png
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as CFG  # noqa: E402

SOURCE = "Fuente: elaboración propia con datos de OpenRouter e Internet Archive."
DIFF = 4          # semanas de la diferencia
HAC_LAGS = 4


def latest(pattern):
    files = sorted(glob.glob(str(CFG.RAW / pattern)))
    if not files:
        raise FileNotFoundError(f"Falta {pattern} en data/raw/")
    return files[-1]


def catalog_series(archive=None, models=None, panel=None):
    """Número de modelos (variantes base) y de creadores disponibles por semana y segmento."""
    a = archive if archive is not None else pd.read_csv(latest("archive_prices_*.csv"),
                                                        parse_dates=["snapshot"])
    a = a.dropna(subset=["id"]).copy()
    a["base"] = a["id"].str.split(":").str[0]
    a["author"] = a["base"].str.split("/").str[0]
    a["week"] = a["snapshot"].dt.to_period("W").dt.start_time
    # apertura: hugging_face_id del catálogo actual; si el modelo ya no existe, el nido del panel
    cur = models if models is not None else json.loads(Path(latest("openrouter_models_*.json")).read_text())
    hf = {m["id"].split(":")[0]: bool(m.get("hugging_face_id")) for m in cur}
    p = panel if panel is not None else pd.read_csv(CFG.CLEAN / "panel.csv")
    nest = p.drop_duplicates("base_slug").set_index("base_slug")["nest"].eq(CFG.NEST_OPEN).to_dict()
    canon = a.drop_duplicates("base").set_index("base")["canonical_slug"].to_dict()
    def is_open(b):
        if b in hf:
            return hf[b]
        c = canon.get(b)
        return nest.get(c, nest.get(b, np.nan))
    a["open"] = a["base"].map(is_open)
    a = a.drop_duplicates(["week", "base"])
    rows = []
    for w, g in a.groupby("week"):
        rows.append({"week": w, "modelos": g["base"].nunique(), "creadores": g["author"].nunique(),
                     "modelos_abiertos": (g["open"] == True).sum(),  # noqa: E712
                     "modelos_cerrados": (g["open"] == False).sum(),  # noqa: E712
                     "apertura_desconocida": g["open"].isna().sum()})
    return pd.DataFrame(rows)


def market_series(rankings=None, panel=None):
    r = rankings if rankings is not None else pd.read_csv(latest("openrouter_rankings_week_*.csv"))
    r = r.copy()
    r["week"] = pd.to_datetime(r["date"]).dt.to_period("W").dt.start_time
    r["tokens"] = pd.to_numeric(r["total_tokens"])
    S = r.groupby("week")["tokens"].sum().rename("S")
    p = panel if panel is not None else pd.read_csv(CFG.CLEAN / "panel.csv", parse_dates=["week"])
    p = p.copy()
    p["author"] = p["model"].str.split("/").str[0]

    def hhi(df):
        sh = df.groupby("author")["tokens"].sum()
        sh = 100 * sh / sh.sum()
        return (sh ** 2).sum()
    out = pd.DataFrame({"S": S})
    out["hhi_total"] = p.groupby("week").apply(hhi, include_groups=False)
    for nest in (CFG.NEST_OPEN, CFG.NEST_CLOSED):
        out[f"hhi_{nest}"] = p[p["nest"] == nest].groupby("week").apply(hhi, include_groups=False)
    return out.reset_index()


def hac_ols(y, X):
    X = sm.add_constant(X)
    ok = y.notna() & X.notna().all(axis=1)
    return sm.OLS(y[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def regressions(df):
    d = df.sort_values("week").reset_index(drop=True)
    d["lnS"] = np.log(d["S"])
    d["t"] = np.arange(len(d))
    outcomes = {"Modelos disponibles": "modelos", "Creadores disponibles": "creadores",
                "Modelos abiertos": "modelos_abiertos", "Modelos cerrados": "modelos_cerrados",
                "IHH por creador (total)": "hhi_total",
                f"IHH dentro de {CFG.NEST_OPEN}s": f"hhi_{CFG.NEST_OPEN}",
                f"IHH dentro de {CFG.NEST_CLOSED}s": f"hhi_{CFG.NEST_CLOSED}"}
    rows = []
    for label, col in outcomes.items():
        y = np.log(d[col].where(d[col] > 0))
        dy, dS = y.diff(DIFF), d["lnS"].diff(DIFF)
        r_diff = hac_ols(dy, dS.rename("dlnS"))
        r_lvl = hac_ols(y, d[["lnS", "t"]])
        row = {"variable (en ln)": label, "semanas": int(y.notna().sum()),
               f"elasticidad Δ{DIFF} sem": r_diff.params.get("dlnS", np.nan),
               "ee Δ": r_diff.bse.get("dlnS", np.nan),
               "elasticidad niveles + tendencia": r_lvl.params.get("lnS", np.nan),
               "ee niveles": r_lvl.bse.get("lnS", np.nan)}
        if col.startswith("modelos") or col == "creadores":
            b, se = row[f"elasticidad Δ{DIFF} sem"], row["ee Δ"]
            row["p H0: elasticidad = 1"] = float(2 * (1 - _ncdf(abs((b - 1) / se)))) if se > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), d


def _ncdf(z):
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))


def rank_size(panel=None):
    """Pendiente rango-participación (tipo Zipf) en las últimas 8 semanas, participación en el top 50."""
    p = panel if panel is not None else pd.read_csv(CFG.CLEAN / "panel.csv", parse_dates=["week"])
    last = p[p["week"] > p["week"].max() - pd.Timedelta(weeks=8)]
    sh = last.groupby("model")["tokens"].sum().sort_values(ascending=False)
    sh = sh / sh.sum()
    rank = np.arange(1, len(sh) + 1)
    fit = sm.OLS(np.log(sh.to_numpy()), sm.add_constant(np.log(rank))).fit(cov_type="HC1")
    return sh, rank, fit


def figures(d, sh, rank, fit):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, lab in (("modelos", "Todos"), ("modelos_abiertos", "Abiertos"),
                     ("modelos_cerrados", "Cerrados")):
        ok = d[col] > 0
        ax.scatter(d.loc[ok, "lnS"], np.log(d.loc[ok, col]), s=12, alpha=0.7, label=lab)
    x0 = d["lnS"].min()
    y0 = np.log(d["modelos"].iloc[d["lnS"].idxmin()])
    xs = np.linspace(d["lnS"].min(), d["lnS"].max(), 10)
    ax.plot(xs, y0 + (xs - x0), color="grey", ls="--", label="Pendiente 1 desde la primera semana (competencia monopolística)")
    ax.set_xlabel("ln tokens totales por semana (tamaño de mercado)")
    ax.set_ylabel("ln número de modelos disponibles")
    ax.set_title("Figura 7. Variedades disponibles y tamaño de mercado", fontsize=10)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, SOURCE + " Un punto = una semana con captura del catálogo.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CFG.FIGURES / "figura7_variedades_vs_mercado.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, lab in (("hhi_total", "Todos"), (f"hhi_{CFG.NEST_OPEN}", "Dentro de abiertos"),
                     (f"hhi_{CFG.NEST_CLOSED}", "Dentro de cerrados")):
        ax.scatter(d["lnS"], d[col], s=12, alpha=0.7, label=lab)
    ax.set_xlabel("ln tokens totales por semana (tamaño de mercado)")
    ax.set_ylabel("IHH por creador (0–10,000)")
    ax.set_title("Figura 8. Concentración y tamaño de mercado", fontsize=10)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, "Fuente: elaboración propia con datos de OpenRouter (top 50).", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CFG.FIGURES / "figura8_concentracion_vs_mercado.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(np.log(rank), np.log(sh.to_numpy()), s=12, color="#1f4e79")
    ax.plot(np.log(rank), fit.params[0] + fit.params[1] * np.log(rank), color="#c0504d",
            label=f"Pendiente = {fit.params[1]:.2f} (e.e. {fit.bse[1]:.2f})")
    ax.set_xlabel("ln rango del modelo")
    ax.set_ylabel("ln participación en tokens del top 50")
    ax.set_title("Figura 9. Rango y participación, últimas 8 semanas", fontsize=10)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, "Fuente: elaboración propia con datos de OpenRouter (top 50).", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CFG.FIGURES / "figura9_rango_participacion.png", dpi=200)
    plt.close(fig)


def main(archive=None, models=None, rankings=None, panel=None):
    cat = catalog_series(archive, models, panel)
    mk = market_series(rankings, panel)
    df = cat.merge(mk, on="week", how="inner")
    df.to_csv(CFG.TABLES / "serie_variedades.csv", index=False)
    if len(df) < 20:
        print(f"Solo {len(df)} semanas con catálogo y tokens; muy pocas para estimar.")
        return
    t11, d = regressions(df)
    sh, rank, fit = rank_size(panel)
    t11.to_csv(CFG.TABLES / "tabla11_sutton.csv", index=False)
    unk = 100 * df["apertura_desconocida"].sum() / df["modelos"].sum()
    (CFG.TABLES / "tabla11_sutton.md").write_text(
        "Tabla 11. Variedades, concentración y tamaño de mercado (elasticidades respecto a ln tokens "
        "totales por semana)\n\n" + t11.round(3).to_markdown(index=False)
        + f"\n\nNota: {len(df)} semanas con captura del catálogo. Diferencias de {DIFF} semanas y niveles "
          f"con tendencia lineal; errores HAC (Newey-West, {HAC_LAGS} rezagos). Competencia monopolística "
          "predice elasticidad ≈ 1 en el número de modelos y elasticidad negativa del IHH; costos hundidos "
          "endógenos (Sutton, 1991) predicen elasticidad < 1 y un IHH que no baja. "
          f"Apertura desconocida: {unk:.1f}% de los modelos-semana. Pendiente rango-participación de las "
          f"últimas 8 semanas: {fit.params[1]:.2f} (e.e. {fit.bse[1]:.2f}).\n{SOURCE}\n")
    figures(d, sh, rank, fit)
    print(t11.round(3).to_string())
    print(f"Pendiente rango-participación: {fit.params[1]:.2f} (e.e. {fit.bse[1]:.2f})")


if __name__ == "__main__":
    main()
