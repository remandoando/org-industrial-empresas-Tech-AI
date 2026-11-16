"""Estimación principal de H2.

Tabla 2: logit MCO, logit VI, logit anidado MCO y logit anidado VI (especificación principal).
Tabla 3: razones de desvío por nido, con intervalos al 95% (solo dependen de σ).
(La Tabla 4 del monopolista hipotético ya no se genera: requiere un precio identificado.)
Figura 3: mapa de calor de razones de desvío entre los 15 modelos con más tokens.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import nl  # noqa: E402

SOURCE = "Fuente: elaboración propia con datos de OpenRouter (openrouter.ai/rankings) y LMArena."
NOTE = ("Efectos fijos de modelo y de semana. Errores estándar agrupados por modelo entre paréntesis. "
        "Instrumentos: diferenciación en calidad y número de rivales en el nido (Gandhi y Houde, 2019).")


def base_sample(panel, lam=C.LAMBDA_BASE, include_free=False, drop_launch=0, group="nest",
                observed_quality_only=False):
    p = panel.copy()
    if not include_free:
        p = p[~p["is_free"]]
    if drop_launch:
        p = p[p["age_weeks"] >= drop_launch]
    if observed_quality_only:
        p = p[~p["quality_imputed"]]
    p = nl.add_shares(p, lam, group)
    return nl.add_diff_ivs(p, C.CLOSE_QUALITY_BAND, group)


def table2(d):
    specs = {"(1) Logit MCO": dict(nested=False, iv=False),
             "(2) Logit VI": dict(nested=False, iv=True),
             "(3) Anidado MCO": dict(nested=True, iv=False),
             "(4) Anidado VI": dict(nested=True, iv=True)}
    cols, fits = {}, {}
    for name, kw in specs.items():
        res = nl.estimate(d, **kw)
        s = nl.summarize(res, **kw)
        fits[name] = res
        col = {"Precio (−α)": f"{-s['alpha']:.4f}", "  ee": f"({s['se_alpha']:.4f})"}
        if kw["nested"]:
            col["ln s_j|g (σ)"] = f"{s['sigma']:.4f}"
            col["  ee "] = f"({s['se_sigma']:.4f})"
            col["p-valor H0: σ ≤ 0"] = f"{s['p_sigma_pos']:.4f}"
        for k, v in s.items():
            if k.startswith("F_primera_etapa"):
                col[f"F primera etapa ({k.replace('F_primera_etapa_', '')})"] = f"{v:.1f}"
        if "sobreid_p" in s:
            col["p-valor sobreidentificación"] = f"{s['sobreid_p']:.3f}"
        col["Observaciones"] = f"{s['n_obs']:,}"
        cols[name] = col
    t2 = pd.DataFrame(cols)
    order = ["Precio (−α)", "  ee", "ln s_j|g (σ)", "  ee ", "p-valor H0: σ ≤ 0",
             "F primera etapa (price)", "F primera etapa (ln_sjg)", "p-valor sobreidentificación",
             "Observaciones"]
    t2 = t2.reindex([r for r in order if r in t2.index])
    t2.to_csv(C.TABLES / "tabla2_demanda.csv")
    (C.TABLES / "tabla2_demanda.md").write_text(
        "Tabla 2. Estimación de demanda. Variable dependiente: ln s_jt − ln s_0t\n\n"
        + t2.fillna("").to_markdown() + f"\n\nNota: {NOTE}\n{SOURCE}\n")
    return fits["(4) Anidado VI"]


def table3(d, res):
    a, s = -res.params["price"], res.params["ln_sjg"]
    point = nl.nest_summary(d, a, s)
    ci, n_ok = nl.bootstrap_nest_summary(d, res, C.N_BOOT, C.SEED)
    cols = ["desvio_mismo_nido", "desvio_otro_nido", "desvio_externo", "mismo_nido_entre_internos"]
    share = d.groupby("nest")["tokens"].sum() / d["tokens"].sum()   # referencia del criterio (b) de H2
    rows = []
    for nest in point.index:
        for v in cols:
            lo, hi = ci.loc[(nest, v)]
            rows.append({"nido de origen": nest, "medida": v, "estimación": point.loc[nest, v],
                         "IC95 inf": lo, "IC95 sup": hi})
        rows.append({"nido de origen": nest, "medida": "participacion_del_nido (tokens)",
                     "estimación": share.get(nest, np.nan), "IC95 inf": np.nan, "IC95 sup": np.nan})
    t3 = pd.DataFrame(rows)
    t3.to_csv(C.TABLES / "tabla3_elasticidades_desvio.csv", index=False)
    (C.TABLES / "tabla3_elasticidades_desvio.md").write_text(
        "Tabla 3. Destino de los tokens perdidos ante un alza de precio (razones de desvío), "
        "promedio ponderado por tokens\n\n" + t3.round(3).to_markdown(index=False)
        + f"\n\nNota: intervalos por simulación paramétrica de σ ({n_ok} extracciones en (0, 1), "
          f"semilla {C.SEED}); los desvíos no dependen de α. No se reportan elasticidades porque el "
          "coeficiente de precio no está identificado (Tabla 2). Los tres desvíos suman 1. "
          "'mismo_nido_entre_internos' es la fracción que se queda en el mismo nido entre los tokens que "
          "no se van al bien externo; H2 se cumple si es claramente mayor que la participación del nido.\n"
          f"{SOURCE}\n")
    return point


def figure3(d, res):
    a, s = -res.params["price"], res.params["ln_sjg"]
    last = d[d["week"] == d["week"].max()].reset_index(drop=True)
    div = nl.diversion(last, a, s)
    top = last["s"].nlargest(C.TOP_N_HEATMAP).index
    order = last.loc[top].sort_values(["nest", "s"], ascending=[True, False]).index
    M = pd.DataFrame(div[np.ix_(order, order)] * 100,
                     index=last.loc[order, "model"].str.split("/").str[-1],
                     columns=last.loc[order, "model"].str.split("/").str[-1])
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M.values, cmap="Blues")
    ax.set_xticks(range(len(M)), M.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(M)), [f"{n} ({g[0]})" for n, g in zip(M.index, last.loc[order, "nest"])],
                  fontsize=7)
    ax.set_xlabel("Modelo que recibe los tokens")
    ax.set_ylabel("Modelo que sube su precio (a = abierto, c = cerrado)")
    fig.colorbar(im, ax=ax, label="Razón de desvío (%)")
    ax.set_title(f"Figura 3. Razones de desvío, semana {last['week'].max():%Y-%m-%d}")
    fig.text(0.01, 0.01, SOURCE, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(C.FIGURES / "figura3_desvio.png", dpi=200)
    plt.close(fig)


def main(panel=None):
    panel = panel if panel is not None else pd.read_csv(C.CLEAN / "panel.csv", parse_dates=["week"])
    d = base_sample(panel)
    res = table2(d)
    point = table3(d, res)
    figure3(d, res)
    print("Especificación principal (4):")
    print(f"  alpha = {-res.params['price']:.4f}, sigma = {res.params['ln_sjg']:.4f}")
    print(point.drop(columns=["elast_propia"]).round(3))


if __name__ == "__main__":
    main()
