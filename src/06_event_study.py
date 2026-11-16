"""Validación de H2 sin precios: estudio de eventos con la entrada de modelos al top 50.

Lógica: con logit simple (sin anidamiento, IIA) la entrada de un modelo reduce en la misma
proporción los tokens de todos los modelos establecidos; con efectos fijos de evento×semana esa
caída común queda absorbida y la diferencia entre nidos es cero. Si H2 es cierta (σ > 0), los
establecidos del MISMO nido que el entrante pierden más: β_k < 0 para k >= 0.

    ln q_jet = μ_je + τ_et + Σ_{k≠-1} β_k 1[t−E_e = k] × MismoNido_je + ε_jet

Salidas:
  output/tables/tabla7_eventos.csv/.md     diferencia en diferencias agregada y robustez
  output/tables/eventos_lista.csv          lanzamientos usados (para el apéndice)
  output/figures/figura4_eventos.png       β_k con IC 95%
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from linearmodels.iv.absorbing import AbsorbingLS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

SOURCE = "Fuente: elaboración propia con datos de OpenRouter (openrouter.ai/rankings) y LMArena."


def prepare(panel):
    p = panel[~panel["is_free"]].copy()
    p["author"] = p["model"].str.split("/").str[0]
    weeks = np.sort(p["week"].unique())
    p["t"] = p["week"].map({w: i for i, w in enumerate(weeks)})
    p["rank"] = p.groupby("t")["tokens"].rank(ascending=False, method="first")
    p["ln_q"] = np.log(p["tokens"])
    return p, len(weeks)


def find_events(p, n_weeks, top_rank, exclusion):
    """Entrantes: aparecen por primera vez después de las primeras EVENT_PRE semanas del panel,
    tienen al menos 4 semanas posteriores observables y llegan al top `top_rank` en sus primeras
    4 semanas."""
    first = p.groupby("model")["t"].min()
    ev = []
    for m, t0 in first.items():
        if t0 < C.EVENT_PRE or t0 > n_weeks - 5:
            continue
        early = p[(p["model"] == m) & (p["t"].between(t0, t0 + 3))]
        if early["rank"].min() <= top_rank:
            row = early.iloc[0]
            ev.append({"entrante": m, "t_entrada": t0, "semana": row["week"], "nido": row["nest"],
                       "creador": row["author"], "mejor_lugar": int(early["rank"].min())})
    ev = pd.DataFrame(ev).sort_values("t_entrada").reset_index(drop=True)
    if exclusion and len(ev):
        t = ev["t_entrada"].to_numpy()
        crowded = np.array([(np.abs(t - ti) <= exclusion).sum() > 1 for ti in t])
        ev = ev[~crowded].reset_index(drop=True)
    return ev


def stack(p, events, exclude_same_author=True):
    blocks = []
    for e, ev in events.iterrows():
        lo, hi = ev["t_entrada"] - C.EVENT_PRE, ev["t_entrada"] + C.EVENT_POST
        w = p[p["t"].between(lo, hi) & (p["model"] != ev["entrante"])].copy()
        if exclude_same_author:
            w = w[w["author"] != ev["creador"]]
        pre = w[w["t"] < ev["t_entrada"]].groupby("model")["t"].nunique()
        w = w[w["model"].isin(pre[pre >= C.EVENT_MIN_PRE_WEEKS].index)]
        w["k"] = w["t"] - ev["t_entrada"]
        w["same"] = (w["nest"] == ev["nido"]).astype(int)
        w["event"] = e
        w["entrant_nest"] = ev["nido"]
        blocks.append(w)
    s = pd.concat(blocks, ignore_index=True)
    s["em"] = s["event"].astype(str) + "|" + s["model"]
    s["et"] = s["event"].astype(str) + "|" + s["t"].astype(str)
    return s


def fit(s, regressors):
    absorb = pd.DataFrame({"em": pd.Categorical(s["em"]), "et": pd.Categorical(s["et"])})
    clusters = pd.Categorical(s["model"]).codes
    return AbsorbingLS(s["ln_q"], s[regressors], absorb=absorb).fit(
        cov_type="clustered", clusters=clusters)


def dynamic(s):
    ks = [k for k in range(-C.EVENT_PRE, C.EVENT_POST + 1) if k != -1]
    cols = []
    for k in ks:
        c = f"k{k:+d}"
        s[c] = ((s["k"] == k) & (s["same"] == 1)).astype(float)
        cols.append(c)
    r = fit(s, cols)
    ci = r.conf_int()
    return pd.DataFrame({"k": ks, "beta": r.params[cols].to_numpy(),
                         "lo": ci.loc[cols, "lower"].to_numpy(), "hi": ci.loc[cols, "upper"].to_numpy()})


def did(s):
    s = s.copy()
    s["pre_x_mismo"] = ((s["k"] <= -2) & (s["same"] == 1)).astype(float)
    s["post_x_mismo"] = ((s["k"] >= 0) & (s["same"] == 1)).astype(float)
    r = fit(s, ["pre_x_mismo", "post_x_mismo"])
    return {"β post (mismo nido)": r.params["post_x_mismo"], "ee post": r.std_errors["post_x_mismo"],
            "p post": r.pvalues["post_x_mismo"], "β pre (tendencia previa)": r.params["pre_x_mismo"],
            "ee pre": r.std_errors["pre_x_mismo"], "p pre": r.pvalues["pre_x_mismo"],
            "eventos": s["event"].nunique(), "obs": len(s)}


def main(panel=None):
    panel = panel if panel is not None else pd.read_csv(C.CLEAN / "panel.csv", parse_dates=["week"])
    p, n_weeks = prepare(panel)
    events = find_events(p, n_weeks, C.EVENT_TOP_RANK, C.EVENT_EXCLUSION)
    if len(events) < 5:
        print(f"Solo {len(events)} eventos: relaja EVENT_TOP_RANK o EVENT_EXCLUSION en config.py")
        return
    events.to_csv(C.TABLES / "eventos_lista.csv", index=False)
    s = stack(p, events)

    rows = [{"especificación": "Base (todos los entrantes)", **did(s)}]
    for nest in (C.NEST_OPEN, C.NEST_CLOSED):
        sub = s[s["entrant_nest"] == nest]
        if sub["event"].nunique() >= 3:
            rows.append({"especificación": f"Solo entrantes de pesos {nest}s", **did(sub)})
    alt = [("Entrante en top 5", dict(top_rank=5, exclusion=C.EVENT_EXCLUSION), True),
           ("Entrante en top 20", dict(top_rank=20, exclusion=C.EVENT_EXCLUSION), True),
           ("Sin excluir eventos cercanos", dict(top_rank=C.EVENT_TOP_RANK, exclusion=0), True),
           ("Excluye eventos a ≤ 2 semanas", dict(top_rank=C.EVENT_TOP_RANK, exclusion=2), True),
           ("Incluye modelos del mismo creador", dict(top_rank=C.EVENT_TOP_RANK,
                                                      exclusion=C.EVENT_EXCLUSION), False)]
    for label, kw, excl_author in alt:
        ev = find_events(p, n_weeks, **kw)
        if len(ev) >= 5:
            rows.append({"especificación": label, **did(stack(p, ev, excl_author))})
    t7 = pd.DataFrame(rows)
    t7["cambio % post"] = 100 * (np.exp(t7["β post (mismo nido)"]) - 1)
    t7.to_csv(C.TABLES / "tabla7_eventos.csv", index=False)
    (C.TABLES / "tabla7_eventos.md").write_text(
        "Tabla 7. Estudio de eventos: entrada de un modelo al top 50 y tokens de los modelos "
        "establecidos del mismo nido frente al otro nido\n\n" + t7.round(4).to_markdown(index=False)
        + "\n\nNota: variable dependiente ln tokens semanales del modelo establecido. Efectos fijos de "
          "evento×modelo y evento×semana; errores agrupados por modelo. 'β post' = diferencia promedio "
          f"en semanas 0 a {C.EVENT_POST} entre establecidos del mismo nido que el entrante y del otro "
          "nido; H2 predice β post < 0. 'β pre' compara semanas −4 a −2 contra −1 y debe ser ≈ 0. "
          "Se excluyen del grupo de comparación los modelos del mismo creador que el entrante.\n"
          + SOURCE + "\n")

    dyn = dynamic(s)
    dyn.to_csv(C.TABLES / "figura4_coeficientes.csv", index=False)
    ref = pd.DataFrame({"k": [-1], "beta": [0.0], "lo": [0.0], "hi": [0.0]})
    dyn = pd.concat([dyn, ref]).sort_values("k")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(-0.5, color="grey", lw=1, ls="--")
    ax.errorbar(dyn["k"], 100 * dyn["beta"], yerr=[100 * (dyn["beta"] - dyn["lo"]),
                                                    100 * (dyn["hi"] - dyn["beta"])],
                fmt="o", color="#1f4e79", ecolor="#7fa7cf", capsize=3)
    ax.set_xlabel("Semanas desde la entrada al top 50 (−1 = referencia)")
    ax.set_ylabel("Diferencia en ln tokens × 100, mismo nido − otro nido")
    ax.set_title(f"Figura 4. Efecto de una entrada sobre los establecidos del mismo nido "
                 f"({s['event'].nunique()} eventos)", fontsize=10)
    fig.text(0.01, 0.01, SOURCE + " Barras: IC 95%, errores agrupados por modelo.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(C.FIGURES / "figura4_eventos.png", dpi=200)
    plt.close(fig)
    print(t7.round(3).to_string())


if __name__ == "__main__":
    main()
