"""H7: competencia entre proveedores que sirven el MISMO modelo abierto (producto casi homogéneo).

Cada modelo de pesos abiertos es un mercado; el número de proveedores N varía entre mercados.
  (a) Precios contra N.  Cournot simétrico: P_N − c = (a − c)/(N + 1); el margen sigue cayendo con
      cada entrada.  Bertrand homogéneo: el precio llega al costo con N = 2 y ya no cambia.
      Prueba: en ln(precio) ~ dummies de N, H0 Bertrand = coeficientes de N = 3, 4, 5+ iguales al de N = 2.
  (b) Umbrales de entrada (Bresnahan y Reiss, 1991): probit ordenado de N sobre el tamaño de
      mercado.  Si la demanda necesaria POR PROVEEDOR crece con N, la entrada intensifica la
      competencia (márgenes que caen, como en Cournot); si la razón es ≈ 1, no.

Entradas: data/raw/openrouter_endpoint_prices_*.csv (01_download.py --only models),
          data/raw/openrouter_rankings_week_*.csv y data/clean/panel.csv.
Salidas:  output/tables/tabla8_precios_vs_N.*, tabla9_umbrales_entrada.*, mercados_proveedores.csv
          output/figures/figura5_precio_vs_N.png, figura6_umbrales.png
"""
import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.miscmodels.ordinal_model import OrderedModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as CFG  # noqa: E402  (no 'C': patsy usa C() en las fórmulas)

SOURCE = "Fuente: elaboración propia con datos de OpenRouter (openrouter.ai) y LMArena."


def latest(pattern):
    files = sorted(glob.glob(str(CFG.RAW / pattern)))
    if not files:
        raise FileNotFoundError(f"Falta {pattern}: corre  python src/01_download.py --only models")
    return files[-1]


def parse_params(text):
    """Parámetros en miles de millones desde el nombre (p. ej. 'Qwen3-235B-A22B' -> activos 22)."""
    t = str(text)
    act = re.search(r"[-_]A(\d+(?:\.\d+)?)B\b", t)
    if act:
        return float(act.group(1))
    tot = re.findall(r"(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", t)
    return float(tot[-1]) if tot else np.nan


def build_markets(endpoints=None, rankings=None, panel=None):
    e = endpoints if endpoints is not None else pd.read_csv(latest("openrouter_endpoint_prices_*.csv"))
    e = e[(e["hugging_face_id"].fillna("") != "") & ~e["id"].str.contains(":free")]
    e = e[e["status"].isna() | (e["status"] == 0)]
    e = e[e["service_tier"].isna()]
    for c in ("prompt", "completion"):
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e["price"] = 1e6 * (CFG.W_PROMPT * e["prompt"] + (1 - CFG.W_PROMPT) * e["completion"])
    e = e[e["price"] > 0]
    # un precio por proveedor (el más bajo si ofrece varias cuantizaciones)
    e = e.sort_values("price").drop_duplicates(["canonical_slug", "provider_name"])

    g = e.groupby("canonical_slug")
    m = pd.DataFrame({
        "N": g["provider_name"].nunique(), "p_min": g["price"].min(), "p_med": g["price"].median(),
        "p_max": g["price"].max(), "cv": g["price"].std() / g["price"].mean(),
        "quant_main": g["quantization"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "nd"),
        "ctx": g["model_context_length"].first(), "hf": g["hugging_face_id"].first()}).reset_index()
    m["N_bin"] = np.minimum(m["N"], CFG.N_CAP)
    m["N_cat"] = m["N_bin"].map(lambda n: f"{n}+" if n == CFG.N_CAP else str(n))
    m["ln_p_min"], m["ln_p_med"] = np.log(m["p_min"]), np.log(m["p_med"])
    m["ln_ctx"] = np.log(m["ctx"].clip(lower=1))
    m["params_b"] = m["hf"].map(parse_params)
    m["ln_params"] = np.log(m["params_b"])

    # tamaño de mercado: tokens semanales recientes del modelo (variantes de pago)
    r = rankings if rankings is not None else pd.read_csv(latest("openrouter_rankings_week_*.csv"))
    r = r[(r["model_permaslug"] != "other") & ~r["model_permaslug"].str.contains(":free")].copy()
    r["slug"] = r["model_permaslug"].str.split(":").str[0]
    r["week"] = pd.to_datetime(r["date"]).dt.to_period("W").dt.start_time
    r["tokens"] = pd.to_numeric(r["total_tokens"])
    recent = r[r["week"] > r["week"].max() - pd.Timedelta(weeks=CFG.MARKET_SIZE_WEEKS)]
    w = recent.groupby(["slug", "week"])["tokens"].sum().groupby("slug").agg(["mean", "count"])
    w = w[w["count"] >= CFG.MARKET_SIZE_WEEKS // 2]
    m["S"] = m["canonical_slug"].map(w["mean"])
    m["ln_S"] = np.log(m["S"])

    p = panel if panel is not None else pd.read_csv(CFG.CLEAN / "panel.csv", parse_dates=["week"])
    q = p.sort_values("week").groupby("base_slug")["quality"].last()
    m["quality"] = m["canonical_slug"].map(q)
    return m


def price_regressions(m):
    specs = {"(1) Base": ("ln_ctx", 15),
             "(2) + parámetros": ("ln_ctx + ln_params", 15),
             "(3) + calidad y tamaño": ("ln_ctx + quality + ln_S", 40)}   # (control, mínimo de mercados)
    m = m.copy()
    m["N_grp"] = np.where(m["N"] >= 3, "3+", m["N"].astype(int).astype(str))
    groupings = {"detallado": ("N_cat", [c for c in ["2", "3", "4", f"{CFG.N_CAP}+"]
                                         if c in set(m["N_cat"])]),
                 "agrupado (1, 2, 3+)": ("N_grp", ["2", "3+"])}
    rows, fits = [], {}
    for y in ("ln_p_min", "ln_p_med"):
        for gname, (gvar, cats) in groupings.items():
            for name, (ctrl, min_obs) in specs.items():
                vs = [v.strip() for v in ctrl.split("+")]
                d = m.dropna(subset=[y] + vs)
                if d[gvar].nunique() < 3 or len(d) < min_obs:
                    continue
                ctrl_ok = " + ".join(v for v in vs if d[v].nunique() > 1) or "1"  # sin controles constantes
                term = f"C({gvar}, Treatment('1'))"
                res = smf.ols(f"{y} ~ {term} + {ctrl_ok}", d).fit(cov_type="HC1")
                if gname == "detallado":
                    fits[(y, name)] = res
                row = {"precio": "mínimo" if y == "ln_p_min" else "mediano", "N": gname,
                       "especificación": name, "mercados": int(res.nobs)}
                for c in cats:
                    k = f"{term}[T.{c}]"
                    if k in res.params:
                        row[f"N={c}"] = res.params[k]
                        row[f"ee N={c}"] = res.bse[k]
                base = f"{term}[T.2]"
                later = [f"{term}[T.{c}]" for c in cats[1:] if f"{term}[T.{c}]" in res.params]
                if later and base in res.params:
                    test = res.wald_test(", ".join(f"{k} = {base}" for k in later), scalar=True)
                    row["p H0 Bertrand (N≥3 = N=2)"] = float(test.pvalue)
                rows.append(row)
    out = pd.DataFrame(rows)
    pcol = "p H0 Bertrand (N≥3 = N=2)"
    if pcol in out:
        out = out[[c for c in out.columns if c != pcol] + [pcol]]
    return out, fits


def entry_thresholds(m, n_boot, seed):
    d = m.dropna(subset=["ln_S"]).copy()
    d["N_ord"] = pd.Categorical(d["N_bin"], ordered=True)
    counts = d["N_bin"].value_counts().reindex(range(1, CFG.N_CAP + 1), fill_value=0)
    if len(d) < 20 or (counts < 5).any():
        return None, f"{len(d)} mercados con tamaño observado; por N: {counts.to_dict()} " \
                     "(se requieren todas las categorías con al menos 5)"

    def fit(dd):
        cats = pd.Categorical(dd["N_bin"], ordered=True)
        mod = OrderedModel(cats, dd[["ln_S"]], distr="probit")
        r = mod.fit(method="bfgs", disp=False, maxiter=2000)
        beta = r.params["ln_S"]
        cut = mod.transform_threshold_params(r.params)[1:-1]
        n_levels = list(cats.categories[1:])          # N que se alcanza al cruzar cada corte
        lnS = cut / beta
        S = pd.Series(np.exp(lnS), index=n_levels)
        per_firm = S / S.index.to_numpy()
        prev = per_firm.reindex(per_firm.index.to_numpy() - 1).to_numpy()   # solo niveles consecutivos
        ratio = pd.Series(per_firm.to_numpy() / prev, index=per_firm.index)
        return S, ratio

    S, ratio = fit(d)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        bd = d.sample(len(d), replace=True, random_state=rng.integers(1e9))
        if pd.Categorical(bd["N_bin"]).categories.size != d["N_ord"].cat.categories.size:
            continue
        try:
            boots.append(fit(bd)[1])
        except Exception:
            continue
    B = pd.concat(boots, axis=1) if boots else pd.DataFrame(index=ratio.index)
    out = pd.DataFrame({"N alcanzado": S.index, "umbral S_N (tokens/semana)": S.to_numpy(),
                        "por proveedor S_N/N": (S / S.index.to_numpy()).to_numpy(),
                        "razón (S_N/N)/(S_{N-1}/(N-1))": ratio.to_numpy(),
                        "IC95 inf": B.quantile(0.025, axis=1).reindex(S.index).to_numpy(),
                        "IC95 sup": B.quantile(0.975, axis=1).reindex(S.index).to_numpy()})
    return out, len(d)


def figure5(fits):
    # se prefiere la especificación que controla por tamaño del modelo si cubre >= 30 mercados
    key = ("ln_p_min", "(2) + parámetros")
    if key not in fits or fits[key].nobs < 30:
        key = ("ln_p_min", "(1) Base")
    if key not in fits:
        return
    res = fits[key]
    control = "ventana de contexto y parámetros" if "parámetros" in key[1] else "ventana de contexto"
    cats = ["1"] + [c for c in ["2", "3", "4", f"{CFG.N_CAP}+"]
                    if f"C(N_cat, Treatment('1'))[T.{c}]" in res.params]
    b = [0.0] + [res.params[f"C(N_cat, Treatment('1'))[T.{c}]"] for c in cats[1:]]
    se = [0.0] + [res.bse[f"C(N_cat, Treatment('1'))[T.{c}]"] for c in cats[1:]]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, 100 * np.array(b), yerr=196 * np.array(se), fmt="o", color="#1f4e79",
                ecolor="#7fa7cf", capsize=3, label="Estimación (IC 95%)")
    if len(b) > 1:
        ax.hlines(100 * b[1], 1, x[-1], colors="#c0504d", linestyles="--",
                  label="Bertrand: sin cambio después de N = 2")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(x, cats)
    ax.set_xlabel("Número de proveedores del modelo (N)")
    ax.set_ylabel("Precio mínimo vs N = 1 (diferencia en ln × 100)")
    ax.set_title("Figura 5. Precio de un modelo abierto según su número de proveedores", fontsize=10)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, SOURCE + f"\nControla por {control}; errores robustos HC1.", fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(CFG.FIGURES / "figura5_precio_vs_N.png", dpi=200)
    plt.close(fig)


def figure6(t9):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = t9.dropna(subset=["razón (S_N/N)/(S_{N-1}/(N-1))"])
    x = np.arange(len(t))
    y = t["razón (S_N/N)/(S_{N-1}/(N-1))"].to_numpy()
    err = np.vstack([y - t["IC95 inf"].to_numpy(), t["IC95 sup"].to_numpy() - y])
    ax.bar(x, y, color="#7fa7cf")
    ax.errorbar(x, y, yerr=np.nan_to_num(err), fmt="none", ecolor="#1f4e79", capsize=3)
    ax.axhline(1, color="#c0504d", ls="--", label="1 = la entrada no cambia la competencia")
    ax.set_xticks(x, [f"{int(a) - 1}→{int(a)}{'+' if int(a) == CFG.N_CAP else ''}"
                      for a in t["N alcanzado"]])
    ax.set_xlabel("Paso de proveedores")
    ax.set_ylabel("Demanda necesaria por proveedor (razón)")
    ax.set_title("Figura 6. Umbrales de entrada por proveedor (Bresnahan y Reiss)", fontsize=10)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, SOURCE + " IC 95% por remuestreo de mercados.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CFG.FIGURES / "figura6_umbrales.png", dpi=200)
    plt.close(fig)


def main(markets=None):
    m = markets if markets is not None else build_markets()
    m.to_csv(CFG.TABLES / "mercados_proveedores.csv", index=False)
    desc = m.groupby("N_cat").agg(mercados=("N", "size"), precio_min_mediana=("p_min", "median"),
                                  precio_mediano_mediana=("p_med", "median"),
                                  dispersion_cv=("cv", "mean"),
                                  con_tamano_de_mercado=("S", lambda s: s.notna().sum()))
    print("Mercados por número de proveedores:\n", desc.round(3))

    t8, fits = price_regressions(m)
    t8.to_csv(CFG.TABLES / "tabla8_precios_vs_N.csv", index=False)
    (CFG.TABLES / "tabla8_precios_vs_N.md").write_text(
        "Tabla 8. Precio de un modelo abierto según su número de proveedores (N = 1 es la base)\n\n"
        + desc.round(3).to_markdown() + "\n\n" + t8.round(4).to_markdown(index=False)
        + "\n\nNota: variable dependiente ln del precio mínimo o mediano entre proveedores (USD por "
          "millón de tokens, mezcla 75/25 entrada/salida). Un mercado = un modelo de pesos abiertos. "
          "Errores robustos HC1. 'agrupado' junta N = 3, 4 y 5+ en una categoría 3+ para ganar potencia; "
          "la especificación (3) solo se estima con 40 mercados o más. 'p H0 Bertrand' prueba que los coeficientes de N ≥ 3 son iguales al de "
          "N = 2; Cournot predice que el precio sigue bajando. 'cv' = coeficiente de variación de "
          "precios entre proveedores (asimetría de costos).\n" + SOURCE + "\n")
    figure5(fits)

    t9, n = entry_thresholds(m, CFG.N_BOOT_BR, CFG.SEED)
    if t9 is None:
        print(f"Umbrales de entrada no estimados: {n}.")
        (CFG.TABLES / "tabla9_umbrales_entrada.md").write_text(
            f"Tabla 9 no estimada: {n}. Con los datos disponibles solo los modelos que estuvieron en el "
            "top 50 tienen tamaño de mercado observado.\n")
    else:
        t9.to_csv(CFG.TABLES / "tabla9_umbrales_entrada.csv", index=False)
        (CFG.TABLES / "tabla9_umbrales_entrada.md").write_text(
            "Tabla 9. Umbrales de entrada de proveedores (probit ordenado, Bresnahan y Reiss, 1991)\n\n"
            + t9.round(3).to_markdown(index=False)
            + f"\n\nNota: mercados con tokens en el top 50 en al menos {CFG.MARKET_SIZE_WEEKS // 2} de "
              f"las últimas {CFG.MARKET_SIZE_WEEKS} semanas. Una razón > 1 indica que cada proveedor "
              "adicional necesita más demanda por proveedor: la entrada reduce los márgenes. Intervalos "
              f"por {CFG.N_BOOT_BR} remuestreos de mercados (semilla {CFG.SEED}).\n" + SOURCE + "\n")
        figure6(t9)
    print(t8.round(3).to_string())
    if t9 is not None:
        print(t9.round(3).to_string())


if __name__ == "__main__":
    main()
