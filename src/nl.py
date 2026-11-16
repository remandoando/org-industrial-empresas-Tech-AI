"""Funciones compartidas: instrumentos, estimación del logit anidado y sus elasticidades.

Notación (logit anidado de un nivel, Berry 1994):
    ln s_jt - ln s_0t = x_jt'b - alpha p_jt + sigma ln s_{jt|g} + mu_j + tau_t + xi_jt
El coeficiente de precio estimado es -alpha.
"""
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

# ----------------------------------------------------------------------------- variables

def add_shares(df, lam, group="nest"):
    """Participaciones dentro del mercado potencial M_t = lam * tokens totales de la semana."""
    d = df.copy()
    d["M"] = lam * d["week_total_tokens"]
    d["s"] = d["tokens"] / d["M"]
    d["s0"] = 1 - d.groupby("week")["s"].transform("sum")
    if (d["s0"] <= 0).any():
        raise ValueError(f"lambda={lam} deja s0<=0; usa un lambda mayor.")
    d["s_g"] = d.groupby(["week", group])["s"].transform("sum")
    d["s_jg"] = d["s"] / d["s_g"]
    d["y"] = np.log(d["s"]) - np.log(d["s0"])
    d["ln_sjg"] = np.log(d["s_jg"])
    return d


def add_diff_ivs(df, band, group="nest"):
    """Instrumentos de diferenciación por semana: conteos y distancias en calidad,
    con todos los rivales y solo con rivales del mismo nido."""
    out = []
    for _, w in df.groupby("week", sort=False):
        q = w["quality"].to_numpy()
        g = w[group].to_numpy()
        diff = q[:, None] - q[None, :]
        same = g[:, None] == g[None, :]
        np.fill_diagonal(same, False)
        rival = ~np.eye(len(w), dtype=bool)
        close = (np.abs(diff) < band) & rival
        sq = (diff / 100.0) ** 2 * rival
        w = w.copy()
        w["iv_n_nest"] = same.sum(1)
        w["iv_close_all"] = close.sum(1)
        w["iv_close_nest"] = (close & same).sum(1)
        w["iv_sq_all"] = sq.sum(1)
        w["iv_sq_nest"] = (sq * same).sum(1)
        out.append(w)
    return pd.concat(out)


IV_SETS = {
    "todos": ["iv_n_nest", "iv_close_all", "iv_close_nest", "iv_sq_all", "iv_sq_nest"],
    "solo_calidad": ["iv_close_all", "iv_close_nest", "iv_sq_all", "iv_sq_nest"],
    "solo_conteos": ["iv_n_nest", "iv_close_all", "iv_close_nest"],
}

# ----------------------------------------------------------------------------- estimación

def _formula(endog, ivs, controls):
    ctrl = " + ".join(controls) if controls else ""
    fe = "C(model_id) + C(week_id)"
    rhs = " + ".join([x for x in (ctrl, fe) if x])
    if ivs:
        return f"y ~ 1 + {rhs} + [{' + '.join(endog)} ~ {' + '.join(ivs)}]"
    return f"y ~ 1 + {rhs} + {' + '.join(endog)}"


def estimate(df, nested=True, iv=True, ivs=None, controls=("quality", "ln_age"),
             cluster="model_id"):
    """OLS o 2SLS con efectos fijos de modelo y semana; errores agrupados."""
    endog = ["price", "ln_sjg"] if nested else ["price"]
    d = df.copy()
    d["model_id"] = d["model"].astype("category").cat.codes
    d["week_id"] = d["week"].astype("category").cat.codes
    clusters = d[cluster] if cluster in d else d["model"].astype("category").cat.codes
    f = _formula(endog, IV_SETS[ivs or "todos"] if iv else None, list(controls))
    res = IV2SLS.from_formula(f, d).fit(cov_type="clustered", clusters=clusters)
    return res


def summarize(res, nested=True, iv=True):
    """Coeficientes clave, errores estándar y diagnósticos de instrumentos."""
    r = {"alpha": -res.params["price"], "se_alpha": res.std_errors["price"],
         "n_obs": int(res.nobs)}
    if nested:
        r.update(sigma=res.params["ln_sjg"], se_sigma=res.std_errors["ln_sjg"],
                 p_sigma_pos=float(1 - _norm_cdf(res.params["ln_sjg"] / res.std_errors["ln_sjg"])))
    if iv:
        fs = res.first_stage.diagnostics
        for v in fs.index:
            r[f"F_primera_etapa_{v}"] = float(fs.loc[v, "f.stat"])
        try:
            j = res.wooldridge_overid
            r["sobreid_stat"], r["sobreid_p"] = float(j.stat), float(j.pval)
        except Exception:  # exactamente identificado
            pass
    return r


def _norm_cdf(z):
    from math import erf, sqrt
    return 0.5 * (1 + erf(z / sqrt(2)))

# ----------------------------------------------------------------------------- elasticidades

def derivatives(w, alpha, sigma, group="nest"):
    """Matriz ds_k/dp_j (fila j = producto cuyo precio cambia) de una semana."""
    s = w["s"].to_numpy()
    sjg = w["s_jg"].to_numpy()
    g = w[group].to_numpy()
    same = g[:, None] == g[None, :]
    a = sigma / (1 - sigma)
    # ds_k/dp_j = alpha * s_j * ( a*s_k|g*1[mismo nido] + s_k ),  j != k
    D = alpha * s[:, None] * (a * sjg[None, :] * same + s[None, :])
    own = -alpha * s * (1 / (1 - sigma) - a * sjg - s)
    np.fill_diagonal(D, own)
    return D


def elasticities(w, alpha, sigma, group="nest"):
    D = derivatives(w, alpha, sigma, group)
    p, s = w["price"].to_numpy(), w["s"].to_numpy()
    return D * p[:, None] / s[None, :]


def diversion(w, alpha, sigma, group="nest"):
    """D_jk = -(ds_k/dp_j)/(ds_j/dp_j). Fila j suma, con el bien externo, 1."""
    D = derivatives(w, alpha, sigma, group)
    own = np.diag(D).copy()
    div = -D / own[:, None]
    np.fill_diagonal(div, 0.0)
    return div


def group_diversion(w, alpha, sigma, group="nest"):
    """Para cada producto: fracción de tokens perdidos que va a su propio nido, al otro nido
    y al bien externo, más su elasticidad propia."""
    div = diversion(w, alpha, sigma, group)
    g = w[group].to_numpy()
    same = (g[:, None] == g[None, :])
    to_same = (div * same).sum(1)
    to_other = (div * ~same).sum(1)
    E = elasticities(w, alpha, sigma, group)
    return pd.DataFrame({"model": w["model"].to_numpy(), group: g, "s": w["s"].to_numpy(),
                         "elast_propia": np.diag(E), "desvio_mismo_nido": to_same,
                         "desvio_otro_nido": to_other,
                         "desvio_externo": 1 - to_same - to_other})


def nest_summary(panel, alpha, sigma, group="nest"):
    """Promedios ponderados por tokens, sobre todas las semanas, por nido de origen."""
    parts = [group_diversion(w, alpha, sigma, group) for _, w in panel.groupby("week")]
    gd = pd.concat(parts)
    # entre los tokens que se quedan en modelos del top 50, qué fracción se queda en el nido
    gd["mismo_nido_entre_internos"] = gd["desvio_mismo_nido"] / (gd["desvio_mismo_nido"]
                                                                 + gd["desvio_otro_nido"])
    cols = ["elast_propia", "desvio_mismo_nido", "desvio_otro_nido", "desvio_externo",
            "mismo_nido_entre_internos"]
    def wavg(x):  # promedio ponderado por tokens que ignora celdas no definidas
        return pd.Series({c: np.average(x[c][x[c].notna()], weights=x["s"][x[c].notna()])
                          for c in cols})
    return gd.groupby(group).apply(wavg, include_groups=False)


def bootstrap_nest_summary(panel, res, n, seed, group="nest"):
    """Intervalos al 95% de las razones de desvío extrayendo solo σ de su distribución asintótica.
    En el logit anidado los desvíos no dependen de α (se cancela en el cociente de derivadas), así que
    no se condiciona al signo del precio, que no está identificado. Se descartan σ fuera de (0, 1)."""
    rng = np.random.default_rng(seed)
    s_hat, s_se = res.params["ln_sjg"], res.std_errors["ln_sjg"]
    draws = rng.normal(s_hat, s_se, size=n)
    draws = draws[(draws > 0) & (draws < 1)]
    a_ref = 1.0                                   # cualquier α > 0 da los mismos desvíos
    cols = ["desvio_mismo_nido", "desvio_otro_nido", "desvio_externo", "mismo_nido_entre_internos"]
    sims = [nest_summary(panel, a_ref, d, group)[cols].stack() for d in draws]
    sims = pd.concat(sims, axis=1)
    lo, hi = sims.quantile(0.025, axis=1), sims.quantile(0.975, axis=1)
    return pd.DataFrame({"ic95_inf": lo, "ic95_sup": hi}), len(draws)


def group_elasticity(panel_week, alpha, sigma, nest_value, pct=0.05, group="nest"):
    """Respuesta de los tokens de todo un nido a un alza uniforme de pct en sus precios
    (insumo de la prueba del monopolista hipotético). Recalcula shares con logit anidado."""
    w = panel_week
    delta = np.log(w["s"]) - np.log(w["s0"]) - sigma * np.log(w["s_jg"])
    up = w[group].eq(nest_value).to_numpy()
    delta_new = delta - alpha * w["price"].to_numpy() * pct * up

    def shares(dl):
        e = np.exp(dl / (1 - sigma))
        Dg = pd.Series(e).groupby(w[group].to_numpy()).transform("sum").to_numpy()
        Dg_unique = pd.Series(e).groupby(w[group].to_numpy()).sum()
        denom = 1 + (Dg_unique ** (1 - sigma)).sum()
        return e / Dg * Dg ** (1 - sigma) / denom

    s_old, s_new = shares(delta.to_numpy()), shares(delta_new)
    q_old = s_old[up].sum()
    q_new = s_new[up].sum()
    return (q_new - q_old) / q_old


def critical_loss(margin, pct=0.05):
    """Pérdida crítica de la prueba SSNIP con márgenes de Lerner supuestos."""
    return pct / (pct + margin)
