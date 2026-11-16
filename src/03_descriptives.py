"""Tabla 1 (descriptivos por nido), Figura 1 (participación por creador e IHH) y
Figura 2 (precio contra calidad)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

SOURCE = "Fuente: elaboración propia con datos de OpenRouter (openrouter.ai/rankings) y LMArena."


def main(panel=None):
    p = panel if panel is not None else pd.read_csv(C.CLEAN / "panel.csv", parse_dates=["week"])
    p = p[~p["is_free"]].copy()
    p["tokens_mm"] = p["tokens"] / 1e9

    # Tabla 1
    vars_ = {"tokens_mm": "Tokens por semana (miles de millones)",
             "price": "Precio mezclado (USD por millón de tokens)",
             "quality": "Calificación Arena (puntos)", "age_weeks": "Antigüedad (semanas)"}
    rows = []
    for nest, g in p.groupby("nest"):
        for v, lab in vars_.items():
            rows.append({"nido": nest, "variable": lab, "media": g[v].mean(), "desv_est": g[v].std(),
                         "min": g[v].min(), "max": g[v].max(), "obs": g[v].notna().sum()})
    t1 = pd.DataFrame(rows)
    t1.to_csv(C.TABLES / "tabla1_descriptivos.csv", index=False)
    (C.TABLES / "tabla1_descriptivos.md").write_text(
        "Tabla 1. Estadística descriptiva por nido, modelo-semana\n\n"
        + t1.round(2).to_markdown(index=False) + f"\n\n{SOURCE}\n")

    # Figura 1: participación en tokens por creador (dentro del top 50) e IHH
    p["author"] = p["model"].str.split("/").str[0]
    sh = p.groupby(["week", "author"])["tokens"].sum().unstack(fill_value=0)
    sh = sh.div(sh.sum(1), axis=0) * 100
    top = sh.mean().nlargest(8).index
    plot = pd.concat([sh[top], sh.drop(columns=top).sum(1).rename("otros")], axis=1)
    hhi = (sh ** 2).sum(1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(plot.index, plot.T.values, labels=plot.columns, alpha=0.85)
    ax.set_ylabel("Participación en tokens del top 50 (%)")
    ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot(hhi.index, hhi.values, color="black", lw=1.5, label="IHH (eje derecho)")
    ax2.set_ylabel("Índice Herfindahl-Hirschman (0–10,000)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=8)
    ax.set_title("Figura 1. Participación semanal por creador e IHH")
    fig.text(0.01, 0.01, SOURCE, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(C.FIGURES / "figura1_participacion_ihh.png", dpi=200)
    plt.close(fig)

    # Figura 2: precio contra calidad (promedio por modelo)
    m = p.groupby(["model", "nest"], as_index=False).agg(price=("price", "mean"),
                                                         quality=("quality", "mean"),
                                                         tokens=("tokens", "sum"))
    fig, ax = plt.subplots(figsize=(7, 5))
    for nest, g in m.groupby("nest"):
        ax.scatter(g["quality"], g["price"], s=20 + 300 * g["tokens"] / m["tokens"].max(),
                   alpha=0.6, label=f"Pesos {nest}s")
    ax.set_yscale("log")
    ax.set_xlabel("Calificación Arena (puntos)")
    ax.set_ylabel("Precio mezclado (USD por millón de tokens, escala log)")
    ax.legend(title="Tamaño = tokens totales")
    ax.set_title("Figura 2. Precio y calidad por modelo")
    fig.text(0.01, 0.01, SOURCE, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(C.FIGURES / "figura2_precio_calidad.png", dpi=200)
    plt.close(fig)
    print("Descriptivos listos")


if __name__ == "__main__":
    main()
