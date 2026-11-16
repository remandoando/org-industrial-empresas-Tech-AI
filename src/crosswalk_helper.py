"""Ayuda a completar la tabla de equivalencias OpenRouter -> LMArena.

Para cada modelo del panel busca los 3 nombres de Arena más parecidos y marca la confianza:
  alta  = mismo creador, mismos números de versión y nombre muy parecido (se llena solo)
  media = hay candidatos, pero hay que elegir a mano
  sin candidato = el modelo probablemente no está en Arena

Salida: data/crosswalk/crosswalk_propuesto.csv, ordenado por tokens (los más importantes arriba).
Revísalo (sobre todo 'media'), corrige 'arena_name' y 'open_weights', y guárdalo como
data/crosswalk/crosswalk.csv. Después:  python run_all.py --no-download

Uso: python src/crosswalk_helper.py   (después de haber corrido 02_build_panel.py)
"""
import glob
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as CFG  # noqa: E402

DROP_WORDS = {"preview", "latest", "exp", "experimental", "instruct", "chat", "it", "hf", "beta"}
# cuantización y prefijos de proveedor: no cambian el modelo, solo cómo se sirve o quién lo publica
QUANT_WORDS = {"nvfp4", "fp4", "fp8", "bf16", "fp16", "int4", "int8", "awq", "gguf"}
VENDOR_WORDS = {"nvidia", "google", "openai", "anthropic", "meta", "microsoft", "ibm", "amazon"}
# variantes de esfuerzo de razonamiento que Arena lista por separado (orden = prioridad al elegir)
EFFORT_ORDER = ["medium", "high", "low", "xhigh", "thinking", "think", "reasoning", "max", "minimal"]
EFFORT = set(EFFORT_ORDER)
AUTHOR_ALIAS = {"x-ai": ["xai"], "z-ai": ["zai", "zhipu"], "moonshotai": ["moonshot"],
                "meta-llama": ["meta"], "mistralai": ["mistral"], "qwen": ["alibaba", "qwen"],
                "nousresearch": ["nous"]}


def tokens(name):
    """Separa un nombre en palabras y números de versión, sin fechas ni sufijos."""
    n = str(name).lower().split("/")[-1].split(":")[0]
    n = re.sub(r"[()\[\],]", " ", n)                               # 'GPT-6 Luna (Max)' -> 'gpt-6 luna max'
    n = re.sub(r"20\d{2}[-_]?\d{2}[-_]?\d{2}", " ", n)          # fechas 2026-07-09 / 20260709
    n = re.sub(r"(?<![\d.])(0[1-9]|1[0-2])(\d{2})$", " ", n)      # fechas cortas al final (0127)
    n = re.sub(r"-(0[1-9]|1[0-2])-(\d{2})$", " ", n)               # fechas cortas al final (06-17)
    parts = re.split(r"[-_\s/]+", n)
    out, nums = [], []
    for p in parts:
        if not p or p in DROP_WORDS or p in QUANT_WORDS or p in VENDOR_WORDS:
            continue
        if re.fullmatch(r"\d+(\.\d+)?", p):
            nums.append(p)                                          # versión partida: 4 6 -> 4.6
            continue
        if nums:
            out.append(".".join(nums))
            nums = []
        out.append(p)
    if nums:
        out.append(".".join(nums))
    return out


def versions(toks):
    # números de versión; se ignoran parámetros activos tipo 'a55b' que Arena suele omitir
    return {t for t in toks if re.search(r"\d", t) and not re.fullmatch(r"a\d+b", t)}


def split_variant(ta, tb):
    """Tokens de Arena que son variante de esfuerzo y NO están en el nombre de OpenRouter.
    Así 'qwen3-max' conserva 'max' (es parte del nombre), pero 'claude-opus-5.5-high' -> variante 'high'."""
    var = [t for t in tb if (t in EFFORT or re.fullmatch(r"\d+k", t)) and t not in ta]
    return [t for t in tb if t not in var], var


def variant_rank(var):
    # 0 = nombre simple; después según EFFORT_ORDER; presupuestos tipo '32k' al final
    if not var:
        return 0
    return 1 + min(EFFORT_ORDER.index(t) if t in EFFORT else len(EFFORT_ORDER) for t in var)


def score(a, b):
    ta, tb_full = tokens(a), tokens(b)
    tb, var = split_variant(ta, tb_full)
    sa, sb = set(ta), set(tb)
    jac = len(sa & sb) / max(1, len(sa | sb))
    seq = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    exact = set(t for t in ta if not re.fullmatch(r"a\d+b", t)) == \
        set(t for t in tb if not re.fullmatch(r"a\d+b", t))
    return 0.5 * jac + 0.5 * seq, versions(ta) == versions(tb), exact, var


def same_author(author, org):
    o = re.sub(r"[^a-z]", "", str(org).lower())
    for a in [author] + AUTHOR_ALIAS.get(author, []):           # acepta alias y nombre original
        a = re.sub(r"[^a-z]", "", a)
        if a and o and (a[:4] in o or o[:4] in a):
            return True
    return False


def main():
    arena_file = sorted(glob.glob(str(CFG.RAW / "lmarena_*.csv")))[-1]
    ar = pd.read_csv(arena_file, usecols=lambda c: c in {"model_name", "organization"})
    ar = ar.drop_duplicates("model_name")
    cw = pd.read_csv(CFG.CROSSWALK / "crosswalk_auto.csv")
    p = pd.read_csv(CFG.CLEAN / "panel.csv")
    tok = p.groupby("model")["tokens"].sum()

    rows = []
    for _, r in cw.iterrows():
        author = r["model"].split("/")[0]
        pool = ar[ar["organization"].map(lambda o: same_author(author, o))] \
            if "organization" in ar else ar
        if pool.empty:
            pool = ar
        sc = [(n, *score(r["base_slug"], n)) for n in pool["model_name"]]
        # exacto > variante preferida (simple primero) > misma versión > parecido > nombre más corto
        sc.sort(key=lambda x: (not x[3], variant_rank(x[4]) if x[3] else 0, not x[2], -x[1], len(x[0])))
        top = sc[:3]
        best, s, same_v, exact, var = top[0] if top else (None, 0, False, False, [])
        if pd.notna(r.get("arena_name")):
            conf, pick = "alta (empate exacto)", r["arena_name"]
        elif best and exact:
            # mismas palabras y versiones (p. ej. claude-4.6-sonnet = claude-sonnet-4-6)
            conf, pick = ("alta" if not var else f"alta (variante {'-'.join(var)})"), best
        elif best and s >= 0.45:
            conf, pick = "media", ""
        else:
            conf, pick = "sin candidato", ""
        rows.append({"model": r["model"], "base_slug": r["base_slug"], "tokens": tok.get(r["model"], 0),
                     "arena_name": pick, "confianza": conf,
                     "open_weights": r.get("open_weights"),
                     **{f"candidato_{i + 1}": f"{n} ({v:.2f}{', misma versión' if sv else ''})"
                        for i, (n, v, sv, _, _) in enumerate(top)}})
    out = pd.DataFrame(rows).sort_values("tokens", ascending=False)
    out.to_csv(CFG.CROSSWALK / "crosswalk_propuesto.csv", index=False)
    print(out["confianza"].value_counts().to_string())
    w = out.groupby("confianza")["tokens"].sum() / out["tokens"].sum()
    print("\nParticipación en tokens por nivel de confianza:\n" + (100 * w).round(1).to_string())
    print("\nRevisa data/crosswalk/crosswalk_propuesto.csv y guárdalo como crosswalk.csv")


if __name__ == "__main__":
    main()
