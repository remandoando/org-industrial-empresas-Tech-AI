"""Descarga los datos crudos. Corre una vez; los archivos quedan con fecha en data/raw/.

Fuentes (todas públicas):
  1. OpenRouter Data API: tokens semanales de los 50 modelos más usados + fila "other" (CC BY 4.0).
     Requiere OPENROUTER_API_KEY en el entorno o en .env.
  2. OpenRouter /api/v1/models: catálogo actual (precios, contexto, hugging_face_id).
  3. OpenRouter /api/v1/models/{author}/{slug}/endpoints: número de proveedores por modelo.
  4. Internet Archive: capturas históricas de /api/v1/models para reconstruir precios por semana.
  5. LMArena leaderboard-dataset (Hugging Face): calificación Arena por modelo y fecha.

Uso:  python src/01_download.py [--skip-archive] [--skip-arena]
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

OR_BASE = "https://openrouter.ai/api/v1"
TODAY = dt.date.today().isoformat()
UA = {"User-Agent": "llm-api-io research (academic, non-commercial)"}


def _key():
    key = os.environ.get("OPENROUTER_API_KEY")   # config.py ya cargó .env
    if not key:
        sys.exit("Falta OPENROUTER_API_KEY (ver .env.example).")
    return key


def get_json(url, params=None, headers=None, tries=5):
    for i in range(tries):
        r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=60)
        if r.status_code == 429:
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Demasiados 429 en {url}")


def download_rankings():
    """Tokens semanales. Se pide por ventanas de 90 días para no depender de un rango máximo."""
    key = _key()
    start = pd.Timestamp(C.START_DATE)
    end = pd.Timestamp(C.END_DATE) if C.END_DATE else pd.Timestamp(TODAY) - pd.Timedelta(days=1)
    rows, metas = [], []
    while start <= end:
        stop = min(start + pd.Timedelta(days=89), end)
        js = get_json(f"{OR_BASE}/datasets/rankings-daily",
                      params={"start_date": start.date().isoformat(),
                              "end_date": stop.date().isoformat(), "period": "week"},
                      headers={"Authorization": f"Bearer {key}"})
        rows += js["data"]
        metas.append(js["meta"])
        start = stop + pd.Timedelta(days=1)
        time.sleep(2.5)  # límite: 30 solicitudes por minuto
    df = pd.DataFrame(rows).drop_duplicates(["date", "model_permaslug"])
    out = C.RAW / f"openrouter_rankings_week_{TODAY}.csv"
    df.to_csv(out, index=False)
    (C.RAW / f"openrouter_rankings_meta_{TODAY}.json").write_text(json.dumps(metas, indent=2))
    print(f"[rankings] {len(df):,} filas -> {out.name}")


def download_models_and_endpoints():
    models, url = [], f"{OR_BASE}/models"
    params = {"output_modalities": "text"}
    while url:
        js = get_json(url, params=params)
        models += js["data"]
        nxt = (js.get("links") or {}).get("next")
        url, params = (f"https://openrouter.ai{nxt}" if nxt else None), None
    (C.RAW / f"openrouter_models_{TODAY}.json").write_text(json.dumps(models))
    print(f"[models] {len(models)} modelos")

    counts, rows = [], []
    for m in models:
        mid = m["id"]
        try:
            js = get_json(f"{OR_BASE}/models/{mid}/endpoints")
            eps = js["data"].get("endpoints", [])
            n = len(eps)
        except requests.HTTPError:
            eps, n = [], None
        counts.append({"id": mid, "canonical_slug": m.get("canonical_slug"), "n_providers": n})
        # una fila por proveedor: insumo de 07_provider_cournot.py
        for e in eps:
            pr = e.get("pricing") or {}
            lat = e.get("latency_last_30m") or {}
            thr = e.get("throughput_last_30m") or {}
            rows.append({"id": mid, "canonical_slug": m.get("canonical_slug"),
                         "hugging_face_id": m.get("hugging_face_id") or "",
                         "model_context_length": m.get("context_length"),
                         "provider_name": e.get("provider_name"), "tag": e.get("tag"),
                         "quantization": e.get("quantization"), "service_tier": e.get("service_tier"),
                         "status": e.get("status"), "prompt": pr.get("prompt"),
                         "completion": pr.get("completion"), "discount": pr.get("discount"),
                         "context_length": e.get("context_length"),
                         "uptime_last_1d": e.get("uptime_last_1d"),
                         "latency_p50": lat.get("p50"), "throughput_p50": thr.get("p50")})
        time.sleep(0.3)
    pd.DataFrame(counts).to_csv(C.RAW / f"openrouter_endpoints_{TODAY}.csv", index=False)
    pd.DataFrame(rows).to_csv(C.RAW / f"openrouter_endpoint_prices_{TODAY}.csv", index=False)
    print(f"[endpoints] {len(rows):,} filas proveedor-modelo")


def decode_snapshot(raw):
    """Las capturas del Internet Archive pueden venir comprimidas (gzip o brotli).
    Devuelve el texto JSON o None si no se puede leer."""
    import gzip
    import zlib
    for fn in (lambda b: b, gzip.decompress, zlib.decompress, _brotli):
        try:
            return fn(raw).decode("utf-8")
        except Exception:
            continue
    return None


def _brotli(b):
    import brotli
    return brotli.decompress(b)


def download_archive_prices():
    """Una captura del catálogo por semana ISO desde el Internet Archive."""
    cdx = requests.get("https://web.archive.org/cdx/search/cdx", headers=UA, timeout=120, params={
        "url": "openrouter.ai/api/v1/models", "output": "json", "filter": "statuscode:200",
        "from": C.START_DATE.replace("-", "")[:8], "collapse": "timestamp:8"}).json()
    if len(cdx) <= 1:
        print("[archive] sin capturas: usa data/manual/prices_manual.csv")
        return
    snaps = pd.DataFrame(cdx[1:], columns=cdx[0])
    snaps["ts"] = pd.to_datetime(snaps["timestamp"], format="%Y%m%d%H%M%S")
    snaps["week"] = snaps["ts"].dt.to_period("W").dt.start_time
    snaps = snaps.sort_values("ts").groupby("week").tail(1)
    outdir = C.RAW / "archive_models"
    outdir.mkdir(exist_ok=True)
    recs = []
    for _, s in snaps.iterrows():
        f = outdir / f"{s['timestamp']}.json"
        if not f.exists():
            url = f"https://web.archive.org/web/{s['timestamp']}id_/{s['original']}"
            for intento in range(4):  # el Archive corta conexiones si se le pide rápido
                try:
                    r = requests.get(url, headers={**UA, "Accept-Encoding": "gzip, deflate"},
                                     timeout=120)
                    if r.ok:
                        f.write_bytes(r.content)
                    break
                except requests.exceptions.RequestException:
                    espera = 30 * (intento + 1)
                    print(f"  [archive] conexión rechazada, reintento en {espera} s: {f.name}")
                    time.sleep(espera)
            time.sleep(4)
        if not f.exists():
            print(f"  [archive] sin captura tras 4 intentos, se omite: {f.name}")
            continue
        text = decode_snapshot(f.read_bytes())
        try:
            data = json.loads(text).get("data", []) if text else None
        except json.JSONDecodeError:
            data = None
        if data is None:
            print(f"  [archive] captura ilegible, se omite: {f.name}")
            continue
        for m in data:
            p = m.get("pricing") or {}
            recs.append({"snapshot": s["ts"], "id": m.get("id"),
                         "canonical_slug": m.get("canonical_slug"),
                         "prompt": p.get("prompt"), "completion": p.get("completion")})
    pd.DataFrame(recs).to_csv(C.RAW / f"archive_prices_{TODAY}.csv", index=False)
    print(f"[archive] {snaps.shape[0]} semanas con captura, {len(recs):,} precios")


def download_arena():
    from datasets import get_dataset_config_names, load_dataset
    repo = "lmarena-ai/leaderboard-dataset"
    token = os.environ.get("HF_TOKEN") or None   # de .env (config.py); opcional para datos públicos
    print(f"[arena] token de Hugging Face: {'sí' if token else 'no (descarga anónima)'}")
    configs = get_dataset_config_names(repo, token=token)
    # subconjunto de texto general, sin control de estilo
    cand = [c for c in configs if c.lower().startswith("text") and "style" not in c.lower()]
    cfg = cand[0] if cand else configs[0]
    ds = load_dataset(repo, cfg, split="full", token=token)
    df = ds.to_pandas()
    df.to_csv(C.RAW / f"lmarena_{cfg}_{TODAY}.csv", index=False)
    print(f"[arena] config='{cfg}', columnas={list(df.columns)}, {len(df):,} filas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-archive", action="store_true")
    ap.add_argument("--skip-arena", action="store_true")
    ap.add_argument("--only", nargs="+", choices=["rankings", "models", "archive", "arena"],
                    help="descarga solo estos pasos (p. ej. --only archive arena)")
    a = ap.parse_args()
    steps = a.only or ["rankings", "models", "archive", "arena"]
    if "rankings" in steps:
        download_rankings()
    if "models" in steps:
        download_models_and_endpoints()
    if "archive" in steps and not a.skip_archive:
        download_archive_prices()
    if "arena" in steps and not a.skip_arena:
        download_arena()
