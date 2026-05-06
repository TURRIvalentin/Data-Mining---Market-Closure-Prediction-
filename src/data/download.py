"""
Polymarket download pipeline — Fase 2.

Flujo:
  1. Paginar Gamma API /markets para recolectar todos los mercados candidatos
     que pasen los filtros de inclusión. Resultado cacheado en data/raw/candidates.json.
  2. Muestreo estratificado por trimestre de creación.
  3. Descargar serie de precios diarios (CLOB /prices-history, primeros 7 días)
     para cada mercado muestreado.

Reanudable: cada archivo de precios se escribe individualmente. Al relanzar,
se saltean los mercados cuyo archivo ya existe.

Uso:
    python -m src.data.download              # 500 mercados, seed=42
    python -m src.data.download --sample 50  # prueba rápida
    python -m src.data.download --dry-run    # muestra plan sin descargar nada
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    CLOB_API_BASE,
    GAMMA_API_BASE,
    API_PAGE_SIZE,
    API_RATE_LIMIT_PAUSE,
    DATE_START,
    MIN_MARKET_DURATION_DAYS,
    MIN_PRICE_POINTS,
    OBSERVATION_WINDOW_DAYS,
    RANDOM_SEED,
)

# CLOB API no tiene rate limit estricto (verificado empíricamente: 15 req/s sin 429).
# Usamos 0.15s como pausa mínima por respeto al servidor, no por restricción técnica.
# Gamma API (deprecated) usa el API_RATE_LIMIT_PAUSE original (0.5s) por ser más sensible.
CLOB_PAUSE = 0.15

# Parada por plateau: si las últimas PLATEAU_WINDOW páginas suman < PLATEAU_MIN_NEW candidatos,
# la API ya entró en territorio pre-DATE_START y no vale la pena seguir paginando.
PLATEAU_WINDOW  = 20
PLATEAU_MIN_NEW = 2

# ── Categorización heurística ────────────────────────────────────────────────
# Las reglas viven en src/features/categorization.py (auditables, versionadas).
from src.features.categorization import infer_category_coarse  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR         = Path("data/raw")
MARKETS_DIR     = RAW_DIR / "markets"   # un JSON por mercado (metadata Gamma)
PRICES_DIR      = RAW_DIR / "prices"    # un JSON por mercado (serie de precios)
CANDIDATES_FILE = RAW_DIR / "candidates.json"   # cache de candidatos
RUN_META_FILE   = RAW_DIR / "download_run.json" # resumen de la corrida

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── Utilidades de fecha ───────────────────────────────────────────────────────

def parse_dt(s: str) -> datetime:
    """Parsea strings de fecha de Polymarket a datetime UTC.

    Maneja formatos observados en la API:
      "2024-09-24T00:00:00Z"
      "2026-05-03 14:29:04+00"
      "2024-01-15T12:00:00+00:00"
    """
    s = s.strip()
    s = s.replace(" ", "T")           # normaliza separador
    s = re.sub(r"\+00$", "+00:00", s) # +00 → +00:00
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def get_start_dt(market: dict) -> datetime:
    """Retorna el datetime de inicio de actividad real del mercado.

    Usa startDate con fallback a createdAt.

    Justificación: startDate es la fecha en que el mercado abre al trading.
    createdAt puede ser hasta 2 días anterior (el mercado se registra en la
    base de datos antes de activarse). La ventana de observación debe comenzar
    en el primer día de trading, no en el de registro.
    """
    sd = market.get("startDate") or market.get("startDateIso")
    ca = market.get("createdAt")
    raw = sd if sd else ca
    return parse_dt(raw)


# ── Filtros de inclusión ──────────────────────────────────────────────────────

def passes_filters(market: dict) -> tuple[bool, str]:
    """Evalúa si un mercado cumple los criterios de inclusión.

    Returns:
        (True, "ok") si pasa todos los filtros.
        (False, reason) con el primer criterio que falla.
    """
    # 1. Debe estar cerrado/resuelto
    if not market.get("closed"):
        return False, "not_closed"

    # 2. Debe tener clobTokenIds (necesarios para /prices-history)
    raw_tokens = market.get("clobTokenIds", "[]")
    try:
        token_ids = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
    except (json.JSONDecodeError, TypeError):
        return False, "bad_token_ids"
    if not token_ids:
        return False, "no_token_ids"

    # 3. Debe ser binario: exactamente ["Yes","No"] o ["No","Yes"]
    raw_outcomes = market.get("outcomes", "[]")
    try:
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes
    except (json.JSONDecodeError, TypeError):
        return False, "bad_outcomes"
    if len(outcomes) != 2 or {o.lower() for o in outcomes} != {"yes", "no"}:
        return False, "not_binary"

    # 4. Resolución confirmada: outcomePrices debe ser exactamente [1,0] o [0,1]
    #    Valores intermedios (0.5, etc.) indican cancelación o resolución N/A.
    raw_op = market.get("outcomePrices", "[]")
    try:
        op = json.loads(raw_op) if isinstance(raw_op, str) else raw_op
        p_yes = float(op[0])
        p_no  = float(op[1])
    except (json.JSONDecodeError, TypeError, IndexError, ValueError):
        return False, "bad_outcome_prices"
    if not ((p_yes == 1.0 and p_no == 0.0) or (p_yes == 0.0 and p_no == 1.0)):
        return False, "cancelled_or_partial"

    # 5. Debe tener fechas de inicio y cierre
    # Usamos startDate (primer día de trading) con fallback a createdAt.
    # closedTime es la fecha REAL de resolución; endDate es la programada y puede
    # diferir hasta 302 días (verificado empíricamente).
    start_raw   = market.get("startDate") or market.get("startDateIso") or market.get("createdAt")
    closed_time = market.get("closedTime")
    if not start_raw or not closed_time:
        return False, "missing_dates"
    try:
        start_dt = parse_dt(start_raw)
        closed_dt = parse_dt(closed_time)
    except Exception:
        return False, "date_parse_error"

    # 6. Inicio dentro del período de estudio (lee DATE_START de config — no hardcodeado)
    period_start = datetime.fromisoformat(DATE_START).replace(tzinfo=timezone.utc)
    if start_dt < period_start:
        return False, "before_period"

    # 7. Duración mínima de actividad real (startDate → closedTime)
    if (closed_dt - start_dt).days < MIN_MARKET_DURATION_DAYS:
        return False, "too_short"

    return True, "ok"


# ── HTTP con backoff ──────────────────────────────────────────────────────────

def get_with_backoff(url: str, params: dict, max_retries: int = 6) -> requests.Response:
    """GET con backoff exponencial en 429 (rate limit) y 5xx (errores de servidor)."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                log.warning(f"Rate limit (429). Esperando {wait:.1f}s (intento {attempt + 1})")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = (2 ** attempt) + random.uniform(0, 1)
                log.warning(f"Error servidor {r.status_code}. Esperando {wait:.1f}s (intento {attempt + 1})")
                time.sleep(wait)
                continue
            return r
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            log.warning(f"Error de red: {e}. Esperando {wait:.1f}s (intento {attempt + 1})")
            time.sleep(wait)
    raise RuntimeError(f"Se agotaron los reintentos para {url}")


# ── Fase 1: recolección de candidatos ─────────────────────────────────────────

def fetch_all_candidates(force_refresh: bool = False) -> list[dict]:
    """Pagina Gamma API /markets y recolecta todos los mercados que pasan filtros.

    Si CANDIDATES_FILE ya existe, carga el cache (evita re-descargar la lista).
    Usar force_refresh=True para forzar una nueva descarga completa.
    """
    if CANDIDATES_FILE.exists() and not force_refresh:
        log.info(f"Cache encontrado: {CANDIDATES_FILE}. Cargando candidatos ...")
        with open(CANDIDATES_FILE) as f:
            data = json.load(f)
        log.info(
            f"  {len(data['candidates'])} candidatos cargados "
            f"(descargados el {data.get('fetched_at', 'desconocido')[:19]})"
        )
        log.info(f"  Excluidos: {data['excluded']}")
        return data["candidates"]

    candidates = []
    excluded   = defaultdict(int)
    offset     = 0
    page_num   = 0
    period_start = datetime.fromisoformat(DATE_START).replace(tzinfo=timezone.utc)

    # Paginamos en orden descendente por closedTime (mercados más recientes primero).
    # Parada por plateau: si las últimas PLATEAU_WINDOW páginas suman < PLATEAU_MIN_NEW
    # candidatos nuevos, asumimos que el resto del catálogo es anterior a DATE_START.
    # Nota: startDateMin es ignorado por la API; el corte temporal se aplica en Python.
    log.info(
        f"Descargando lista de mercados de Gamma API "
        f"(closed=true, order=closedTime desc, corte Python >= {DATE_START}) ..."
    )
    recent_new_per_page: list[int] = []

    while True:
        params = {
            "closed":    "true",
            "limit":     API_PAGE_SIZE,
            "offset":    offset,
            "order":     "closedTime",
            "ascending": "false",   # más recientes primero → parada anticipada
        }
        r = get_with_backoff(f"{GAMMA_API_BASE}/markets", params)

        if r.status_code != 200:
            log.error(f"Status inesperado {r.status_code} en offset={offset}: {r.text[:200]}")
            break

        page = r.json()
        if not isinstance(page, list) or len(page) == 0:
            break

        page_all_old = True   # bandera fallback: toda la página es anterior al período
        prev_n       = len(candidates)

        for m in page:
            ct_raw = m.get("closedTime", "")
            try:
                if parse_dt(ct_raw) >= period_start:
                    page_all_old = False
            except Exception:
                page_all_old = False

            ok, reason = passes_filters(m)
            if ok:
                candidates.append(m)
            else:
                excluded[reason] += 1

        page_num += 1
        offset   += len(page)

        # Plateau tracking
        new_this_page = len(candidates) - prev_n
        recent_new_per_page.append(new_this_page)
        if len(recent_new_per_page) > PLATEAU_WINDOW:
            recent_new_per_page.pop(0)

        if page_num % 5 == 0:
            log.info(
                f"  Página {page_num:3d} | offset={offset:6d} "
                f"| candidatos: {len(candidates)}"
            )

        # Parada por plateau
        if (len(recent_new_per_page) == PLATEAU_WINDOW
                and sum(recent_new_per_page) < PLATEAU_MIN_NEW):
            log.info(
                f"  PLATEAU en página {page_num} (offset={offset}): "
                f"{sum(recent_new_per_page)} candidatos nuevos en últimas "
                f"{PLATEAU_WINDOW} páginas. Deteniendo paginación."
            )
            break

        # Parada por página completamente anterior al período
        if page_all_old:
            log.info(
                f"  Parada anticipada en página {page_num}: "
                f"toda la página es anterior a {DATE_START}"
            )
            break

        if len(page) < API_PAGE_SIZE:
            break  # última página

        time.sleep(API_RATE_LIMIT_PAUSE)

    excluded_sorted = dict(sorted(excluded.items(), key=lambda x: -x[1]))
    total_excl = sum(excluded_sorted.values())
    log.info(
        f"Paginación completa: {page_num} páginas | {offset} mercados procesados | "
        f"{len(candidates)} candidatos | {total_excl} excluidos"
    )
    log.info(f"Excluidos por criterio: {excluded_sorted}")

    CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_FILE, "w") as f:
        json.dump(
            {
                "candidates": candidates,
                "excluded":   excluded_sorted,
                "fetched_at": datetime.utcnow().isoformat(),
                "pagination": {
                    "pages_scanned":     page_num,
                    "markets_processed": offset,
                    "plateau_window":    PLATEAU_WINDOW,
                    "plateau_min_new":   PLATEAU_MIN_NEW,
                },
                "filters": {
                    "date_start":          DATE_START,
                    "min_duration_days":   MIN_MARKET_DURATION_DAYS,
                    "min_price_points":    MIN_PRICE_POINTS,
                    "requires_binary":     True,
                    "requires_confirmed_outcome": True,
                },
            },
            f,
            indent=2,
        )
    log.info(f"Candidatos guardados en {CANDIDATES_FILE}")
    return candidates


# ── Fase 2: muestreo estratificado ────────────────────────────────────────────

def stratified_sample(candidates: list[dict], n: int, seed: int) -> list[dict]:
    """Muestrea n mercados estratificados por trimestre de creación.

    Asignación proporcional al tamaño de cada estrato; muestreo aleatorio
    dentro de cada trimestre. Seed fijo para reproducibilidad.

    Justificación: el muestreo por trimestre garantiza representación temporal
    uniforme, lo cual es más importante que estratificar por categoría para
    un análisis con potencial drift temporal.
    """
    buckets: dict[str, list] = defaultdict(list)
    for m in candidates:
        try:
            dt = get_start_dt(m)
            q  = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        except Exception:
            q = "unknown"
        buckets[q].append(m)

    total = len(candidates)
    rng   = random.Random(seed)

    # Asignación proporcional
    allocated: dict[str, tuple[list, int]] = {}
    for q, group in sorted(buckets.items()):
        alloc = max(1, round(len(group) / total * n))
        allocated[q] = (group, alloc)

    # Ajuste por redondeo para llegar exactamente a n
    total_alloc = sum(a for _, a in allocated.values())
    diff = n - total_alloc
    if diff != 0:
        biggest_q = max(allocated, key=lambda q: len(allocated[q][0]))
        g, a = allocated[biggest_q]
        allocated[biggest_q] = (g, max(1, a + diff))

    log.info(f"Muestreo estratificado: {n} de {total} candidatos en {len(buckets)} trimestres")
    sampled = []
    for q, (group, alloc) in sorted(allocated.items()):
        actual = min(alloc, len(group))
        picked = rng.sample(group, actual)
        sampled.extend(picked)
        log.info(f"  {q}: {len(group):4d} disponibles → {actual} seleccionados")

    log.info(f"  Total muestreado: {len(sampled)}")
    return sampled


# ── Fase 3: descarga de precios ───────────────────────────────────────────────

def fetch_prices_for_market(market: dict) -> dict | None:
    """Descarga la serie de precios diarios del token YES para los primeros
    OBSERVATION_WINDOW_DAYS días de vida del mercado.

    Returns None si la llamada a la API falla.
    """
    cid       = market["conditionId"]
    raw_tok   = market.get("clobTokenIds", "[]")
    token_ids = json.loads(raw_tok) if isinstance(raw_tok, str) else raw_tok
    token_yes = token_ids[0]  # token[0] = YES según documentación de Polymarket

    # Ventana desde startDate (primer día de trading), no desde createdAt.
    # El gap createdAt→startDate puede ser hasta 2 días; usando startDate se
    # recuperan hasta 2 puntos extra de precio en la ventana de observación.
    start_dt = get_start_dt(market)
    end_dt   = start_dt + timedelta(days=OBSERVATION_WINDOW_DAYS)
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())

    params = {
        "market":   token_yes,
        "startTs":  start_ts,
        "endTs":    end_ts,
        "fidelity": 1440,  # diario (1440 minutos)
    }
    r = get_with_backoff(f"{CLOB_API_BASE}/prices-history", params)

    if r.status_code != 200:
        log.warning(
            f"  prices-history error para {cid[:20]}...: "
            f"{r.status_code} {r.text[:120]}"
        )
        return None

    history = r.json().get("history", [])

    return {
        "condition_id": cid,
        "token_yes":    token_yes,
        "start_ts":     start_ts,
        "end_ts":       end_ts,
        "n_points":     len(history),
        "history":      history,  # [{t: unix_ts, p: float}, ...]
    }


def download_prices(sampled: list[dict], log_every: int = 50) -> dict:
    """Descarga precios para todos los mercados muestreados.

    Reanudable: saltea mercados cuyo archivo de precios ya existe.
    Escribe cada archivo individualmente apenas lo recibe (no acumula en memoria).
    """
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    total      = len(sampled)
    downloaded = 0
    skipped    = 0
    failed     = []
    start_time = time.time()

    log.info(f"Descargando precios para {total} mercados (fidelity=1440, window={OBSERVATION_WINDOW_DAYS}d) ...")

    for i, market in enumerate(sampled):
        cid      = market["conditionId"]
        out_file = PRICES_DIR / f"{cid}.json"

        if out_file.exists():
            skipped += 1
            continue

        try:
            prices = fetch_prices_for_market(market)

            if prices is None:
                failed.append({"cid": cid, "reason": "api_error"})
                # No escribimos el archivo: en el próximo run se reintentará
                continue

            if prices["n_points"] == 0:
                # Mercado con precio sin historia (puede ocurrir en los primeros días)
                # Guardamos igual para no reintentar en el próximo run
                prices["note"] = "empty_history"
                log.debug(f"  Sin historia de precios: {cid[:30]}...")

            with open(out_file, "w") as f:
                json.dump(prices, f)

            downloaded += 1

        except Exception as e:
            log.error(f"  Error inesperado para {cid[:20]}...: {e}")
            failed.append({"cid": cid, "reason": str(e)})
            continue

        # Log de progreso
        done = downloaded + skipped
        if done % log_every == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate    = downloaded / elapsed if elapsed > 0 else 0
            log.info(
                f"  [{i + 1:4d}/{total}] "
                f"descargados={downloaded} saltados={skipped} fallidos={len(failed)} "
                f"| {elapsed:.0f}s | {rate:.1f} mercados/s"
            )

        time.sleep(CLOB_PAUSE)  # 0.15s: sin rate limit detectado, pausa por respeto al servidor

    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


# ── Guardado de metadata de mercados ─────────────────────────────────────────

def save_market_metadata(sampled: list[dict]) -> None:
    """Guarda el JSON de metadata de Gamma API para cada mercado muestreado."""
    MARKETS_DIR.mkdir(parents=True, exist_ok=True)
    for m in sampled:
        cid = m["conditionId"]
        out = MARKETS_DIR / f"{cid}.json"
        if not out.exists():
            with open(out, "w") as f:
                json.dump(m, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_sample: int = 500, seed: int = RANDOM_SEED,
         dry_run: bool = False, use_all: bool = False,
         refresh: bool = False) -> None:
    mode = "ALL" if use_all else f"sample={n_sample}"
    log.info("=" * 65)
    log.info(f"Polymarket download  |  {mode}  seed={seed}  dry_run={dry_run}")
    log.info("=" * 65)

    # ── Fase 1: candidatos
    candidates = fetch_all_candidates(force_refresh=refresh)
    if not candidates:
        log.error("Sin candidatos. Verificar acceso a la API y filtros.")
        return
    log.info(f"\nCandidatos disponibles: {len(candidates)}")

    # ── Fase 2: selección
    if use_all:
        # Usar todos los candidatos, ordenados por closedTime (alineado con split temporal)
        sampled = sorted(candidates, key=lambda m: m.get("closedTime", ""))
        log.info(f"Modo --all: usando los {len(sampled)} candidatos sin muestreo")
    else:
        sampled = stratified_sample(candidates, n=n_sample, seed=seed)

    if dry_run:
        log.info(f"\nDRY RUN activado — {len(candidates)} candidatos totales (pre-filtro puntos)")
        _print_sample_preview(sampled, seed=seed)
        return

    # ── Guarda metadata de mercados
    save_market_metadata(sampled)

    # ── Fase 3: precios
    stats = download_prices(sampled)

    # ── Estadísticas del dataset final
    dataset_stats = _compute_dataset_stats(sampled)

    # ── Metadata de la corrida
    run_meta = {
        "run_at":            datetime.utcnow().isoformat(),
        "n_candidates":      len(candidates),
        "n_downloaded":      len(sampled),
        "use_all":           use_all,
        "seed":              seed if not use_all else None,
        "prices_downloaded": stats["downloaded"],
        "prices_skipped":    stats["skipped"],
        "prices_failed":     len(stats["failed"]),
        "failed_markets":    stats["failed"],
        "dataset_final":     dataset_stats["n_final"],
        "below_min_points":  dataset_stats["below_min_points"],
    }
    RUN_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_META_FILE, "w") as f:
        json.dump(run_meta, f, indent=2)

    log.info("\n" + "=" * 65)
    log.info("DESCARGA COMPLETADA")
    log.info(f"  Candidatos totales   : {len(candidates)}")
    log.info(f"  Descargados          : {len(sampled)}")
    log.info(f"  Precios OK           : {stats['downloaded']}")
    log.info(f"  Saltados (ya exist)  : {stats['skipped']}")
    log.info(f"  Fallidos             : {len(stats['failed'])}")
    log.info(f"  Con < {MIN_PRICE_POINTS} puntos (excluidos): {dataset_stats['below_min_points']}")
    log.info(f"  Dataset final        : {dataset_stats['n_final']}")
    log.info("")
    log.info("  Balance YES/NO:")
    log.info(f"    YES: {dataset_stats['yes']} ({dataset_stats['yes_pct']:.1f}%)")
    log.info(f"    NO : {dataset_stats['no']} ({dataset_stats['no_pct']:.1f}%)")
    log.info("")
    log.info("  Distribucion n_puntos_precio:")
    for pts, cnt in sorted(dataset_stats["points_dist"].items()):
        pct = cnt / dataset_stats["n_final"] * 100 if dataset_stats["n_final"] else 0
        bar = "#" * int(pct / 3)
        log.info(f"    {pts:2d} pts: {cnt:4d} ({pct:5.1f}%)  {bar}")
    log.info("")
    log.info("  Distribucion category_coarse:")
    for cat, cnt in sorted(dataset_stats["cat_dist"].items(), key=lambda x: -x[1]):
        pct = cnt / dataset_stats["n_final"] * 100 if dataset_stats["n_final"] else 0
        log.info(f"    {cat:15s}: {cnt:4d} ({pct:5.1f}%)")
    log.info("=" * 65)


def _compute_dataset_stats(sampled: list[dict]) -> dict:
    """Lee archivos de precios descargados y calcula estadísticas del dataset final.

    Aplica el filtro MIN_PRICE_POINTS y reporta distribuciones de n_points,
    category_coarse y balance YES/NO.
    """
    from collections import Counter as _Counter
    points_dist: _Counter = _Counter()
    cat_dist:    _Counter = _Counter()
    yes = no = 0
    below = 0

    for m in sampled:
        cid      = m.get("conditionId", "")
        pfile    = PRICES_DIR / f"{cid}.json"
        if not pfile.exists():
            continue
        with open(pfile) as f:
            p = json.load(f)
        n_pts = p.get("n_points", 0)
        if n_pts < MIN_PRICE_POINTS:
            below += 1
            continue

        points_dist[n_pts] += 1
        cat_dist[infer_category_coarse(m.get("question", ""))] += 1

        raw_op = m.get("outcomePrices", "[]")
        op = json.loads(raw_op) if isinstance(raw_op, str) else raw_op
        if float(op[0]) == 1.0:
            yes += 1
        else:
            no += 1

    n_final = yes + no
    return {
        "n_final":          n_final,
        "below_min_points": below,
        "yes":              yes,
        "no":               no,
        "yes_pct":          yes / n_final * 100 if n_final else 0,
        "no_pct":           no  / n_final * 100 if n_final else 0,
        "points_dist":      dict(points_dist),
        "cat_dist":         dict(cat_dist),
    }


def _print_sample_preview(sampled: list[dict], seed: int = RANDOM_SEED) -> None:
    """Muestra resumen del sample + sondea puntos de precio en subconjunto."""
    by_q: dict[str, int] = defaultdict(int)
    by_cat: dict[str, int] = defaultdict(int)
    outcomes = {"yes": 0, "no": 0}

    for m in sampled:
        try:
            dt = get_start_dt(m)
            by_q[f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"] += 1
        except Exception:
            by_q["unknown"] += 1
        by_cat[infer_category_coarse(m.get("question", ""))] += 1
        raw_op = m.get("outcomePrices", "[]")
        op = json.loads(raw_op) if isinstance(raw_op, str) else raw_op
        if float(op[0]) == 1.0:
            outcomes["yes"] += 1
        else:
            outcomes["no"] += 1

    log.info("\nDistribucion por trimestre:")
    for q, c in sorted(by_q.items()):
        log.info(f"  {q}: {c}")
    log.info("\nDistribucion por categoria (top 10):")
    for cat, c in sorted(by_cat.items(), key=lambda x: -x[1])[:10]:
        log.info(f"  {cat or 'sin categoria':30s}: {c}")
    log.info(f"\nBalance de clases  YES={outcomes['yes']}  NO={outcomes['no']}")

    # 5 ejemplos aleatorios
    rng = random.Random(seed)
    log.info("\n5 ejemplos aleatorios del sample:")
    log.info(f"  {'conditionId[:20]':<22} {'start':>10} {'closed':>10} {'dur':>5} {'vol':>10} {'cat':>12} out")
    log.info("  " + "-" * 80)
    for m in rng.sample(sampled, min(5, len(sampled))):
        cid = m.get("conditionId", "?")[:20]
        cat = (m.get("category") or "?")[:12]
        ct  = (m.get("closedTime") or "")[:10]
        vol = m.get("volumeNum") or 0
        try:
            sd  = get_start_dt(m)
            cd  = parse_dt(m["closedTime"])
            dur = (cd - sd).days
            sd_str = sd.strftime("%Y-%m-%d")
        except Exception:
            dur    = "?"
            sd_str = "?"
        raw_op = m.get("outcomePrices", "[]")
        op = json.loads(raw_op) if isinstance(raw_op, str) else raw_op
        out = "YES" if float(op[0]) == 1.0 else "NO"
        log.info(f"  {cid:<22} {sd_str:>10} {ct:>10} {str(dur):>5} {float(vol):>10,.0f} {cat:>12} {out}")
        log.info(f"    Q: {m.get('question','')[:75]}")

    _probe_price_points(sampled, probe_n=60, seed=seed)


def _probe_price_points(sampled: list[dict], probe_n: int = 60,
                        seed: int = RANDOM_SEED) -> None:
    """Descarga precios de probe_n mercados aleatorios y muestra distribucion de puntos.

    Sirve para estimar cuantos mercados del sample sobreviviran el filtro MIN_PRICE_POINTS.
    No escribe nada a disco.
    """
    from collections import Counter as _Counter
    rng   = random.Random(seed)
    probe = rng.sample(sampled, min(probe_n, len(sampled)))

    log.info(f"\nSondeo de puntos de precio ({len(probe)} mercados) ...")
    counts: list[int] = []
    failed = 0
    for m in probe:
        result = fetch_prices_for_market(m)
        if result is None:
            failed += 1
        else:
            counts.append(result["n_points"])
        time.sleep(CLOB_PAUSE)

    if not counts:
        log.info("  Sin datos de precio disponibles en el sondeo.")
        return

    dist = _Counter(counts)
    log.info(f"  Distribucion de n_puntos_precio (n={len(counts)}, fallidos={failed}):")
    for n_pts in sorted(dist.keys()):
        pct = dist[n_pts] / len(counts) * 100
        bar = "#" * int(pct / 3)
        log.info(f"  {n_pts:2d} pts: {dist[n_pts]:4d} ({pct:5.1f}%)  {bar}")

    pass_n  = sum(v for k, v in dist.items() if k >= MIN_PRICE_POINTS)
    fail_n  = sum(v for k, v in dist.items() if k < MIN_PRICE_POINTS)
    pass_rt = pass_n / len(counts)
    log.info(f"\n  >= {MIN_PRICE_POINTS} puntos : {pass_n} ({pass_rt*100:.1f}%)")
    log.info(f"  <  {MIN_PRICE_POINTS} puntos : {fail_n} ({fail_n/len(counts)*100:.1f}%) -> se descartaran")
    log.info(
        f"  Dataset final estimado: ~{int(len(sampled) * pass_rt)} mercados "
        f"(extrapolado de {len(counts)} sondeos)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga datos de Polymarket")
    parser.add_argument("--sample",    type=int,  default=500, help="N mercados a muestrear (ignorado con --all)")
    parser.add_argument("--seed",      type=int,  default=RANDOM_SEED)
    parser.add_argument("--dry-run",   action="store_true", help="Mostrar plan sin descargar")
    parser.add_argument("--refresh",   action="store_true", help="Ignorar cache de candidatos y re-descargar lista")
    parser.add_argument("--all",       action="store_true", dest="use_all",
                        help="Descargar todos los candidatos sin muestrear")
    args = parser.parse_args()
    main(n_sample=args.sample, seed=args.seed, dry_run=args.dry_run,
         use_all=args.use_all, refresh=args.refresh)
