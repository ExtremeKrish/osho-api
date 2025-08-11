# main.py
import os
import re
import difflib
from math import ceil
from typing import Optional, List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from psycopg2.pool import SimpleConnectionPool
import psycopg2.extras

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("Please set DATABASE_URL in .env")

app = FastAPI(title="Osho Discourse Search API (in-memory exact + trigram fallback)")

# ---------- Pool & In-memory cache ----------
POOL: Optional[SimpleConnectionPool] = None
# DATA: list of dicts with description and lowercased fields for fast in-memory search
DATA: List[Dict] = []

# snippet radius (chars before and after match)
SNIPPET_RADIUS = 160


class SearchResponse(BaseModel):
    query: str
    totalFound: int
    page: int
    totalPages: int
    limit: int
    results: list


@app.on_event("startup")
def startup_event():
    global POOL, DATA
    try:
        POOL = SimpleConnectionPool(minconn=1, maxconn=8, dsn=DB_URL)
    except Exception as e:
        raise RuntimeError(f"Failed to create DB pool: {e}")

    # preload minimal dataset into memory to serve *exact phrase* searches instantly
    conn = POOL.getconn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("""
            SELECT id, title, slug, description, language, series_name, audiofile
            FROM discourses
        """)
        rows = cur.fetchall()
        DATA = []
        for r in rows:
            desc = r["description"] or ""
            DATA.append({
                "id": int(r["id"]),
                "title": r["title"],
                "slug": r["slug"],
                "description": desc,
                "description_lower": desc.lower(),
                "language": (r["language"] or "").lower(),
                "seriesName": r["series_name"],
                "seriesName_lower": (r["series_name"] or "").lower(),
                "audioFile": r["audiofile"],
            })
    finally:
        POOL.putconn(conn)


@app.on_event("shutdown")
def shutdown_event():
    global POOL
    if POOL:
        POOL.closeall()


def db_conn():
    if not POOL:
        raise HTTPException(status_code=500, detail="DB pool not initialized")
    try:
        return POOL.getconn()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB connection error: {e}")


def release_conn(conn):
    if POOL and conn:
        try:
            POOL.putconn(conn)
        except Exception:
            pass


def make_snippet_and_highlight(orig_text: str, match_span: Optional[tuple], query: str) -> str:
    """
    Create a snippet around match_span and wrap the matched substring with <em>..</em>.
    match_span is (start, end) in original string. If None, return the start of the text truncated.
    """
    if not orig_text:
        return ""
    if match_span:
        s, e = match_span
        start = max(0, s - SNIPPET_RADIUS)
        end = min(len(orig_text), e + SNIPPET_RADIUS)
        before = orig_text[start:s]
        matched = orig_text[s:e]
        after = orig_text[e:end]
        return (before + "<em>" + matched + "</em>" + after).strip()
    # fallback snippet (first 2*SNIPPET_RADIUS chars)
    return (orig_text[: SNIPPET_RADIUS * 2] + ("..." if len(orig_text) > SNIPPET_RADIUS * 2 else "")).strip()


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1, description="Search text (Hindi or English)"),
    limit: int = Query(10, ge=1, le=200, description="Results per page"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    lang: Optional[str] = Query(None, description="Filter by language (exact)"),
    series: Optional[str] = Query(None, description="Filter by series_name (exact)"),
):
    """
    Fast exact-phrase search (in-memory). If no exact hits, fallback to trigram search in DB.
    """
    q_raw = query.strip()
    if not q_raw:
        raise HTTPException(status_code=400, detail="Empty query")

    q_lower = q_raw.lower()
    offset = (page - 1) * limit

    # Build filters (lowercased) for in-memory check
    lang_l = lang.lower() if lang else None
    series_l = series.lower() if series else None

    # --- 1) In-memory exact phrase search (case-insensitive) ---
    filtered = []
    if lang_l or series_l:
        # apply filters first to reduce scanning
        for item in DATA:
            if lang_l and item["language"] != lang_l:
                continue
            if series_l and item["seriesName_lower"] != series_l:
                continue
            filtered.append(item)
    else:
        filtered = DATA  # no filters, use full cache

    exact_matches = []
    # Use regex search with re.IGNORECASE to get original-span for highlight safely (handles unicode)
    escaped_q = re.escape(q_raw)
    pattern = re.compile(escaped_q, flags=re.IGNORECASE)

    for item in filtered:
        m = pattern.search(item["description"])
        if m:
            s, e = m.start(), m.end()
            snippet = make_snippet_and_highlight(item["description"], (s, e), q_raw)
            exact_matches.append({
                "title": item["title"],
                "slug": item["slug"],
                "score": 1.0,
                "ftRank": 1.0,
                "trigramSim": 0.0,
                "language": item.get("language", ""),
                "seriesName": item.get("seriesName"),
                "audioFile": item.get("audioFile"),
                "highlight": snippet
            })

    total_found = len(exact_matches)

    # If we have exact matches, return paginated slice
    if total_found > 0:
        page_slice = exact_matches[offset: offset + limit]
        total_pages = ceil(total_found / limit) if limit else 1
        return {
            "query": q_raw,
            "totalFound": total_found,
            "page": page,
            "totalPages": total_pages,
            "limit": limit,
            "results": page_slice
        }

    # --- 2) No exact match: trigram similarity fallback via DB (uses trigram index) ---
    # Build filter SQL and params
    filters_sql_parts = []
    params: List = []
    if lang:
        filters_sql_parts.append("LOWER(language) = %s")
        params.append(lang.lower())
    if series:
        filters_sql_parts.append("LOWER(series_name) = %s")
        params.append(series.lower())
    filter_sql = (" AND " + " AND ".join(filters_sql_parts)) if filters_sql_parts else ""

    # Count matches using trigram operator % (fast with gin_trgm_ops index)
    count_sql = f"SELECT COUNT(*) FROM discourses WHERE description %% %s {filter_sql}"
    count_params = [q_raw] + params

    # Search SQL using similarity + ts_headline for highlight
    search_sql = f"""
        SELECT
            title,
            slug,
            similarity(description, %s) AS trigram_sim,
            language,
            series_name,
            audioFile,
            ts_headline(
                'simple',
                description,
                plainto_tsquery('simple', %s),
                'startsel=<em>,stopsel=</em>,maxfragments=2,maxwords=200,highlightall=true'
            ) AS highlight
        FROM discourses
        WHERE description %% %s {filter_sql}
        ORDER BY trigram_sim DESC
        LIMIT %s OFFSET %s
    """
    # parameters: similarity(%s), ts_headline(%s), WHERE %% %s, then filters, limit, offset
    search_params = [q_raw, q_raw, q_raw] + params + [limit, offset]

    conn = db_conn()
    try:
        cur = conn.cursor()
        cur.execute(count_sql, count_params)
        total_found = cur.fetchone()[0]
        cur.close()

        results: List[Dict] = []
        if total_found > 0:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute(search_sql, search_params)
            rows = cur.fetchall()
            for r in rows:
                trig = float(r["trigram_sim"] or 0.0)
                results.append({
                    "title": r["title"],
                    "slug": r["slug"],
                    "score": float(trig),
                    "ftRank": 0.0,
                    "trigramSim": trig,
                    "language": r["language"],
                    "seriesName": r["series_name"],
                    "audioFile": r["audiofile"],
                    "highlight": r["highlight"] or ""
                })
            cur.close()

    except Exception as e:
        # make sure connection is released and raise friendly error
        release_conn(conn)
        raise HTTPException(status_code=500, detail=f"DB error (trigram fallback): {e}")
    finally:
        release_conn(conn)

    total_pages = ceil(total_found / limit) if limit else 1
    return {
        "query": q_raw,
        "totalFound": total_found,
        "page": page,
        "totalPages": total_pages,
        "limit": limit,
        "results": results
    }