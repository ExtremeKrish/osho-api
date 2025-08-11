# main.py
import os
from math import ceil
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("Please set DATABASE_URL in .env")

app = FastAPI(title="Osho Discourse Search API")

# ---------- Connection Pool ----------
pool: SimpleConnectionPool = None

@app.on_event("startup")
def startup_event():
    global pool
    pool = SimpleConnectionPool(minconn=1, maxconn=5, dsn=DB_URL)

@app.on_event("shutdown")
def shutdown_event():
    global pool
    if pool:
        pool.closeall()

def get_conn():
    return pool.getconn()

def release_conn(conn):
    pool.putconn(conn)


class SearchResponse(BaseModel):
    query: str
    total_found: int
    page: int
    total_pages: int
    limit: int
    results: list


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1, description="Search text (Hindi or English)"),
    limit: int = Query(10, ge=1, le=200, description="Results per page"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    lang: Optional[str] = Query(None, description="Filter by language, e.g. 'hindi' or 'english'"),
    series: Optional[str] = Query(None, description="Filter by series_name, exact match"),
):
    """
    Hybrid search:
      - Full-text search using tsvector (simple config)
      - Trigram similarity fallback (pg_trgm)
      - Highlight matched snippet with <em>..</em> using ts_headline
    """

    offset = (page - 1) * limit

    # Base WHERE clause (full-text OR trigram)
    base_where = "(search_vector @@ plainto_tsquery('simple', %s) OR description %% %s)"
    params_for_base = [query, query]

    # Additional filters
    filters = []
    params = []

    if lang:
        filters.append("language = %s")
        params.append(lang)
    if series:
        filters.append("series_name = %s")
        params.append(series)

    filter_sql = (" AND " + " AND ".join(filters)) if filters else ""

    # ---------- COUNT ----------
    count_sql = f"""
        SELECT COUNT(*) FROM discourses
        WHERE {base_where}
        {filter_sql}
    """
    count_params = params_for_base + params

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(count_sql, count_params)
        total_found = cur.fetchone()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error (count): {e}")
    finally:
        cur.close()
        release_conn(conn)

    # ---------- SEARCH (paginated) ----------
    search_sql = f"""
        SELECT
            title,
            slug,
            COALESCE(ts_rank_cd(search_vector, plainto_tsquery('simple', %s)), 0) AS ft_rank,
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
        WHERE {base_where}
        {filter_sql}
        ORDER BY ft_rank DESC, trigram_sim DESC
        LIMIT %s OFFSET %s
    """
    search_params = [query, query, query] + params_for_base + params + [limit, offset]

    results = []
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(search_sql, search_params)
        rows = cur.fetchall()
        for r in rows:
            ft_rank = r["ft_rank"] or 0.0
            trigram = r["trigram_sim"] or 0.0
            score = max(ft_rank, trigram)
            results.append({
                "title": r["title"],
                "slug": r["slug"],
                "score": float(score),
                "ft_rank": float(ft_rank),
                "trigram_sim": float(trigram),
                "language": r["language"],
                "series_name": r["series_name"],
                "audioFile": r["audiofile"],
                "highlight": r["highlight"] or ""
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error (search): {e}")
    finally:
        cur.close()
        release_conn(conn)

    total_pages = ceil(total_found / limit) if limit else 1

    return {
        "query": query,
        "total_found": total_found,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "results": results
    }
