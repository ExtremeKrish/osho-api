# main.py
import os
from math import ceil
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import psycopg2
import psycopg2.extras

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("Please set DATABASE_URL in .env")

app = FastAPI(title="Osho Discourse Search API")


def get_conn():
    # Simple connection per request. For higher load, swap to a pool.
    return psycopg2.connect(DB_URL)


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

    # base WHERE clause (full-text OR trigram)
    # We'll build SQL and params carefully to avoid injection.
    base_where = "(search_vector @@ plainto_tsquery('simple', %s) OR description %% %s)"
    params_for_base = [query, query]  # used multiple times below

    # Additional filters
    filters = []
    params = []  # final params list used for each statement

    # We'll build a 'filter_sql' string and attach params accordingly.
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
    # params order for count: [query, query] + params
    count_params = params_for_base + params

    # run count
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(count_sql, count_params)
        total_found = cur.fetchone()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error (count): {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass

    # ---------- SEARCH (paginated) ----------
    # We compute:
    #   - ft_rank = ts_rank_cd(search_vector, plainto_tsquery('simple', query))
    #   - trigram_sim = similarity(description, query)
    # final order: first by ft_rank desc (if present), then by trigram_sim desc
    # We also produce a highlighted snippet with ts_headline (simple config).
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

    # params for search:
    # Notice order matters: first plainto_tsquery for ft_rank, then similarity param,
    # then plainto_tsquery for ts_headline, then base_where's query params, then filters, then limit, offset.
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
            # final score: combine both. simple: max of the two normalized measures.
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
                "highlight": r["highlight"] or (r["description"][:200] + "..." if r.get("description") else "")
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error (search): {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass

    total_pages = ceil(total_found / limit) if limit else 1

    return {
        "query": query,
        "total_found": total_found,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "results": results
    }