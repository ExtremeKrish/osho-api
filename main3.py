import os
from math import ceil
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
from psycopg2 import pool

# Load .env variables
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("Please set DATABASE_URL in .env")

# Connection pool instead of reconnecting each time
try:
    conn_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=DB_URL
    )
except Exception as e:
    raise RuntimeError(f"Failed to connect to database: {e}")

app = FastAPI(title="Osho Discourse Search API")

def get_conn():
    try:
        return conn_pool.getconn()
    except:
        raise HTTPException(status_code=500, detail="DB connection error")

def release_conn(conn):
    try:
        conn_pool.putconn(conn)
    except:
        pass

class SearchResponse(BaseModel):
    query: str
    total_found: int
    page: int
    total_pages: int
    limit: int
    results: list

@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=200),
    page: int = Query(1, ge=1),
    lang: Optional[str] = Query(None),
    series: Optional[str] = Query(None),
):
    offset = (page - 1) * limit

    base_where = "(search_vector @@ plainto_tsquery('simple', %s) OR description %% %s)"
    params_for_base = [query, query]

    filters = []
    params = []
    if lang:
        filters.append("language = %s")
        params.append(lang)
    if series:
        filters.append("series_name = %s")
        params.append(series)
    filter_sql = (" AND " + " AND ".join(filters)) if filters else ""

    # Count query
    count_sql = f"""
        SELECT COUNT(*) 
        FROM discourses
        WHERE {base_where} {filter_sql}
    """
    count_params = params_for_base + params

    # Search query
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
        WHERE {base_where} {filter_sql}
        ORDER BY ft_rank DESC, trigram_sim DESC
        LIMIT %s OFFSET %s
    """
    search_params = [query, query, query] + params_for_base + params + [limit, offset]

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(count_sql, count_params)
        total_found = cur.fetchone()[0]

        cur.close()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(search_sql, search_params)
        rows = cur.fetchall()

        results = []
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
                "highlight": r["highlight"]
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
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
