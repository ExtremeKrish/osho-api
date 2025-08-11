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

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set specific origins ["https://example.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    query: str = Query(..., min_length=1, description="Exact phrase to search in description"),
    limit: int = Query(10, ge=1, le=200),
    page: int = Query(1, ge=1),
    lang: Optional[str] = None,
    series: Optional[str] = None,
):
    offset = (page - 1) * limit

    # Build WHERE clause
    filters = ["description ILIKE %s"]  # Exact phrase match, case-insensitive
    params = [f"%{query}%"]

    if lang:
        filters.append("language = %s")
        params.append(lang)
    if series:
        filters.append("series_name = %s")
        params.append(series)

    filter_sql = " AND ".join(filters)

    # ---------- COUNT ----------
    count_sql = f"SELECT COUNT(*) FROM discourses WHERE {filter_sql}"
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(count_sql, params)
        total_found = cur.fetchone()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error (count): {e}")
    finally:
        cur.close()
        conn.close()

    # ---------- FETCH PAGINATED ----------
    search_sql = f"""
        SELECT
            title,
            slug,
            language,
            series_name,
            audioFile,
            description
        FROM discourses
        WHERE {filter_sql}
        ORDER BY id ASC
        LIMIT %s OFFSET %s
    """
    params_with_limit = params + [limit, offset]

    results = []
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(search_sql, params_with_limit)
        rows = cur.fetchall()
        for r in rows:
            desc = r["description"]
            lower_desc = desc.lower()
            lower_query = query.lower()
        
            if lower_query in lower_desc:
                # Find where it matched
                start_idx = lower_desc.index(lower_query)
                end_idx = start_idx + len(query)
        
                # Split description into words
                words = desc.split()
                
                # Find the word index of the match
                char_count = 0
                match_word_index = None
                for i, w in enumerate(words):
                    if char_count <= start_idx < char_count + len(w):
                        match_word_index = i
                        break
                    char_count += len(w) + 1  # +1 for space
        
                if match_word_index is not None:
                    start_word = max(match_word_index - 5, 0)
                    end_word = min(match_word_index + 6, len(words))
                    snippet_words = words[start_word:end_word]
                    
                    # Highlight matched word(s)
                    snippet_words[match_word_index - start_word] = (
                        f"<b>{snippet_words[match_word_index - start_word]}</b>"
                    )
        
                    highlight = " ".join(snippet_words)
                else:
                    highlight = desc  # fallback
            else:
                highlight = desc  # no match
        
            results.append({
                "title": r["title"],
                "slug": r["slug"],
                "language": r["language"],
                "series_name": r["series_name"],
                "audioFile": r["audiofile"],
                "highlight": highlight
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error (search): {e}")
    finally:
        cur.close()
        conn.close()

    total_pages = ceil(total_found / limit) if limit else 1

    return {
        "query": query,
        "total_found": total_found,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
        "results": results
    }