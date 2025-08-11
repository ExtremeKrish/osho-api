# main.py
import os
from math import ceil
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
import re
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

def extract_highlight(text: str, query: str, context_words: int = 5):
    # Replace <br> with spaces and remove other HTML tags
    clean_text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"<.*?>", "", clean_text)

    # Tokenize words preserving original case
    words = clean_text.split()
    lower_words = [w.lower() for w in words]
    query_words = query.lower().split()

    # Find the starting index of the exact phrase
    for i in range(len(lower_words) - len(query_words) + 1):
        if lower_words[i:i+len(query_words)] == query_words:
            start_word = max(0, i - context_words)
            end_word = min(len(words), i + len(query_words) + context_words)
            snippet = words[start_word:end_word]

            # Highlight the matched phrase
            match_in_snippet_start = i - start_word
            match_in_snippet_end = match_in_snippet_start + len(query_words)
            snippet[match_in_snippet_start:match_in_snippet_end] = [
                f"<b>{w}</b>" for w in snippet[match_in_snippet_start:match_in_snippet_end]
            ]

            return " ".join(snippet)

    return text  # fallback if not found


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
            highlight = extract_highlight(r["description"], query, context_words=5)
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