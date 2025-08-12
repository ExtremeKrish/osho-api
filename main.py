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
    # Remove <br> and all HTML tags
    clean_text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"<.*?>", "", clean_text)

    words = clean_text.split()
    lower_words = [w.lower() for w in words]
    query_words = query.lower().split()

    # Find the first exact phrase match
    for i in range(len(lower_words) - len(query_words) + 1):
        if lower_words[i:i+len(query_words)] == query_words:
            start = max(0, i - context_words)
            end = min(len(words), i + len(query_words) + context_words)
            snippet = words[start:end]

            # Wrap the exact phrase inside <b>...</b>
            match_start = i - start
            match_end = match_start + len(query_words)
            phrase = " ".join(snippet[match_start:match_end])
            snippet[match_start:match_end] = [f"<b>{phrase}</b>"]

            return " ".join(snippet)

    # If no match found, return empty string or some default text
    return ""


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1, description="Phrase to search"),
    limit: int = Query(10, ge=1, le=200),
    page: int = Query(1, ge=1),
    lang: Optional[str] = None,
    series: Optional[str] = None,
    exactMatch: Optional[bool] = Query(False, description="Use exact phrase match with phraseto_tsquery"),
):
    offset = (page - 1) * limit

    filters = []
    params = []

    # Choose which tsquery function to use based on exactMatch param
    tsquery_func = "phraseto_tsquery" if exactMatch else "plainto_tsquery"
    filters.append(f"search_vector @@ {tsquery_func}('simple', %s)")
    params.append(query)

    if lang:
        filters.append("language = %s")
        params.append(lang)
    if series:
        filters.append("series_name = %s")
        params.append(series)

    filter_sql = " AND ".join(filters)

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
    
    
@app.get("/jinda")
def ping():
    # This does nothing, just keeps the app awake
    return {"status": "ha"}
