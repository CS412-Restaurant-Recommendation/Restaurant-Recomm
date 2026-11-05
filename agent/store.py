# file: agent/store.py
#
# Read-only access to the agent data built by build_agent_store.py:
# restaurant details, the SQLite review store, and user first names.

import re
import sqlite3
import threading
from pathlib import Path

import pandas as pd

data_dir = Path(__file__).resolve().parent.parent / "Dataset"

details = pd.read_pickle(data_dir / "business_details.pkl").set_index("business_id")
user_names = pd.read_pickle(data_dir / "user_names.pkl")

SNIPPET_CHARS = 300

# SQLite connections can't be shared across threads, and FastAPI runs sync
# endpoints in a thread pool, so each thread opens its own read-only connection.
_local = threading.local()


def _conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(f"file:{data_dir / 'reviews.db'}?mode=ro", uri=True)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def _clip(text, n=SNIPPET_CHARS):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def get_user_name(user_id):
    return user_names.get(user_id)


def get_details(business_id):
    """Address, hours, price range and amenities; None if unknown. Missing values are dropped."""
    if business_id not in details.index:
        return None
    row = details.loc[business_id].to_dict()
    clean = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            if value:
                clean[key] = value
        elif not pd.isna(value):
            clean[key] = value.item() if hasattr(value, "item") else value
    return clean


def rating_summary(business_id):
    """Star histogram from the reviews in our dataset; None if the business has none."""
    row = _conn().execute(
        "SELECT * FROM business_review_stats WHERE business_id = ?", (business_id,)
    ).fetchone()
    if row is None:
        return None
    n = row["n_reviews"]
    histogram = {f"{s}_star": row[f"stars_{s}"] for s in range(5, 0, -1)}
    return {
        "reviews_in_dataset": n,
        "avg_stars_in_dataset": round(row["avg_stars"], 2),
        "histogram": histogram,
        "percent": {k: round(100 * v / n, 1) for k, v in histogram.items()},
    }


def get_reviews(business_id, min_stars=None, max_stars=None, limit=5, order="highest"):
    """Reviews of one business, optionally filtered by star range, sorted by stars."""
    sql = "SELECT stars, text FROM reviews WHERE business_id = ?"
    params = [business_id]
    if min_stars is not None:
        sql += " AND stars >= ?"
        params.append(int(min_stars))
    if max_stars is not None:
        sql += " AND stars <= ?"
        params.append(int(max_stars))
    sql += " ORDER BY stars " + ("ASC" if order == "lowest" else "DESC") + ", rowid LIMIT ?"
    params.append(int(limit))
    return [{"stars": r["stars"], "text": _clip(r["text"])} for r in _conn().execute(sql, params)]


def search_reviews(business_id, keywords, limit=8):
    """
    Full-text search within one business's reviews (stemmed, any keyword matches).
    Returns (total_matches, [{stars, snippet}]) ranked by relevance.
    """
    # Keep only word tokens and quote them, so free text from the model can
    # never produce an FTS5 syntax error.
    words = re.findall(r"\w+", keywords.lower())
    words = [w for w in words if w not in {"and", "or", "not", "near"}]
    if not words:
        return 0, []
    match = f'business_id:"{business_id}" AND text:(' + " OR ".join(f'"{w}"' for w in words) + ")"

    conn = _conn()
    total = conn.execute(
        "SELECT COUNT(*) FROM reviews_fts WHERE reviews_fts MATCH ?", (match,)
    ).fetchone()[0]
    rows = conn.execute(
        """
        SELECT r.stars, snippet(reviews_fts, 0, '**', '**', '…', 40) AS snippet
        FROM reviews_fts JOIN reviews r ON r.rowid = reviews_fts.rowid
        WHERE reviews_fts MATCH ?
        ORDER BY bm25(reviews_fts, 1.0, 0.0)
        LIMIT ?
        """,
        (match, int(limit)),
    ).fetchall()
    return total, [{"stars": r["stars"], "snippet": " ".join(r["snippet"].split())} for r in rows]
