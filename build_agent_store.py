# file: build_agent_store.py
#
# One-time build of the data the chat agent needs beyond the recommender pickles:
#   1. business_details.pkl  - address, hours, price range, amenities per restaurant
#   2. reviews.db            - SQLite review store with full-text search (FTS5)
#                              and per-business rating stats
#   3. user_names.pkl        - user_id -> first name, for friendly greetings
#
# Only restaurants in business_df.pkl (the recommender's catalogue) are kept.
#
# Usage:
#   python build_agent_store.py                 # build everything
#   python build_agent_store.py --only reviews  # rebuild one part

import argparse
import ast
import json
import sqlite3
import time
import zipfile
from pathlib import Path

import pandas as pd

data_dir = Path(__file__).resolve().parent / "Dataset"

business_path = data_dir / "yelp_academic_dataset_business.json"
review_path = data_dir / "filtered_yelp_review.json"
user_path = data_dir / "filtered_yelp_user.json"
archive_path = data_dir / "Archive.zip"  # holds filtered_yelp_review.json if not extracted

details_out = data_dir / "business_details.pkl"
reviews_db_out = data_dir / "reviews.db"
user_names_out = data_dir / "user_names.pkl"

BATCH_SIZE = 50_000

# Yelp attribute name -> our column name. Values are parsed with ast.literal_eval.
SCALAR_ATTRIBUTES = {
    "RestaurantsPriceRange2": "price_range",
    "RestaurantsTakeOut": "takeout",
    "RestaurantsDelivery": "delivery",
    "OutdoorSeating": "outdoor_seating",
    "RestaurantsReservations": "reservations",
    "RestaurantsGoodForGroups": "good_for_groups",
    "GoodForKids": "good_for_kids",
    "RestaurantsTableService": "table_service",
    "WheelchairAccessible": "wheelchair_accessible",
    "HappyHour": "happy_hour",
    "DogsAllowed": "dogs_allowed",
    "DriveThru": "drive_thru",
    "Caters": "caters",
    "HasTV": "has_tv",
    "BusinessAcceptsCreditCards": "accepts_credit_cards",
    "WiFi": "wifi",
    "Alcohol": "alcohol",
    "NoiseLevel": "noise_level",
    "RestaurantsAttire": "attire",
}
# Dict-valued attributes are reduced to the list of keys that are True,
# e.g. BusinessParking "{'lot': True, 'street': False}" -> ["lot"]
DICT_ATTRIBUTES = {
    "BusinessParking": "parking",
    "Ambience": "ambience",
    "GoodForMeal": "good_for_meal",
}


def load_catalogue_ids():
    business = pd.read_pickle(data_dir / "business_df.pkl")
    return set(business["business_id"])

# ==========================
# 1. BUSINESS DETAILS
# ==========================

def parse_attribute(raw):
    """Parse a Yelp attribute string ("True", "u'free'", "2", "None") to a Python value."""
    if raw is None:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def parse_hours(hours):
    """{"Monday": "8:0-18:30"} -> {"Monday": "08:00-18:30"}"""
    if not hours:
        return None

    def fmt(t):
        h, m = t.split(":")
        return f"{int(h):02d}:{int(m):02d}"

    out = {}
    for day, span in hours.items():
        try:
            start, end = span.split("-")
            out[day] = f"{fmt(start)}-{fmt(end)}"
        except ValueError:
            out[day] = span
    return out


def build_business_details(keep_ids):
    rows = []
    with open(business_path, "r", encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            if record["business_id"] not in keep_ids:
                continue

            attrs = record.get("attributes") or {}
            row = {
                "business_id": record["business_id"],
                "address": record.get("address"),
                "state": record.get("state"),
                "postal_code": record.get("postal_code"),
                "latitude": record.get("latitude"),
                "longitude": record.get("longitude"),
                "is_open": bool(record.get("is_open")),
                "hours": parse_hours(record.get("hours")),
            }
            for yelp_name, col in SCALAR_ATTRIBUTES.items():
                value = parse_attribute(attrs.get(yelp_name))
                # Strings like 'free' / 'full_bar' stay strings; "None" becomes None
                row[col] = value
            for yelp_name, col in DICT_ATTRIBUTES.items():
                value = parse_attribute(attrs.get(yelp_name))
                row[col] = sorted(k for k, v in value.items() if v) if isinstance(value, dict) else None
            rows.append(row)

    details = pd.DataFrame(rows)
    details["price_range"] = pd.to_numeric(details["price_range"], errors="coerce").astype("Int64")
    details.to_pickle(details_out)

    print("Saved business_details.pkl for", len(details), "of", len(keep_ids), "restaurants.")
    missing = len(keep_ids) - details["business_id"].nunique()
    if missing:
        print("WARNING:", missing, "catalogue restaurants not found in", business_path.name)

# ==========================
# 2. REVIEW STORE (SQLite + FTS5)
# ==========================

def open_review_source():
    """Read the extracted review JSON if present, else stream it straight from Archive.zip."""
    if review_path.exists():
        print("Reading reviews from", review_path.name)
        return open(review_path, "rb")
    print("Streaming reviews from", archive_path.name, "->", review_path.name)
    return zipfile.ZipFile(archive_path).open(review_path.name)


def build_review_store(keep_ids):
    # Build into a temp file and swap in at the end, so a failed run never
    # leaves a half-built reviews.db behind.
    tmp_path = reviews_db_out.with_suffix(".db.tmp")
    tmp_path.unlink(missing_ok=True)

    conn = sqlite3.connect(tmp_path)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("""
        CREATE TABLE reviews (
            review_id   TEXT PRIMARY KEY,
            business_id TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            stars       INTEGER NOT NULL,
            text        TEXT NOT NULL
        )
    """)

    insert_sql = "INSERT OR IGNORE INTO reviews VALUES (?, ?, ?, ?, ?)"
    batch = []
    seen = kept = 0
    start = time.time()

    with open_review_source() as fin:
        for line in fin:
            seen += 1
            record = json.loads(line)
            if record["business_id"] not in keep_ids or not record.get("text"):
                continue
            batch.append((
                record["review_id"],
                record["business_id"],
                record["user_id"],
                int(record["stars"]),
                record["text"],
            ))
            if len(batch) >= BATCH_SIZE:
                conn.executemany(insert_sql, batch)
                kept += len(batch)
                batch = []
                if kept % 500_000 == 0:
                    print(f"  {kept:,} reviews kept / {seen:,} read ({time.time() - start:.0f}s)")
    if batch:
        conn.executemany(insert_sql, batch)
        kept += len(batch)
    conn.commit()
    print(f"Inserted {kept:,} of {seen:,} reviews ({time.time() - start:.0f}s).")

    print("Creating indexes...")
    conn.execute("CREATE INDEX idx_reviews_business ON reviews(business_id, stars)")
    conn.execute("CREATE INDEX idx_reviews_user ON reviews(user_id)")

    # External-content FTS5 index over review text (stemmed, so "parking"
    # also matches "park" / "parked"). business_id is indexed too, so a search
    # within one restaurant is a single FTS lookup instead of matching the
    # keyword across all reviews first. Query example:
    #   reviews_fts MATCH 'business_id:"<id>" AND text:(parking)'
    #   ORDER BY bm25(reviews_fts, 1.0, 0.0)   -- rank on text only
    print("Building full-text index...")
    conn.execute("""
        CREATE VIRTUAL TABLE reviews_fts USING fts5(
            text, business_id, content='reviews', content_rowid='rowid',
            tokenize='porter unicode61'
        )
    """)
    conn.execute("INSERT INTO reviews_fts(reviews_fts) VALUES ('rebuild')")
    conn.execute("INSERT INTO reviews_fts(reviews_fts) VALUES ('optimize')")

    # Per-business rating summary (computed from the reviews in this dataset,
    # which can differ from Yelp's own review_count)
    print("Computing per-business rating stats...")
    conn.execute("""
        CREATE TABLE business_review_stats AS
        SELECT
            business_id,
            COUNT(*)                                  AS n_reviews,
            ROUND(AVG(stars), 3)                      AS avg_stars,
            SUM(CASE WHEN stars = 1 THEN 1 ELSE 0 END) AS stars_1,
            SUM(CASE WHEN stars = 2 THEN 1 ELSE 0 END) AS stars_2,
            SUM(CASE WHEN stars = 3 THEN 1 ELSE 0 END) AS stars_3,
            SUM(CASE WHEN stars = 4 THEN 1 ELSE 0 END) AS stars_4,
            SUM(CASE WHEN stars = 5 THEN 1 ELSE 0 END) AS stars_5
        FROM reviews
        GROUP BY business_id
    """)
    conn.execute("CREATE UNIQUE INDEX idx_stats_business ON business_review_stats(business_id)")
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    tmp_path.replace(reviews_db_out)
    size_gb = reviews_db_out.stat().st_size / 1e9
    print(f"Saved reviews.db ({size_gb:.2f} GB) in {time.time() - start:.0f}s.")

# ==========================
# 3. USER NAMES
# ==========================

def build_user_names():
    # Only users the recommender knows (those with an inferred home city)
    known_users = set(pd.read_pickle(data_dir / "user_city_map.pkl").index)

    users = pd.read_json(user_path, lines=True)[["user_id", "name"]]
    users = users.dropna(subset=["user_id", "name"])
    users = users[users["user_id"].isin(known_users)].drop_duplicates("user_id")

    pd.Series(users["name"].values, index=users["user_id"]).to_pickle(user_names_out)
    print("Saved user_names.pkl for", len(users), "of", len(known_users), "known users.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the chat agent's data store.")
    parser.add_argument("--only", choices=["details", "reviews", "users"],
                        help="build just one part (default: all)")
    args = parser.parse_args()

    if args.only in (None, "details", "reviews"):
        keep_ids = load_catalogue_ids()
        print("Catalogue restaurants:", len(keep_ids))
    if args.only in (None, "details"):
        build_business_details(keep_ids)
    if args.only in (None, "reviews"):
        build_review_store(keep_ids)
    if args.only in (None, "users"):
        build_user_names()
