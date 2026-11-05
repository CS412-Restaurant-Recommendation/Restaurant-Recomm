# file: agent/tools.py
#
# Tools the chat agent can call. Each tool is a plain function fn(ctx, **args)
# plus a hand-written Gemini FunctionDeclaration (explicit JSON Schema).
#
# ctx is the chat Session (see agent/agent.py). The user's identity comes from
# ctx.user_id, which the server sets; it is never a declared argument, so the
# model cannot choose whose data to read.

import difflib
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from google.genai import types

import recommend as rec
from agent import store
from new_user_recommendation import filter_by_categories, get_new_user_recommendations


class ToolError(Exception):
    """A problem the model can fix (unknown id, bad value); returned as {"error": ...}."""


@dataclass
class ToolSpec:
    fn: Callable
    declaration: types.FunctionDeclaration


TOOLS: dict[str, ToolSpec] = {}


def tool(description, properties=None, required=()):
    """Register fn as a tool with an explicit JSON-Schema declaration."""
    def register(fn):
        declaration = types.FunctionDeclaration(
            name=fn.__name__,
            description=description,
            parameters_json_schema={
                "type": "object",
                "properties": properties or {},
                "required": list(required),
            },
        )
        TOOLS[fn.__name__] = ToolSpec(fn, declaration)
        return fn
    return register

# ==========================
# Shared helpers
# ==========================

business = rec.business
KNOWN_CATEGORIES = set(rec.business_features.columns)
# Canonical spelling per lowercase city (the data mixes e.g. "AVON" and "Avon")
CITY_NAMES = business.groupby(business["city"].str.lower())["city"].agg(lambda s: s.mode().iloc[0]).to_dict()

BUSINESS_ID = {"type": "string", "description": "Restaurant business_id from a previous tool result."}
CUISINES = {
    "type": "array", "items": {"type": "string"},
    "description": "Yelp category names, lowercase, e.g. ['italian'], ['indian', 'vegetarian']. "
                   "Call list_cuisines if unsure of the exact name.",
}


def _clamp(value, low, high):
    return max(low, min(high, int(value)))


def _restaurant(row, extra=()):
    """Compact restaurant record from a business_df-like row."""
    out = {
        "business_id": row["business_id"],
        "name": row["name"],
        "city": row["city"],
        "stars": float(row["stars"]),
        "review_count": int(row["review_count"]),
        "categories": row["categories"],
    }
    for col in extra:
        if col in row and pd.notna(row[col]):
            out[col] = round(float(row[col]), 3)
    return out


def _get_business(business_id):
    rows = business[business["business_id"] == business_id]
    if rows.empty:
        raise ToolError(f"Unknown business_id '{business_id}'. Use search_restaurants to find the id.")
    return rows.iloc[0]


def _split_cuisines(cuisines):
    """-> (known lowercase categories, unknown names)"""
    wanted = [c.strip().lower() for c in (cuisines or []) if c and c.strip()]
    return [c for c in wanted if c in KNOWN_CATEGORIES], [c for c in wanted if c not in KNOWN_CATEGORIES]


def _price(level):
    return "$" * int(level) if level else None

# ==========================
# Recommendations
# ==========================

@tool(
    "Recommend restaurants for the current user. Personalized for known users "
    "(collaborative filtering + category similarity, in their home city); based on stated "
    "preferences for onboarded new users; otherwise popular highly-rated places.",
    {
        "n": {"type": "integer", "description": "How many to return (1-20).", "default": 5},
        "cuisines": CUISINES,
        "exclude_ids": {"type": "array", "items": {"type": "string"},
                        "description": "business_ids to skip, e.g. ones already shown."},
    },
)
def recommend_restaurants(ctx, n=5, cuisines=None, exclude_ids=None):
    n = _clamp(n, 1, 20)
    categories, unknown = _split_cuisines(cuisines)
    if cuisines and not categories:
        raise ToolError(f"Unknown cuisine(s) {unknown}. Call list_cuisines to see valid names.")

    if ctx.user_known:
        meta = rec.get_recommendations_with_meta(
            ctx.user_id, top_n=n, categories=categories, exclude_biz_ids=exclude_ids
        )
        result = {"source": meta["source"], "fallback_reason": meta["fallback_reason"],
                  "home_city": meta["city"],
                  "recommendations": [_restaurant(r, ("cf_score", "cat_sim")) for r in meta["recommendations"]]}

    elif ctx.onboarding:
        prefs = ctx.onboarding
        profile = rec.build_profile_from_categories(prefs["cuisines"])
        df = rec.recommend_from_profile(profile, city=prefs["city"], top_n=n,
                                        categories=categories, exclude_biz_ids=exclude_ids)
        if profile is not None and not df.empty:
            result = {"source": "stated_preferences", "preferences": prefs,
                      "recommendations": [_restaurant(r, ("cat_sim",)) for _, r in df.iterrows()]}
        else:
            recs = get_new_user_recommendations(top_n=n, city=prefs["city"], categories=categories,
                                                exclude_biz_ids=exclude_ids)
            result = {"source": "cold_start", "fallback_reason": "no_preference_matches",
                      "preferences": prefs, "recommendations": [_restaurant(r) for r in recs]}

    else:
        recs = get_new_user_recommendations(top_n=n, categories=categories, exclude_biz_ids=exclude_ids)
        result = {"source": "cold_start", "fallback_reason": "new_user_not_onboarded",
                  "hint": "Ask the user for favourite cuisines and city, then call set_onboarding_preferences.",
                  "recommendations": [_restaurant(r) for r in recs]}

    if unknown:
        result["ignored_unknown_cuisines"] = unknown
    for i, r in enumerate(result["recommendations"], start=1):
        r["rank"] = i
    found = len(result["recommendations"])
    if found < n:
        result["note"] = (f"Only {found} of the {n} requested restaurants match these filters. "
                          "Present these and tell the user; don't keep searching for more.")
    return result


@tool(
    "Explain why a restaurant is recommended to the current user: how similar users rated "
    "it (collaborative filtering) and which categories it shares with the user's taste.",
    {"business_id": BUSINESS_ID},
    required=["business_id"],
)
def explain_recommendation(ctx, business_id):
    _get_business(business_id)
    explanation = rec.explain_recommendation(ctx.user_id, business_id)
    if not ctx.user_known:
        explanation["note"] = "New user: no rating history, so there is no personalized evidence."
        if ctx.onboarding:
            cats = set(_get_business(business_id)["category_list"])
            explanation["matches_stated_cuisines"] = sorted(cats & set(ctx.onboarding["cuisines"]))
    return explanation


@tool(
    "Find restaurants similar to a given one by category profile.",
    {
        "business_id": BUSINESS_ID,
        "n": {"type": "integer", "description": "How many (1-10).", "default": 5},
        "same_city": {"type": "boolean", "description": "Only the same city (default true).", "default": True},
    },
    required=["business_id"],
)
def similar_restaurants(ctx, business_id, n=5, same_city=True):
    target = _get_business(business_id)
    df = rec.similar_restaurants(business_id, top_n=_clamp(n, 1, 10), same_city=bool(same_city))
    return {"similar_to": _restaurant(target),
            "results": [_restaurant(r, ("cat_sim",)) for _, r in df.iterrows()],
            "note": None if not df.empty else "No restaurants share categories with this one in scope."}

# ==========================
# Restaurant lookup and details
# ==========================

@tool(
    "Search the restaurant catalogue by (partial) name, cuisine and/or city. Use this to turn "
    "a restaurant name the user mentions into a business_id.",
    {
        "name": {"type": "string", "description": "Full or partial restaurant name."},
        "cuisine": {"type": "string", "description": "One category name, e.g. 'thai'."},
        "city": {"type": "string"},
        "limit": {"type": "integer", "description": "Max results (1-20).", "default": 10},
    },
)
def search_restaurants(ctx, name=None, cuisine=None, city=None, limit=10):
    if not (name or cuisine or city):
        raise ToolError("Give at least one of name, cuisine or city.")
    df = business
    if city:
        df = df[df["city"].str.lower() == city.strip().lower()]
    if cuisine:
        df = filter_by_categories(df, [cuisine])
    if name:
        needle = name.strip().lower()
        lowered = df["name"].str.lower()
        df = df[lowered.str.contains(needle, regex=False)]
        df = df.assign(_exact=(lowered[df.index] == needle))
        df = df.sort_values(["_exact", "review_count"], ascending=[False, False])
    else:
        df = df.sort_values("review_count", ascending=False)
    return {"total_matches": int(len(df)),
            "results": [_restaurant(r) for _, r in df.head(_clamp(limit, 1, 20)).iterrows()]}


@tool(
    "Get a restaurant's details: address, hours, price range, open/closed, and amenities "
    "(takeout, delivery, outdoor seating, parking, wifi, alcohol, noise level, ...).",
    {"business_id": BUSINESS_ID},
    required=["business_id"],
)
def get_restaurant_details(ctx, business_id):
    result = _restaurant(_get_business(business_id))
    details = store.get_details(business_id) or {}
    if "price_range" in details:
        details["price"] = _price(details.pop("price_range"))
    result.update(details)
    return result


@tool(
    "Get a restaurant's rating summary: Yelp's overall stars/review count and the star "
    "histogram of the reviews in our dataset.",
    {"business_id": BUSINESS_ID},
    required=["business_id"],
)
def get_rating_summary(ctx, business_id):
    biz = _get_business(business_id)
    return {"business_id": business_id, "name": biz["name"],
            "yelp_stars": float(biz["stars"]), "yelp_review_count": int(biz["review_count"]),
            "dataset": store.rating_summary(business_id)}


@tool(
    "Compare 2-4 restaurants side by side (stars, price, rating histogram, key amenities).",
    {"business_ids": {"type": "array", "items": {"type": "string"}, "description": "2-4 business_ids."}},
    required=["business_ids"],
)
def compare_restaurants(ctx, business_ids):
    if not 2 <= len(business_ids) <= 4:
        raise ToolError("Give between 2 and 4 business_ids.")
    keys = ("price_range", "is_open", "noise_level", "alcohol", "outdoor_seating",
            "delivery", "takeout", "reservations", "parking")
    rows = []
    for bid in business_ids:
        row = _restaurant(_get_business(bid))
        details = store.get_details(bid) or {}
        row.update({k: details[k] for k in keys if k in details})
        if "price_range" in row:
            row["price"] = _price(row.pop("price_range"))
        summary = store.rating_summary(bid)
        if summary:
            row["dataset_percent_by_star"] = summary["percent"]
        rows.append(row)
    return {"restaurants": rows}

# ==========================
# Reviews
# ==========================

@tool(
    "Read reviews of a restaurant, optionally only within a star range (e.g. max_stars=2 "
    "for complaints). Sorted by stars; the data has no review dates.",
    {
        "business_id": BUSINESS_ID,
        "min_stars": {"type": "integer", "description": "1-5"},
        "max_stars": {"type": "integer", "description": "1-5"},
        "limit": {"type": "integer", "description": "1-10", "default": 5},
        "order": {"type": "string", "enum": ["highest", "lowest"], "default": "highest"},
    },
    required=["business_id"],
)
def get_reviews(ctx, business_id, min_stars=None, max_stars=None, limit=5, order="highest"):
    biz = _get_business(business_id)
    reviews = store.get_reviews(business_id, min_stars, max_stars, _clamp(limit, 1, 10), order)
    return {"business_id": business_id, "name": biz["name"], "reviews": reviews}


@tool(
    "Search a restaurant's reviews for keywords about an aspect (e.g. 'parking', "
    "'service waiter', 'price expensive', 'wait time'). Any keyword matches; word stems match.",
    {
        "business_id": BUSINESS_ID,
        "keywords": {"type": "string", "description": "Space-separated keywords."},
        "limit": {"type": "integer", "description": "1-15", "default": 8},
    },
    required=["business_id", "keywords"],
)
def search_reviews(ctx, business_id, keywords, limit=8):
    biz = _get_business(business_id)
    total, snippets = store.search_reviews(business_id, keywords, _clamp(limit, 1, 15))
    return {"business_id": business_id, "name": biz["name"], "keywords": keywords,
            "matching_reviews": total, "snippets": snippets}

# ==========================
# The user's own taste
# ==========================

@tool("Summarize the current user's taste: home city, rating habits and favourite categories.")
def get_my_taste_profile(ctx):
    if not ctx.user_known:
        return {"user_known": False, "stated_preferences": ctx.onboarding,
                "note": "New user: no rating history."}
    taste = rec.get_user_taste(ctx.user_id, top_restaurants=5)
    if taste is None:
        return {"user_known": True, "note": "No ratings found in the training data."}
    taste.pop("user_id", None)
    return taste


@tool(
    "List the restaurants the current user rated highest.",
    {"limit": {"type": "integer", "description": "1-20", "default": 10}},
)
def get_my_history(ctx, limit=10):
    if not ctx.user_known:
        return {"user_known": False, "restaurants": [], "note": "New user: no rating history."}
    taste = rec.get_user_taste(ctx.user_id, top_restaurants=_clamp(limit, 1, 20))
    return {"restaurants": taste["top_rated_restaurants"] if taste else []}

# ==========================
# Catalogue and onboarding
# ==========================

@tool(
    "List the most common restaurant categories (cuisines etc.), optionally in one city, "
    "with restaurant counts. Use to map the user's words to exact category names.",
    {"city": {"type": "string"}, "limit": {"type": "integer", "description": "1-100", "default": 50}},
)
def list_cuisines(ctx, city=None, limit=50):
    df = business
    if city:
        df = df[df["city"].str.lower() == city.strip().lower()]
        if df.empty:
            raise ToolError(f"No restaurants in city '{city}'.")
    counts = Counter(cat for cats in df["category_list"] for cat in cats)
    return {"city": city, "categories": [{"category": c, "restaurants": k}
                                         for c, k in counts.most_common(_clamp(limit, 1, 100))]}


@tool(
    "Save a NEW user's stated favourite cuisines and city so recommendations can be "
    "personalized to them. Only for users without rating history.",
    {"cuisines": CUISINES, "city": {"type": "string"}},
    required=["cuisines", "city"],
)
def set_onboarding_preferences(ctx, cuisines, city):
    if ctx.user_known:
        raise ToolError("This user already has rating history; onboarding is only for new users.")
    canonical = CITY_NAMES.get(city.strip().lower())
    if canonical is None:
        close = difflib.get_close_matches(city.strip().lower(), CITY_NAMES.keys(), n=3)
        raise ToolError(f"Unknown city '{city}'. Closest matches: {[CITY_NAMES[c] for c in close]}")
    categories, unknown = _split_cuisines(cuisines)
    if not categories:
        raise ToolError(f"Unknown cuisine(s) {unknown}. Call list_cuisines to see valid names.")
    ctx.onboarding = {"cuisines": categories, "city": canonical}
    return {"saved": ctx.onboarding, "ignored_unknown_cuisines": unknown}
