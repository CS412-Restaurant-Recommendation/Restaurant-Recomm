# AI Agent Integration: Progress Tracker

This adds a Gemini-powered chat agent that answers **only from this project's data and models**. It has no internet access and no Google Search grounding. The agent calls our Python functions as tools (Gemini function calling) through a **hand-written tool-calling loop**; the SDK's automatic function calling is turned off. The user ID is bound to the chat session on the server and is never taken from the chat text.

UI: FastAPI `POST /chat` plus a Streamlit frontend.

## Status

| Step | Description | Status |
|---|---|---|
| 1 | Changes to existing code (filters, explanations, cold-start fix, new helpers) | ✅ Done (not committed) |
| 2 | `build_agent_store.py`: business details + SQLite review store with full-text search | ✅ Done (not committed) |
| 3 | `agent/` package (tools + hand-written Gemini tool loop), `POST /session` and `POST /chat`, Streamlit `app_chat.py` | ✅ Done (not committed); a few live checks pending Gemini quota |

---

## Step 1: Changes to existing code ✅

### `recommend.py`
- **Data path:** `data_dir` is now `Path(__file__).resolve().parent / "Dataset"`, so it works from any working directory.
- **`get_neighbors(user_id, k_neighbors=30)` (new):** the similar-users search, pulled out of `compute_cf_scores_for_user()`. It's cached per user with `lru_cache`, and callers must not mutate the returned Series. `compute_cf_scores_for_user()` now calls it, with the same output as before.
- **`recommend_for_user(..., categories=None, exclude_biz_ids=None)`:** new optional filters.
  - `categories`: case-insensitive exact match against `category_list`, e.g. `["italian", "pizza"]`.
  - `exclude_biz_ids`: skip restaurants already shown.
- **`get_recommendations_with_meta(user_id, top_n, min_review_count, categories, exclude_biz_ids)` (new):** returns:
  - `source`: `"personalized"` or `"cold_start"`
  - `fallback_reason`: `None`, `"unknown_user"` or `"no_personalized_results"`
  - `city`: the user's inferred home city (`None` for unknown users)
  - `recommendations`: list of dicts
- **`get_recommendations()`:** now a thin wrapper around the function above. Its signature and return type are unchanged, so `GET /recommendations` is backward compatible.
- **`explain_recommendation(user_id, business_id)` (new):** returns:
  - `restaurant`: basic info
  - `collaborative`: similar users considered, how many of them rated it, their weighted average rating and similarity, and `cf_score`
  - `content`: category similarity, shared categories, and restaurants the user rated 4★ or higher that share those categories
- **`build_profile_from_categories(categories)` (new):** a normalized category vector from stated preferences, for cold-start onboarding.
- **`recommend_from_profile(profile, city, top_n, categories, min_review_count, exclude_biz_ids)` (new):** content-only ranking (`cat_sim`, then `review_count`, then `stars`). The city match is case-insensitive.
- **`similar_restaurants(business_id, top_n=5, same_city=True)` (new):** item-item category cosine similarity. Places with zero shared categories are excluded.
- **`get_user_taste(user_id)` (new):** home city, review count, average stars given, top profile categories and top-rated restaurants.
- **`get_sample_users(n=20)` (new):** the known users (both in the CF matrix and in the city map) with the most training reviews.

### `new_user_recommendation.py`
- It loads `business_df.pkl` once at import. Before, it re-read `filtered_yelp_business.json` on every call, and that file only exists inside `Archive.zip`, so the path was broken.
- The IMDB-style weighted rating (`m = 300`) is computed once at import.
- `get_new_user_recommendations(top_n, city=None, categories=None, exclude_biz_ids=None)` takes new optional filters, and results now include `city`.
- New shared helper `filter_by_categories(df, categories)`. It lives here to avoid a circular import, since `recommend.py` imports from this module.

### `main.py`
- New endpoint `GET /users/sample?n=20` for the chat UI's user picker. `GET /recommendations` is unchanged.

### `build_train_test_and_profiles.py`, `build_user_city.py`, `eval_recommender.py`
- Same file-relative `data_dir` fix (one line each).

### Verification
- **Regression:** the top-10 lists (`business_id`, `cf_score`, `cat_sim`) for 8 known users were identical before and after the refactor.
- The category filter, exclusions, cold-start path, explanations, taste profile, similar restaurants, sample users and both API handlers were all exercised successfully from outside the project directory.
- The full `eval_recommender.py` was not rerun. The ranking path is unchanged, so Hit@N should match.

### Notes / known behavior
- **Not done yet:** `POST /session` and `POST /chat` are deferred to Step 3, because they depend on the agent.
- **Cold-start ranking:** it now uses `business_df.pkl` (top 100 businesses per city) instead of the full filtered JSON, so results can differ slightly.
- **New user + cuisine, no city:** results come from all cities. Step 3 onboarding will ask for a city and use `recommend_from_profile(city=...)`.
- **Small cities:** with only 100 restaurants per city, a cuisine filter can come back empty (e.g. only one Greek place in West Chester). Personalized requests then fall back with `fallback_reason="no_personalized_results"`.
- **`data_cleaning.py`:** left untouched (a one-off script with hard-coded string paths).

---

## Step 2: Agent data store ✅

### New file: `build_agent_store.py`
Run `python build_agent_store.py` (or `--only details|reviews|users`). A full build takes about 2 minutes. Only restaurants in `business_df.pkl` (23,367) are kept, and no new packages are needed (`sqlite3` is in the standard library).

| Output (in `Dataset/`, git-ignored) | Contents |
|---|---|
| `business_details.pkl` (7 MB) | One row per restaurant (100% coverage) |
| `reviews.db` (2.84 GB) | SQLite review store with full-text search and rating stats |
| `user_names.pkl` (58 MB) | user_id → first name for 1,505,695 of 1,505,705 known users |

**`business_details.pkl`**
- Columns: `address`, `state`, `postal_code`, `latitude`, `longitude`, `is_open`, `hours` (normalized to `{"Monday": "08:00-18:30"}`) and `price_range` (1–4, `Int64`).
- Yes/no and text amenities: takeout, delivery, outdoor seating, reservations, good for groups, good for kids, table service, wheelchair accessible, happy hour, dogs allowed, drive-thru, caters, has TV, accepts credit cards, `wifi`, `alcohol`, `noise_level`, `attire`.
- `parking`, `ambience` and `good_for_meal` are lists of the options that are true, e.g. `["lot", "street"]`.
- Yelp attribute strings (`"True"`, `"u'free'"`, `"{'lot': True}"`, `"None"`) are parsed with `ast.literal_eval`. A missing attribute is `None`.

**`reviews.db`**
- `reviews(review_id PK, business_id, user_id, stars, text)` holds 2,612,372 reviews, indexed on `(business_id, stars)` and on `user_id`.
- `reviews_fts` is a full-text (FTS5) index over `text` **and** `business_id`. It uses Porter stemming, so "parking" also matches "park" and "parked".
- `business_review_stats(business_id, n_reviews, avg_stars, stars_1 … stars_5)` has one row per restaurant.
- The reviews are streamed straight from `Archive.zip`, so the 3.5 GB JSON is not extracted. An extracted `Dataset/filtered_yelp_review.json` is used instead if present.
- The database is built into `reviews.db.tmp` and swapped in at the end, so a failed run never leaves a half-built database.

**Query pattern for Step 3** (2–27 ms, including the restaurant with the most reviews):
```sql
SELECT r.* FROM reviews_fts JOIN reviews r ON r.rowid = reviews_fts.rowid
WHERE reviews_fts MATCH 'business_id:"<id>" AND text:(parking)'
ORDER BY bm25(reviews_fts, 1.0, 0.0)   -- rank on text only
LIMIT 5;
```
Filtering with `r.business_id = ?` outside the MATCH took 2.5 s for a typical restaurant and 45 s for the largest. That's why `business_id` is inside the index.

### Other changes
- **`README.md`:** added "Step 4 — Build the chat agent's data store" to the Data Pipeline section.

### Verification
- 100% of catalogue restaurants have details, and `business_id` is unique.
- Review count matches a separate counting pass: 2,612,372 of 5,126,287 reviews belong to catalogue restaurants.
- There are stats rows for all 23,367 restaurants, and `PRAGMA quick_check` reports ok.
- Keyword search, low-star filtering and rating stats all returned correct results. For example, Opa Taverna has 220 reviews, 4.34★ average, histogram 12/14/11/33/150.

### Notes / known behavior
- **No dates or useful votes:** the review file only has `review_id`, `user_id`, `business_id`, `stars` and `text`. Sorting by "most useful" or "most recent" isn't possible, so Step 3's `get_reviews` will sort by star rating or search relevance instead.
- **Two different counts:** `n_reviews` in `business_review_stats` counts the reviews in our dataset and can differ from Yelp's `review_count` (Opa Taverna: 220 vs 214). The agent should say which number it is quoting.
- **Ambiguous hours:** Yelp writes `"00:00-00:00"` for some days, which can mean closed or open 24 hours. It's kept as is.
- **Sentiment model skipped:** `sentiment_rating_model_1to5.joblib` is not in the repo. It can be regenerated from `sentimentAnalysis.ipynb` later if wanted.
- **Disk:** the outputs add about 2.9 GB to `Dataset/`.

## Step 3: Agent + chat UI ✅

### Files
| File | Change |
|---|---|
| `agent/__init__.py` | new, empty |
| `agent/store.py` | new: read access to `business_details.pkl`, `user_names.pkl` and `reviews.db` (one read-only SQLite connection per thread, since FastAPI uses a thread pool) |
| `agent/tools.py` | new: 13 tools, each a function `fn(ctx, **args)` registered with a hand-written JSON-Schema `FunctionDeclaration` via the `@tool(...)` decorator |
| `agent/agent.py` | new: Gemini client, system prompt, `Session`, `run_turn()` loop, `execute_tool()`, rate-limit retry, groundedness check |
| `main.py` | added `POST /session` and `POST /chat` (plus INFO logging) |
| `app_chat.py` | new: Streamlit chat UI with a user picker and a "debug" expander per reply |
| `tests/test_agent_loop.py`, `tests/__init__.py` | new: 14 offline tests of the loop using a fake Gemini client |
| `requirements.txt` | added `google-genai`, `streamlit`, `python-dotenv`, `requests` |
| `.gitignore` | added `.env` |
| `.env.example` | new template; your real `.env` holds the key and is git-ignored |
| `README.md` | new "AI Chat Agent" section (setup, run, endpoints, tests, free-tier note) |

Installed into `myenv`: `google-genai` 2.33.0, `streamlit` 1.66.1, `requests` 2.32.5.

**How to run:** `uvicorn main:app`, then `streamlit run app_chat.py`. To run the tests: `python -m tests.test_agent_loop`.

### The tool-calling loop (`agent/agent.py`)
- Automatic function calling is **off** (`AutomaticFunctionCallingConfig(disable=True)`), and tools are declared explicitly.
- Each `Session` keeps its own `history: list[types.Content]`.
- No Google Search tool is declared, so the model has no internet access.
- `run_turn()` is 37 lines and follows the approved design:
  1. Send the history and tools to the model.
  2. Append the model's turn **exactly as returned** (same object, so thought signatures survive).
  3. If the turn has function calls, run all of them and append all results in **one** user turn, in order, each `FunctionResponse` carrying the call's `id` and `name`.
  4. Call the model again, repeating until it returns plain text.

| # | Requirement | Implementation |
|---|---|---|
| 1 | No automatic function calling; explicit tools; history per session | `disable=True`; `@tool` declarations; `Session.history` |
| 2 | Append the model's turn as returned | `history.append(response.candidates[0].content)`; a test checks it's the identical object with its signature |
| 3 | Several calls per turn | one combined function-response turn, `id`s matched, order kept |
| 4 | Inject `user_id` | Tools get the session as `ctx`. `user_id` is in no declaration, so if the model sends it the call is rejected. |
| 5 | Errors become tool results | Unknown tool, bad arguments (checked with `inspect.signature().bind()` before calling), `ToolError`, or any exception all become `{"error": "..."}` and the loop continues |
| 6 | Cap of 6 rounds with a graceful fallback | One final call with function calling set to `NONE`; if that fails, `CAP_REPLY` |
| 7 | Log and return tool calls | One log line per call (`tool=… args=… 59ms 1003 chars ok`). `/chat` returns `{round, name, args, duration_ms, result_chars, error, preview}` for each call. |
| 8 | Groundedness check after the loop | `check_groundedness()` returns `grounded` and `ungrounded_names`; it only flags and logs |

Other safeguards:
- **API failure:** the whole turn is rolled back, so history never ends on an unanswered call.
- **History trimming:** keeps the last 20 user messages, cut only at user text turns.
- **Locking:** a per-session lock, so turns can't interleave.

### Additions beyond the approved plan
These came out of testing:
- **Per-session system prompt:** the config is built per session, adding one line about who is chatting (known user with name and city, or new user who needs onboarding).
- **Rate-limit handling:** your key is on the **free tier: 5 requests per minute** for `gemini-3.6-flash`, and there's a daily cap too.
  - `generate()` retries per-minute 429 errors up to 2 times, after the delay the API suggests (capped at 30 s).
  - Daily-quota 429 errors aren't retried.
  - Users see specific messages (`RATE_LIMIT_REPLY`, `DAILY_QUOTA_REPLY`) instead of a generic error.
- **Groundedness tuning:**
  - Single-word catalogue names ("Coffee", "Tavern") match case-sensitively, so ordinary words don't trigger the flag.
  - A shortened mention of a returned name isn't flagged. The model wrote "Rai Rai Ramen" for "Rai Rai Ramen & Hawaiian BBQ", which collided with another restaurant's name.
  - Names returned earlier in the session count as grounded, so follow-ups like "why #2?" pass.
- **"Fewer results" note:** `recommend_restaurants` now says when it found fewer than requested, and the system prompt tells the model to present partial results. Before this, a request for 3 Indian places in West Chester (only 1 exists) made the model search until it hit the round cap.
- **Default model:** if `GEMINI_MODEL` is blank, the code uses `gemini-flash-latest`. Your `.env` sets `gemini-3.6-flash`, and I confirmed that model is available to your key.

### Tools (`agent/tools.py`)
| Tool | Backs onto |
|---|---|
| `recommend_restaurants(n, cuisines?, exclude_ids?)` | `get_recommendations_with_meta`. Onboarded new users get `recommend_from_profile`; everyone else gets cold-start. Returns `source`, `rank` and the fewer-results note. |
| `explain_recommendation(business_id)` | `explain_recommendation` (for new users: their stated cuisines matched) |
| `similar_restaurants(business_id, n?, same_city?)` | `similar_restaurants` |
| `search_restaurants(name?, cuisine?, city?, limit?)` | `business_df` (exact name first, then by review count) |
| `get_restaurant_details(business_id)` | `business_df` + `business_details.pkl` (price shown as `$`–`$$$$`) |
| `get_rating_summary(business_id)` | Yelp stars and count + the dataset histogram and percentages |
| `compare_restaurants(business_ids)` | 2–4 places: price, amenities, star percentages |
| `get_reviews(business_id, min_stars?, max_stars?, limit?, order?)` | `reviews` table, sorted by stars |
| `search_reviews(business_id, keywords, limit?)` | FTS5 search with highlighted snippets. Keywords are reduced to quoted words, so model input can't break FTS syntax. |
| `get_my_taste_profile()` / `get_my_history(limit?)` | `get_user_taste` |
| `list_cuisines(city?, limit?)` | category counts |
| `set_onboarding_preferences(cuisines, city)` | validates the city (suggesting near matches, e.g. "Philadelfia" → "Philadelphia") and the cuisines, then saves them to the session |

### Verification
- **Offline tests (14, all pass):**
  - text only; one call then text, with the model turn being the identical object
  - parallel calls
  - unknown tool, missing argument, `user_id` passed, bad id, and a tool that raises
  - `user_id` injection
  - round cap with a final answer, and the round-cap fallback
  - API failure rollback
  - groundedness flags, and shortened names not flagged
  - rate-limit retry, retries exhausted, and daily quota not retried
  - history trimming
- **Tool checks against real data:** all 13 tools and their error paths work, in 0–560 ms.
- **Streamlit smoke test (`AppTest`) against the live API:** 25 sample users load, a session starts ("Karen (Philadelphia)"), the chat input appears, and new-user mode works.
- **Live Gemini run, known user "Johanna", West Chester:** these turns were correct and `grounded=true`:
  - 3 recommendations with explanations (parallel `explain_recommendation` calls)
  - "why the second one?"
  - parking reviews at Opa Taverna
  - rating breakdown
  - similar places anywhere
  - a comparison table
- **Regression:** `GET /recommendations` rankings are unchanged.

### Not yet verified live (blocked by the free-tier daily quota)
- The new-user onboarding conversation, "what kind of food do I like?", and an out-of-scope question.
- A rerun of "Only Indian food, 3 options" with the fewer-results fix.
- To finish: rerun these when the quota resets, or use a paid-tier key. The script used is in the session scratchpad; or chat through the Streamlit UI.

## Open questions
- None. Live verification of the remaining scenarios is waiting on the Gemini quota.
