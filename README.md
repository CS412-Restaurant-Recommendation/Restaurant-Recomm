# 🍽️ Restaurant Recommender

A hybrid restaurant recommendation system built on the [Yelp Open Dataset](https://www.yelp.com/dataset). It combines **user-based collaborative filtering**, **content-based category similarity**, and a **cold-start (new-user) fallback** — all served through a **FastAPI REST endpoint**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Pipeline (one-time)](#data-pipeline-one-time)
- [Running the API](#running-the-api)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Sentiment Analysis](#sentiment-analysis)
- [Branches](#branches)

---

## Overview

Given a Yelp `user_id`, the system recommends restaurants in that user's **home city** (inferred from their review history) that they have **not yet rated**. Recommendations are ranked by:

1. **Collaborative Filtering (CF) score** — weighted average rating from the top-30 most similar users (cosine similarity on a precomputed user-item matrix).
2. **Category Similarity** — cosine similarity between the user's category preference vector and each candidate restaurant's category vector.
3. **Popularity** — `review_count` then `stars` as final tie-breakers.

For users with no review history (**cold-start**), the system falls back to a global **IMDB-style weighted rating** ranking.

---

## Architecture

```
Raw Yelp JSON
      │
      ▼
data_cleaning.py          ← filters businesses to a curated training set
      │
      ▼
build_train_test_and_profiles.py
  ├─ 70/30 per-user train/test split
  ├─ MultiLabelBinarizer → business_features.pkl (category vectors)
  └─ Pivot table → user_item_cf.pkl (user × restaurant rating matrix)
      │
      ▼
build_user_city.py        ← infers each user's home city from train reviews
      │
      ▼
recommend.py              ← CF + content hybrid recommender (inference)
      │
      ├─ new_user_recommendation.py  ← cold-start fallback
      │
      ▼
main.py (FastAPI)         ← REST API  GET /recommendations
```

---

## Project Structure

```
Restaurant-Recomm/
├── main.py                          # FastAPI app — exposes /recommendations
├── recommend.py                     # Core hybrid recommender (CF + content)
├── new_user_recommendation.py       # Cold-start fallback (IMDB weighted rating)
├── build_train_test_and_profiles.py # Offline pipeline: splits, features, CF matrix
├── build_user_city.py               # Offline pipeline: user → home city mapping
├── data_cleaning.py                 # Filters raw business JSON to training IDs
├── eval_recommender.py              # Hit@N evaluation on the held-out test set
├── sentimentAnalysis.ipynb          # Exploratory sentiment analysis notebook
├── requirements.txt                 # Python dependencies
└── Dataset/                         # Data directory (large files not in git)
    ├── yelp_academic_dataset_business.json   # Raw Yelp business data
    ├── filtered_yelp_business.json           # Filtered business data
    ├── filtered_yelp_review.json             # Filtered review data
    ├── filtered_yelp_user.json               # Filtered user data
    ├── train-business-ids-only.csv           # Curated business ID list
    ├── business_df.pkl                       # Processed business DataFrame
    ├── business_features.pkl                 # One-hot category feature matrix
    ├── bizid_to_idx.pkl                      # business_id → index mapping
    ├── cat_feature_names.pkl                 # Category label names
    ├── reviews_train.pkl                     # 70% train split
    ├── reviews_test.pkl                      # 30% test split
    ├── user_city_map.pkl                     # user_id → home city mapping
    ├── user_city_df.pkl                      # Full user-city DataFrame
    ├── user_item_cf.pkl                      # User × item CF rating matrix
    └── user_index_cf.pkl                     # user_id → row index in CF matrix
```

> **Note:** The `Dataset/` directory contains large binary and JSON files that are **not tracked in git** (see `.gitignore`). You must download the Yelp dataset and run the data pipeline yourself (see below).

---

## Dataset

This project uses the **[Yelp Open Dataset](https://www.yelp.com/dataset)**. Download it and place the following files inside the `Dataset/` directory:

| File | Description |
|------|-------------|
| `yelp_academic_dataset_business.json` | Business metadata (name, city, categories, stars) |
| `yelp_academic_dataset_review.json` | User reviews with star ratings |
| `yelp_academic_dataset_user.json` | User metadata |

> The dataset is ~10 GB compressed. Only the business and review files are required to run the pipeline.

---

## Setup

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```
fastapi
pandas
scikit-learn
uvicorn
```

> For the sentiment analysis notebook you will also need: `jupyter`, `transformers`, and `torch`.

---

## Data Pipeline (one-time)

Run these scripts **once** in order to build all precomputed artifacts before starting the API.

### Step 1 — Filter raw business data

```bash
python data_cleaning.py
```

Filters `filtered_yelp_business.json` to only businesses present in `train-business-ids-only.csv`, cleans category strings (removes generic labels like `"Restaurants"` and `"Food"`), and writes `filtered_yelp_business(new).json`.

### Step 2 — Build train/test splits, category features, and CF matrix

```bash
python build_train_test_and_profiles.py
```

This script:
- Keeps the **top-100 most-reviewed restaurants per city** (configurable via `TOP_K_PER_CITY`).
- Drops users with fewer than 5 reviews (`MIN_USER_REVIEWS`).
- Performs a **per-user 70/30 train/test split**.
- Builds a **one-hot category feature matrix** (`business_features.pkl`).
- Builds a **user × restaurant rating pivot table** for CF (`user_item_cf.pkl`), capped at the top 10,000 users and 10,000 businesses.

Artifacts saved to `Dataset/`:
`business_df.pkl`, `business_features.pkl`, `bizid_to_idx.pkl`, `cat_feature_names.pkl`, `reviews_train.pkl`, `reviews_test.pkl`, `user_item_cf.pkl`, `user_index_cf.pkl`

### Step 3 — Build user → city mapping

```bash
python build_user_city.py
```

Infers each user's **home city** as the city where they've reviewed the most restaurants (mode city from train reviews).

Artifacts saved: `user_city_df.pkl`, `user_city_map.pkl`

---

## Running the API

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

---

## API Reference

### `GET /recommendations`

Returns a ranked list of restaurant recommendations for a given user.

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | `string` | **required** | Yelp user ID |
| `top_n` | `int` | `10` | Number of recommendations to return (1–100) |
| `min_review_count` | `int` | `1` | Minimum number of reviews a restaurant must have |

#### Example Request

```bash
curl "http://127.0.0.1:8000/recommendations?user_id=abc123&top_n=5&min_review_count=10"
```

#### Example Response

```json
{
  "user_id": "abc123",
  "count": 5,
  "recommendations": [
    {
      "business_id": "xyz789",
      "name": "The Italian Place",
      "city": "Philadelphia",
      "stars": 4.5,
      "review_count": 312,
      "categories": "Italian, Pizza, Fine Dining",
      "cat_sim": 0.87,
      "cf_score": 4.21
    }
  ]
}
```

**New / unknown users** receive a cold-start response using the global IMDB-style weighted rating (no `cat_sim` or `cf_score` fields).

---

## Evaluation

The evaluation script measures **Hit@N** — the fraction of users for whom at least one test-set restaurant appears in the top-N recommendations.

```bash
python eval_recommender.py
```

Default settings (configurable at the top of the file):

| Parameter | Value |
|-----------|-------|
| `TOP_N` | 50 |
| `MIN_REVIEW_COUNT` | 10 |
| Users evaluated | 500 (first 500 in test set) |

Sample output:
```
Evaluated users: 487
Hit@50: 0.3127
```

---

## Sentiment Analysis

`sentimentAnalysis.ipynb` is an exploratory notebook for analyzing sentiment in Yelp reviews. It can be used to understand review quality signals and potentially augment the recommendation scoring in future iterations.

To run:
```bash
jupyter notebook sentimentAnalysis.ipynb
```

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable, production-ready code |
| `release` | Release candidates |
| `sanjith_dev` | Active development |

---

## License

This project uses the [Yelp Open Dataset](https://www.yelp.com/dataset/terms), which is subject to Yelp's dataset terms of use. This codebase is for **academic and research purposes only**.
