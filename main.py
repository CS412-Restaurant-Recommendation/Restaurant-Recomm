from fastapi import FastAPI, Query
from recommend import get_recommendations

app = FastAPI(title="Restaurant Recommender API")

@app.get("/recommendations")
def recommendations(
    user_id: str = Query(...),
    top_n: int = Query(10, ge=1, le=100),
    min_review_count: int = Query(1, ge=0),
):
    recs = get_recommendations(user_id=user_id, top_n=top_n, min_review_count=min_review_count)

    # recs will always be a list here
    return {
        "user_id": user_id,
        "count": len(recs),
        "recommendations": recs,
    }
