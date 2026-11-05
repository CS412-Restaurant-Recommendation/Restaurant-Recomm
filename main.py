import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recommend import get_recommendations, get_sample_users
from agent import agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Restaurant Recommender API")


class SessionRequest(BaseModel):
    user_id: Optional[str] = None  # omit or leave empty for a new user


class ChatRequest(BaseModel):
    session_id: str
    message: str

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

@app.get("/users/sample")
def sample_users(n: int = Query(20, ge=1, le=200)):
    # Known users with the most history, for the chat UI's user picker
    users = get_sample_users(n=n)
    return {"count": len(users), "users": users}

@app.post("/session")
def start_session(req: SessionRequest):
    # Binds the user ID to a new chat session; the agent never takes it from chat text
    session = agent.create_session(req.user_id)
    return {
        "session_id": session.session_id,
        "user_known": session.user_known,
        "name": session.name,
        "home_city": session.home_city,
    }

@app.post("/chat")
def chat(req: ChatRequest):
    session = agent.SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id; start one with POST /session")
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Message is empty")
    return agent.chat(session, req.message)
