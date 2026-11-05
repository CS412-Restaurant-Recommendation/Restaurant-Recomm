# file: agent/agent.py
#
# Gemini chat agent with a hand-written tool-calling loop.
#
# Automatic function calling in the google-genai SDK is disabled: we declare
# the tools ourselves, run the calls the model asks for, and send the results
# back, keeping the conversation history per session. See run_turn().

import inspect
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import recommend as rec
from agent import store
from agent.tools import TOOLS, ToolError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("agent")

MODEL = os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"
MAX_TOOL_ROUNDS = 6
MAX_USER_TURNS_KEPT = 20
PREVIEW_CHARS = 400
RATE_LIMIT_RETRIES = 2
MAX_RETRY_WAIT_S = 30

EMPTY_REPLY = "Sorry, I don't have an answer for that. Could you rephrase?"
CAP_REPLY = "I couldn't finish looking that up — could you narrow the question?"
ERROR_REPLY = "Sorry, something went wrong while contacting the model. Please try again."
RATE_LIMIT_REPLY = "I'm being rate-limited by the Gemini API right now. Please wait a minute and try again."
DAILY_QUOTA_REPLY = "The Gemini API's daily request quota for this key is used up. Please try again tomorrow."

SYSTEM_PROMPT = """You are a restaurant recommendation assistant for a Yelp-based recommender system.

Rules:
- Answer ONLY from the results of your tools. Never invent restaurants, ratings, reviews,
  addresses or other facts, and never use outside knowledge about restaurants.
- If the tools don't have the answer, say so plainly.
- When the user names a restaurant, call search_restaurants to get its business_id first.
- When recommendations have source "cold_start", say they are popular, highly-rated picks,
  not personalized ones.
- To explain a recommendation, call explain_recommendation and describe the evidence in plain
  words, e.g. "9 users with similar taste rated it 4.2 stars on average" or "you rated
  Zoe's Kitchen 4 stars and this place is also Greek".
- Our dataset's review counts can differ from Yelp's review_count; say which one you quote.
- For a new user with no stated preferences, ask for their favourite cuisines and city, then
  call set_onboarding_preferences before recommending.
- If a tool returns {"error": ...}, fix the arguments and retry, or explain the problem.
- If a tool returns fewer results than asked for, present what it returned and say so;
  don't keep searching. Offer to look in other cities or cuisines instead.
- Politely decline requests unrelated to restaurants in this dataset.
- Be concise. Use short lists for multiple restaurants (name, city, stars, one-line reason).
"""

TOOL_DECLARATIONS = [types.Tool(function_declarations=[t.declaration for t in TOOLS.values()])]

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (add it to .env)")
        _client = genai.Client(api_key=api_key)
    return _client

# ==========================
# Sessions
# ==========================

@dataclass
class Session:
    session_id: str
    user_id: str | None
    user_known: bool
    name: str | None = None
    home_city: str | None = None
    onboarding: dict | None = None            # new users' stated cuisines + city
    history: list = field(default_factory=list)  # list[types.Content]
    grounded_names: set = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)
    config: types.GenerateContentConfig | None = None

    def trim_history(self):
        """Keep the last MAX_USER_TURNS_KEPT user messages, cutting only at a user text turn."""
        starts = [i for i, c in enumerate(self.history)
                  if c.role == "user" and any(p.text for p in c.parts)]
        if len(starts) > MAX_USER_TURNS_KEPT:
            del self.history[:starts[-MAX_USER_TURNS_KEPT]]


SESSIONS: dict[str, Session] = {}


def create_session(user_id=None):
    user_id = (user_id or "").strip() or None
    known = user_id is not None and user_id in rec.user_city_map
    session = Session(
        session_id=uuid.uuid4().hex,
        user_id=user_id if known else None,
        user_known=known,
        name=store.get_user_name(user_id) if known else None,
        home_city=rec.user_city_map.get(user_id) if known else None,
    )
    if known:
        who = f"The user is {session.name or 'a known user'}, home city {session.home_city}, with rating history."
    else:
        who = "The user is NEW (no rating history). Onboard them before recommending."
    session.config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT + "\n" + who,
        tools=TOOL_DECLARATIONS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    SESSIONS[session.session_id] = session
    return session


def chat(session, message):
    with session.lock:   # one turn at a time per session
        return run_turn(session, message)

# ==========================
# The tool-calling loop
# ==========================

def _is_rate_limit(error):
    return isinstance(error, genai_errors.APIError) and getattr(error, "code", None) == 429


def _is_daily_quota(error):
    return _is_rate_limit(error) and "PerDay" in str(error)


def generate(history, config):
    """One model call. Retries per-minute rate limits (429) after the delay the API suggests."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            response = get_client().models.generate_content(model=MODEL, contents=history, config=config)
            break
        except genai_errors.APIError as e:
            # Waiting can't help once the daily quota is used up
            if not _is_rate_limit(e) or _is_daily_quota(e) or attempt == RATE_LIMIT_RETRIES:
                raise
            suggested = re.search(r"retry in ([\d.]+)s", str(e))
            wait = min(float(suggested.group(1)) + 1 if suggested else 10, MAX_RETRY_WAIT_S)
            logger.warning("Rate-limited by Gemini; retrying in %.0fs", wait)
            time.sleep(wait)
    if not response.candidates or response.candidates[0].content is None:
        raise RuntimeError(f"No usable candidate (prompt feedback: {response.prompt_feedback})")
    return response


def run_turn(session, user_message):
    history = session.history
    start_len = len(history)                     # to roll back on API failure
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))
    trace, tool_results = [], []

    try:
        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            response = generate(history, session.config)
            history.append(response.candidates[0].content)   # exactly as returned: keeps thought signatures

            calls = response.function_calls or []
            if not calls:                        # plain text -> done
                reply = response.text or EMPTY_REPLY
                break

            # Run every call from this turn; answer them all in ONE user turn,
            # in the same order, each FunctionResponse carrying its call's id and name.
            parts = []
            for call in calls:
                result, record = execute_tool(session, call, round_no)
                trace.append(record)
                tool_results.append(result)
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=call.id, name=call.name, response=result)))
            history.append(types.Content(role="user", parts=parts))
        else:                                    # loop never broke: hit the round cap
            reply = answer_without_tools(session)

    except Exception as e:                       # Gemini/API failure, blocked response, etc.
        logger.exception("Gemini call failed")
        del history[start_len:]                  # never leave a dangling function call in history
        reply = DAILY_QUOTA_REPLY if _is_daily_quota(e) else RATE_LIMIT_REPLY if _is_rate_limit(e) else ERROR_REPLY

    grounding = check_groundedness(reply, tool_results, session)
    session.trim_history()
    return {"reply": reply, "tool_calls": trace, **grounding}


def answer_without_tools(session):
    """After MAX_TOOL_ROUNDS, ask once more with function calling off, so the model answers from what it has."""
    logger.warning("Tool round cap (%d) reached; asking for a final answer without tools", MAX_TOOL_ROUNDS)
    config = session.config.model_copy(update={"tool_config": types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="NONE"))})
    try:
        response = generate(session.history, config)
        if response.text:
            session.history.append(response.candidates[0].content)
            return response.text
    except Exception:
        logger.exception("Final no-tools call failed")
    # Close the turn with a model message so the history stays well-formed
    session.history.append(types.Content(role="model", parts=[types.Part.from_text(text=CAP_REPLY)]))
    return CAP_REPLY


def execute_tool(session, call, round_no):
    """Run one function call. Never raises: problems come back as {"error": ...} for the model."""
    args = dict(call.args or {})
    spec = TOOLS.get(call.name)
    start = time.perf_counter()
    try:
        if spec is None:
            raise ToolError(f"Unknown tool: {call.name}")
        try:
            # Validate before calling, so a TypeError inside the tool isn't reported as bad arguments.
            # The session is injected as ctx; user_id is in no declaration, so passing it fails here.
            inspect.signature(spec.fn).bind(session, **args)
        except TypeError as e:
            raise ToolError(f"Bad arguments for {call.name}: {e}") from None
        result = spec.fn(session, **args)
    except ToolError as e:
        result = {"error": str(e)}
    except Exception as e:
        logger.exception("Tool %s raised", call.name)
        result = {"error": f"{type(e).__name__}: {e}"}

    result = _json_safe(result if isinstance(result, dict) else {"result": result})
    text = json.dumps(result, ensure_ascii=False)
    duration_ms = round((time.perf_counter() - start) * 1000)
    error = result.get("error")
    logger.info("tool=%s args=%s %dms %d chars %s", call.name, json.dumps(args), duration_ms,
                len(text), f"ERROR: {error}" if error else "ok")
    record = {"round": round_no, "name": call.name, "args": args, "duration_ms": duration_ms,
              "result_chars": len(text), "error": error, "preview": text[:PREVIEW_CHARS]}
    return result, record


def _json_safe(value):
    """Round-trip through JSON: NumPy/pandas values -> plain types, NaN/Infinity -> None."""
    def default(o):
        return o.item() if hasattr(o, "item") else str(o)
    return json.loads(json.dumps(value, default=default), parse_constant=lambda _: None)

# ==========================
# Groundedness check
# ==========================

# Catalogue restaurant names worth scanning for (short names cause false hits)
CATALOGUE_NAMES = {n.strip() for n in rec.business["name"].dropna().unique() if len(n.strip()) >= 4}


def _collect_names(value, names):
    if isinstance(value, dict):
        for key, v in value.items():
            if key == "name" and isinstance(v, str):
                names.add(v.lower())
            else:
                _collect_names(v, names)
    elif isinstance(value, list):
        for v in value:
            _collect_names(v, names)


def check_groundedness(reply, tool_results, session):
    """
    Flag catalogue restaurant names in the reply that no tool returned, this turn or
    earlier in the session (so follow-ups like "why #2?" aren't flagged). Never blocks.
    """
    for result in tool_results:
        _collect_names(result, session.grounded_names)

    text = reply
    for name in sorted(session.grounded_names, key=len, reverse=True):
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)  # so "Opa" inside "Opa Taverna" isn't flagged
    lower = text.lower()

    # Multi-word names match case-insensitively; single-word names ("Coffee", "Tavern")
    # must match case-sensitively, or ordinary words in the reply would be flagged.
    ungrounded = sorted(
        name for name in CATALOGUE_NAMES
        if name.lower() in lower and re.search(
            rf"(?<!\w){re.escape(name)}(?!\w)", text, flags=re.IGNORECASE if " " in name else 0)
        # A shortened mention of a returned name ("Rai Rai Ramen" for "Rai Rai Ramen &
        # Hawaiian BBQ") can collide with another catalogue name; don't flag those.
        and not any(name.lower() in g for g in session.grounded_names)
    )
    if ungrounded:
        logger.warning("Ungrounded restaurant names in reply: %s", ungrounded)
    return {"grounded": not ungrounded, "ungrounded_names": ungrounded}
