# file: tests/test_agent_loop.py
#
# Offline tests for the hand-written tool-calling loop in agent/agent.py.
# A fake Gemini client returns scripted responses, so no API key or network is needed.
#
# Run:  python -m tests.test_agent_loop      (or: pytest tests/)

from google.genai import errors as genai_errors
from google.genai import types

from agent import agent
from agent.tools import TOOLS, ToolSpec

KNOWN_USER = "--_r6E98SNIrGU7weyNxbw"
OPA = "ctC-HU2Jbv04I42ykXFjLA"  # Opa Taverna


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []  # (contents snapshot, config) per call

    def generate_content(self, model, contents, config):
        self.requests.append((list(contents), config))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def text_response(text):
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def call_response(*calls):
    """calls: (name, args) pairs. Each part carries a thought signature, like a thinking model."""
    parts = [types.Part(function_call=types.FunctionCall(id=f"call-{i}", name=name, args=args),
                        thought_signature=b"signature")
             for i, (name, args) in enumerate(calls)]
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])


def setup(script, user_id=KNOWN_USER):
    fake = FakeClient(script)
    agent._client = fake
    return agent.create_session(user_id), fake


def function_responses(content):
    return [p.function_response for p in content.parts]


def test_text_only():
    session, fake = setup([text_response("Hello!")])
    out = agent.run_turn(session, "hi")
    assert out["reply"] == "Hello!"
    assert out["tool_calls"] == []
    assert [c.role for c in session.history] == ["user", "model"]
    assert fake.models.requests[0][1].automatic_function_calling.disable is True


def test_one_call_then_text_keeps_model_turn_as_returned():
    call = call_response(("get_rating_summary", {"business_id": OPA}))
    session, fake = setup([call, text_response("Opa Taverna has 4.5 stars.")])
    out = agent.run_turn(session, "rating of opa?")

    assert out["reply"] == "Opa Taverna has 4.5 stars."
    assert [c.role for c in session.history] == ["user", "model", "user", "model"]
    # The model turn is the exact object returned (thought signature intact), not a rebuilt copy
    assert session.history[1] is call.candidates[0].content
    assert session.history[1].parts[0].thought_signature == b"signature"
    # ...and it is what the second request sent back to the model
    assert fake.models.requests[1][0][1] is call.candidates[0].content

    (resp,) = function_responses(session.history[2])
    assert resp.id == "call-0" and resp.name == "get_rating_summary"
    assert resp.response["yelp_stars"] == 4.5
    (record,) = out["tool_calls"]
    assert record["round"] == 1 and record["error"] is None and record["result_chars"] > 0


def test_parallel_calls_answered_in_one_turn_in_order():
    call = call_response(("get_rating_summary", {"business_id": OPA}),
                         ("search_restaurants", {"name": "opa taverna"}))
    session, _ = setup([call, text_response("done")])
    out = agent.run_turn(session, "two things")

    assert [c.role for c in session.history] == ["user", "model", "user", "model"]
    responses = function_responses(session.history[2])
    assert [(r.id, r.name) for r in responses] == [("call-0", "get_rating_summary"),
                                                  ("call-1", "search_restaurants")]
    assert [r["name"] for r in out["tool_calls"]] == ["get_rating_summary", "search_restaurants"]


def test_errors_become_tool_results_and_loop_continues():
    def boom(ctx):
        raise RuntimeError("kaboom")
    TOOLS["boom"] = ToolSpec(boom, types.FunctionDeclaration(name="boom", description="test"))
    try:
        call = call_response(
            ("no_such_tool", {}),
            ("get_rating_summary", {}),                                   # missing argument
            ("get_rating_summary", {"business_id": OPA, "user_id": "x"}),  # model tries to pick the user
            ("get_rating_summary", {"business_id": "bad-id"}),             # ToolError
            ("boom", {}),                                                  # tool raises
        )
        session, _ = setup([call, text_response("recovered")])
        out = agent.run_turn(session, "break things")
    finally:
        del TOOLS["boom"]

    assert out["reply"] == "recovered"
    errors = [r.response["error"] for r in function_responses(session.history[2])]
    assert errors[0] == "Unknown tool: no_such_tool"
    assert "missing a required argument" in errors[1]
    assert "unexpected keyword argument 'user_id'" in errors[2]
    assert "Unknown business_id" in errors[3]
    assert errors[4] == "RuntimeError: kaboom"
    assert all(r["error"] for r in out["tool_calls"])


def test_user_id_is_injected_from_session():
    seen = {}
    def whoami(ctx):
        seen["user_id"] = ctx.user_id
        return {"ok": True}
    TOOLS["whoami"] = ToolSpec(whoami, types.FunctionDeclaration(name="whoami", description="test"))
    try:
        session, _ = setup([call_response(("whoami", {})), text_response("ok")])
        agent.run_turn(session, "who am i")
    finally:
        del TOOLS["whoami"]
    assert seen["user_id"] == KNOWN_USER


def test_round_cap_asks_for_final_answer_without_tools():
    rounds = [call_response(("search_restaurants", {"name": "opa"})) for _ in range(agent.MAX_TOOL_ROUNDS)]
    session, fake = setup(rounds + [text_response("Here is what I found.")])
    out = agent.run_turn(session, "loop forever")

    assert out["reply"] == "Here is what I found."
    assert len(out["tool_calls"]) == agent.MAX_TOOL_ROUNDS
    assert len(fake.models.requests) == agent.MAX_TOOL_ROUNDS + 1
    final_config = fake.models.requests[-1][1]
    assert final_config.tool_config.function_calling_config.mode == "NONE"
    assert session.history[-1].role == "model"


def test_round_cap_fallback_when_final_call_fails():
    rounds = [call_response(("search_restaurants", {"name": "opa"})) for _ in range(agent.MAX_TOOL_ROUNDS)]
    session, _ = setup(rounds + [RuntimeError("API down")])
    out = agent.run_turn(session, "loop forever")
    assert out["reply"] == agent.CAP_REPLY
    assert session.history[-1].role == "model"  # history still ends well-formed


def test_api_failure_rolls_back_the_turn():
    session, _ = setup([text_response("first answer"),
                        call_response(("search_restaurants", {"name": "opa"})),
                        RuntimeError("API down")])
    agent.run_turn(session, "first")
    before = list(session.history)

    out = agent.run_turn(session, "second")
    assert out["reply"] == agent.ERROR_REPLY
    assert session.history == before  # no dangling user message or function call


def test_groundedness_flags_names_no_tool_returned():
    call = call_response(("search_restaurants", {"name": "opa taverna"}))
    session, _ = setup([call, text_response(
        "Opa Taverna is great, and so is Market Street Grill. Good coffee nearby.")])
    out = agent.run_turn(session, "opa?")
    assert out["grounded"] is False
    assert out["ungrounded_names"] == ["Market Street Grill"]  # "coffee" is not flagged

    # Names grounded earlier in the session stay grounded in follow-ups
    agent._client = FakeClient([text_response("Opa Taverna again.")])
    assert agent.run_turn(session, "tell me more")["grounded"] is True


def rate_limit_error():
    return genai_errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded. Please retry in 3.5s."}})


def test_rate_limit_is_retried_after_suggested_delay():
    waits = []
    real_sleep, agent.time.sleep = agent.time.sleep, waits.append
    try:
        session, fake = setup([rate_limit_error(), text_response("after retry")])
        out = agent.run_turn(session, "hi")
    finally:
        agent.time.sleep = real_sleep
    assert out["reply"] == "after retry"
    assert waits == [4.5]  # suggested 3.5s + 1s margin
    assert len(fake.models.requests) == 2


def test_rate_limit_retries_exhausted_gives_specific_reply():
    real_sleep, agent.time.sleep = agent.time.sleep, lambda s: None
    try:
        session, _ = setup([rate_limit_error()] * (agent.RATE_LIMIT_RETRIES + 1))
        out = agent.run_turn(session, "hi")
    finally:
        agent.time.sleep = real_sleep
    assert out["reply"] == agent.RATE_LIMIT_REPLY
    assert session.history == []  # turn rolled back


def test_daily_quota_is_not_retried():
    error = genai_errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "Quota exceeded, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 3s."}})
    session, fake = setup([error])
    out = agent.run_turn(session, "hi")
    assert out["reply"] == agent.DAILY_QUOTA_REPLY
    assert len(fake.models.requests) == 1


def test_shortened_returned_name_is_not_flagged():
    session, _ = setup([text_response("You loved Rai Rai Ramen.")])
    session.grounded_names.add("rai rai ramen & hawaiian bbq")  # returned by a tool earlier
    assert agent.run_turn(session, "what did I like?")["grounded"] is True


def test_trim_history_cuts_only_at_user_text_turns():
    script = []
    for i in range(agent.MAX_USER_TURNS_KEPT + 3):
        script += [call_response(("search_restaurants", {"name": "opa"})), text_response(f"answer {i}")]
    session, _ = setup(script)
    for i in range(agent.MAX_USER_TURNS_KEPT + 3):
        agent.run_turn(session, f"question {i}")

    user_text_turns = [c for c in session.history if c.role == "user" and c.parts[0].text]
    assert len(user_text_turns) == agent.MAX_USER_TURNS_KEPT
    assert session.history[0].parts[0].text == "question 3"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"{len(tests)} tests passed")
