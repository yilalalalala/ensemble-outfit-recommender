"""M7a/M7b logic that needs no model: parsing, gap detection, grounding, message conversion."""
import pytest

from ensemble.llm.client import LLMResponse
from ensemble.vision.outfit import gaps, parse_items


def test_parse_items_validates_boxes_and_categories():
    text = '{"items": [{"category": "shoes", "colour": "black", "description": "boots", "box": [10, 10, 60, 60]},' \
           ' {"category": "spaceship", "colour": "", "description": "", "box": [0, 0, 5, 5]},' \
           ' {"category": "top", "colour": "red", "description": "tee", "box": [50, 50, 9999, 60]}]}'
    items = parse_items(text, 100, 100)
    assert [i["category"] for i in items] == ["shoes", "top"]
    assert items[0]["box"] == [10, 10, 60, 60] and items[1]["box"] is None


def test_parse_items_tolerates_garbage():
    assert parse_items("not json at all", 100, 100) == []


def test_gap_detection():
    assert gaps([{"category": "dress"}]) == ["shoes", "accessories"]
    assert gaps([{"category": "top"}, {"category": "bottom"}, {"category": "shoes"}, {"category": "bag"}]) == []
    assert gaps([{"category": "top"}]) == ["lower", "shoes", "accessories"]


class ScriptedLLM:
    """Replays a fixed sequence of responses: a tool call, then a final answer."""

    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, system, messages, tools=None, **kw):
        return self.responses.pop(0)


@pytest.mark.data
def test_grounding_drops_invented_ids(cfg):
    if not cfg.path("serving_db").exists():
        pytest.skip("serving store not built")
    import sqlite3

    from ensemble.assistant.agent import new_session, run_turn
    con = sqlite3.connect(cfg.path("serving_db"))
    c, a = con.execute("SELECT customer_idx, article_id FROM for_you ORDER BY customer_idx, rank LIMIT 1").fetchone()
    llm = ScriptedLLM([
        LLMResponse(text="", tool_calls=[{"id": "t1", "name": "recommend_for_customer",
                                          "arguments": {"customer_idx": c, "max_results": 3}}]),
        LLMResponse(text=f"Try [[{a}]] and also [[111111111]]."),
    ])
    s = new_session(c, None)
    r = run_turn(llm, s, "recommend me something")
    assert r["cited"] == [a]
    assert r["hallucinated"] == [111111111]
    assert "111111111" not in r["answer"]
    assert r["trace"][0]["tool"] == "recommend_for_customer"


def test_claude_message_conversion_merges_tool_results():
    from ensemble.llm.client import ClaudeClient
    conv = ClaudeClient._convert(None, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "x", "arguments": {}},
                                                            {"id": "b", "name": "y", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "a", "name": "x", "content": "{}"},
        {"role": "tool", "tool_call_id": "b", "name": "y", "content": "{}"},
    ])
    assert [m["role"] for m in conv] == ["user", "assistant", "user"]
    assert len(conv[2]["content"]) == 2   # parallel tool results go back in one user message
