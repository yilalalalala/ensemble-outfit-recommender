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


def test_parse_items_keeps_small_accessory_boxes_but_not_tiny_garment_boxes():
    """Sunglasses on a full-length photo cover ~0.2% of it: their box must survive (it is what frames them in the
    UI); a garment box that small is a detection error and is still dropped."""
    text = '{"items": [{"category": "sunglasses", "colour": "black", "description": "sunglasses", "box": [450, 300, 510, 325]},' \
           ' {"category": "top", "colour": "grey", "description": "tee", "box": [450, 300, 510, 325]},' \
           ' {"category": "jewellery", "colour": "gold", "description": "earring", "box": [100, 100, 103, 104]}]}'
    items = parse_items(text, 1000, 750)
    assert [i["box"] for i in items] == [[450, 300, 510, 325], None, None]


def test_claude_detection_uses_normalised_boxes_scaled_to_the_image():
    """Claude is asked for 0-1000 boxes, which are converted to the pixels of the image the API holds."""
    from PIL import Image

    from ensemble.vision import outfit

    class ClaudeClient:   # the detector only looks at the client's type name
        def chat(self, system, messages, **kw):
            assert outfit.NORMALISED in messages[0]["content"]
            return LLMResponse(text='{"items": [{"category": "top", "colour": "red", "description": "tee", "box": [250, 100, 750, 500]}]}')
    items, small = outfit.detect(Image.new("RGB", (1200, 800), "white"), ClaudeClient(), parse_boxes=False)
    assert items[0]["box"] == [round(.25 * small.width), round(.1 * small.height), round(.75 * small.width), round(.5 * small.height)]


def test_boxes_are_parsed_for_claude_but_not_for_qwen(monkeypatch):
    """D-051: clothes parsing replaces Claude's loose boxes; Qwen-VL keeps its own (they score higher)."""
    from PIL import Image

    from ensemble.vision import outfit, parsing
    calls = []
    monkeypatch.setattr(parsing, "refine", lambda items, img: calls.append(len(items)) or items)
    reply = '{"items": [{"category": "top", "colour": "red", "description": "tee", "box": [250, 100, 750, 500]}]}'

    class ClaudeClient:
        def chat(self, *a, **k): return LLMResponse(text=reply)

    class OllamaClient:
        def chat(self, *a, **k): return LLMResponse(text=reply)
    img = Image.new("RGB", (1200, 800), "white")
    outfit.detect(img, ClaudeClient())
    outfit.detect(img, OllamaClient())
    assert calls == [1]


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
