"""M7b: conversational shopping assistant (DESIGN §7.4–7.5, D-009, D-012).

The LLM is the orchestration layer: it understands the request, calls tools,
and writes the answer. Products come only from tools. The answer cites products
as [[article_id]]; the server renders cards from those citations and drops any
id no tool returned in this session (counted as a hallucination).
"""
from __future__ import annotations

import json
import re
import time
import uuid

from ensemble.assistant.tools import TOOL_SPECS, Toolbox

SYSTEM = """You are Ensemble, a fashion shopping assistant for a clothing catalogue.

How you work:
- Products exist for you only through your tools. Never invent a product, price, or article id.
- Cite every product you recommend as [[article_id]] using ids returned by a tool in this conversation.
- Pick the tool that fits: explicit product search → search_catalog; "what should I buy" → recommend_for_customer;
  "what goes with X" → complete_the_look; an uploaded photo → analyze_outfit_photo first.
- Turn the customer's words into tool arguments (slot, colour, product type, price). "Cheaper" means price="budget".
- Recommend at most 4 products per answer unless asked for more. For each, give one short reason drawn from the tool
  output (e.g. its evidence or reasons field). Keep answers under 120 words.
- If a tool returns nothing, say so plainly and suggest relaxing a filter. If the request is not about clothing
  or outfits, say briefly that you can only help with shopping this catalogue.
- When a photo match might be imperfect, say which catalogue item you matched and ask the customer to confirm."""

CITE = re.compile(r"\[\[(\d{6,10})\]\]")


def new_session(customer_idx: int | None, vlm) -> dict:
    return {"id": uuid.uuid4().hex[:12], "customer_idx": customer_idx, "messages": [], "photos": {},
            "grounded": set(), "vlm": vlm, "log": []}


def run_turn(llm, session: dict, text: str, photo=None, max_steps: int = 6) -> dict:
    """One user turn. Returns the grounded answer, cited cards and trace metrics."""
    tb = Toolbox(session)
    content = text
    if session.get("customer_idx") is not None and not session["messages"]:
        content = f"(Signed-in customer_idx: {session['customer_idx']})\n{text}"
    if photo is not None:
        pid = f"photo_{len(session['photos']) + 1}"
        session["photos"][pid] = photo
        content += f"\n(The customer uploaded a photo: photo_id={pid})"
    session["messages"].append({"role": "user", "content": content})
    trace, t0, cost, tokens_in, tokens_out = [], time.time(), 0.0, 0, 0
    answer = ""
    for _ in range(max_steps):
        r = llm.chat(SYSTEM, session["messages"], TOOL_SPECS, max_tokens=2048, purpose="assistant")
        cost += r.cost_usd
        tokens_in += r.input_tokens
        tokens_out += r.output_tokens
        session["messages"].append({"role": "assistant", "content": r.text, "tool_calls": r.tool_calls,
                                    "raw_content": r.raw_content})
        if not r.tool_calls:
            answer = r.text
            break
        for c in r.tool_calls:
            args = c["arguments"] if isinstance(c["arguments"], dict) else json.loads(c["arguments"] or "{}")
            result = tb.call(c["name"], args)
            trace.append({"tool": c["name"], "arguments": args,
                          "n_results": len(result.get("results", [])) if isinstance(result, dict) else 0})
            session["messages"].append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                                        "content": json.dumps(result, default=str)[:6000]})
    # Ids the customer typed themselves are not hallucinations when echoed back.
    user_ids = {int(x) for m in session["messages"] if m["role"] == "user" for x in re.findall(r"\d{6,10}", m["content"])}
    # Any article id in the answer counts, whatever the format ([[id]], "(article id)", bare): grounded
    # ids are citations; ids no tool returned (and the customer did not type) are hallucinations.
    mentioned = [int(x) for x in re.findall(r"(?<!\d)0?(\d{6,9})(?!\d)", answer)]
    cited = [a for a in dict.fromkeys(mentioned) if a in session["grounded"]]
    hallucinated = [a for a in dict.fromkeys(mentioned) if a not in session["grounded"] and a not in user_ids]
    clean = CITE.sub(lambda m: m.group(0) if int(m.group(1)) in session["grounded"] else "", answer)
    grounded = cited
    out = {"answer": clean, "cited": grounded, "hallucinated": hallucinated, "trace": trace,
           "latency_s": round(time.time() - t0, 2), "cost_usd": round(cost, 5),
           "input_tokens": tokens_in, "output_tokens": tokens_out}
    session["log"].append(out)
    return out
