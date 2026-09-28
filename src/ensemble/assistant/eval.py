"""M7b evaluation: the same fixed conversation set run against each LLM back end.

Automatic metrics (no LLM needed):
  tool_accuracy        expected tool(s) called in the turn
  argument_accuracy    expected argument values present (slot, colour, price, type)
  constraint_pass      every cited article satisfies the turn's constraints, checked on catalogue metadata
  hallucination_rate   share of turns citing an id no tool returned (target 0)
  no_product_pass      turns that must not recommend products (off-topic, unknown id) cite none
  latency, cost, tokens
LLM-as-judge (Claude, rubric 1–5) scores helpfulness and faithfulness of each final answer. The judge is
the same model family as one contestant, so self-preference bias is possible; its scores are reported
next to the automatic metrics, not instead of them.

  python -m ensemble.assistant.eval ollama|claude [--judge]
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time

from PIL import Image

from ensemble.assistant.agent import new_session, run_turn
from ensemble.config import load_config
from ensemble.llm.client import ClaudeClient, OllamaClient, spend


def fixtures(cfg) -> dict:
    con = sqlite3.connect(cfg.path("serving_db"))
    q = lambda s, a=(): con.execute(s, a).fetchone()  # noqa: E731
    cust = q("SELECT customer_idx FROM demo_customers WHERE segment='returning' ORDER BY customer_idx LIMIT 1")[0]
    new = q("SELECT customer_idx FROM demo_customers WHERE segment='new' ORDER BY customer_idx LIMIT 1")[0]
    anchor = lambda slot: q("""SELECT c.anchor, a.prod_name FROM complete_the_look c JOIN articles a ON a.article_id=c.anchor
                              WHERE a.slot=? GROUP BY c.anchor HAVING count(DISTINCT c.slot) >= 3
                              ORDER BY count(*) DESC, c.anchor LIMIT 1""", (slot,))  # noqa: E731
    return {"C": cust, "NEW": new, "LOWER": anchor("lower"), "FULL": anchor("full"), "UPPER": anchor("upper")}


def cases(f: dict) -> list[dict]:
    L, F, U = f["LOWER"], f["FULL"], f["UPPER"]
    return [
        {"id": "reco_open", "customer": f["C"], "turns": [{"text": "Recommend me something new to wear.",
          "tools": ["recommend_for_customer"]}]},
        {"id": "reco_slot", "customer": f["C"], "turns": [{"text": "Any trousers or skirts for me?",
          "tools": ["recommend_for_customer", "search_catalog"], "any_tool": True, "cons": {"slot": ["lower"]}}]},
        {"id": "reco_new_customer", "customer": f["NEW"], "turns": [{"text": "I'm new here, what should I buy?",
          "tools": ["recommend_for_customer", "search_catalog"], "any_tool": True}]},
        {"id": "search_earrings", "turns": [{"text": "I'm looking for gold hoop earrings.", "tools": ["search_catalog"],
          "cons": {"type": ["earring"], "colour": ["gold"]}}]},
        {"id": "search_boots", "turns": [{"text": "Show me black ankle boots.", "tools": ["search_catalog"],
          "args": {"colour": "black"}, "cons": {"slot": ["shoes"], "colour": ["black"]}}]},
        {"id": "search_hoodie", "turns": [{"text": "Men's grey hoodie please.", "tools": ["search_catalog"],
          "cons": {"slot": ["upper"], "colour": ["grey"]}}]},
        {"id": "search_wedding", "turns": [{"text": "I need an elegant dress to wear as a wedding guest.",
          "tools": ["search_catalog"], "cons": {"slot": ["full"]}}]},
        {"id": "search_swim", "turns": [{"text": "A swimsuit for my summer holiday?", "tools": ["search_catalog"],
          "cons": {"slot": ["swimwear"]}}]},
        {"id": "refine_cheaper", "turns": [
            {"text": "Show me white sneakers.", "tools": ["search_catalog"], "cons": {"slot": ["shoes"]}},
            {"text": "Cheaper ones please.", "tools": ["search_catalog"], "args": {"price": "budget"}, "cons": {"slot": ["shoes"]}}]},
        {"id": "ctl_open", "turns": [{"text": f"What goes with the '{L[1]}' (article {L[0]})?", "tools": ["complete_the_look"],
          "args": {"article_id": L[0]}}]},
        {"id": "ctl_shoes", "turns": [{"text": f"Which shoes would go with article {L[0]}?", "tools": ["complete_the_look"],
          "args": {"slot": "shoes"}, "cons": {"slot": ["shoes"]}}]},
        {"id": "ctl_accessories", "turns": [{"text": f"I bought the '{F[1]}' (article {F[0]}). What accessories complete it?",
          "tools": ["complete_the_look"], "args": {"slot": "accessories"}, "cons": {"slot": ["accessories"]}}]},
        {"id": "ctl_followup", "customer": f["C"], "turns": [
            {"text": "Recommend me a top.", "tools": ["recommend_for_customer", "search_catalog"], "any_tool": True},
            {"text": "I like the first one. What bottoms go with it?", "tools": ["complete_the_look"], "args": {"slot": "lower"},
             "cons": {"slot": ["lower"]}}]},
        {"id": "ctl_top", "turns": [{"text": f"Help me style article {U[0]} for the office.", "tools": ["complete_the_look"]}]},
        {"id": "photo_missing", "photo": "full_07.jpg", "turns": [{"text": "What's missing from this outfit?",
          "tools": ["analyze_outfit_photo"]}]},
        {"id": "photo_bag", "photo": "full_13.jpg", "turns": [{"text": "What bag would complete this look?",
          "tools": ["analyze_outfit_photo"], "cons": {"slot": ["accessories"]}}]},
        {"id": "photo_similar", "photo": "full_04.jpg", "turns": [
            {"text": "Find me a coat like the one in this photo.", "tools": ["analyze_outfit_photo", "search_catalog"],
             "any_tool": True, "cons": {"slot": ["upper"]}},
            {"text": "What shoes would you wear with it?", "tools": ["complete_the_look", "analyze_outfit_photo", "search_catalog"],
             "any_tool": True, "cons": {"slot": ["shoes"]}}]},
        {"id": "offtopic", "turns": [{"text": "What's the weather in Stockholm tomorrow?", "tools": [], "no_products": True}]},
        {"id": "unknown_id", "turns": [{"text": "Tell me about article 123456789.", "tools": [], "no_products": True,
                                        "optional_tools": True}]},
        {"id": "brand", "turns": [{"text": "Do you have Gucci loafers?", "tools": ["search_catalog"], "optional_tools": True}]},
    ]


def check_constraints(con, ids: list[int], cons: dict) -> bool:
    if not ids or not cons:
        return True
    rows = con.execute(f"""SELECT slot, lower(product_type_name), lower(colour_group_name) FROM articles
                           WHERE article_id IN ({','.join('?' * len(ids))})""", ids).fetchall()
    for slot, ptype, colour in rows:
        if "slot" in cons and slot not in cons["slot"]:
            return False
        if "type" in cons and not any(t in ptype for t in cons["type"]):
            return False
        if "colour" in cons and not any(c in colour for c in cons["colour"]):
            return False
    return True


def run(backend: str, judge: bool = False) -> dict:
    cfg = load_config()
    llm = ClaudeClient("claude-opus-5") if backend == "claude" else OllamaClient("qwen3-vl:8b-instruct")
    vlm = llm if backend == "claude" else OllamaClient("qwen2.5vl:7b")
    con = sqlite3.connect(cfg.path("serving_db"))
    f = fixtures(cfg)
    photos = cfg.path("raw") / "outfit_photos"
    turns_out = []
    for c in cases(f):
        s = new_session(c.get("customer"), vlm)
        for i, t in enumerate(c["turns"]):
            photo = Image.open(photos / c["photo"]) if c.get("photo") and i == 0 else None
            try:
                r = run_turn(llm, s, t["text"], photo)
            except Exception as e:  # a failed turn is scored as failed, not skipped
                r = {"answer": f"[error: {e}]", "cited": [], "hallucinated": [], "trace": [], "latency_s": 0,
                     "cost_usd": 0, "input_tokens": 0, "output_tokens": 0, "error": str(e)}
            called = [x["tool"] for x in r["trace"]]
            exp = t["tools"]
            if not exp:
                tool_ok = t.get("optional_tools", False) or not called
            elif t.get("any_tool"):
                tool_ok = any(x in called for x in exp)
            else:
                tool_ok = exp[0] in called or bool(t.get("optional_tools") and not called)
            args_ok = True
            for k, v in t.get("args", {}).items():
                vals = [str(x["arguments"].get(k, "")).lower() for x in r["trace"]]
                args_ok &= any(str(v).lower() in x for x in vals)
            row = {"case": c["id"], "turn": i + 1, "text": t["text"], "answer": r["answer"], "tools_called": called,
                   "tool_ok": tool_ok, "args_ok": args_ok, "cited": r["cited"], "hallucinated": r["hallucinated"],
                   "constraint_ok": check_constraints(con, r["cited"], t.get("cons", {})),
                   "no_products_ok": (not r["cited"]) if t.get("no_products") else None,
                   "latency_s": r["latency_s"], "cost_usd": r["cost_usd"],
                   "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"], "error": r.get("error")}
            turns_out.append(row)
            (cfg.path("reports") / f"m7b_assistant_{backend}.partial.json").write_text(json.dumps(turns_out, default=str))
            print(f"  [{backend}] {c['id']}#{i + 1}: tools={called} ok={tool_ok}/{args_ok} cited={len(r['cited'])} "
                  f"halluc={len(r['hallucinated'])} cons={row['constraint_ok']} {r['latency_s']}s ${r['cost_usd']:.4f}", flush=True)
    if judge:
        judge_scores(turns_out)
    n = len(turns_out)
    summary = {
        "backend": backend, "model": getattr(llm, "model", ""), "turns": n,
        "tool_accuracy": sum(t["tool_ok"] for t in turns_out) / n,
        "argument_accuracy": sum(t["args_ok"] for t in turns_out) / n,
        "constraint_pass": sum(t["constraint_ok"] for t in turns_out) / n,
        "hallucination_rate": sum(bool(t["hallucinated"]) for t in turns_out) / n,
        "no_product_pass": [t["no_products_ok"] for t in turns_out if t["no_products_ok"] is not None],
        "turns_with_products": sum(bool(t["cited"]) for t in turns_out) / n,
        "latency_median_s": statistics.median(t["latency_s"] for t in turns_out),
        "latency_p90_s": sorted(t["latency_s"] for t in turns_out)[int(0.9 * (n - 1))],
        "cost_usd_total": sum(t["cost_usd"] for t in turns_out),
        "errors": sum(bool(t["error"]) for t in turns_out),
    }
    if judge:
        js = [t["judge"] for t in turns_out if t.get("judge")]
        summary["judge_helpfulness"] = statistics.mean(j["helpfulness"] for j in js) if js else None
        summary["judge_faithfulness"] = statistics.mean(j["faithfulness"] for j in js) if js else None
    out = cfg.path("reports") / f"m7b_assistant_{backend}.json"
    out.write_text(json.dumps({"summary": summary, "turns": turns_out}, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print("LLM spend so far:", spend())
    return summary


JUDGE = """You grade a fashion shopping assistant's reply. The assistant may only recommend products returned by its tools,
cited as [[id]]. Score two things from 1 (bad) to 5 (excellent):
- helpfulness: does the reply address what the customer asked, with suitable suggestions and a useful next step?
- faithfulness: does it avoid claims not supported by the tool results (invented products, prices, brands)?
Return JSON only: {"helpfulness": n, "faithfulness": n, "reason": "<one sentence>"}"""


def judge_scores(turns: list[dict]) -> None:
    j = ClaudeClient("claude-opus-5")
    for t in turns:
        prompt = (f"Customer: {t['text']}\nTools called: {t['tools_called']}\nAssistant reply:\n{t['answer']}")
        try:
            r = j.chat(JUDGE, [{"role": "user", "content": prompt}], max_tokens=400, purpose="judge")
            m = __import__("re").search(r"\{.*\}", r.text, __import__("re").S)
            t["judge"] = json.loads(m.group(0)) if m else None
        except Exception as e:  # budget or parse failure: leave unjudged
            t["judge"] = None
            t["judge_error"] = str(e)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "ollama", judge="--judge" in sys.argv)
