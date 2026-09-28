"""LLM abstraction layer (D-018): one interface, two back ends.

  OllamaClient  local open-source model (free), used for development
  ClaudeClient  Anthropic API (paid), used for the final comparison

Messages use one neutral format:
  {"role": "user", "content": str, "images": [bytes, ...]}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Tools are JSON-schema dicts: {"name", "description", "parameters"}.

Every paid call is metered into reports/llm_spend.json; a call that would start
after cumulative spend reaches the budget raises BudgetExceeded.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field

from ensemble.config import ROOT

# USD per million tokens (Anthropic first-party list prices).
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}
BUDGET_USD = float(os.environ.get("ENSEMBLE_LLM_BUDGET_USD", "8.0"))
SPEND_FILE = ROOT / "reports" / "llm_spend.json"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    raw_content: object = None       # provider-native assistant content, for faithful replay


def spend() -> dict:
    return json.loads(SPEND_FILE.read_text()) if SPEND_FILE.exists() else {"total_usd": 0.0, "by_purpose": {}}


def _record(cost: float, purpose: str) -> None:
    s = spend()
    s["total_usd"] = round(s["total_usd"] + cost, 6)
    s["by_purpose"][purpose] = round(s["by_purpose"].get(purpose, 0.0) + cost, 6)
    SPEND_FILE.parent.mkdir(exist_ok=True)
    SPEND_FILE.write_text(json.dumps(s, indent=2))


def api_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        m = re.search(r"^ANTHROPIC_API_KEY=(.+)$", env.read_text(), re.M)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


class OllamaClient:
    name = "ollama"

    def __init__(self, model: str = "qwen3-vl:8b-instruct", host: str = "http://localhost:11434", think: bool = False):
        self.model, self.host, self.think = model, host, think

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 2048, json_mode: bool = False, purpose: str = "dev") -> LLMResponse:
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                msgs.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
            elif m["role"] == "assistant":
                msgs.append({"role": "assistant", "content": m.get("content", ""),
                             "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in m.get("tool_calls", [])]})
            else:
                d = {"role": "user", "content": m["content"]}
                if m.get("images"):
                    d["images"] = [base64.b64encode(b).decode() for b in m["images"]]
                msgs.append(d)
        body = {"model": self.model, "messages": msgs, "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16384}}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        if json_mode:
            body["format"] = "json"
        if self.model.startswith("qwen3"):
            body["think"] = self.think       # thinking off by default: latency matters in chat
        t = time.time()
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=600))
        msg = out.get("message", {})
        calls = [{"id": f"call_{i}", "name": c["function"]["name"], "arguments": c["function"].get("arguments", {})}
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        return LLMResponse(text=msg.get("content", ""), tool_calls=calls,
                           input_tokens=out.get("prompt_eval_count", 0), output_tokens=out.get("eval_count", 0),
                           latency_s=time.time() - t)


class ClaudeClient:
    name = "claude"

    def __init__(self, model: str = "claude-opus-5"):
        import anthropic
        key = api_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (see .env)")
        self.client, self.model = anthropic.Anthropic(api_key=key), model

    def _convert(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.b64encode(b).decode()}} for b in m.get("images", [])]
                content.append({"type": "text", "text": m["content"]})
                out.append({"role": "user", "content": content})
            elif m["role"] == "assistant":
                if m.get("raw_content") is not None:
                    out.append({"role": "assistant", "content": m["raw_content"]})
                else:
                    blocks = ([{"type": "text", "text": m["content"]}] if m.get("content") else []) + [
                        {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                        for c in m.get("tool_calls", [])]
                    out.append({"role": "assistant", "content": blocks})
            else:  # tool results are sent as user turns; consecutive ones are merged
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and all(b.get("type") == "tool_result" for b in out[-1]["content"]):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 4096, json_mode: bool = False, purpose: str = "eval") -> LLMResponse:
        if spend()["total_usd"] >= BUDGET_USD:
            raise BudgetExceeded(f"LLM budget of ${BUDGET_USD} reached")
        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=self._convert(messages),
                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}])
        if tools:
            ts = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
            ts[-1]["cache_control"] = {"type": "ephemeral"}   # cache tool definitions + system prompt
            kwargs["tools"] = ts
        if self.model.startswith("claude-opus-5"):
            kwargs["output_config"] = {"effort": "medium"}
        t = time.time()
        r = self.client.messages.create(**kwargs)
        u = r.usage
        pin, pout = PRICES.get(self.model, (5.0, 25.0))
        cache_w = getattr(u, "cache_creation_input_tokens", 0) or 0
        cache_r = getattr(u, "cache_read_input_tokens", 0) or 0
        cost = (u.input_tokens * pin + cache_w * pin * 1.25 + cache_r * pin * 0.1 + u.output_tokens * pout) / 1e6
        _record(cost, purpose)
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": b.input} for b in r.content if b.type == "tool_use"]
        if r.stop_reason == "refusal":
            text = text or "[refused]"
        return LLMResponse(text=text, tool_calls=calls, input_tokens=u.input_tokens + cache_w + cache_r,
                           output_tokens=u.output_tokens, cost_usd=cost, latency_s=time.time() - t,
                           raw_content=[b.model_dump(exclude_none=True) for b in r.content])


LOCAL_MODELS = {"assistant": "qwen3-vl:8b-instruct",  # vision + tool calling, no thinking (the default 8b tag ignores think=false)
                "vision": "qwen2.5vl:7b"}      # garment detection with pixel boxes (no tool support in Ollama)


def get_client(role: str = "assistant", backend: str | None = None, model: str | None = None):
    """``role`` is "assistant" (tool calling) or "vision" (garment detection)."""
    backend = backend or os.environ.get("ENSEMBLE_LLM", "ollama")
    if backend == "claude":
        return ClaudeClient(model or "claude-opus-5")
    return OllamaClient(model or LOCAL_MODELS[role])
