"""Stable, versioned vocabularies for string categorical features.

Replaces ``hash(value) % N`` encodings, which can collide and change with the
hash function. Each vocabulary maps the sorted distinct values of a column in
the static ``articles``/``customers`` tables to consecutive integer codes:

- code 0 is reserved for a NULL in the source (an explicit "missing" category);
- codes 1..n are the sorted distinct values;
- a value not in the vocabulary (e.g. a later catalogue) encodes to NULL, which
  LightGBM treats as missing rather than as a known category.

The vocabularies depend only on catalogue/customer metadata, never on
transactions, so they carry no temporal information. They are written once to
``data/processed/vocab/<VERSION>.json``; ``version`` is a hash of the content,
so a model can record exactly which encoding it was trained with.
"""
from __future__ import annotations

import hashlib
import json

VOCAB_COLUMNS = {
    "product_group": ("articles", "product_group_name"),
    "club_status": ("customers", "club_member_status"),
    "news_freq": ("customers", "fashion_news_frequency"),
}


def build_vocab(con) -> dict:
    vocab = {}
    for name, (table, col) in VOCAB_COLUMNS.items():
        vals = [r[0] for r in con.execute(
            f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY 1").fetchall()]
        vocab[name] = {v: i + 1 for i, v in enumerate(vals)}
    body = json.dumps(vocab, sort_keys=True)
    return {"version": hashlib.sha1(body.encode()).hexdigest()[:10], "maps": vocab}


def load_vocab(con, cfg=None) -> dict:
    """Load the stored vocabulary, creating it from the database on first use."""
    if cfg is None:
        from ensemble.config import load_config
        cfg = load_config()
    path = cfg.path("processed") / "vocab" / "vocab.json"
    if path.exists():
        return json.loads(path.read_text())
    v = build_vocab(con)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(v, indent=1, sort_keys=True))
    return v


def register(con, vocab: dict) -> None:
    """Create TEMP TABLE ``_vocab_<name>(value, code)`` for each vocabulary."""
    for name, m in vocab["maps"].items():
        rows = ", ".join(f"('{k.replace(chr(39), chr(39) * 2)}', {v})" for k, v in m.items())
        con.execute(f"CREATE OR REPLACE TEMP TABLE _vocab_{name} AS SELECT * FROM (VALUES {rows}) t(value, code)")


def encode(name: str, table_alias: str, col: str) -> tuple[str, str]:
    """(LEFT JOIN clause, select expression): 0 if NULL, the code if known, NULL if unknown."""
    v = f"_v_{name}"
    return (f"LEFT JOIN _vocab_{name} {v} ON {v}.value = {table_alias}.{col}",
            f"CASE WHEN {table_alias}.{col} IS NULL THEN 0 ELSE {v}.code END")
