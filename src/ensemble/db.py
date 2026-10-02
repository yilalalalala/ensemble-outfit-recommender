"""DuckDB connection factory (D-007)."""
from __future__ import annotations

import duckdb

from ensemble.config import Config, load_config


def connect(cfg: Config | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg = cfg or load_config()
    path = cfg.path("database")
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    tmp = cfg.path("interim") / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{cfg.duckdb.memory_limit}'")
    con.execute(f"SET threads={int(cfg.duckdb.threads)}")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    return con
