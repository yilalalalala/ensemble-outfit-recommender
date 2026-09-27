import pytest

from ensemble.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def con(cfg):
    if not cfg.path("database").exists():
        pytest.skip("database not built; run `make ingest`")
    from ensemble.db import connect

    c = connect(cfg, read_only=True)
    yield c
    c.close()
