from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from cohort_builder.config import REPO_ROOT, Settings  # noqa: E402
from cohort_builder.db import connect  # noqa: E402
from cohort_builder.orchestrator import CohortBuilder  # noqa: E402
from cohort_builder.synthetic import build_demo_database  # noqa: E402
from fake_llm import GOLDEN_INTENTS, LAAD_INTENTS, FakeLLM  # noqa: E402


@pytest.fixture(scope="session")
def demo_db_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("db") / "template.duckdb"
    con = connect(path)
    build_demo_database(con, n_persons=3000, seed=7)
    con.close()
    return path


@pytest.fixture
def settings(demo_db_template, tmp_path) -> Settings:
    db = tmp_path / "test.duckdb"
    shutil.copy(demo_db_template, db)
    return dataclasses.replace(Settings(), db_path=db, model="fake-model", llm_mode="cached")


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(dict(GOLDEN_INTENTS))


@pytest.fixture
def builder(settings, fake_llm) -> CohortBuilder:
    b = CohortBuilder(settings, backend=fake_llm)
    yield b
    b.con.close()


@pytest.fixture
def laad_llm() -> FakeLLM:
    return FakeLLM(dict(LAAD_INTENTS))


@pytest.fixture
def laad_builder(settings, laad_llm) -> CohortBuilder:
    b = CohortBuilder(dataclasses.replace(settings, dataset="iqvia_laad"), backend=laad_llm)
    yield b
    b.con.close()


@pytest.fixture
def example_ir_path() -> Path:
    return REPO_ROOT / "examples" / "t2dm_metformin_hba1c.json"
