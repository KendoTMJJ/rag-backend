import json
import os
from pathlib import Path
import pytest
from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")


def _load_cases(filename: str) -> list[dict]:
    here = Path(__file__).parent
    with (here / filename).open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def api_url() -> str:
    url = os.getenv("RAG_API_URL")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def cases() -> list[dict]:
    return _load_cases("cases_sin_contexto.json")
