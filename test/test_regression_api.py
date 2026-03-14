from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests


def load_cases() -> List[dict]:
    cases_path = Path(__file__).parent / "cases_sin_contexto.json"
    with cases_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("cases.json debe ser un arreglo JSON []")
    return data


CASES = load_cases()


# ← ya NO hay fixture api_url aquí, viene del conftest.py


def post(api_url: str, question: str, session_id: str) -> Dict[str, Any]:
    payload = {"question": question, "chatSessionId": session_id}
    r = requests.post(api_url, json=payload, timeout=25)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise AssertionError(f"Respuesta no es JSON object. resp={data}")
    return data


def assert_route(case: dict, resp: dict):
    route = resp.get("route")

    exp = case.get("expect") or {}
    exp_any = case.get("expectAny") or {}

    if "route" in exp:
        expected = exp["route"]
        assert route == expected, f"route esperado={expected}, obtenido={route}, resp={resp}"

    if "route" in exp_any:
        expected_list = exp_any["route"]
        assert route in expected_list, f"route esperado in {expected_list}, obtenido={route}, resp={resp}"


def assert_contains(case: dict, resp: dict):
    """
    Validación opcional:
      "contains": {"data.message": ["SNIES", "programa"]}
    """
    contains = case.get("contains")
    if not contains:
        return

    def get_path(d: dict, path: str):
        cur: Any = d
        for p in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    for path, needles in contains.items():
        val = get_path(resp, path)
        if val is None:
            raise AssertionError(f"Falta path '{path}' en resp={resp}")
        s = str(val)
        if isinstance(needles, str):
            needles = [needles]
        for n in needles:
            assert n in s, f"No contiene '{n}' en {path}. valor={s}"


def assert_resolved(case: dict, resp: dict):
    """
    Validación opcional:
      "expectResolved": true/false
      Valida data.resolved sin importar la ruta.

      "resolvedByRoute": {"NARRATIVE_SQL": true, "NARRATIVE_NOT_FOUND": false}
      Valida data.resolved condicionado a la ruta que llegó.
      Si la ruta recibida no está en el mapa, no se valida.
    """
    data = resp.get("data") or {}
    route = resp.get("route", "")

    if "expectResolved" in case:
        expected = case["expectResolved"]
        actual = data.get("resolved")
        if actual is None:
            raise AssertionError(
                f"data.resolved ausente. expectResolved={expected}, resp={resp}"
            )
        assert actual == expected, \
            f"resolved esperado={expected}, obtenido={actual}"

    rbr = case.get("resolvedByRoute") or {}
    if route in rbr:
        expected = rbr[route]
        actual = data.get("resolved")
        if actual is None:
            raise AssertionError(
                f"data.resolved ausente para route={route}. "
                f"resolvedByRoute esperaba={expected}, resp={resp}"
            )
        assert actual == expected, \
            f"resolved para route={route}: esperado={expected}, obtenido={actual}"


def assert_snies(case: dict, resp: dict):
    """
    Opcional:
      "expectSnies": "116978"
      "expectSniesAny": ["116978","109020"]
    """
    data = resp.get("data") or {}
    sn = data.get("snies")
    if sn is None:
        return

    if "expectSnies" in case:
        assert str(sn) == str(
            case["expectSnies"]), f"snies esperado={case['expectSnies']}, obtenido={sn}"

    if "expectSniesAny" in case:
        allowed = [str(x) for x in case["expectSniesAny"]]
        assert str(
            sn) in allowed, f"snies esperado in {allowed}, obtenido={sn}"


@pytest.mark.regression
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.get("name", "SIN_NOMBRE"))
def test_regression_case(api_url, case):
    name = case.get("name", "SIN_NOMBRE")
    session_id = case.get("session", "reg-default")
    question = case.get("question", "")

    if not question:
        pytest.fail(f"[{name}] Caso sin 'question'")

    t0 = time.time()
    resp = post(api_url, question, session_id)
    dt = round(time.time() - t0, 3)

    try:
        assert_route(case, resp)
        assert_contains(case, resp)
        assert_resolved(case, resp)
        assert_snies(case, resp)
    except AssertionError as e:
        raise AssertionError(
            f"\n❌ Caso: {name}\nQ: {question}\nResp: {resp}\nTiempo: {dt}s\nError: {e}"
        ) from e
