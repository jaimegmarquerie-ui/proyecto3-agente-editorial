import json
from pathlib import Path

import requests

from editorial_agent import graph as graph_module


PROFILE_PATH = Path("profiles/moscardo.json")


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {
                    "title": "Nuestra parte de noche",
                    "url": "https://example.test/book",
                    "content": "Fragmento de prueba",
                }
            ]
        }


def test_graph_searches_with_only_query_and_saves_evidence(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        graph_module, "dotenv_values", lambda _path: {"TAVILY_API_KEY": "test-only-key"}
    )

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(graph_module.requests, "post", fake_post)
    app = graph_module.build_graph(evidence_dir=tmp_path)
    result = app.invoke(
        {"profile_path": str(PROFILE_PATH), "query": "consulta de prueba"}
    )

    assert len(calls) == 1
    assert calls[0]["json"] == {
        "query": "consulta de prueba",
        "search_depth": "basic",
        "max_results": 5,
    }
    assert result["profile"]["name"] == "La Insumisa"
    assert result["results"][0]["url"] == "https://example.test/book"
    assert result.get("error") is None

    evidence_path = Path(result["evidence_path"])
    assert evidence_path.parent == tmp_path
    assert evidence_path.name.startswith("editorial_")
    evidence_text = evidence_path.read_text(encoding="utf-8")
    evidence = json.loads(evidence_text)
    assert evidence["query"] == "consulta de prueba"
    assert evidence["profile_name"] == "La Insumisa"
    assert evidence["results"][0]["url"] == "https://example.test/book"
    assert "test-only-key" not in evidence_text


def test_graph_saves_tavily_http_error_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(
        graph_module, "dotenv_values", lambda _path: {"TAVILY_API_KEY": "test-only-key"}
    )

    response = requests.Response()
    response.status_code = 503

    def fake_post(*_args, **_kwargs):
        return response

    monkeypatch.setattr(graph_module.requests, "post", fake_post)
    app = graph_module.build_graph(evidence_dir=tmp_path)
    result = app.invoke(
        {"profile_path": str(PROFILE_PATH), "query": "consulta de prueba"}
    )

    assert result["error"] == "Tavily devolvió un error HTTP 503."
    evidence = json.loads(Path(result["evidence_path"]).read_text(encoding="utf-8"))
    assert evidence["error"] == result["error"]
    assert evidence["results"] == []
