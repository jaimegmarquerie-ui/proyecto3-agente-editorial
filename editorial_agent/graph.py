"""Minimal LangGraph flow: load profile, search Tavily, save evidence."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

import requests
from dotenv import dotenv_values
from langgraph.graph import END, START, StateGraph

from editorial_agent.profile import load_profile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class AgentState(TypedDict, total=False):
    profile_path: str
    query: str
    profile: dict[str, Any] | None
    results: list[dict[str, str]]
    error: str | None
    evidence_path: str
    created_at_utc: str


def build_graph(evidence_dir: str | Path | None = None):
    """Build the three-node flow; evidence_dir can be overridden in tests."""
    output_dir = Path(evidence_dir) if evidence_dir else PROJECT_ROOT / "evidence"

    def read_profile(state: AgentState) -> dict[str, Any]:
        try:
            profile = load_profile(state["profile_path"])
        except (OSError, ValueError) as error:
            return {"profile": None, "results": [], "error": str(error)}
        return {"profile": profile, "results": [], "error": None}

    def search_tavily(state: AgentState) -> dict[str, Any]:
        if state.get("error"):
            return {}

        api_key = dotenv_values(ENV_FILE).get("TAVILY_API_KEY")
        if not api_key or not api_key.strip():
            return {
                "error": "Falta TAVILY_API_KEY en el archivo .env; no se realizó la búsqueda."
            }

        response = None
        try:
            response = requests.post(
                TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {api_key.strip()}"},
                json={
                    "query": state["query"],
                    "search_depth": "basic",
                    "max_results": 5,
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as error:
            failed_response = error.response if error.response is not None else response
            status_code = (
                failed_response.status_code
                if failed_response is not None
                else "desconocido"
            )
            return {"error": f"Tavily devolvió un error HTTP {status_code}."}
        except requests.RequestException:
            return {"error": "No se pudo conectar con Tavily."}
        except ValueError:
            return {"error": "Tavily devolvió una respuesta que no es JSON válido."}

        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            return {"error": "La respuesta de Tavily no contiene una lista de resultados."}

        results = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "content": str(item.get("content", "")),
            }
            for item in raw_results[:5]
            if isinstance(item, dict)
        ]
        return {"results": results, "error": None}

    def save_result(state: AgentState) -> dict[str, str]:
        created_at = datetime.now(timezone.utc)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"editorial_{created_at:%Y%m%d_%H%M%S}_"
            f"{uuid4().hex[:8]}.json"
        )
        evidence_path = output_dir / filename
        profile = state.get("profile")
        evidence = {
            "query": state.get("query", ""),
            "created_at_utc": created_at.isoformat(timespec="seconds"),
            "profile_name": profile.get("name") if isinstance(profile, dict) else None,
            "results": state.get("results", []),
            "error": state.get("error"),
        }
        evidence_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "evidence_path": str(evidence_path),
            "created_at_utc": created_at.isoformat(timespec="seconds"),
        }

    graph = StateGraph(AgentState)
    graph.add_node("read_profile", read_profile)
    graph.add_node("search_tavily", search_tavily)
    graph.add_node("save_result", save_result)
    graph.add_edge(START, "read_profile")
    graph.add_edge("read_profile", "search_tavily")
    graph.add_edge("search_tavily", "save_result")
    graph.add_edge("save_result", END)
    return graph.compile()
