"""Run one basic Tavily search and save its source snippets as evidence."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values


QUERY = "Mariana Enríquez Nuestra parte de noche Anagrama"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
EVIDENCE_FILE = PROJECT_ROOT / "evidence" / "tavily_probe.json"


def main() -> int:
    api_key = dotenv_values(ENV_FILE).get("TAVILY_API_KEY")
    if not api_key or not api_key.strip():
        print(
            f"Falta TAVILY_API_KEY en el archivo {ENV_FILE.name}; "
            "no se realizó la búsqueda.",
            file=sys.stderr,
        )
        return 1

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            json={"query": QUERY, "search_depth": "basic", "max_results": 5},
            timeout=30,
        )
    except requests.RequestException:
        print(
            "No se pudo conectar con Tavily; no se guardó la evidencia.",
            file=sys.stderr,
        )
        return 1

    if not response.ok:
        print(
            f"Tavily devolvió un error HTTP {response.status_code}; "
            "no se guardó la evidencia.",
            file=sys.stderr,
        )
        return 1

    try:
        payload = response.json()
    except ValueError:
        print(
            "Tavily devolvió una respuesta que no es JSON válido; "
            "no se guardó la evidencia.",
            file=sys.stderr,
        )
        return 1

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        print(
            "La respuesta de Tavily no contiene una lista de resultados; "
            "no se guardó la evidencia.",
            file=sys.stderr,
        )
        return 1

    evidence = {
        "query": QUERY,
        "datetime_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
            }
            for result in results[:5]
            if isinstance(result, dict)
        ],
    }

    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_FILE.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Búsqueda completada: {len(evidence['results'])} resultados. "
        f"Evidencia guardada en {EVIDENCE_FILE.relative_to(PROJECT_ROOT)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
