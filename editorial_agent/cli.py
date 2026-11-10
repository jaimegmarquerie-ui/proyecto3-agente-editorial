"""Command-line entry point for the editorial agent."""

import argparse
from typing import Sequence

from editorial_agent.graph import build_graph


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consulta el agente editorial.")
    parser.add_argument("--profile", required=True, help="Ruta al perfil JSON")
    parser.add_argument("--query", required=True, help="Consulta editorial")
    args = parser.parse_args(argv)

    result = build_graph().invoke(
        {"profile_path": args.profile, "query": args.query}
    )
    if result.get("error"):
        print(f"Error: {result['error']}")
    else:
        print(f"Resultados encontrados: {len(result.get('results', []))}")
    print(f"Dosier Markdown: {result['dossier_path']}")
    print(f"Evidencia: {result['evidence_path']}")
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
