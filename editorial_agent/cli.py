"""Command-line entry point for the editorial agent."""

import argparse
from typing import Sequence

from editorial_agent.graph import build_graph


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consulta el agente editorial.")
    parser.add_argument("--profile", required=True, help="Ruta al perfil JSON")
    parser.add_argument("--query", required=True, help="Consulta editorial")
    parser.add_argument("--author", help="Autor del que buscar libros en Open Library")
    parser.add_argument(
        "--max-works",
        type=int,
        choices=range(1, 11),
        metavar="N",
        help="Obras de Open Library a revisar (1-10; por defecto 5)",
    )
    args = parser.parse_args(argv)

    state = {"profile_path": args.profile, "query": args.query, "author": args.author}
    if args.max_works is not None:
        state["max_works"] = args.max_works
    result = build_graph().invoke(state)
    if result.get("error"):
        print(f"Error: {result['error']}")
    else:
        print(f"Resultados encontrados: {len(result.get('results', []))}")
    print(f"Dosier Markdown: {result['dossier_path']}")
    print(f"Evidencia: {result['evidence_path']}")
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
