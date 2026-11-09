"""Command-line entry point for the editorial agent skeleton."""

import argparse
from typing import Sequence

from editorial_agent.profile import load_profile


def _format_values(values: object) -> str:
    if isinstance(values, list):
        labels = [
            str(value.get("label", value)) if isinstance(value, dict) else str(value)
            for value in values
        ]
        return ", ".join(labels)
    return str(values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consulta el agente editorial.")
    parser.add_argument("--profile", required=True, help="Ruta al perfil JSON")
    parser.add_argument("--query", required=True, help="Consulta editorial")
    args = parser.parse_args(argv)

    try:
        profile = load_profile(args.profile)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Librería: {profile['name']}")
    print(f"Valores: {_format_values(profile['values'])}")
    print(f"Idioma de salida: {profile['output_language']}")
    print(f"Consulta: {args.query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
