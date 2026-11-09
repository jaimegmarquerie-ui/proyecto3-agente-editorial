"""Loading and validating bookshop profiles."""

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "name",
    "values",
    "genres_sections",
    "output_language",
    "location",
)


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load a JSON profile and check that required top-level fields exist."""
    profile_path = Path(path)

    try:
        content = profile_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"No se encontró el archivo de perfil: {profile_path}"
        ) from error
    except OSError as error:
        raise OSError(
            f"No se pudo leer el archivo de perfil '{profile_path}': {error}"
        ) from error

    try:
        profile = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"El archivo de perfil '{profile_path}' no contiene JSON válido "
            f"(línea {error.lineno}, columna {error.colno})."
        ) from error

    if not isinstance(profile, dict):
        raise ValueError(
            f"El perfil '{profile_path}' debe ser un objeto JSON."
        )

    missing_fields = [field for field in REQUIRED_FIELDS if field not in profile]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"Faltan campos obligatorios en el perfil '{profile_path}': {missing}."
        )

    return profile
