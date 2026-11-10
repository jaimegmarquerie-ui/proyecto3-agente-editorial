"""Free diagnostic for the Gemini judge (no Tavily credits used).

Without arguments: lists the Gemini models this API key can see and says
whether graph.GEMINI_MODEL is among them.

With model names: sends a tiny test prompt to each one and reports whether
it really answers. A key on the free tier without billing cannot be charged.

The key itself is never printed.

Run from the project folder:
    python -m editorial_agent.gemini_check
    python -m editorial_agent.gemini_check gemini-3.8-flash gemini-3.5-flash
"""

import sys

import requests
from dotenv import dotenv_values

from editorial_agent.graph import ENV_FILE, GEMINI_GENERATE_CONTENT_URL, GEMINI_MODEL

LIST_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _service_message(response: requests.Response, api_key: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return "respuesta no JSON"
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = str(error.get("message", "sin mensaje")) if isinstance(error, dict) else str(error)
    return " ".join(message.replace(api_key, "[REDACTED]").split())[:300]


def list_models(api_key: str) -> int:
    try:
        response = requests.get(
            LIST_MODELS_URL,
            headers={"x-goog-api-key": api_key},
            params={"pageSize": 1000},
            timeout=30,
        )
    except requests.RequestException:
        print("No se pudo conectar con Gemini.")
        return 1

    print(f"HTTP {response.status_code}")
    if not response.ok:
        print(f"Mensaje del servicio: {_service_message(response, api_key)}")
        return 1

    usable = sorted(
        model["name"].removeprefix("models/")
        for model in response.json().get("models", [])
        if "generateContent" in model.get("supportedGenerationMethods", [])
    )
    print(f"Modelo configurado: {GEMINI_MODEL}")
    print(f"¿Aparece en la lista?: {'SI' if GEMINI_MODEL in usable else 'NO'}")
    print("Modelos 'flash' en la lista:")
    for name in usable:
        if "flash" in name:
            print(f"  - {name}")
    return 0


def ping_model(api_key: str, model: str) -> bool:
    try:
        response = requests.post(
            GEMINI_GENERATE_CONTENT_URL.format(model=model),
            headers={"x-goog-api-key": api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": "Responde solo: OK"}]}],
                "generationConfig": {"maxOutputTokens": 50},
            },
            timeout=45,
        )
    except requests.RequestException:
        print(f"{model}: no se pudo conectar")
        return False

    if response.ok:
        print(f"{model}: FUNCIONA (HTTP {response.status_code})")
        return True
    print(f"{model}: NO (HTTP {response.status_code}) {_service_message(response, api_key)}")
    return False


def main(argv: list[str]) -> int:
    api_key = (dotenv_values(ENV_FILE).get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        print("Falta GEMINI_API_KEY en .env.")
        return 1
    if not argv:
        return list_models(api_key)
    results = [ping_model(api_key, model) for model in argv]
    return 0 if any(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
