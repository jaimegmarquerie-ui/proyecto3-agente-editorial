import json

import pytest

from editorial_agent.profile import load_profile


def test_load_valid_moscardo_profile():
    profile = load_profile("profiles/moscardo.json")

    assert profile["name"] == "La Insumisa"
    assert profile["values"]
    assert profile["genres_sections"]
    assert profile["output_language"] == "es"
    assert profile["location"]


def test_load_profile_reports_missing_required_field(tmp_path):
    path = tmp_path / "incomplete.json"
    path.write_text(
        json.dumps(
            {
                "name": "Librería de prueba",
                "values": [],
                "genres_sections": [],
                "output_language": "es",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="location"):
        load_profile(path)


def test_load_profile_reports_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{ "name": ', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON válido"):
        load_profile(path)
