# Editorial Research Agent (Project 3 · Ironhack AI Consulting)

> **Status: work in progress (MVP, day 2 of 5).** Not a final submission.

An autonomous research agent for **independent bookshops**. Given a topic or an author and a bookshop profile, it searches the web, finds **verified Spanish editions** of relevant books, judges how well each book fits the bookshop's values, and writes a Markdown dossier with sources. **The bookseller makes the final decision.**

## How it works (LangGraph)

```
START → read_profile → search_tavily → fetch_books → judge_results → write_dossier → save_result → END
```

| Node | What it does | External tool |
|------|--------------|---------------|
| `read_profile` | Loads and validates the bookshop profile (`profiles/moscardo.json`) | — |
| `search_tavily` | Web search for context | **Tavily** |
| `fetch_books` | Finds the author's works, keeps the first Spanish ISBN (978-84 / 84) and verifies that edition | **Open Library** |
| `judge_results` | Decides `util` / `no_util` per book, citing the profile values or sections it matches | **Gemini** (`gemini-3.5-flash`, fallback `gemini-3-flash-preview`) |
| `write_dossier` | Fills a Markdown template **only with data already retrieved** (no LLM writing) | — |
| `save_result` | Saves a JSON evidence file | — |

**Grounding rules:** bibliographic fields appear only if present in a source; otherwise they show "no verificado". The agent's opinions are labelled "Interpretación del agente". Source text is treated as data, never as instructions.

**Error handling:** provider error messages are saved (never keys or headers); Gemini retries on 429/503, timeouts and connection errors (waits 2 s and 5 s), then tries a fallback model; if judging fails, the dossier is marked **NO FINAL**.

## Setup

```bash
conda create -n proyecto3 python=3.11 -y
conda activate proyecto3
pip install -r requirements.txt
cp .env.example .env   # then add your own TAVILY_API_KEY and GEMINI_API_KEY
```

## Run

```bash
python -m editorial_agent.cli --profile profiles/moscardo.json \
  --query "libros escritos por Leila Guerriero" --author "Leila Guerriero"
```

Outputs: `samples/dossier_<date>_<id>.md` and `evidence/editorial_<timestamp>_<id>.json`.

Diagnostic (free, no Tavily credits): `python -m editorial_agent.gemini_check`

## Tests

```bash
pytest
```

All external calls are mocked in tests (no network, no cost).

## Known limitations (10 Nov 2026)

- Synopsis is available only when Open Library has a work description; otherwise "no verificado".
- "Relacionado con" currently lists the other retrieved books, not curated related titles.
- Tavily and LLM results are not deterministic; a repeatable judge test with saved results is pending.
- Gemini free tier is sometimes overloaded (HTTP 503); retries and a fallback model mitigate it.
- Only Spanish-registered editions (978-84) are considered, by design.

## Docs

- `stack_decision.md` — why LangGraph
- `docs/architecture.md` — architecture draft
- `docs/requirements_checklist.md` — requirements status
- `gtm_future_sprints.md` — future GTM sprints and ROI hypothesis
