# Stack decision

**Industry / use case:** independent bookshops — editorial research dossiers.
**Primary stack:** **LangGraph (Python)**. n8n is out of scope for this MVP (possible secondary use later).

## Why LangGraph fits this problem and this user

Our agent must make decisions that depend on intermediate results: if the first search finds too few useful books, it rewrites the query and searches again; if a publisher page contradicts Google Books, both values are shown; if a tool fails, the flow takes a different path instead of guessing. LangGraph gives us explicit, inspectable state (query, bookshop profile, candidates, sources, errors, attempts) and **conditional edges** for those branches, written and tested in Python.

The core value of the product for a bookseller is **validation logic**: work vs edition, the most recent Spanish edition, ISBN checks, one source per field, a synopsis that is not a foreword note, a genre whose evidence appears in the publisher's text. That logic is easier to unit-test in Python than in visual nodes: the project has 49 automated tests, and they found real bugs during development.

## Why n8n is only secondary here

n8n would be faster for wiring integrations and for delivering the report to Slack or Gmail. Our MVP saves the dossier and its evidence in the repository, so that advantage matters less for now. A thin n8n workflow could later act as a **trigger** (for example, run the agent every Monday and email the dossier to the bookseller).

## Evidence from this week

- **Conditional decision (ReAct-style loop):** after the judge, a routing function checks how many books are useful and how many attempts are left; if needed, the agent rewrites the query (Gemini, with a fixed fallback rule) and searches again, at most twice. Real runs: `evidence/editorial_20261111_150400_*.json` (2 attempts: 0 → 10 books) and `evidence/editorial_20261111_150801_*.json` (1 attempt, enough).
- **Grounding without a vector database:** live sources with a URL per field, "no verificado" when missing, a template-based dossier, and Python checks on what the LLM proposes. See the README, section 4.

## Trade-offs we accept

- Slower to build than an n8n workflow; more code to maintain (mitigated with Codex and automated tests).
- No visual workflow out of the box: the graph is documented as a diagram in the README and in `docs/architecture.md`.
- HTTP calls, retries and fallbacks are written by us, not provided as ready-made nodes.
