# Project 3 — Requirements checklist

Sources: official "Project 3: Final Deliverables" (shared by the teacher, 12 Nov 2026), Project 3 brief and rubric, and our product decisions.

**Deadline:** GitHub repository URL submitted in the Student Portal (`project-3`) **before the presentations on Friday 13 November, from 11:00**.

**Status:** ☑ done · ◐ in progress · ☐ to do (last update: 12 Nov 2026)

## Official deliverables

| # | Deliverable | Where | Status |
|---|---|---|---|
| 1 | Working MVP: trigger → research / tools → structured report; entry point in README | `main.py`, `editorial_agent/`, README §2 | ☑ |
| 2 | Stack decision: primary stack, why it fits, why the other is secondary | `stack_decision.md` | ☑ |
| 3 | `gtm_future_sprints.md` in repo root: 2–3 sprints × 5 parts (goal, buyer, channel, deliverable, metric) | `gtm_future_sprints.md` | ☑ |
| 4 | Slides as PDF in the repo | `presentation/project3_slides.pdf` | ☑ (to review and rehearse) |
| 5 | README: setup, how to run, file map, architecture, tools/APIs, env vars | `README.md` | ☑ |
| 6 | Demo recording link in README (opens without login) | README, top | ☐ record and add link |
| 6 | Live demo inside a 6-minute presentation, with a backup | presentation | ☐ rehearse with a timer |
| — | `.env.example` in repo; no `.env`, no keys committed | `.env.example`, `.gitignore`, `git log -- .env` empty | ☑ |
| — | No real personal data in the repo | profile of the founder's own bookshop; public book data | ☑ |
| — | GitHub URL submitted in the Student Portal (`project-3`) | Student Portal | ☐ Friday before 11:00 (aim for 10:00) |
| — | Teacher can open the repository (currently private) | GitHub settings | ☐ make public or add collaborator |

## Rubric points and our evidence

| Point | Evidence |
|---|---|
| ≥3 real external tools | Tavily, Open Library, Google Books (+ Gemini as the LLM). Real runs in `evidence/` (11 Nov, 4–5 services, 0 failures) |
| Autonomy after the trigger | One command runs the whole graph; conditional edge rewrites the query when research is weak |
| Branching / ReAct-style loop | `route_after_judge` + `reformulate_query`, max 2 attempts; real runs 150400 (2 attempts) and 150801 (1 attempt) |
| Errors and validation | Retries, fallback model, rate-limit stop, NOT FINAL dossier, Python checks on LLM output; 47 mocked tests |
| Grounding | Live sources with URL per field, "no verificado", template dossier (no vector DB; argued in README §4) |
| ≥2 sample reports | `samples/` (several dossiers from 10–11 Nov) |
| Free tiers | $0 per report |

## Product decisions (ours)

| Decision | Why |
|---|---|
| Agent for independent bookshops; demo profile = La Insumisa (Moscardó, Madrid) | The founder's real use case |
| Editable JSON bookshop profile | Same engine, different recommendations per bookshop |
| Tools: Tavily, Open Library, Google Books; GDELT postponed (429 per IP) | Real, free, verifiable |
| LLM: Gemini free tier (Cohere dropped after 401 + paid key) | $0 constraint |
| Publisher page as primary source; most recent Spanish edition | Bookseller's criteria |
| No invented ISBN, stock, sales or events; "no verificado" when missing | Honesty of the dossier |
| Reader popularity is information only, never a penalty | Independent bookshops value non-mainstream books |
