# Demo script (6 minutes in total, hard limit)

Speaker notes with exact words are inside the slides (PowerPoint: View → Notes).

## Before you start (Friday morning)

1. `conda activate proyecto3`, then `pytest` (expect all tests to pass).
2. One full test run, to check that the keys and services work: open `run_demo.py` in VS Code and press Run ▶ (or `python main.py --profile profiles/moscardo.json --query "libros escritos por Leila Guerriero" --author "Leila Guerriero"`). It prints the run time.
3. Keep open in VS Code: the newest good dossier in `samples/` (backup) and the terminal in the project folder.
4. Keep the demo recording link open in a browser tab (second backup).
5. Font size: zoom VS Code (Ctrl and +) so the audience can read the terminal and the dossier.

## Timing

| Time | Slide | What you do |
|---|---|---|
| 0:00–0:20 | 1 Title | **Start the agent first** (`run_demo.py` → Run ▶). It takes 3–9 min, so it may not finish in time |
| 0:20–1:00 | 2 Problem | Who, what they do today, ~4 hours per dossier (estimate) |
| 1:00–1:30 | 3 How it works | Five steps in plain words; it retries on its own |
| 1:30–3:45 | 4 Demo | Show the live terminal running → open the **backup dossier** (complete run) → one book → click one source → if the live run finished, show its last lines |
| 3:45–4:15 | 5 Result | 3–9 minutes instead of ~4 hours; sources; honest "not verified" |
| 4:15–4:55 | 6 Under the hood | LangGraph, Gemini, 3 data tools, grounding, errors, $0, 3–9 min |
| 4:55–5:25 | 7 GTM | Three sprints, one number each |
| 5:25–5:40 | 8 Thank you | "The agent recommends; the bookseller decides." |

## Important: run time

On 12 Nov the run took 8 min 50 s because the free Gemini tier answered 503 (busy) and the agent retried. Plan the demo around the **backup dossier** (`samples/dossier_20261111_7a1a1973.md`, a complete run with 0 failed tools) and treat the live run as a bonus that proves it really runs.

## Demo steps (slide 4)

1. Terminal: point at the last two lines (dossier path and evidence path).
2. Open the dossier: show **Investigación** (attempts, query, status "suficiente", tools used, no failed tools).
3. Scroll to one book (e.g. *Teoría de la gravedad* or *La llamada*): verified ISBN, most recent edition, synopsis with source, genre (or "no verificado"), themes labelled as interpretation, judge's reason citing the profile.
4. Ctrl+click one link (Open Library or the publisher page).
5. Only if there is time: show `evidence/editorial_20261111_150400_*.json` → `search_history`: attempt 1 "obras de Leila Guerriero" found 0 books → attempt 2 rewritten query found 10.

## If something fails

- Run still going when you reach slide 4: talk through slide 4 steps on the backup dossier, then switch to the live one if it finishes.
- API error or "NO FINAL": say *"Live APIs can fail. When that happens the agent marks the dossier NOT FINAL instead of guessing. Here is a complete run from yesterday."* and open the backup dossier.
- No internet: play the recording.
