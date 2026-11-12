# Architecture

## Graph (LangGraph)

```mermaid
flowchart TD
  START([Trigger: CLI request + bookshop profile]) --> read_profile
  read_profile --> search_tavily[search_tavily<br/>web search, counts attempts]
  search_tavily --> fetch_books[fetch_books<br/>Open Library: 5 works by default (max 10),<br/>latest Spanish edition, dedupe]
  fetch_books --> fetch_google_books[fetch_google_books<br/>cross-check, discrepancies]
  fetch_google_books --> fetch_publisher_pages[fetch_publisher_pages<br/>Tavily limited to the publisher's site]
  fetch_publisher_pages --> describe_books[describe_books<br/>Gemini proposes synopsis, genre, themes<br/>Python validates against sources]
  describe_books --> judge_results[judge_results<br/>Gemini: useful for this bookshop?]
  judge_results --> route{route_after_judge}
  route -->|enough useful books| write_dossier
  route -->|not enough, attempts < 2| reformulate_query[reformulate_query<br/>Gemini or fallback rule]
  reformulate_query --> search_tavily
  route -->|not enough, no attempts left| write_dossier
  route -->|tool error| write_dossier
  write_dossier[write_dossier<br/>template, no LLM writing] --> save_result[save_result<br/>evidence JSON]
  save_result --> END([END])
```

**State** (shared dictionary): query, author, profile, search results, books (edition, Google Books data, publisher page, validated description, discrepancies), judgments, attempts and search history, research status, errors per source, tools used/failed, output paths.

**Sufficiency rule:** ≥2 useful books, or at least 1 book found and all of them useful.

## Failure paths

| Failure | Behaviour |
|---|---|
| Invalid profile or CLI input | Stop with a clear error; no credits spent |
| Tavily or judge fails | No retry of the search; dossier marked **NOT FINAL**; no invented results |
| Gemini 429/503/timeout | Retries with waits, then fallback model |
| Gemini returns invalid JSON | Error with `finishReason`; describe runs in batches of 3, so one failed batch only affects its books |
| Google Books 429/503 after retries | Stops calling for the rest of the run; data marked missing |
| Open Library / publisher page / Google Books error for one book | Recorded in its own error list; the flow continues |
| Edition language unknown | Accepted with "idioma no verificado" (explicit non-Spanish editions are rejected) |
| Retrieved text contains instructions | Treated as data only (prompt-injection guard) |
| Not enough useful books after 2 attempts | Dossier says so honestly ("No se encontraron suficientes libros útiles…") |
