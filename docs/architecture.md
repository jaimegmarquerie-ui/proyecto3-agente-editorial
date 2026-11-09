# Architecture — happy path (≤10 steps) and failure paths — DRAFT

```mermaid
flowchart TD
  A[1. CLI trigger<br/>profile.json + query] --> B[2. Validate inputs<br/>profile schema, query type]
  B -->|invalid| X1[Stop: clear error message]
  B --> C[3. Verify event if the query claims one<br/>GDELT + Tavily]
  C -->|not verified| C2[Mark as unverified in state<br/>never present as fact]
  C --> D[4. Discover sources<br/>Tavily: publisher pages first]
  C2 --> D
  D --> E[5. Enrich metadata<br/>Google Books]
  E --> F[6. Normalize work / edition / ISBN<br/>flag contradictions]
  F --> G[7. Filter and rank vs profile<br/>LLM, explicit criteria]
  G --> H[8. Grounding check<br/>every claim has a source URL]
  H -->|0 reliable records| X2[Report: no verified results + reasons]
  H --> I[9. Write dossier ≤5 records<br/>samples/ as Markdown + JSON]
  I --> J[10. Run log: tools called, status, errors]
```

## Failure paths
| Failure | Behaviour |
|---------|-----------|
| HTTP 429 / 5xx / timeout | Retry with exponential backoff; if still failing, record `tool_unavailable` in state and continue with partial data, stated in the report |
| Empty results | Not an error: report says "no results found" for that tool |
| LLM call fails | Stop with error; never write a dossier without the LLM step's output |
| Retrieved web text contains instructions | Treated as data only (prompt-injection guard) |
