# Stack decision — DRAFT (9 Nov 2026)

**Industry / use case:** independent bookshops — editorial research dossiers.
**Primary stack:** ☑ LangGraph (Python) ☐ n8n

## Why LangGraph (draft, 5 sentences)

Our agent must make decisions that depend on intermediate results: if a news event cannot be verified, if a publisher page contradicts Google Books, or if a tool fails, the flow must take a different path. LangGraph gives us explicit, inspectable state (query, bookshop profile, candidates, sources, errors) and conditional edges for those branches, written and tested in Python. The core value of the product is validation logic — work vs edition, ISBN checks, one source URL per claim — which is easier to unit-test in Python code than in visual nodes. n8n would be faster to wire integrations and send the report to Slack or Gmail, but our MVP saves reports to the repository, so that advantage matters less. We keep n8n out of scope for the MVP; a thin n8n trigger could be a post-MVP option.

## Trade-offs we accept
- Slower to build than an n8n workflow; more code to maintain.
- No visual workflow out of the box: we must document the graph (diagram in README).
- Integrations (HTTP calls, retries) are written by us, not provided as nodes.

## To confirm with evidence this week
- ReAct-style tool loop: where it fits (Day 3).
- Grounding alternative to a vector DB (R10) — argued in README.
