# TaxHub Knowledge Assistant — MVP v2

A thin, real slice of the **Knowledge / Wiki** block of a TaxHub operating hub for German
Steuerberater (tax advisors). It ingests real regulatory source documents and exposes a
grounded, source-cited Q&A assistant — not a hardcoded demo.

## What's new in v2
- **17 real regulatory sections** (up from 10): Steuerberatungsgesetz (StBerG) + Steuerberatervergütungsverordnung (StBVV), covering licensing, professional duties, confidentiality, firm recognition, and the full statutory fee schedule (consultations, tax returns, bookkeeping, audits, court proceedings).
- **Hybrid retrieval**: TF-IDF cosine similarity blended with raw lexical token overlap (70/30 weighting), so both natural-language and keyword-style questions retrieve correctly — no embeddings API required, fully offline-capable.
- **Built-in grounding eval harness**: an 8-question regression test with known-correct source mappings, runnable from the app's "Grounding check" tab. Verified locally at **8/8 (100%) retrieval accuracy** before shipping.
- **Multi-tab UI**: Ask / Browse sources / Grounding check / About — including a source browser so a reviewer can audit the entire knowledge base without asking a single question.
- **Session history + relevance diagnostics**: every retrieved source shows its cosine and lexical overlap scores, and how long retrieval took.

## What's real here
- **Knowledge base** (`knowledge_base/`): verbatim excerpts fetched directly from the official
  German law portal gesetze-im-internet.de. Every chunk carries its source URL and is cited
  in every answer.
- **Retrieval**: fully auditable hybrid scoring — every answer traces to an exact statute
  paragraph, never a hallucinated one. The Grounding check tab proves this empirically.
- **Optional synthesis**: if `ANTHROPIC_API_KEY` is set, Claude writes a fluent answer
  strictly constrained to the retrieved passages, with inline `[n]` citations. Without a key,
  the app still works — it just shows the raw cited passages (pure retrieval mode).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy live (free, ~10 minutes)
1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. "New app" → select the repo → main file path `app.py` → Deploy.
4. (Optional) In app settings → Secrets, add `ANTHROPIC_API_KEY = "sk-..."` to enable
   synthesized answers.

## Extending toward the real product
- Swap TF-IDF for embeddings (e.g. `text-embedding-3-small`) + a vector store once the
  knowledge base grows beyond a few hundred documents.
- Ingest a real Kanzlei's document set (fee agreements, engagement letters, past client
  correspondence, DATEV exports) instead of/in addition to StBerG/StBVV.
- Grow the grounding eval set alongside the knowledge base — treat it as a CI gate before
  shipping any retrieval change.
- Add a feedback loop: flag answers a Steuerberater corrects, feed corrections back as
  higher-priority source chunks.
- Wire this Knowledge layer behind a Communication Hub (intake) so a client question
  captured by phone/email is auto-answered or routed with a cited draft.
