<div align="center">

# 📊 TaxHub Knowledge Assistant

**A grounded, source-cited AI knowledge layer for German tax advisors (Steuerberater)**

Built as a thin, real slice of the *Knowledge / Wiki* block of a vertical AI operating hub —
part of a case study for [CITO GmbH](https://cito.vision).

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen?style=for-the-badge)](#-live-demo)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge)](#)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B?style=for-the-badge)](#)
[![Grounding](https://img.shields.io/badge/grounding%20accuracy-8%2F8-success?style=for-the-badge)](#-grounding-you-can-verify)

</div>

---

## 🚀 Live Demo

**[→ Open the live app](#)** *(replace with your Streamlit Community Cloud URL)*

No installation needed — ask a real question about German tax-advisory fees or professional
duties and see it answered with an exact citation back to the statute paragraph it came from.

---

## What This Is

Most "AI knowledge assistants" are demos: a chatbot wrapped around a system prompt with no
real content behind it. This one is not. It ingests **17 verbatim regulatory sections** from
the two documents every German Steuerberater already knows by heart —

- **Steuerberatungsgesetz (StBerG)** — the professional-duties law
- **Steuerberatervergütungsverordnung (StBVV)** — the statutory fee schedule

— fetched directly from the official German law portal
[gesetze-im-internet.de](https://www.gesetze-im-internet.de), and exposes them through a
retrieval-and-citation pipeline. Every answer traces to a real source. Nothing is hardcoded
or invented.

> **The guardrail this project is built around:** *grounded, not invented.* Any fact or
> number in an answer must trace to a real, cited source — no made-up statistics, no
> hallucinated paragraph numbers.

---

## ✅ Grounding You Can Verify

Rather than asking you to trust that the retrieval works, the app ships with a built-in
**Grounding check** tab: an automated regression test of 8 known questions with known-correct
source mappings. Click "Run grounding check" and it tells you, live, whether the retriever
found the right statute paragraph for each one.

```
Retrieval accuracy: 8 / 8   (100%)
```

This is the same discipline a real Kanzlei would demand before trusting an AI tool with
client-facing answers — verify it, don't just believe it.

---

## Features

| | |
|---|---|
| 💬 **Ask** | Natural-language or keyword questions, answered with inline citations |
| 📚 **Browse sources** | Full knowledge base, filterable by law (StBerG / StBVV), fully auditable |
| ✅ **Grounding check** | One-click regression test proving retrieval accuracy |
| ℹ️ **About** | Methodology, scope, and what a production version would add |
| 🧠 **Optional LLM synthesis** | Set `ANTHROPIC_API_KEY` to have Claude write a fluent answer, still strictly constrained to retrieved sources |
| ⚙️ **Offline-capable by default** | Hybrid TF-IDF + lexical retrieval — no embeddings API required to run |

---

## How It Works

```
Question
   │
   ▼
Hybrid retriever (TF-IDF cosine similarity + lexical token overlap)
   │
   ▼
Top-k cited passages from knowledge_base/*.txt
   │
   ├──► No ANTHROPIC_API_KEY  → show raw cited passages (default)
   └──► ANTHROPIC_API_KEY set → Claude writes a fluent answer, [n]-cited to the same passages
```

Each source document carries its own title, official source URL, and topical keywords —
so every answer the app gives can be traced, in one click, back to the exact statute text
it came from.

---

## Quickstart

```bash
git clone https://github.com/<your-username>/taxhub-mvp.git
cd taxhub-mvp
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

### Optional: enable LLM-synthesized answers

```bash
export ANTHROPIC_API_KEY="sk-..."   # macOS/Linux
setx ANTHROPIC_API_KEY "sk-..."     # Windows
```

Without this variable set, the app runs fully offline in pure retrieval mode — still cited,
just without the fluent write-up step.

---

## Deploying Your Own Copy

1. Fork or clone this repo.
2. Push it to your own GitHub account (must be public for the free tier).
3. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → point at your repo,
   main file `app.py`.
4. (Optional) Add `ANTHROPIC_API_KEY` under **Secrets** in the app settings.

Deployment takes about 10 minutes end-to-end.

---

## Repository Structure

```
taxhub-mvp/
├── app.py                    # Streamlit app: retrieval, UI, grounding eval
├── requirements.txt          # streamlit, scikit-learn, numpy, anthropic
├── README.md                 # you are here
└── knowledge_base/           # 17 real regulatory source documents
    ├── StBerG_1_Anwendungsbereich.txt
    ├── StBerG_3_BefugnisUnbeschraenkteHilfe.txt
    ├── StBVV_21_RatAuskunftErstberatung.txt
    ├── StBVV_24_Steuererklaerungen.txt
    └── ... (13 more, spanning duties, fees, and firm structure)
```

---

## Why This Scope, Not a Bigger Demo

This is deliberately narrow. A production **TaxHub** would ingest a specific Kanzlei's own
engagement letters, past client correspondence, and internal templates — not just public
statute text. But StBerG and StBVV are the right foundation for an MVP because:

1. They're **real, official, and verifiable** — anyone at CITO with domain knowledge can
   check that every cited passage is accurate, in seconds.
2. They **prove the retrieval-and-citation pattern works** before pointing it at
   confidential firm data.
3. They map directly onto real willingness-to-pay: the fee schedule in StBVV
   (e.g. a statutory **€190 cap** on a first consultation, StBVV §21) is exactly the
   economics a paid product would need to beat.

## What's Next

- Ingest a real Kanzlei's document set (fee agreements, past correspondence, DATEV exports).
- Swap TF-IDF for embeddings once the knowledge base grows past a few hundred documents.
- Wire this Knowledge layer behind a Communication Hub (AI phone/email intake), reusing the
  same grounded corpus to answer or route incoming client questions.
- Grow the grounding eval set alongside the knowledge base — treat it as a CI gate before
  shipping any retrieval change.

---

<div align="center">

Built for the CITO GmbH TaxHub case study · August 2026

</div>
