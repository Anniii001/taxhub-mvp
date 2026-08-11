"""
TaxHub Knowledge Assistant — MVP v2.2
A source-cited, hybrid-retrieval Q&A layer over real German tax-advisory
regulation (Steuerberatungsgesetz / StBerG and Steuerberatervergütungsverordnung / StBVV).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy live (free, ~10 min): push this folder to a public GitHub repo, then
deploy on https://share.streamlit.io pointing at app.py. Optionally set
ANTHROPIC_API_KEY as a secret to enable synthesized (still strictly cited) answers.
"""

import os
import re
import glob
import time
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Knowledge base loading + hybrid retrieval (TF-IDF cosine + lexical overlap)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    chunks, sources = [], []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        lines = text.split("\n")
        title = lines[0].strip()
        source_line = next((l for l in lines if l.startswith("Source:")), "")
        keyword_line = next((l for l in lines if l.startswith("Keywords:")), "")
        source_url = source_line.replace("Source:", "").strip()
        keywords = keyword_line.replace("Keywords:", "").strip()
        idx_kw = lines.index(keyword_line) if keyword_line in lines else 1
        body = "\n".join(lines[idx_kw + 1:]).strip()
        for para in [p.strip() for p in body.split("\n\n") if p.strip()]:
            display = f"{title}\n{para}"
            index_text = f"{title} {keywords} {para}"
            chunks.append(display)
            sources.append({
                "title": title, "url": source_url, "keywords": keywords,
                "index_text": index_text, "file": os.path.basename(path),
                "law": "StBVV" if "StBVV" in path or "StBVV" in title else "StBerG",
            })
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([s["index_text"] for s in sources])
    return chunks, sources, vectorizer, matrix


def _lexical_overlap_score(query, index_text):
    q_tokens = set(re.findall(r"[a-zA-ZäöüÄÖÜß]{3,}", query.lower()))
    d_tokens = set(re.findall(r"[a-zA-ZäöüÄÖÜß]{3,}", index_text.lower()))
    if not q_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)


def retrieve(query, chunks, sources, vectorizer, matrix, top_k=4, thresh=0.02):
    """Hybrid score = weighted blend of TF-IDF cosine similarity and raw lexical
    token overlap, so both semantically-loaded and keyword-style queries work."""
    q_vec = vectorizer.transform([query])
    cos_sims = cosine_similarity(q_vec, matrix).flatten()
    lex_sims = np.array([_lexical_overlap_score(query, s["index_text"]) for s in sources])
    hybrid = 0.7 * cos_sims + 0.3 * lex_sims

    top_idx = np.argsort(hybrid)[::-1][:top_k]
    results = []
    for i in top_idx:
        if hybrid[i] > thresh:
            results.append({
                "text": chunks[i], "source": sources[i],
                "score": float(hybrid[i]), "cos": float(cos_sims[i]), "lex": float(lex_sims[i]),
            })
    return results


def synthesize_with_claude(query, results):
    import anthropic
    client = anthropic.Anthropic()
    context = "\n\n---\n\n".join(
        f"[{i+1}] {r['source']['title']} ({r['source']['url']})\n{r['text']}"
        for i, r in enumerate(results)
    )
    prompt = f"""You are a knowledge assistant for a German tax advisory firm (Steuerberater).
Answer the staff/client question ONLY using the numbered sources below. Cite sources
inline like [1], [2]. If the sources do not contain the answer, say so explicitly —
never invent a fact, paragraph number, or fee.

Sources:
{context}

Question: {query}

Answer (in English, with inline [n] citations):"""
    msg = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ---------------------------------------------------------------------------
# Built-in grounding evaluation harness — proves the retriever isn't guessing
# ---------------------------------------------------------------------------

EVAL_SET = [
    {"q": "What is the fee cap for a first consultation with a consumer client?",
     "expect_file": "StBVV_21_RatAuskunftErstberatung.txt"},
    {"q": "What is the hourly time-based fee range under the StBVV?",
     "expect_file": "StBVV_13_Zeitgebuehr.txt"},
    {"q": "Are Steuerberater employees bound to confidentiality?",
     "expect_file": "StBerG_62_VerschwiegenheitspflichtBeschaeftigter.txt"},
    {"q": "What is the fee range for a corporate tax return?",
     "expect_file": "StBVV_24_Steuererklaerungen.txt"},
    {"q": "Can a Steuerberater work as an employee elsewhere?",
     "expect_file": "StBerG_57_AllgemeineBerufspflichten.txt"},
    {"q": "Who is authorized to give tax advice in Germany?",
     "expect_file": "StBerG_3_BefugnisUnbeschraenkteHilfe.txt"},
    {"q": "How much does a Steuerberater charge for a tax audit attendance?",
     "expect_file": "StBVV_29_TeilnahmeAnPruefungen.txt"},
    {"q": "Can a Steuerberater bill the same matter twice?",
     "expect_file": "StBVV_12_Abgeltungsbereich.txt"},
]


def run_eval(chunks, sources, vectorizer, matrix):
    hits = 0
    rows = []
    for case in EVAL_SET:
        results = retrieve(case["q"], chunks, sources, vectorizer, matrix, top_k=1, thresh=0.0)
        top_file = results[0]["source"]["file"] if results else None
        ok = top_file == case["expect_file"]
        hits += int(ok)
        rows.append({"question": case["q"], "expected": case["expect_file"],
                     "retrieved": top_file, "correct": "✅" if ok else "❌"})
    return hits, len(EVAL_SET), rows


# ---------------------------------------------------------------------------
# Page config + global styling
# ---------------------------------------------------------------------------

st.set_page_config(page_title="TaxHub Knowledge Assistant", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 12px;
       background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
       border: 1px solid #334155; margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 0.3rem 0; font-size: 1.9rem;}
.hero p {margin: 0; color: #94a3b8; font-size: 0.95rem;}
.source-card {background-color: rgba(37, 99, 235, 0.08); border-left: 4px solid #2563eb;
              padding: 0.7rem 1rem; border-radius: 6px; margin-bottom: 0.6rem;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {padding: 0.5rem 1.1rem; border-radius: 8px 8px 0 0;}
section[data-testid="stSidebar"] {background-color: #0f172a;}
</style>
""", unsafe_allow_html=True)

chunks, sources, vectorizer, matrix = load_knowledge_base()
n_docs = len(set(s["file"] for s in sources))

# ---------------------------------------------------------------------------
# Hero header with credibility metrics (instead of a bare title)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <h1>📊 TaxHub Knowledge Assistant</h1>
      <p>Grounded, source-cited Q&A over German tax-advisory regulation (StBerG &amp; StBVV) —
      a thin, real slice of the Knowledge/Wiki block for a Steuerberater-focused operating hub.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Source documents", n_docs, help="Real statute sections, fetched verbatim from gesetze-im-internet.de")
m2.metric("Indexed passages", len(chunks))
m3.metric("Laws covered", "2", help="StBerG (professional duties) + StBVV (statutory fee schedule)")
m4.metric("Grounding accuracy", "—", help="Run the check in the 'Grounding check' tab to see the live score")

# ---------------------------------------------------------------------------
# Sidebar: explains every control instead of leaving bare widgets on the page
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Retrieval settings")
    st.caption(
        "Controls how the assistant searches the knowledge base below. These settings apply "
        "to every question you ask in the **Ask** tab."
    )
    top_k = st.slider(
        "Sources per answer",
        min_value=1, max_value=6, value=4,
        help="How many cited regulatory passages the assistant retrieves and shows for each "
             "question. More sources give broader context; fewer keep answers tighter and "
             "more focused on the single best match.",
    )
    st.caption(f"Currently retrieving up to **{top_k}** source passage(s) per question.")

    st.divider()
    st.markdown("### 🧠 Answer mode")
    if USE_LLM:
        st.success("LLM synthesis **enabled** — Claude writes a fluent, source-constrained answer.", icon="✅")
    else:
        st.info(
            "**Offline retrieval mode.** No ANTHROPIC_API_KEY is set, so the assistant shows the "
            "exact regulatory passages it retrieves rather than a generated sentence. Every "
            "fact still traces to a real, cited source — this mode simply skips the fluent "
            "write-up step.",
            icon="ℹ️",
        )

    st.divider()
    st.markdown("### 📎 About this MVP")
    st.caption(
        "Built for the CITO TaxHub case study. Knowledge base: 17 real sections of the German "
        "Steuerberatungsgesetz and Steuerberatervergütungsverordnung. See the **About** tab for "
        "the full methodology."
    )

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_chat, tab_sources, tab_eval, tab_about = st.tabs(
    ["💬 Ask", "📚 Browse sources", "✅ Grounding check", "ℹ️ About"]
)

with tab_chat:
    st.subheader("Ask a question")
    st.caption(
        "Try an example below, or type your own. Answers are grounded strictly in the "
        "knowledge base — if nothing relevant is found, the assistant says so instead of guessing."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    examples = [q["q"] for q in EVAL_SET[:5]]
    choice = st.selectbox("Example questions", ["—"] + examples)

    # Widget key depends on `choice` so the text box re-initializes with the
    # new default value whenever a different example is picked (a fixed key
    # would make Streamlit ignore `value` after the first render).
    query = st.text_input(
        "Or type your own question about tax-advisory regulation or fees:",
        value="" if choice == "—" else choice,
        key=f"query_input_{choice}",
        placeholder="e.g. What is the monthly fee range for bookkeeping under the StBVV?",
    )

    ask = st.button("Ask", type="primary", use_container_width=False)

    if ask and query.strip():
        t0 = time.time()
        with st.spinner("Retrieving grounded sources..."):
            results = retrieve(query, chunks, sources, vectorizer, matrix, top_k=top_k)
        elapsed = time.time() - t0

        if not results:
            st.warning("No sufficiently relevant passage found in the knowledge base for this question.")
        else:
            if USE_LLM:
                try:
                    answer = synthesize_with_claude(query, results)
                    st.markdown("#### Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"LLM synthesis failed ({e}); showing retrieved passages instead.")

            st.markdown(f"#### Sources  ·  retrieved in {elapsed*1000:.0f} ms")
            for i, r in enumerate(results):
                law_badge = "🟦 StBVV" if r["source"]["law"] == "StBVV" else "🟩 StBerG"
                with st.expander(
                    f"[{i+1}] {law_badge} · {r['source']['title']}  ·  relevance {r['score']:.2f}",
                    expanded=(i == 0),
                ):
                    st.markdown(f'<div class="source-card">{r["text"]}</div>', unsafe_allow_html=True)
                    st.caption(f"cosine similarity = {r['cos']:.2f}  ·  lexical overlap = {r['lex']:.2f}")
                    st.markdown(f"[View original source ↗]({r['source']['url']})")

            st.session_state.history.append({"q": query, "n_sources": len(results)})

    if st.session_state.history:
        st.divider()
        st.caption(f"Session: {len(st.session_state.history)} question(s) asked, all grounded.")

with tab_sources:
    st.subheader("Full knowledge base")
    st.caption(f"{n_docs} source documents, {len(chunks)} indexed passages — every one traceable to an official source.")
    law_filter = st.radio("Filter by law", ["All", "StBerG", "StBVV"], horizontal=True)
    seen_files = set()
    for chunk, src in zip(chunks, sources):
        if src["file"] in seen_files:
            continue
        if law_filter != "All" and src["law"] != law_filter:
            continue
        seen_files.add(src["file"])
        with st.expander(f"{src['title']}"):
            st.write(chunk.split("\n", 1)[1] if "\n" in chunk else chunk)
            st.markdown(f"[Official source ↗]({src['url']})")

with tab_eval:
    st.subheader("Grounding self-check")
    st.caption(
        "An automated regression test: for each question, does the retriever surface the "
        "exact statute paragraph it should? This is how we — and any Kanzlei evaluating this — "
        "verify the assistant isn't guessing."
    )
    if st.button("▶ Run grounding check", type="primary"):
        hits, total, rows = run_eval(chunks, sources, vectorizer, matrix)
        pct = 100 * hits / total
        c1, c2 = st.columns(2)
        c1.metric("Retrieval accuracy", f"{hits}/{total}")
        c2.metric("Percentage correct", f"{pct:.0f}%")
        if pct == 100:
            st.success("All grounding checks passed — every answer traced to its correct source.", icon="✅")
        st.table(rows)

with tab_about:
    st.subheader("Methodology")
    st.markdown("""
    **Knowledge base**: 17 real regulatory sections from the *Steuerberatungsgesetz* (StBerG)
    and *Steuerberatervergütungsverordnung* (StBVV), fetched verbatim from the official German
    law portal [gesetze-im-internet.de](https://www.gesetze-im-internet.de). Every answer traces
    to an exact source document — nothing here is invented or hardcoded.

    **Retrieval**: a hybrid of TF-IDF cosine similarity and raw lexical token overlap, so both
    natural-language and keyword-style questions work well without needing an embeddings API
    (keeping this MVP fully offline-capable).

    **Why this scope**: the case calls for something "grounded in real content, with sources,
    not a hardcoded demo." StBerG and StBVV are the two documents every German Steuerberater
    already knows by heart — which makes it easy for a domain expert to verify this assistant
    isn't inventing paragraphs or fees. A production TaxHub would ingest a specific Kanzlei's own
    engagement letters, past cases, and internal templates on top of this statutory backbone.

    **Optional LLM layer**: set `ANTHROPIC_API_KEY` to let Claude write a fluent, still strictly
    source-constrained answer instead of raw passages.
    """)
