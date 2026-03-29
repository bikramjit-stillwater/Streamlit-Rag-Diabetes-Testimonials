import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Diabetes Testimonial Chatbot", layout="wide")

# -----------------------------
# Gemini setup
# -----------------------------
genai.configure(api_key="AIzaSyCWI5G0f4l75e6_CT_0D8IO24-uxTraR_I")
model = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# Load and prepare data
# -----------------------------
@st.cache_resource
def load_rag_system():
    csv_path = "diabetes_testimonials_only.csv"

    df = pd.read_csv(csv_path)
    df = df[["title", "url", "transcript"]].copy()

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["transcript"] = df["transcript"].fillna("").astype(str).str.strip()

    df = df[df["transcript"] != ""].reset_index(drop=True)

    all_docs = []
    for i, row in df.iterrows():
        block = f"""TESTIMONIAL_ID: {i}
TITLE: {row['title']}
URL: {row['url']}
TRANSCRIPT:
{row['transcript']}

{'='*100}
"""
        all_docs.append(block)

    with open("all_patient_testimonials.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_docs))

    documents = []
    for i, row in df.iterrows():
        doc_text = f"""TITLE: {row['title']}
URL: {row['url']}
TRANSCRIPT:
{row['transcript']}"""

        documents.append({
            "doc_id": i,
            "title": row["title"],
            "url": row["url"],
            "text": doc_text
        })

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    doc_texts = [d["text"] for d in documents]
    doc_embeddings = embed_model.encode(
        doc_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(doc_embeddings.astype("float32"))

    return df, documents, embed_model, index


df, documents, embed_model, index = load_rag_system()

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, top_k=3):
    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = documents[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results

# -----------------------------
# RAG
# -----------------------------
def ask_rag(query, top_k=3):
    results = retrieve(query, top_k=top_k)

    context = "\n\n".join([
        f"""SOURCE {i+1}
TITLE: {r['title']}
URL: {r['url']}
CONTENT:
{r['text']}"""
        for i, r in enumerate(results)
    ])

    prompt = f"""
You are a testimonial-based assistant.

Rules:
1. Answer only from the provided testimonial context.
2. If the answer is not clearly present, say: "Not found clearly in the testimonials."
3. Do not give medical advice.
4. Mention relevant source title and URL.
5. Keep the answer clear and short.

User question:
{query}

Context:
{context}
"""

    response = model.generate_content(prompt)

    return {
        "query": query,
        "answer": response.text,
        "sources": [
            {"title": r["title"], "url": r["url"], "score": r["score"]}
            for r in results
        ]
    }

# -----------------------------
# UI
# -----------------------------
st.title("Diabetes Testimonial Chatbot")
st.write("Ask questions from patient testimonial data.")

preset_questions = [
    "Find testimonials where people reduced diabetes medicine after switching to plant-based diet",
    "Find patient stories that talk about plant-based diet helping diabetes",
    "Find testimonials where patients describe how long they had diabetes"
]

st.subheader("Try sample questions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("1. Reduced medicine after plant-based diet"):
        st.session_state["selected_query"] = preset_questions[0]

with col2:
    if st.button("2. Plant-based diet helping diabetes"):
        st.session_state["selected_query"] = preset_questions[1]

with col3:
    if st.button("3. Duration of diabetes in testimonials"):
        st.session_state["selected_query"] = preset_questions[2]

default_query = st.session_state.get("selected_query", "")
query = st.text_input("Type your question", value=default_query)

if st.button("Ask") and query.strip():
    with st.spinner("Searching testimonials and generating answer..."):
        result = ask_rag(query.strip(), top_k=3)

    st.markdown("## Answer")
    st.write(result["answer"])

    st.markdown("## Sources")
    for i, src in enumerate(result["sources"], start=1):
        st.markdown(
            f"**{i}. {src['title']}**  \n"
            f"URL: {src['url']}  \n"
            f"Score: {round(src['score'], 4)}"
        )
