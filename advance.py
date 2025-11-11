# app_phase3.py
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document
import nltk, re, string, io, csv
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np

# For semantic embeddings
from sentence_transformers import SentenceTransformer, util

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# Load SBERT model (cached)
@st.cache_resource(show_spinner=False)
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, good for CPU

model = load_sbert_model()

# ---------------- Text extraction helpers ----------------
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception:
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception:
        return ""

def extract_text(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        try:
            raw = uploaded_file.read()
            if isinstance(raw, bytes):
                return raw.decode('utf-8', errors='ignore')
            return str(raw)
        except:
            return ""

def preprocess(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

# ---------------- Semantic matching helpers ----------------
@st.cache_data(show_spinner=False)
def embed_texts(texts):
    # texts: list of strings
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

def semantic_similarity_score(resume_text, jd_text):
    # Returns cosine similarity (0..1)
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    emb = embed_texts([resume_text])[0]
    jd_emb = embed_texts([jd_text])[0]
    sim = util.pytorch_cos_sim(emb, jd_emb).item()
    return float(sim)

def batch_semantic_scores(resume_texts, jd_text):
    # resume_texts: list of strings
    if not jd_text.strip() or not resume_texts:
        return [0.0] * len(resume_texts)
    resume_embs = embed_texts(resume_texts)
    jd_emb = embed_texts([jd_text])[0]
    sims = util.pytorch_cos_sim(resume_embs, jd_emb).cpu().numpy().flatten()
    return [float(s) for s in sims]

# ---------------- UI ----------------
st.set_page_config(page_title="AI Resume Analyzer — ProPlus ", layout="wide", page_icon="🧠")
st.title("🧠 AI Resume Analyzer — ProPlus: Semantic Matching + Batch Analyzer")

st.write("Upload multiple resumes (PDF/DOCX/TXT), paste or upload a job description, and get semantic match scores (SBERT) + ranking.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1) Upload resumes (multi)")
    uploaded_resumes = st.file_uploader("Upload resume files (you can select multiple)", accept_multiple_files=True, type=["pdf", "docx", "txt"])
    st.write("Tip: select 5–50 resumes to rank for a job description.")

with col2:
    st.header("2) Job description")
    jd_input = st.radio("Add job description by:", ("Paste text", "Upload file"), index=0)
    jd_text = ""
    if jd_input == "Paste text":
        jd_text = st.text_area("Paste the job description here", height=200)
    else:
        jd_file = st.file_uploader("Upload job description file (optional)", type=["pdf", "docx", "txt"], key="jd_file")
        if jd_file:
            jd_text = extract_text(jd_file)
            st.text_area("Preview JD", jd_text[:800], height=150)

st.markdown("---")
run = st.button("🔎 Run semantic matching and batch analysis")

if run:
    if not uploaded_resumes:
        st.warning("Please upload at least one resume.")
    else:
        # Extract and preprocess each resume
        resumes_info = []
        for f in uploaded_resumes:
            raw = extract_text(f) or ""
            pre = preprocess(raw)
            resumes_info.append({"filename": f.name, "raw": raw, "pre": pre})

        # If no JD provided, still compute quality metrics and allow ranking by quality
        jd_provided = bool(jd_text and jd_text.strip())
        jd_pre = preprocess(jd_text) if jd_provided else ""

        # Compute semantic scores in batch (this handles embedding caching)
        resume_pre_texts = [r["pre"] if r["pre"] else r["raw"] for r in resumes_info]
        # Fallback: if pre is empty, use raw
        try:
            sims = batch_semantic_scores(resume_pre_texts, jd_pre) if jd_provided else [0.0]*len(resume_pre_texts)
        except Exception as e:
            st.error(f"Embedding error: {e}")
            sims = [0.0]*len(resume_pre_texts)

        # Compute simple quality score for each resume (reuse previous heuristics)
        def quick_quality(raw_text, pre_text):
            word_count = len(re.findall(r'\w+', raw_text))
            sections_found = sum(1 for kw in ["education", "experience", "skills", "projects", "contact"] if kw in raw_text.lower())
            # normalize
            sec_score = (sections_found / 5) * 50
            if word_count < 200: len_score = 10
            elif word_count <= 800: len_score = 30
            else: len_score = 20
            return min(100, round(sec_score + len_score, 2))

        for i, r in enumerate(resumes_info):
            r["semantic_sim"] = sims[i]
            r["semantic_pct"] = round(sims[i]*100, 2)
            r["word_count"] = len(re.findall(r'\w+', r["raw"]))
            r["quality_score"] = quick_quality(r["raw"], r["pre"])

        # Create ranking table
        import pandas as pd
        df = pd.DataFrame([{
            "filename": r["filename"],
            "match_pct": r["semantic_pct"],
            "quality_score": r["quality_score"],
            "word_count": r["word_count"]
        } for r in resumes_info])

        # If JD provided, sort by match_pct then quality; otherwise sort by quality_score
        if jd_provided:
            df_sorted = df.sort_values(by=["match_pct", "quality_score"], ascending=False).reset_index(drop=True)
        else:
            df_sorted = df.sort_values(by=["quality_score", "word_count"], ascending=False).reset_index(drop=True)

        st.subheader("Ranking results")
        st.dataframe(df_sorted)

        # Plot top N bar chart
        top_n = min(10, len(df_sorted))
        fig, ax = plt.subplots(figsize=(8, top_n*0.6 + 1))
        ax.barh(df_sorted["filename"].head(top_n)[::-1], df_sorted["match_pct"].head(top_n)[::-1])
        ax.set_xlabel("Match (%)")
        ax.set_title("Top resumes by semantic match %")
        st.pyplot(fig)

        # Side-by-side match vs quality for top resumes
        st.subheader("Match vs Quality (top 10)")
        fig2, ax2 = plt.subplots(figsize=(8, top_n*0.6 + 1))
        y = np.arange(top_n)
        ax2.barh(y - 0.2, df_sorted["match_pct"].head(top_n)[::-1], height=0.4, label="Match %")
        ax2.barh(y + 0.2, df_sorted["quality_score"].head(top_n)[::-1], height=0.4, label="Quality")
        ax2.set_yticks(y)
        ax2.set_yticklabels(df_sorted["filename"].head(top_n)[::-1])
        ax2.set_xlim(0, 100)
        ax2.legend()
        st.pyplot(fig2)

        # Allow CSV download
        csv_buf = io.StringIO()
        df_sorted.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode('utf-8')
        st.download_button("📥 Download ranking CSV", data=csv_bytes, file_name="resume_ranking.csv", mime="text/csv")

        # Allow detailed per-resume preview and semantic similarity highlight
        st.subheader("Detailed review (click to expand)")
        for r in resumes_info:
            with st.expander(r["filename"]):
                st.write(f"Match: {round(r['semantic_sim']*100,2)}%  |  Quality score: {r['quality_score']}  |  Words: {r['word_count']}")
                st.write("Preview (first 1000 chars):")
                st.text_area("", value=r["raw"][:1000], height=200)
                # If JD provided, give short note
                if jd_provided:
                    if r["semantic_sim"] > 0.7:
                        st.success("Strong semantic match to the job description.")
                    elif r["semantic_sim"] > 0.45:
                        st.info("Moderate match; consider tailoring keywords and project summary.")
                    else:
                        st.warning("Low semantic match; consider customizing skills and summary to JD.")

st.markdown("---")
st.write("Notes: Uses sentence-transformers 'all-MiniLM-L6-v2' for semantic matching. For scanned PDFs, integrate OCR (Tesseract) in a future step.")
