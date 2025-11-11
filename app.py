# app.py
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document
import nltk
from nltk.corpus import stopwords
import string
import re
import matplotlib.pyplot as plt
import io

# Ensure required NLTK data is present
nltk.download('stopwords', quiet=True)

# -------------------
# Configuration
# -------------------
SKILLS_DICT = {
    "Programming": ["python", "java", "c++", "r", "sql", "javascript"],
    "Machine Learning": ["machine learning", "supervised", "unsupervised", "classification", "regression"],
    "Deep Learning": ["tensorflow", "keras", "cnn", "rnn", "lstm", "neural network", "pytorch"],
    "Data Science": ["pandas", "numpy", "matplotlib", "seaborn", "data analysis", "feature engineering"],
    "NLP": ["nlp", "text", "bert", "transformer", "tokenization", "word2vec", "nlp pipeline"],
    "Tools": ["git", "linux", "jupyter", "colab", "streamlit", "docker"]
}

SECTION_KEYWORDS = {
    "education": ["education", "degree", "bachelor", "master", "graduat", "university", "college"],
    "experience": ["experience", "intern", "worked", "role", "responsibilit", "employment"],
    "skills": ["skills", "technical skills", "proficient", "familiar", "competent"],
    "projects": ["project", "projects", "portfolio", "github", "implemented"],
    "contact": ["email", "phone", "contact", "linkedin", "address"]
}

STOP_WORDS = set(stopwords.words('english'))

# -------------------
# Helpers: Text extraction
# -------------------
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return ""

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
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
        # assume text file
        try:
            raw = uploaded_file.read()
            if isinstance(raw, bytes):
                return raw.decode('utf-8', errors='ignore')
            return str(raw)
        except:
            return ""

# -------------------
# Helpers: Preprocess & keyword functions
# -------------------
def preprocess(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)

def detect_sections(text):
    text_low = text.lower()
    present = {}
    for sec, keywords in SECTION_KEYWORDS.items():
        present[sec] = any(k in text_low for k in keywords)
    return present

def skill_strength_analysis(preprocessed_text):
    results = {}
    for category, keywords in SKILLS_DICT.items():
        # count keywords that appear (match exact keyword or simple phrase)
        count = 0
        for kw in keywords:
            if kw in preprocessed_text:
                count += 1
        strength = (count / len(keywords)) * 100
        results[category] = round(strength, 2)
    return results

def resume_quality_score(sections_present, word_count, skill_strengths):
    # sections: reward for having each section
    sec_score = sum(1 for v in sections_present.values() if v) / max(1, len(sections_present)) * 50  # up to 50
    # length score: ideal between 300 and 800 words
    if word_count < 200:
        len_score = 10
    elif word_count <= 800:
        len_score = 30
    else:
        len_score = 20
    # skill coverage average
    avg_skill = sum(skill_strengths.values()) / max(1, len(skill_strengths))
    skill_score = (avg_skill / 100) * 20  # up to 20
    total = sec_score + len_score + skill_score
    total = min(100, round(total, 2))
    return total

# -------------------
# Helpers: Matching & missing keywords
# -------------------
def compute_tfidf_similarity(resume_text, jd_text):
    # if either empty, return 0.0
    if not resume_text.strip() or not jd_text.strip():
        return 0.0, None
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    try:
        tfidf = vec.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return sim, (vec, tfidf)
    except Exception:
        return 0.0, None

def top_jd_keywords(jd_text, vec_and_tfidf, top_n=15):
    # Returns JD's top weighted tokens (by tfidf)
    if vec_and_tfidf is None:
        return []
    vec, tfidf = vec_and_tfidf
    jd_vec = tfidf[1].toarray().flatten()
    feature_names = vec.get_feature_names_out()
    idx_sorted = jd_vec.argsort()[::-1]
    top = []
    for idx in idx_sorted:
        if jd_vec[idx] <= 0:
            continue
        top.append(feature_names[idx])
        if len(top) >= top_n:
            break
    return top

# -------------------
# UI
# -------------------
st.set_page_config(page_title="AI Resume Analyzer Pro", page_icon="🤖", layout="wide")
st.title("🤖 AI Resume Analyzer Pro")
st.write("Upload your resume and a job description (or paste it) to get a detailed analysis and actionable suggestions.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1) Upload Resume")
    uploaded_resume = st.file_uploader("Upload resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
    if uploaded_resume:
        raw_resume = extract_text(uploaded_resume)
        if not raw_resume:
            st.error("Couldn't extract text from this file. Try a different resume file or a .txt file.")
        else:
            st.success("Resume loaded.")
            # Show small preview (first 500 chars)
            st.text_area("Resume preview (first 800 chars):", raw_resume[:800], height=150)

with col2:
    st.header("2) Job Description")
    jd_input_method = st.radio("Add job description by:", ("Paste text", "Upload file (TXT/PDF/DOCX)"), index=0)
    jd_text = ""
    if jd_input_method == "Paste text":
        jd_text = st.text_area("Paste the job description here:", height=180)
    else:
        uploaded_jd = st.file_uploader("Upload job description (optional)", type=["pdf", "docx", "txt"], key="jd_uploader")
        if uploaded_jd:
            jd_text = extract_text(uploaded_jd)
            st.text_area("Job description preview:", jd_text[:800], height=150)

st.markdown("---")
run_button = st.button("🔍 Analyze Resume & Match")

if run_button:
    # Basic checks
    if not uploaded_resume:
        st.warning("Please upload a resume file to analyze.")
    else:
        raw_resume = extract_text(uploaded_resume) or ""
        resume_text = preprocess(raw_resume)
        jd_text_raw = jd_text or ""
        jd_text_pre = preprocess(jd_text_raw)

        # Section detection
        sections = detect_sections(raw_resume)
        st.subheader("✔ Sections detected")
        cols = st.columns(len(sections))
        for i, (sec, present) in enumerate(sections.items()):
            with cols[i]:
                if present:
                    st.metric(label=sec.capitalize(), value="Present")
                else:
                    st.metric(label=sec.capitalize(), value="Not found")

        # Word count
        word_count = len(re.findall(r'\w+', raw_resume))
        st.write(f"*Word count (approx):* {word_count}")

        # Skill strengths
        skill_strengths = skill_strength_analysis(resume_text)
        st.subheader("💪 Skill Category Strengths")
        # plot horizontal bar
        fig1, ax1 = plt.subplots()
        categories = list(skill_strengths.keys())
        scores = list(skill_strengths.values())
        ax1.barh(categories, scores)
        ax1.set_xlabel("Strength (%)")
        ax1.set_xlim(0, 100)
        ax1.set_title("Skill coverage by category")
        st.pyplot(fig1)

        # Resume quality score
        quality = resume_quality_score(sections, word_count, skill_strengths)
        st.metric(label="Resume Quality Score (0-100)", value=f"{quality}")

        # Compute similarity
        sim, vec_and_tfidf = compute_tfidf_similarity(resume_text, jd_text_pre)
        st.subheader("📈 Resume ↔ Job Description Matching")
        if jd_text_raw.strip():
            st.write(f"*Match score (cosine similarity):* {sim*100:.2f}%")
            # missing keywords based on JD top tokens
            jd_top_tokens = top_jd_keywords(jd_text_pre, vec_and_tfidf, top_n=20)
            resume_tokens = set(resume_text.split())
            missing = [t for t in jd_top_tokens if t not in resume_tokens]
            if missing:
                st.write("*Top missing / underrepresented keywords (from job description):*")
                st.write(", ".join(missing[:20]))
            else:
                st.success("Your resume includes most of the important keywords from the JD.")
            # Chart: match vs avg skill coverage
            avg_skill_cov = sum(scores)/len(scores)
            fig2, ax2 = plt.subplots()
            ax2.bar(["Match (%)", "Avg Skill Coverage (%)"], [sim*100, avg_skill_cov])
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("Percentage")
            ax2.set_title("Match vs Skill Coverage")
            st.pyplot(fig2)
        else:
            st.info("No job description provided — only resume-level analysis shown.")

        # Suggestions
        st.subheader("🛠 Suggestions & Improvements")
        suggestions = []
        # section suggestions
        for sec, present in sections.items():
            if not present:
                suggestions.append(f"Add a '{sec.capitalize()}' section to your resume.")
        # length suggestion
        if word_count < 300:
            suggestions.append("Resume is short — add concise project details, responsibilities, and technologies used.")
        elif word_count > 1200:
            suggestions.append("Resume is long — try to keep it concise (1-2 pages).")
        # skills suggestion
        weak_skills = [cat for cat, val in skill_strengths.items() if val < 50]
        if weak_skills:
            suggestions.append("Consider adding more details or projects that demonstrate: " + ", ".join(weak_skills))
        # JD suggestions
        if jd_text_raw.strip():
            if sim < 0.4:
                suggestions.append("Align your resume keywords more closely to the job description (add top JD keywords).")
            elif sim < 0.7:
                suggestions.append("Good match — consider tailoring a short 'Summary' at top to highlight key JD skills.")
            else:
                suggestions.append("Strong match! You can further strengthen by adding measurable results in projects/experience.")

        for s in suggestions:
            st.write("- " + s)

        # ---------------------------
        # Downloadable report
        # ---------------------------
        report_lines = []
        report_lines.append("AI Resume Analyzer Report\n")
        report_lines.append(f"Resume Quality Score: {quality}/100\n")
        report_lines.append(f"Word count (approx): {word_count}\n")
        report_lines.append("Sections detected:\n")
        for sec, present in sections.items():
            report_lines.append(f" - {sec.capitalize()}: {'Present' if present else 'Missing'}\n")
        report_lines.append("\nSkill category strengths:\n")
        for cat, val in skill_strengths.items():
            report_lines.append(f" - {cat}: {val}%\n")
        if jd_text_raw.strip():
            report_lines.append(f"\nMatch score vs job description: {sim*100:.2f}%\n")
            if missing:
                report_lines.append("Top missing JD keywords:\n")
                report_lines.append(", ".join(missing[:20]) + "\n")
        report_lines.append("\nSuggestions:\n")
        for s in suggestions:
            report_lines.append(" - " + s + "\n")

        report_text = "\n".join(report_lines)

        st.download_button(
            label="📥 Download analysis report (TXT)",
            data=report_text,
            file_name="resume_analysis_report.txt",
            mime="text/plain"
        )

st.markdown("---")
st.write("Made with ❤ — AI Resume Analyzer Pro.")
