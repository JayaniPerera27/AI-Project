import json
import os
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.analysis import analyze
from src.documents import extract_document_text
from src.scoring import model_status


load_dotenv()

st.set_page_config(
    page_title="ResumeFit AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #15231d;
        --muted: #66756e;
        --line: #dce4df;
        --surface: #ffffff;
        --soft: #f4f7f5;
        --green: #176b4d;
        --green-soft: #e8f3ed;
        --amber: #a86108;
        --red: #a33a3a;
    }
    .stApp { background: #f7f9f8; color: var(--ink); }
    .block-container { max-width: 1320px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: 0 !important; color: var(--ink); }
    h1 { font-size: 2.2rem !important; font-weight: 720 !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1rem !important; }
    [data-testid="stSidebar"] { background: #12251d; }
    [data-testid="stSidebar"] * { color: #eef5f1; }
    [data-testid="stSidebar"] hr { border-color: #355047; }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 1rem 1.1rem;
        min-height: 116px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--soft);
        border: 1px dashed #9aaba2;
        border-radius: 7px;
    }
    .app-kicker {
        color: var(--green); font-size: .78rem; font-weight: 700;
        text-transform: uppercase; margin-bottom: .35rem;
    }
    .app-subtitle { color: var(--muted); margin-top: -.6rem; margin-bottom: 1.4rem; }
    .status-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: .42rem 0; border-bottom: 1px solid #355047; font-size: .88rem;
    }
    .status-on { color: #8fd4b5; font-weight: 700; }
    .status-off { color: #f0b2a9; font-weight: 700; }
    .section-label {
        color: var(--muted); font-size: .77rem; font-weight: 700;
        text-transform: uppercase; margin: .5rem 0 .75rem;
    }
    .skill-wrap { display: flex; flex-wrap: wrap; gap: .42rem; margin-bottom: 1rem; }
    .skill {
        border-radius: 4px; padding: .28rem .5rem; font-size: .8rem;
        border: 1px solid var(--line); background: white;
    }
    .skill.match { border-color: #9bcbb5; background: var(--green-soft); color: #15553f; }
    .skill.missing { border-color: #e5c1a1; background: #fff4e8; color: #854b0b; }
    .empty-state {
        color: var(--muted); border: 1px dashed var(--line); padding: .8rem;
        border-radius: 6px; font-size: .86rem;
    }
    .report-block {
        background: white; border-left: 3px solid var(--green);
        padding: .8rem 1rem; margin-bottom: .8rem;
    }
    .report-block strong { display: block; color: var(--ink); margin-bottom: .2rem; }
    .report-block span { color: #52615a; font-size: .93rem; }
    .risk-note {
        background: #fff7e9; border: 1px solid #eed9ad; border-radius: 6px;
        padding: .75rem .9rem; color: #694811; font-size: .82rem;
    }
    div.stButton > button, div.stDownloadButton > button { border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def skill_markup(skills, kind):
    if not skills:
        return '<div class="empty-state">No skills detected in this group.</div>'
    tags = "".join(f'<span class="skill {kind}">{escape(skill)}</span>' for skill in skills)
    return f'<div class="skill-wrap">{tags}</div>'


def score_chart(scores):
    labels = ["Overall", "Skill coverage", "TF-IDF"]
    values = [
        scores["overall"] * 100,
        st.session_state.analysis["skills"]["skill_coverage"] * 100,
        scores["tfidf"] * 100,
    ]
    if scores["bert"] is not None:
        labels.append("BERT")
        values.append(scores["bert"] * 100)
    if scores["word2vec"] is not None:
        labels.append("Word2Vec")
        values.append(scores["word2vec"] * 100)

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#176b4d", "#4b9274", "#d59b49", "#40709b", "#8565a5"][: len(values)],
            text=[f"{value:.1f}%" for value in values],
            textposition="inside",
        )
    )
    figure.update_layout(
        height=260,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor="#e9eeeb", ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#44534c"),
    )
    return figure


def report_markup(feedback):
    sections = [
        ("Overall match", feedback.get("overall_match", "")),
        ("Strengths", feedback.get("strengths", "")),
        ("Gaps", feedback.get("gaps", "")),
        ("Recommendations", feedback.get("recommendations", "")),
        ("Limitations", feedback.get("limitations", "")),
    ]
    return "".join(
        f'<div class="report-block"><strong>{escape(title)}</strong><span>{escape(str(text))}</span></div>'
        for title, text in sections
    )


with st.sidebar:
    st.markdown("## ◈ ResumeFit AI")
    st.caption("Model operations")
    st.divider()

    status = model_status()
    st.markdown("**Artifact status**")
    for name, available in status.items():
        state_class = "status-on" if available else "status-off"
        state_text = "Ready" if available else "Missing"
        st.markdown(
            f'<div class="status-row"><span>{name}</span><span class="{state_class}">{state_text}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    use_llm = st.toggle(
        "Generate AI feedback",
        value=bool(os.getenv("OPENAI_API_KEY")),
        help="Requires OPENAI_API_KEY in the .env file.",
    )
    st.caption("All scores are decision support only. A human reviewer must make final hiring decisions.")


st.markdown('<div class="app-kicker">Candidate intelligence workspace</div>', unsafe_allow_html=True)
st.title("Resume and Job Match Analyzer")
st.markdown(
    '<div class="app-subtitle">Compare role requirements with resume evidence and review explainable match signals.</div>',
    unsafe_allow_html=True,
)

input_left, input_right = st.columns(2, gap="large")

with input_left:
    st.subheader("Candidate resume")
    input_mode = st.segmented_control(
        "Resume input method",
        options=["Upload document", "Paste text"],
        default="Upload document",
        label_visibility="collapsed",
    )

    if input_mode == "Upload document":
        resume_file = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
        )
        try:
            resume_text = extract_document_text(resume_file) if resume_file else ""
            if resume_text:
                st.success(f"Extracted {len(resume_text.split()):,} words from {resume_file.name}.")
                with st.expander("Preview extracted text"):
                    st.text(resume_text[:5000])
        except Exception as exc:
            resume_text = ""
            st.error(f"Could not read the document: {exc}")
    else:
        resume_text = st.text_area(
            "Resume text",
            height=285,
            placeholder="Paste the candidate resume here...",
        )

with input_right:
    st.subheader("Job description")
    job_description = st.text_area(
        "Job description",
        height=333,
        placeholder="Paste the complete job description here...",
        label_visibility="collapsed",
    )

action_left, action_right = st.columns([1, 5])
with action_left:
    analyze_clicked = st.button("Analyze match", type="primary", width="stretch")
with action_right:
    st.caption("The trained BERT model is used automatically when its files are available.")

if analyze_clicked:
    if not resume_text.strip() or not job_description.strip():
        st.error("Add both a resume and job description before analyzing.")
    else:
        with st.spinner("Analyzing resume evidence and role requirements..."):
            try:
                st.session_state.analysis = analyze(resume_text, job_description, use_llm=use_llm)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

if "analysis" in st.session_state:
    result = st.session_state.analysis
    skills = result["skills"]
    scores = result["scores"]
    feedback = result["feedback"]

    st.divider()
    st.markdown('<div class="section-label">Analysis overview</div>', unsafe_allow_html=True)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Overall match", f"{scores['overall'] * 100:.1f}%")
    metric_2.metric("Skill coverage", f"{skills['skill_coverage'] * 100:.1f}%")
    metric_3.metric("Matched skills", len(skills["matched_skills"]))
    metric_4.metric("Missing skills", len(skills["missing_skills"]))

    if scores["overall"] >= 0.75:
        st.success(f"Strong match signal · {scores['source']}")
    elif scores["overall"] >= 0.45:
        st.warning(f"Potential match with notable gaps · {scores['source']}")
    else:
        st.error(f"Low current match signal · {scores['source']}")

    result_tab, skills_tab, report_tab, diagnostics_tab = st.tabs(
        ["Score profile", "Skill evidence", "Feedback report", "Diagnostics"]
    )

    with result_tab:
        chart_col, context_col = st.columns([3, 2], gap="large")
        with chart_col:
            st.plotly_chart(score_chart(scores), width="stretch")
        with context_col:
            st.markdown("#### Review context")
            st.metric("Resume length", f"{result['resume_word_count']:,} words")
            st.metric("Job description length", f"{result['job_word_count']:,} words")
            st.markdown(
                '<div class="risk-note">A high score indicates textual alignment, not candidate suitability, identity, or hiring eligibility.</div>',
                unsafe_allow_html=True,
            )

    with skills_tab:
        matched_col, missing_col = st.columns(2, gap="large")
        with matched_col:
            st.markdown("#### Matched requirements")
            st.markdown(skill_markup(skills["matched_skills"], "match"), unsafe_allow_html=True)
        with missing_col:
            st.markdown("#### Missing or unclear requirements")
            st.markdown(skill_markup(skills["missing_skills"], "missing"), unsafe_allow_html=True)

        with st.expander("All detected skills"):
            st.markdown("**Resume skills**")
            st.write(", ".join(skills["resume_skills"]) or "None detected")
            st.markdown("**Job description skills**")
            st.write(", ".join(skills["job_skills"]) or "None detected")

    with report_tab:
        st.caption(f"Generated by: {feedback.get('provider', 'Unknown')}")
        st.markdown(report_markup(feedback), unsafe_allow_html=True)

    with diagnostics_tab:
        diagnostics = pd.DataFrame(
            [
                {"Signal": "Overall score", "Score": scores["overall"], "Available": True},
                {"Signal": "BERT regression", "Score": scores["bert"], "Available": scores["bert"] is not None},
                {"Signal": scores["tfidf_source"], "Score": scores["tfidf"], "Available": True},
                {"Signal": "Word2Vec similarity", "Score": scores["word2vec"], "Available": scores["word2vec"] is not None},
                {"Signal": "Skill coverage", "Score": skills["skill_coverage"], "Available": True},
            ]
        )
        diagnostics["Score"] = diagnostics["Score"].map(
            lambda value: f"{value * 100:.1f}%" if value is not None else "Not loaded"
        )
        st.dataframe(diagnostics, width="stretch", hide_index=True)

    export_result = {
        "scores": scores,
        "skills": skills,
        "feedback": feedback,
        "resume_word_count": result["resume_word_count"],
        "job_word_count": result["job_word_count"],
    }
    st.download_button(
        "Download analysis JSON",
        data=json.dumps(export_result, indent=2),
        file_name="resume_match_analysis.json",
        mime="application/json",
    )
