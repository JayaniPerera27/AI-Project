from src.feedback import generate_feedback
from src.scoring import score_resume
from src.skills import compare_skills


def analyze(resume_text, job_description, use_llm=False):
    if not resume_text.strip() or not job_description.strip():
        raise ValueError("Both resume text and job description are required.")

    skills = compare_skills(resume_text, job_description)
    scores = score_resume(
        resume_text,
        job_description,
        skills["skill_coverage"],
    )
    feedback = generate_feedback(
        resume_text,
        job_description,
        scores["overall"],
        skills["matched_skills"],
        skills["missing_skills"],
        use_llm,
    )

    return {
        "skills": skills,
        "scores": scores,
        "feedback": feedback,
        "resume_word_count": len(resume_text.split()),
        "job_word_count": len(job_description.split()),
    }

