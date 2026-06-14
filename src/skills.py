import json
import re
from functools import lru_cache

from src.config import SKILLS_PATH


DEFAULT_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "mysql", "postgresql", "mongodb", "redis", "oracle",
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "data analysis", "data science", "generative ai",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "spark",
    "docker", "kubernetes", "git", "linux", "aws", "azure", "gcp", "ci/cd",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "power bi", "tableau", "excel", "project management", "agile", "scrum",
    "communication", "leadership", "problem solving", "teamwork",
    "financial analysis", "marketing", "sales", "recruitment",
]


@lru_cache(maxsize=1)
def load_skills():
    if SKILLS_PATH.exists():
        with SKILLS_PATH.open("r", encoding="utf-8") as file:
            values = json.load(file)
        if isinstance(values, list):
            return sorted({str(value).strip().lower() for value in values if value})
    return DEFAULT_SKILLS


def extract_skills(text):
    normalized = str(text).lower()
    found = []
    for skill in load_skills():
        pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            found.append(skill)
    return sorted(set(found))


def compare_skills(resume_text, job_description):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))
    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    coverage = len(matched) / len(job_skills) if job_skills else 0.0

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "skill_coverage": coverage,
    }

