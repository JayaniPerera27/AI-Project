from src.skills import compare_skills, extract_skills


def test_extract_skills_is_case_insensitive():
    skills = extract_skills("Built PYTHON services using Docker and PostgreSQL.")
    assert "python" in skills
    assert "docker" in skills
    assert "postgresql" in skills


def test_compare_skills_reports_coverage():
    result = compare_skills(
        "Python developer with SQL experience",
        "Requires Python, SQL, Docker, and AWS",
    )
    assert result["matched_skills"] == ["python", "sql"]
    assert result["missing_skills"] == ["aws", "docker"]
    assert result["skill_coverage"] == 0.5

