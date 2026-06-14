import os


def _deterministic_feedback(score, matched_skills, missing_skills):
    if score >= 0.75:
        assessment = "The resume shows strong alignment with the role."
    elif score >= 0.45:
        assessment = "The resume shows partial alignment, with several clear gaps."
    else:
        assessment = "The resume currently shows limited alignment with the role."

    strengths = (
        f"The strongest evidence is the overlap in {', '.join(matched_skills[:8])}."
        if matched_skills
        else "No explicit skill overlap was detected from the configured skill vocabulary."
    )
    gaps = (
        f"Important missing or unclear skills include {', '.join(missing_skills[:8])}."
        if missing_skills
        else "No missing skills were detected from the configured skill vocabulary."
    )
    recommendation = (
        "Add evidence-based project or work examples for the missing skills, and quantify relevant outcomes."
        if missing_skills
        else "Strengthen the resume with measurable outcomes and role-specific achievements."
    )

    return {
        "provider": "Local structured feedback",
        "overall_match": assessment,
        "strengths": strengths,
        "gaps": gaps,
        "recommendations": recommendation,
        "limitations": "This assessment checks textual evidence only and must not replace human hiring judgment.",
    }


def generate_feedback(resume, job_description, score, matched_skills, missing_skills, use_llm):
    fallback = _deterministic_feedback(score, matched_skills, missing_skills)
    api_key = os.getenv("OPENAI_API_KEY")
    if not use_llm or not api_key:
        return fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        prompt = f"""
You are an HR screening assistant. Use only evidence explicitly present in the
resume and job description. Do not infer protected characteristics and do not
invent qualifications.

Job description:
{job_description}

Resume:
{resume}

Match score: {score:.2f}
Detected matched skills: {matched_skills}
Detected missing skills: {missing_skills}

Return concise JSON with string fields: overall_match, strengths, gaps,
recommendations, limitations.
"""
        response = client.responses.create(model=model, input=prompt)

        import json
        text = response.output_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)
        result["provider"] = f"OpenAI ({model})"
        return result
    except Exception as exc:
        fallback["provider"] += f" - LLM unavailable: {exc}"
        return fallback

