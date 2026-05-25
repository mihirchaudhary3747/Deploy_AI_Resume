import re
import os
from dotenv import load_dotenv

# Load .env BEFORE anything else
load_dotenv()

from groq import Groq


def get_api_key():
    # Streamlit Cloud
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    # Local .env (already loaded above)
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    raise ValueError("GROQ_API_KEY not found in .env or Streamlit secrets!")


client = Groq(api_key=get_api_key())


def analyze_resume_ai(resume_text):
    try:
        prompt = f"""You are an expert ATS resume analyzer with 15+ years of HR experience.

Analyze the resume below and reply STRICTLY in this exact format (no markdown, no asterisks, no extra text):

Score: <0-100>
ATS_Score: <0-100>
Field: <job role>
Level: <Fresher / Junior / Mid-Level / Senior / Expert>
Current_Skills: <comma separated list>
Missing_Skills: <comma separated list>
Recommended_Skills: <comma separated list>

Strengths:
- <strength 1>
- <strength 2>
- <strength 3>

Weaknesses:
- <weakness 1>
- <weakness 2>
- <weakness 3>

Improvements:
- <improvement 1>
- <improvement 2>
- <improvement 3>
- <improvement 4>
- <improvement 5>

ATS_Tips:
- <tip 1>
- <tip 2>
- <tip 3>

Career_Path: <Junior Title> → <Mid Title> → <Senior Title>
Salary_Range: <INR range like 4-8 LPA>

Resume:
{resume_text[:3000]}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict ATS resume analyzer. "
                        "Always analyze the ACTUAL resume content provided. "
                        "Give REAL scores based on quality. Be specific. "
                        "Follow the output format exactly — no extra commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1500,
        )

        result = response.choices[0].message.content
        print("RAW AI OUTPUT:\n", result)
        return _parse_result(result)

    except Exception as e:
        print("GROQ API ERROR:", e)
        return _error_response(str(e))


def _parse_result(text):
    """Parse the structured AI response into a dict."""

    def get_line(pattern, default="N/A"):
        """Extract a single value from a key: value line."""
        match = re.search(pattern, text)
        return match.group(1).strip() if match else default

    def get_csv(pattern):
        """Extract a comma-separated line into a list."""
        match = re.search(pattern, text)
        if match:
            return [s.strip() for s in match.group(1).split(",") if s.strip()]
        return []

    def get_bullets(section_label):
        """Extract bullet points under a section heading."""
        # Match section heading and capture all following '- item' lines
        pattern = rf"(?m)^{re.escape(section_label)}:\s*\n((?:\s*-[^\n]+\n?)+)"
        match = re.search(pattern, text)
        if match:
            return [
                item.strip()
                for item in re.findall(r"-\s*(.+)", match.group(1))
                if item.strip()
            ]
        return []

    # FIX: use \bScore: to avoid matching ATS_Score
    raw_score = get_line(r"(?<!\w)Score:\s*(\d+)", "60")
    raw_ats   = get_line(r"ATS_Score:\s*(\d+)", "55")

    score     = max(0, min(100, int(raw_score)))
    ats_score = max(0, min(100, int(raw_ats)))

    return {
        "score":              score,
        "ats_score":          ats_score,
        "field":              get_line(r"Field:\s*([^\n]+)", "General"),
        "level":              get_line(r"Level:\s*([^\n]+)", "Fresher"),
        "career_path":        get_line(r"Career_Path:\s*([^\n]+)", "Not specified"),
        "salary_range":       get_line(r"Salary_Range:\s*([^\n]+)", "Not specified"),
        "current_skills":     get_csv(r"Current_Skills:\s*([^\n]+)"),
        "skills_missing":     get_csv(r"Missing_Skills:\s*([^\n]+)"),
        "skills_recommended": get_csv(r"Recommended_Skills:\s*([^\n]+)"),
        "strengths":          get_bullets("Strengths"),
        "weaknesses":         get_bullets("Weaknesses"),
        "improvements":       get_bullets("Improvements"),
        "ats_tips":           get_bullets("ATS_Tips"),
    }


def _error_response(error_msg="Unknown error"):
    return {
        "score":              0,
        "ats_score":          0,
        "field":              "Error",
        "level":              "Unknown",
        "career_path":        "N/A",
        "salary_range":       "N/A",
        "current_skills":     [],
        "skills_missing":     [f"Analysis failed: {error_msg}"],
        "skills_recommended": ["Please try again"],
        "strengths":          ["Could not analyze"],
        "weaknesses":         ["Could not analyze"],
        "improvements":       ["Please re-upload your resume"],
        "ats_tips":           ["Try again"],
    }
