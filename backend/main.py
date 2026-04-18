import io
import os
import re
from typing import Any

import docx
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class InferredSkills(BaseModel):
    skills: list[str]


class MatchResult(BaseModel):
    match_score: int = Field(..., ge=0, le=100)
    title_match: str
    summary: str
    matched_skills: list[str]
    missing_skills: list[str]
    suggested_improvements: list[str]
    keyword_overlap: list[str]
    inferred_skills: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    success: bool
    filename: str
    used_fallback: bool
    result: MatchResult


def build_app() -> FastAPI:
    app = FastAPI(title="Resume vs Job Matcher API", version="1.0.0")

    cors_origins = os.getenv("CORS_ORIGINS", "*")
    origins = ["*"] if cors_origins.strip() == "*" else [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
    allow_credentials = origins != ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "status": "ok",
            "message": "Resume vs Job Matcher API is running",
            "endpoints": {
                "POST /api/analyze": "Upload resume and submit a job description",
                "GET /health": "Health check",
            },
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        }

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_resume(
        resume: UploadFile = File(...),
        job_description: str = Form(...),
    ) -> AnalyzeResponse:
        validate_upload(resume)

        file_bytes = await resume.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="File is too large. Max size is 10 MB.")

        resume_text = extract_resume_text(resume.filename or "", file_bytes)
        cleaned_job_description = clean_text(job_description)

        if len(resume_text) < 30:
            raise HTTPException(
                status_code=400,
                detail="Could not extract enough text from the resume file.",
            )

        if len(cleaned_job_description) < 20:
            raise HTTPException(
                status_code=400,
                detail="Job description is too short.",
            )

        used_fallback = False
        inferred_skills: list[str] = []
        try:
            enhanced_resume_text, inferred_skills = enhance_resume_with_skills(resume_text)
            result = analyze_with_gemini(enhanced_resume_text, cleaned_job_description)
            result["inferred_skills"] = inferred_skills
        except Exception as e:
            print(f"Error during analysis: {e}")
            used_fallback = True
            result = fallback_match(resume_text, cleaned_job_description)

        return AnalyzeResponse(
            success=True,
            filename=resume.filename or "resume",
            used_fallback=used_fallback,
            result=MatchResult(**result),
        )

    return app


app = build_app()


def validate_upload(resume: UploadFile) -> None:
    filename = resume.filename or ""
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload PDF, DOCX, or TXT.",
        )


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)
    return clean_text("\n".join(text_parts))


def extract_text_from_docx(file_bytes: bytes) -> str:
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return clean_text("\n".join(paragraphs))


def extract_text_from_txt(file_bytes: bytes) -> str:
    return clean_text(file_bytes.decode("utf-8", errors="ignore"))


def extract_resume_text(filename: str, file_bytes: bytes) -> str:
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if extension == ".docx":
        return extract_text_from_docx(file_bytes)
    if extension == ".txt":
        return extract_text_from_txt(file_bytes)

    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Please upload PDF, DOCX, or TXT.",
    )


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def extract_keywords(text: str) -> set[str]:
    stopwords = {
        "about",
        "also",
        "and",
        "are",
        "build",
        "built",
        "but",
        "for",
        "from",
        "have",
        "into",
        "job",
        "looking",
        "must",
        "need",
        "nice",
        "our",
        "role",
        "team",
        "that",
        "the",
        "their",
        "this",
        "with",
        "you",
        "your",
    }

    matches = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{1,}", text.lower())
    return {match for match in matches if match not in stopwords and len(match) > 1}


def score_to_title(score: int) -> str:
    if score >= 80:
        return "Strong match"
    if score >= 60:
        return "Moderate match"
    if score >= 40:
        return "Partial match"
    return "Low match"


def fallback_match(resume_text: str, job_description: str) -> dict[str, Any]:
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)

    overlap = sorted(resume_keywords.intersection(job_keywords))
    missing = sorted(job_keywords - resume_keywords)

    if not job_keywords:
        score = 0
    else:
        score = min(100, round((len(overlap) / len(job_keywords)) * 100))

    return {
        "match_score": score,
        "title_match": score_to_title(score),
        "summary": "This result was generated with a keyword-based fallback because the AI analysis was unavailable.",
        "matched_skills": overlap[:15],
        "missing_skills": missing[:15],
        "suggested_improvements": [
            "Add the missing skills you genuinely have with project evidence.",
            "Rewrite bullet points to match the job description language more closely.",
            "Include measurable achievements, tools, and frameworks for each relevant project.",
        ],
        "keyword_overlap": overlap[:20],
        "inferred_skills": [],
    }


def enhance_resume_with_skills(resume_text: str) -> tuple[str, list[str]]:
    client = get_gemini_client()
    prompt = f"""
Analyze the following project description or resume. Identify not only the mentioned tools but also the implicit technical skills required to achieve the described results (e.g., if the user mentions 'low latency chat,' infer 'WebSockets' and 'Concurrency Management').

Return a JSON object with a single field 'skills' containing a list of these inferred implicit skills.

Resume:
{resume_text}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=InferredSkills,
        ),
    )

    inferred_skills: list[str] = []
    if getattr(response, "parsed", None):
        parsed = response.parsed
        if isinstance(parsed, InferredSkills):
            inferred_skills = parsed.skills
        elif isinstance(parsed, dict):
            inferred_skills = parsed.get("skills", [])

    if not inferred_skills:
        raw_text = (getattr(response, "text", "") or "").strip()
        if raw_text:
            try:
                parsed_json = InferredSkills.model_validate_json(raw_text)
                inferred_skills = parsed_json.skills
            except Exception:
                pass

    enhanced_text = resume_text + "\n\nInferred Implicit Skills:\n" + ", ".join(inferred_skills)
    return enhanced_text, inferred_skills


def analyze_with_gemini(resume_text: str, job_description: str) -> dict[str, Any]:
    client = get_gemini_client()

    prompt = f"""
You are an expert ATS resume matcher.

Compare the resume and the job description. Return strict JSON only.

Instructions:
- Score the resume from 0 to 100.
- Be honest and strict, but fair.
- Only mark a skill as matched if the resume clearly supports it.
- Put missing requirements into missing_skills.
- suggested_improvements must contain specific actions to improve this resume for this job.
- summary must be short and easy to show in a UI card.

Resume:
{resume_text}

Job Description:
{job_description}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=MatchResult,
        ),
    )

    if getattr(response, "parsed", None):
        parsed = response.parsed
        if isinstance(parsed, MatchResult):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return MatchResult(**parsed).model_dump()

    raw_text = (getattr(response, "text", "") or "").strip()
    if not raw_text:
        raise RuntimeError("Gemini returned an empty response.")

    return MatchResult.model_validate_json(raw_text).model_dump()
