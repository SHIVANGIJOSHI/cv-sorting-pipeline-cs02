"""
parser.py - Robust Document Parsing and Semantic Card Generator
Ingests PDF and TXT CVs and Job Descriptions, applies security guardrails,
and normalizes unstructured resumes into typed Pydantic models.
"""

import os
import re
from typing import List, Set
from pypdf import PdfReader
from schemas import JobDescriptionSchema, CandidateStructuredCard
from guardrails import SecurityGuardrails


class DocumentParser:
    """
    Ingests raw documents from disk, sanitizes contents against prompt injection,
    and structures text for downstream retrieval and evaluation.
    """

    # Common technical skills dictionary for heuristic lexical extraction
    COMMON_SKILL_LEXICON = {
        "python", "java", "c++", "c#", "rust", "go", "golang", "scala", "javascript", 
        "typescript", "sql", "postgresql", "mysql", "mongodb", "redis", "opensearch",
        "pytorch", "tensorflow", "jax", "keras", "scikit-learn", "numpy", "pandas",
        "transformers", "huggingface", "langchain", "llama-index", "spacy", "nltk",
        "fastapi", "flask", "django", "spring boot", "react", "docker", "kubernetes",
        "aws", "azure", "gcp", "git", "ci/cd", "rag", "fine-tuning", "llm", "lora"
    }

    def __init__(self):
        self.guardrails = SecurityGuardrails()

    def extract_raw_text(self, file_path: str) -> str:
        """
        Reads plain text from PDF or TXT files.
        Handles missing files and corrupted streams gracefully.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file does not exist: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()

        elif ext == ".pdf":
            try:
                reader = PdfReader(file_path)
                pages_text = []
                for idx, page in enumerate(reader.pages):
                    extracted = page.extract_text()
                    if extracted:
                        pages_text.append(extracted)
                return "\n".join(pages_text).strip()
            except Exception as e:
                raise RuntimeError(f"Failed to parse PDF '{file_path}': {str(e)}")

        else:
            raise ValueError(f"Unsupported format '{ext}'. Only .pdf and .txt files are accepted.")

    def parse_job_description(self, jd_path: str) -> JobDescriptionSchema:
        """
        Loads the Job Description and extracts key constraints:
        title, mandatory skills, and minimum experience requirements.
        """
        raw_text = self.extract_raw_text(jd_path)
        sanitized_text, _, _ = self.guardrails.sanitize_text(raw_text)

        # Heuristic 1: Extract skills matching the common lexicon
        tokens = re.findall(r"\b[A-Za-z0-9\+#\./-]+\b", sanitized_text.lower())
        token_set = set(tokens)
        detected_skills = sorted(list(token_set.intersection(self.COMMON_SKILL_LEXICON)))

        # Heuristic 2: Extract minimum years of experience using regex (e.g., '5+ years', '3-5 years')
        exp_match = re.search(r"(\d+)\+?\s*(?:to\s*\d+\s*)?(?:years|yrs)", sanitized_text, re.IGNORECASE)
        min_years = float(exp_match.group(1)) if exp_match else 0.0

        # Heuristic 3: Extract role title from the first non-empty line
        first_line = sanitized_text.splitlines()[0] if sanitized_text.splitlines() else "Target Technical Role"
        title = first_line[:60].strip()

        return JobDescriptionSchema(
            title=title,
            mandatory_skills=detected_skills,
            min_years_experience=min_years,
            raw_text=sanitized_text
        )

    def parse_candidate_cv(self, cv_path: str, candidate_id: str) -> CandidateStructuredCard:
        """
        Loads a candidate CV, neutralizes adversarial injection prompts,
        and constructs a structured semantic card.
        """
        file_name = os.path.basename(cv_path)
        raw_text = self.extract_raw_text(cv_path)

        # Apply security guardrails
        sanitized_text, threat_detected, threat_details = self.guardrails.sanitize_text(raw_text)

        # Extract skills present in CV
        tokens = re.findall(r"\b[A-Za-z0-9\+#\./-]+\b", sanitized_text.lower())
        token_set = set(tokens)
        detected_skills = sorted(list(token_set.intersection(self.COMMON_SKILL_LEXICON)))

        # Estimate experience
        exp_matches = re.findall(
            r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+experience)?", 
            sanitized_text, 
            re.IGNORECASE
        )
        exp_years = max([float(x) for x in exp_matches]) if exp_matches else 0.0

        # Inferred candidate name from header
        first_few_lines = [l.strip() for l in sanitized_text.splitlines() if l.strip()]
        inferred_name = first_few_lines[0][:40] if first_few_lines else f"Candidate_{candidate_id}"

        return CandidateStructuredCard(
            candidate_id=candidate_id,
            file_name=file_name,
            inferred_name=inferred_name,
            detected_skills=detected_skills,
            estimated_experience_years=exp_years,
            raw_text=sanitized_text,
            has_injection_risk=threat_detected,
            threat_details=threat_details
        )