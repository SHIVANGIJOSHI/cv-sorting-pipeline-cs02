"""
schemas.py - Data Contracts and Pydantic Schemas for CV Sorting Engine (CS[02])
Ensures deterministic data exchange between parsing, retrieval, reranking, and evaluation stages.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class JobDescriptionSchema(BaseModel):
    """Structured representation of the Job Description."""
    title: str = Field(default="Target Role", description="Job position title")
    mandatory_skills: List[str] = Field(default_factory=list, description="Non-negotiable required technical skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Desirable or secondary technical skills")
    min_years_experience: float = Field(default=0.0, description="Minimum required years of professional experience")
    domain_keywords: List[str] = Field(default_factory=list, description="Industry, domain, or architectural keywords")
    raw_text: str = Field(description="Original unedited text of the job description")


class CandidateStructuredCard(BaseModel):
    """Structured semantic card representing an ingested candidate resume."""
    candidate_id: str = Field(description="Unique internal candidate identifier")
    file_name: str = Field(description="Source file name of the CV")
    inferred_name: str = Field(default="Candidate", description="Extracted candidate name")
    detected_skills: List[str] = Field(default_factory=list, description="Identified technical and domain competencies")
    estimated_experience_years: float = Field(default=0.0, description="Parsed years of experience")
    raw_text: str = Field(description="Sanitized raw text representation")
    has_injection_risk: bool = Field(default=False, description="Flag indicating if jailbreak patterns were intercepted")
    threat_details: List[str] = Field(default_factory=list, description="Forensic log of caught injection patterns")


class Stage1RetrievalResult(BaseModel):
    """Outputs from Stage 1: Coarse Hybrid Retrieval (Dense Vector + Lexical Overlap)."""
    candidate_id: str
    file_name: str
    dense_similarity_score: float = Field(ge=0.0, le=100.0, description="Bi-encoder cosine similarity mapped to 0-100")
    lexical_overlap_score: float = Field(ge=0.0, le=100.0, description="Token/skill coverage percentage mapped to 0-100")
    stage1_composite: float = Field(ge=0.0, le=100.0, description="Weighted preliminary retrieval score")


class RubricScores(BaseModel):
    """Multi-attribute evaluation breakdown."""
    technical_depth: float = Field(ge=0.0, le=100.0, description="Skill alignment and engineering depth")
    experience_relevance: float = Field(ge=0.0, le=100.0, description="Alignment of project history and seniority")
    domain_fit: float = Field(ge=0.0, le=100.0, description="Industry and system architecture relevance")
    education_certs: float = Field(ge=0.0, le=100.0, description="Degrees and relevant certifications")


class LLMCandidateAuditOutput(BaseModel):
    """Structured Pydantic schema enforced on generative LLM outputs."""
    technical_depth: float = Field(ge=0.0, le=100.0, description="Score on technical skills alignment and depth")
    experience_relevance: float = Field(ge=0.0, le=100.0, description="Score evaluating candidate tenure against JD minimum")
    domain_fit: float = Field(ge=0.0, le=100.0, description="Score evaluating relevant industry and system architecture experience")
    education_certs: float = Field(ge=0.0, le=100.0, description="Score on degrees, education, or professional credentials")
    qualitative_critique: str = Field(description="2-3 sentence transparent qualitative recruiter assessment")


class CandidateFinalEvaluation(BaseModel):
    """Complete evaluation and audit profile for an individual candidate."""
    rank: int = Field(default=0, description="Final ordinal rank after complete pipeline execution")
    candidate_id: str
    file_name: str
    inferred_name: str
    final_score: float = Field(ge=0.0, le=100.0, description="Final calibrated score (0 to 100)")
    stage1_dense_score: float = Field(ge=0.0, le=100.0)
    stage1_lexical_score: float = Field(ge=0.0, le=100.0)
    stage2_rerank_score: float = Field(ge=0.0, le=100.0)
    rubric_breakdown: RubricScores
    matched_skills: List[str]
    missing_mandatory_skills: List[str]
    audit_justification: str = Field(description="Detailed qualitative rationale explaining the final rank")
    security_flag: bool = Field(default=False, description="Whether adversarial text was neutralized")


class FinalRankingPayload(BaseModel):
    """Root export schema for CLI and machine-readable output."""
    job_description_path: str
    total_candidates_processed: int
    top_candidates_count: int
    ranked_candidates: List[CandidateFinalEvaluation]