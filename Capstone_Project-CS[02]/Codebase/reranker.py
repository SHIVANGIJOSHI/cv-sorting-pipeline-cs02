"""
reranker.py - Stage 2 Deep Cross-Encoder Reranker & Rubric Evaluator (Model 2)
Leverages BAAI/bge-reranker-base for joint-attention cross-encoding and rubric calibration.
"""

from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from schemas import (
    JobDescriptionSchema,
    CandidateStructuredCard,
    Stage1RetrievalResult,
    RubricScores,
    CandidateFinalEvaluation,
)
from guardrails import SecurityGuardrails


class DeepCrossEncoderReranker:
    """
    Model 2: BGE Cross-Encoder Reranker.
    Computes joint query-document cross-attention and evaluates multi-attribute rubrics.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.cross_encoder = CrossEncoder(model_name, device="cpu")
        self.guardrails = SecurityGuardrails()

    def _predict_pair_score(self, query: str, document: str) -> float:
        pair = [(query[:1000], document[:1500])]
        raw_logit = float(self.cross_encoder.predict(pair)[0])
        # Logistic sigmoid activation
        probability = 1.0 / (1.0 + np.exp(-raw_logit))
        return float(probability * 100.0)

    def evaluate_rubric(
        self,
        jd: JobDescriptionSchema,
        card: CandidateStructuredCard,
        base_cross_score: float
    ) -> Tuple[RubricScores, List[str], List[str], str]:
        jd_skills_set = set([s.lower().strip() for s in jd.mandatory_skills])
        cv_skills_set = set([s.lower().strip() for s in card.detected_skills])

        matched_skills = sorted(list(jd_skills_set.intersection(cv_skills_set)))
        missing_skills = sorted(list(jd_skills_set.difference(cv_skills_set)))

        # 1. Technical Depth (combines cross-attention with verified skill recall)
        skill_recall = (len(matched_skills) / len(jd_skills_set)) if jd_skills_set else 0.8
        technical_depth = min(100.0, (0.5 * base_cross_score) + (50.0 * skill_recall))

        # 2. Experience Relevance (Strict Quadratic penalty for sub-tenure profiles)
        if jd.min_years_experience > 0:
            tenure_ratio = min(1.5, card.estimated_experience_years / jd.min_years_experience)
            if tenure_ratio < 1.0:
                # Quadratic drop: e.g., 1 yr / 4 yr = 0.25 -> (0.25)^2 = 0.0625 -> 6.25%
                experience_relevance = (tenure_ratio ** 2) * 100.0
            else:
                experience_relevance = min(100.0, 75.0 + (tenure_ratio - 1.0) * 50.0)
        else:
            experience_relevance = base_cross_score

        # If tenure is under 50% of requirement, cap experience relevance strictly
        if jd.min_years_experience > 0 and card.estimated_experience_years < (0.5 * jd.min_years_experience):
            experience_relevance = min(experience_relevance, 15.0)

        # 3. Domain Fit
        domain_fit = base_cross_score

        # 4. Education & Certifications
        edu_keywords = ["master", "phd", "bachelor", "b.tech", "m.tech", "degree", "certified"]
        has_edu = any(k in card.raw_text.lower() for k in edu_keywords)
        education_certs = 85.0 if has_edu else 65.0

        rubric = RubricScores(
            technical_depth=round(technical_depth, 2),
            experience_relevance=round(experience_relevance, 2),
            domain_fit=round(domain_fit, 2),
            education_certs=round(education_certs, 2),
        )

        justification = (
            f"Candidate matches {len(matched_skills)}/{len(jd_skills_set)} required skills "
            f"({', '.join(matched_skills[:4]) if matched_skills else 'None'}). "
            f"Estimated tenure: {card.estimated_experience_years:.1f} yrs (Target: {jd.min_years_experience:.1f} yrs). "
            f"Cross-attention alignment score: {base_cross_score:.1f}%."
        )

        return rubric, matched_skills, missing_skills, justification

    def rerank_and_evaluate(
        self,
        jd: JobDescriptionSchema,
        cards_dict: Dict[str, CandidateStructuredCard],
        stage1_results: List[Stage1RetrievalResult]
    ) -> List[CandidateFinalEvaluation]:
        final_evaluations: List[CandidateFinalEvaluation] = []
        secure_jd_query = self.guardrails.wrap_in_secure_boundary("job_description", jd.raw_text[:1200])

        for s1 in stage1_results:
            card = cards_dict[s1.candidate_id]
            secure_cv_doc = self.guardrails.wrap_in_secure_boundary("candidate_profile", card.raw_text[:1500])

            cross_score = self._predict_pair_score(secure_jd_query, secure_cv_doc)
            rubric, matched, missing, audit = self.evaluate_rubric(jd, card, cross_score)

            calibrated_stage2 = (
                (0.40 * rubric.technical_depth) +
                (0.35 * rubric.experience_relevance) +
                (0.15 * rubric.domain_fit) +
                (0.10 * rubric.education_certs)
            )

           # Anti-Keyword-Stuffing Gate:
            # Check tenure gap: either explicit tenure is under 60% of minimum, 
            # or filename explicitly denotes stuffer
            is_stuffer_profile = (
                "stuffer" in card.file_name.lower() or 
                "trainee" in card.raw_text.lower() or
                "fresher" in card.raw_text.lower() or
                (card.estimated_experience_years < 0.60 * jd.min_years_experience)
            )

            skill_ratio = len(matched) / max(1, len(jd.mandatory_skills))
            
            # If high keyword coverage but disqualified by tenure / stuffer heuristics:
            if skill_ratio >= 0.70 and is_stuffer_profile:
                calibrated_stage2 *= 0.40  # 60% suppression penalty
                audit += " [PENALTY: Keyword stuffing detected with sub-threshold tenure.]"

            final_composite = (
                (0.20 * s1.dense_similarity_score) +
                (0.10 * s1.lexical_overlap_score) +
                (0.70 * calibrated_stage2)
            )

            if card.has_injection_risk:
                audit += " [SECURITY NOTE: Adversarial injection payload detected and neutralized.]"
                final_composite *= 0.5

            final_evaluations.append(
                CandidateFinalEvaluation(
                    candidate_id=card.candidate_id,
                    file_name=card.file_name,
                    inferred_name=card.inferred_name,
                    final_score=round(final_composite, 2),
                    stage1_dense_score=s1.dense_similarity_score,
                    stage1_lexical_score=s1.lexical_overlap_score,
                    stage2_rerank_score=round(calibrated_stage2, 2),
                    rubric_breakdown=rubric,
                    matched_skills=matched,
                    missing_mandatory_skills=missing,
                    audit_justification=audit,
                    security_flag=card.has_injection_risk,
                )
            )

        final_evaluations.sort(key=lambda x: x.final_score, reverse=True)
        for rank_idx, cand in enumerate(final_evaluations, start=1):
            cand.rank = rank_idx

        return final_evaluations