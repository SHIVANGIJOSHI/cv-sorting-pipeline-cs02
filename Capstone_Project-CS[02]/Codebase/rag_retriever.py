"""
rag_retriever.py - Stage 1 Coarse Hybrid Retrieval Engine (Model 1)
Leverages BAAI/bge-base-en-v1.5 dense vector embeddings with in-memory matrix projection
and exact lexical skill coverage.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from schemas import JobDescriptionSchema, CandidateStructuredCard, Stage1RetrievalResult


class DenseLexicalRetriever:
    """
    Model 1: BGE-base Bi-Encoder Dense Retrieval paired with Lexical Token Overlap.
    Optimized for CPU execution.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.bi_encoder = SentenceTransformer(model_name, device="cpu")
        self.card_index: List[CandidateStructuredCard] = []
        self.dense_matrix: Optional[np.ndarray] = None

    def index_candidates(self, cards: List[CandidateStructuredCard]) -> None:
        """
        Embeds candidate structured cards into normalized unit vectors.
        """
        self.card_index = cards
        corpus = [
            f"Skills: {', '.join(card.detected_skills)}. Experience: {card.estimated_experience_years} years. {card.raw_text[:2000]}"
            for card in cards
        ]

        # Compute normalized L2 vectors for dot-product cosine similarity
        self.dense_matrix = self.bi_encoder.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def compute_lexical_coverage(self, jd_skills: List[str], cv_skills: List[str]) -> float:
        """
        Calculates percentage of required JD skills found in CV.
        """
        if not jd_skills:
            return 50.0

        jd_set = set([s.lower().strip() for s in jd_skills])
        cv_set = set([s.lower().strip() for s in cv_skills])

        overlap = jd_set.intersection(cv_set)
        coverage = (len(overlap) / len(jd_set)) * 100.0
        return float(min(100.0, coverage))

    def retrieve(
        self,
        jd: JobDescriptionSchema,
        w_dense: float = 0.60,
        w_lexical: float = 0.40
    ) -> List[Stage1RetrievalResult]:
        """
        Performs coarse hybrid retrieval against the Job Description query.
        Uses BGE asymmetric instruction prefix on the query.
        """
        if self.dense_matrix is None or len(self.card_index) == 0:
            return []

        # BGE asymmetric retrieval instruction prefix
        query_text = (
            f"Represent this sentence for searching relevant passages: "
            f"Required Skills: {', '.join(jd.mandatory_skills)}. "
            f"Min Experience: {jd.min_years_experience} years. {jd.raw_text[:2000]}"
        )

        query_embedding = self.bi_encoder.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # O(N) vectorized cosine similarity
        dot_products = np.dot(self.dense_matrix, query_embedding)

        results: List[Stage1RetrievalResult] = []
        for idx, card in enumerate(self.card_index):
            raw_cos = float(dot_products[idx])
            dense_score = max(0.0, min(100.0, ((raw_cos + 1.0) / 2.0) * 100.0))
            lexical_score = self.compute_lexical_coverage(jd.mandatory_skills, card.detected_skills)
            stage1_composite = (w_dense * dense_score) + (w_lexical * lexical_score)

            results.append(
                Stage1RetrievalResult(
                    candidate_id=card.candidate_id,
                    file_name=card.file_name,
                    dense_similarity_score=round(dense_score, 2),
                    lexical_overlap_score=round(lexical_score, 2),
                    stage1_composite=round(stage1_composite, 2)
                )
            )

        results.sort(key=lambda x: x.stage1_composite, reverse=True)
        return results