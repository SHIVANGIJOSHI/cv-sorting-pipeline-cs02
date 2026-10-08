"""
main.py - Unified Command-Line Interface (CLI) Entry Point
Project: Capstone_Project-CS[02] - CV Sorting using LLMs
Orchestrated via LangChain LCEL Runnables
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
from typing import List

from schemas import CandidateFinalEvaluation, FinalRankingPayload
from orchestrator import LangChainCVSortingPipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CS[02]: LangChain-Orchestrated Two-Stage Information Retrieval CV Sorter"
    )
    parser.add_argument("--jd_path", type=str, required=True, help="Path to Job Description file (.txt or .pdf)")
    parser.add_argument("--cv_dir", type=str, required=True, help="Directory containing candidate CV files")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top candidates to display (default: 10)")
    parser.add_argument("--output_file", type=str, default="ranked_results.json", help="Path to export JSON output")
    parser.add_argument("--mode", type=str, choices=["local", "api"], default="local", help="Execution mode: 'local' (BGE neural) or 'api' (gpt-4o-mini)")
    parser.add_argument("--api_key", type=str, default=None, help="Optional OpenAI API key for API mode")
    return parser.parse_args()


def display_terminal_table(ranked_candidates: List[CandidateFinalEvaluation], top_k: int) -> None:
    display_subset = ranked_candidates[:top_k]

    print("\n" + "=" * 96)
    print(
        f"{'RANK':<5} | {'SCORE':<8} | {'DENSE':<7} | {'LEXICAL':<8} | {'STAGE2':<10} | {'NAME / FILE':<24} | {'FLAG'}"
    )
    print("=" * 96)

    for cand in display_subset:
        flag_str = "[ALERT]" if cand.security_flag else "CLEAN"
        display_label = cand.inferred_name if cand.inferred_name != "Candidate" else cand.file_name
        display_label = display_label[:24]

        print(
            f"{cand.rank:<5} | "
            f"{cand.final_score:<8.2f} | "
            f"{cand.stage1_dense_score:<7.1f} | "
            f"{cand.stage1_lexical_score:<8.1f} | "
            f"{cand.stage2_rerank_score:<10.1f} | "
            f"{display_label:<24} | "
            f"{flag_str}"
        )

    print("=" * 96)


def main() -> None:
    args = parse_arguments()

    if not os.path.exists(args.jd_path):
        print(f"[FATAL ERROR] Job Description file not found at: {args.jd_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.cv_dir):
        print(f"[FATAL ERROR] CV directory not found at: {args.cv_dir}", file=sys.stderr)
        sys.exit(1)

    supported_exts = (".pdf", ".txt")
    cv_filenames = [f for f in os.listdir(args.cv_dir) if f.lower().endswith(supported_exts)]
    if not cv_filenames:
        print(f"[FATAL ERROR] No supported CV files found in {args.cv_dir}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "#" * 60)
    print(f"#  CS[02]: CV SORTING PIPELINE ({args.mode.upper()} MODE)          #")
    print("#" * 60)

    # Initialize LangChain Orchestrator
    pipeline = LangChainCVSortingPipeline(mode=args.mode, api_key=args.api_key)

    # Execute LCEL Chain
    result_state = pipeline.run(
        jd_path=args.jd_path,
        cv_dir=args.cv_dir,
        cv_files=cv_filenames,
        top_k=args.top_k,
        output_file=args.output_file
    )

    final_rankings = result_state["final_rankings"]

    # Render results table
    display_terminal_table(final_rankings, top_k=args.top_k)

    if final_rankings:
        top_cand = final_rankings[0]
        print(f"\n>>> Top Match Highlight (Rank #{top_cand.rank}): {top_cand.file_name}")
        print(f"    Inferred Identity   : {top_cand.inferred_name}")
        print(f"    Composite Fit Score : {top_cand.final_score:.2f} / 100.00")
        print(
            f"    Rubrics Breakdown   : Tech={top_cand.rubric_breakdown.technical_depth:.1f}, "
            f"Exp={top_cand.rubric_breakdown.experience_relevance:.1f}, "
            f"Domain={top_cand.rubric_breakdown.domain_fit:.1f}, "
            f"Edu={top_cand.rubric_breakdown.education_certs:.1f}"
        )
        print(f"    Audit Justification : {top_cand.audit_justification}\n")

    # Export structured JSON
    output_payload = FinalRankingPayload(
        job_description_path=args.jd_path,
        total_candidates_processed=len(final_rankings),
        top_candidates_count=min(args.top_k, len(final_rankings)),
        ranked_candidates=final_rankings
    )

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        f_out.write(output_payload.model_dump_json(indent=2))

    print(f"[STATUS] Results successfully exported via LangChain pipeline to: {args.output_file}\n")


if __name__ == "__main__":
    main()