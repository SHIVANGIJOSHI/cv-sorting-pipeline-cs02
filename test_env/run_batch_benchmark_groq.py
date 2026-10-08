"""
run_batch_benchmark_groq.py - Rate-Governed Evaluation for Mode B (Groq API)
Executes Mode B across benchmark suites with adaptive pacing and exponential backoff
to strictly comply with Groq free-tier limits (30 RPM, 8K TPM).
"""

import os
import sys
import glob
import time
import random
from typing import List, Dict, Any

# 1. Resolve paths to import from Codebase
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODEBASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../Capstone_Project-CS[02]/Codebase"))

if not os.path.exists(CODEBASE_DIR):
    raise FileNotFoundError(f"Codebase directory not found at: {CODEBASE_DIR}")

sys.path.insert(0, CODEBASE_DIR)

from orchestrator import LangChainCVSortingPipeline

# 2. Benchmark directory lookup
BENCHMARK_ROOT = os.path.join(CURRENT_DIR, "benchmark_suite")
sample_dirs = sorted([d for d in glob.glob(f"{BENCHMARK_ROOT}/sample_*") if os.path.isdir(d)])

if not sample_dirs:
    raise FileNotFoundError(f"No sample folders found in {BENCHMARK_ROOT}. Please run generate_10_jds_1000_cvs.py first.")

# 3. Read Groq API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[ERROR] Please export your GROQ_API_KEY before running this script.")
    print("Example: export GROQ_API_KEY='gsk_...'")
    sys.exit(1)

# Helper function to classify candidate cohorts
def is_cohort(candidate: Any, prefix: str) -> bool:
    fn = getattr(candidate, "file_name", "").lower()
    cid = getattr(candidate, "candidate_id", "").lower()
    return prefix in fn or prefix in cid

def execute_safe_groq_benchmark(
    suites_to_run: List[str],
    candidates_per_suite: int = 100,
    delay_between_calls: float = 2.2
):
    """
    Runs Mode B with rate-limit dampening.
    - delay_between_calls = 2.2s guarantees max ~27 RPM (under 30 RPM limit).
    """
    print("\n" + "=" * 88)
    print(f"  MODE B (GROQ API: openai/gpt-oss-20b) RATE-GOVERNED BENCHMARK")
    print(f"  Pacing: {delay_between_calls}s inter-call sleep | Evaluating {len(suites_to_run)} suite(s)")
    print("=" * 88 + "\n")

    pipeline = LangChainCVSortingPipeline(mode="api", api_key=GROQ_API_KEY)

    aggregate_p20 = []
    aggregate_mrr = []
    aggregate_stuffer_ranks = []
    total_threats_caught = 0

    for idx, s_dir in enumerate(suites_to_run, start=1):
        suite_name = os.path.basename(s_dir)
        jd_path = os.path.join(s_dir, "job_description.txt")
        cv_dir = os.path.join(s_dir, "cvs")
        cv_files = sorted([f for f in os.listdir(cv_dir) if f.endswith(".txt")])[:candidates_per_suite]

        print(f"[*] Processing [{idx}/{len(suites_to_run)}]: {suite_name} ({len(cv_files)} CVs)...")

        # Execute with retry wrapper
        max_retries = 5
        success = False
        attempt = 0

        while attempt < max_retries and not success:
            try:
                # Run the LangChain pipeline in Mode B
                result = pipeline.run(
                    jd_path=jd_path,
                    cv_dir=cv_dir,
                    cv_files=cv_files,
                    top_k=candidates_per_suite
                )
                success = True
            except Exception as e:
                err_msg = str(e).lower()
                attempt += 1
                if "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg:
                    sleep_time = (2 ** attempt) * 5 + random.uniform(1.0, 3.0)
                    print(f"  [WARN] Rate limit hit (429). Backing off for {sleep_time:.1f}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    print(f"  [ERROR] Unhandled exception during suite execution: {e}")
                    raise e

        rankings = result["final_rankings"]

        # 1. Precision@20
        top_20 = rankings[:20]
        seniors_in_top20 = sum(1 for c in top_20 if is_cohort(c, "c1_senior"))
        p20 = seniors_in_top20 / 20.0
        aggregate_p20.append(p20)

        # 2. MRR
        first_senior_rank = next((c.rank for c in rankings if is_cohort(c, "c1_senior")), 100)
        mrr = 1.0 / first_senior_rank if first_senior_rank > 0 else 0.0
        aggregate_mrr.append(mrr)

        # 3. Stuffer Demotion
        stuffers = [c.rank for c in rankings if is_cohort(c, "c3_stuffer")]
        avg_stuffer = (sum(stuffers) / len(stuffers)) if stuffers else 0.0
        aggregate_stuffer_ranks.append(avg_stuffer)

        # 4. Threats Neutralized
        alerts = 0
        for c in rankings:
            flag = getattr(c, "has_security_alert", None)
            if flag is None:
                flag = getattr(c, "security_flag", False)
            if flag and is_cohort(c, "c4_adversarial"):
                alerts += 1
        total_threats_caught += alerts

        print(
            f"  --> {suite_name:<30} | "
            f"P@20: {p20 * 100:>5.1f}% | "
            f"MRR: {mrr:.2f} | "
            f"Avg Stuffer: #{avg_stuffer:<4.1f} | "
            f"Threats: {alerts}/10"
        )

        # Cool-down between entire suites to reset rolling token minute window
        if idx < len(suites_to_run):
            cooldown = 15.0
            print(f"  [PAUSE] Cooling down for {cooldown}s to reset rolling TPM window...\n")
            time.sleep(cooldown)

    print("\n" + "=" * 88)
    print("MODE B AGGREGATE BENCHMARK RESULTS:")
    print(f"  - Evaluated Suites            : {len(suites_to_run)}")
    print(f"  - Mean Precision@20           : {sum(aggregate_p20) / len(aggregate_p20) * 100:.2f}%")
    print(f"  - Mean Reciprocal Rank (MRR)  : {sum(aggregate_mrr) / len(aggregate_mrr):.4f}")
    print(f"  - Avg Keyword Stuffer Rank    : #{sum(aggregate_stuffer_ranks) / len(aggregate_stuffer_ranks):.1f}")
    print(f"  - Adversarial Detection Rate  : {total_threats_caught}/{len(suites_to_run) * 10} ({(total_threats_caught / (len(suites_to_run) * 10)) * 100:.1f}%)")
    print("=" * 88 + "\n")

if __name__ == "__main__":
    # Choose either single-suite verification (Suite 01) or full run
    # For a safe test without hitting daily token quotas, run Suite 01 first:
    target_suites = [sample_dirs[0]] 
    
    # If you want to evaluate multiple suites, pass a slice (e.g., sample_dirs[:3]):
    # target_suites = sample_dirs[:3]

    execute_safe_groq_benchmark(
        suites_to_run=target_suites,
        candidates_per_suite=100,
        delay_between_calls=2.2
    )