"""
run_batch_benchmark.py - Corrected Metric Extraction
"""

import os
import sys
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODEBASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../Capstone_Project-CS[02]/Codebase"))

if not os.path.exists(CODEBASE_DIR):
    raise FileNotFoundError(f"Codebase directory not found at: {CODEBASE_DIR}")

sys.path.insert(0, CODEBASE_DIR)

from orchestrator import LangChainCVSortingPipeline

BENCHMARK_ROOT = os.path.join(CURRENT_DIR, "benchmark_suite")
sample_dirs = sorted([d for d in glob.glob(f"{BENCHMARK_ROOT}/sample_*") if os.path.isdir(d)])

if not sample_dirs:
    raise FileNotFoundError(f"No sample folders found in {BENCHMARK_ROOT}.")

pipeline = LangChainCVSortingPipeline(mode="local")

print("\n" + "=" * 84)
print(f"  RUNNING BENCHMARK EVALUATION ACROSS {len(sample_dirs)} SUITES (1,000 TOTAL CVs)")
print("=" * 84)

total_p20 = []
total_mrr = []
stuffer_ranks = []
threats_caught = 0

for s_dir in sample_dirs:
    suite_name = os.path.basename(s_dir)
    jd_path = os.path.join(s_dir, "job_description.txt")
    cv_dir = os.path.join(s_dir, "cvs")
    cv_files = [f for f in os.listdir(cv_dir) if f.endswith(".txt")]

    result = pipeline.run(
        jd_path=jd_path,
        cv_dir=cv_dir,
        cv_files=cv_files,
        top_k=100
    )
    rankings = result["final_rankings"]

    # Helper function to check filename or candidate_id
    def is_cohort(candidate, prefix):
        fn = getattr(candidate, "file_name", "").lower()
        cid = getattr(candidate, "candidate_id", "").lower()
        return prefix in fn or prefix in cid

    # Metric 1: Precision@20 (Expected: Cohort 1 'c1_senior' in top 20)
    top_20 = rankings[:20]
    seniors_in_top20 = sum(1 for c in top_20 if is_cohort(c, "c1_senior"))
    p20 = seniors_in_top20 / 20.0
    total_p20.append(p20)

    # Metric 2: MRR (Rank of the first qualified senior)
    first_senior_rank = next((c.rank for c in rankings if is_cohort(c, "c1_senior")), 100)
    mrr = 1.0 / first_senior_rank if first_senior_rank > 0 else 0.0
    total_mrr.append(mrr)

    # Metric 3: Stuffer Rank Demotion (c3_stuffer should be pushed down)
    stuffers = [c.rank for c in rankings if is_cohort(c, "c3_stuffer")]
    avg_stuffer = (sum(stuffers) / len(stuffers)) if stuffers else 0.0
    stuffer_ranks.append(avg_stuffer)

    # Metric 4: Security Alerts Intercepted
    # Check both possible schema attribute names: has_security_alert or security_flag
    alerts = 0
    for c in rankings:
        flag = getattr(c, "has_security_alert", None)
        if flag is None:
            flag = getattr(c, "security_flag", False)
        if flag and is_cohort(c, "c4_adversarial"):
            alerts += 1
    threats_caught += alerts

    print(
        f"Suite: {suite_name:<30} | "
        f"P@20: {p20 * 100:>5.1f}% | "
        f"MRR: {mrr:.2f} | "
        f"Avg Stuffer: #{avg_stuffer:<4.1f} | "
        f"Threats: {alerts}/10"
    )

print("=" * 84)
print("AGGREGATE BENCHMARK RESULTS (10 SUITES / 1,000 CVs):")
print(f"  - Mean Precision@20           : {sum(total_p20) / len(total_p20) * 100:.2f}%")
print(f"  - Mean Reciprocal Rank (MRR)  : {sum(total_mrr) / len(total_mrr):.4f}")
print(f"  - Avg Keyword Stuffer Rank    : #{sum(stuffer_ranks) / len(stuffer_ranks):.1f}")
print(f"  - Adversarial Detection Rate  : {threats_caught}/100 ({threats_caught:.1f}%)")
print("=" * 84 + "\n")