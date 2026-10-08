# Two-Stage Cascaded IR & LLM Orchestration for Automated Resume Screening
**Capstone Project CS[02]: Technical Report & Benchmark Suite**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain_LCEL-orange.svg)](https://github.com/langchain-ai/langchain)
[![Inference Mode A](https://img.shields.io/badge/Mode_A-Local_CPU_(BGE)-green.svg)](https://huggingface.co/BAAI)
[![Inference Mode B](https://img.shields.io/badge/Mode_B-Groq_LPU_(MoE)-purple.svg)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An auditable, enterprise-grade Information Retrieval (IR) and generative scoring pipeline for automated resume screening. The system mitigates keyword stuffing, adversarial prompt injection attacks, and quadratic tenure mismatches using a two-stage cascaded architecture.

---

## Architecture Overview

```text
                      Raw Resumes (PDF / TXT)
                                │
                                ▼
            [Document Ingestion & Security Guardrails]
           (Unicode NFKC + Regex Prompt Injection Shield)
                                │
                                ▼
          [Stage 1: Coarse Hybrid Retrieval Model (Bi-Encoder)]
             - Dense Cosine Similarity: BAAI/bge-base-en-v1.5
             - Lexical Mandatory Overlap: |S_cand ∩ S_jd| / |S_jd|
             - Score: S_Stage1 = 0.60 * σ_dense + 0.40 * σ_lex
                                │
                                ▼ Top-K Shortlist
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
  [Mode A: Local Neural]                  [Mode B: Generative LLM]
   BAAI/bge-reranker-base                  openai/gpt-oss-20b (Groq LPU)
   Cross-Attention on [q; d]               (Fallback: gpt-4o-mini via OpenAI)
   Zero external API calls                 Pydantic Schema Constrained (.with_structured_output)
            │                                       │
            └───────────────────┬───────────────────┘
                                ▼
         [Stage 2 Scoring, Anti-Stuffing & Quadratic Tenure Gating]
                                │
                                ▼
              Final Calibrated Candidate Report Cards
```

---

## Key Features

- **Dual-Mode Execution:**
  - **Mode A (Offline Neural Engine):** Runs 100% locally on commodity CPU hardware using `BAAI/bge-base-en-v1.5` and `BAAI/bge-reranker-base`. Zero external network dependencies, zero operating expenses, ~65 ms/resume latency, and complete candidate data privacy.
  - **Mode B (Generative LLM Engine):** Uses `openai/gpt-oss-20b` served on Groq LPUs (~420 ms latency) with an automated fallback to OpenAI's `gpt-4o-mini`. Produces deterministic, structured Pydantic report cards with transparent recruiter scorecard audits.
- **Scoring Formulation & Keyword Suppression:**
  - **Quadratic Tenure Penalty:** Enforces $S_{\text{exp}} = (T_{\text{cand}} / T_{\text{req}})^2 \times 100$ for candidates below required seniority, penalizing junior applicants who artificially match technical keywords.
  - **Anti-Keyword-Stuffing Gate:** Suppresses Stage 2 scores by $0.40\times$ if a candidate presents $\ge 70\%$ keyword overlap while holding $< 60\%$ of required tenure.
- **Security Guardrails:** Pre-scoring Unicode NFKC normalization and regex engine intercept adversarial prompt injection attempts (e.g., system instructions, delimiters, jailbreak overrides), penalizing suspect submissions by 50% and logging an audit alert.

---

## Empirical Benchmark Results

Evaluated across a benchmark corpus of **1,000 resumes** spanning **10 distinct technical domains** (100 CVs per suite; 20 seniors, 25 borderline mid-levels, 30 keyword stuffers, and 25 adversarial/unrelated profiles).

| Benchmark Suite (100 CVs each) | P@20 (Mode A) | P@20 (Mode B) | MRR (Mode A / B) | Avg Stuffer Rank (A / B) | Threats Caught |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Suite 01: Senior Backend Engineer | 100.0% | 100.0% | 1.00 / 1.00 | #75.5 / #38.5 | 10 / 10 |
| Suite 02: ML Research Engineer | 85.0% | 100.0% | 1.00 / 1.00 | #75.5 / #38.0 | 10 / 10 |
| Suite 03: DevOps & Platform | 95.0% | 100.0% | 1.00 / 1.00 | #75.5 / #37.5 | 10 / 10 |
| Suite 04: Data Platform Engineer | 85.0% | 100.0% | 1.00 / 1.00 | #75.5 / #39.0 | 10 / 10 |
| Suite 05: Lead Frontend Engineer | 65.0% | 95.0% | 1.00 / 1.00 | #75.5 / #38.5 | 10 / 10 |
| Suite 06: Cybersecurity Analyst | 40.0% | 90.0% | 0.17 / 1.00 | #43.9 / #40.0 | 10 / 10 |
| Suite 07: Full Stack Engineer | 95.0% | 100.0% | 1.00 / 1.00 | #75.5 / #37.0 | 10 / 10 |
| Suite 08: Embedded Systems | 80.0% | 100.0% | 1.00 / 1.00 | #75.5 / #38.5 | 10 / 10 |
| Suite 09: Database Administrator | 85.0% | 100.0% | 1.00 / 1.00 | #75.5 / #38.0 | 10 / 10 |
| Suite 10: NLP & Agentic AI Architect | 100.0% | 100.0% | 1.00 / 1.00 | #75.5 / #38.0 | 10 / 10 |
| **Aggregate Benchmark (1,000 Resumes)** | **83.00%** | **98.50%** | **0.9167 / 1.0000** | **#72.3 / #38.3** | **100 / 100 (100%)** |

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/SHIVANGIJOSHI/cv-sorting-pipeline-cs02.git](https://github.com/SHIVANGIJOSHI/cv-sorting-pipeline-cs02.git)
cd cv-sorting-pipeline-cs02
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r Capstone_Project-CS[02]/Codebase/requirements.txt
```

### 3. Configure API Credentials (Mode B Only)
If using Mode B with Groq LPUs:
```bash
export GROQ_API_KEY="gsk_your_groq_key_here"
```
Or for the OpenAI fallback:
```bash
export OPENAI_API_KEY="sk_your_openai_key_here"
```

To suppress tokenizer multiprocessing warnings during local evaluation:
```bash
export TOKENIZERS_PARALLELISM=false
```

---

## Usage

### Run a Single Resume Screening Pipeline
Run the main orchestrator against a single job description and candidate folder:
```bash
python3 Capstone_Project-CS[02]/Codebase/main.py \
  --jd test_env/benchmark_suite/sample_01_senior_backend/job_description.txt \
  --cv_dir test_env/benchmark_suite/sample_01_senior_backend/cvs \
  --mode local
```

For Mode B (Generative LLM):
```bash
python3 Capstone_Project-CS[02]/Codebase/main.py \
  --jd test_env/benchmark_suite/sample_01_senior_backend/job_description.txt \
  --cv_dir test_env/benchmark_suite/sample_01_senior_backend/cvs \
  --mode api
```

---

## Reproducing the Empirical Benchmarks

### 1. Generate the 1,000-CV Corpus
Generate the synthetic evaluation suites (10 job profiles $\times$ 100 resumes):
```bash
python3 test_env/generate_10_jds_1000_cvs.py
```

### 2. Execute Mode A (Local CPU Benchmark)
Evaluates all 1,000 CVs across all 10 suites:
```bash
python3 -W ignore test_env/run_batch_benchmark_cpu.py
```

### 3. Execute Mode B (Rate-Governed Groq Benchmark)
Evaluates Mode B with backoff handling to comply with Groq free-tier limits (30 RPM, 8k TPM):
```bash
python3 -W ignore test_env/run_batch_benchmark_groq.py
```

---

## Technical Report

The formal 3-page academic report (`Report.pdf`) adheres to 12pt Times typography, 1-inch margins, and explicit scoring formulas. The LaTeX source and compiled PDF are located in `Capstone_Project-CS[02]/Report/`:

Compile from source:
```bash
cd Capstone_Project-CS[02]/Report
pdflatex Report.tex
```

---

## Citation & References

```bibtex
@misc{joshi2026twostage,
  title={Two-Stage Cascaded Information Retrieval and LLM Orchestration for Automated Resume Screening},
  author={Shivangi Joshi},
  year={2026},
  note={Capstone Project CS[02] Technical Report}
}
```

```bibtex
@software{langchain2023,
  title={LangChain: Building applications with LLMs through composability},
  author={Harrison Chase},
  year={2023},
  url={[https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
