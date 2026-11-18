# Task 2 — RAG System from Scratch

## Overview

This project implements a Retrieval-Augmented Generation (RAG) retrieval pipeline from scratch. It processes a PDF document, divides its content into smaller chunks, generates semantic embeddings, and retrieves relevant passages using dense, sparse, and hybrid retrieval techniques.

The objective is to evaluate how chunk size and retrieval strategies affect document retrieval performance.

## Architecture

PDF Document → Text Extraction → Text Chunking → Embedding Generation → FAISS Vector Search + BM25 → Hybrid Retrieval → Recall@K Evaluation

## Technologies Used

- **Python** — Core implementation.
- **PyMuPDF / PyPDF** — PDF text extraction.
- **Sentence Transformers** — Semantic embedding generation.
- **BAAI/bge-small-en-v1.5** — Embedding model.
- **FAISS** — Efficient vector similarity search.
- **BM25** — Keyword-based sparse retrieval.
- **Reciprocal Rank Fusion (RRF)** — Combines dense and sparse retrieval rankings.
- **NumPy** — Numerical operations.

## Implementation

### 1. Document Processing
- Extracted text from a 77-page research paper.
- Divided the extracted text into overlapping chunks.
- Evaluated two chunk sizes: 128 and 384 tokens.

### 2. Embedding Generation
- Generated semantic vector representations using BAAI/bge-small-en-v1.5.
- Stored the embeddings for similarity-based retrieval.

### 3. Dense Retrieval
- Used FAISS to perform vector similarity searches.
- Retrieved passages based on semantic relevance to the input query.

### 4. Sparse Retrieval
- Implemented BM25 for keyword-based document retrieval.
- Retrieved passages using lexical matching.

### 5. Hybrid Retrieval
- Combined dense and BM25 rankings using Reciprocal Rank Fusion (RRF).
- Evaluated whether combining semantic and keyword retrieval improves retrieval performance.

### 6. Evaluation
- Created 15 evaluation questions.
- Measured Recall@5 using a gold-passage matching criterion.
- Compared retrieval performance across different chunk sizes.

## Benchmark Results

| Chunk Size | Dense Recall@5 | BM25 Recall@5 | Hybrid Recall@5 |
|---|---:|---:|---:|
| 128 | 26.7% | 40.0% | 40.0% |
| 384 | 53.3% | 53.3% | 53.3% |

### Key Findings

- Increasing chunk size from 128 to 384 improved dense retrieval recall from 26.7% to 53.3%.
- BM25 achieved 40% recall with 128-token chunks.
- Hybrid retrieval achieved 53.3% recall with 384-token chunks.
- Larger chunks provided better retrieval coverage in this evaluation.

## Project Structure

```text
task_2_tndrr/
├── rag.py
├── questions_template.json
├── questions.json
└── results/
    ├── recall.json
    ├── failures_128.json
    └── failures_384.json
```

*The results and evaluation files above are generated during execution; include them in the repository when available.*

## Execution

Install dependencies:

```bash
pip install numpy faiss-cpu pypdf sentence-transformers PyMuPDF
```

Prepare the source PDF and evaluation questions, then run:

```bash
python rag.py
```

Refer to the script for its actual configuration and execution requirements.

## Key Learning Outcomes

- Understanding RAG architecture and retrieval pipelines.
- Implementing dense vector retrieval using FAISS.
- Understanding sparse retrieval using BM25.
- Combining retrieval methods using Reciprocal Rank Fusion.
- Evaluating retrieval quality using Recall@K.
- Understanding the impact of chunk size on retrieval performance.

## Conclusion

This project demonstrates an end-to-end retrieval pipeline combining semantic search, keyword-based search, and hybrid ranking. The evaluation showed that 384-token chunks achieved the highest Recall@5 of 53.3% across the tested configurations.
# Task 3 — Local LLM Benchmarking and Serving Plan

## Overview

This project evaluates locally hosted Large Language Models (LLMs) using Ollama. Two lightweight models are benchmarked against a common set of 10 prompts to compare generation speed, memory consumption, and response quality.

The project also includes a GPU serving-capacity estimation for handling multiple concurrent users.

## Technologies Used

- **Python** — Benchmark implementation and evaluation.
- **Ollama** — Local LLM hosting and inference.
- **Qwen 2.5 3B** — First benchmark model.
- **Llama 3.2 3B** — Second benchmark model.
- **REST API** — Communication with the local Ollama server.
- **JSON** — Benchmark result storage.
- **CUDA / NVIDIA GPU** — GPU-accelerated inference.

## Architecture

Prompt Dataset → Ollama API → Local LLM Inference → Response Generation → Performance Measurement → JSON Results → Model Comparison Report

## Implementation

### 1. Local Model Setup

Installed Ollama and downloaded two models:

- `qwen2.5:3b`
- `llama3.2:3b`

Both models were evaluated using the same prompts and generation settings.

### 2. Benchmark Dataset

Created a set of 10 prompts covering:

- Concept explanation.
- Python programming.
- Mathematical reasoning.
- Summarization.
- Structured JSON generation.
- Technical comparisons.
- General knowledge.
- Translation.
- SQL query generation.
- Argument generation.

### 3. Benchmark Execution

The benchmark script:

- Connects to the local Ollama API.
- Loads and tests both models.
- Performs warm-up inference.
- Uses temperature 0 and seed 0.
- Generates responses for all 10 prompts.
- Measures generation speed and prefill speed.
- Records token counts and model memory usage.
- Saves benchmark results as JSON files.

### 4. Model Comparison

The comparison script evaluates:

- Mean tokens per second.
- Median tokens per second.
- Loaded model memory.
- VRAM usage.
- Generated response quality.

## Benchmark Results

| Metric | Qwen 2.5 3B | Llama 3.2 3B |
|---|---:|---:|
| Mean generation speed | 81.2 tok/s | 76.9 tok/s |
| Median generation speed | 80.5 tok/s | 77.0 tok/s |
| Loaded memory | 2.09 GB | 2.32 GB |
| Reported VRAM usage | 2.09 GB | 2.32 GB |

### Key Findings

- Qwen 2.5 3B achieved a higher mean generation speed.
- Qwen achieved approximately 5.6% higher mean throughput than Llama.
- Qwen also reported lower loaded memory consumption.
- Both models were evaluated using the same 10-prompt benchmark.

### Response Quality Observations

- Both models produced accurate photosynthesis summaries.
- Both extracted the correct name and age in the JSON task.
- Both correctly identified Canberra as Australia's capital.
- Some TCP/UDP responses were incomplete.
- Hindi translation quality needed improvement.
- Both generated SQL queries for retrieving the top three salaries.

## Serving Capacity Estimation

A separate Python script estimates the serving requirements for 20 concurrent users on a hypothetical 24 GB GPU.

### Assumptions

- Model: Llama 3.1 8B.
- Weight quantization: 4-bit AWQ.
- GPU memory: 24 GB.
- Concurrent users: 20.
- Context length: 2,048 tokens per user.
- KV cache precision: FP16.

### Estimated Results

| Metric | Estimate |
|---|---:|
| KV cache per token | 128 KiB |
| Total KV cache requirement | 5.4 GB |
| Available KV cache budget | 14.6 GB |
| Maximum KV cache tokens | 111,389 |
| Estimated throughput per user | 43 tok/s |
| Aggregate estimated throughput | 868 tok/s |

**Note:** These are theoretical estimates based on the serving script's assumptions, not measured production performance. The serving estimate is separate from the actual 3B Ollama benchmarks.

## Project Structure

```text
task_3_tndrr/
├── bench.py
├── compare.py
├── prompts.txt
├── serving_plan.py
└── results/
    ├── qwen2.5_3b.json
    ├── llama3.2_3b.json
    └── compare.md
```

*The results directory and its files are generated during benchmark execution. Include the actual generated files in the repository when available.*

## Execution

### 1. Install Ollama

Download and install Ollama from:

https://ollama.com/

### 2. Download Models

```bash
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Run Benchmark

Create the results directory:

```bash
mkdir -p results
```

Run:

```bash
python bench.py --models qwen2.5:3b llama3.2:3b
```

### 5. Generate Comparison Report

```bash
python compare.py
```

### 6. Run Serving Plan

```bash
python serving_plan.py
```

## Key Learning Outcomes

- Running LLMs locally using Ollama.
- Understanding local model inference.
- Measuring tokens-per-second performance.
- Comparing model memory consumption.
- Evaluating response quality across different tasks.
- Understanding concurrent inference and KV cache memory.
- Estimating GPU serving capacity.

## Conclusion

This project demonstrates practical local LLM benchmarking and serving-capacity estimation. Qwen 2.5 3B achieved the higher measured generation speed in the tested environment, while the serving analysis explored the theoretical requirements for supporting concurrent LLM inference.
