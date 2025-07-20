
# dsRAG + FinanceBench

This repository presents a complete implementation of a high-performance Retrieval-Augmented Generation (RAG) pipeline using the `dsRAG` framework, tailored for financial question-answering on the `FinanceBench` dataset.

The pipeline includes:
- End-to-end retrieval, reranking, and generation using local and API-based LLMs
- Custom PDF chunking strategies for 10-K, 10-Q, and earnings call reports
- Cohere reranker integration for improved relevance
- Semantic evaluation comparing model answers with FinanceBench ground-truth
- Performance tracking (accuracy, throughput, refusals, benchmark errors)

---

## 📊 Evaluation Results

| Metric              | Value         |
|---------------------|---------------|
| Total Questions     | 150           |
| Correct Answers     | 80            |
| Aligned (Paraphrased) Answers | 17  |
| Incorrect Answers   | 28            |
| Refusals            | 15            |
| Benchmark Errors    | 10            |
| **Final Accuracy**  | **69.33%**    |

Compared against Ragie and Mafin 2.5 baselines.

---

## 🧩 Key Components

### 🔹 Chunking (`chunking/`)
- `chunk_pdf.py`: Splits SEC filings into semantically meaningful chunks.
- `chunk_debug_viewer.py`: View chunks extracted from individual PDFs.

### 🔹 Knowledge Base (`kb/`)
- `create_kb.py`: Builds a vector store using `SentenceTransformer` embeddings.
- `auto_context_model.py`: Adds context automatically using a smaller LLM.

### 🔹 Retrieval & Reranking
- `retriever/safe_query_kb.py`: Retrieves top-k chunks safely with fallback logic.
- `reranker/cohere_reranker.py`: Uses Cohere's reranker to improve result ordering.

### 🔹 LLM Integration (`llm/`)
- `mistral_local.py`: Interface for running Mistral or DeepSeek locally via vLLM.

### 🔹 Evaluation (`evaluation/`)
- `run_evaluation.py`: Runs evaluation on FinanceBench with result logging.
- `metrics_tracker.py`: Tracks correctness, GPU usage, token consumption.
- `evaluation_prompt.txt`: Prompt for financial question answering.
- Output saved in `evaluation/output/summary_report.csv` and Excel files.

---

## ⚙️ Setup Instructions

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Setup Environment
```bash
bash setup.sh
```

### 3. Build the Knowledge Base
```bash
python kb/create_kb.py
```

### 4. Run Evaluation
```bash
python evaluation/run_evaluation.py
```

---

## 📂 Folder Structure

```
dsrag-financebench/
├── chunking/           # PDF chunking logic
├── kb/                 # Vector store creation & auto-context models
├── retriever/          # Chunk retriever
├── reranker/           # Cohere CrossEncoder reranker
├── llm/                # Mistral / DeepSeek wrapper
├── evaluation/         # Full pipeline and outputs
├── utils/              # Token/GPU usage tracking
├── data/               # (Placeholder for FinanceBench PDFs)
└── external/           # Attribution to dsRAG and FinanceBench repos
```

---

## 🔗 References & Acknowledgments

This project builds on:

- **dsRAG** by D-Star AI: [https://github.com/D-Star-AI/dsRAG](https://github.com/D-Star-AI/dsRAG)
- **FinanceBench** by Patronus AI: [https://github.com/patronus-ai/FinanceBench](https://github.com/patronus-ai/FinanceBench)

---

## 👤 Author

**Praneet Kulkarni**  
Email: praneet@akashx.ai  
LinkedIn: https://www.linkedin.com/in/praneet-kulkarni/

If you use this work, please consider citing the original repositories and this implementation.
