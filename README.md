
# dsRAG + FinanceBench

This repository presents a complete implementation of a high-performance Retrieval-Augmented Generation (RAG) pipeline using the `dsRAG` framework, tailored for financial question-answering on the `FinanceBench` dataset.

The pipeline includes:
- End-to-end retrieval, reranking, and generation using local and API-based LLMs
- Custom PDF chunking strategies for 10-K, 10-Q, and earnings call reports
- Various reranker integration for improved relevance
- Semantic evaluation comparing model answers with FinanceBench ground-truth
- Performance tracking (accuracy, throughput, refusals, benchmark errors)

---

## 📊 Evaluation Results Anthropic Claude 4.0 Sonnet (Best performing so far)

| Metric              | Value         |
|---------------------|---------------|
| Total Questions     | 150           |
| Correct Answers     | 122           |
| Incorrect Answers   | 09            |
| Refusals            | 13            |
| Benchmark Errors    | 6             |
| **Final Accuracy**  | **81.33%**    |

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


API Keys: Specify API Keys for the respective models you use.

**Provider**
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... # if using GPT-4o-mini CO_API_KEY=... # if using Cohere reranker VOYAGE_API_KEY=... # if using Voyage reranker

Chunking_Test_Script.py Chunking / Auto-Context Model: (GPT-4o-mini)

**Specify the Chunking model you want to use:**

from dsrag.llm import OpenAIChatAPI
auto_context_model = OpenAIChatAPI(model="gpt-4o-mini", temperature=0.2, max_tokens=600)

**Uses GPT-4o-mini to generate document titles, summaries, and semantic sections for chunking.**

**Specify the Embedding model you want to use: Embedding Model (OpenAI text-embedding-3-large)**

from dsrag.embedding import OpenAIEmbedding
embedding_model = OpenAIEmbedding(model="text-embedding-3-large")

**Uses OpenAI’s text-embedding-3-large to convert each chunk into 3072-dimensional vectors for retrieval.**

**Specify the Reranking model you want to use:**

Reranker Model (Cohere Reranker v3)

from dsrag.reranker import CohereReranker
reranker = CohereReranker(model="rerank-english-v3.0")

**Use Cohere’s reranker to reorder retrieved chunks by semantic relevance to the query.**

Note: Similarly, if you want to configure different models, simply call the respective class (OpenAIChatAPI, OpenAIEmbedding, CohereReranker, etc.) and pass the desired model name as an argument with their respective classes configured in the llm.py, embedding.py and reranker.py. This lets you run experiments with different model backends and configurations interchangeably.

**Inference.py: Inference**

**Specify the inference model you want to use:**

from dsrag.llm import OpenAIChatAPI llm = OpenAIChatAPI(model="gpt-4o-mini", temperature=0.2, max_tokens=800)

**Specify the reranker model you want to use:**

from dsrag.reranker import VoyageReranker reranker = VoyageReranker(model="rerank-2.5-lite") # or "rerank-2.5"

Lastly, After getting the inference results, I sent the Question, Ground_truth answer and the answer returned by my selected model to the GPT-4o model for Semantic similarity to identify if the results are correct or not. You could refer the python file - semantic_evaluation.py.

## 👤 Author

**Praneet Kulkarni**  
Email: praneetk@umich.edu  
LinkedIn: https://www.linkedin.com/in/praneet-kulkarni/

If you use this work, please consider citing the original repositories and this implementation.
