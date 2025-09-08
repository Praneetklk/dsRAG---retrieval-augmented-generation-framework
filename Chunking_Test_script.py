import sys
import os
import time
import glob
import pickle

# --- dsRAG imports ---
sys.path.insert(0, "/root/Antropic_Test/Claude/dsRAG")
# from dsrag.llm import OpenAIChatAPI
# from dsrag.llm import LocalTransformersLLM
from dsrag.llm import AnthropicChatAPI

from dsrag.knowledge_base import KnowledgeBase
from dsrag.reranker import CohereReranker
from dsrag.embedding import OpenAIEmbedding


from dotenv import load_dotenv
load_dotenv()

# =========================
# Config
# =========================
KB_ID = "Antropic_chunks"  # Unique ID for your KnowledgeBase
STORAGE_DIR = "/root/Antropic_Test/storage_dir"
CHUNK_STORAGE_DIR = os.path.join(STORAGE_DIR, "chunk_storage")
PDF_DIR = "/root/TEST/financebench/pdfs"
# PDF_DIR = "/root/TEST/financebench/Demo_pdf"  # for quick testing


auto_context_model = AnthropicChatAPI(
    model="claude-sonnet-4-20250514",
    temperature=0.2,
    max_tokens=600,
)


embedding_model = OpenAIEmbedding(model="text-embedding-3-large")
# =========================
# Helpers (chunk counting)
# =========================
def pick_kb_pickle(chunk_dir: str, kb_id: str) -> str | None:
    """
    Prefer exact chunk_storage/{kb_id}.pkl if it exists; else pick the newest among {kb_id}*.pkl.
    """
    exact = os.path.join(chunk_dir, f"{kb_id}.pkl")
    if os.path.exists(exact):
        return exact

    candidates = glob.glob(os.path.join(chunk_dir, f"{kb_id}*.pkl"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def count_chunks_schema_dict_of_dict(path: str) -> tuple[int, dict]:
    """
    Your pickle layout (confirmed): 
      top-level dict: { "<DOC>.pdf": { 0: <chunk_dict>, 1: <chunk_dict>, ... }, ... }
    Returns (total_chunks, per_doc_counts).
    """
    with open(path, "rb") as f:
        store = pickle.load(f)

    if not isinstance(store, dict):
        return 0, {}

    per_doc_counts = {}
    for doc, inner in store.items():
        if isinstance(inner, dict):
            per_doc_counts[doc] = len(inner)
        else:
            per_doc_counts[doc] = 0
    return sum(per_doc_counts.values()), per_doc_counts



reranker = CohereReranker()

# ---- Token counting patch on the embedding model itself ---
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")  # works for OpenAI v3 embeddings
except Exception:
    _enc = None

def _tok_len(seq):
    if isinstance(seq, str):
        seq = [seq]
    if _enc:
        return sum(len(_enc.encode(s)) for s in seq)
    # fallback heuristic (~4 chars per token)
    return sum(max(1, len(s) // 4) for s in seq)

# Keep original method
_original_get_embeddings = embedding_model.get_embeddings

def _counting_get_embeddings(text, input_type=None):
    # init counter
    if not hasattr(embedding_model, "token_counter"):
        embedding_model.token_counter = {"total_tokens": 0}
    # count & accumulate
    embedding_model.token_counter["total_tokens"] += _tok_len(text)
    # delegate to the real call
    return _original_get_embeddings(text, input_type)

# Monkey-patch the instance
embedding_model.get_embeddings = _counting_get_embeddings


kb = KnowledgeBase(
    kb_id=KB_ID,
    storage_directory=STORAGE_DIR,
    embedding_model=embedding_model,
    auto_context_model=auto_context_model,
    reranker=reranker,
    # Set to True if you want to append to an existing KB; False to require fresh.
    exists_ok=False,
)

# =========================
# Build document list
# =========================
documents = []
for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(PDF_DIR, filename)
        doc_id = filename.replace("/", "_")
        documents.append({
            "doc_id": doc_id,
            "file_path": file_path,
            "document_title": filename.rsplit(".", 1)[0],
            "auto_context_config": {
                "use_generated_title": True,
                "get_document_summary": True,
                "get_section_summaries": False
            },
            "file_parsing_config": {
                "use_vlm": False,
                "always_save_page_images": False
            },

            "semantic_sectioning_config": {
                "use_semantic_sectioning": True,
                "llm_provider": "anthropic",
                "model": "claude-3-7-sonnet-20250219"
            },

            "chunking_config": {
                "chunk_size": 800,
                "min_length_for_chunking": 2000
            },
            "metadata": {
                "source": "FinanceBench",
                "filename": filename
            }
        })

print(f"[INFO] Adding {len(documents)} documents to KnowledgeBase...")
start_time = time.time()

# --- Token counting wrapper for embeddings ---
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")   # <- works for OpenAI v3 embeddings

except Exception:
    _enc = None

# Keep original
_original_get = kb._get_embeddings

def _tok_len(seq):
    if isinstance(seq, str):
        seq = [seq]
    if _enc:
        return sum(len(_enc.encode(t)) for t in seq)
    # fallback heuristic ~4 chars per token
    return sum(max(1, len(t) // 4) for t in seq)

def _wrapped_get(text, input_type: str = ""):
    # init counter
    if not hasattr(embedding_model, "token_counter"):
        embedding_model.token_counter = {"total_tokens": 0}
    # count & accumulate
    embedding_model.token_counter["total_tokens"] += _tok_len(text)
    # delegate
    return _original_get(text, input_type)

# monkey-patch
kb._get_embeddings = _wrapped_get


try:
    successful = kb.add_documents(
        documents=documents,
        max_workers=5,       # increase if your machine can handle it
        show_progress=True,
        rate_limit_pause=0.5
    )
    duration = time.time() - start_time

    # --- Token stats (best-effort; requires your embedding path to populate token_counter) ---
    total_tokens = 0
    if hasattr(embedding_model, "token_counter"):
        total_tokens = int(embedding_model.token_counter.get("total_tokens", 0))

    print(f"\n[INFO] Ingestion finished in {duration:.2f} seconds.")
    print(f"🔢 Total tokens encoded: {total_tokens}")
    if duration > 0:
        print(f"⚡ Tokens/sec: {total_tokens / duration:.2f}")

    # --- Docs stats ---
    if isinstance(successful, list):
        num_docs = sum(1 for x in successful if x is True) if all(isinstance(x, bool) for x in successful) else len(successful)
    elif isinstance(successful, int):
        num_docs = successful
    else:
        num_docs = len(documents)  # fallback
    if duration > 0 and num_docs:
        print(f"📄 Docs/sec: {num_docs / duration:.2f}")
    else:
        print(f"📄 Docs processed: {num_docs}")

    # --- Chunk stats from persisted storage (schema-specific) ---
    chunk_path = pick_kb_pickle(CHUNK_STORAGE_DIR, KB_ID)
    if not chunk_path:
        print(f"[WARN] No chunk storage file found for kb_id='{KB_ID}' in {CHUNK_STORAGE_DIR}")
        total_chunks = 0
    else:
        total_chunks, per_doc_counts = count_chunks_schema_dict_of_dict(chunk_path)
        print(f"[INFO] Counted chunks from: {chunk_path}")
        print("[INFO] Per-doc chunk counts:")
        for doc, n in per_doc_counts.items():
            print(f"  - {doc}: {n}")

    print(f"📊 Total chunks (from storage): {total_chunks}")
    if duration > 0 and total_chunks:
        print(f"📦 Chunks/sec: {total_chunks / duration:.2f}")

except Exception as e:
    print(f"[❌ ERROR] Document ingestion failed: {e}")
