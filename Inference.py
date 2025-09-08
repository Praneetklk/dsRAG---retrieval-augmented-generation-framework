import sys
sys.path.insert(0, "/root/Antropic_Test/Claude/dsRAG")

from dsrag.knowledge_base import KnowledgeBase
# from dsrag.llm import OpenAIChatAPI
# from dsrag.llm import DeepSeekAPI
# from dsrag.reranker import CohereReranker
from datasets import load_dataset
# from sentence_transformers import SentenceTransformer
# from dsrag.llm import LocalTransformersLLM
from dsrag.embedding import OpenAIEmbedding

import torch
from dsrag.llm import AnthropicChatAPI

import pandas as pd
import time
import psutil
import os
from pynvml import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import requests
from dotenv import load_dotenv
load_dotenv()



nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#embedding_model = SentenceTransformer("infgrad/stella-base-en-v2")

llm = AnthropicChatAPI(
    model="claude-sonnet-4-20250514",
    temperature=0.2,
    max_tokens=800,
)

# auto_context_model = OpenAIChatAPI(model="gpt-4o-mini")
# reranker = CohereReranker()

from dsrag.reranker import VoyageReranker

reranker = VoyageReranker(model="rerank-2.5-lite")  # or "rerank-2.5"


# embedding_model = MiniLMEmbedding(model_name="all-MiniLM-L6-v2")

# ✅ Setup KnowledgeBase & LLM
kb = KnowledgeBase(
    kb_id="Antropic_chunks",
    reranker=reranker,
    storage_directory="/root/Antropic_Test/storage_dir",
    exists_ok=True
)

_ = kb.query(["warm-up"])

# llm = OpenAIChatAPI(model="gpt-4o-mini")
# llm = AnthropicChatAPI(model="claude-sonnet-4-20250514")
# ✅ Load FinanceBench dataset
dataset = load_dataset("PatronusAI/financebench", split="train")
print(f"Loaded {len(dataset)} FinanceBench questions.")

results_list = []
chunk_details_dict = {}

def safe_llm_call(prompt: str):
    """
    Calls dsRAG's LLM wrapper robustly:
    1) Try messages = [{role, content}] (Anthropic/OpenAI style)
    2) Fallback to plain string prompt
    Returns a dict with 'content' and token placeholders.
    """
    try:
        # Prefer messages format (common for Anthropic/OpenAI chat wrappers)
        messages = [
            {"role": "system", "content": "You are a helpful assistant answering financial questions using retrieved information. Answer the following question strictly using the retrieved context from 10-K/10-Q filings. If the context does not provide enough information, reply exactly: 'The provided documents do not contain sufficient information to answer this question.' and nothing else"},
            {"role": "user", "content": prompt},
        ]
        resp = llm.make_llm_call(messages)   # many wrappers accept messages
    except TypeError:
        # Fallback: some wrappers accept a single prompt string
        resp = llm.make_llm_call(prompt)
    except Exception as e:
        print(f"❌ Error in safe_llm_call: {type(e).__name__} — {e}")
        raise

    # Normalize response to string
    if isinstance(resp, dict):
        # Some wrappers might return {'content': '...'} or {'choices': [...]}
        if "content" in resp and isinstance(resp["content"], str):
            text = resp["content"]
        elif "choices" in resp and resp["choices"]:
            # defensive: OpenAI-like shape
            choice = resp["choices"][0]
            if isinstance(choice, dict):
                msg = choice.get("message") or {}
                text = (msg.get("content") if isinstance(msg, dict) else "") or choice.get("text", "")
            else:
                text = str(choice)
        else:
            text = str(resp)
    else:
        # Most dsRAG wrappers return a plain string
        text = str(resp)

    return {
        "content": text.strip(),
        "prompt_tokens": -1,
        "completion_tokens": -1,
        "total_tokens": -1,
    }


def safe_query_kb(question):
    try:
        retrieved = kb.query([question])

        return retrieved if retrieved else []
    except Exception as e:
        raise

def process_question(idx, sample):
    question = sample["question"]
    q_start = time.time()

    try:
        retrieved = safe_query_kb(question)
        

        chunks_used = [chunk["content"] for chunk in retrieved] if retrieved else []
        context = "\n".join(chunks_used) if chunks_used else "No relevant context found."

        prompt = f"""
You are a helpful assistant answering financial questions using retrieved information.
Answer the following question strictly using the retrieved context from 10-K/10-Q filings. 
If the context does not provide enough information, reply exactly: 
'The provided documents do not contain sufficient information to answer this question.' and nothing else


Context:
{context}

Question:
{question}"""

        llm_response = safe_llm_call(prompt)

        if isinstance(llm_response, dict):
            model_answer = llm_response.get("content", "").strip()
            prompt_tokens = llm_response.get("prompt_tokens", -1)
            completion_tokens = llm_response.get("completion_tokens", -1)
            total_tokens = llm_response.get("total_tokens", -1)
        else:
            model_answer = llm_response.strip()
            prompt_tokens = completion_tokens = total_tokens = -1


        time_taken_sec = round(time.time() - q_start, 2)

        result = {
            "FinanceBench ID": sample.get("financebench_id", ""),
            "Company": sample.get("company", ""),
            "Question": question,
            "Ground Truth Answer": sample.get("answer", ""),
            "Model Response": model_answer,
            "Error": ""
        }

        # Store detailed chunk info
        chunk_details_dict[f"Q{idx+1}"] = {
            "Question": question,
            "Ground Truth Answer": sample.get("answer", ""),
            "Model Response": model_answer,
            "Chunks Used": chunks_used
        }

    except Exception as e:
        print(f"[{idx+1}] ❌ Error processing question: {e}")
        result = {
            "FinanceBench ID": sample.get("financebench_id", ""),
            "Company": sample.get("company", ""),
            "Question": question,
            "Ground Truth Answer": sample.get("answer", ""),
            "Model Response": "",
            "Error": str(e)
        }

        chunk_details_dict[f"Q{idx+1}"] = {
            "Question": question,
            "Ground Truth Answer": sample.get("answer", ""),
            "Model Response": "Error",
            "Chunks Used": []
        }

    print(f"[{idx+1}/{len(dataset)}] ✅")
    return result

# ✅ Run the pipeline
pipeline_start = time.time()
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_question, idx, sample) for idx, sample in enumerate(dataset)]
    for future in as_completed(futures):
        results_list.append(future.result())

# ✅ Save result spreadsheet
os.makedirs("Latest_Prompt_Test", exist_ok=True)
result_path = os.path.join("Latest_Prompt_Test", "Antropic_test_Anthropic_Inference_Voyage_Latest_Prompt.xlsx")
df = pd.DataFrame(results_list)
df.to_excel(result_path, index=False)

# ✅ Save per-question chunk details in separate sheets
chunk_excel_path = os.path.join("Latest_Prompt_Test", "Antropic_test_Anthropic_Inference_Voyage_Latest_Prompt_With_chunks.xlsx")

import re

def clean_excel_string(value):
    if isinstance(value, str):
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    return value

with pd.ExcelWriter(chunk_excel_path, engine="openpyxl") as writer:
    for qid, info in chunk_details_dict.items():
        chunk_data = {
            "Field": ["Question", "Ground Truth Answer", "Model Response"] + [f"Chunk {i+1}" for i in range(len(info["Chunks Used"]))],
            "Content": [info["Question"], info["Ground Truth Answer"], info["Model Response"]] + info["Chunks Used"]
        }

        # Clean illegal characters
        chunk_data["Field"] = [clean_excel_string(f) for f in chunk_data["Field"]]
        chunk_data["Content"] = [clean_excel_string(c) for c in chunk_data["Content"]]

        chunk_df = pd.DataFrame(chunk_data)
        chunk_df.to_excel(writer, sheet_name=qid, index=False)


# ✅ Final summary
total_time_sec = round(time.time() - pipeline_start, 2)
print(f"🕒 Total time taken: {total_time_sec} seconds")
print(f"✅ Saved result summary to '{result_path}'")
print(f"✅ Saved chunk details to '{chunk_excel_path}'")
