import pandas as pd
import time
import openai
import os

# 🔐 Use OpenAI API (Set your key here)
openai.api_key = "<api-key"

# 📄 Load the original results
df = pd.read_excel("Latest_Prompt_Test/Antropic_test_Anthropic_Inference_Voyage_Latest_Prompt.xlsx")

# 🧠 Function to check semantic match using GPT-4o
def is_semantic_match(question, ground_truth, prediction):
    prompt = f"""You are a financial QA evaluator. Your task is to determine whether the model-generated answer is semantically equivalent to the ground-truth answer for a financial question.

Use the following criteria to judge:

If the model's answer expresses the same meaning, reasoning, or factual conclusion as the ground-truth—even if the wording or phrasing differs—it should be marked "Yes".

If the model's answer is numerical, treat it as correct if it is within a reasonable margin of error (e.g., ±1% difference) unless the question explicitly requires exact figures.

If the model’s answer omits key details, changes the interpretation, or gives incorrect or irrelevant information, mark it "No".


Question:
{question}

Ground Truth Answer:
{ground_truth}

Model Response:
{prediction}

Does the model response answer the question correctly and semantically match the ground truth?"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial QA evaluator."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response["choices"][0]["message"]["content"].strip().lower()
        return 1 if "yes" in answer else 0
    except Exception as e:
        print(f"❌ Error for Q: '{question[:60]}...': {e}")
        return -1

# 🧮 Run evaluation for each row
results = []
for i, row in df.iterrows():
    q = str(row["Question"])
    gt = str(row["Ground Truth Answer"])
    pred = str(row["Model Response"])
    verified = is_semantic_match(q, gt, pred)
    results.append(verified)
    print(f"[{i+1}/{len(df)}] ✅ Verified = {verified}")
    time.sleep(1.0)  # optional: helps avoid OpenAI rate limits

# 📝 Save results
df["LLM Verified"] = results

os.makedirs("Latest_Prompt_Results", exist_ok=True)
output_path = "Latest_Prompt_Results/Semantic_Results_Antropic_test_Anthropic_Inference_Voyage_WithoutGPU_Usage_Correct_Prompt.xlsx"
df.to_excel(output_path, index=False)

print(f"\n✅ Evaluation complete. Results saved to '{output_path}'")
