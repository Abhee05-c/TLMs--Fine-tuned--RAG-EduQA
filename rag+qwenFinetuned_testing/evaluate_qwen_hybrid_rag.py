import json
import time
import torch
import evaluate
import numpy as np
from math import exp
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "outputs/qwen_qlora"   #finetuned QLoRA model 
dataset_path = "dataset/evaluation_datasetUnique.jsonl"
max_new_tokens = 300
output_csv = "rag+qwenFinetuned_testing/Evaluation_result_per_question/evaluation_resultsUnique5.csv"


# Load evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Running on: {device.upper()}")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load base model + tokenizer
print("🔹 Loading base Qwen model in 4-bit mode...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("🔹 Attaching fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load JSONL dataset
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = load_jsonl(dataset_path)
print(f"Loaded {len(data)} samples for evaluation.\n")

records = []

#Perplexity
def perplexity_score(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return exp(loss.item())

#sem_cosine
def accuracy_score(reference, prediction):
    emb1 = embedder.encode(reference, convert_to_tensor=True)
    emb2 = embedder.encode(prediction, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

for i, sample in enumerate(tqdm(data, desc="Evaluating Qwen Hybrid (RAG + LoRA)")):
    context = sample["context"]
    question = sample["question"]
    ground_truth =sample["answer"]

    
    prompt = f"""You are a serious knowledgeable teaching assistant, your job is to answer every single question asked by a student using the provided context.
    Here, is a strict follow up, You can never skip any question.
    I repeat, not a single question should be left unanswered.
    Always provide a clear, concise, detail explanation like a teacher and provide a clear cum easy explanation to the student's question everytime you were asked, 
    and answer always with the context below only:

    Context:{context}
    Question: {question}
    Answer:  
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    latency = time.time() - start_time

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = prediction.split("Answer:")[-1].strip()

    if not prediction.strip():
        bleu_val, rouge_val, acc_val, ppl_val = 0.0, 0.0, 0.0, np.nan
    else:
        try:
            bleu_val = bleu.compute(predictions=[prediction], references=[ground_truth])["bleu"]
        except ZeroDivisionError:
            bleu_val = 0.0

        try:
            rouge_val = rouge.compute(predictions=[prediction], references=[ground_truth])["rougeL"]
        except Exception:
            rouge_val = 0.0

        acc_val = accuracy_score(ground_truth, prediction)
        ppl_val = perplexity_score(model, tokenizer, prediction)

    records.append({
        "index": i + 1,
        "question": question,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "semantic_accuracy": acc_val,
        "bleu": bleu_val,
        "rougeL": rouge_val,
        "perplexity": ppl_val,
        "latency_sec": latency
    })

df = pd.DataFrame(records)
print("\nQwen Hybrid (Fine-Tuned + RAG) Evaluation:")
print(f"Semantic Accuracy: {df['semantic_accuracy'].mean():.4f}")
print(f"BLEU: {df['bleu'].mean():.4f}")
print(f"ROUGE-L: {df['rougeL'].mean():.4f}")
print(f"Avg Perplexity: {df['perplexity'].mean():.2f}")
print(f"Avg Latency: {df['latency_sec'].mean():.2f} sec")


df.to_csv(output_csv, index=False)
print(f"saved: {output_csv}")
