import json
import time
import torch
import evaluate
import numpy as np
from math import exp
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
dataset_path = "dataset/evaluation_datasetUnique.jsonl"
max_new_tokens = 128

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

#model + tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto",load_in_8bit=True, trust_remote_code=True
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load JSONL dataset
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = load_jsonl(dataset_path)


bleu_scores, rouge_scores, acc_scores, latencies, perplexities = [], [], [], [], []

#perplexity
def perplexity_score(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return exp(loss.item())

#sem_accuracy
def accuracy_score(reference, prediction):
    emb1 = embedder.encode(reference, convert_to_tensor=True)
    emb2 = embedder.encode(prediction, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()


for sample in tqdm(data, desc="Evaluating Qwen-RAG"):
    context = sample["context"]
    question = sample["question"]
    ground_truth = sample["answer"]

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

    bleu_val = bleu.compute(predictions=[prediction], references=[ground_truth])["bleu"]
    rouge_val = rouge.compute(predictions=[prediction], references=[ground_truth])["rougeL"]
    acc_val = accuracy_score(ground_truth, prediction)
    ppl_val = perplexity_score(model, tokenizer, prediction)

    bleu_scores.append(bleu_val)
    rouge_scores.append(rouge_val)
    acc_scores.append(acc_val)
    latencies.append(latency)
    perplexities.append(ppl_val)

print("\nQwen RAG Evaluation ==========")
print(f"Semantic Accuracy: {np.mean(acc_scores):.4f}")
print(f"BLEU: {np.mean(bleu_scores):.4f}")
print(f"ROUGE-L: {np.mean(rouge_scores):.4f}")
print(f"Avg Perplexity: {np.mean(perplexities):.2f}")
print(f"Avg Latency: {np.mean(latencies):.2f} sec")