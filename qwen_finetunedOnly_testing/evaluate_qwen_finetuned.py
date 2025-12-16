import json, time, numpy as np, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "outputs/qwen_qlora"
DATA_PATH = "dataset/evaluation_datasetUnique.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.7

print(f"\nEvaluating Fine-Tuned Model: {ADAPTER_PATH}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(DEVICE)
model.eval()

sim_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

print(f"Load {len(dataset)} samples.\n")

accuracies, bleu_scores, rouge_scores, latencies, perplexities = [], [], [], [], []

for data in tqdm(dataset):
    q, ctx, ref = data["question"], data["context"], data["answer"].strip().lower()
    prompt = f"""You are a serious knowledgeable teaching assistant, your job is to answer every single question asked by a student using the provided context.
    Here, is a strict follow up, You can never skip any question.
    I repeat, not a single question should be left unanswered.
    Always provide a clear, concise, detail explanation like a teacher and provide a clear cum easy explanation to the student's question everytime you were asked, 
    and answer always with the context below only:

    Context:{ctx}
    Question: {q}
    Answer:  
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    latency = time.time() - start

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in resp:
        resp = resp.split("Answer:")[-1]
    resp = resp.strip().lower()

    emb_ref = sim_model.encode(ref, convert_to_tensor=True)
    emb_resp = sim_model.encode(resp, convert_to_tensor=True)
    sim = util.cos_sim(emb_ref, emb_resp).item()
    acc = 1 if sim >= THRESHOLD else 0

    bleu = sentence_bleu([ref.split()], resp.split())
    rouge_score = rouge.score(ref, resp)["rougeL"].fmeasure
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
        ppl = torch.exp(out.loss).item()

    accuracies.append(acc)
    bleu_scores.append(bleu)
    rouge_scores.append(rouge_score)
    latencies.append(latency)
    perplexities.append(ppl)

results = {
    "model": f"{BASE_MODEL} (Fine-tuned)",
    "semantic_accuracy": float(np.mean(accuracies) * 100),
    "latency": float(np.mean(latencies)),
    "bleu": float(np.mean(bleu_scores)),
    "rougeL": float(np.mean(rouge_scores)),
    "perplexity": float(np.mean(perplexities))
}

print("\nBase Qwen Results:")
for k, v in results.items():
    if isinstance(v, (float, int)):
        print(f"{k:20}: {v:.3f}")
    else:
        print(f"{k:20}: {v}")

with open("results_qwen_finetuned.json", "w") as f:
    json.dump(results, f, indent=4)
print("\nSaved as results_qwen_finetuned.json")
