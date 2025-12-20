This is a project based on testing and evaluation of Tiny Language Models(TLMs) to generate context specific answers ensuring factual grounding on a educational synthetically generated JSONL datset that is usually developed from teacher authored/written class notes from four subjects namely AI, Python, Operating Systems ,OOPs. To ensure Factual awareness for the generated answers, the models were incorporated with RAG in subsequent experiments.

- The Context Specific Output Generation Pipeline Operated in 4 Phases
<img width="1054" height="580" alt="image" src="https://github.com/user-attachments/assets/c9e46b14-6bc9-453f-bfc1-33160255b599" />

## Workline Summary:
- It started with 6 baseline models, namely TinyLlama-1.1B-Chat-v1.0, flan-t5-small, SmolLM2-135M, SmolLM2-360M-Instruct4, SmolLM2-1.7B-Instruct, and Qwen2.5-1.5B-Instruct. Out of which , Qwen-2.5-1.5B-Instruct outperformed all others and was chosen for further experimentation.
The chosen model has been fine-tuned using a synthetically prepared domain-specific dataset based on classroom notes written by teachers and then transformed into structured JSON-based question-answer pairs. The findings indicate that parameter-efficient fine-tuning results in a significant enhancement of semantic alignment and pedagogical coherence within the established hardware resource constraints. 
- Fine-tuning was done using the QLoRA methodology, which combines low-rank adaptation with quantized model weights, significantly reducing memory consumption. The PEFT Library has been used to attach trainable LoRA adapters to the frozen base model, while the bitsandbytes library was used to enable 4-bit quantization during training. This setup allowed for efficient fine-tuning within limited hardware resources-specifically, one NVIDIA GeForce RTX 3050 GPU with 6 GB of VRAM.
- The fine-tuned Qwen-2.5-1.5B-Instruct model yields a remarkably improved semantic accuracy of 87.7% on average in the evaluation dataset. This clearly suggests that domain-specific fine-tuning improves the model's ability to generate semantically consistent responses for scholarly questions. Similarly, improved BLEU and ROUGE-L scores indicate a better structural resemblance to reference answers, although these metrics remain quite modest due to the generative and paraphrasing nature of instruction-tuned outputs.
- However, the conducted experiments also demonstrated that fine-tuning itself is not sufficient in ensuring factual grounding while responding to academic content-specific queries.
- To address this limitation, RAG was incorporated into the system to allow it to dynamically access instructional material at inference time. The hybrid configuration that combines fine-tuning with retrieval exhibits balanced improvements across semantic accuracy, response fluency, and factual consistency, while sustaining acceptable inference latency for educational use cases.
The hybrid model shows an overall good performance, with a semantic accuracy of 81.8%, better than the RAG-only configuration and competitive with fine-tuning alone. This also indicates that retrieval augmentation is serving its purpose of complementing fine-tuning by enhancing contextual relevance while retaining coherent instructional responses. The improvements in BLEU and ROUGE-L scores further indicate an increase in structural alignment with reference answers compared to the base RAG setup.

## The instructional prompt used:
- prompt = f"""
    You are a serious knowledgeable teaching assistant, your job is to answer every single question asked by a student using the provided context.
    Here, is a strict follow up, You can never skip any question.
    I repeat, not a single question should be left unanswered.
    Always provide a clear, concise, detail explanation like a teacher and provide a clear cum easy explanation to the student's question everytime you were asked, 
    and answer always with the context below only:
    - Context:{context}
    - Question: {question}
    - Answer:  
    """

This work, therefore, substantiates that TLMs can effectively support education question answering with proper selection and augmentation by fine-tuning and retrieval mechanisms without depending on large-scale language models. The results reveal the practical viability of TLM-based systems in classroom applications and form a starting point for further research in scalable low-resource educational assistants.

## Frontend Integration for Educational Access:

Frontend integration make the platform usable and accessible in actual educational environments by presenting the model outputs in a way that makes it easy for students to interact with the question-answering process. This assisted in bridging the gap between backend model experimentation and actual educational usage, ensuring that students can effectively access the generated responses for learning and revision.



## Folder Structure:

- /dataset/evaluation_datasetUnique.jsonl - It contains the JSONL dataset having 253 objects ensuring 40% factual, 40% conceptual, 20% reasoning based questions, across all four subjects to ensure academic fairness while traing the models.
- /qwen_baseModel_Testing/evaluation_qwenBase_model.py - This is the code base file, which generally has how the base model was trained. It only has the qwen2.5-1.5B-Instruct model as it was the best among all trained and selected for finetuning and further experimentation.
- /qwen_finetunedOnly_testing/evaluate_qwen_finetuned.py - This is the code base file, for the evaluation of finetuned model.
- /rag+qwenFinetuned_testing/evaluate_qwen_hybrid_rag.py - This is the code base file, where hybrid approach was used i.e., incorporating RAG with finetuned Qwen model.
- /rag_testing/ragMetrics_testing.py - This is the code base file, where the Qwen base variant model integrated with RAG was used to for the evaluation.




