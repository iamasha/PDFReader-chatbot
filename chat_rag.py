from indexer import FaissIndex
from pdf_loader import extract_text_from_pdf, chunk_text
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os


os.environ["TRANSFORMERS_OFFLINE"] = "1"
LLM_MODEL = os.environ.get("LLM_MODEL", "google/flan-t5-base")


def build_index_from_pdf(pdf_path,index_path="faiss.index",meta_path="meta.pk1"):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text,chunk_size=500,overlap=100)
    idx = FaissIndex()
    idx.add(chunks)
    idx.save(index_path,meta_path)
    print("index built. chunks:",len(chunks))
    return idx

def load_index(index_path="faiss_index",meta_path="meta.pk1"):
    idx=FaissIndex()
    idx.load(index_path,meta_path)
    return idx

#simple generator function using HF casual model
def generate_answer(context_chunks,question,max_new_tokens=256):
    #compose the prompt
    question = question.replace("Question:", "").strip()
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"if you are a helpful assistant. Use the following extracted document snippets to answer the question. Only use the information in the snippets; if answer is not contained, say you dont know.\n\nContext:\n\n{context}\n\nQuestion:{question}\nAnswer:"
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.eval()
    input_ids = tokenizer(prompt,return_tensors="pt").input_ids
    with torch.no_grad():
        # outputs = model.generate(
        #             input_ids,
        #             max_new_tokens=100,  # ✅ limit to 100 for quick response
        #             do_sample=False
        #         )
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)

        # outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    answer = tokenizer.decode(outputs[0],skip_special_tokens=True)

    return answer.strip()

def answer_query(index,question,top_k=4):
    chunks = index.query(question,top_k=top_k)
    chunks = chunks[:2]
    answer = generate_answer(chunks,question)
    return answer

