import os
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import PyPDF2
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

app = FastAPI(title="Knowledge Base Search Engine")

# ---- Enable CORS for frontend ----
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # use ["*"] for dev, specify frontend URL for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Embeddings model ----
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384

# ---- FAISS index & document storage ----
if os.path.exists("faiss_index.index") and os.path.exists("doc_texts.pkl") and os.path.exists("doc_meta.pkl"):
    index = faiss.read_index("faiss_index.index")
    with open("doc_texts.pkl", "rb") as f:
        doc_texts = pickle.load(f)
    with open("doc_meta.pkl", "rb") as f:
        doc_meta = pickle.load(f)
else:
    index = faiss.IndexFlatL2(embedding_dim)
    doc_texts = []
    doc_meta = []

# ---- Local LLM (CPU) ----
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

# ---- Upload documents ----
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    count = 0
    uploaded_names = []
    for file in files:
        content = ""
        try:
            if file.filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(file.file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            else:
                content = (await file.read()).decode()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read {file.filename}: {str(e)}"}

        # Split content into sentence chunks with overlap
        sentences = sent_tokenize(content)
        chunk_size = 5
        overlap = 1
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = " ".join(sentences[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        for idx, chunk in enumerate(chunks):
            vec = embed_model.encode([chunk], convert_to_numpy=True).astype('float32')
            index.add(vec)
            doc_texts.append(chunk)
            doc_meta.append({"file": file.filename, "chunk_id": idx})

        count += 1
        uploaded_names.append(file.filename)

    # Save index and docs
    faiss.write_index(index, "faiss_index.index")
    with open("doc_texts.pkl", "wb") as f:
        pickle.dump(doc_texts, f)
    with open("doc_meta.pkl", "wb") as f:
        pickle.dump(doc_meta, f)

    return {"status": "success", "documents_added": count, "uploaded_files": uploaded_names}

# ---- Query endpoint ----
@app.get("/query")
def query(q: str, top_k: int = 3):
    if len(doc_texts) == 0:
        return {"answer": "No documents uploaded yet.", "retrieved_docs": []}

    # Encode query
    query_vec = embed_model.encode([q], convert_to_numpy=True).astype('float32')
    D, I = index.search(query_vec, top_k)

    retrieved = []
    for i, score in zip(I[0], D[0]):
        if i < len(doc_texts):
            text = re.sub(r'\s+', ' ', doc_texts[i]).strip()
            retrieved.append({
                "text": text,
                "file": doc_meta[i]["file"],
                "chunk_id": doc_meta[i]["chunk_id"],
                "score": float(score)
            })

    # Merge retrieved chunks for LLM context
    context_text = "\n".join([f"[{r['file']}] {r['text']}" for r in retrieved])

    prompt = f"Using the following context from documents, answer the question succinctly:\n\n{context_text}\n\nQuestion: {q}\nAnswer:"

    answer = qa_pipeline(prompt, max_new_tokens=150)[0]['generated_text'].strip()

    return {"answer": answer, "retrieved_docs": retrieved}

# ---- List all uploaded files ----
@app.get("/files")
def list_files():
    unique_files = list({meta["file"] for meta in doc_meta})
    return {"files": unique_files}

@app.post("/delete-file")
def delete_file(payload: dict):
    fname = payload.get("filename")
    if not fname:
        return {"status": "error", "message": "No filename provided."}

    global doc_texts, doc_meta, index

    # Find indices of chunks belonging to the file
    indices_to_keep = [i for i, meta in enumerate(doc_meta) if meta["file"] != fname]

    # Filter doc_texts and doc_meta
    doc_texts = [doc_texts[i] for i in indices_to_keep]
    doc_meta = [doc_meta[i] for i in indices_to_keep]

    # Rebuild FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    if doc_texts:
        import numpy as np
        vectors = np.array([embed_model.encode([text], convert_to_numpy=True).astype('float32')[0] for text in doc_texts])
        index.add(vectors)

    # Save updated index and pickles
    faiss.write_index(index, "faiss_index.index")
    with open("doc_texts.pkl", "wb") as f:
        pickle.dump(doc_texts, f)
    with open("doc_meta.pkl", "wb") as f:
        pickle.dump(doc_meta, f)

    return {"status": "success", "message": f"{fname} deleted successfully."}
