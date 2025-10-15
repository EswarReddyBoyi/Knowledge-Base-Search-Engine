# Knowledge Base Search Engine

A web application to upload documents (PDF or text) and ask questions using a local LLM with semantic search. The system uses embeddings (SentenceTransformers) and FAISS for document retrieval, and a text-to-text generation model (Flan-T5-small) for answering queries.

---

## Features

- Upload multiple files (PDF or text) and store them with chunked embeddings.
- Search across uploaded documents using semantic similarity.
- Answer user queries using a local LLM with relevant document context.
- Copy or download the generated answer.
- List and delete previously uploaded documents.
- Responsive frontend compatible with desktop and mobile.

---

## Tech Stack

**Backend:**

- Python 3.12+
- [FastAPI](https://fastapi.tiangolo.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [NLTK](https://www.nltk.org/)
- PyPDF2 (PDF parsing)
- Pickle (document persistence)

**Frontend:**

- HTML, CSS, JavaScript
- Simple, responsive design

**Deployment:**

- Backend: Railway (Free tier) / Render (requires >512MB RAM)
- Frontend: Netlify / Vercel

---

## Installation

1. Clone the repository:

`````
git clone https://github.com/EswarReddyBoyi/knowledge-base-search.git
cd knowledge-base-search
``````

2. Create a virtual environment:
````
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
````

3. Install dependencies:
````
pip install --upgrade pip
pip install -r requirements.txt
````

***Run Locally***

Start the FastAPI backend:
````
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
````
------------------
Open index.html in a browser or deploy frontend separately.

Access the frontend at http://127.0.0.1:5500 (if using Live Server in VSCode).

-------------------
### Usage

Upload Documents: Select multiple files (PDF or text) and click Upload.

Ask a Question: Type your query, select the number of top results, and click Search.

Copy/Download Answer: Use the buttons to copy or download the generated answer.

Manage Documents: List uploaded files and delete any unwanted files.

-----------------

### Deployment Guide
***Backend***

Railway: 1 GB free RAM is recommended for SentenceTransformers + Flan-T5-small.

Render: Free tier has 512MB RAM → may crash with LLM in memory.

Ensure the backend port is exposed (Railway automatically detects PORT).

***Frontend***

Netlify or Vercel: Static deployment for index.html, main.js, and style.css.

Update the backend URL in main.js to the deployed backend.

---------------------

***Notes***

For large document sets, ensure enough RAM for embeddings and FAISS index.

Models are loaded into memory; consider lazy loading to reduce startup memory.

PDF text extraction may not work for scanned documents (requires OCR).

