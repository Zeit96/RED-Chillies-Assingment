# Simple RAG Application

A small Retrieval-Augmented Generation (RAG) app that answers questions using
information retrieved from a local document collection. Runs entirely for
free, with no API keys — embeddings and the LLM both run on your own machine.

## How it works (my approach)

1. **Load documents** — `load_documents()` reads every `.txt`, `.md`, and
   `.pdf` file from `sample_docs/`.
2. **Chunk** — `chunk_text()` splits each document into ~500-character
   pieces with 100 characters of overlap, so an idea split across a chunk
   boundary isn't lost.
3. **Embed** — each chunk is converted into a vector using the
   `all-MiniLM-L6-v2` model from `sentence-transformers`. This is a small,
   free, local embedding model (no API calls, downloads once from
   Hugging Face the first time you run it).
4. **Retrieve** — when you ask a question, it's embedded the same way, and
   compared against every chunk's embedding using cosine similarity
   (implemented as a dot product on normalized vectors, with numpy). The
   top 3 most similar chunks are selected.
5. **Generate** — the question + retrieved chunks are combined into a
   prompt and sent to a local LLM served by **Ollama** (default model:
   `llama3.2`), which also runs for free with no API key.
6. **Display answer** — the model's response is printed to the terminal.
7. **Show sources** — for every answer, the app prints which document and
   chunk number the retrieved context came from, along with a similarity
   score, so you can verify the answer is grounded in the source material.

### Why these specific tools
- **No vector database (FAISS/Chroma):** for a handful of sample documents,
  plain numpy cosine similarity is simple, transparent, and fast enough.
  In a real/production app with thousands of documents, I'd swap this for
  a proper vector database for faster search at scale.
- **Ollama instead of an API-based LLM (OpenAI/Claude/etc.):** avoids
  needing an API key or paying per request, which fits a free local setup.
  The code can be pointed at a paid API instead by changing the `ask_llm()`
  function if you have API access.

## Setup Instructions

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (the local LLM engine)
Download and install from **https://ollama.com/download** for your OS
(Windows/Mac/Linux). This is a normal installer — no coding needed.

### 3. Pull a model
Open a terminal and run:
```bash
ollama pull llama3.2
```
This downloads the model (a few GB) — it happens once.

### 4. Start Ollama
```bash
ollama serve
```
Leave this running in its own terminal window. (On Mac/Windows, the
Ollama app usually starts this automatically in the background — if so
you can skip this step.)

### 5. Run the RAG app
In a **new** terminal window, from this project folder:
```bash
python rag.py
```

You'll see it load documents, chunk them, build embeddings, and then
prompt you:
```
Your question: What is the home office stipend amount?
```

Type `exit` to quit.

## Using your own documents
Replace or add files inside `sample_docs/` — `.txt`, `.md`, and `.pdf` are
all supported. No code changes needed; the app automatically picks up
every file in that folder.

## Sample documents included
- `remote_work_policy.md` — a company remote work policy
- `product_faq.md` — FAQ for a fictional cloud backup product
- `onboarding_guide.md` — a new-employee onboarding guide

Try questions like:
- "How much is the home office stipend?"
- "How long is the free trial for deleted file recovery?"
- "When can I start using my PTO?"
