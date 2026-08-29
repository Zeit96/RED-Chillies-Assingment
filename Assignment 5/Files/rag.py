"""
Simple RAG (Retrieval-Augmented Generation) Application
==========================================================

WHAT THIS SCRIPT DOES (in order — matches the assignment's 7 steps):
1. Load documents        -> load every .txt / .md / .pdf file from sample_docs/
2. Split into chunks     -> break long documents into smaller overlapping pieces
3. Generate embeddings   -> turn each chunk into a vector using a local model
                             (no API key needed — runs on your own machine)
4. Retrieve chunks       -> for a user question, find the most similar chunks
                             using cosine similarity
5. Send to an LLM        -> pass the question + retrieved chunks to a local
                             LLM served by Ollama (also free, no API key)
6. Display the answer    -> print the generated answer
7. Show sources          -> print which document + chunk the answer came from

WHY THESE TOOLS:
- sentence-transformers (all-MiniLM-L6-v2): a small, fast, free, local
  embedding model. No internet API calls needed after the first download.
- Ollama: runs an open-source LLM (e.g. llama3.2) locally for free.
  You must install it separately - see README.md.
- No FAISS/Chroma vector database: for a small sample document set, plain
  numpy cosine similarity is simpler to understand and explain, and is
  fast enough. (Mention in your writeup that a real production RAG app
  would swap this for a vector database like Chroma/FAISS/Pinecone as
  the document collection grows.)
"""

import os
import glob
import json
import textwrap
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DOCS_FOLDER = "sample_docs"
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between consecutive chunks
TOP_K = 3                 # how many chunks to retrieve per question
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"           # change to any model you've pulled with `ollama pull`
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------------------------
# STEP 1: LOAD DOCUMENTS
# ---------------------------------------------------------------------------
def load_documents(folder):
    """Read every .txt, .md, and .pdf file in `folder` and return a list of
    dicts: {"source": filename, "text": full_text}"""
    documents = []
    filepaths = glob.glob(os.path.join(folder, "*"))

    for path in filepaths:
        filename = os.path.basename(path)

        if path.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif path.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue  # skip unsupported file types

        documents.append({"source": filename, "text": text})

    return documents


# ---------------------------------------------------------------------------
# STEP 2: SPLIT INTO CHUNKS
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of `chunk_size` characters.
    Overlap helps avoid cutting an idea in half between two chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_chunk_store(documents):
    """Turn all documents into a flat list of chunks, each tagged with its
    source document so we can cite it later."""
    chunk_store = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunk_store.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk,
            })
    return chunk_store


# ---------------------------------------------------------------------------
# STEP 3: GENERATE + STORE EMBEDDINGS
# ---------------------------------------------------------------------------
def embed_chunks(chunk_store, model):
    texts = [c["text"] for c in chunk_store]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


# ---------------------------------------------------------------------------
# STEP 4: RETRIEVE RELEVANT CHUNKS FOR A QUESTION
# ---------------------------------------------------------------------------
def retrieve(question, model, chunk_store, chunk_embeddings, top_k=TOP_K):
    query_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Since embeddings are normalized, dot product == cosine similarity
    scores = chunk_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            **chunk_store[idx],
            "score": float(scores[idx]),
        })
    return results


# ---------------------------------------------------------------------------
# STEP 5: SEND RETRIEVED CONTEXT TO A LOCAL LLM (via Ollama)
# ---------------------------------------------------------------------------
def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        f"[Source: {c['source']} - chunk {c['chunk_id']}]\n{c['text']}"
        for c in retrieved_chunks
    )
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the
context provided below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def ask_llm(prompt):
    """Call a local Ollama server. Requires Ollama to be installed and
    running, and the model to have been pulled (see README.md)."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return ("[ERROR] Could not connect to Ollama at localhost:11434.\n"
                "Make sure Ollama is installed and running (`ollama serve`),\n"
                f"and that you've pulled the model with: ollama pull {OLLAMA_MODEL}")
    except Exception as e:
        return f"[ERROR] {e}"


# ---------------------------------------------------------------------------
# STEPS 6 & 7: DISPLAY ANSWER + SOURCES
# ---------------------------------------------------------------------------
def main():
    print("Loading documents...")
    documents = load_documents(DOCS_FOLDER)
    print(f"  Loaded {len(documents)} document(s): {[d['source'] for d in documents]}")

    print("Splitting into chunks...")
    chunk_store = build_chunk_store(documents)
    print(f"  Created {len(chunk_store)} chunks.")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings for all chunks...")
    chunk_embeddings = embed_chunks(chunk_store, model)

    print("\nReady! Ask questions about the documents in sample_docs/.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        retrieved = retrieve(question, model, chunk_store, chunk_embeddings)
        prompt = build_prompt(question, retrieved)
        answer = ask_llm(prompt)

        print("\n--- ANSWER ---")
        print(textwrap.fill(answer, width=90))

        print("\n--- SOURCES USED ---")
        for c in retrieved:
            preview = c["text"][:120].replace("\n", " ")
            print(f"  [{c['source']} | chunk {c['chunk_id']} | score={c['score']:.3f}] {preview}...")
        print()


if __name__ == "__main__":
    main()
