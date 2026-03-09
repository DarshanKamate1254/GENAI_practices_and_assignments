import os
from dotenv import load_dotenv

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# DOCUMENTS
# -----------------------------

DOCUMENT1 = """Operating the Climate Control System.
Your Googlecar has a climate control system that allows you to adjust temperature and airflow."""

DOCUMENT2 = """Your Googlecar has a touchscreen display for navigation, entertainment, and music.
Tap the Music icon to play songs."""

DOCUMENT3 = """Shifting Gears.
Your Googlecar has an automatic transmission with Park, Reverse, Neutral, and Drive."""

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

# -----------------------------
# EMBEDDING MODEL
# -----------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = embedding_model.encode(input).tolist()
        return embeddings


# -----------------------------
# CHROMADB SETUP
# -----------------------------

DB_NAME = "groq_rag_db_1"

embed_fn = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.Client()

db = chroma_client.get_or_create_collection(
    name=DB_NAME,
    embedding_function=embed_fn
)

db.add(
    documents=documents,
    ids=[str(i) for i in range(len(documents))]
)

print("Documents in DB:", db.count())

# -----------------------------
# QUERY
# -----------------------------

query = "How to shift gears?"

result = db.query(
    query_texts=[query],
    n_results=1
)

[retrieved_docs] = result["documents"]

# -----------------------------
# PROMPT BUILDING
# -----------------------------

prompt = f"""
You are a helpful assistant that answers questions using the reference passage.

QUESTION: {query}
"""

for passage in retrieved_docs:
    prompt += f"\nPASSAGE: {passage}"

# -----------------------------
# GROQ LLM
# -----------------------------

response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.2,
)

answer = response.choices[0].message.content

print("\nAnswer:\n")
print(answer)