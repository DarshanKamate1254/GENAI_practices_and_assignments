import gc
import ctypes
from time import time
import requests
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# CONFIG
# -----------------------------

NUM_TITLES = 5
MAX_SEQ_LEN = 512

MODEL_PATH = "models/bge-small-en"
FAISS_INDEX_PATH = "vectordb/faiss.index"
WIKI_DATASET_PATH = "data/wiki_dataset"
TEST_DATA = "data/test.csv"

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "llama-3"


# -----------------------------
# MEMORY CLEAN
# -----------------------------

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass


# -----------------------------
# LOAD TEST DATA
# -----------------------------

df = pd.read_csv(TEST_DATA, index_col="id")


# -----------------------------
# SENTENCE TRANSFORMER
# -----------------------------

class SentenceTransformer:

    def __init__(self, checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def transform(self, batch):

        tokens = self.tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN
        )

        return tokens.to(self.device)

    def get_dataloader(self, sentences):

        sentences = [
            "Represent this sentence for searching relevant passages: " + s
            for s in sentences
        ]

        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)

        return DataLoader(dataset, batch_size=16)

    def encode(self, sentences):

        dataloader = self.get_dataloader(sentences)

        embeddings = []

        for batch in dataloader:

            with torch.no_grad():

                e = self.model(**batch).pooler_output
                e = F.normalize(e, p=2, dim=1)

                embeddings.append(e.cpu().numpy())

        return np.concatenate(embeddings, axis=0)


# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

print("Loading embedding model...")

start = time()

model = SentenceTransformer(MODEL_PATH)

print(f"Model loaded in {time()-start:.1f}s")


# -----------------------------
# EMBED QUESTIONS
# -----------------------------

print("Embedding prompts...")

f = lambda row: " ".join([
    row["prompt"],
    row["A"],
    row["B"],
    row["C"],
    row["D"],
    row["E"]
])

inputs = df.apply(f, axis=1).values

prompt_embeddings = model.encode(inputs)


# -----------------------------
# LOAD FAISS INDEX
# -----------------------------

print("Loading FAISS index...")

faiss_index = faiss.read_index(FAISS_INDEX_PATH)


# -----------------------------
# VECTOR SEARCH
# -----------------------------

print("Searching knowledge base...")

search_index = faiss_index.search(
    np.float32(prompt_embeddings),
    NUM_TITLES
)[1]


# -----------------------------
# LOAD WIKI DATASET
# -----------------------------

print("Loading wiki dataset...")

dataset = load_from_disk(WIKI_DATASET_PATH)


# -----------------------------
# EXTRACT CONTEXT
# -----------------------------

print("Extracting contexts...")

for i in range(len(df)):

    df.loc[i, "context"] = "- " + "\n- ".join(
        [dataset[int(j)]["text"] for j in search_index[i]]
    )


clean_memory()


# -----------------------------
# LM STUDIO CALL
# -----------------------------

def ask_llm(prompt):

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post(LMSTUDIO_URL, json=payload)

    return response.json()["choices"][0]["message"]["content"]


# -----------------------------
# ANSWER EVALUATION
# -----------------------------

predictions = []

print("Running LLM reasoning...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    context = row["context"]

    prompt = f"""
Below is an instruction that describes a task.

Context:
{context}

Question:
{row['prompt']}

Options:
A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}
E. {row['E']}

Return ONLY the correct option letter.
"""

    answer = ask_llm(prompt).strip()

    predictions.append(answer)


df["prediction"] = predictions


# -----------------------------
# SAVE RESULTS
# -----------------------------

df[["prediction"]].to_csv("submission.csv")

print("Submission saved: submission.csv")