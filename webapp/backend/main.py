from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from models import TwoTowerModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import json
from gensim.models import KeyedVectors
import gensim.downloader as api
import faiss

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_QUERY_LEN = 26
MAX_ANSWER_LEN = 201
hidden_size_query = 125
hidden_size_answer = 125
word2vec = api.load('word2vec-google-news-300')

# word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
# Get the vocabulary and embeddings
vocab_size = len(word2vec)
embedding_dim = word2vec.vector_size
weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

# Create a PyTorch Embedding layer
embedding_layer = nn.Embedding.from_pretrained(weights, freeze=False)  # Set freeze=True if you don't want to fine-tune

word2index = {word: i for i, word in enumerate(word2vec.index_to_key)}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TwoTowerModel(MAX_QUERY_LEN, MAX_ANSWER_LEN, hidden_size_query, hidden_size_answer).to(device)
model.load_state_dict(torch.load('models/checkpoint_epoch_6.pt')['model_state_dict'])
model.eval()
answer_model = model.answer_tower




def word_to_tensor(word):
    """Convert a word into a tensor index for the embedding layer"""
    if word in word2index:
        return torch.tensor([word2index[word]], dtype=torch.long)
    else:
        return torch.tensor([word2index["unk"]], dtype=torch.long)
    

def preprocess_answer(query, answer_model, max_length):
    query_words = query.split()
    query_tensor = torch.cat([word_to_tensor(word) for word in query_words])
    print(query_tensor.shape)
    answer_length = len(query_words)
    
    padded_answer_indices = torch.stack([torch.nn.functional.pad(
        query_tensor[:answer_length], 
        (0, max_length - answer_length)
    )]).long()
    print(padded_answer_indices.shape)
    padded_answers = embedding_layer(padded_answer_indices)
    print(padded_answers.shape)
    answer_length = torch.tensor([answer_length], dtype=torch.long)
    print(answer_length.shape)
    answer_embeddings = answer_model(padded_answers, answer_length)
    print(answer_embeddings.shape)
    return answer_embeddings

with open("data/answer_embeddings.json", "r") as f:
    document_mapping = json.load(f)

index = faiss.read_index("data/answer_embeddings.faiss")

class TextInput(BaseModel):
    text: str

@app.post("/submit")
async def submit_text(input_data: TextInput, k: int = 10):
    print("kk")
    try:
        # Prepare the input text
        print("loool")
        print(input_data.text)
        # tokens = trainer.prepare_data_minimal(input_data.text, word_to_id, tokenizer)
        query_embeddings = preprocess_answer(input_data.text, answer_model, MAX_QUERY_LEN).unsqueeze(0)
        print(query_embeddings.shape, "blop")
        
        # if not tokens:
        distances, indices = index.search(query_embeddings.cpu().detach().numpy(), k)
        results = ""
        for i in range(k):
            idx = str(indices[0][i])  # Convert index to string since JSON keys are strings
            print(f"Index {idx}: {document_mapping[idx]}")
            print(f"Distance: {distances[0][i]}\n")
            results += f"{document_mapping[idx]} {distances[0][i]}\n"
        #     return {"error": "No valid tokens found in input"}
            
        # Find similar words for each token
        documents = []
        for i in range(k):
            documents.append({
                "document": str(document_mapping[str(indices[0][i])]),
                "distance": str(distances[0][i]),
                "results": results
            })
        
        return {"results": documents}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "API is running"} 