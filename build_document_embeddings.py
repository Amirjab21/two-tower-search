import torch
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import torch.nn as nn

from models import QADataset, TwoTowerModel
import tqdm
import faiss
import json
import wandb
from pathlib import Path
#We need to process all the data and then save the embeddings to faiss
# IMPORTANT, you must add answer to the Dataset class for this to work:
        # return {
        #     'query': query_tensor,
        #     'answer': answer_tensor,
        #     'query_length': len(query_words),
        #     'answer_length': len(answer_words),
        #     "original_answer": answer
        # }

df_val = pd.read_parquet("data/qa_formatted_validation.parquet")
df = pd.read_parquet("data/qa_formatted.parquet")
query_answer_pairs = []

for query_id, group in df.groupby('query_id'):
    query = group['query'].iloc[0]
    
    for _, row in group.iterrows():
        query_answer_pairs.append((query, row['answer']))

for query_id, group in df_val.groupby('query_id'):
    query = group['query'].iloc[0]
    
    for _, row in group.iterrows():
        query_answer_pairs.append((query, row['answer']))


# Load Google's pretrained Word2Vec model
model_path = "data/GoogleNews-vectors-negative300.bin"
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
# Get the vocabulary and embeddings
vocab_size = len(word2vec)
embedding_dim = word2vec.vector_size
weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

embedding_layer = nn.Embedding.from_pretrained(weights, freeze=False)

word2index = {word: i for i, word in enumerate(word2vec.index_to_key)}

def word_to_tensor(word):
    """Convert a word into a tensor index for the embedding layer"""
    if word in word2index:
        return torch.tensor([word2index[word]], dtype=torch.long)
    else:
        return torch.tensor([word2index["unk"]], dtype=torch.long)
    
max_query_len = 26
max_answer_len = 231

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    
    query_lengths = [min(item['query'].size(0), max_query_len) for item in batch]
    answer_lengths = [min(item['answer'].size(0), max_answer_len) for item in batch]
    
    padded_query_indices = torch.stack([
        torch.nn.functional.pad(
            item['query'][:query_lengths[i]], 
            (0, max_query_len - query_lengths[i])
        )
        for i, item in enumerate(batch)
    ]).long()
    
    padded_answer_indices = torch.stack([
        torch.nn.functional.pad(
            item['answer'][:answer_lengths[i]], 
            (0, max_answer_len - answer_lengths[i])
        )
        for i, item in enumerate(batch)
    ]).long()
    
    # Convert indices to embeddings using the embedding layer
    padded_queries = embedding_layer(padded_query_indices)  # Shape: [batch_size, max_query_len, 300]
    padded_answers = embedding_layer(padded_answer_indices)  # Shape: [batch_size, max_answer_len, 300]
    
    return {
        'query': padded_queries,
        'answer': padded_answers,
        'query_length': query_lengths,
        'answer_length': answer_lengths,
        "original_answer": [item['original_answer'] for item in batch]
    }

dataset = QADataset(query_answer_pairs, word2index)
batch_size = 512
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

def save_embeddings(device, answer_model, embedding_dimension_output, batch_size):
    progress_bar = tqdm.tqdm(dataloader)
    embeddings_list = []
    document_to_embedding_id = {}
    current_idx = 0
    for batch_idx, batch in enumerate(progress_bar):
        answer = batch['answer'].to(device)
        answer_length = batch['answer_length']
        original_answer = batch['original_answer']
        # print(original_answer)
        answer_embeddings = answer_model(answer, answer_length)
        embedding_numpy = answer_embeddings.cpu().detach().numpy()
        embeddings_list.append(embedding_numpy)

        # print(batch_idx, type(current_idx))
        for i, orig_answer in enumerate(original_answer):
            if (batch_idx == 0 and i == 3):
                print(orig_answer, current_idx + i, embedding_numpy[i])
            document_to_embedding_id[current_idx + i] = orig_answer

        current_idx += batch_size
    
    embeddings_numpy = np.concatenate(embeddings_list, axis=0)
    index = faiss.IndexFlatL2(embedding_dimension_output)
    index.add(embeddings_numpy)
    faiss.write_index(index, "data/answer_embeddings.faiss")
    with open("data/answer_embeddings.json", "w") as f:
        json.dump(document_to_embedding_id, f)

device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
model = TwoTowerModel(query_len=max_query_len, answer_len=max_answer_len, hidden_size_query=125, hidden_size_answer=125)
model.train()
model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_6.pt")['model_state_dict'])
answer_model = model.answer_tower
save_embeddings(device, answer_model, embedding_dimension_output=125, batch_size=batch_size)

model.eval()
document_text = "Install ceramic tile floor to match shower-Average prices for installation are between $11 to $22 per square foot; 2  A light/fan combination-Averages at $180 and one hour of installation; 3  Insulate and re-finish ceilings and walls-Fiberglass wall insulation with R-30 value will cost $2.25 per square foo"


def preprocess_answer(answer, answer_model):
    query_words = answer.split()
    query_tensor = torch.cat([word_to_tensor(word) for word in query_words])
    # print(query_tensor)
    answer_length = len(query_words)
    # print(answer_length, max_answer_len, 'cmon')
    padded_answer_indices = torch.stack([torch.nn.functional.pad(
        query_tensor[:answer_length], 
        (0, max_answer_len - answer_length)
    )]).long()
    padded_answers = embedding_layer(padded_answer_indices)
    answer_length = torch.tensor([answer_length], dtype=torch.long)
    # print(padded_answers)
    answer_embeddings = answer_model(padded_answers, answer_length)
    return answer_embeddings

answer_embeddings = preprocess_answer(document_text, answer_model).unsqueeze(0)
# print(answer_embeddings.shape, answer_embeddings)
index = faiss.read_index("data/answer_embeddings.faiss")


# data_directory = Path("data")
# index_path = data_directory / "answer_embeddings.faiss"

wandb.init(project="document_embeddings")
distances, indices = index.search(answer_embeddings.cpu().detach().numpy(), 4)
for i in range(4):
    print(indices[0][i])

with open("data/answer_embeddings.json", "r") as f:
    document_mapping = json.load(f)

# Print the corresponding answers for each index
for i in range(4):
    idx = str(indices[0][i])  # Convert index to string since JSON keys are strings
    print(f"Index {idx}: {document_mapping[idx]}")
    print(f"Distance: {distances[0][i]}\n")

