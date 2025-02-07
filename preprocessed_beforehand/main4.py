import pandas as pd
from pathlib import Path
import torch
from gensim.models import KeyedVectors
import tqdm
import torch
import torch.nn as nn
from models4 import QADataset, TwoTowerModel
import wandb
import gensim.downloader as api
import json

def save_checkpoint(model, optimizer, epoch, val_loss):
    checkpoint_dir = Path("checkpoints")
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    return checkpoint_path



df = pd.read_parquet('data/selected_only.parquet')
df_val = pd.read_parquet('data/qa_formatted_validation.parquet').head(1024)




# Load Google's pretrained Word2Vec model
model_path = "data/GoogleNews-vectors-negative300.bin"
# word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
word2vec = api.load('word2vec-google-news-300')
# Get the vocabulary and embeddings
vocab_size = len(word2vec)
embedding_dim = word2vec.vector_size
weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

# Create a PyTorch Embedding layer
embedding_layer = nn.Embedding.from_pretrained(weights, freeze=True)  # Set freeze=True if you don't want to fine-tune


# word2index = {word: i for i, word in enumerate(word2vec.index_to_key)}
with open('data/word2index.json', 'r') as f:
    word2index = json.load(f)


def word_to_tensor(word):
    """Convert a word into a tensor index for the embedding layer"""
    if word in word2index:
        return torch.tensor([word2index[word]], dtype=torch.long)
    else:
        return torch.tensor([word2index["unk"]], dtype=torch.long)  # Handle OOV words

def word_to_numpy(word):
    if word in word2index:
        return word2index[word]
    else:
        return word2index["unk"]

def sentence_to_tensor(sentence):
    return [word_to_tensor(word) for word in sentence.split()]


query_answer_pairs = []

for query_id, group in df.groupby('query_id'):
    query = group['query'].iloc[0]
    
    for _, row in group.iterrows():
        query_answer_pairs.append((query, row['answer']))

query_answer_pairs_val = []
for query_id, group in df_val.groupby('query_id'):
    query = group['query'].iloc[0]
    
    for _, row in group.iterrows():
        query_answer_pairs_val.append((query, row['answer']))



def preprocess(query_answer_pairs):
    processed_pairs = []
    MAX_QUERY_LENGTH = 26
    PAD_TOKEN = 300001
    MAX_ANSWER_LENGTH = 201
    for query, answer in query_answer_pairs:
        if not isinstance(query, str) or not isinstance(answer, str):
            continue
        query_tokens = [word_to_numpy(word) for word in query.split()]
        answer_tokens = [word_to_numpy(word) for word in answer.split()]

        # Store original lengths before padding/truncating
        query_len = min(len(query_tokens), MAX_QUERY_LENGTH)
        answer_len = min(len(answer_tokens), MAX_ANSWER_LENGTH)

        # Pad or truncate query
        if len(query_tokens) < MAX_QUERY_LENGTH:
            query_tokens.extend([PAD_TOKEN] * (MAX_QUERY_LENGTH - len(query_tokens)))
        else:
            query_tokens = query_tokens[:MAX_QUERY_LENGTH]

        # Pad or truncate answer
        if len(answer_tokens) < MAX_ANSWER_LENGTH:
            answer_tokens.extend([PAD_TOKEN] * (MAX_ANSWER_LENGTH - len(answer_tokens)))
        else:
            answer_tokens = answer_tokens[:MAX_ANSWER_LENGTH]
            
        processed_pairs.append(((query_tokens, query_len), (answer_tokens, answer_len)))
    return processed_pairs

query_answer_pairs_indices = preprocess(query_answer_pairs)
query_answer_pairs_indices_val = preprocess(query_answer_pairs_val)


  
def evaluate_model(model, df_val, max_query_len, max_answer_len, collate_fn, num_samples, margin):
    model.eval()
    # query_answer_pairs_indices_val = preprocess(query_answer_pairs_val)
    dataset = QADataset(query_answer_pairs_indices_val, word2index, embedding_layer)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True) #Removed collate_fn

    
    with torch.no_grad():
        for batch in val_loader:

            query, answer = batch['query'].to(device), batch['answer'].to(device)
            query_length, answer_length = batch['query_length'], batch['answer_length']
            query_embeddings, answer_embeddings = model(query, answer, query_length, answer_length)
            
            # Compute similarities matrix
            positive_similarities = nn.functional.cosine_similarity(
                query_embeddings,  
                answer_embeddings,
                dim=1
            )

            shifted_answer_embeddings = torch.roll(answer_embeddings, shifts=1, dims=0)
            negative_similarities = nn.functional.cosine_similarity(
                query_embeddings,
                shifted_answer_embeddings,
                dim=1
            )

            positive_distances = 1 - positive_similarities
            negative_distances = 1 - negative_similarities
            
            batch_loss = torch.mean(
                torch.clamp(positive_distances - negative_distances + margin, min=0.0)
            )
                
    return batch_loss.item()

max_query_len = max(len(query.split()) for query, _ in query_answer_pairs)
max_answer_len = max(len(answer.split()) for _, answer in query_answer_pairs)

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""

    query_lengths = [min(item['query'].size(0), max_query_len) for item in batch]
    answer_lengths = [min(item['answer'].size(0), max_answer_len) for item in batch]

    padded_query_indices = torch.stack([
        torch.nn.functional.pad(
            item['query'][:query_lengths[i]], 
            (0, max_query_len - query_lengths[i]),
            value=word2index["~pad~"]
        )
        for i, item in enumerate(batch)
    ]).long()
    
    padded_answer_indices = torch.stack([
        torch.nn.functional.pad(
            item['answer'][:answer_lengths[i]], 
            (0, max_answer_len - answer_lengths[i]),
            value=word2index["~pad~"]
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
        'answer_length': answer_lengths
    }

dataset = QADataset(query_answer_pairs_indices, word2index, embedding_layer)
batch_size = 2048
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    # collate_fn=collate_fn,
    num_workers=8
)



def train(train_loader: torch.utils.data.DataLoader, device, learning_rate, num_epochs, batch_size, hidden_size_query, hidden_size_answer, margin=0.1, wandb_yes = False):
    """Train the model"""
    if wandb_yes:
        wandb.init(
            project="two-tower-training",
            config={
                "query_max_len": max_query_len,
                "answer_max_len": max_answer_len,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "hidden_size_query": hidden_size_query,
                "hidden_size_answer": hidden_size_answer,
            },
        )
    
    

    model = TwoTowerModel(max_query_len, max_answer_len, hidden_size_query, hidden_size_answer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0
        progress_bar = tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        )
        for batch_idx, batch in enumerate(progress_bar):
            query, answer = batch['query'].to(device), batch['answer'].to(device)
            query_length, answer_length = batch['query_length'], batch['answer_length']
            batch_size = query.shape[0]

            optimizer.zero_grad()
            query_embeddings, answer_embeddings = model(query, answer, query_length, answer_length)
            

            positive_similarities = nn.functional.cosine_similarity(
                query_embeddings,  
                answer_embeddings,
                dim=1
            )

            shifted_answer_embeddings = torch.roll(answer_embeddings, shifts=1, dims=0)
            negative_similarities = nn.functional.cosine_similarity(
                query_embeddings,
                shifted_answer_embeddings,
                dim=1
            )
            positive_distances = 1 - positive_similarities
            negative_distances = 1 - negative_similarities
            
            # Mask out diagonal (positive pairs) from negative distances
            batch_loss = torch.mean(
                torch.clamp(positive_distances - negative_distances + margin, min=0.0)
            )

            progress_bar.set_postfix({"loss": batch_loss.item()})

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item() * batch_size  # Accumulate the total loss

            if wandb_yes:
                wandb.log({
                    "batch_loss": batch_loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })

        val_loss = evaluate_model(model, df_val, max_query_len, max_answer_len,collate_fn, 1024, margin)
        epoch_loss = total_loss / len(train_loader.dataset)  # Average loss over all samples
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        if wandb_yes:
            wandb.log({
                "epoch_loss": epoch_loss,
                "epoch": epoch,
                "val_loss": val_loss
            })

        if wandb_yes:
            save_checkpoint(model, optimizer, epoch, epoch_loss)


    return model



hidden_size_query = 250
hidden_size_answer = 250
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = train(dataloader, device, learning_rate=0.001, num_epochs=8, batch_size=batch_size, hidden_size_query=hidden_size_query, hidden_size_answer=hidden_size_answer, wandb_yes=True)
