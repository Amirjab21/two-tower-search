import pandas as pd
from pathlib import Path
import torch
from gensim.models import KeyedVectors
import tqdm
import psutil
import os
import torch
import torch.nn as nn
from models import QADataset
from models import QueryTower, AnswerTower, TwoTowerModel
# from two_tower_trainer import TwoTowerTrainer
import wandb

# from testing import save_checkpoint

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
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
# Get the vocabulary and embeddings
vocab_size = len(word2vec)
embedding_dim = word2vec.vector_size
weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

# Create a PyTorch Embedding layer
embedding_layer = nn.Embedding.from_pretrained(weights, freeze=False)  # Set freeze=True if you don't want to fine-tune

word2index = {word: i for i, word in enumerate(word2vec.index_to_key)}


def word_to_tensor(word):
    """Convert a word into a tensor index for the embedding layer"""
    if word in word2index:
        return torch.tensor([word2index[word]], dtype=torch.long)
    else:
        return torch.tensor([word2index["unk"]], dtype=torch.long)  # Handle OOV words

def sentence_to_tensor(sentence):
    return [word_to_tensor(word) for word in sentence.split()]



training_examples = []
# Create array of (query, answer) tuples
query_answer_pairs = []

# def prepare_data(df):
#     query_answer_pairs = []
#     for query_id, group in df.groupby('query_id'):
#         query = group['query'].iloc[0]
#         for _, row in group.iterrows():
#             query_answer_pairs.append((query, row['answer']))
#     return query_answer_pairs

# Group by query_id to get all answers for each query
for query_id, group in df.groupby('query_id'):
    query = group['query'].iloc[0]
    
    # Add each answer as a separate tuple with the query
    for _, row in group.iterrows():
        query_answer_pairs.append((query, row['answer']))

query_answer_pairs_val = []
for query_id, group in df_val.groupby('query_id'):
    query = group['query'].iloc[0]
    
    # Add each answer as a separate tuple with the query
    for _, row in group.iterrows():
        query_answer_pairs_val.append((query, row['answer']))
  
# def evaluate_model(model, val_dataset, word2vec, device, num_samples=1024):
    # model.eval()
    # query_answer_pairs_val = []
    # for query_id, group in df_val.groupby('query_id'):
    #     query = group['query'].iloc[0]
    #     for _, row in group.iterrows():
    #         if len(query_answer_pairs_val) >= num_samples:  # Add limit
    #             break
    #         query_answer_pairs_val.append((query, row['answer']))
    #     if len(query_answer_pairs_val) >= num_samples:  # Add limit
    #         break
    # # val_queries, val_docs = prepare_data(val_dataset, num_samples=num_samples)
    # dataset = QADataset(query_answer_pairs_val, word2index)
    # val_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    # total_loss = 0
    
    # total_loss = 0
    # margin = 0.1  # Should match training margin
    
    # with torch.no_grad():
    #     for batch in val_loader:
    #         query = batch['query'].to(device)
    #         answer = batch['answer'].to(device)
    #         query_length = batch['query_length']
    #         answer_length = batch['answer_length']
    #         batch_size = query.shape[0]

    #         query_embeddings, answer_embeddings = model(query, answer, query_length, answer_length)
            
    #         # Compute similarities matrix
    #         similarities = nn.functional.cosine_similarity(
    #             query_embeddings.unsqueeze(1),
    #             answer_embeddings.unsqueeze(0),
    #             dim=2
    #         )
            
    #         # Calculate loss same way as training
    #         positive_similarities = torch.diagonal(similarities)
    #         positive_distances = 1 - positive_similarities
    #         negative_distances = 1 - similarities
            
    #         mask = torch.eye(batch_size, device=device)
    #         negative_distances = negative_distances.masked_fill(mask.bool(), float('inf'))
    #         hardest_negative_distances = torch.min(negative_distances, dim=1)[0]
            
    #         batch_loss = torch.mean(
    #             torch.clamp(positive_distances - hardest_negative_distances + margin, min=0.0)
    #         )
            
    #         total_loss += batch_loss.item() * batch_size
    
    # return total_loss / len(val_loader)
max_query_len = max(len(query.split()) for query, _ in query_answer_pairs)
max_answer_len = max(len(answer.split()) for _, answer in query_answer_pairs)
print(max_query_len, max_answer_len)
def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    
    query_lengths = [item['query'].size(0) for item in batch]
    answer_lengths = [item['answer'].size(0) for item in batch]
    # First pad the indices
    padded_query_indices = torch.stack([
        torch.nn.functional.pad(item['query'], (0, max_query_len - item['query'].size(0)))
        for item in batch
    ]).long()
    padded_answer_indices = torch.stack([
        torch.nn.functional.pad(item['answer'], (0, max_answer_len - item['answer'].size(0)))
        for item in batch
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

dataset = QADataset(query_answer_pairs, word2index)
batch_size = 512
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
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
    # criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, metrics='loss')

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=1
    # )
    
    zero = torch.tensor(0.0).to(device)
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
            # print(answer.shape, answer[0], answer[1], answer[37])
            query_embeddings, answer_embeddings = model(query, answer, query_length, answer_length)
            
            # print(query_embeddings.shape, "query_embeddings")
            # Compute all pairwise similarities at once
            positive_similarities = nn.functional.cosine_similarity(
                query_embeddings,  # [B, 1, D]
                answer_embeddings,# [1, B, D]
                dim=1
            )  # [B, B]

            shifted_answer_embeddings = torch.roll(answer_embeddings, shifts=1, dims=0)
            negative_similarities = nn.functional.cosine_similarity(
                query_embeddings,
                shifted_answer_embeddings,
                dim=1
            )


            # print(similarities.shape, "similarities")
            
            # Positive similarities are on the diagonal
            # positive_similarities = torch.diagonal(similarities)  # [B]
            
            # Compute loss using matrix operations
            positive_distances = 1 - positive_similarities
            negative_distances = 1 - negative_similarities
            
            # Mask out diagonal (positive pairs) from negative distances
            batch_loss = torch.mean(
                torch.clamp(positive_distances - negative_distances + margin, min=0.0)
            )
            
            # Compute loss
            
            # batch_loss.backward()
            # print(batch_loss, "batch_loss")
            # Update progress bar with current batch loss
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

        # scheduler.step()
        # val_loss = evaluate_model(model, df_val, word2index, device, 1024)
        # print(val_loss, "val_loss")
        epoch_loss = total_loss / len(train_loader.dataset)  # Average loss over all samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        if wandb_yes:
            wandb.log({
                "epoch_loss": epoch_loss,
                "epoch": epoch,
                # "val_loss": val_loss
            })



        # val_loss = validation_loss(model, query_answer_pairs_val, max_query_len, max_answer_len, device)
        # wandb.log({
        #     "val_loss": val_loss,
        #     "epoch": epoch,
        # })
        

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, epoch_loss)

        # test_retrieval(model, "How do I improve my programming skills?", dataset, word_to_tensor, max_query_len, collate_fn, embedding_layer, 5)


    return model



hidden_size_query = 125
hidden_size_answer = 125
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = train(dataloader, device, learning_rate=0.001, num_epochs=7, batch_size=batch_size, hidden_size_query=hidden_size_query, hidden_size_answer=hidden_size_answer, wandb_yes=False)
# device = torch.device("cpu")
# device = "cpu"
# model = TwoTowerModel(max_query_len, max_answer_len, hidden_size_query, hidden_size_answer).to(device)
# test_retrieval(model, "checkpoints/checkpoint_epoch_2.pt", "What is the reserve bank of australia?", dataset, word_to_tensor, max_query_len, collate_fn, embedding_layer, 5)








