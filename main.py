import pandas as pd
from pathlib import Path
import torch
from gensim.models import KeyedVectors
import tqdm

import torch
import torch.nn as nn
from models import QADataset
from models import QueryTower, AnswerTower, TwoTowerModel
# from two_tower_trainer import TwoTowerTrainer
import wandb

from testing import test_retrieval, save_checkpoint


df = pd.read_parquet('data/qa_formatted.parquet')


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

# Group by query_id to get all answers for each query
for query_id, group in df.groupby('query_id'):
    query = group['query'].iloc[0]
    
    # Add each answer as a separate tuple with the query
    for _, row in group.iterrows():
        query_answer_pairs.append((query, row['answer']))
    
    # Rest of the code can stay if you want to keep the original functionality too
    selected_answers = group[group['is_selected'] == 1]['answer'].tolist()
    non_selected_answers = group[group['is_selected'] == 0]['answer'].tolist()
    all_answers = selected_answers + non_selected_answers
    
    training_example = {
        'query': query,
        'answers': all_answers,
        'num_selected': len(selected_answers)
    }
    training_examples.append(training_example)

# Print first few query-answer pairs to verify
print("\nFirst few query-answer pairs:")
print(query_answer_pairs[:3])

# Convert to DataFrame
# training_df = pd.DataFrame(training_examples)

# Save to parquet
# training_df.to_parquet('training_examples.parquet')
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

# Example usage:
for batch in dataloader:
    print("Batch shape:")
    print(f"Queries: {batch['query'].shape}")
    print(f"Answers: {batch['answer'].shape}")
    break







def train(train_loader: torch.utils.data.DataLoader, device, learning_rate, num_epochs, batch_size):
    """Train the model"""
    wandb.init(
            project="two-tower-training",
            config={
                "query_max_len": max_query_len,
                "answer_max_len": max_answer_len,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
            },
        )
    
    hidden_size_query = 6
    hidden_size_answer = 6

    model = TwoTowerModel(max_query_len, max_answer_len, hidden_size_query, hidden_size_answer).to(device)
    # criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer2, mode="min", factor=0.5, patience=1
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
            # print(answer_embeddings.shape, answer_embeddings[0], answer_embeddings[1], answer_embeddings[37])
            batch_loss = torch.tensor(0.0).to(device)
            
            margin = 0.5  # Hyperparameter you can tune
            for i in range(batch_size):
                # Positive pair
                query_emb = query_embeddings[i]
                pos_answer_emb = answer_embeddings[i]
                
                # Calculate positive similarity
                pos_similarity = torch.dot(query_emb, pos_answer_emb)
                
                # Randomly select 3 negative indices
                available_indices = [j for j in range(batch_size) if j != i]
                # Just select one random negative index
                neg_idx = available_indices[torch.randint(len(available_indices), (1,)).item()]
                neg_answer_emb = answer_embeddings[neg_idx]

                # Calculate distances (using dot product similarity, convert to distance)
                pos_distance = 1 - pos_similarity  # Convert similarity to distance
                
                # Calculate negative similarity (for single negative example)
                neg_distance = 1 - torch.dot(query_emb, neg_answer_emb)
                # print(neg_idx, i)
                # print(answer_embeddings[0], answer_embeddings[1], answer_embeddings[37])
                loss = torch.max(zero, pos_distance - neg_distance + margin)
                batch_loss += loss
            

            # Update progress bar
            batch_loss = batch_loss / batch_size
            
            # Update progress bar with current batch loss
            progress_bar.set_postfix({"loss": batch_loss.item()})

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item() * batch_size  # Accumulate the total loss

            wandb.log({
                "batch_loss": batch_loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })


        epoch_loss = total_loss / len(train_loader.dataset)  # Average loss over all samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        wandb.log({
            "epoch_loss": epoch_loss,
            "epoch": epoch,
        })
        

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, epoch_loss)

        # test_retrieval(model, "How do I improve my programming skills?", dataset, word_to_tensor, max_query_len, collate_fn, embedding_layer, 5)


    return model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
train(dataloader, device, learning_rate=0.001, num_epochs=3, batch_size=batch_size)
# device = "cpu"
# model = TwoTowerModel(max_query_len, max_answer_len, 6, 6).to(device)
# test_retrieval(model, "Results-Based Accountability® (also known as RBA) is a disciplined way of thinking", dataset, word_to_tensor, max_query_len, collate_fn, embedding_layer, 5)








