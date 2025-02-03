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


df = pd.read_parquet('data/qa_formatted.parquet').head(10000)


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
    
    # Pad sequences
    padded_queries = torch.stack([
        torch.nn.functional.pad(item['query'], (0, max_query_len - item['query'].size(0)))
        for item in batch
    ]).float()
    padded_answers = torch.stack([
        torch.nn.functional.pad(item['answer'], (0, max_answer_len - item['answer'].size(0)))
        for item in batch
    ]).float()
    
    return {
        'query': padded_queries,
        'answer': padded_answers
    }

dataset = QADataset(query_answer_pairs, word2index)
batch_size = 32
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

    model = TwoTowerModel(max_query_len, max_answer_len).to(device)
    # criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer2, mode="min", factor=0.5, patience=1
    # )

    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0
        progress_bar = tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
           
            query, answer = batch['query'].to(device), batch['answer'].to(device)
            batch_size = query.shape[0]

            optimizer.zero_grad()

            query_embeddings, answer_embeddings = model(query, answer)

            total_loss = 0
            for i in range(batch_size):
                # Positive pair
                query_emb = query_embeddings[i]
                pos_answer_emb = answer_embeddings[i]
                
                # Calculate positive similarity
                pos_similarity = torch.dot(query_emb, pos_answer_emb)
                
                # Get negative examples (all other answers in batch)
                negative_indices = [j for j in range(batch_size) if j != i]
                neg_answer_embs = answer_embeddings[negative_indices]
                
                # Calculate negative similarities
                neg_similarities = torch.matmul(query_emb, neg_answer_embs.T)
                
                # Combine positive and negative similarities
                all_similarities = torch.cat([pos_similarity.unsqueeze(0), neg_similarities])
                
                # Create label (positive example should have highest similarity)
                label = torch.zeros(len(all_similarities), device=device)
                label[0] = 1  # First position is the positive example
                
                # Calculate loss for this query
                loss = nn.CrossEntropyLoss()(all_similarities.unsqueeze(0), label.unsqueeze(0).argmax(dim=1))
                total_loss += loss

            # Update progress bar
            batch_loss = total_loss / batch_size
            
            # Update progress bar with current batch loss
            progress_bar.set_postfix({"loss": batch_loss.item()})

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            wandb.log({
                "batch_loss": batch_loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

            # Log to WandB

        epoch_loss = total_loss / len(train_loader.dataset)  # Average loss over all samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Log epoch metrics to WandB
        wandb.log({
            "epoch_loss": epoch_loss.item(),
            "epoch": epoch,
        })
        # logging.info(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        # scheduler.step(avg_loss)

        # Perform similarity check for 'dog' and other test words
        # test_word = "love"
        
        # similar_words = self.find_similar_words(model, test_word, n=10)
        # print(similar_words)

        # Save checkpoint
        # self.save_checkpoint(model, optimizer, epoch, avg_loss)

    return model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
train(dataloader, device, learning_rate=0.001, num_epochs=10, batch_size=batch_size)









