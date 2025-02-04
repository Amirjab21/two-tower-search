import torch
import torch.nn as nn
from models import TwoTowerModel, QADataset
from pathlib import Path
import wandb
def test_retrieval(model, checkpoint_path, query, dataset, word_to_tensor, max_query_len, collate_fn,embedding_layer, k=5):
    """
    Test the model's retrieval capabilities
    Args:
        model: Trained TwoTowerModel
        query: String query to test
        dataset: QADataset containing all query-answer pairs
        word2index: Dictionary mapping words to indices
        k: Number of top results to return
    """
    # device = next(model.parameters()).device
    # model = load_checkpoint(model, checkpoint_path)
    device = "cpu"
    model.eval()

    with torch.no_grad():
        # Convert query to tensor
        query_tokens = [word_to_tensor(word) for word in query.split()]
        # Create a tensor of indices first
        query_indices = torch.zeros(max_query_len, dtype=torch.long)
        query_indices = query_indices.to(device)
        for i, token in enumerate(query_tokens[:max_query_len]):
            query_indices[i] = token
        
        # Convert indices to embeddings using the embedding layer
        query_tensor = embedding_layer(query_indices).to(device)  # Shape: [max_query_len, 300]

        query_embedding = model.query_tower(query_tensor)
        
        # Get all answer embeddings
        all_answers = []
        all_answer_embeddings = []
        
        # Process answers in batches to avoid memory issues
        batch_size = 512
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        for batch in dataloader:
            answers = batch['answer'].to(device)
            answer_embeddings = model.answer_tower(answers)
            print(f"Single batch answer_embeddings shape: {answer_embeddings.shape}")
            all_answer_embeddings.append(answer_embeddings)
            all_answers.extend([dataset.query_answer_pairs[i][1] for i in range(len(answers))])
            print(f"Current length of all_answers list: {len(all_answers)}")
        
        # Concatenate all answer embeddings
        all_answer_embeddings = torch.cat(all_answer_embeddings, dim=0)
        # print(f"Final all_answer_embeddings shape: {all_answer_embeddings.shape}")
        # print(f"Final length of all_answers list: {len(all_answers)}")
        
        # Calculate similarities
        similarities = torch.matmul(query_embedding, all_answer_embeddings.T)
        # print(f"Similarities shape: {similarities.shape}")
        # print(f"Similarities: {similarities[0]}")
        # Get top k results
        top_k_similarities, top_k_indices = torch.topk(similarities, k=k)
        
        results = []
        for sim, idx in zip(top_k_similarities, top_k_indices):
            results.append({
                'answer': all_answers[idx],
                'similarity_score': sim.item()
            })
        
        print("\nTop 5 Retrieved Answers:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Answer: {result['answer']}")
            print(f"   Similarity Score: {result['similarity_score']:.4f}")
        model.train()
        return results

# Example usage
# if __name__ == "__main__":
#     # Load your trained model
    # model = TwoTowerModel(max_query_len, max_answer_len).to(device)
    # Load the saved model weights if you have them
    # model.load_state_dict(torch.load('path_to_saved_model.pth'))
    
    # Test queries
# test_queries = [
#     "How do I improve my programming skills?",
#     "What's the best way to learn machine learning?",
#     # Add more test queries...
# ]

# for query in test_queries:
#     print(f"\nQuery: {query}")
#     results = test_retrieval(model, query, dataset, word2index)
#     print("\nTop 5 Retrieved Answers:")
#     for i, result in enumerate(results, 1):
#         print(f"\n{i}. Answer: {result['answer']}")
#         print(f"   Similarity Score: {result['similarity_score']:.4f}")


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

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint.
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        tuple: (epoch, val_loss) from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return model
# device = torch.device("cpu")
# model = TwoTowerModel(22, 191,6,6 ).to(device)
# test_retrieval(model, "checkpoints/checkpoint_epoch_2.pt", "How do I improve my programming skills?",)