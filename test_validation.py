import pandas as pd
import torch
from models import TwoTowerModel, QADataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import gensim.downloader as api
import faiss
import json
import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df = pd.read_parquet('data/qa_formatted_validation.parquet').head(50)

df_selected = df[df['is_selected'] == 1]

selected_answers = dict(zip(
    df[df['is_selected'] == 1]['query'],
    df[df['is_selected'] == 1]['answer']
))

# print(selected_answers['what causes alkalosis'])



def load_model(checkpoint_path, max_query_len, max_answer_len, hidden_size_query, hidden_size_answer):
    model = TwoTowerModel(query_len=max_query_len, answer_len=max_answer_len, hidden_size_query=hidden_size_query, hidden_size_answer=hidden_size_answer)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    return model

models = {
    "checkpoint_path": "checkpoints/checkpoint_epoch_6.pt",
    "max_query_len": 26,
    "max_answer_len": 201,
    "hidden_size_query": 125,
    "hidden_size_answer": 125
}

model = load_model(**models)


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


def word_to_tensor(word):
    """Convert a word into a tensor index for the embedding layer"""
    if word in word2index:
        return torch.tensor([word2index[word]], dtype=torch.long)
    else:
        return torch.tensor([word2index["unk"]], dtype=torch.long)





def preprocess_answer(query, answer_model, max_length):
    query_words = query.split()
    query_tensor = torch.cat([word_to_tensor(word) for word in query_words])
    # print(query_tensor.shape)
    answer_length = len(query_words)
    
    padded_answer_indices = torch.stack([torch.nn.functional.pad(
        query_tensor[:answer_length], 
        (0, max_length - answer_length)
    )]).long()
    # print(padded_answer_indices.shape)
    padded_answers = embedding_layer(padded_answer_indices)
    # print(padded_answers.shape)
    answer_length = torch.tensor([answer_length], dtype=torch.long)
    # print(answer_length.shape)
    answer_embeddings = answer_model(padded_answers, answer_length)
    # print(answer_embeddings.shape)
    return answer_embeddings

def evaluate_model(model, df_val, max_query_len, max_answer_len, collate_fn, word2index, device, k=10):
    model.eval()
    query_answer_pairs_val = []
    for query_id, group in df_val.groupby('query_id'):
        query = group['query'].iloc[0]
        for _, row in group.iterrows():
            query_answer_pairs_val.append((query, row['answer']))
    print(len(query_answer_pairs_val))
    dataset = QADataset(query_answer_pairs_val, word2index)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=collate_fn)

    index = faiss.read_index("data/answer_embeddings.faiss")
    with open("data/answer_embeddings.json", "r") as f:
        document_mapping = json.load(f)

    with torch.no_grad():
        # for batch in val_loader:
        # query, answer = batch['query'].to(device), batch['answer'].to(device)
        # query_length, answer_length = batch['query_length'], batch['answer_length']
        query_embeddings = preprocess_answer(query, model.query_tower, max_query_len).unsqueeze(0)
        
        distances, indices = index.search(query_embeddings.cpu().detach().numpy(), k)

        answers = [document_mapping[str(indices[0][i])] for i in range(k)]


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    
    query_lengths = [min(item['query'].size(0), models['max_query_len']) for item in batch]
    answer_lengths = [min(item['answer'].size(0), models['max_answer_len']) for item in batch]
    
    padded_query_indices = torch.stack([
        torch.nn.functional.pad(
            item['query'][:query_lengths[i]], 
            (0, models['max_query_len'] - query_lengths[i])
        )
        for i, item in enumerate(batch)
    ]).long()
    
    padded_answer_indices = torch.stack([
        torch.nn.functional.pad(
            item['answer'][:answer_lengths[i]], 
            (0, models['max_answer_len'] - answer_lengths[i])
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
                
def get_val_docs(df_val):
    val_docs = []
    val_doc_to_queries = {}
    query_to_relevant = {}
    
    for row in tqdm.tqdm(df_val.itertuples(), total=len(df_val)):
        query = row.query
        answer = row.answer
        is_selected = row.is_selected
        
        if query not in query_to_relevant:
            query_to_relevant[query] = []
            
        if is_selected == 1:
            query_to_relevant[query].append(answer)  # Append just the answer instead of selected_answers
            if answer not in val_doc_to_queries:
                val_doc_to_queries[answer] = []
            val_doc_to_queries[answer].append(query)
            
        if answer not in val_docs:
            val_docs.append(answer)

    return val_docs, val_doc_to_queries, query_to_relevant
    
val_docs, val_doc_to_queries, query_to_relevant = get_val_docs(df_selected)

with open('data/val_docs.json', 'w') as f:
    json.dump(val_docs, f)

with open('data/val_doc_to_queries.json', 'w') as f:
    json.dump(val_doc_to_queries, f)

with open('data/query_to_relevant.json', 'w') as f:
    json.dump(query_to_relevant, f)

# with open('data/val_docs.json', 'r') as f:
#     val_docs = json.load(f)

# with open('data/val_doc_to_queries.json', 'r') as f:
#     val_doc_to_queries = json.load(f)

# with open('data/query_to_relevant.json', 'r') as f:
#     query_to_relevant = json.load(f)

doc_encodings = []
batch_size = 128
for i in tqdm.tqdm(range(0, len(val_docs), batch_size)):
    batch_docs = val_docs[i:i + batch_size]
    batch_encodings = []
    for doc in batch_docs:
        print(doc, "doc")
        doc_vec = preprocess_answer(doc, model.answer_tower, models['max_answer_len'])
        batch_encodings.append(doc_vec)
    doc_encodings.extend(batch_encodings)
doc_encodings = torch.cat(doc_encodings, dim=0)
# evaluate_model(model, df, models['max_query_len'], models['max_answer_len'], collate_fn, word2index, device, k=10)


def search_ms_marco(query: str, k: int = 5):
    query_vec = preprocess_answer(query, model.query_tower, models['max_query_len'])
    print(query_vec.shape, doc_encodings.shape)
    similarities = torch.nn.functional.cosine_similarity(query_vec, doc_encodings)
    top_k = torch.topk(similarities, k=k)
    results = []
    for idx, score in zip(top_k.indices, top_k.values):
        results.append((val_docs[idx], score.item()))
    return results

num_test_queries = 500
for sample in tqdm.tqdm(df_selected.itertuples(), total=len(df_selected)):
    mrr_sum = 0
    test_count = 0
    query = sample.query
    if not query in query_to_relevant or not query_to_relevant[query]:
        continue
    relevant_docs = set(query_to_relevant[query])
    results = search_ms_marco(query, k=10)
    

    
    # Calculate MRR
    mrr = 0
    for rank, (doc, score) in enumerate(results, 1):
        is_relevant = "✓" if doc in relevant_docs else " "
        
        if doc in relevant_docs and mrr == 0:
            mrr = 1.0 / rank
    
    mrr_sum += mrr
    test_count += 1
    
    print(f"\nQuery: {query}")
    print(f"MRR: {mrr:.4f}")
    for rank, (doc, score) in enumerate(results[:3], 1):
        is_relevant = "✓" if doc in relevant_docs else " "
        print(f"{rank}. [{is_relevant}] ({score:.4f}) {doc[:100]}...")
    
    if test_count >= num_test_queries:
        break

avg_mrr = mrr_sum / test_count

print(f"Average MRR: {avg_mrr:.4f}")


# with open('data/val_docs.json', 'w') as f:
#     json.dump(val_docs, f)

# with open('data/val_doc_to_queries.json', 'w') as f:
#     json.dump(val_doc_to_queries, f)

# with open('data/query_to_relevant.json', 'w') as f:
#     json.dump(query_to_relevant, f)
    

with open('data/mrr_score.txt', 'w') as f:
    f.write(f"Average MRR: {avg_mrr:.4f}\n")
    f.write(f"Number of test queries: {test_count}\n")

def get_MRR_score(model, df_val, max_query_len, max_answer_len, collate_fn, word2index, device, k=10):
    df_val_set = pd.read_parquet('data/qa_formatted_validation.parquet')
    df_val_set = df_val_set.head(1024)
    mrr_sum = 0
    test_count = 0
    for sample in tqdm.tqdm(df_val_set.itertuples(), total=len(df_val_set)):
        mrr_sum = 0
        