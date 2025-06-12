import torch
import torch.nn as nn
import numpy as np

class QueryTower(nn.Module):
    def __init__(self, input_dim=12, hidden_size=3):  # Update input dimension to match your data
        super(QueryTower, self).__init__()
        self.rnn = nn.GRU(input_size=300, 
                         hidden_size=hidden_size,
                         num_layers=1,
                         bidirectional=True,
                         dropout=0.1,
                         batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
    def forward(self, x, lengths=None):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed_x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        embedding = self.proj(hidden.squeeze(0))  # Remove dimension of size 1
        return embedding
    
class AnswerTower(nn.Module):
    def __init__(self, input_dim=32, hidden_size=3):  # Added hidden_size parameter
        super(AnswerTower, self).__init__()
        self.rnn = nn.GRU(input_size=300,
                         hidden_size=hidden_size,
                         num_layers=1,
                         bidirectional=True,
                         dropout=0.1,
                         batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x, lengths=None):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed_x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        embedding = self.proj(hidden.squeeze(0))  # Remove dimension of size 1
        return embedding
    
class TwoTowerModel(nn.Module):
    def __init__(self, query_len, answer_len, hidden_size_query, hidden_size_answer):
        super().__init__()
        self.query_tower = QueryTower(input_dim=query_len, hidden_size=hidden_size_query)
        self.answer_tower = AnswerTower(input_dim=answer_len, hidden_size=hidden_size_answer)
    
    def forward(self, query, answer, query_lengths=None, answer_lengths=None):
        query_embeddings = self.query_tower(query, query_lengths)
        answer_embeddings = self.answer_tower(answer, answer_lengths)
        return query_embeddings, answer_embeddings


class QADataset(torch.utils.data.Dataset):
    def __init__(self, query_answer_pairs, word2index, embedding_layer):
        self.query_answer_pairs = query_answer_pairs
        self.word2index = word2index
        self.embedding_layer = embedding_layer
    
    def __len__(self):
        return len(self.query_answer_pairs)
    
    def __getitem__(self, idx):
        ((query, query_length), (answer, answer_length)) = self.query_answer_pairs[idx]
        
        query_tensor = torch.cat([self.embedding_layer(torch.tensor([word], dtype=torch.long)) for word in query])
        answer_tensor = torch.cat([self.embedding_layer(torch.tensor([word], dtype=torch.long)) for word in answer])
        
        return {
            'query': query_tensor,
            'answer': answer_tensor,
            'query_length': query_length,
            'answer_length': answer_length,
            # "original_answer": answer
        }

    def word_to_tensor(self, word):
        """Convert a word into a tensor index for the embedding layer"""
        if word in self.word2index:
            return torch.tensor([self.word2index[word]], dtype=torch.long)
        else:
            return torch.tensor([self.word2index["unk"]], dtype=torch.long)  # Handle OOV words
    