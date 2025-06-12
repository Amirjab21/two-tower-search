import torch
import torch.nn as nn


class QueryTower(nn.Module):
    def __init__(self, input_dim=12, hidden_size=3):  # Update input dimension to match your data
        super(QueryTower, self).__init__()
        self.rnn = nn.RNN(input_size=300, 
                         hidden_size=hidden_size,
                         num_layers=1,
                         batch_first=True)
    def forward(self, x, lengths=None):
        if lengths is not None:
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            output, hidden = self.rnn(packed_x)
        else:
            output, hidden = self.rnn(x)
        # print("Input shape:", x.shape)
        # print("First sequence in batch:", x[0][:5])  # First 5 elements
        # print("Last sequence in batch:", x[3][:5])  # First 5 elements
        
        # Remove the first dimension and print sample elements
        embedding = hidden.squeeze(0)  # Remove dimension of size 1
        # print("Query embedding shape:", embedding.shape)
        # print("First query in batch:", embedding[0])
        # print("Last query in batch:", embedding[-1])
        return embedding
    
class AnswerTower(nn.Module):
    def __init__(self, input_dim=32, hidden_size=3):  # Added hidden_size parameter
        super(AnswerTower, self).__init__()
        self.rnn = nn.RNN(input_size=300,
                         hidden_size=hidden_size,
                         num_layers=1,
                         batch_first=True)
    
    def forward(self, x, lengths=None):
        if lengths is not None:
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            output, hidden = self.rnn(packed_x)
        else:
            output, hidden = self.rnn(x)
        # print("Output shape:", output.shape)

        # Remove the first dimension and print sample elements
        embedding = hidden.squeeze(0)  # Remove dimension of size 1
        # print("Answer embedding shape:", embedding.shape)
        # print("First answer in batch:", embedding[0])
        # print("Last answer in batch:", embedding[-1])
        return embedding
    
class TwoTowerModel(nn.Module):
    def __init__(self, query_len, answer_len, hidden_size_query, hidden_size_answer):
        super().__init__()
        self.query_tower = QueryTower(input_dim=query_len, hidden_size=hidden_size_query)
        self.answer_tower = AnswerTower(input_dim=answer_len, hidden_size=hidden_size_answer)
    
    def forward(self, query, answer, negative_answer, query_lengths=None, answer_lengths=None, negative_answer_lengths=None):
        query_embeddings = self.query_tower(query, query_lengths)
        answer_embeddings = self.answer_tower(answer, answer_lengths)
        negative_answer_embeddings = self.answer_tower(negative_answer, negative_answer_lengths)
        return query_embeddings, answer_embeddings, negative_answer_embeddings


class QADataset(torch.utils.data.Dataset):
    def __init__(self, df, embedding_layer):
        self.df = df
        self.embedding_layer = embedding_layer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # query, answer = self.query_answer_pairs[idx]

        # Convert to tensors only when accessed
        query_words = self.embedding_layer(self.df['query_padded'].iloc[idx])
        answer_words = self.embedding_layer(self.df['answer_padded'].iloc[idx])
        answer_length = self.df['answer_word_count'].iloc[idx]
        query_length = self.df['query_word_count'].iloc[idx]
        # query_tensor = torch.cat([self.word_to_tensor(word) for word in query_words])
        # answer_tensor = torch.cat([self.word_to_tensor(word) for word in answer_words])
        
        return {
            'query': query_words,
            'answer': answer_words,
            'query_length': query_length,
            'answer_length': answer_length
        }

    def word_to_tensor(self, word):
        """Convert a word into a tensor index for the embedding layer"""
        if word in self.word2index:
            return torch.tensor([self.word2index[word]], dtype=torch.long)
        else:
            return torch.tensor([self.word2index["unk"]], dtype=torch.long)  # Handle OOV words
        