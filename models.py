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
    
    def forward(self, query, answer, query_lengths=None, answer_lengths=None):
        query_embeddings = self.query_tower(query, query_lengths)
        answer_embeddings = self.answer_tower(answer, answer_lengths)
        return query_embeddings, answer_embeddings


class QADataset(torch.utils.data.Dataset):
    def __init__(self, query_answer_pairs, word2index):
        self.query_answer_pairs = query_answer_pairs
        self.word2index = word2index
    
    def __len__(self):
        return len(self.query_answer_pairs)
    
    def __getitem__(self, idx):
        query, answer = self.query_answer_pairs[idx]

        # Convert to tensors only when accessed
        query_words = query.split()
        query_tensor = torch.cat([self.word_to_tensor(word) for word in query_words])
        answer_words = answer.split()
        answer_tensor = torch.cat([self.word_to_tensor(word) for word in answer_words])
        
        return {
            'query': query_tensor,
            'answer': answer_tensor,
            'query_length': len(query_words),
            'answer_length': len(answer_words)
        }

    def word_to_tensor(self, word):
        """Convert a word into a tensor index for the embedding layer"""
        if word in self.word2index:
            return torch.tensor([self.word2index[word]], dtype=torch.long)
        else:
            return torch.tensor([self.word2index["unk"]], dtype=torch.long)  # Handle OOV words
        

# class BiRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, rnn_type='LSTM', dropout=0.2):
#         """
#         Bidirectional RNN (LSTM or GRU) model
        
#         Args:
#             input_size: Size of input features
#             hidden_size: Number of hidden units
#             num_layers: Number of RNN layers
#             num_classes: Number of output classes
#             rnn_type: Type of RNN ('LSTM' or 'GRU')
#             dropout: Dropout probability
#         """
#         super(BiRNN, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn_type = rnn_type
        
#         # Choose between LSTM and GRU
#         self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
#                              batch_first=True, bidirectional=True, dropout=dropout)
            
#         # The output layer - multiply hidden_size by 2 because of bidirectional
#         self.fc = nn.Linear(hidden_size * 2, num_classes)
        
#     def forward(self, x):
#         """
#         Forward pass
        
#         Args:
#             x: Input tensor of shape (batch_size, sequence_length, input_size)
            
#         Returns:
#             output: Tensor of shape (batch_size, num_classes)
#         """
#         # Initialize hidden state with zeros
#         batch_size = x.size(0)
#         h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
#             # Initialize cell state
#         c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
#         # Forward propagate LSTM
#         out, _ = self.rnn(x, (h0, c0))
        
#         # Decode the hidden state of the last time step
#         out = self.fc(out[:, -1, :])
#         return out