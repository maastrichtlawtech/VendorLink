"""
Python version : 3.8.12
Description : Contains the architecture of BiGRU Classifier trained on pre-trained BERT embeddings.
"""

# %% Importing Libraries
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# %% GRUCLassifier layer with Bert Embeddings
class BiGRUBertClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_units, max_seq_len, batch_size, n_layers, output_size, dropout=0.65):
        super(BiGRUBertClassifier, self).__init__()

        # self.embedding = nn.Embedding(len(vocab_size), embedding_size)
        self.hidden_units = hidden_units
        self.gru = nn.GRU(input_size=embedding_size,
                           hidden_size=hidden_units,
                           num_layers=n_layers,
                           batch_first=True,
                           bidirectional=True)
        self.output_size = output_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2*hidden_units, 4000)
        # self.fc2 = nn.Linear(4000, 768)
        self.label = nn.Linear(4000, output_size)

    def forward(self, text_emb):
        # text_emb = self.embedding(text)
        text_len = torch.tensor([self.max_seq_len]*self.batch_size)
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        output_forward = output[range(len(output)), text_len - 1, :self.hidden_units]
        output_reverse = output[:, 0, self.hidden_units:]
        output_cat = torch.cat((output_forward, output_reverse), 1)
        output = self.drop(output_cat)

        fc_out = self.fc(output)
        # fc_out = self.fc2(fc_out_)
        logits = self.label(fc_out)

        return logits


class BiGRUFasttextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 4000)
        self.fc2 = nn.Linear(4000, output_dim)
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, text, text_lengths):
        
        # text = [batch size, sent len]
        
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        packed_output, hidden = self.rnn(packed_embedded)
        
        #unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))
                
        #hidden = [batch size, hid dim * num directions]
            
        return output