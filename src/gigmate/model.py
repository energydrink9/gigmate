import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
import math
from gigmate.constants import get_params

MODEL_FILE_NAME = 'output/model.chk'

def look_ahead_mask(size: int) -> torch.FloatTensor:  
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask[mask.bool()] = -float('inf')
    return mask

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        sequence_length = x.size(1)
        mask = look_ahead_mask(sequence_length).to(x.device)

        attn_output, _ = self.mha(x, x, x, attn_mask=mask, is_causal=True)

        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)  # Use padding_idx for masking
        self.max_seq_len = max_seq_len
        self.transformer_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.dense = nn.Linear(d_model, vocab_size)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)

    def forward(self, inputs):

        # Combine embeddings
        x = self.embedding(inputs)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        
        # Pass through transformer blocks
        for layer in self.transformer_layers:
            x = layer(x)

        # Final output layer
        x = self.dense(x)

        return x

def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    
    pos_encoding = torch.zeros(1, seq_len, d_model)
    pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
    pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding

def get_model(params = get_params()):
    model = TransformerModel(
        params['num_layers'],
        params['d_model'],
        params['num_heads'],
        params['dff'],
        params['vocab_size'],
        params['max_seq_len'],
        params['dropout_rate']
    )

    return model