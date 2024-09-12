import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from gigmate.utils.constants import get_params
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

MODEL_FILE_NAME = 'output/model.chk'

def look_ahead_mask(size: int) -> torch.BoolTensor:  
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
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
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, attn_mask=mask, is_causal=True, need_weights=False)
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
        self.pos_encoding = PositionalEncoding1D(d_model)

    def forward(self, inputs):

        # Combine embeddings
        x = self.embedding(inputs)

        # Add positional encoding
        x = Summer(self.pos_encoding)(x).to(x.device)

        sequence_length = x.size(1)
        mask = look_ahead_mask(sequence_length).to(x.device)

        # Pass through transformer blocks
        for layer in self.transformer_layers:
            x = layer(x, mask)

        # Final output layer
        x = self.dense(x)

        return x

def get_model(params = get_params(), checkpoint_path = None, device = None):
    model = TransformerModel(
        params['num_layers'],
        params['d_model'],
        params['num_heads'],
        params['dff'],
        params['vocab_size'],
        params['max_seq_len'],
        params['dropout_rate']
    )

    model.to(device)

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace("model._orig_mod.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict, strict=True)

    return model
