from typing import List, Optional, Tuple, cast
import torch
import torch.nn as nn
import torchao
from torch import Tensor
from gigmate.model.decoder import Decoder
from gigmate.model.encoder import Encoder
from gigmate.utils.constants import get_pad_token_id, get_params
from gigmate.utils.device import Device

ENABLE_QUANTIZATION = False

# [1]. Use alibi positional embeddings
# [2]. Use causal local attention
# [3]. Fix torch compile optimization
# [4]. Apply delay interleaving pattern
# [5]. Use Meta Encodec 32khz for encoding to tokens and decoding back to audio
# [6]. Apply quantization aware training: https://github.com/pytorch/ao
# [7]. Add support for stems
# [8]. Handle padding using appropriate attention mask or builting pytorch nested tensors
# [9]. Create music dataset
# # [1]. Find platform
# # [2]. Download songs on online storage
# # [3]. Create metadata consisting in song (multiple versions), (guitar stem)
# # [4]. Create pipeline step to add noise to the song and save separately
# # [5]. Encode everything with EnCodec 32khz
# # [6]. Publish
# # [7]. Write code for the dataset (splitting and loading of song tracks for training, validation and test)
# [10]. Define evaluation metrics (Frechet Audio Distance)
# [11]. Finish flex attention implementation
# 12. check out grouped query attention
# 13. Perform training
# 14. Evaluate model
# 15. Implement inference
# [16]. Implement encoder-decoder version


# Helper function for initializing weights. Ref: https://www.yadavsaurabh.com/good-initialisation-of/
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if hasattr(m, 'special_init'):
            nn.init.constant_(m.weight, 0)


class TransformerModel(nn.Module):
    def __init__(self, encoder_layers: int, decoder_layers: int, d_model: int, codebooks: int, num_heads: int, dff: int, vocab_size: int, batch_size: int, sliding_window_size: int, dropout: float = 0.1, padding_value=0):
        super(TransformerModel, self).__init__()

        self.num_heads = num_heads
        self.codebooks = codebooks
        self.num_layers = decoder_layers

        embeddings = [nn.Embedding(vocab_size, d_model, padding_idx=padding_value) for _ in range(codebooks)]
        self.embeddings = nn.ModuleList(embeddings)

        self.encoder = Encoder(encoder_layers, d_model, num_heads, dff, sliding_window_size, dropout)
        self.decoder = Decoder(decoder_layers, d_model, num_heads, dff, sliding_window_size, dropout)
        
        linears = [nn.Linear(d_model, vocab_size) for _ in range(codebooks)]
        self.linears = nn.ModuleList(linears)

        self.apply(init_weights)
    
    def forward(self, input: Tensor, conditioning_input: Tensor, sequence_lengths: Optional[List[int]] = None, use_cache: bool = False, cache: Optional[List[Tensor]] = None, cache_index: Optional[int] = None) -> Tuple[Tensor, List[Tensor]]:

        full_track_x = sum([self.embeddings[k](conditioning_input[:, k]) for k in range(self.codebooks)])
        x = sum([self.embeddings[k](input[:, k]) for k in range(self.codebooks)])

        # TODO: Implement kv cache for the encoder
        cross_attention_src, _ = self.encoder(full_track_x, sequence_lengths)
        x, updated_cache = self.decoder(x, sequence_lengths=sequence_lengths, cross_attention_src=cross_attention_src, use_cache=use_cache, cache=cache, cache_index=cache_index)

        logits = torch.stack([self.linears[k](x) for k in range(self.codebooks)], dim=1)  # [B, K, S, vocab_size]

        return logits, updated_cache


def load_ckpt(model, checkpoint_path: str, device: Device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)['state_dict']
    
    # Workaround to load the model
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "").replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)


def get_model(params=get_params(), checkpoint_path=None, device: Device = 'cpu', compile=True) -> TransformerModel:
    model = TransformerModel(
        encoder_layers=params['encoder_layers'],
        decoder_layers=params['decoder_layers'],
        d_model=params['d_model'],
        codebooks=params['codebooks'],
        num_heads=params['num_heads'],
        dff=params['dff'],
        vocab_size=params['vocab_size'],
        batch_size=params['batch_size'],
        sliding_window_size=params['sliding_window_size'],
        dropout=params['dropout_rate'],
        padding_value=get_pad_token_id(),
    )

    model.to(device)

    if checkpoint_path is not None:
        print(f'Loading ckpt {checkpoint_path}')
        load_ckpt(model, checkpoint_path, device)

    if compile is True:
        # TODO: fix torch compile full graph
        backend = 'aot_eager' if device == 'mps' or device == 'cuda' else 'inductor'
        model = cast(TransformerModel, torch.compile(model, fullgraph=False, backend=backend))

    # TODO: enable quantization if on CUDA
    if device == 'cuda':
        # Probably the next line is not needed when quantization is enabled as it would be taken care of by torchao
        torch.set_float32_matmul_precision('high')
        
        if ENABLE_QUANTIZATION is True:
            model = torchao.autoquant(model)

    return model
