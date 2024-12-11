from torch import Tensor
import torch
from gigmate.utils.constants import get_end_of_sequence_token_id, get_pad_token_id, get_start_of_sequence_token_id


def remove_forbidden_tokens(outputs: Tensor, forbidden_tokens: list[int]) -> Tensor:
    outputs[..., forbidden_tokens] = float('-inf')
    return outputs


# logits: B, K, S, V shape or B, K, V shape
def sample_from_logits(logits: Tensor, temperature: float, no_special_tokens=False) -> Tensor:

    if no_special_tokens is True:
        forbidden_tokens = [get_start_of_sequence_token_id(), get_end_of_sequence_token_id(), get_pad_token_id()]
        logits = remove_forbidden_tokens(logits, forbidden_tokens)

    # If temp is 0 then next_token is the argmax of logits
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1)
    # If temp is not 0 then next_token is sampled out of logits
    else:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        # TODO: Vectorize this operation
        batches = []
        for b in range(probs.shape[0]):
            codebooks = []
            for c in range(probs.shape[1]):
                sample = torch.multinomial(probs[b, c], num_samples=1)
                codebooks.append(sample)
  
            batches.append(torch.stack(codebooks))
 
        next_token = torch.stack(batches).squeeze(-1)
 
    return next_token



