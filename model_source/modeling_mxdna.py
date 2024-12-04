""" PyTorch MxDNA model."""
# torch
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchvision.ops import deform_conv2d

# transformers
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils import ModelOutput
from transformers.activations import ACT2FN

# misc
import numpy as np
import math
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

from .configuration_mxdna import MxDNAConfig
from .kernel.build.BasicUnitNMS import basic_unit_nms_fn
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input

logger = logging.get_logger(__name__)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HELPER FUNCTIONS AND CONSTANTS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@dataclass
class MxDNAModelOutput(ModelOutput):
    router_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_mask_center: Optional[Tuple[torch.FloatTensor, ...]] = None
    nucleotide_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    nucleotide_attention_mask: Optional[torch.FloatTensor] = None
    token_hidden_states: torch.FloatTensor = None
    token_attention_mask: Optional[torch.FloatTensor] = None
    
@dataclass
class LanaguageModelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    router_logits: torch.FloatTensor = None

@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    router_logits: torch.FloatTensor = None
    
def load_balancing_loss_func_helper(router_logits: torch.Tensor, token_mask:torch.Tensor = None) -> float:
    # router_logits is a tensor of shape [batch_size , seq_len, num_experts]
    # token_mask is a tensor of shape [batch_size , seq_len], containing int of range [0, num_experts] representing the expert index for each token, where -1 is for padding tokens
    compute_device = router_logits.device
    batch_size, seq_len, num_experts = router_logits.size()
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)

    padding_mask = token_mask != -1
    
    # unpad the tokens
    routing_weights_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(routing_weights, padding_mask) # [num_tokens, num_experts]
    token_mask_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(token_mask.unsqueeze(-1), padding_mask) # [num_tokens]
    
    # cast
    token_mask_unpad = token_mask_unpad.squeeze(-1).to(torch.int64)
    
    # compute the fraction of tokens dispatched to each expert
    expert_mask = torch.nn.functional.one_hot(token_mask_unpad, num_classes=num_experts).float() # [num_tokens, num_experts]
    # cast to float32
    expert_mask = expert_mask.to(torch.float32)
    fraction_tokens_dispatched = torch.mean(expert_mask, dim=0) # [num_experts]
    # compute the fraciton of router probabilities allocated to each expert
    fraction_router_prob = torch.mean(routing_weights_unpad, dim=0) # [num_experts]
    
    load_balance_loss = torch.mean(fraction_router_prob * fraction_tokens_dispatched) * (num_experts ** 2)
    
    return load_balance_loss

def load_balancing_loss_func(
    router_logits: List[torch.Tensor], token_mask_center: List[torch.Tensor]
) -> float:
    return load_balancing_loss_func_helper(router_logits, token_mask_center)

def find_closest_factor(N, M):
    """Find the closest factor of N to M.

    Args:
        N (int): The number to find the closest factor of.
        M (int): The number to find the closest factor to.

    Returns:
        int: The closest factor of N to M.
    """
    closest_factor = None
    min_difference = float('inf')
    
    for d in range(1, int(N**0.5) + 1):
        if N % d == 0:
            # Check both the factor and its pair
            for factor in [d, N // d]:
                difference = abs(factor - M)
                if difference < min_difference:
                    min_difference = difference
                    closest_factor = factor
    
    return int(closest_factor)

def reorder_fn(sequences, attention_mask):
    """Reorder the sequences based on the attention mask: gather the valid tokens and put them in the front of the sequence.

    Args:
        sequences (List[torch.Tensor]): The sequences to reorder.
        attention_mask (torch.Tensor): The attention mask.

    Returns:
        Tuple[List[torch.Tensor], Callable]: The reordered sequences and the function to reorder back.
    """
    device = attention_mask.device
    num_seqs = len(sequences)

    batch_size,seq_len = attention_mask.shape[:2]
    attention_mask = attention_mask.bool()
    
    valid_counts = attention_mask.sum(dim = -1)
    max_length = valid_counts.max()
    
    output_attention_mask = torch.zeros((batch_size,max_length),dtype=attention_mask.dtype,device = device)
    
    range_tensor = torch.arange(max_length,device = device)
    
    range_tensor = range_tensor.unsqueeze(0).expand(batch_size, -1)
    
    valid_positions = range_tensor < valid_counts.unsqueeze(1)
    
    output_attention_mask[valid_positions] = True
    
    new_sequences = [torch.zeros([batch_size,max_length,*t.shape[2:]],dtype=t.dtype,device = device) for t in sequences]

    for s in range(num_seqs):
        new_sequences[s][output_attention_mask] = sequences[s][attention_mask]
    
    return new_sequences

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_isolated(x, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch_size,
    num_key_value_heads, seq_len, head_dim) to (batch_size, num_key_value_heads * n_rep, seq_len,
    """
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# BASIC BLOCK
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class MxDNALayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
    
    def forward(self, hidden_states):
        return self.norm(hidden_states)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->MxDNA
class MxDNARotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class MxDNAFlashAttention2(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. 
    """

    def __init__(self, config: MxDNAConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
            
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.projection_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.projection_k = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.projection_v = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.projection_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MxDNARotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )



    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):

        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.projection_q(hidden_states)
        key_states = self.projection_k(hidden_states)
        value_states = self.projection_v(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = seq_len
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids[:,:rotary_seq_len])

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.projection_q.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            seq_len,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size).contiguous()

        attn_output = self.projection_o(attn_output)

        return attn_output

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        seq_len_q,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states,indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_states, attention_mask)
            key_states, indices_k, cu_seqlens_k, max_seqlen_in_batch_k = unpad_input(key_states, attention_mask)
            value_states, indices_v, cu_seqlens_v, max_seqlen_in_batch_v = unpad_input(value_states, attention_mask)

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=False,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, seq_len_q)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=False,
            )

        return attn_output
        
class MxDNAFlashCrossAttention2(nn.Module):
    def __init__(self, config: MxDNAConfig, layer_idx: Optional[int] = None):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.projection_q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.projection_k = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.projection_v = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.projection_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = MxDNARotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    def forward(
        self,
        hidden_states_q: torch.Tensor,
        hidden_states_kv: torch.Tensor,
        attention_mask_q: Optional[torch.Tensor] = None,
        attention_mask_kv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        
        """It is a cross-attention layer that applies the Flash Attention mechanism.

        Args:
            hidden_states_q (`torch.Tensor`):
                The query hidden states of shape `(batch_size, seq_len_q, hidden_size)`.
            hidden_states_kv (`torch.Tensor`):
                The key and value hidden states of shape `(batch_size, seq_len_kv, hidden_size)`.
            attention_mask_query (`torch.Tensor`, `optional`):
                The query padding mask of shape `(batch_size, seq_len_q)`.
            attention_mask_kv (`torch.Tensor`, `optional`):
                The key and value padding mask of shape `(batch_size, seq_len_kv)`.
            position_ids (`torch.LongTensor`, `optional`):
                The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
        """
        batch_size, seq_len_q, _ = hidden_states_q.size()
        batch_size, seq_len_kv, _ = hidden_states_kv.size()  
        
        query_states = self.projection_q(hidden_states_q)
        key_states = self.projection_k(hidden_states_kv)
        value_states = self.projection_v(hidden_states_kv)  

        query_states = query_states.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len_kv, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len_kv, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        
        rotary_seqeuence_length_q = seq_len_q
        cos_q, sin_q = self.rotary_emb(query_states, seq_len=rotary_seqeuence_length_q)
        
        rotary_seq_len_kv = seq_len_kv
        cos_kv, sin_kv = self.rotary_emb(value_states, seq_len=rotary_seq_len_kv)

        query_states = apply_rotary_pos_emb_isolated(query_states, cos_q, sin_q, position_ids[:,:rotary_seqeuence_length_q])
        key_states = apply_rotary_pos_emb_isolated(key_states, cos_kv, sin_kv, position_ids[:,:rotary_seq_len_kv])


        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.projection_q.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask_kv,
            attention_mask_q,
            seq_len_kv,
            seq_len_q,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(batch_size,seq_len_q, self.hidden_size).contiguous()
        
        attn_output = self.projection_o(attn_output)

        return attn_output
    
    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask_kv,
        attention_mask_q,
        seq_len_kv,
        seq_len_q,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask_kv (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len_kv)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            attention_mask_q (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len_q)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            seq_len_kv (`int`):
                The length of the key value sequence
            seq_len_q (`int`):
                The length of the query sequence
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        
        # Contains at least one padding token in the sequence
        if attention_mask_kv is None and attention_mask_q is None:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=False
            )
        else:
            if attention_mask_kv is None:
                attention_mask_kv = torch.ones(query_states.shape[0], seq_len_kv).to(query_states.device)
            if attention_mask_q is None:
                attention_mask_q = torch.ones(query_states.shape[0], seq_len_q).to(query_states.device)
            batch_size = query_states.shape[0]
            query_states, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_states, attention_mask_q)
            key_states, indices_k, cu_seqlens_k, max_seqlen_in_batch_k = unpad_input(key_states, attention_mask_kv)
            value_states, indices_v, cu_seqlens_v, max_seqlen_in_batch_v = unpad_input(value_states, attention_mask_kv)
        
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=False
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, seq_len_q)
        
        return attn_output


class MxDNAFFNBlock(nn.Module):
    def __init__(self, config: MxDNAConfig):
        super().__init__()
        self.ffn = MxDNAMLP(config)
        
    def forward(self, hidden_states,attention_mask = None):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if attention_mask is not None:
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask)
        current_hidden_states = self.ffn(hidden_states)
        if attention_mask is not None:
            current_hidden_states = pad_input(current_hidden_states, indices, batch_size, seq_len)
        return current_hidden_states

class MxDNAMLP(nn.Module):
    def __init__(self, config: MxDNAConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size


        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class MxDNAConvNet(nn.Module):
    """
    This is a convolutional network that is used as an expert in the MOE block. Pointwise Conv -> GLU -> grouped Conv -> LayerNorm -> Swish -> Pointwise Conv
    """
    def __init__(self, config, kernel_size: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.pointwise_conv_pre = nn.Linear(hidden_dim, hidden_dim * 2,bias=False)
        self.glu = nn.GLU(dim=-1)
        self.grouped_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, groups=find_closest_factor(hidden_dim,kernel_size),stride=kernel_size,bias=False)
        # initialize the weights normal
        torch.nn.init.trunc_normal_(self.grouped_conv.weight,mean=0.0,std=config.initializer_range)
        self.norm = nn.LayerNorm(hidden_dim)
        self.swish = nn.SiLU()
        self.pointwise_conv_post = nn.Linear(hidden_dim, hidden_dim,bias=False)
        
    def forward(self, hidden_states: torch.Tensor):
        """Forward pass of the convolutional network.

        Args:
            hidden_states (torch.Tensor): The input hidden states. Shape: (batch_size, seq_len, hidden_dim)
            
        Returns:
            torch.Tensor: The output hidden states. Shape: (batch_size, seq_len, hidden_dim)
        """
        hidden_states = self.pointwise_conv_pre(hidden_states)
        hidden_states = self.glu(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2)
        hidden_states = self.grouped_conv(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).contiguous()
        hidden_states = self.norm(hidden_states)
        hidden_states = self.swish(hidden_states)
        hidden_states = self.pointwise_conv_post(hidden_states)
        return hidden_states
    
        
class MxDNAConvMoeBlock(nn.Module):
    """
    This is a MOE block with convolution layers as the experts. The block applies the experts to the input hidden states based on the router logits and basic_unit masks.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.expert_kernel_sizes =  config.expert_kernel_sizes
        # Define experts as a ModuleList of MxDNAConvNet of different kernel sizes
        self.experts = nn.ModuleList([MxDNAConvNet(config, kernel_size, self.hidden_dim) for kernel_size in config.expert_kernel_sizes])

    def forward(self,hidden_states: torch.Tensor,router_logits , basic_unit_mask_center, basic_unit_mask_all):
        """Forward pass of the MOE block. It is sparsely activated by extracting the hidden states of nucleotides belonging to the same basic_unit and applying the expert to the extracted hidden states.
        Extracting and placing back the hidden states of nucleotides belonging to the same basic_unit is done by the unpad_input and pad_input functions.
        Applying the expert (kernel size == stride) to the extracted hidden states of nucleotides belonging is made possible by the non-overlapping nms

        Args:
            hidden_states (torch.Tensor): The input hidden states. Shape: (batch_size, seq_len, hidden_dim)
            router_logits (torch.Tensor): The logits of the routers. Shape: (batch_size, seq_len, num_experts)
            basic_unit_mask_center (torch.Tensor): The basic_unit mask for the center nucleotide of each basic_unit. Shape: (batch_size, seq_len)
            basic_unit_mask_all (torch.Tensor): The basic_unit mask for all nucleotides in each basic_unit. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: The output hidden states. Shape: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        router_weights = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]

        # Initialize the final output hidden states with zeros
        final_hidden_states = torch.zeros_like(hidden_states)
        
        for expert_idx, expert in enumerate(self.experts):
            
            # extract the hidden states of nucleotides belonging to the basic_units with corresponding lengths
            expert_input_unpad, indices, cu_seqlens, max_seqlen_in_batch= unpad_input(hidden_states, basic_unit_mask_all==expert_idx)  
                      
            # if no nucleotide belongs to the basic_unit, skip the expert
            if expert_input_unpad.size(0) == 0:
                continue
            
            # Initialize the extracted final hidden states with zeros
            final_hidden_states_unpad = torch.zeros_like(expert_input_unpad)

            # extract the basic_unit mask for the center nucleotide positions of the basic_units
            basic_unit_mask_center_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(basic_unit_mask_center.unsqueeze(-1), basic_unit_mask_all==expert_idx)
            
            # apply the expert to the extracted hidden states of nucleotides belonging to the basic_unit
            # since the expert are convolution with stride = kernel_size, the nucleotides are succesfully aggregated
            expert_output = expert(expert_input_unpad)
            
            # update the extracted final hidden states with the expert output at the center nucleotide positions
            final_hidden_states_unpad[basic_unit_mask_center_unpad.squeeze(-1) == expert_idx] = expert_output.to(final_hidden_states_unpad.dtype)
            
            # unpad the extracted final hidden states and add them to the final output hidden states
            final_hidden_states += pad_input(final_hidden_states_unpad, indices, batch_size, seq_len) * ((router_weights[:, :, expert_idx]).unsqueeze(-1))
                
        return final_hidden_states

class MxDNADeforambleConvBlock(nn.Module):
    """
    This is a block that applies deformable convolution to the input hidden states. It adapts the orignal two-dimensional deformable convolution (with built-in implementation of torchvision) to the one-dimensional case.
    """
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size

        self.kh = config.deformable_conv_kernel_size
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.offset_conv = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.kh, kernel_size=self.kh, padding=self.kh // 2, bias=True)
        self.modulator_conv = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.kh, kernel_size=self.kh, padding=self.kh // 2, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.regular_conv = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(self.kh, 1), padding=(self.kh // 2, 0), bias=False)
        torch.nn.init.trunc_normal_(self.regular_conv.weight, mean = 0.0, std = config.initializer_range)

    def forward(self, hidden_states, attention_mask, special_tokens_mask):
        """Forward pass of the deformable convolution block.

        Args:
            hidden_states (torch.Tensor): The input hidden states. Shape: (batch_size, seq_len, hidden_dim)
            attention_mask (torch.Tensor): The attention mask. Shape: (batch_size, seq_len)
            special_tokens_mask (torch.Tensor): The special tokens mask. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: The output hidden states. Shape: (batch_size, seq_len, hidden_dim)
        """
        
        batch_size, seq_len, hidden_dim = hidden_states.size()
        residual = hidden_states
        
        # prenorm
        hidden_states = self.input_layernorm(hidden_states)
    
        hidden_states = hidden_states.transpose(-2, -1)  # Transpose to put channels in the second dimension

        x_offsets = self.offset_conv(hidden_states)  # Generate x offsets for deformable convolution
        modulator = 2 * torch.sigmoid(self.modulator_conv(hidden_states))  # Generate modulation parameters

        # Prepare interleaved offset array with zero y offsets
        zero_y_offsets = torch.zeros_like(x_offsets)  # Create a zero tensor for the y component
        offset = torch.stack((x_offsets, zero_y_offsets), dim=2)  # Interleave x and y
        offset = offset.reshape(batch_size, 2*self.kh, seq_len)  # Reshape to fit deform_conv2d input requirements
        offset = offset.unsqueeze(-1)

        modulator = modulator.unsqueeze(-1) # Repeat modulation across the kernel width

        # Mask hidden states to avoid processing special and masked tokens
        masked_hidden_states = hidden_states * (1 - special_tokens_mask[:, None, :]) * attention_mask[:, None, :]
        
        # Reshape masked_hidden_states to match the input requirement of deform_conv2d
        masked_hidden_states = masked_hidden_states.unsqueeze(-1)  # Add an extra dimension to simulate width of 1

        # Apply deformable convolution
        out = deform_conv2d(input=masked_hidden_states, offset=offset, mask=modulator, weight=self.regular_conv.weight, bias=self.regular_conv.bias,
                            padding=(self.kh // 2, 0), stride=1, dilation=1)

        # Remove the extra width dimension and transpose back to original dimension ordering
        out = out.squeeze(-1).transpose(-2, -1)  # Removing width and swapping back dimensions

        # Residual connection
        out = out + residual
        return out


class LMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x
    
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODEL LAYER
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class MxDNALearntTokenizationLayer(nn.Module):
    """
    This is a layer that learns the tokenization of the input sequence. Scoring (router) -> Selection (nms) -> Aggregation (experts) -> Deformable Convolution (assembly)
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.deformable_conv_kernel_size = config.deformable_conv_kernel_size
        self.expert_kernel_sizes = np.array(config.expert_kernel_sizes,dtype=np.int32)
        self.jitter_noise = config.router_jitter_noise
        self.basic_unit_masked_token_embedding = nn.Parameter(torch.zeros(self.hidden_size))
        # initialize the weights normal
        torch.nn.init.trunc_normal_(self.basic_unit_masked_token_embedding,mean=0.0,std=config.initializer_range)

        self.gate = nn.Linear(config.hidden_size,config.num_local_experts,bias=False)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.conv_moe = MxDNAConvMoeBlock(config)
        self.deform_conv = MxDNADeforambleConvBlock(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        **kwargs,
    ):
        """Forward pass of the learnt tokenization layer.

        Args:
            hidden_states (torch.Tensor): The input hidden states. Shape: (batch_size, seq_len, hidden_dim)
            attention_mask (torch.Tensor): The attention mask. Shape: (batch_size, seq_len).
            special_tokens_mask (torch.Tensor): The special tokens mask. Shape: (batch_size, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable]: The output hidden states, attention mask, special tokens mask, and the reorder back function.
            
        """


        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Initialize the basic_unit_mask with the special_tokens_mask
        basic_unit_mask_cpu = special_tokens_mask.clone().detach().cpu().numpy().astype(np.int32)
    
        # Convert the tensors to numpy arrays
        attention_mask_cpu = attention_mask.clone().detach().cpu().numpy().astype(np.int32)

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states_group = hidden_states * torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        else:
            hidden_states_group = hidden_states
              
        router_logits = self.gate(hidden_states_group * (1 - special_tokens_mask[ :, :,None]))   

        # Convert the tensors to numpy arrays
        router_logits_cpu = router_logits.clone().detach().cpu().numpy().astype(np.float32)

        # apply nms to select the most significant basic_units
        basic_unit_mask_center_cpu, basic_unit_mask_all_cpu = basic_unit_nms_fn(router_logits_cpu, self.expert_kernel_sizes, attention_mask_cpu)
        
        # Update the basic_unit_mask with the center nucleotide positions of the selected basic_units
        basic_unit_mask_cpu = basic_unit_mask_cpu + (basic_unit_mask_center_cpu!=-1)
        
        # Convert the numpy arrays back to tensors
        basic_unit_mask_center = torch.tensor(basic_unit_mask_center_cpu, device=hidden_states.device)
        basic_unit_mask_all = torch.tensor(basic_unit_mask_all_cpu, device=hidden_states.device)

        # apply the experts to the hidden states based on the router logits and basic_unit masks, this aggregating the hidden states of the nucleotides belonging to the same basic_unit
        final_hidden_states = self.conv_moe(hidden_states,router_logits,basic_unit_mask_center,basic_unit_mask_all)
            
        hidden_states = residual + final_hidden_states
        attention_mask = torch.tensor((basic_unit_mask_cpu*attention_mask_cpu)!=0, device=hidden_states.device,dtype=torch.long)

        seqs = reorder_fn(sequences=[hidden_states, attention_mask,special_tokens_mask], attention_mask=attention_mask)
        
        hidden_states, attention_mask,special_tokens_mask = seqs
        
        hidden_states = self.deform_conv(hidden_states, attention_mask, special_tokens_mask)
        
        return hidden_states, attention_mask, special_tokens_mask,router_logits,basic_unit_mask_center


class MxDNAEncoderLayer(nn.Module):
    def __init__(self, config: MxDNAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MxDNAFlashAttention2(config, layer_idx)
        self.ffn = MxDNAFFNBlock(config)
        self.input_layernorm = MxDNALayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = MxDNALayerNorm(config.hidden_size, eps=config.layer_norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, seq_len)` where padding elements are indicated by 0.
        """

        # MHA
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = residual + hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states,attention_mask)
        hidden_states = residual + hidden_states

        return hidden_states
    
class MxDNADecoderLayer(nn.Module):
    def __init__(self, config: MxDNAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cross_attn = MxDNAFlashCrossAttention2(config, layer_idx)
        self.ffn = MxDNAFFNBlock(config)
        self.input_layernorm = MxDNALayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = MxDNALayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states_q: torch.Tensor,
        hidden_states_kv: torch.Tensor,
        attention_mask_q: Optional[torch.Tensor] = None,
        attention_mask_kv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
            
        residual = hidden_states_q
        hidden_states_q = self.input_layernorm(hidden_states_q)
        hidden_states_kv = self.input_layernorm(hidden_states_kv)
        hidden_states_q = self.cross_attn(
            hidden_states_q=hidden_states_q,
            hidden_states_kv=hidden_states_kv,
            attention_mask_query=attention_mask_q,
            attention_mask_kv=attention_mask_kv,
            position_ids=position_ids,
        )
        hidden_states_q = residual + hidden_states_q
        
        residual = hidden_states_q
        hidden_states_q = self.post_attention_layernorm(hidden_states_q)
        hidden_states_q = self.ffn(hidden_states_q,attention_mask_q)
        hidden_states_q = residual + hidden_states_q
        
        return hidden_states_q
        

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODEL
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class MxDNAPreTrainedModel(PreTrainedModel):
    config_class = MxDNAConfig
    base_model_prefix = "model"
    _no_split_modules = ["MxDNALearntTokenizationLayer", "MxDNADecoderLayer", "MxDNAEncoderLayer"]
    _supports_flash_attn_2 = True
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight,mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight,mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MxDNAModel(MxDNAPreTrainedModel):
    """
    Transformer encoder consisting of *config.num_hidden_layers* layers. Each layer is a ["MxDNALearntTokenizationLayer", "MxDNADecoderLayer", "MxDNAEncoderLayer"]

    Args:
        config: MxDNAConfig
    """

    def __init__(self, config: MxDNAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        nucleotide_layers = []
        token_layers = []
        
        self.conversion_layer_idx = config.conversion_layer_idx
        for layer_idx in range(0, self.conversion_layer_idx):
            nucleotide_layers.append(MxDNAEncoderLayer(config, layer_idx))
        self.nucleotide_layers = nn.ModuleList(nucleotide_layers)
        
        self.conversion_layer = MxDNALearntTokenizationLayer(config, self.conversion_layer_idx)
        
        for layer_idx in range(self.conversion_layer_idx+1, config.num_hidden_layers):
            token_layers.append(MxDNAEncoderLayer(config, layer_idx))
        self.token_layers = nn.ModuleList(token_layers)
        
        self.norm = MxDNALayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, MxDNAModelOutput]:

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = attention_mask
     
        hidden_states = inputs_embeds

        nucleotide_attention_mask = attention_mask
        # nucleotide layers
        for nucleotide_layer in self.nucleotide_layers:
            hidden_states = nucleotide_layer(
                    hidden_states,
                    attention_mask=nucleotide_attention_mask,
                    position_ids=position_ids,
                )
            
        nucleotide_hidden_states = hidden_states
        
        hidden_states, token_attention_mask, token_special_tokens_mask,router_logits, token_mask_center = self.conversion_layer(
                hidden_states,
                attention_mask=nucleotide_attention_mask,
                special_tokens_mask = special_tokens_mask,
        )

        # token layers
        for token_layer in self.token_layers:
            
            hidden_states = token_layer(
                    hidden_states,
                    attention_mask=token_attention_mask,
                    position_ids=position_ids,
                )
            
        token_hidden_states = self.norm(hidden_states)

        return MxDNAModelOutput(
            token_hidden_states=token_hidden_states,
            nucleotide_hidden_states=nucleotide_hidden_states,
            token_attention_mask=token_attention_mask,
            nucleotide_attention_mask=nucleotide_attention_mask,
            router_logits=router_logits,
            token_mask_center=token_mask_center,
        )

class MxDNAForMaskedLM(MxDNAPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = MxDNAModel(config)
        self.vocab_size = config.vocab_size
        self.decoder = MxDNADecoderLayer(config,config.num_hidden_layers)
        self.lm_head = LMHead(config)
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm = MxDNALayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        load_balancing_loss: Optional[bool] = None,
    ) -> Union[Tuple, LanaguageModelingOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""
        
        batch_size,seq_len = input_ids.shape
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )


        token_hidden_states = outputs.token_hidden_states
        nucleotide_hidden_states = outputs.nucleotide_hidden_states
        token_attention_mask = outputs.token_attention_mask
        nucleotide_attention_mask = attention_mask

        last_hidden_states = self.decoder(
            hidden_states_q=nucleotide_hidden_states,
            hidden_states_kv=token_hidden_states,
            attention_mask_q=nucleotide_attention_mask,
            attention_mask_kv=token_attention_mask,
            position_ids=position_ids,
        )
        
        last_hidden_states = self.norm(last_hidden_states)
        
        logits = self.lm_head(last_hidden_states)
        logits = logits.float()


        loss = None
        lm_loss = None        
        aux_loss = None
        
        if labels is not None:
            # No shift for masked language modeling
            mlm_logits = logits[..., :, :].contiguous()
            mlm_labels = labels[..., :].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            mlm_logits = mlm_logits.view(-1, self.config.vocab_size)
            mlm_labels = mlm_labels.view(-1)
            # Enable model parallelism
            mlm_labels = mlm_labels.to(mlm_logits.device)
            lm_loss = loss_fct(mlm_logits, mlm_labels)
        
        loss = lm_loss

        if  load_balancing_loss:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                outputs.token_mask_center,)

            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
        
        return LanaguageModelingOutput(
            loss=loss,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            logits=logits,
            router_logits = outputs.router_logits,
        )
        
class MxDNAForSequenceClassification(MxDNAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MxDNAModel(config)
        # self.decoder = MxDNADecoderLayer(config,config.num_hidden_layers)
        self.task_head= ClassificationHead(config)
        # Initialize weights and apply final processing
        self.norm = MxDNALayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        load_balancing_loss: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        batch_size,seq_len = input_ids.shape
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()
            
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        token_hidden_states = outputs.token_hidden_states

        last_hidden_states = token_hidden_states
        last_hidden_states = self.norm(last_hidden_states)
        pooled_logits = self.task_head(last_hidden_states[:,0])
        loss = None
        cls_loss = None
        aux_loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    cls_loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    cls_loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                cls_loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                cls_loss = loss_fct(pooled_logits, labels)
                
        loss = cls_loss

        if  load_balancing_loss:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                outputs.token_mask_center,)
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
                
        return SequenceClassifierOutput(
            loss=loss,
            cls_loss=cls_loss,
            aux_loss=aux_loss,
            logits=pooled_logits,
            router_logits=outputs.router_logits,
        )
        