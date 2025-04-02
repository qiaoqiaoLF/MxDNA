""" PyTorch MxDNA model."""
import warnings
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import ModelOutput
from transformers.activations import ACT2FN
import numpy as np

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)
from .configuration_mxdna import MxDNAConfig
from torchvision.ops import nms, batched_nms

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input  # noqa
from .MotifMaskingModule.MotifMasking import motif_masking_fn

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MxDNAConfig"
@dataclass
class MxDNAModelOutput(ModelOutput):
    
    motif_hidden_states: torch.FloatTensor = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    motif_mask_center_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    nucleotide_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    nucleotide_attention_mask: Optional[torch.FloatTensor] = None
    motif_attention_mask: Optional[torch.FloatTensor] = None
    
@dataclass
class LanaguageModelingOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    z_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    router_logits: torch.FloatTensor = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    z_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    router_logits: torch.FloatTensor = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    


def router_z_loss_func_helper(router_logits: torch.Tensor, attention_mask: torch.Tensor = None) -> float:
    
    # unpad the router logits
    if attention_mask is not None:
        router_logits_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(router_logits, attention_mask)
    else:
        router_logits_unpad = router_logits.reshape(-1, router_logits.size(-1))
        
    
    num_tokens, num_experts = router_logits_unpad.size()
    log_z = torch.logsumexp(router_logits_unpad, dim=-1)
    z_loss = log_z**2
    return torch.mean(z_loss)

def router_z_loss_func(
    router_logits_list: List[torch.Tensor], attention_mask_list: List[torch.Tensor]
) -> float:
    z_loss = 0.0
    for router_logits, attention_mask in zip(router_logits_list, attention_mask_list):
        z_loss += router_z_loss_func_helper(router_logits, attention_mask)
    return z_loss / len(router_logits_list)

def load_balancing_loss_func_helper(router_logits: torch.Tensor, motif_mask:torch.Tensor = None) -> float:
    # router_logits is a tensor of shape [batch_size , seq_len, num_experts]
    # motif_mask is a tensor of shape [batch_size , seq_len], containing int of range [0, num_experts] representing the expert index for each token, where -1 is for padding tokens
    compute_device = router_logits.device
    batch_size, seq_len, num_experts = router_logits.size()
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)

    padding_mask = motif_mask != -1
    
    # unpad the tokens
    routing_weights_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(routing_weights, padding_mask) # [num_tokens, num_experts]
    motif_mask_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(motif_mask.unsqueeze(-1), padding_mask) # [num_tokens]
    
    # cast
    motif_mask_unpad = motif_mask_unpad.squeeze(-1).to(torch.int64)
    
    # compute the fraction of tokens dispatched to each expert
    expert_mask = torch.nn.functional.one_hot(motif_mask_unpad, num_classes=num_experts).float() # [num_tokens, num_experts]
    # cast to float32
    expert_mask = expert_mask.to(torch.float32)
    fraction_tokens_dispatched = torch.mean(expert_mask, dim=0) # [num_experts]
    # compute the fraciton of router probabilities allocated to each expert
    fraction_router_prob = torch.mean(routing_weights_unpad, dim=0) # [num_experts]

    # compute the loss: \sum_{i = 1}^{num_experts} |fraction_tokens_dispatched[i] * fraction_router_prob[i]} * num_experts
    # ideally fraction_tokens_dispatched[i] should be equal to fraction_router_prob[i] to 1/num_experts
    # Thus the loss should be \sum_{i = 1}^{num_experts} (1 / num_experts * 1 / num_experts) * num_experts = 1
    
    load_balance_loss = torch.mean(fraction_router_prob * fraction_tokens_dispatched) * (num_experts ** 2)
    
    return load_balance_loss

def load_balancing_loss_func(
    router_logits_list: List[torch.Tensor], motif_mask_center_list: List[torch.Tensor]
) -> float:
    load_balancing_loss = 0.0
    for router_logits, motif_mask_center in zip(router_logits_list, motif_mask_center_list):
        load_balancing_loss += load_balancing_loss_func_helper(router_logits, motif_mask_center)
    return load_balancing_loss / len(router_logits_list)


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
        output_attentions: bool = False,
        **kwargs,
    ):

        batch_size, seq_len, _ = hidden_states.size()

        # unpad the hidden states
        if attention_mask is not None:
            hidden_states, indices, cu_seqlens, max_seq_lens_in_batch = unpad_input(hidden_states, attention_mask)
        query_states = self.projection_q(hidden_states)
        key_states = self.projection_k(hidden_states)
        value_states = self.projection_v(hidden_states)
        # pad the hidden states
        if attention_mask is not None:
            query_states = pad_input(query_states, indices, batch_size, seq_len)
            key_states = pad_input(key_states, indices, batch_size, seq_len)
            value_states = pad_input(value_states, indices, batch_size, seq_len)

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
        if attention_mask is not None:
            attn_output, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(attn_output, attention_mask)
        attn_output = self.projection_o(attn_output)
        if attention_mask is not None:
            attn_output = pad_input(attn_output, indices, batch_size, seq_len)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

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
        output_attentions: bool = False,
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
            output_attentions (`bool`, `optional`, defaults to `False`):
        """
        batch_size, seq_len_q, _ = hidden_states_q.size()
        batch_size, seq_len_kv, _ = hidden_states_kv.size()
        # unpad the hidden states
        if attention_mask_q is not None:
            hidden_states_q, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(hidden_states_q, attention_mask_q)
        if attention_mask_kv is not None:
            hidden_states_kv, indices_kv, cu_seqlens_kv, max_seqlen_in_batch_kv = unpad_input(hidden_states_kv, attention_mask_kv)
            
        
        query_states = self.projection_q(hidden_states_q)
        key_states = self.projection_k(hidden_states_kv)
        value_states = self.projection_v(hidden_states_kv)
        
        # pad the hidden states
        if attention_mask_q is not None:
            query_states = pad_input(query_states, indices_q, batch_size, seq_len_q)
        if attention_mask_kv is not None:
            key_states = pad_input(key_states, indices_kv, batch_size, seq_len_kv)
            value_states = pad_input(value_states, indices_kv, batch_size, seq_len_kv)
        
        

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
        
        if attention_mask_q is not None:
            attn_output, indices_out, cu_seqlens_out, max_seqlen_in_batch_out = unpad_input(attn_output, attention_mask_q)
        attn_output = self.projection_o(attn_output)
        if attention_mask_q is not None:
            attn_output = pad_input(attn_output, indices_out, batch_size, seq_len_q)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
    
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

    

def find_closest_factor(N, M):
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


class MxDNAConvNet(nn.Module):
    def __init__(self, config: MxDNAConfig, kernel_size: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.pointwise_conv_pre = nn.Linear(hidden_dim, hidden_dim * 2,bias=False)
        self.glu = nn.GLU(dim=-1)
        self.depthwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, groups=find_closest_factor(hidden_dim,kernel_size),stride=kernel_size,bias=False)
        # initialize the weights normal
        torch.nn.init.trunc_normal_(self.depthwise_conv.weight,mean=0.0,std=config.initializer_range)
        self.norm = MxDNALayerNorm(hidden_dim)
        self.swish = nn.SiLU()
        self.pointwise_conv_post = nn.Linear(hidden_dim, hidden_dim,bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional network.

        Args:
            hidden_states (torch.Tensor): The input hidden states. Shape: (batch_size, seq_len, hidden_dim)
            1 for special tokens and 0 for non-special tokens. Shape: (batch_size, seq_len). Shape: (batch_size, seq_len)
            

        Returns:
            torch.Tensor: The output hidden states. Shape: (batch_size, seq_len, hidden_dim)
        """
         # (batch_size, hidden_dim, seq_len)
        # we do not want to let convolution to interact with special tokens
        hidden_states = self.pointwise_conv_pre(hidden_states)
        hidden_states = self.glu(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2)
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).contiguous()
        hidden_states = self.norm(hidden_states)
        hidden_states = self.swish(hidden_states)
        hidden_states = self.pointwise_conv_post(hidden_states)
        return hidden_states
    
        
class MxDNAConvMoeBlock(nn.Module):
    """
    This is a MOE block with convolution layers as the experts.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.expert_kernel_sizes =  config.expert_kernel_sizes
        # Define experts as a ModuleList of convolutional layers (or blocks)
        self.experts = nn.ModuleList([MxDNAConvNet(config, kernel_size, self.hidden_dim) for kernel_size in config.expert_kernel_sizes])
        # self.masks_pooling = nn.ModuleList([nn.MaxPool1d( kernel_size,stride=kernel_size) for kernel_size in config.expert_kernel_sizes])

    def forward(self,hidden_states: torch.Tensor,router_logits , motif_mask_center, motif_mask_all,masked_tokens_mask) -> torch.Tensor:

        batch_size, seq_len, hidden_dim = hidden_states.shape
        router_weights = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]

        final_hidden_states = torch.zeros_like(hidden_states)
        
        process_masked_tokens_mask =  masked_tokens_mask!=None
        if process_masked_tokens_mask:
            final_masked_tokens_mask = torch.zeros_like(masked_tokens_mask)
        else:
            final_masked_tokens_mask= None

        # Efficiently compute contributions from top-k experts
        for expert_idx, expert in enumerate(self.experts):
            expert_input_unpad, indices, cu_seqlens, max_seqlen_in_batch= unpad_input(hidden_states, motif_mask_all==expert_idx)            
            # if no token has the kernel size, we skip the expert
            
            if expert_input_unpad.size(0) == 0:
                continue
            
            final_hidden_states_unpad = torch.zeros_like(expert_input_unpad)

            motif_mask_center_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(motif_mask_center.unsqueeze(-1), motif_mask_all==expert_idx)
            
            expert_output = expert(expert_input_unpad)
            
            final_hidden_states_unpad[motif_mask_center_unpad.squeeze(-1) == expert_idx] = expert_output.to(final_hidden_states_unpad.dtype)
            final_hidden_states += pad_input(final_hidden_states_unpad, indices, batch_size, seq_len) * ((router_weights[:, :, expert_idx]).unsqueeze(-1))
            
            if process_masked_tokens_mask:

                masked_tokens_mask_unpad, indices, cu_seqlens, max_seqlen_in_batch= unpad_input(masked_tokens_mask.unsqueeze(-1), motif_mask_all==expert_idx)
                final_masked_tokens_mask_unpad = torch.zeros_like(masked_tokens_mask_unpad)
                masks_pooling_output = self.masks_pooling[expert_idx](masked_tokens_mask_unpad.float().transpose(-2,-1)).transpose(-2,-1).long().contiguous()
                final_masked_tokens_mask_unpad[motif_mask_center_unpad.squeeze(-1) == expert_idx] = masks_pooling_output.to(final_masked_tokens_mask_unpad.dtype)
                final_masked_tokens_mask += pad_input(final_masked_tokens_mask_unpad,indices, batch_size, seq_len).squeeze(-1)
                
        return final_hidden_states,final_masked_tokens_mask

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
    
from torchvision.ops import deform_conv2d

class MxDNADeforambleConvBlock(nn.Module):
    def __init__(self, config: MxDNAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.kh = config.deformable_conv_kernel_size
        self.input_layernorm = MxDNALayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.offset_conv = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.kh, kernel_size=self.kh, padding=self.kh // 2, bias=True)
        self.modulator_conv = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.kh, kernel_size=self.kh, padding=self.kh // 2, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.regular_conv = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=(self.kh, 1), padding=(self.kh // 2, 0), bias=False)
        torch.nn.init.trunc_normal_(self.regular_conv.weight, mean = 0.0, std = config.initializer_range)

    def forward(self, hidden_states, attention_mask, special_tokens_mask):
  
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


class MxDNAConversionLayer(nn.Module):
    def __init__(self, config: MxDNAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.num_motif_groups = config.num_motif_groups
        self.deformable_conv_kernel_size = config.deformable_conv_kernel_size
        self.expert_kernel_sizes = np.array(config.expert_kernel_sizes,dtype=np.int32)
        self.jitter_noise = config.router_jitter_noise
        self.motif_masked_token_embedding = nn.Parameter(torch.zeros(self.hidden_size))
        # initialize the weights normal
        torch.nn.init.trunc_normal_(self.motif_masked_token_embedding,mean=0.0,std=config.initializer_range)

        self.gate = nn.Linear(config.hidden_size,config.num_local_experts,bias=False)

        self.input_layernorm = MxDNALayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.conv_moe = MxDNAConvMoeBlock(config)
        self.deform_conv = MxDNADeforambleConvBlock(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        masked_tokens_mask: Optional[torch.Tensor] = None,
        output_router_logits: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, seq_len)` where padding elements are indicated by 0.\
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        final_hidden_states = torch.zeros_like(hidden_states)
        process_masked_tokens_mask =  masked_tokens_mask!=None
        if process_masked_tokens_mask:
            final_masked_tokens_mask = torch.zeros_like(masked_tokens_mask)
        else:
            final_masked_tokens_mask= None
        
        
        special_tokens_mask_cpu = special_tokens_mask.clone().detach().cpu().numpy().astype(np.int32)
        motif_mask_cpu = special_tokens_mask_cpu.copy()
        if attention_mask is None:
            attention_mask_cpu = np.ones_like(special_tokens_mask_cpu, dtype=np.int32)
        else:
            attention_mask_cpu = attention_mask.clone().detach().cpu().numpy().astype(np.int32)
             
        router_logits_list = []           
        motif_mask_center_list = []
        motif_mask_all_list= []

        for group_idx in range(self.num_motif_groups):
            # Scoring
            if self.training and self.jitter_noise > 0:
                # Multiply the token inputs by the uniform distribution - adding some noise
                hidden_states_group = hidden_states * torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            else:
                hidden_states_group = hidden_states
            router_logits = self.gate(hidden_states_group * (1 - special_tokens_mask[ :, :,None]))
            
            # Seleting Algorithm
            router_logits_cpu = router_logits.clone().detach().cpu().numpy().astype(np.float32)

            
            motif_mask_center_cpu,motif_mask_all_cpu = motif_masking_fn(router_logits_cpu, self.expert_kernel_sizes, attention_mask_cpu)
            motif_mask_cpu = motif_mask_cpu + (motif_mask_center_cpu!=-1)
            
            motif_mask_center_list.append(torch.tensor(motif_mask_center_cpu, device=hidden_states.device))
            motif_mask_all_list.append(torch.tensor(motif_mask_all_cpu, device=hidden_states.device))
            router_logits_list.append(router_logits)
            
            final_hidden_states_new,final_masked_tokens_mask_new=self.conv_moe(hidden_states,router_logits_list[group_idx],motif_mask_center_list[group_idx],motif_mask_all_list[group_idx],masked_tokens_mask)
            final_hidden_states += final_hidden_states_new
            if process_masked_tokens_mask:
                final_masked_tokens_mask += final_masked_tokens_mask_new
        if process_masked_tokens_mask:
            final_hidden_states[final_masked_tokens_mask!=0] = self.motif_masked_token_embedding
        hidden_states = residual + final_hidden_states
        attention_mask = torch.tensor((motif_mask_cpu*attention_mask_cpu)!=0, device=hidden_states.device,dtype=torch.long)
              
        # no cpu below
        motif_mask_center_cpu,motif_mask_all_cpu,special_tokens_mask_cpu,attention_mask_cpu,motif_mask_cpu,router_logits_cpu= None,None,None,None,None,None
        
        hidden_states = self.deform_conv(hidden_states, attention_mask, special_tokens_mask)
        
        outputs = (hidden_states,)

        if output_router_logits:
            outputs += (router_logits_list,motif_mask_center_list)
            

        return outputs, attention_mask
    
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
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, seq_len)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        # MHA
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states,attention_mask)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)


        return outputs
    
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
        output_attentions: Optional[bool] = False,
    ):
            
        residual = hidden_states_q
        hidden_states_q = self.input_layernorm(hidden_states_q)
        hidden_states_kv = self.input_layernorm(hidden_states_kv)
        hidden_states_q, self_attn_weights = self.cross_attn(
            hidden_states_q=hidden_states_q,
            hidden_states_kv=hidden_states_kv,
            attention_mask_query=attention_mask_q,
            attention_mask_kv=attention_mask_kv,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden_states_q = residual + hidden_states_q
        
        residual = hidden_states_q
        hidden_states_q = self.post_attention_layernorm(hidden_states_q)
        hidden_states_q = self.ffn(hidden_states_q,attention_mask_q)
        hidden_states_q = residual + hidden_states_q
        
        outputs = (hidden_states_q,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs
        

class MxDNAPreTrainedModel(PreTrainedModel):
    config_class = MxDNAConfig
    base_model_prefix = "model"
    _no_split_modules = ["MxDNAConversionLayer", "MxDNADecoderLayer", "MxDNAEncoderLayer"]
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
    Transformer encoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MxDNAConversionLayer`]

    Args:
        config: MxDNAConfig
    """

    def __init__(self, config: MxDNAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        nucleotide_layers = []
        motif_layers = []
        
        self.conversion_layer_idx = config.conversion_layer_idx
        for layer_idx in range(0, self.conversion_layer_idx):
            nucleotide_layers.append(MxDNAEncoderLayer(config, layer_idx))
        self.nucleotide_layers = nn.ModuleList(nucleotide_layers)
        
        self.conversion_layer = MxDNAConversionLayer(config, self.conversion_layer_idx)
        
        for layer_idx in range(self.conversion_layer_idx+1, config.num_hidden_layers):
            motif_layers.append(MxDNAEncoderLayer(config, layer_idx))
        self.motif_layers = nn.ModuleList(motif_layers)
        
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
        masked_tokens_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        output_nucleotide_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, MxDNAModelOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_nucleotide_hidden_states = (
            output_nucleotide_hidden_states if output_nucleotide_hidden_states is not None else self.config.output_nucleotide_hidden_states
        )

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


        
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
     
        hidden_states = inputs_embeds

        # encoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        router_logits_list = None
        motif_mask_center_list = None
        nucleotide_hidden_states = None

        nucleotide_attention_mask = attention_mask
        # nucleotide layers
        for nucleotide_layer in self.nucleotide_layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = nucleotide_layer(
                    hidden_states,
                    attention_mask=nucleotide_attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
            
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        if output_nucleotide_hidden_states:
            nucleotide_hidden_states = hidden_states
        
        # conversion layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_outputs,motif_attention_mask= self.conversion_layer(
                hidden_states,
                attention_mask=nucleotide_attention_mask,
                special_tokens_mask = special_tokens_mask,
                masked_tokens_mask=masked_tokens_mask,
                output_router_logits=output_router_logits,
            )
        hidden_states = layer_outputs[0]
        if output_router_logits:
            router_logits_list = layer_outputs[-2]
            motif_mask_center_list = layer_outputs[-1]
            

        # motif layers
        for motif_layer in self.motif_layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = motif_layer(
                    hidden_states,
                    attention_mask=motif_attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
            
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last encoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MxDNAModelOutput(
            motif_hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            nucleotide_hidden_states=nucleotide_hidden_states,
            motif_attention_mask=motif_attention_mask,
            nucleotide_attention_mask=nucleotide_attention_mask,
            all_attentions=all_self_attns,
            router_logits_list=router_logits_list,
            motif_mask_center_list=motif_mask_center_list,
        )
import math

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

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

class MxDNAForTokenExtraction(MxDNAPreTrainedModel):
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        load_balancing_loss: Optional[bool] = None,
        z_loss: Optional[bool] = None,
    ) -> Union[Tuple, LanaguageModelingOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
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
            masked_tokens_mask=None,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_nucleotide_hidden_states=True,
            output_router_logits=output_router_logits,
        )


        motif_hidden_states = outputs.motif_hidden_states
        motif_attention_mask = outputs.motif_attention_mask

        return motif_hidden_states[motif_attention_mask*(1-special_tokens_mask)!=0]
        
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        load_balancing_loss: Optional[bool] = None,
        z_loss: Optional[bool] = None,
    ) -> Union[Tuple, LanaguageModelingOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
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
            masked_tokens_mask=None,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_nucleotide_hidden_states=True,
            output_router_logits=output_router_logits,
        )


        motif_hidden_states = outputs.motif_hidden_states
        nucleotide_hidden_states = outputs.nucleotide_hidden_states
        motif_attention_mask = outputs.motif_attention_mask
        nucleotide_attention_mask = attention_mask

        decoder_layer_outputs = self.decoder(
            hidden_states_q=nucleotide_hidden_states,
            hidden_states_kv=motif_hidden_states,
            attention_mask_q=nucleotide_attention_mask,
            attention_mask_kv=motif_attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        

        
        last_hidden_states = decoder_layer_outputs[0]
        last_hidden_states = self.norm(last_hidden_states)
        
        logits = self.lm_head(last_hidden_states)
        logits = logits.float()


        loss = None
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
            loss = loss_fct(mlm_logits, mlm_labels)
        

        aux_loss = None
        router_z_loss = None
        if output_router_logits and load_balancing_loss:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits_list,
                outputs.motif_mask_center_list,)

            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
        if output_router_logits and z_loss:  
            router_z_loss = router_z_loss_func(
                outputs.router_logits_list,
                outputs.motif_mask_center_list,)
            loss += self.router_aux_loss_coef * router_z_loss.to(loss.device)
        
        return LanaguageModelingOutput(
            loss=loss,
            aux_loss=aux_loss,
            z_loss=router_z_loss,
            logits=logits,
            router_logits = outputs.router_logits_list,
            all_hidden_states=outputs.all_hidden_states+decoder_layer_outputs[0] if output_hidden_states else None,
            all_attentions=outputs.all_attentions+decoder_layer_outputs[1] if output_attentions else None,
        )
        
class MxDNAForSequenceClassification(MxDNAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MxDNAModel(config)
        self.decoder = MxDNADecoderLayer(config,config.num_hidden_layers)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        load_balancing_loss: Optional[bool] = None,
        z_loss: Optional[bool] = None,
        **kwargs,
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
            
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_nucleotide_hidden_states=True,
        )
        motif_hidden_states = transformer_outputs.motif_hidden_states
        nucleotide_hidden_states = transformer_outputs.nucleotide_hidden_states
        router_logits_list = transformer_outputs.router_logits_list
        motif_mask_center_list = transformer_outputs.motif_mask_center_list
        
        motif_attention_mask = transformer_outputs.motif_attention_mask
        nucleotide_attention_mask = attention_mask
        
        decoder_attention_mask_q = torch.zeros_like(motif_attention_mask)
        decoder_attention_mask_q[:,0]=1
        decoder_attention_mask_kv = torch.cat((motif_attention_mask,nucleotide_attention_mask),dim=1)
        decoder_hidden_states_kv = torch.cat((motif_hidden_states,nucleotide_hidden_states),dim=1)
        decoder_position_ids = torch.cat((position_ids,position_ids),dim = 1)
        decoder_layer_outputs = self.decoder(
            hidden_states_q=motif_hidden_states,
            hidden_states_kv=decoder_hidden_states_kv,
            attention_mask_q=decoder_attention_mask_q,
            attention_mask_kv=decoder_attention_mask_kv,
            position_ids=decoder_position_ids,
            output_attentions=output_attentions,
        )
        
        last_hidden_states = decoder_layer_outputs[0]
        last_hidden_states = self.norm(last_hidden_states)
        pooled_logits = self.task_head(last_hidden_states[:,0])
        loss = None

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
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        aux_loss=None
        router_z_loss=None
        if output_router_logits and load_balancing_loss:
            aux_loss = load_balancing_loss_func(
                router_logits_list,
                motif_mask_center_list,)
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
          
        if output_router_logits and z_loss:  
            router_z_loss = router_z_loss_func(
                router_logits_list,
                motif_mask_center_list,)
            loss += self.router_aux_loss_coef * router_z_loss.to(loss.device)
            
                
        return SequenceClassifierOutput(
            loss=loss,
            aux_loss=aux_loss,
            z_loss=router_z_loss,
            logits=pooled_logits,
            router_logits=router_logits_list,
            all_hidden_states=transformer_outputs.all_hidden_states+decoder_layer_outputs[0] if output_hidden_states else None,
            all_attentions=transformer_outputs.all_attentions+decoder_layer_outputs[1] if output_attentions else None,
        )
        
class MxDNAForSequenceClassificationND(MxDNAPreTrainedModel):
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
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
            
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            output_nucleotide_hidden_states=False,
        )
        motif_hidden_states = transformer_outputs.motif_hidden_states
        nucleotide_hidden_states = transformer_outputs.nucleotide_hidden_states
        router_logits_list = transformer_outputs.router_logits_list
        motif_mask_center_list = transformer_outputs.motif_mask_center_list
        
        last_hidden_states = motif_hidden_states
        last_hidden_states = self.norm(last_hidden_states)
        pooled_logits = self.task_head(last_hidden_states[:,0])
        loss = None

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
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        aux_loss=None
        
        if output_router_logits and load_balancing_loss:
            aux_loss = load_balancing_loss_func(
                router_logits_list,
                motif_mask_center_list,)
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
                
        return SequenceClassifierOutput(
            loss=loss,
            aux_loss=aux_loss,
            logits=pooled_logits,
            router_logits=router_logits_list,
            all_hidden_states=transformer_outputs.all_hidden_states,
            all_attentions=transformer_outputs.all_attentions,
        )
        