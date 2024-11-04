
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .BasicUnitNMS import basic_unit_nms_fn
from torchvision.ops import deform_conv2d
from flash_attn.bert_padding import pad_input, unpad_input


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

    def reorder_back(sorted_x):
        original_x = torch.zeros([batch_size,seq_len,*sorted_x.shape[2:]],dtype=sorted_x.dtype,device = device)
        original_x[attention_mask] = sorted_x[output_attention_mask]
        return original_x
    
    return new_sequences, reorder_back


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

        seqs,reorder_back_fn = reorder_fn(sequences=[hidden_states, attention_mask,special_tokens_mask], attention_mask=attention_mask)
        
        hidden_states, attention_mask,special_tokens_mask = seqs
        
        hidden_states = self.deform_conv(hidden_states, attention_mask, special_tokens_mask)
        
        return hidden_states, attention_mask, special_tokens_mask, reorder_back_fn
