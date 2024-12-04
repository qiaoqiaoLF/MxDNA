from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class MxDNAConfig(PretrainedConfig):

    model_type = "mxdna"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id =0,
        unk_token_id = 1,
        cls_token_id = 2,
        sep_token_id = 3,
        mask_token_id = 4,
        rope_theta=1e6,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        router_aux_loss_coef=0.001,
        expert_kernel_sizes=[],
        conversion_layer_idx=-1,
        num_motif_groups=1,
        router_jitter_noise=None,
        deformable_conv_kernel_size=-1,
        **kwargs,
    ):

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.router_aux_loss_coef = router_aux_loss_coef
        
        self.expert_kernel_sizes = expert_kernel_sizes
        self.conversion_layer_idx = conversion_layer_idx
        self.deformable_conv_kernel_size = deformable_conv_kernel_size
        
        self.router_jitter_noise = router_jitter_noise
        self.num_motif_groups = num_motif_groups
        super().__init__(
            pad_token_id=pad_token_id,
            unk_token_id =unk_token_id,
            cls_token_id = cls_token_id,
            sep_token_id=sep_token_id,
            mask_token_id=mask_token_id,
            **kwargs,
        )
