from model_source.modeling_mxdna import MxDNAForMaskedLM 
from model_source.configuration_mxdna import MxDNAConfig
from transformers import AutoTokenizer
import torch

# Load configuration and tokenizer
config = MxDNAConfig.from_pretrained('./model_source/config/config.json')
tokenizer = AutoTokenizer.from_pretrained('./model_source/1mertokenizer')

# Initialize the model
model = MxDNAForMaskedLM._from_config(config=config).cuda()

# Define a sample DNA sequence
masked_sequence = 'A T [MASK] G A G [MASK] T A G T A G C T A G T C G T A G T C G T G T A'
sequence   = 'A T C G A G C T A G T A G C T A G T C G T A G T C G T G T A'
# Tokenize the sequence
inputs = tokenizer.encode_plus(
    masked_sequence,
    add_special_tokens=True,
    return_attention_mask=True, 
    return_special_tokens_mask=True,
    return_tensors='pt'
)
labels = tokenizer.encode_plus(
    sequence,
    add_special_tokens=True,
    return_attention_mask=False, 
    return_special_tokens_mask=False,
    return_tensors='pt'
)

# Move inputs to GPU
input_ids = inputs["input_ids"].cuda()
attention_mask = inputs["attention_mask"].cuda()
special_tokens_mask = inputs["special_tokens_mask"].cuda()

labels = labels['input_ids'].cuda()

# Perform a forward pass
with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        special_tokens_mask=special_tokens_mask,
        labels=labels, 
        load_balancing_loss=True
    )