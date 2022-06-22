# SkipBERT

Code associated with the paper **[SkipBERT: Efficient Inference with Shallow Layer Skipping](https://aclanthology.org/2022.acl-long.503/)**, at ACL 2022

Thank you for your interests! The code is still under construction so should be updated frequently.

## Download Pre-trained Checkpoints

- [SkipBERT6+6](https://skipbert.s3.amazonaws.com/skipbert-L6-6.zip)
- [SkipBERT6+4](https://skipbert.s3.amazonaws.com/skipbert-L6-4.zip)
- [SkipBERT6+2](https://skipbert.s3.amazonaws.com/skipbert-L6-2.zip)

## Quick Start

```python
import psutil, os
import torch
from skipbert import SkipBertModel
from transformers import BertTokenizerFast, BertConfig

p = psutil.Process(os.getpid())
p.nice(100)  # set process priority
print('nice:', p.nice())
torch.set_num_threads(1) # set num of torch threads

# Input Related
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

inputs = tokenizer(
    ["Good temper decides everything"],
    return_tensors='pt', padding='max_length', max_length=128
)

inputs = {
   k: (v.to(device) if isinstance(v, torch.Tensor) and k != 'input_ids' else v) for k, v in inputs.items()
}

# Model Related
config = BertConfig.from_pretrained(PATH_TO_MODEL)
config.plot_mode = 'plot_passive'

model = SkipBertModel.from_pretrained(PATH_TO_MODEL, config=config)
model.eval()

# Inference
# first time will compute the shallow layers
ret = model(**inputs)

# second time will retrieve hidden states from PLOT
ret = model(**inputs)
```
