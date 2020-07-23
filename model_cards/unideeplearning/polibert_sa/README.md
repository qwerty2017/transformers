---
language: it
thumbnail: https://www.unideeplearning.com/wp-content/uploads/2019/10/logo_unideep-2.png
tags:
- sentiment
- Italian
license: MIT
widget:
- text: 'Giuseppe Rossi è un ottimo politico'
---

# 🤗 + polibert_SA - POLItic BERT based Sentiment Analysis
  
## Model description  
  
This model performs sentiment analysis on Italian political twitter sentences. It was trained starting from an instance of "bert-base-italian-uncased-xxl" and fine-tuned on an Italian dataset of tweets.
  
## Intended uses & limitations  
  
#### How to use  
  
```python
import torch
from torch import nn 

text = "Giueseppe Rossi è un pessimo politico"
input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors= 'pt')

logits, = model(input_ids)
logits = logits.squeeze(0)

prob = nn.functional.softmax(logits, dim=0)

# 0 Negative, 1 Neutral, 2 Positive 
print(prob.argmax().tolist())
```  
  
#### Hyperparameters

- Optimizer: **AdamW** with learning rate of **2e-5**, epsilon of **1e-8**
- Max epochs: **2**
- Batch size: **16**

## Acknowledgments

Thanks to the support from: 
the [Hugging Face](https://huggingface.co/), Unione Professionisti (https://www.unioneprofessionisti.com/)
