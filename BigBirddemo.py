import warnings
warnings.filterwarnings('ignore')
from transformers import BigBirdTokenizer, BigBirdModel, BigBirdForMaskedLM, AutoTokenizer
import os
import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda


doc_names = os.listdir('./doc/')
batch_size = 2
representation = np.zeros((len(doc_names), 768))
ROUND = len(doc_names) // batch_size + 1
texts = []
for doc_name in doc_names:
    with open('./doc/' + doc_name, 'r', encoding='utf-8') as f:
        texts.append(f.read())

# texts = texts

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")

model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
model = model.eval().cuda()

for i in range(ROUND):
    down = i * batch_size
    up = (i + 1) * batch_size if (i + 1) * batch_size <= len(doc_names) else len(doc_names)

    tokens_pt = tokenizer(texts[down : up], return_tensors='pt', padding=True)
    for key in tokens_pt.data.keys():
        tokens_pt.data[key] = tokens_pt.data[key].cuda()

    with torch.no_grad():
        output = model.forward(**tokens_pt)
        representation[down : up, :] = output.pooler_output.cpu().numpy()
        del output, tokens_pt
        cuda.empty_cache()

print(representation.shape)
print(representation[:10])


