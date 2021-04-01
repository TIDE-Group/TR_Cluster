from transformers import BigBirdTokenizer, BigBirdForTokenClassification, AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
import numpy as np
import torch
import pandas as pd

df = pd.read_excel('./doc.xlsx')
print(df.columns)
texts = df[df['content'].notna()]['content'].tolist()
label_list = [
     "O",       # Outside of a named entity
     "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
     "I-MISC",  # Miscellaneous entity
     "B-PER",   # Beginning of a person's name right after another person's name
     "I-PER",   # Person's name
     "B-ORG",   # Beginning of an organisation right after another organisation
     "I-ORG",   # Organisation
     "B-LOC",   # Beginning of a location right after another location
     "I-LOC"    # Location
]

# model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english").cuda()
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = BigBirdForTokenClassification.from_pretrained('google/bigbird-roberta-base').cuda()
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

# ner_model = pipeline('ner', model=model, tokenizer=tokenizer, framework='pt')
with torch.no_grad():   
    # results = ner_model(texts[:3])
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(texts[0])))
    inputs = tokenizer.encode(texts[0], return_tensors="pt", padding=True).cuda()
    # for key in inputs.data.keys():
    #     inputs.data[key] = inputs.data[key].cuda()
    
    outputs = model(inputs).logits

    predictions = torch.argmax(outputs, dim=2)

    results = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].cpu().numpy()) if prediction != 0]

print(results)
pass
