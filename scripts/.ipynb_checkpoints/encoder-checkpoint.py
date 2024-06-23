# Transformers imports
from transformers import (
    AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, 
    DataCollatorWithPadding, TrainingArguments, Trainer, TextClassificationPipeline, 
    AdamW, get_scheduler, pipeline, RobertaTokenizerFast
)

import torch 
import numpy as np

def encode_texts(model_name, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            if torch.cuda.is_available():
                encoded_input = {key: val.to('cuda') for key, val in encoded_input.items()}
            output = model(**encoded_input)
            cls_embedding = output.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.cpu().numpy())
    print("embeddings are generated")
    return np.vstack(embeddings)

def encode_texts_biolm(model_name, texts):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", max_len=512)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            if torch.cuda.is_available():
                encoded_input = {key: val.to('cuda') for key, val in encoded_input.items()}
            output = model(**encoded_input)
            cls_embedding = output.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.cpu().numpy())
    print("embeddings are generated")
    return np.vstack(embeddings)


