import os
import os.path as p
from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn.functional as F

class DS_argument(Dataset):
    def __init__(self, tokenizer, pos_df, neg_df):       
        inputs = []
        targets = []
        
        for p_n, df in zip(('pos', 'neg'), (pos_df, neg_df)):
            conclusion = df['Conclusion'].tolist()
            stance = df['Stance'].tolist()
            premise = df['Premise'].tolist()
            
            if p_n == 'pos':
                for c, s, pr in zip(conclusion, stance, premise):
                    if s == 'in favor of':
                        q = f'Tell me what you would say about the following statement. Statement: {c} Answer:'
                        resp = f'I would say, "I agree with that because {pr}." about {c}'
                    if s == 'against':
                        q = f'Tell me what you would say about the following statement. Statement: {c} Answer:'
                        resp = f'I would say, "I disagree with that because {pr}." about {c}'
                    inputs.append(q)
                    targets.append(resp)
                    
            if p_n == 'neg':
                for c, s, pr in zip(conclusion, stance, premise):
                    if s == 'in favor of':
                        q = f'Tell me what you would not say about the following statement. Statement: {c} Answer:'
                        resp = f'I would not say, "I agree with that because {pr}." about {c}'
                    if s == 'against':
                        q = f'Tell me what you would not say about the following statement. Statement: {c} Answer:'
                        resp = f'I would not say, "I disagree with that because {pr}." about {c}'
                    inputs.append(q)
                    targets.append(resp)
        
        batch_size = len(inputs)
        max_length = 128
        
        self.model_inputs = tokenizer(inputs)
        self.labels = tokenizer(targets)

        for i in range(batch_size):
            sample_input_ids = self.model_inputs["input_ids"][i]
            label_input_ids = self.labels["input_ids"][i] + [tokenizer.pad_token_id]
            self.model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            self.labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            self.model_inputs["attention_mask"][i] = [1] * len(self.model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = self.model_inputs["input_ids"][i]
            label_input_ids = self.labels["input_ids"][i]
            self.model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            self.model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + self.model_inputs[
                "attention_mask"
            ][i]
            self.labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            self.model_inputs["input_ids"][i] = self.model_inputs["input_ids"][i][:max_length]
            self.model_inputs["attention_mask"][i] = self.model_inputs["attention_mask"][i][:max_length]
            self.labels["input_ids"][i] = self.labels["input_ids"][i][:max_length]
        self.model_inputs["labels"] = self.labels["input_ids"]
    
    def __len__(self):
        return len(self.model_inputs["labels"])
    
    def __getitem__(self, idx):
        return (self.model_inputs['input_ids'][idx], self.model_inputs['attention_mask'][idx], self.model_inputs['labels'][idx])
    