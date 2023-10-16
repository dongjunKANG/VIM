import pandas as pd
import os
import os.path as p
import numpy
import random  

def gen_argument_train_data(country_name, threshold):
    country_df = pd.read_csv('./data/country_and_group.csv', sep='\t')
    country_list = country_df['Country'].tolist()
    country_idx = country_list.index(country_name)
    country_row = country_df.loc[country_idx]
    target_score = list(country_row)[-10:]

    sample_df = pd.read_csv('./data/valueEval_10.csv', sep='\t')
    
    columns = sample_df.columns.tolist()
    value_idx = columns.index('Achievement')
    positive_arg_idx = []
    negative_arg_idx = []
    index = list(sample_df.index)
    scores = []
    for i, idx in enumerate(index):
        row = list(sample_df.iloc[i])[value_idx:value_idx+10]
        score = []     
        for r, s in zip(row, target_score):
            if r == 1:
                score.append(s)
        min_score = min(score)
        scores.append(min_score)
        if min_score >= threshold:
            positive_arg_idx.append(idx)
        if min_score < threshold:
            negative_arg_idx.append(idx)

    sample_df['Score'] = scores
    pos_df = sample_df.iloc[positive_arg_idx]
    neg_df = sample_df.iloc[negative_arg_idx]

    pos_train_df = pos_df.sample(frac=0.8, random_state=42)
    pos_temp_df = pos_df.drop(pos_train_df.index)
    neg_train_df = neg_df.sample(frac=0.8, random_state=42)
    neg_temp_df = neg_df.drop(neg_train_df.index)
    
    pos_valid_df = pos_temp_df.sample(frac=0.5, random_state=42)
    pos_test_df = pos_temp_df.drop(pos_valid_df.index)
    neg_valid_df = neg_temp_df.sample(frac=0.5, random_state=42)
    neg_test_df = neg_temp_df.drop(neg_valid_df.index)
    
    
    drop_columns = [j for j in pos_train_df.columns.tolist() if 'named' in j]
    pos_train_df = pos_train_df.drop(columns=drop_columns)
    pos_valid_df = pos_valid_df.drop(columns=drop_columns)
    pos_test_df = pos_test_df.drop(columns=drop_columns)
    neg_train_df = neg_train_df.drop(columns=drop_columns)
    neg_valid_df = neg_valid_df.drop(columns=drop_columns)
    neg_test_df = neg_test_df.drop(columns=drop_columns)
    
    os.makedirs(f'./data/country/argument/train/TH_{threshold}/pos', exist_ok=True)
    os.makedirs(f'./data/country/argument/train/TH_{threshold}/neg', exist_ok=True)
    os.makedirs(f'./data/country/argument/valid/TH_{threshold}/pos', exist_ok=True)
    os.makedirs(f'./data/country/argument/valid/TH_{threshold}/neg', exist_ok=True)
    os.makedirs(f'./data/country/argument/test/TH_{threshold}/pos', exist_ok=True)
    os.makedirs(f'./data/country/argument/test/TH_{threshold}/neg', exist_ok=True)
    pos_train_df.to_csv(f'./data/country/argument/train/TH_{threshold}/pos/{country_name}.csv', sep='\t')
    pos_valid_df.to_csv(f'./data/country/argument/valid/TH_{threshold}/pos/{country_name}.csv', sep='\t')
    pos_test_df.to_csv(f'./data/country/argument/test/TH_{threshold}/neg/{country_name}.csv', sep='\t')
    neg_train_df.to_csv(f'./data/country/argument/train/TH_{threshold}/neg/{country_name}.csv', sep='\t')
    neg_valid_df.to_csv(f'./data/country/argument/valid/TH_{threshold}/neg/{country_name}.csv', sep='\t')
    neg_test_df.to_csv(f'./data/country/argument/test/TH_{threshold}/neg/{country_name}.csv', sep='\t')
    
    
if __name__ == '__main__':  
    country_df = pd.read_csv('./data/country_and_group.csv', sep='\t')
    country_list = country_df['Country'].tolist()
    threshold = 3
    
    for country_name in country_list:
        gen_argument_train_data(country_name, threshold)