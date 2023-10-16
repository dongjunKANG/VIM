import pandas as pd
import os
import os.path as p
import math
import torch
import torch.nn as nn
from scipy import stats 
from sklearn.metrics import mean_squared_error    
import sys   


def calc_valuenet_mse(country_name, model_name, mode, strategy, threshold):
    value_list = ['Achievement', 'Benevolence', 'Conformity', 'Hedonism', 'Power', 'Security', 'Self-direction', 'Stimulation', 'Tradition', 'Universalism']
    if mode == 'base':
        gen_df = pd.read_csv(f'./results/valuenet/base/ess_base.csv', sep='\t')
    if mode == 'argument':
        gen_df = pd.read_csv(f'./results/valuenet/argument/{model_name}/TH_{threshold}/{country_name}/result_{country_name}.csv', sep='\t')
    if mode == 'argument_survey':
        gen_df = pd.read_csv(f'./results/valuenet/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country_name}/result_{country_name}.csv', sep='\t')
    if mode == 'survey':
        gen_df = pd.read_csv(f'./results/valuenet/survey/{model_name}/{strategy}/{country_name}/result_{country_name}.csv', sep='\t')

    country_df = pd.read_csv(f'./data/country_and_group.csv', sep='\t')
    country_list = country_df['Country'].tolist()
    country_idx = country_list.index(country_name)
    country_row = country_df.loc[country_idx]
    print(list(country_row))
    target_score = list(country_row)[-10:]
    
    
    ratio_list = []
    value_mse_list = []
    
    for target_scr, value in zip(target_score, value_list):
        val_df = gen_df.groupby('Value').get_group(f'{value}')
        pos_df = val_df.groupby('Stance').get_group('positive')
        neg_df = val_df.groupby('Stance').get_group('negative')
        
        pos_answer = pos_df['Answer'].tolist()
        neg_answer = neg_df['Answer'].tolist()     
        
        pos_count = 0
        neg_count = 0
        
        for pos_a in pos_answer:
            if 'I agree' in str(pos_a) or 'i agree' in str(pos_a):
                pos_count += 1
            if 'I disagree' in str(pos_a) or 'i disagree' in str(pos_a):
                neg_count += 1
        
        for neg_a in neg_answer:
            if 'I agree' in str(neg_a) or 'i agree' in str(neg_a):
                neg_count += 1
            if 'I disagree' in str(neg_a) or 'i disagree' in str(neg_a):
                pos_count += 1
        
        ratio = pos_count/(pos_count+neg_count)
        ratio_list.append(ratio)
    
        rescaled_scr = (target_scr-1)/5
        value_mse = mean_squared_error([ratio], [rescaled_scr])
        value_mse_list.append(value_mse)

    mse = mean_squared_error(ratio_list, target_score)
    return mse, value_mse_list
    
        
    
    
if __name__ == '__main__':
    mode = sys.argv[1]          # argument, survey, argument_survey
    strategy = 'min'
    threshold = 3
        
    model_name = 'llama'       
    
    if mode == 'survey':
        mode_path = f'./results/valuenet/survey/{model_name}/{strategy}'
    if mode == 'argument':
        mode_path = f'./results/valuenet/argument/{model_name}/TH_{threshold}'
    if mode == 'argument_survey':
        mode_path = f'./results/valuenet/argument_survey/{model_name}/{strategy}_TH_{threshold}'    
    country_list = os.listdir(mode_path)
    country_list.sort()

    mse_list = []
    name_list = []

    pred_scores = []
    target_scores = []
    
    value_mse = [[], [], [], [], [], [], [], [], [], []]
    
    
    for country_name in country_list:
        print(country_name)
        mse, v_mse_list = calc_valuenet_mse(country_name, model_name, mode, strategy, threshold)
        mse_list.append(mse)
        name_list.append(country_name)
        for i, v_mse in enumerate(v_mse_list):
            value_mse[i].append(v_mse)
    
    name_list.append('TOTAL')
    mse_list.append(sum(mse_list)/len(mse_list))
    for i in range(10):
        value_mse[i].append(sum(value_mse[i])/len(value_mse[i]))
    
    data = {
        'Country':name_list,
        'MSE':mse_list
    }
    
    value_mse_data = {
        'Country':name_list,
        'Achievement':value_mse[0],
        'Benevolence':value_mse[1],
        'Conformity':value_mse[2],
        'Hedonism':value_mse[3],
        'Power':value_mse[4],
        'Security':value_mse[5],
        'Self-direction':value_mse[6],
        'Stimulation':value_mse[7],
        'Tradition':value_mse[8],
        'Universalism':value_mse[9]
    }
    
    
    df = pd.DataFrame(data=data)
    df.to_csv(f'./results/valuenet/{mode}_nmse.csv', sep='\t')
    
    value_mse_df = pd.DataFrame(data=value_mse_data)
    value_mse_df.to_csv(f'./results/valuenet/{mode}_value_nmse.csv', sep='\t')
        
    
    