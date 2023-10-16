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
    value_nmse_list = []
    
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
        value_nmse_list.append(value_mse)

    nmse = mean_squared_error(ratio_list, target_score)
    return nmse, value_nmse_list
    
        
    
    
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

    nmse_list = []
    name_list = []

    pred_scores = []
    target_scores = []
    
    value_nmse = [[], [], [], [], [], [], [], [], [], []]
    
    
    for country_name in country_list:
        print(country_name)
        nmse, v_nmse_list = calc_valuenet_mse(country_name, model_name, mode, strategy, threshold)
        nmse_list.append(nmse)
        name_list.append(country_name)
        for i, v_nmse in enumerate(v_nmse_list):
            value_nmse[i].append(v_nmse)
    
    name_list.append('TOTAL')
    nmse_list.append(sum(nmse_list)/len(nmse_list))
    for i in range(10):
        value_nmse[i].append(sum(value_nmse[i])/len(value_nmse[i]))
    
    data = {
        'Country':name_list,
        'NMSE':nmse_list
    }
    
    value_nmse_data = {
        'Country':name_list,
        'Achievement':value_nmse[0],
        'Benevolence':value_nmse[1],
        'Conformity':value_nmse[2],
        'Hedonism':value_nmse[3],
        'Power':value_nmse[4],
        'Security':value_nmse[5],
        'Self-direction':value_nmse[6],
        'Stimulation':value_nmse[7],
        'Tradition':value_nmse[8],
        'Universalism':value_nmse[9]
    }
    
    
    df = pd.DataFrame(data=data)
    df.to_csv(f'./results/valuenet/{mode}_nmse.csv', sep='\t')
    
    value_nmse_df = pd.DataFrame(data=value_nmse_data)
    value_nmse_df.to_csv(f'./results/valuenet/{mode}_value_nmse.csv', sep='\t')
        
    
    
