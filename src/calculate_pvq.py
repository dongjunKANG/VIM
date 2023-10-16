import pandas as pd
import os
import os.path as p
import math
import torch
import torch.nn as nn
from scipy import stats 
from sklearn.metrics import mean_squared_error    
import sys   

def calc_survey_scores(country_name, model_name, mode, strategy, threshold):
    value_list = ['Achievement', 'Benevolence', 'Conformity', 'Hedonism', 'Power', 'Security', 'Self-direction', 'Stimulation', 'Tradition', 'Universalism']
    value_list_2 = ['Achievement', 'Benevolence', 'Conformity', 'Hedonism', 'Power', 'Security', 'Self-direction', 'Stimulation', 'Tradition', 'Universalism']
    pvq40_coding_key = {
        'Conformity':[7, 16, 28, 36],
        'Tradition':[9, 20, 25, 38],
        'Benevolence':[12, 18, 27, 33],
        'Universalism':[3, 8, 19, 23, 29, 40],
        'Self-direction':[1, 11, 22, 34],
        'Stimulation':[6, 15, 30],
        'Hedonism':[10, 26, 37],
        'Achievement':[4, 13, 24, 32],
        'Power':[2, 17, 39],
        'Security':[5, 14, 21, 31, 35]
    }
    if mode == 'argument':
        gen_df = pd.read_csv(f'./results/pvq_results/argument/{model_name}/TH_{threshold}/{country_name}/result_{country_name}.csv', sep='\t')
    elif mode == 'argument_survey':
        gen_df = pd.read_csv(f'./results/pvq_results/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country_name}/result_{country_name}.csv', sep='\t')
    else:
        gen_df = pd.read_csv(f'./results/pvq_results/survey/{model_name}/{strategy}/{country_name}/result_{country_name}.csv', sep='\t')
  
    answers = gen_df['Answer'].tolist()
    answer_scores = []
    for a in answers:
        if "1" in str(a) or 'Not like me at all' in str(a):
            answer_scores.append(1)        
        elif "2" in str(a) or 'Not like me' in str(a):
            answer_scores.append(2)
        elif "3" in str(a) or 'A little like me' in str(a):
            answer_scores.append(3)
        elif "4" in str(a) or 'Somewhat like me' in str(a):
            answer_scores.append(4)
        elif "5" in str(a) or 'Like me' in str(a):
            answer_scores.append(5)
        elif "6" in str(a) or 'Very much like me' in str(a):
            answer_scores.append(6)
        else:
            answer_scores.append(-1000)
                
    scores = []
    for v in value_list_2:
        v_question = pvq40_coding_key[v]
        v_score = 0
        for q in v_question:
            v_score += answer_scores[q-1]
        v_score = v_score/len(v_question)
        scores.append(v_score)

    data = {
        'value':value_list,
        'score':scores
    }
    score_df = pd.DataFrame(data=data)
    
    if mode == 'argument':
        os.makedirs(f'./results/pvq_scores/argument/{model_name}/TH_{threshold}/{country_name}', exist_ok=True)
        score_df.to_csv(f'./results/pvq_scores/argument/{model_name}/TH_{threshold}/{country_name}/result_{country_name}.csv', sep='\t')
    elif mode == 'argument_survey':
        os.makedirs(f'./results/pvq_scores/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country_name}', exist_ok=True)
        score_df.to_csv(f'./results/pvq_scores/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country_name}/result_{country_name}.csv', sep='\t')
    else:
        os.makedirs(f'./results/pvq_scores/survey/{model_name}/{strategy}/{country_name}', exist_ok=True)
        score_df.to_csv(f'./results/pvq_scores/survey/{model_name}/{strategy}/{country_name}/result_{country_name}.csv', sep='\t')
       
    return scores

  
if __name__ == '__main__':
    mode = sys.argv[1]          # argument, survey, argument_survey
    strategy = 'min'
    threshold = 3
        
    model_name = 'llama'       
    
    if mode == 'survey':
        mode_path = f'./results/pvq_results/survey/{model_name}/{strategy}'
    if mode == 'argument':
        mode_path = f'./results/pvq_results/argument/{model_name}/TH_{threshold}'
    if mode == 'argument_survey':
        mode_path = f'./results/pvq_results/argument_survey/{model_name}/{strategy}_TH_{threshold}'    
    country_list = os.listdir(mode_path)
    country_list.sort()

    
    nmse_list = []
    name_list = []

    pred_scores = []
    target_scores = []
    
    for country_name in country_list:
        pred_score = calc_survey_scores(country_name=country_name, 
                                        model_name=model_name, 
                                        mode=mode, 
                                        strategy=strategy,
                                        threshold=threshold
                                        )

        country_df = pd.read_csv('./data/country_and_group.csv', sep='\t')
        country_list = country_df['Country'].tolist()
        country_idx = country_list.index(country_name)
        
        country_row = country_df.loc[country_idx]
        target_score = list(country_row)[-10:]
        
        norm_pred_score = [(scr-1)/5 for scr in pred_score]
        norm_target_score = [(scr-1)/5 for scr in target_score]
        nmse = mean_squared_error(norm_pred_score, norm_target_score)
        nmse_list.append(nmse)
        
        

        print(f"{country_name} MSE: {nmse}")
        name_list.append(country_name)
        


        pred_scores.append(pred_score)
        target_scores.append(target_score)
    
    avg_nmse = sum(nmse_list)/len(nmse_list)
    nmse_list.append(avg_nmse)
    
    
    names = name_list.copy()
    names.append('TOTAL AVERAGE')
    
    nmse_data = {
        'Country':names,
        'NMSE':nmse_list
    }
    
    score_data = {
        'Country':name_list,
        'Pred_score':pred_scores,
        'Target_score':target_scores
    }
    
    if mode == 'survey':
        result_path = f'./results/pvq_scores/survey/llama/{strategy}'
    if mode == 'argument':
        result_path = f'./results/pvq_scores/argument/llama/TH_{threshold}'
    if mode == 'argument_survey':
        result_path = f'./results/pvq_scores/argument_survey/llama/{strategy}_TH_{threshold}' 
   
    os.makedirs(result_path, exist_ok=True)
    
    nmse_df = pd.DataFrame(data=nmse_data)
    nmse_df.to_csv(p.join(result_path, 'NMSE_TOTAL.csv'), sep='\t')
    
    score_df = pd.DataFrame(data=score_data)
    score_df.to_csv(p.join(result_path, 'SCORE_TOTAL.csv'), sep='\t')
