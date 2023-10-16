import pandas as pd
import os
import os.path as p
import math
import torch
import torch.nn as nn
from scipy import stats 
from sklearn.metrics import mean_squared_error    
import sys

def calc_ess_answer(country_name, model_name, mode, strategy, threshold):
    chapters = ['media_and_social_trust', 'personal_and_social_well-being', 'politics', 'subjective_well-being', 'understanding_democracy']
    ch_nmse_list = []
    
    for ch in chapters:
        # print(ch)
        if mode == 'base':
            gen_df = pd.read_csv(f'./results/ess/{ch}/base/ess_base.csv', sep='\t')
        if mode == 'argument':
            gen_df = pd.read_csv(f'./results/ess/{ch}/argument/{model_name}/TH_{threshold}/{country_name}/ess_{country_name}.csv', sep='\t')
        if mode == 'argument_survey':
            gen_df = pd.read_csv(f'./results/ess/{ch}/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country_name}/ess_{country_name}.csv', sep='\t')
        if mode == 'survey':
            gen_df = pd.read_csv(f'./results/ess/{ch}/survey/{model_name}/{strategy}/{country_name}/ess_{country_name}.csv', sep='\t')
        
        gold_df = pd.read_csv(f'./data/ESS/country_and_group/{ch}/mean/{country_name}.csv', sep='\t')
        gold_score = gold_df['Mean'].tolist()
        code_list = gold_df['Code'].tolist()
        
        answer = gen_df['Answer'].tolist()
        pred_score = []
        scaled_gold = []
        scaled_pred = []
        
        if ch == 'media_and_social_trust':
            for a, gold, code in zip(answer, gold_score, code_list):
                gen = a.split("my answer is")[1:]
                gen = " ".join(gen)
                
                print(gen)
                if code == 'tvtot' or code == 'tvpol':
                    gen = gen[:45]
                    count = []
                    if 'No time at all' in gen:
                        count.append(0)
                    if 'Less than 0.5 hour' in gen:
                        count.append(1)
                    if '0.5 to 1 hour' in gen:
                        count.append(2)
                    if 'More than 1 hour, up to 1.5 hours' in gen:
                        count.append(3)
                    if 'More than 1.5 hours, up to 2 hours' in gen:
                        count.append(4)
                    if 'More than 2 hours, up to 2.5 hours' in gen:
                        count.append(5)
                    if 'More than 2.5 hours, up to 3 hours' in gen:
                        count.append(6)
                    if 'More than 3 hours' in gen:
                        count.append(7)
                    if len(count) == 1:
                        pred_score.append(count[0])
                        scaled_gold.append(gold/7)
                        scaled_pred.append(count[0]/7)
                    else:
                        pred_score.append(99)
                    
                        
                else:
                    gen = gen.split(".")[0]
                    count = []
                    if "0" in str(gen)[:5]:
                        if "10" in str(gen)[:5]:
                            count.append(10)
                        else:
                            count.append(0)
                    if "1" in str(gen)[:5]:
                        if "10" not in str(gen)[:5]:
                            count.append(1)   
                    elif "2" in str(gen)[:5]:
                        count.append(2)
                    elif "3" in str(gen)[:5]:
                        count.append(3)
                    elif "4" in str(gen)[:5]:
                        count.append(4)
                    elif "5" in str(gen)[:5]:
                        count.append(5)
                    elif "6" in str(gen)[:5]:
                        count.append(6)
                    elif "7" in str(gen)[:5]:
                        count.append(7)
                    elif "8" in str(gen)[:5]:
                        count.append(8)
                    elif "9" in str(gen)[:5]:
                        count.append(9)
                    elif "two" in str(gen)[:10]:
                        count.append(2)
                    elif "three" in str(gen)[:10]:
                        count.append(3)
                    elif "four" in str(gen)[:10]:
                        count.append(4)
                    elif "five" in str(gen)[:10]:
                        count.append(5)
                    elif "six" in str(gen)[:10]:
                        count.append(6)
                    elif "seven" in str(gen)[:10]:
                        count.append(7)
                    elif "eight" in str(gen)[:10]:
                        count.append(8)
                    elif "nine" in str(gen)[:10]:
                        count.append(9)
                    elif "ten" in str(gen)[:10]:
                        count.append(10)
                    if len(count) == 1:
                        pred_score.append(count[0])
                        scaled_gold.append(gold/10)
                        scaled_pred.append(count[0]/10)
                    else:
                        pred_score.append(99)
                    
            
            
            real_pred, real_gold = [], []
            normalized = []
            print(f'---{ch}---')
            print(f'Pred: {pred_score}')
            print(f'Gold: {gold_score}')
            for pre, gol in zip(pred_score, gold_score):
                if pre < 90:
                    real_pred.append(pre)
                    real_gold.append(gol)
                    normalized.append(0)
            
            norm_mse = mean_squared_error(scaled_pred, scaled_gold)    
            ch_nmse_list.append(norm_mse)
            
        
        if ch == 'personal_and_social_well-being':
            for a, gold, code in zip(answer, gold_score, code_list):
                if code != 'cldgng':
                    gen = a.split("my answer is")[1:]
                    gen = " ".join(gen)
                    gen = gen.split(".")[0]
                    gen = gen.strip()
                    
                    
                    if code == 'wkvlorg':
                        count = []
                        if 'At least once a week' in gen or 'at least once a week' in gen:
                            count.append(1)
                        if 'At least once a month' in gen or 'at least once a month' in gen:
                            count.append(2)
                        if 'At least once every three months' in gen or 'at least once every three months' in gen:
                            count.append(3)
                        if 'At least once every six months' in gen or 'at least once every six months' in gen:
                            count.append(4)
                        if 'Less often' in gen or 'less often' in gen:
                            count.append(5)
                        if 'Never' in gen or 'never' in gen:
                            count.append(6)
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold-1)/5)
                            scaled_pred.append((count[0]-1)/5)
                        else:
                            pred_score.append(99)
                        

                    elif code == 'optftr' or code == 'pstvms' or code == 'flrms' or code == 'dclvlf' or code == 'lchshcp' or code == 'accdng' or code == 'wrbknrm' or code == 'dngval' or code == 'nhpftr' or code == 'lotsgot' or code == 'lfwrs' or code == 'flclpla':
                        count = []
                        
                        if 'Agree' in gen or 'agree' in gen:
                            if 'Disagree' in gen or 'disagree' in gen:
                                if 'Neither agree nor disagree' in gen or 'neither agree nor disagree' in gen:
                                    count.append(3)
                                elif 'Disagree strongly' in gen or 'disagree strongly' in gen:
                                    count.append(5)
                                else:
                                    count.append(4)                            
                            elif 'Agree strongly' in gen or 'agree strongly' in gen:
                                count.append(1)
                            else:
                                count.append(2)
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold-1)/4)
                            scaled_pred.append((count[0]-1)/4)
                        else:
                            pred_score.append(99)
                        
                            
                        
                    elif code == 'fltdpr' or code == 'flteeff' or code == 'slprl' or code == 'wrhpp' or code == 'fltlnl' or code == 'enjlf' or code == 'fltsd' or code == 'cldgng' or code == 'enrglot' or code == 'fltanx' or code == 'fltpcfl':
                        count = []
                        if 'None or almost none of the time' in gen or 'none or almost none of the time' in gen:
                            count.append(1)
                        elif 'Some of the time' in gen or 'some of the time' in gen:
                            count.append(2)
                        elif 'Most of the time' in gen or 'most of the time' in gen:
                            count.append(3)
                        elif 'All or almost of the time' in gen or 'all or almost of the time' in gen:
                            count.append(4)
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold-1)/3)
                            scaled_pred.append((count[0]-1)/3)
                        else:
                            pred_score.append(99)
                        
                        
                    
                    elif code == 'tmdotwa' or code == 'flapppl' or code == 'deaimpp' or code == 'tmimdng' or code == 'tmabdng' or code == 'tmendng' or code == 'tnapsur' or code == 'sedirlf' or code == 'rehlppl' or code == 'prhlppl' or code == 'plinsoc':
                        count = []
                        if "0" in str(gen)[:5]:
                            if "10" in str(gen)[:5]:
                                count.append(10)
                            else:
                                count.append(0)
                        if "1" in str(gen)[:5]:
                            if "10" not in str(gen)[:5]:
                                count.append(1)   
                        elif "2" in str(gen)[:5]:
                            count.append(2)
                        elif "3" in str(gen)[:5]:
                            count.append(3)
                        elif "4" in str(gen)[:5]:
                            count.append(4)
                        elif "5" in str(gen)[:5]:
                            count.append(5)
                        elif "6" in str(gen)[:5]:
                            count.append(6)
                        elif "7" in str(gen)[:5]:
                            count.append(7)
                        elif "8" in str(gen)[:5]:
                            count.append(8)
                        elif "9" in str(gen)[:5]:
                            count.append(9)
                        elif "two" in str(gen)[:10]:
                            count.append(2)
                        elif "three" in str(gen)[:10]:
                            count.append(3)
                        elif "four" in str(gen)[:10]:
                            count.append(4)
                        elif "five" in str(gen)[:10]:
                            count.append(5)
                        elif "six" in str(gen)[:10]:
                            count.append(6)
                        elif "seven" in str(gen)[:10]:
                            count.append(7)
                        elif "eight" in str(gen)[:10]:
                            count.append(8)
                        elif "nine" in str(gen)[:10]:
                            count.append(9)
                        elif "ten" in str(gen)[:10]:
                            count.append(10)
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold)/10)
                            scaled_pred.append((count[0])/10)
                        else:
                            pred_score.append(99)
                        
                
                    elif code == 'physact':
                        count = []
                        if 'No days' in gen or 'no days' in gen or '0' in gen[:5]:
                            count.append(0)
                        if 'One day' in gen or 'one day' in gen or '1' in gen[:5]:
                            count.append(1)
                        if 'Two days' in gen or 'two days' in gen or '2' in gen[:5]:
                            count.append(2)
                        if 'Three days' in gen or 'three days' in gen or '3' in gen[:5]:
                            count.append(3)
                        if 'Four days' in gen or 'four days' in gen or '4' in gen[:5]:
                            count.append(4)
                        if 'Five days' in gen or 'five days' in gen or '5' in gen[:5]:
                            count.append(5)
                        if 'Six days' in gen or 'six days' in gen or '6' in gen[:5]:
                            count.append(6)
                        if 'Seven days' in gen or 'seven days' in gen or '7' in gen[:5]:
                            count.append(7)
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold)/7)
                            scaled_pred.append((count[0])/7)
                        else:
                            pred_score.append(99)
                        
                    else:
                        count = []
                        if "1" in str(gen)[:5]:
                            count.append(1)      
                        elif "2" in str(gen)[:5]:
                            count.append(2)
                        elif "3" in str(gen)[:5]:
                            count.append(3)
                        elif "4" in str(gen)[:5]:
                            count.append(4)
                        elif "5" in str(gen)[:5]:
                            count.append(5)
                        elif "6" in str(gen)[:5]:
                            count.append(6)
                        elif "two" in str(gen)[:10]:
                            count.append(2)
                        elif "three" in str(gen)[:10]:
                            count.append(3)
                        elif "four" in str(gen)[:10]:
                            count.append(4)
                        elif "five" in str(gen)[:10]:
                            count.append(5)
                        elif "six" in str(gen)[:10]:
                            count.append(6)

                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold-1)/5)
                            scaled_pred.append((count[0]-1)/5)
                        else:
                            pred_score.append(99)

            real_pred, real_gold = [], []
            normalized = []
            
            print(f'---{ch}---')
            print(f'Pred: {pred_score}')
            print(f'Gold: {gold_score}')
            
            for pre, gol in zip(pred_score, gold_score):
                if pre < 90:
                    real_pred.append(pre)
                    real_gold.append(gol)
                    normalized.append(0)
            
            norm_mse = mean_squared_error(scaled_pred, scaled_gold)    
            ch_nmse_list.append(norm_mse)
        
        if ch == 'politics':
            gen = answer[0].split("my answer is")[1:]
            gen = " ".join(gen)
            gen = gen.split(".")[0]
            gen = gen.strip()
            
            count = []
            if 'Very interested' in gen or 'very interested' in gen:
                count.append(1)
            if 'Quite interested' in gen or 'quite interested' in gen:
                count.append(2)
            if 'Hardly interested' in gen or 'hardly interested' in gen:
                count.append(3)
            if 'Not at all interested' in gen or 'not at all interested' in gen:
                count.append(4)
            if len(count) == 1:
                pred_score.append(count[0])
                scaled_gold.append((gold_score[0]-1)/3)
                scaled_pred.append((count[0]-1)/3)
            else:
                pred_score.append(99)
                
            for a, gold, code in zip(answer[1:], gold_score[1:], code_list[1:]):
                if code != 'wrkorg':
                    count = []
                    gen = a.split("my answer is")[1:]
                    gen = " ".join(gen)
                    gen = gen.split(".")[0]
                    gen = gen.strip()
                    
                    if code == 'trstlgl' or code == 'trstplc' or code == 'trstplt' or code == 'trstep' or code == 'trstun' or code == 'trstprt' or code == 'trstprl' or code == 'implvdm' or code == 'dmcntov' or code == 'lrscale' or code == 'stflife' or code == 'stfeco' or code == 'stfgov' or code == 'stfdem' or code == 'stfedu' or code == 'stfhlth' or code == 'euftf' or code == 'imbgeco' or code == 'imueclt' or code == 'imwbcnt':
                        count = []
                        if "0" in str(gen)[:5]:
                            if "10" in str(gen)[:5]:
                                count.append(10)
                            else:
                                count.append(0)
                        if "1" in str(gen)[:5]:
                            if "10" not in str(gen)[:5]:
                                count.append(1)   
                        elif "2" in str(gen)[:5]:
                            count.append(2)
                        elif "3" in str(gen)[:5]:
                            count.append(3)
                        elif "4" in str(gen)[:5]:
                            count.append(4)
                        elif "5" in str(gen)[:5]:
                            count.append(5)
                        elif "6" in str(gen)[:5]:
                            count.append(6)
                        elif "7" in str(gen)[:5]:
                            count.append(7)
                        elif "8" in str(gen)[:5]:
                            count.append(8)
                        elif "9" in str(gen)[:5]:
                            count.append(9)
                        elif "two" in str(gen)[:10]:
                            count.append(2)
                        elif "three" in str(gen)[:10]:
                            count.append(3)
                        elif "four" in str(gen)[:10]:
                            count.append(4)
                        elif "five" in str(gen)[:10]:
                            count.append(5)
                        elif "six" in str(gen)[:10]:
                            count.append(6)
                        elif "seven" in str(gen)[:10]:
                            count.append(7)
                        elif "eight" in str(gen)[:10]:
                            count.append(8)
                        elif "nine" in str(gen)[:10]:
                            count.append(9)
                        elif "ten" in str(gen)[:10]:
                            count.append(10)
                                    
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold_score[0])/10)
                            scaled_pred.append((count[0])/10)
                        else:
                            pred_score.append(99)
                            
                    elif code == 'vote':
                        count = []
                        if 'Yes' in gen or 'yes' in gen:
                            count.append(1)
                        if 'No' in gen or 'no' in gen:
                            if 'Not eligible to vote' in gen or 'not eligible to vote' in gen:                    
                                count.append(3)
                            else:
                                count.append(2)
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold-1)/2)
                            scaled_pred.append((count[0]-1)/2)
                        else:
                            pred_score.append(99)    
                    
                    elif code == 'wrkprty' or code == 'wrkorg' or code == 'badge' or code == 'sgnptit' or code == 'pbldmn' or code == 'bctprd' or code == 'clsprty' or code == 'contplt':
                        count = []
                        gen = gen[:10]
                        if 'Yes' in gen or 'yes' in gen:
                            count.append(1)
                        if 'No' in gen or 'no' in gen:
                            count.append(2)
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold-1))
                            scaled_pred.append((count[0]-1))
                        else:
                            pred_score.append(99) 
                    
                    elif code == 'gincdif' or code == 'freehms':
                        count = []
                        if 'Neither agree nor disagree' in gen or 'neither agree nor disagree' in gen:
                            count.append(3)
                        elif 'Disagree strongly' in gen or 'disagree strongly' in gen:
                            count.append(5)
                        elif 'Agree strongly' in gen or 'agree strongly' in gen:
                            if 'Disagree strongly' not in gen and 'disagree strongly' not in gen:
                                count.append(1)
                        elif 'Disagree' in gen or 'disagree' in gen:
                            if 'Disagree strongly' not in gen and 'disagree strongly' not in gen:
                                count.append(4)
                        elif 'Agree' in gen or 'agree' in gen:
                            if 'Agree strongly' not in gen and 'agree strongly' not in gen and 'Neither agree nor disagree' not in gen and 'neither agree nor disagree' not in gen and 'Disagree' not in gen and 'disagree' not in gen:
                                count.append(2)
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold-1)/4)
                            scaled_pred.append((count[0]-1)/4)
                        else:
                            pred_score.append(99)
                    
                    elif code == 'imsmetn' or code == 'imdfetn' or code == 'impcntr':
                        count = []
                        if 'Allow many to come and live here' in gen or 'allow many to come and live here' in gen:
                            count.append(1)
                        if 'Allow some' in gen or 'allow some' in gen:
                            count.append(2)
                        if 'Allow a few' in gen or 'allow a few' in gen or 'allow few' in gen or 'Allow few' in gen:
                            count.append(3)
                        if 'Allow none' in gen or 'allow none' in gen:
                            count.append(4)                           
                        
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold-1)/3)
                            scaled_pred.append((count[0]-1)/3)
                        else:
                            pred_score.append(99)
                                    
                    else:
                        count = []
                        if "1" in str(gen)[:2]:
                            count.append(1)      
                        elif "2" in str(gen)[:2]:
                            count.append(2)
                        elif "3" in str(gen)[:2]:
                            count.append(3)
                        elif "4" in str(gen)[:2]:
                            count.append(4)
                        elif "5" in str(gen)[:2]:
                            count.append(5)
                        elif "6" in str(gen)[:2]:
                            count.append(6)
                        if len(count) == 1:
                            pred_score.append(count[0])
                            scaled_gold.append((gold-1)/5)
                            scaled_pred.append((count[0]-1)/5)
                        else:
                            pred_score.append(99)            


            real_pred, real_gold = [], []
            normalized = []
            
            print(f'---{ch}---')
            print(f'Pred: {pred_score}')
            print(f'Gold: {gold_score}')
            
            for pre, gol in zip(pred_score, gold_score):
                if pre < 90:
                    real_pred.append(pre)
                    real_gold.append(gol)
                    normalized.append(0)
            
            norm_mse = mean_squared_error(scaled_pred, scaled_gold)    
            ch_nmse_list.append(norm_mse)
                
        
        if ch == 'understanding_democracy':
            for a, gold, code in zip(answer, gold_score, code_list):
                count = []
                gen = a.split('my answer is')[1:]
                gen = " ".join(gen)
                gen = gen.split(".")[0]
                gen = gen.strip()
                
                if code == 'fplvdm':
                    count = []
                    if 'free to express their political views openly' in gen:
                        count.append(1)
                    elif 'prevented from expressing' in gen:
                        count.append(2)
                    elif 'It depends on the circumstances' in gen:
                        count.append(5)
                    if len(count) == 1:
                        pred_score.append(count[0])
                        scaled_gold.append((gold-1)/4)
                        scaled_pred.append((count[0]-1)/4)
                    else:
                        pred_score.append(99)  
                
                elif code == 'chpldm':
                    count = []
                    if 'government should change its' in gen:
                        count.append(1)
                    elif 'government should stick to its planned policies' in gen:
                        count.append(2)
                    elif 'It depends on the circumstances' in gen:
                        count.append(5)
                    if len(count) == 1:
                        pred_score.append(count[0])
                        scaled_gold.append((gold-1)/4)
                        scaled_pred.append((count[0]-1)/4)
                    else:
                        pred_score.append(99) 
                
                elif code == 'gvspcdm':
                    count = []
                    if 'single party forms the government' in gen:
                        count.append(1)
                    elif 'or more parties in coalition' in gen:
                        count.append(2)
                    elif 'It depends on the circumstances' in gen:
                        count.append(5)
                    if len(count) == 1:
                        pred_score.append(count[0])
                        scaled_gold.append((gold-1)/4)
                        scaled_pred.append((count[0]-1)/4)
                    else:
                        pred_score.append(99)  
                        
                else:
                    count = []
                    if "0" in str(gen)[:10]:
                        if "10" in str(gen)[:10]:
                            count.append(10)
                        else:
                            count.append(0)
                    if "1" in str(gen)[:10]:
                        if "10" not in str(gen)[:10]:
                            count.append(1)   
                    elif "2" in str(gen)[:10]:
                        count.append(2)
                    elif "3" in str(gen)[:10]:
                        count.append(3)
                    elif "4" in str(gen)[:10]:
                        count.append(4)
                    elif "5" in str(gen)[:10]:
                        count.append(5)
                    elif "6" in str(gen)[:10]:
                        count.append(6)
                    elif "7" in str(gen)[:10]:
                        count.append(7)
                    elif "8" in str(gen)[:10]:
                        count.append(8)
                    elif "9" in str(gen)[:10]:
                        count.append(9)
                    elif "zero" in str(gen)[:10]:
                        count.append(0)
                    # elif "one" in str(gen)[:10]:
                    #     count.append(1)
                    elif "two" in str(gen)[:10]:
                        count.append(2)
                    elif "three" in str(gen)[:10]:
                        count.append(3)
                    elif "four" in str(gen)[:10]:
                        count.append(4)
                    elif "five" in str(gen)[:10]:
                        count.append(5)
                    elif "six" in str(gen)[:10]:
                        count.append(6)
                    elif "seven" in str(gen)[:10]:
                        count.append(7)
                    elif "eight" in str(gen)[:10]:
                        count.append(8)
                    elif "nine" in str(gen)[:10]:
                        count.append(9)
                    elif "ten" in str(gen)[:10]:
                        count.append(10)
                    if len(count) == 1:
                        pred_score.append(count[0])
                        scaled_gold.append((gold)/10)
                        scaled_pred.append((count[0])/10)
                    else:
                        pred_score.append(99)
                
            real_pred, real_gold = [], []
            normalized = []
            
            print(f'---{ch}---')
            print(f'Pred: {pred_score}')
            print(f'Gold: {gold_score}')
            
            for pre, gol in zip(pred_score, gold_score):
                if pre < 90:
                    real_pred.append(pre)
                    real_gold.append(gol)
                    normalized.append(0)
            

            norm_mse = mean_squared_error(scaled_pred, scaled_gold)    
            ch_nmse_list.append(norm_mse)
        
    return ch_nmse_list
    
if __name__ == '__main__':
    mode = sys.argv[1]          # argument, survey, argument_survey
    strategy = 'min'
    threshold = 3
    
    model_name = 'llama'       
    
    if mode == 'base':
        mode_path = f'./results/ess/media_and_social_trust'
    if mode == 'survey':
        mode_path = f'./results/ess/media_and_social_trust/survey/{model_name}/{strategy}'
    if mode == 'argument':
        mode_path = f'./results/ess/media_and_social_trust/argument/{model_name}/TH_{threshold}'
    if mode == 'argument_survey':
        mode_path = f'./results/ess/media_and_social_trust/argument_survey/{model_name}/{strategy}_TH_{threshold}'    
    country_list = os.listdir(mode_path)
    country_list.sort()
    
    chapters = ['media_and_social_trust', 'personal_and_social_well-being', 'politics', 'subjective_well-being', 'understanding_democracy']
    
    
    total_mse = [[], [], [], [], []]
    total_nmse = [[], [], [], [], []]
    
    for country_name in country_list:
        ch_nmse_list = calc_ess_answer(country_name=country_name, 
                                        model_name=model_name, 
                                        mode=mode, 
                                        strategy=strategy,
                                        threshold=threshold
                                        )

        
        total_nmse[0].append(ch_nmse_list[0])
        total_nmse[1].append(ch_nmse_list[1])
        total_nmse[2].append(ch_nmse_list[2])
        total_nmse[3].append(ch_nmse_list[3])

    country_list.append('TOTAL')
    
    total_nmse[0].append(sum(total_nmse[0])/len(total_nmse[0]))
    total_nmse[1].append(sum(total_nmse[1])/len(total_nmse[1]))
    total_nmse[2].append(sum(total_nmse[2])/len(total_nmse[2]))
    total_nmse[3].append(sum(total_nmse[3])/len(total_nmse[3]))

    data = {
        'Country':country_list,
        'media_and_social_trust':total_mse[0],
        'personal_and_social_well-being':total_mse[1],
        'politics':total_mse[2],
        'understanding_democracy':total_mse[3]
    }
    
    norm_data = {
        'Country':country_list,
        'media_and_social_trust':total_nmse[0],
        'personal_and_social_well-being':total_nmse[1],
        'politics':total_nmse[2],
        'understanding_democracy':total_nmse[3]
    }
    
    ESS_norm_df = pd.DataFrame(data=norm_data)
    ESS_norm_df.to_csv(f'./ess_nmse_{mode}.csv', sep='\t')

    
