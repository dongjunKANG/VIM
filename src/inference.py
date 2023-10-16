import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as p
import gc
import sys

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
import torch
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
from tqdm import tqdm
import psutil

import util.option as opt
import util.main_utils as main_utils
from util.train_utils import (
    _collate_fn, _flatten, _find_save_path,
    _load_state, _save_state
)


def inference_argument_generation(model, country, mode, strategy, threshold):
    test_df = pd.read_csv('./data/argument_generation/test.csv', sep='\t')
    conclusions = test_df['Conclusion'].tolist()
    conclusion_list = []
    for c in conclusions:
        if c not in conclusion_list:
            conclusion_list.append(c)
    stance_question = []
    for c in conclusion_list:           
        text_1 = f"""Do you agree or disagree with "{c}"? Answer:"""
        stance_question.append(text_1)

    stance = []
    for question in stance_question:
        inputs = tokenizer(question, return_tensors='pt')
        sample_output = model.generate(
            input_ids=inputs['input_ids'].to(device), 
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=10,
            top_p=0.5
        )
        print("Output:\n" + 100 * '-')
        gen = tokenizer.batch_decode(sample_output.detach().cpu().numpy(), skip_special_tokens=True)
        print(gen)
        resp = gen[0].split('Answer:')[1]
        if 'I agree' in resp[:10]:            
            stance.append('in favor of')
        if 'I disagree' in resp[:10]:
            stance.append('against')


    reason_question = []
    for c, st in zip(conclusion_list, stance):
        if st == 'in favor of':
            text_2 = f"""Explain in detail why you agree with "{c}". Answer:"""
            reason_question.append(text_2)
        if st == 'against':
            text_2 = f"""Explain in detail why you disagree with "{c}". Answer:"""
            reason_question.append(text_2)
        
    reason = []
    for question in reason_question:
        inputs = tokenizer(question, return_tensors='pt')
        sample_output = model.generate(
            input_ids=inputs['input_ids'].to(device), 
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=60,
            top_p=0.5
        )
        print("Output:\n" + 100 * '-')
        gen = tokenizer.batch_decode(sample_output.detach().cpu().numpy(), skip_special_tokens=True)
        print(gen)
        resp = gen[0].split('Answer:')[1]
        reason.append(resp)
    
    data = {
        'Conclusion':conclusion_list,
        'Stance':stance,
        'Premise':reason
    }

    if mode == 'base':    
        path = f'./results/generation/base'
    if mode == 'survey':    
        path = f'./results/generation/survey/{model_name}/{strategy}/{country}'
    if mode == 'argument':    
        path = f'./results/generation/argument/{model_name}/TH_{threshold}/{country}'
    if mode == 'argument_survey':
        path = f'./results/generation/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country}'
                
    csv_path = p.join(path, f'gen_{country}.csv')
    os.makedirs(path, exist_ok=True)
    gen_df = pd.DataFrame(data=data)
    gen_df.to_csv(csv_path, sep='\t')
    

def inference_pvq(model, country, mode, strategy, threshold):
    pvq_df = pd.read_excel('./data/pvq/pvq40.xlsx')
    questions = pvq_df['question'].tolist()
    questionnaire_list = []
    
    for q in questions:        
        text = f"""Indicate for the following statement whether it is 1. 'Not like me at all', 2. 'Not like me', 3. 'A little like me', 4. 'Somewhat like me', 5. 'Like me', 6. 'Very much like me' as a description of you. Statement: '{q}'. Answer:"""           
        questionnaire_list.append(text)
    answers = []
    
    for question in questionnaire_list:
        inputs = tokenizer(question, return_tensors='pt')
        sample_output = model.generate(
            input_ids=inputs['input_ids'].to(device), 
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=20,
            top_p=0.5
        )
        print("Output:\n" + 100 * '-')
        gen = tokenizer.batch_decode(sample_output.detach().cpu().numpy(), skip_special_tokens=True)
        print(gen)
        if mode == 'argument':
            resp = gen[0].split('Answer:')[1]
        else:
            resp = gen[0].split('Answer:')[1]
        
        answers.append(resp)

    data = {
        'Questionnaire':questionnaire_list,
        'Answer':answers
    }

    if mode == 'base':
        path = f'./results/pvq_results/base'
    if mode == 'survey':    
        path = f'./results/pvq_results/survey/{model_name}/{strategy}/{country}'
    if mode == 'argument':    
        path = f'./results/pvq_results/argument/{model_name}/TH_{threshold}/{country}'
    if mode == 'argument_survey':
        path = f'./results/pvq_results/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country}'
    
    if mode == 'base':
        csv_path = p.join(path, f'result_base.csv')
    else:
        csv_path = p.join(path, f'result_{country}.csv')
    os.makedirs(path, exist_ok=True)
    gen_df = pd.DataFrame(data=data)
    gen_df.to_csv(csv_path, sep='\t')
    
def inference_ess(model, country, mode, strategy, threshold):
    ess_df_1 = pd.read_excel('./data/ESS/questionnaire/ESS_round_6_media_and_social_trust.xlsx')
    ess_df_2 = pd.read_excel('./data/ESS/questionnaire/ESS_round_6_personal_and_social_well-being.xlsx')
    ess_df_3 = pd.read_excel('./data/ESS/questionnaire/ESS_round_6_politics.xlsx')
    ess_df_4 = pd.read_excel('./data/ESS/questionnaire/ESS_round_6_subjective_well-being.xlsx')
    ess_df_5 = pd.read_excel('./data/ESS/questionnaire/ESS_round_6_understanding_democracy.xlsx')

    ess_df = [ess_df_1, ess_df_2, ess_df_3, ess_df_4, ess_df_5]
    ess_version = ['media_and_social_trust', 'personal_and_social_well-being', 'politics', 'subjective_well-being', 'understanding_democracy']
    
    for ver, df in zip(ess_version, ess_df):
        code = df['Code'].tolist()
        questionnaire = df['Questionnaire'].tolist()
        answer = []
        for q in questionnaire:
            inputs = tokenizer(q, return_tensors='pt')
            sample_output = model.generate(
                input_ids=inputs['input_ids'].to(device), 
                attention_mask=inputs["attention_mask"].to(device),
                max_new_tokens=40,
                top_p=0.5
            )
            print("Output:\n" + 100 * '-')
            gen = tokenizer.batch_decode(sample_output.detach().cpu().numpy(), skip_special_tokens=True)
            print(gen)
            resp = gen[0].split('Answer:')[1]
            answer.append(resp)
        
        data = {
            'Code':code,
            'Questionnaire':questionnaire,
            'Answer':answer
        }
        
        if mode == 'base':    
            path = f'./results/ess/{ver}/base'
        if mode == 'survey':    
            path = f'./results/ess/{ver}/survey/{model_name}/{strategy}/{country}'
        if mode == 'argument':    
            path = f'./results/ess/{ver}/argument/{model_name}/TH_{threshold}/{country}'
        if mode == 'argument_survey':
            path = f'./results/ess/{ver}/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country}'
                
        csv_path = p.join(path, f'ess_{country}.csv')
        os.makedirs(path, exist_ok=True)
        answer_df = pd.DataFrame(data=data)
        answer_df.to_csv(csv_path, sep='\t')
        
def inference_valuenet(model, country, mode, strategy, threshold):
    value_list = ['Achievement', 'Benevolence', 'Conformity', 'Hedonism', 'Power', 'Security', 'Self-direction', 'Stimulation', 'Tradition', 'Universalism']
    total_questionnaire = []
    total_value = []
    total_stance = []
    answers = []
    
    for value in value_list:
        for stance in (['positive', 'negative']):
            df = pd.read_csv(f'./data/consistency/valuenet/{value}_{stance}.csv', sep='\t')
            scenario = df['scenario'].tolist()
            questionnaire_list = []
            for q in scenario:           
                text = f"""Do you agree the behavior or expression in the following scenario similar to you? Please answer "I agree" or "I disagree". Scenario: '{q}'. Answer:"""           

                questionnaire_list.append(text)
                total_questionnaire.append(text)
                total_value.append(value)
                total_stance.append(stance)
                
            
            
            for question in questionnaire_list:
                inputs = tokenizer(question, return_tensors='pt')
                sample_output = model.generate(
                    input_ids=inputs['input_ids'].to(device), 
                    attention_mask=inputs["attention_mask"].to(device),
                    max_new_tokens=10,
                    top_p=0.5
                )
                print("Output:\n" + 100 * '-')
                gen = tokenizer.batch_decode(sample_output.detach().cpu().numpy(), skip_special_tokens=True)
                print(gen)
                resp = gen[0].split('Answer:')[1]
                answers.append(resp)

    data = {
        'Questionnaire':total_questionnaire,
        'Answer':answers,
        'Value':total_value,
        'Stance':total_stance
    }

    if mode == 'base':
        path = f'./results/valuenet/base'
    if mode == 'survey':    
        path = f'./results/valuenet/survey/{model_name}/{strategy}/{country}'
    if mode == 'argument':    
        path = f'./results/valuenet/argument/{model_name}/TH_{threshold}/{country}'
    if mode == 'argument_survey':
        path = f'./results/valuenet/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country}'
    
    
    if mode == 'base':
        csv_path = p.join(path, f'result_base.csv')
    else:
        csv_path = p.join(path, f'result_{country}.csv')
        
    os.makedirs(path, exist_ok=True)
    gen_df = pd.DataFrame(data=data)
    gen_df.to_csv(csv_path, sep='\t')

if __name__ == '__main__':  
    country = sys.argv[1] 
    GPU_NUM = sys.argv[2]
    mode = sys.argv[3]          # argument, survey, argument_survey
    strategy = 'min'
    threshold = 3
    
    generation = 'yes'
    pvq = 'yes'
    ess = 'yes'
    valuenet = 'yes'
    
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

    model_name = 'llama'
    model_name_or_path = 'decapoda-research/llama-7b-hf'
    
    batch_size = 1
    seed = 42
    set_seed(seed)

    print("Get tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Get model...")   # Get model
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    if mode == 'base':
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model = get_peft_model(model, peft_config)
    if mode == 'survey':
        epoch_num = _find_save_path(f"ckpt/survey/{model_name}/{strategy}/{country}")
        # epoch_num = "epoch_3"
        peft_model_id = f"ckpt/survey/{model_name}/{strategy}/{country}/{epoch_num}"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
    if mode == 'argument':
        epoch_num = _find_save_path(f"ckpt/argument/{model_name}/TH_{threshold}/{country}")
        peft_model_id = f"ckpt/argument/{model_name}/TH_{threshold}/{country}/{epoch_num}"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
    if mode == 'argument_survey':
        epoch_num = _find_save_path(f"ckpt/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country}")
        peft_model_id = f"ckpt/argument_survey/{model_name}/{strategy}_TH_{threshold}/{country}/{epoch_num}"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
    
    
    model = model.to(device)
    if generation == 'yes':
        inference_argument_generation(model, country, mode, strategy, threshold)
    if pvq == 'yes':
        inference_pvq(model, country, mode, strategy, threshold)
    if ess == 'yes':
        inference_ess(model, country, mode, strategy, threshold)
    if valuenet == 'yes':
        inference_valuenet(model, country, mode, strategy, threshold)

