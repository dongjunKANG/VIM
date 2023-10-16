# Title
From Values to Opinions: Predicting Human Behaviors and Stances Using Value-Injected Large Language Models
-------------------
# Overview
This code provides the implementation of "Value Injection Method (VIM)".
-------------------
# How to run
In `src` folder, we make bash script file to train and inference, and evaluate for Value Injection Method (VIM).

Please move the `data` folder into the `software/src` before running.

In this code, train, reference, and performance calculation are conducted for 128 groups (28 country groups, 100 social groups) obtained from the European Social Survey.

- `train.sh` : a bash script file to train LLaMA-7B with VIM

`bash train.sh`
- `inference.sh`: a bash script file to inference LLaMA-7B with VIM 

`bash inference.sh`
- `calculation.sh` : a bash script file to calculate the evaluation task 1, 2, 3, 4

`bash calculation.sh`
- `preprocessing.py` : a python file to preprocessing Touche23-ValueEval dataset

`python preprocessing.py`

-------------------
# Train
This code train VIM, VIM_AG, VIM_QA method.
- VIM
- VIM_AG
- VIM_QA
-------------------
# Model
- LLaMA-7B (7 billion parameters)
-------------------
# Evaluation Task
1. Portrait Values Questionnaire (PVQ)
2. Touché23-ValueEval dataset
3. VALUENET
4. European Social Survey (ESS)
-------------------
# Metric
- Normalized Mean Squared Error (NMSE)
-------------------
# Dataset
We use these public dataset ...
- Touché23-ValueEval dataset : https://touche.webis.de/semeval23/touche23-web/index.html
- VALUENET dataset : https://touche.webis.de/semeval23/touche23-web/index.html
- ESS : https://www.europeansocialsurvey.org/
-------------------
# Tested Environment
- python 3.9.7
- numpy 1.22.4
- pandas 1.3.3
- scikit-learn 1.0
- torch 1.13.1
- tokenizers 0.13.2
- transformers 4.27.0.dev0
- peft 0.3.0.dev0
--------------------

