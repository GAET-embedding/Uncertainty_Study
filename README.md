# Uncertainty Study
A study of the existing state-of-the-art uncertainty methods under program distribution shift:

This is a PyTorch implementation of two programming tasks, namely, code summarization and code completion with three model archiectures for each task. We also use the Java and Python extractors for preprocessing the input code. It can also be extended to other languages, since the PyTorch network is agnostic to the input programming language.

## Requirements
- [https://www.python.org/downloads/release/python-380/](https://www.python.org/downloads/release/python-380/)
- PyTorch 1.9.0 ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))

## Quickstart
### Step 0: Cloning this repository
```
git clone https://github.com/GAET-embedding/Uncertainty_Study.git
cd Uncertainty_Study
```
### Step 1: Download our shifted datasets (from the seven Java projects)
#### Download our preprocessed data for code summarization
```
wget xxxxx 
tar -xvzf xxxxx
```
#### Download our preprocessed data for code completion
```
cd program_tasks/code_completion
wget xxxxx
tar -xvzf xxxxx
```
#### Download the preprocessed Java-large dataset (~16 M examples, compressed: 6.3GB) and Python150k dataset for OOD detection (~150 K examples, compressed: 526MB)
```
wget https://s3.amazonaws.com/code2vec/data/java14m_data.tar.gz
tar -xvzf java14m_data.tar.gz
wget http://files.srl.inf.ethz.ch/data/py150.tar.gz
tar -xzvf py150.tar.gz
```
### Step 2: Training a model
You can either download an already trained model, or train a new model from scratch.
#### Download a trained model
```
wget xxxxx
tar -xvzf xxxxx
```
#### Training a model from scratch
To train a model from scratch:
- Edit the file [program_tasks/code_summary/train.sh](program_tasks/code_summary/train.sh) (code summary) and file [program_tasks/code_completion/train.sh](program_tasks/code_completion/train.sh) (code completion) to point to the right preprocessed data and a specific model archiecture.
- Before training, you can edit the configuration hyper-parameters in these two files.
- Run the two shell scripts:
```
# code summary
bash program_tasks/code_summary/train.sh
# code completion
bash program_tasks/code_completion/train.sh
```
### Step 3: Measuring the five uncertainty scores
- Edit the file [Uncertainty_Eval/get_uncertainty.sh](Uncertainty_Eval/get_uncertainty.sh) to point to the right preprocessed data, a specific task and a specific model.
- Run the script [Uncertainty_Eval/get_uncertainty.sh](Uncertainty_Eval/get_uncertainty.sh):
```
bash Uncertainty_Study/get_uncertainty.sh
```
### Step 4: Evaluation the effectiveness of the five uncertainty methods on both error/success prediction and in-/out-of-distribution detection:
- Edit the file [Uncertainty_Eval/evaluation.py](Uncertainty_Eval/evaluation.py) to point to the target evaluation choice (error/success prediction or in-/out-of-distribution detection).
- Run the script [Uncertainty_Eval/evaluation.py](Uncertainty_Eval/evaluation.py):
```
python Uncertainty_Eval/evaluation.py
```
### Step 5: Evaluation the effectiveness of the five uncertainty methods on input validation:
- Edit the file [filter.py](filter.py) to point to the right preprocessed data, a specific task and a specific model.
- Run the script [filter.py](filter.py):
```
python filter.py
```

