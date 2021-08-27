#!/bin/bash

################################################################################
# MODULE_ID=0 # 0 is code summary
# # MODEL_TYPE=code2vec
# MODEL_TYPE=lstm
# # MODEL_TYPE=codebert
# SHIFT_TYPE=different_time
# # SHIFT_TYPE=different_project
# # SHIFT_TYPE=different_author/elasticsearch
# PROJECT=java_project
# DATA_DIR=java_data/$SHIFT_TYPE/java_pkl
# RES_DIR=program_tasks/code_summary/result/$SHIFT_TYPE/$PROJECT/$MODEL_TYPE
# OOD_DIR=python_pkl
# TRAIN_BATCH_SIZE=128
# TEST_BATCH_SIZE=128
# SAVE_DIR=Uncertainty_Results_new/$SHIFT_TYPE/$PROJECT/$MODEL_TYPE
# # SAVE_DIR=Uncertainty_Results_new/$SHIFT_TYPE/$MODEL_TYPE
################################################################################

################################################################################
MODULE_ID=1 # 1 is code completion
# MODEL_TYPE=word2vec
# MODEL_TYPE=lstm
MODEL_TYPE=codebert
SHIFT_TYPE=different_time
# SHIFT_TYPE=different_project
# SHIFT_TYPE=different_author
PROJECT=java_project
# PROJECT=elasticsearch
DATA_DIR=program_tasks/code_completion/dataset/$SHIFT_TYPE/$PROJECT/
RES_DIR=program_tasks/code_completion/result/$SHIFT_TYPE/$PROJECT/$MODEL_TYPE
OOD_DIR=program_tasks/code_completion/dataset/python_project
TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=32
MAX_SIZE=200
SAVE_DIR=Uncertainty_Results_new/$SHIFT_TYPE/$PROJECT/$MODEL_TYPE
################################################################################

if [ ! -d $SAVE_DIR ]; then
  mkdir $SAVE_DIR
else
  echo dir exist
fi

CUDA_VISIBLE_DEVICES=3 python -m Metric.test_uncertainty \
--module_id=$MODULE_ID --res_dir=$RES_DIR \
--data_dir=$DATA_DIR --save_dir=$SAVE_DIR \
--train_batch_size=$TRAIN_BATCH_SIZE --test_batch_size=$TEST_BATCH_SIZE \
--max_size=$MAX_SIZE --ood_dir=$OOD_DIR

# --train_batch_size=$TRAIN_BATCH_SIZE --test_batch_size=$TEST_BATCH_SIZE \
# --max_size=$MAX_SIZE
