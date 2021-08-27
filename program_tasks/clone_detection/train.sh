#!/bin/bash
PROJECT=gradle
RES_DIR=program_tasks/clone_detection/result/$PROJECT

if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi

EPOCHS=50
BATCH=512
LR=0.005
TRAIN_DATA=data/$PROJECT'_pkl'/train.pkl
TEST_DATA1=data/$PROJECT'_pkl'/test1.pkl
TEST_DATA2=data/$PROJECT'_pkl'/test2.pkl
TEST_DATA3=data/$PROJECT'_pkl'/test3.pkl
# TRAIN_DATA='/glusterfs/data/yxl190090/EmbeddingEvaluation/dataset/java-small-preprocess/train.pkl'
# TEST_DATA1='/glusterfs/data/yxl190090/EmbeddingEvaluation/dataset/java-small-preprocess/test.pkl'
# TEST_DATA2='/glusterfs/data/yxl190090/EmbeddingEvaluation/dataset/java-small-preprocess/test.pkl'
# TEST_DATA3='/glusterfs/data/yxl190090/EmbeddingEvaluation/dataset/java-small-preprocess/test.pkl'


EMBEDDING_TYPE=1
EMBEDDING_DIM=100                 #dimension of vectors
EMBEDDING_PATH='/'                #file for pre-trained vectors
EXPERIMENT_NAME='clone_detection'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=2 python -m program_tasks.clone_detection.main \
--train_data=$TRAIN_DATA \
--test_data1=$TEST_DATA1 \
--test_data2=$TEST_DATA2 \
--test_data3=$TEST_DATA3 \
--embed_type=$EMBEDDING_TYPE --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
--embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH --res_dir=$RES_DIR \
--experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG

