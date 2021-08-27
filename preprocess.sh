#!/usr/bin/env bash
###########################################################
# Change the following values to preprocess a new dataset.
# TRAIN_DIR, VAL_DIR and TEST_DIR should be paths to      
#   directories containing sub-directories with .java files
#   each of {TRAIN_DIR, VAL_DIR and TEST_DIR} should have sub-dirs,
#   and data will be extracted from .java files found in those sub-dirs).
# DATASET_NAME is just a name for the currently extracted 
#   dataset.                                              
# MAX_CONTEXTS is the number of contexts to keep for each 
#   method (by default 200).                              
# WORD_VOCAB_SIZE, PATH_VOCAB_SIZE, TARGET_VOCAB_SIZE -   
#   - the number of words, paths and target words to keep 
#   in the vocabulary (the top occurring words and paths will be kept). 
#   The default values are reasonable for a Tesla K80 GPU 
#   and newer (12 GB of board memory).
# NUM_THREADS - the number of parallel threads to use. It is 
#   recommended to use a multi-core machine for the preprocessing 
#   step and set this value to the number of cores.
# PYTHON - python3 interpreter alias.
# DIR=java_data/different_project/data
# DIR=java_data/different_author/elasticsearch/data
# DIR=java_data/different_time/data
DIR=python_data

TRAIN_DIR=$DIR/train
VAL_DIR=$DIR/val
# TEST_DIR=$DIR/gradle/val
TEST_DIR1=$DIR/test1
TEST_DIR2=$DIR/test2
TEST_DIR3=$DIR/test3
# DATASET_NAME=java_project
DATASET_NAME=python_project
# TRG_DIR=java_data/different_author/$DATASET_NAME
# TRG_DIR=java_data/different_project/$DATASET_NAME
TRG_DIR=$DIR/$DATASET_NAME

### Check if a directory exists ###
if [ -d $TRG_DIR ] 
then
  echo "Directory ${TRG_DIR} exists."
else
  mkdir $TRG_DIR
fi

MAX_CONTEXTS=200
WORD_VOCAB_SIZE=1301136
PATH_VOCAB_SIZE=911417
TARGET_VOCAB_SIZE=261245
NUM_THREADS=64
PYTHON=python3
###########################################################

TRAIN_DATA_FILE=${DATASET_NAME}.train.raw.txt
VAL_DATA_FILE=${DATASET_NAME}.val.raw.txt
TEST_DATA_FILE1=${DATASET_NAME}.test1.raw.txt
TEST_DATA_FILE2=${DATASET_NAME}.test2.raw.txt
TEST_DATA_FILE3=${DATASET_NAME}.test3.raw.txt
# TEST_DATA_FILE=${DATASET_NAME}.test.raw.txt
EXTRACTOR_JAR=JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar

mkdir -p data
mkdir -p data/${DATASET_NAME}

echo "Extracting paths from test set1 ..."
${PYTHON} JavaExtractor/extract.py --dir ${TEST_DIR1} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > ${TEST_DATA_FILE1}
echo "Finished extracting paths from test set1"
echo "Extracting paths from test set2 ..."
${PYTHON} JavaExtractor/extract.py --dir ${TEST_DIR2} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > ${TEST_DATA_FILE2}
echo "Finished extracting paths from test set2"
echo "Extracting paths from test set3 ..."
${PYTHON} JavaExtractor/extract.py --dir ${TEST_DIR3} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > ${TEST_DATA_FILE3}
echo "Finished extracting paths from test set3"
# echo "Extracting paths from test set ..."
# ${PYTHON} JavaExtractor/extract.py --dir ${TEST_DIR} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > ${TEST_DATA_FILE}
# echo "Finished extracting paths from test set"
echo "Extracting paths from validation set ..."
${PYTHON} JavaExtractor/extract.py --dir ${VAL_DIR} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > ${VAL_DATA_FILE}
echo "Finished extracting paths from validation set"
echo "Extracting paths from training set..."
${PYTHON} JavaExtractor/extract.py --dir ${TRAIN_DIR} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} | shuf > ${TRAIN_DATA_FILE}
echo "Finished extracting paths from training set"

TARGET_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.tgt.c2v
ORIGIN_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.ori.c2v
PATH_HISTOGRAM_FILE=data/${DATASET_NAME}/${DATASET_NAME}.histo.path.c2v

echo "Creating histograms from the training data"
cat ${TRAIN_DATA_FILE} | cut -d' ' -f1 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${ORIGIN_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${PATH_HISTOGRAM_FILE}

${PYTHON} preprocess.py \
  --train_data ${TRAIN_DATA_FILE} --val_data ${VAL_DATA_FILE} --test_data1 ${TEST_DATA_FILE1} \
  --test_data2 ${TEST_DATA_FILE2} --test_data3 ${TEST_DATA_FILE3} \
  --max_contexts ${MAX_CONTEXTS} --word_vocab_size ${WORD_VOCAB_SIZE} --path_vocab_size ${PATH_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --word_histogram ${ORIGIN_HISTOGRAM_FILE} \
  --path_histogram ${PATH_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} \
  --output_name ${TRG_DIR}/${DATASET_NAME}

# If all went well, the raw data files can be deleted, because preprocess.py creates new files 
# with truncated and padded number of paths for each example.
rm ${TRAIN_DATA_FILE} ${VAL_DATA_FILE} ${TEST_DATA_FILE1} ${TEST_DATA_FILE2} ${TEST_DATA_FILE3}