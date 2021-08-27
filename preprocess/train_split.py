import os
from posix import listdir
import random
from shutil import move
from tqdm import tqdm

JAVA_PROJECTS = [
    'wildfly', 
    'gradle', 
    'spring-framework', 
    'presto', 
    'hadoop', 
    'hibernate-orm', 
    'elasticsearch',
]

DATA_DIR = 'java_data/different_project/data'
JAVA_PROJ_DIR = {
    j: os.path.join(DATA_DIR, j) for j in JAVA_PROJECTS
}
train_val_ratio = 0.3

# print(JAVA_PROJ_DIR)
for java_proj in JAVA_PROJECTS:
    java_proj_dir = JAVA_PROJ_DIR[java_proj]
    java_proj_train = os.path.join(java_proj_dir, 'train')
    java_proj_val = os.path.join(java_proj_dir, 'val')

    train_proj_name = os.listdir(java_proj_train)[0]

    src_dir = os.path.join(java_proj_train, train_proj_name)
    trg_dir = os.path.join(java_proj_val, train_proj_name)

    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)

    java_files = os.listdir(src_dir) # all java files in train dir
    print('project name: {}, train size: {}'.format(
        java_proj, len(java_files)
    ))
    
    # random sample 30% files from train to validation files
    val_files = random.sample(java_files, int(len(java_files)*train_val_ratio))
    for file in tqdm(val_files):
        move(
            os.path.join(src_dir, file),
            trg_dir
        )
    print("finished transferring project {}".format(java_proj))
    print("# train files: {}, # val files: {}".format(
        len(os.listdir(src_dir)), len(os.listdir(trg_dir))
    ))


