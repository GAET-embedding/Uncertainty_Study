############### find all zip file that needs extrating

target_folder=/glusterfs/data/yxl190090/distribution_shift/python_data
cd $target_folder

zipfile=$(find -mindepth 0 -maxdepth 1 -name '*.zip') # find all zipfiles
echo "ALL zip files in the root dir: \n$zipfile"

for file in $zipfile
do 
    # first extract all files in the zip file
    file=${file#"./"} # get rid of the "./" prefix
    echo "extracting file $file ..." 
    unzip -q $file # -q to suppress the printing of extracted files

    # then copy the .java files to des folder
    file=${file%".zip"} # get rid of the ".zip" suffix
    echo "copying java files from $file ..." 
    cd $file

    # des_dir=$target_folder/$file"_java"
    des_dir=$target_folder/$file"_python"
    mkdir $des_dir
    # java_list=$(find -name '*.java') # find all java files
    python_list=$(find -name '*.py') # find all python files
    
    # for javafile in $java_list
    # do
    #     cp $javafile $des_dir # copy each java file to des folder
    # done 

    for pythonfile in $python_list
    do
        cp $pythonfile $des_dir # copy each python file to des folder
    done 

    cd .. # return to target_folder
done


####################################################################################################
#################################   create train/test/valid   ######################################
####################################################################################################

# cd $target_folder
# mkdir train
# mkdir test1
# mkdir test2
# mkdir test3
# # mkdir test4

# mv gradle-REL_1.9-rc-4 train
# # mv gradle_5.2.0 train
# # mv wildfly_16.0.0.Beta1 train

# mv gradle-REL_2.13 test1
# mv gradle-5.3.0 test2
# mv gradle-6.8.3 test3
# # mv hadoop_3.2.2 test4

