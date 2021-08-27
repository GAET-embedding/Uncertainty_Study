python_dir=/glusterfs/data/yxl190090/distribution_shift/python_data/pyfiles/data
trg_dir=/glusterfs/data/yxl190090/distribution_shift/python_data/extracted_files

mkdir $trg_dir

cd $python_dir
echo "extracting python files from $python_dir ..." 

python_list=$(find -name '*.py') # find all py files

for pyfile in $python_list
    do
        cp $pyfile $trg_dir # copy each C# file to des folder
    done 

cd .. # return to target_folder