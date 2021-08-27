############### find all zip file that needs extrating

target_folder=/glusterfs/data/yxl190090/distribution_shift/csharp_data
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

    des_dir=$target_folder/$file"_cs"
    mkdir $des_dir
    cs_list=$(find -name '*.cs') # find all C# files

    for csfile in $cs_list
    do
        cp $csfile $des_dir # copy each C# file to des folder
    done 

    cd .. # return to target_folder
done