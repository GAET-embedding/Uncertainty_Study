GIT_PATH=java_data/different_author/hibernate-orm/data/hibernate-orm/

# cd that repo directory
cd $GIT_PATH
# set searching author name
AUTHOR1="sebersole"
AUTHOR2="gbadner"
AUTHOR3="dreab8"
AUTHOR3="Sanne"

# git log --pretty="%H" --author=$AUTHOR | 
#     while read commit_hash; 
#     do 
#         git show --oneline --name-only $commit_hash | tail -n+2; 
#     done | sort | uniq

# git log --no-merges --author=$AUTHOR --name-only --pretty=format:"" | sort -u

# git log --pretty= --author=$AUTHOR --name-only | sort -u | wc -l

# git log --author=$AUTHOR1 --diff-filter=A --name-only --format=""

commit_files2=$(git log --author=$AUTHOR2 --diff-filter=A --name-only --format="")

java_files2=$(find $commit_files2 -name '*.java')

echo java_files2

# git log --author=$AUTHOR2 --pretty=oneline --graph --name-status --diff-filter=A

# echo $commit_files2[0]

# git log --author=$AUTHOR3 --diff-filter=A --name-only --format=""

# git log --author=$AUTHOR4 --diff-filter=A --name-only --format=""
