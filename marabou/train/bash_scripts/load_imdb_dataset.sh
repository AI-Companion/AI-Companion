#!/bin/bash
cd $PWD
set -e
export file_url=$1
export zipped_file_name=data/imdb.tar.gz

download_data () {
    echo "----> download ongoing"
    wget -q --show-progress -O $zipped_file_name $file_url
    tar -xzf $zipped_file_name --directory data/
    rm $zipped_file_name
}

if [ ! -d "data" ] # if data folder doesnt exist
then # load data from gdrive
    mkdir -p data/
    download_data
elif [ ! -d "data/aclImdb" ] # check dataset folder exist
then # load data from gdrive
    download_data
else
    echo "----> data already downloaded"
fi