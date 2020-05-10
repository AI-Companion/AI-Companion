#!/bin/bash
cd $PWD
set -e
export file_url=http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
export data_file_name=data/imdb.tar.gz

download_data () {
    echo "download ongoing"
    wget -q --show-progress -O $data_file_name $file_url
    tar -xzf $data_file_name --directory data/
    rm $data_file_name
}

echo "Collecting sentiment analysis data"
if [ ! -d "data" ] # if data folder doesnt exist
then # load data from gdrive
    mkdir -p data/
    download_data
elif [ ! "$(ls -A data)" ] # data folder exists and is empty
then # load data from gdrive
    download_data
else
    echo "data already downloaded"
fi