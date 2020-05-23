#!/bin/bash
cd $PWD
set -e
export file_url=$1
export embedding_file_name=embeddings/stanford_embedding.zip

download_data () {
    echo "download ongoing"
    wget -q --show-progress -O $data_file_name $file_url
    unzip $embedding_file_name -d embeddings/
    rm $embedding_file_name
}

echo "Collecting stanford 6B embeddings"
if [ ! -d "embeddings" ] # if embeddings folder doesnt exist
then # load data
    mkdir -p embeddings/
    download_data
elif [ ! "$(ls -A data)" ] # embedings folder exists and is empty
then # load data
    download_data
else
    echo "data already downloaded"
fi