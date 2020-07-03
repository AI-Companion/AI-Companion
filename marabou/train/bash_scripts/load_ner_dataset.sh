#!/bin/bash
cd $PWD
set -e
export file_url=$1
export file_name=data/ner_dataset.csv

if [ ! -d "data" ] # if data folder doesnt exist
then # uncompress data
    mkdir -p data/
    unzip -q $file_url -d data
elif [ ! -f "$file_name" ] # check if file exists
then # uncompress data
    unzip -q $file_url -d data
else
    echo "----> csv file is already stored, it will be removed after training"
fi