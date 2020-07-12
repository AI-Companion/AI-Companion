#!/bin/bash
cd "$MARABOU_HOME/marabou/train"
set -e
export zipfile=data/ner_dataset.zip
export file_url=data/ner_dataset.csv
export FILEID=$1

if [ -f $file_url ] # dataset exists, do nothing
then
    echo "----> data extracted"
elif [ -f $zipfile ] # zip file exists, inflate it
then
    echo "----> dataset downloaded, inflating ..."
    unzip -q $zipfile -d $PWD"/data"
else # zip file doesnt exist, download then inflate
    echo "----> downloading dataset"
    wget -q --show-progress --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILEID -O $zipfile && rm -rf /tmp/cookies.txt
    echo "----> dataset downloaded, inflating ..."
    unzip -q $zipfile -d $PWD"/data"
fi
