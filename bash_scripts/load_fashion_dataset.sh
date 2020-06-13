#!/bin/bash
cd $PWD
set -e
export zipfile=data/fashion_imagenet.zip
export folder_url=data/fashion_imagenet/classes_url

if [ -d $folder_url ]
then
    echo "----> data extracted"
elif [ -f $zipfile ]
then
    echo "----> inflating"
    unzip -q $zipfile -d $PWD"/data"
fi

