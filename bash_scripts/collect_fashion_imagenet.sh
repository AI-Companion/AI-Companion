#!/bin/bash
cd $PWD
set -e
export folder_url=data/fashion_cnn/classes_url

for f in $folder_url/*
do
    echo "processing $f"
    fname=${f##*/}
    mkdir -p data/fashion_cnn/$fname
    folder_name=data/fashion_cnn/${f##*/}
    while IFS= read -r url
    do
        echo "url $url"
        url_root=${url##*/}
        fname=$folder_name/$url_root
        if [ ! -f $fname ] #check if file exists in the disk
        then
            wget -T 3 -t 2 -q -N --show-progress -O $fname $url || true
        else
            echo "file exists"
        fi
    done < "$f"
done
