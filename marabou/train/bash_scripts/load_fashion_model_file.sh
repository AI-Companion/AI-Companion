#!/bin/bash
cd $PWD
set -e
export fileid_h5_file=$1
export filename_h5_file=$2
export fileid_class_file=$3
export filename_class_file=$4
# collect h5 file
wget -q --show-progress --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$fileid_h5_file -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$fileid_h5_file -O $filename_h5_file && rm -rf /tmp/cookies.txt# collect model file
# collect class file
wget -q --show-progress --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$fileid_class_file -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$fileid_class_file -O $filename_class_file && rm -rf /tmp/cookies.txt