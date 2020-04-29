set -e
mkdir -p data/raw
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -O data/raw/imdb.tar.gz
tar -xzf data/raw/imdb.tar.gz --directory data/raw
rm data/raw/imdb.tar.gz
