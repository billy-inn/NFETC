#!/bin/sh

echo "Downloading corpus"
wget http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip
unzip corpus.zip
rm corpus.zip

echo "Downloading word embeddings..."
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
