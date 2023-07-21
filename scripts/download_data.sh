#!/bin/bash

WORKING_DIR=$(pwd)


if [ ! -d "$WORKING_DIR/data/CLOTH" ]
then
  # Download CLOTH
  wget http://www.cs.cmu.edu/~glai1/data/cloth/CLOTH.zip
  mkdir "$WORKING_DIR/data"
  unzip CLOTH.zip -d "$WORKING_DIR/data"
  rm -rf CLOTH.zip
else
  echo "Directory ${WORKING_DIR}/data/CLOTH exists. To initiate re-download, delete the directory."
fi

if [ ! -d "$WORKING_DIR/data/DGen" ]
then
  # download DGen
  mkdir "$WORKING_DIR/data/DGen"
  cd "$WORKING_DIR/data/DGen"
  wget https://huggingface.co/datasets/AndyChiang/dgen/raw/main/DGen_test_cleaned.json
else
  echo "Directory ${WORKING_DIR}/data/DGen exists. To initiate re-download, delete the directory."
fi