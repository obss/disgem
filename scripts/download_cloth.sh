#!/bin/bash

WORKING_DIR=$(pwd)


if [ ! -d "$WORKING_DIR/data/CLOTH" ]
then
  wget http://www.cs.cmu.edu/~glai1/data/cloth/CLOTH.zip
  mkdir "$WORKING_DIR/data"
  unzip CLOTH.zip -d "$WORKING_DIR/data"
  rm -rf CLOTH.zip
else
  echo "Directory ${WORKING_DIR}/data/CLOTH exists. To initiate re-download, delete the directory."
fi
