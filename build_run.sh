#!/bin/sh

set -ex

project=webcam

#rm -rf ./build/
mkdir -p build

cd build
cmake .. 
#make install
make

# run build project
./$project

# run tests immediately
#ctest --verbose

cd ..

