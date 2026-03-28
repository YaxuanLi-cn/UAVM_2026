#!/bin/bash

# Your need to first download [University-1652] upon request (Usually I will reply you in 5 minutes). You may use the [request template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).
unzip University-Release.zip

# Creating training set
mkdir -p train_tour

cp -r University-Release/train/drone/* train_tour/

rm -rf University-Release/
rm -rf University-Release.zip

# Will download the new competition test set
hf download --repo-type dataset YaxuanLi/UAVM_2026_test --local-dir .

tar -xvf train.tar -C .
tar -xvf test.tar -C .
tar -xvf test_tour.tar -C .

rm -rf train.tar
rm -rf test.tar
rm -rf test_tour.tar
rm -rf .cache
