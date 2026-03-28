#!/bin/bash

hf download --repo-type dataset YaxuanLi/UAVM_Baseline_SuperGlue_Output --local-dir .

unzip train_matches_data.zip -d ../
unzip test_matches_data.zip -d ../

rm -rf train_matches_data.zip
rm -rf test_matches_data.zip


