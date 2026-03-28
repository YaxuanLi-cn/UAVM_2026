#!/bin/bash

hf download --repo-type dataset YaxuanLi/UAVM_2026_test --local-dir .

unzip train_matches_data.zip
unzip test_matches_data.zip

rm -rf train_matches_data.zip
rm -rf test_matches_data.zip


