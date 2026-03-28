unzip University-Release.zip

# 创建新文件夹 tours
mkdir -p train_tour

cp -r University-Release/train/drone/* train_tour/

rm -rf University-Release/
rm -rf University-Release.zip

hf download --repo-type dataset YaxuanLi/UAVM_2026_test --local-dir .

tar -xvf train.tar -C .
tar -xvf test.tar -C .
tar -xvf test_tour.tar -C .

rm -rf train.tar
rm -rf test.tar
rm -rf test_tour.tar
rm -rf .cache
