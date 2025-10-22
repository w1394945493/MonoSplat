# Mono Splat
# 需要安装的包
pip install hydra-core dacite beartype e3nn colorspacious
git clone https://github.com/facebookresearch/dinov2.git

# 数据集：可以在PixelSplat或MVSplat中，下载re10k数据集子集

# 代码(重要的部分):
数据集：src/dataset/dataset_re10k.py
模型推理：src/model/model_wrapper.py

# 跑通推理