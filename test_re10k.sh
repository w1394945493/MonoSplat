
# 10 context views
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=/path/to/checkpoint \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_nctx10.json \
    test.compute_scores=true \
    wandb.mode=disabled

# 2 context views
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=/path/to/checkpoint \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_nctx2.json \
    test.compute_scores=true \
    wandb.mode=disabled


# todo (wys 10.22)
# todo Hydra: 用于配置管理的python框架
'''
# 基础命令：
# python -m src.main: 运行src目录下的main.py模块，Hydra会自动加载yaml配置
# 命令行参数解析：
# 一般写法: key=value 或 +key=value，其中 +表示新增配置文件
# +experiment=re10k 表示在config/experiment/re10k.yaml 下加载实验相关参数
# mode=test 覆盖main.yaml或者re10k.yaml中的mode参数
# dataset/view_sampler=evaluation: 加载dataset/view_sampler/evaluation.yaml文件
# test.compute_scores=true，在yaml文件中写法为：
test:
    compute_scores:true
'''


python -m src.main +experiment=re10k \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=/path/to/checkpoint \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_nctx2.json \
    test.compute_scores=true \
    wandb.mode=disabled



# todo MonoSplat：评估
python /home/lianghao/wangyushen/Projects/MonoSplat/src/main.py \
