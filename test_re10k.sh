
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=/home/yifliu3/code/anysplat_v7/outputs/2024-11-02/15-58-12/checkpoints/epoch_4-step_20000.ckpt \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_tmp.json \
    test.compute_scores=true \
    wandb.mode=disabled