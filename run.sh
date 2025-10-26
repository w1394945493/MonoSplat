# todo Mono Splat 推理/评估
# todo 在re10k数据集上：
python /home/lianghao/wangyushen/Projects/MonoSplat/src/demo.py \
    +experiment=re10k \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=/home/lianghao/wangyushen/data/wangyushen/Weights/monosplat/epoch_63-step_300000.ckpt \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_nctx2.json \
    dataset.name=re10k \
    dataset.roots=[/home/lianghao/wangyushen/data/wangyushen/Datasets/re10k/re10k_subset] \
    data_loader.test.batch_size=1 \
    data_loader.test.num_workers=0 \
    test.output_path=/home/lianghao/wangyushen/data/wangyushen/Output/mono_splat/test \
    test.compute_scores=true \
    wandb.mode=disabled \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/mono_splat \

# todo 在nuScences数据集上
python /home/lianghao/wangyushen/Projects/MonoSplat/src/demo.py \
    +experiment=nuscences \
    mode=test \
    dataset/view_sampler=evaluation \
    checkpointing.load=/home/lianghao/wangyushen/data/wangyushen/Weights/monosplat/epoch_63-step_300000.ckpt \
    dataset.name=nuscences \
    data_loader.test.batch_size=1 \
    data_loader.test.num_workers=0 \
    test.output_path=/home/lianghao/wangyushen/data/wangyushen/Output/mono_splat/test_nuScence \
    test.compute_scores=true \
    wandb.mode=disabled \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/mono_splat \

# todo -------------------------#
# todo MonoSplat 训练
python /home/lianghao/wangyushen/Projects/MonoSplat/src/main.py \
    +experiment=re10k \
    mode=train \
    checkpointing.pretrained_model=/home/lianghao/wangyushen/data/wangyushen/Weights/monosplat/epoch_63-step_300000.ckpt \
    checkpointing.resume=false \
    dataset.view_sampler.index_path=assets/evaluation_index_re10k_nctx2.json \
    dataset.name=re10k \
    dataset.roots=[/home/lianghao/wangyushen/data/wangyushen/Datasets/re10k/re10k_subset] \
    data_loader.train.batch_size=1 \
    data_loader.train.num_workers=0 \
    data_loader.val.batch_size=1 \
    data_loader.val.num_workers=0 \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/mono_splat_train_re10k \

#? ------------------------------------------------#
#? (10.23 wys) MonoSplat 在 nuScences 数据集上训练
CUDA_VISIBLE_DEVICES=3,4,5,6 python /home/lianghao/wangyushen/Projects/MonoSplat/src/main.py \
    +experiment=nuscences \
    mode=train \
    checkpointing.pretrained_model=/home/lianghao/wangyushen/data/wangyushen/Weights/monosplat/epoch_63-step_300000.ckpt \
    checkpointing.resume=false \
    dataset.name=nuscences \
    data_loader.train.batch_size=1 \
    data_loader.train.num_workers=4 \
    data_loader.val.batch_size=1 \
    data_loader.val.num_workers=4 \
    trainer.max_steps=100_001 \
    trainer.val_check_interval=10_000 \
    checkpointing.every_n_train_steps=10_000 \
    train.print_log_every_n_steps=10_000 \
    wandb.mode=disabled \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/mono_splat/nuscenes/train

#? (10.23 wys) MonoSplat 在 nuScences 数据集上训练 分辨率：112×200
CUDA_VISIBLE_DEVICES=7 python /home/lianghao/wangyushen/Projects/MonoSplat/src/main.py \
    +experiment=nuscences \
    mode=train \
    checkpointing.pretrained_model=/home/lianghao/wangyushen/data/wangyushen/Weights/monosplat/epoch_63-step_300000.ckpt \
    checkpointing.resume=false \
    dataset.name=nuscences \
    dataset.resolution=[112,200] \
    data_loader.train.batch_size=1 \
    data_loader.train.num_workers=4 \
    data_loader.val.batch_size=1 \
    data_loader.val.num_workers=4 \
    trainer.max_steps=100_001 \
    trainer.val_check_interval=5_000 \
    checkpointing.every_n_train_steps=10_000 \
    wandb.project=monosplat_112x200 \
    output_dir=/home/lianghao/wangyushen/data/wangyushen/Output/mono_splat/nuscenes/train_2_112x200

#? (10.25 wys) MonoSplat 在 nuScences 数据集上评估 分辨率：112×200
CUDA_VISIBLE_DEVICES=3,4,5,6 python /home/lianghao/wangyushen/Projects/MonoSplat/src/main.py \
    +experiment=nuscences \
    mode=test \
    checkpointing.load=/home/lianghao/wangyushen/data/wangyushen/Output/mono_splat/nuscenes/train_112x200/checkpoints/epoch_0-step_100000.ckpt \
    dataset.name=nuscences \
    dataset.resolution=[112,200] \
    data_loader.test.batch_size=1 \
    data_loader.test.num_workers=4 \
    test.output_path=/home/lianghao/wangyushen/data/wangyushen/Output/mono_splat/nuscenes/test \
    test.compute_scores=true \
    test.save_image=false \
    # wandb.mode=disabled \