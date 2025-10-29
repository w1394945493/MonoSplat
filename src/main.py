# 设置进程名
from setproctitle import setproctitle
setproctitle("wangyushen")

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["COLUMNS"] = "60"
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import TQDMProgressBar

import sys
sys.path.append('/home/lianghao/wangyushen/Projects/MonoSplat/')
# ? -----------------------------------------#
# ? 10.23 注释以下内容(限制的太严格了)
# Configure beartype and jaxtyping.
# with install_import_hook(
#     ("src",), # todo：指定项目中的模块路径
#     ("beartype", "beartype"), # todo python的运行时类型检查工具
# ): #! beartype库：会在函数调用时严格检查参数类型

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.misc.LocalLogger import LocalLogger
from src.misc.step_tracker import StepTracker
from src.misc.wandb_tools import update_checkpoint_path
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config", # todo 配置文件所在文件夹
    config_name="main", # todo Hydra加载main.yaml作为主配置文件
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # todo 结果保存路径
    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    # latest_run = output_dir.parents[1] / "latest-run"
    # os.system(f"rm {latest_run}")
    # os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity, # todo 在wandb的组织或用户名
            project=cfg_dict.wandb.project, # todo 项目名
            mode=cfg_dict.wandb.mode, # todo 运行模式：online联网上传、offline本地保存
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})", # todo 运行名
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir, # todo wandb 本地缓存路径
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True)) # todo LearningRateMonitor：记录学习率变化

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger() #! todo 自定义的logger指存图

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints", # todo 保存路径
            every_n_train_steps=cfg.checkpointing.every_n_train_steps, # todo 每隔多少步保存一次
            save_top_k=cfg.checkpointing.save_top_k, # todo 只保留多少个模型
            monitor="info/global_step", # todo 监控全局训练部署，用于判断哪个checkpoint是最新的
            mode="max",  # save the lastest k ckpt, can do offline test later # todo 步数越大表示越新，取最大
        )
    )
    # callbacks.append(TQDMProgressBar(refresh_rate=10))
    # callbacks.append(RichProgressBar())
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # todo 预训练权重文件路径
    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()
    # todo -----------------------------------#
    # todo Trainer: from pytorch_lightning import Trainer
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto", # todo 自动调用所有可用gpu训练(可用显卡：如CUDA_VISIBLE_DEVICES=3,4,5,6)
        num_nodes=cfg.trainer.num_nodes, # todo 1 单机
        # strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval, # todo 若0-1之间，如0.5，则为每个epoch的50%时评估一次；大于1时(整数)，每隔多少迭代步评估一次
        # enable_progress_bar=True, #todo  是否显示进度条 True/False
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps, # todo 最大训练步数
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps, # todo 正式训练前做的安全检查
        # log_every_n_steps=1, # todo 默认50，每多少步记录一次日志
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    # todo -------------------------#
    # todo (10.22 wys) 定义model
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train, #! cfg.train
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }
    model_wrapper = ModelWrapper(**model_kwargs)

    # todo -------------------------#
    # todo (10.22 wys) 数据集定义
    if cfg.dataset.name == 're10k':
        data_module = DataModule(
            cfg.dataset, #! cfg.dataset.name: re10k
            cfg.data_loader,
            step_tracker,
            global_rank=trainer.global_rank,
        )
    #? -----------------------------#
    #? 增加了nuscences (wys 10.23)
    elif cfg.dataset.name == 'nuscences':
        from src.dataset.data_module_nuscences import DataModuleForNuScences
        data_module = DataModuleForNuScences(
            dataset_cfg=cfg.dataset,
            data_loader_cfg=cfg.data_loader,
            global_rank=trainer.global_rank,
        )

    if cfg.mode == "train":
        strict_load = False
        # todo 加载单目深度模型
        # only load monodepth
        if cfg.checkpointing.pretrained_monodepth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_monodepth, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(cyan(f"Loaded pretrained monodepth: {cfg.checkpointing.pretrained_monodepth}"))

        # todo 加载预训练权重文件
        if cfg.checkpointing.pretrained_model is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
            print(cyan(f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"))

        # todo
        trainer.fit(model_wrapper,
                    datamodule=data_module,
                    ckpt_path=(checkpoint_path if cfg.checkpointing.resume else None)
                )

    else:
        # todo -------------------------#
        # todo test：dataset 和 dataloader 实例化在test()中实现
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
