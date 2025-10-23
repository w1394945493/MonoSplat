# 设置进程名
from setproctitle import setproctitle
setproctitle("wangyushen")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

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


import sys
sys.path.append('/home/lianghao/wangyushen/Projects/MonoSplat/')
# ? 
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
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # todo 预训练权重文件路径
    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    # todo -------------------------#
    # todo (10.22 wys) 定义model
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
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
