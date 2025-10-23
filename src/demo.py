# 设置进程名
from setproctitle import setproctitle
setproctitle("wangyushen")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
from colorama import Fore
from tqdm import tqdm

import sys
sys.path.append('/home/lianghao/wangyushen/Projects/MonoSplat/')
# todo 配置文件相关
from src.config import load_typed_root_config
from src.global_cfg import set_cfg
# todo 模型相关
from src.misc.wandb_tools import update_checkpoint_path
from src.misc.step_tracker import StepTracker
from src.model.encoder import get_encoder
from src.model.decoder import get_decoder
from src.loss import get_losses
from src.model.model_wrapper import ModelWrapper
# todo 数据集相关
from src.dataset.data_module import DataModule


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

@hydra.main(
    version_base=None,
    config_path="../config", # todo 配置文件所在文件夹
    config_name="main", # todo Hydra加载main.yaml作为主配置文件
)
def main(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))

    # todo 预训练权重文件路径
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # todo -------------------------#
    # todo (10.22 wys) 定义model
    step_tracker = StepTracker()
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset), #! 这里用到了cfg.dataset, bg_color
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }
    model_wrapper = ModelWrapper(**model_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_wrapper.to(device)


    # todo -------------------------#
    # todo (10.23 wys) 加载预训练权重模型
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            model_wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
            print(cyan(f"Successfully load checkpoint from {checkpoint_path}."))

    # todo -------------------------#
    # todo (10.22 wys) 数据集定义
    if cfg.dataset.name == 're10k':
        data_module = DataModule(
            cfg.dataset, #! cfg.dataset.name: re10k
            cfg.data_loader,
            step_tracker,
            global_rank=0, # todo global_rank = 0
        )
        val_dataloader = data_module.test_dataloader()
    elif cfg.dataset.name == 'nuscences':
        # todo 定义nuScence dadaset
        from src.dataset.dataset_nuscences import nuScenesDataset
        from torch.utils.data import DataLoader

        demo_dataset = nuScenesDataset(
            cfg.dataset.resolution,
            split="demo",
            use_center=cfg.dataset.use_center,
            use_first=cfg.dataset.use_first,
            use_last = cfg.dataset.use_last,
            near = cfg.dataset.near,
            far = cfg.dataset.far,
        )
        val_dataloader = DataLoader(
            demo_dataset,
            cfg.data_loader.test.batch_size,
            shuffle=False,
            num_workers=cfg.data_loader.test.num_workers
        )
    else:
        raise ValueError(f'unsupported dataset: {cfg.dataset.name}')





    with torch.no_grad():
        model_wrapper.eval()
        for i_iter, batch in enumerate(tqdm(val_dataloader, desc="Processing")):
            # model_wrapper.test_step(batch=batch,batch_idx=i_iter)
            model_wrapper.forward_test(batch=batch,batch_idx=i_iter) #? 增加了一个将数据移动到指定设备的过程
    return






# todo (wys 10.23) 重写一下MonoSplat推理/评估过程
if __name__=='__main__':

    main()