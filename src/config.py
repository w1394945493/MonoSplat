from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg

# todo (wys 10.23)
from typing import Union
from .dataset.dataset_nuscences import DatasetNuScencesCfg

@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    pretrained_model: Optional[str]
    pretrained_monodepth: Optional[str]
    resume: Optional[bool] = True


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int
    num_nodes: Optional[int] = 1

# todo RootCfg：顶层配置，每个字段有类型标注
@dataclass
class RootCfg: # todo：定义了整个项目所有配置的结构和类型
    wandb: dict
    mode: Literal["train", "test"]
    # dataset: DatasetCfg
    dataset: Union[DatasetCfg, DatasetNuScencesCfg]
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    # todo from dacite import from dict:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg), # todo OmegaConf.to_container(): 把Hydra配置转成普通dict
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]

# todo: 将Hydra加载出来的配置字典转换成带检查类型的dataclass实例
def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg, # todo Hydra加载的实例
        RootCfg, # todo 目标dataclass是RootCFg
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )
