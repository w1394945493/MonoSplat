import torch
from torch import Generator
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .data_module import DataLoaderCfg, DataLoaderStageCfg
from .dataset_nuscences import DatasetNuScencesCfg,nuScenesDataset

class DataModuleForNuScences(LightningDataModule):
    dataset_cfg: DatasetNuScencesCfg
    data_loader_cfg: DataLoaderCfg
    global_rank: int
    def __init__(
        self,
        dataset_cfg: DatasetNuScencesCfg,
        data_loader_cfg: DataLoaderCfg,
        global_rank: int = 0,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        train_dataset = nuScenesDataset(
            self.dataset_cfg.resolution,
            split = "demo",
            use_center=self.dataset_cfg.use_center,
            use_first=self.dataset_cfg.use_first,
            use_last = self.dataset_cfg.use_last,
            near = self.dataset_cfg.near,
            far = self.dataset_cfg.far,
        )
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.data_loader_cfg.train.batch_size,
            num_workers=self.data_loader_cfg.train.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):

        val_dataset = nuScenesDataset(
            self.dataset_cfg.resolution,
            split = "demo",
            use_center=self.dataset_cfg.use_center,
            use_first=self.dataset_cfg.use_first,
            use_last = self.dataset_cfg.use_last,
            near = self.dataset_cfg.near,
            far = self.dataset_cfg.far,
        )
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        test_dataset = nuScenesDataset(
            self.dataset_cfg.resolution,
            split = "demo",
            use_center=self.dataset_cfg.use_center,
            use_first=self.dataset_cfg.use_first,
            use_last = self.dataset_cfg.use_last,
            near = self.dataset_cfg.near,
            far = self.dataset_cfg.far,
        )
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            shuffle=False,
        )