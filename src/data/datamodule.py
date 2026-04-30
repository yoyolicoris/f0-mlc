"""Lightning DataModule classes for F0 audio datasets.

Each DataModule wraps one dataset class from dataset_new.py and handles
train/val/test splitting and DataLoader creation.

To use with Hydra, update configs/data/*.yaml to reference
``src.data.datamodules_new.XxxDataModule`` instead of ``src.data.datasets.XxxDataModule``.
"""

from typing import Any, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, random_split
from omegaconf.listconfig import ListConfig


# ---------------------------------------------------------------------------
# Base DataModule
# ---------------------------------------------------------------------------


class F0AudioLitDataModule(LightningDataModule):
    """Base Lightning DataModule for F0 audio datasets."""

    def __init__(
        self,
        datasets_train_val,
        datasets_test,
        batch_size,
        num_workers,
        pin_memory: bool = False,
        train_val_split: Tuple[float, float] = (0.95, 0.05),
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["datasets_train_val", "datasets_test"])

        if isinstance(datasets_train_val, ListConfig):
            self.datasets_train_val = datasets_train_val
        else:
            self.datasets_train_val = [datasets_train_val]

        if isinstance(datasets_test, ListConfig):
            self.datasets_test = datasets_test
        else:
            self.datasets_test = [datasets_test]

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # concatenate datasets
        data_train_val = ConcatDataset(self.datasets_train_val)
        self.data_test = ConcatDataset(self.datasets_test)

        # randomly split into train / val subsets
        self.data_train, self.data_val = random_split(
            dataset=data_train_val,
            lengths=self.hparams.train_val_split,
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass
