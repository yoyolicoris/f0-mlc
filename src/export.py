from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch
from pathlib import Path

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="export.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    args = (torch.rand(8, 64000),)
    m = torch.export.export(
        model.f0_estimator,
        args,
        dynamic_shapes={
            "x": (torch.export.Dim.DYNAMIC, torch.export.Dim.DYNAMIC),
        },
    )
    # print(m)
    with torch.no_grad():
        ep_for_inference = m.run_decompositions(decomp_table={})
        ep_minimal = m.run_decompositions(decomp_table=None)
    print(ep_for_inference.graph_module.print_readable(print_output=False))

    pred_dict = model.f0_estimator(*args)
    inf_dict = ep_for_inference.module()(*args)
    for k in pred_dict.keys():
        torch.testing.assert_close(pred_dict[k], inf_dict[k])

    log_dir = Path(cfg.paths.output_dir) / "exported_models"
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.export.save(ep_for_inference, log_dir / "for_inference.pt2")
    torch.export.save(ep_minimal, log_dir / "minimal.pt2")

    saved_for_inference = torch.export.load(log_dir / "for_inference.pt2")
    saved_minimal = torch.export.load(log_dir / "minimal.pt2")

    inf_dict_2 = saved_for_inference.module()(*args)
    for k in pred_dict.keys():
        torch.testing.assert_close(pred_dict[k], inf_dict_2[k])


if __name__ == "__main__":
    main()
