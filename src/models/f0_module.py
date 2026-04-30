from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from df0.f0_utils import get_f0_targets, hz_to_cents
from lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.aggregation import MeanMetric
from torchmetrics.collections import MetricCollection

from src.utils.evaluate import compute_metrics
from src.data.components.utils import mix_at_snr


class F0LitModule(LightningModule):
    """Lightning module for F0 estimation training and evaluation.

    Wraps an F0 estimator model and handles training loops, validation, testing, and metric computation.
    Supports multi-representation F0 prediction with optional voicing detection and SNR mixing augmentation.
    """

    def __init__(
        self,
        f0_estimator: torch.nn.Module,
        f0_selector: torch.nn.Module,
        cent_tolerance: float,
        snr_range,
        unvoiced_mode: str,
        target_blurring: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scheduler_config: dict,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["f0_estimator", "f0_selector"])

        # Normalize SNR range to [min, max] tuple
        if self.hparams.snr_range is not None:
            snr_range = self.hparams.snr_range
            if isinstance(snr_range, int):
                snr_range = [snr_range, snr_range]
            if snr_range[1] < snr_range[0]:
                snr_range = list(reversed(snr_range))
            self.hparams.snr_range = snr_range

        self.f0_estimator = f0_estimator

        self.f0_min = self.f0_estimator.f0_classes_hz[0]
        self.n_freq = self.f0_estimator.f0_classes_hz.numel()

        f0_classes_cent = hz_to_cents(f_hz=self.f0_estimator.f0_classes_hz, f_ref=self.f0_min)
        self.f0_r_cent = int(f0_classes_cent[1])

        self.f0_selector = f0_selector(f_min=self.f0_min, f0_classes_cent=f0_classes_cent)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = MetricCollection(
            {
                "RPA": MeanMetric(),
                "RCA": MeanMetric(),
                "OA": MeanMetric(),
                "VR": MeanMetric(),
                "VFA": MeanMetric(),
            }
        )

        self.val_acc = MetricCollection(
            {
                "RPA": MeanMetric(),
                "RCA": MeanMetric(),
                "OA": MeanMetric(),
                "VR": MeanMetric(),
                "VFA": MeanMetric(),
            }
        )

        self.test_acc = MetricCollection(
            {
                "RPA": MeanMetric(),
                "RCA": MeanMetric(),
                "OA": MeanMetric(),
                "VR": MeanMetric(),
                "VFA": MeanMetric(),
            }
        )

        # for averaging loss across batches
        self.train_loss = MetricCollection(
            {
                "loss": MeanMetric(),
            }
        )

        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # cent tolerance for f0 evaluation
        self.cent_tolerance = cent_tolerance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.f0_estimator(x)
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x = batch["audio"]

        if self.hparams.snr_range is not None:
            bs = x.shape[0]
            snr_min, snr_max = self.hparams.snr_range
            snr = snr_min + torch.rand(bs) * (snr_max - snr_min)

            signal = batch["audio"]
            noise = batch["audio_noise"]
            x = mix_at_snr(signal, noise, snr_target_db=snr)

        dev = batch["label"].device
        targets_hz = batch["label"]

        # get predictions
        out = self.forward(x)

        # calculate targets
        y = (
            get_f0_targets(
                targets_hz,
                f_min=self.f0_min,
                f_max=None,
                n_freq=self.n_freq,
                cent_step=self.f0_r_cent,
                gaussian_sigma=self.hparams.target_blurring,
                return_as="vector",
            )
            .to(device=dev)
            .transpose(1, -1)
        )

        if "logits_all" in out.keys():
            # add "unvoiced" entry in targets and compute loss
            y_unvoiced = (targets_hz == 0.0).type(torch.float32)
            y_all = torch.cat((y_unvoiced.unsqueeze(dim=1), y), dim=1)
            logits_all = out["logits_all"]
            loss = F.cross_entropy(logits_all.transpose(-1, -2), y_all)
        else:
            loss = torch.as_tensor(0.0)

        # postprocessing to get continuous f0 estimates
        if "logits_f0" in out.keys():
            logits_f0 = out["logits_f0"]
        else:
            logits_f0 = out["logits"]

        probs_f0 = F.softmax(logits_f0, dim=-1)
        preds_hz = self.f0_selector(probs_f0)

        # voicing detection
        if self.hparams.unvoiced_mode == "joint" and "logits_all" in out.keys():
            logits_all = out["logits_all"]
            unvoiced_frames = torch.argmax(logits_all, dim=-1) == 0

        elif self.hparams.unvoiced_mode == "thresh_probs_softmax":
            unvoiced_frames = torch.max(probs_f0, dim=-1).values < 0.5

        elif self.hparams.unvoiced_mode == "thresh_probs_sigmoid":
            unvoiced_frames = torch.max(probs_f0, dim=-1).values < 0.5

        else:
            unvoiced_frames = torch.zeros_like(
                logits_f0[..., 0], dtype=torch.bool, device=preds_hz.device
            )

        preds_hz[unvoiced_frames] = -preds_hz[
            unvoiced_frames
        ]  # negative sign marks frames that are predicted as unvoiced

        losses = {
            "loss": loss,
        }

        return losses, preds_hz, targets_hz

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, preds, targets = self.model_step(batch)
        loss = sum(losses.values())

        # update and log metrics
        for loss_name, value in losses.items():
            self.train_loss[loss_name].update(value)
            self.log(
                f"train/loss/{loss_name}",
                self.train_loss[loss_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        train_acc = compute_metrics(preds, targets, self.cent_tolerance)

        for metric_name, value in train_acc.items():
            self.train_acc[metric_name].update(value)
            self.log(
                f"train/acc/{metric_name}",
                self.train_acc[metric_name],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_start(self) -> None:
        self.log("optimizer/lr", self.lr_schedulers().get_last_lr()[0])

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, preds, targets = self.model_step(batch)
        loss = sum(losses.values())

        # update and log metrics
        self.val_loss(loss)

        val_acc = compute_metrics(preds, targets, self.cent_tolerance)

        for metric_name, value in val_acc.items():
            self.val_acc[metric_name].update(value)
            self.log(
                f"val/acc/{metric_name}",
                self.val_acc[metric_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc["OA"].compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best/OA", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, preds, targets = self.model_step(batch)
        loss = sum(losses.values())

        # update and log metrics
        self.test_loss(loss)

        test_acc = compute_metrics(preds, targets, self.cent_tolerance)

        for metric_name, value in test_acc.items():
            self.test_acc[metric_name].update(value)
            self.log(
                f"test/acc/{metric_name}",
                self.test_acc[metric_name],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, **self.hparams.scheduler_config},
            }
        return {"optimizer": optimizer}
