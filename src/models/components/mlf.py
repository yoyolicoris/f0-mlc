import torch
import torch.nn.functional as F

from df0.dyin import dYIN
from df0.dswipe import dSWIPE
from df0.f0_utils import get_log_frequencies

from src.models.components.cepstrum import CepstrumModule
from src.models.components.vqt_wrapper import VQTModule


class MLF(torch.nn.Module):
    """Multi-representation Learnable F0 estimator.

    Combines multiple F0 estimation methods (dYIN, dSWIPE, cepstrum, VQT) and learns
    to fuse their predictions into a single F0 estimate with voicing detection.
    """

    def __init__(
        self,
        fs=16000,
        f0_min=32.7,
        f0_max=3520.0,
        f0_r_cent=10,
        hop_size=320,
        input_rep_overrides=None,
    ):
        super().__init__()

        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_r_cent = f0_r_cent

        self.f0_classes_hz = get_log_frequencies(
            f_min=f0_min,
            f_max=f0_max,
            cent_step=f0_r_cent,
        )

        self.n_freq = self.f0_classes_hz.numel()

        self.hop_size = hop_size
        self.bins_per_octave = 1200 / f0_r_cent

        self.f0_classes_cent = get_log_frequencies(
            f_min=f0_min,
            f_max=f0_max,
            cent_step=f0_r_cent,
            return_as="cent",
        )

        # Default input representation configurations
        default_configs = {
            "dyin": {"frame_size": 1600},
            "dswipe": {},
            "cepstrum": {"frame_size": 1024},
            "vqt": {
                "log_comp_gamma": 10,
                "filter_scale": 1,
                "norm": True,
                "basis_norm": 1,
                "gamma": 5,
                "window": "hann",
                "pad_mode": "reflect",
                "earlydownsample": True,
                "trainable": False,
                "output_format": "Magnitude",
                "verbose": False,
            },
        }

        # Apply overrides if provided
        if input_rep_overrides:
            for key, vals in input_rep_overrides.items():
                if key not in default_configs:
                    raise ValueError(
                        f"Unknown input representation: {key}. "
                        f"Valid options: {list(default_configs.keys())}"
                    )
                default_configs[key].update(vals)

        # Instantiate input feature extractors
        self.input_reps = torch.nn.ModuleDict(
            {
                "dyin": dYIN(
                    fs=fs,
                    hop_size=hop_size,
                    f0_min=f0_min,
                    f0_max=f0_max,
                    f0_r_cent=f0_r_cent,
                    **default_configs["dyin"],
                ),
                "dswipe": dSWIPE(
                    fs=fs,
                    hop_size=hop_size,
                    f0_min=f0_min,
                    f0_max=f0_max,
                    f0_r_cent=f0_r_cent,
                    **default_configs["dswipe"],
                ),
                "cepstrum": CepstrumModule(
                    fs=fs,
                    hop_size=hop_size,
                    f0_min=f0_min,
                    f0_max=f0_max,
                    f0_r_cent=f0_r_cent,
                    **default_configs["cepstrum"],
                ),
                "vqt": VQTModule(
                    fs=fs,
                    hop_size=hop_size,
                    f0_min=f0_min,
                    f0_max=f0_max,
                    f0_r_cent=f0_r_cent,
                    **default_configs["vqt"],
                ),
            }
        )

        n_input_features = len(self.input_reps)

        self.input_instance_norm = torch.nn.InstanceNorm2d(
            n_input_features, affine=True
        )

        # conv only (toeplitz-like)
        self.conv = torch.nn.Conv1d(
            in_channels=n_input_features,
            out_channels=1,
            kernel_size=int(2 * self.n_freq - 1),
            padding="same",
        )

        self.fc = torch.nn.Linear(
            in_features=3 * n_input_features,
            out_features=1,
        )

    @torch.compile(fullgraph=False)
    def forward(self, x):
        """Extract F0 predictions by fusing multiple representations.

        Args:
            x: Audio waveform, shape (batch_size, num_samples)

        Returns:
            Dictionary with:
                - logits_f0: F0 class logits, shape (batch_size, num_frames, num_f0_classes)
                - logits_unv: Voicing logits, shape (batch_size, num_frames, 1)
                - logits_all: Combined F0+voicing logits, shape (batch_size, num_frames, 1+num_f0_classes)
                - probs_all: Joint F0+voicing probabilities, shape (batch_size, num_frames, 1+num_f0_classes)
        """
        n_frames = (x.shape[-1] - 1) // self.hop_size + 1

        # ----- Feature extraction and normalization -----

        # Stack logits from all input representations
        x_features = [
            feat(x)["logits"][:, :n_frames, :] for feat in self.input_reps.values()
        ]
        x_features = torch.stack(
            x_features, dim=1
        )  # (batch_size, num_reps, num_frames, num_f0_classes)

        x_features_norm = self.input_instance_norm(x_features)

        # ----- F0 logit computation -----

        # Convolve over F0 frequency dimension (Toeplitz-like structure)
        # logits_f0: (batch_size, num_frames, num_f0_classes)
        bs, c, t, f = x_features_norm.shape
        logits_f0 = self.conv(x_features_norm.permute(0, 2, 1, 3).reshape(bs * t, c, f))
        logits_f0 = logits_f0.view(bs, t, 1, f).permute(0, 2, 1, 3)

        logits_f0 = logits_f0.squeeze(dim=1)

        # ----- Voicing logit computation -----

        # Voicing features: max, entropy, and variance across F0 classes
        # x_features_max/ent/var: (batch_size, num_reps, num_frames)
        x_features_max = x_features_norm.max(dim=-1).values
        x_features_probs = F.softmax(x_features_norm, dim=-1)
        x_features_ent = -(
            x_features_probs
            * torch.log(x_features_probs + torch.finfo(torch.float32).tiny)
        ).sum(dim=-1)
        x_features_var = torch.var(x_features_norm, dim=-1)

        # x_features_unv: (batch_size, num_frames, 3*num_reps)
        x_features_unv = torch.cat(
            [x_features_max, x_features_ent, x_features_var], dim=1
        ).permute(0, 2, 1)

        # logits_unv: (batch_size, num_frames, 1)
        logits_unv = self.fc(x_features_unv)

        # ----- Combine F0 and voicing logits -----

        logits_all = torch.cat(
            [logits_unv, logits_f0], dim=-1
        )  # (batch_size, num_frames, 1 + num_f0_classes)

        probs_all = F.softmax(logits_all, dim=-1)

        out = {
            "logits_f0": logits_f0,  # (batch_size, num_frames, num_f0_classes)
            "logits_unv": logits_unv,  # (batch_size, num_frames, 1)
            "logits_all": logits_all,  # (batch_size, num_frames, 1 + num_f0_classes)
            "probs_all": probs_all,  # (batch_size, num_frames, 1 + num_f0_classes)
        }

        return out


if __name__ == "__main__":
    x = torch.rand(8, 160000)

    model = MLF(
        fs=16000,
        f0_min=32.7,
        f0_max=3520.0,
        f0_r_cent=10,
        hop_size=320,
    )

    out = model(x)

    print("Output shapes:")
    for key, val in out.items():
        print(f"  {key}: {val.shape}")
