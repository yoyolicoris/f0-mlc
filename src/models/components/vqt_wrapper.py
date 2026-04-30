import torch
import torch.nn.functional as F
import nnAudio.features

from df0.f0_utils import get_log_frequencies


class VQTModule(torch.nn.Module):
    """Variable-Q Transform (VQT) based F0 estimation module.

    Extracts pitch (F0) predictions using a variable-Q transform with a logarithmic
    frequency scale. Optionally applies log compression to the magnitude spectrogram.
    Returns logits and probabilities over a discrete set of F0 classes.
    """

    def __init__(self, log_comp_gamma=10.0, **kwargs):
        super().__init__()

        bins_per_octave = 1200 / kwargs["f0_r_cent"]

        self.f0_classes_hz = get_log_frequencies(
            f_min=kwargs["f0_min"],
            f_max=kwargs["f0_max"],
            cent_step=kwargs["f0_r_cent"],
            return_as="hz",
        )

        self.vqt = nnAudio.features.VQT(
            sr=kwargs["fs"],
            hop_length=kwargs["hop_size"],
            fmin=kwargs["f0_min"],
            fmax=kwargs["f0_max"],
            n_bins=self.f0_classes_hz.numel(),
            filter_scale=kwargs["filter_scale"],
            bins_per_octave=bins_per_octave,
            norm=kwargs["norm"],
            basis_norm=kwargs["basis_norm"],
            gamma=kwargs["gamma"],
            window=kwargs["window"],
            pad_mode=kwargs["pad_mode"],
            earlydownsample=kwargs["earlydownsample"],
            trainable=kwargs["trainable"],
            output_format=kwargs["output_format"],
            verbose=kwargs["verbose"],
        )

        self.log_comp_gamma = log_comp_gamma

    def forward(self, x):
        """Extract F0 class probabilities from audio using VQT.

        Args:
            x: Audio waveform, shape (batch_size, num_samples)

        Returns:
            Dictionary with:
                - logits: F0 class logits, shape (batch_size, num_frames, num_f0_classes)
                - probs: F0 class probabilities, shape (batch_size, num_frames, num_f0_classes)
        """
        x_vqt = self.vqt(x)

        if self.log_comp_gamma is not None:
            logits = torch.log(1 + self.log_comp_gamma * x_vqt.permute(0, 2, 1))
        else:
            logits = x_vqt.permute(0, 2, 1)

        out = {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
        }

        return out


if __name__ == "__main__":
    x = torch.rand(8, 160000)

    model = VQTModule(
        log_comp_gamma=10.0,
        fs=16000,
        hop_size=320,
        f0_min=32.7,
        f0_max=3520.0,
        f0_r_cent=10,
        filter_scale=1,
        norm=True,
        basis_norm=1,
        gamma=5,
        window="hann",
        pad_mode="reflect",
        earlydownsample=True,
        trainable=False,
        output_format="Magnitude",
        verbose=False,
    )

    out = model(x)
    print(out["logits"].shape)
