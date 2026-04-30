import torch
import torch.nn.functional as F

from df0.f0_utils import get_log_frequencies


class CepstrumModule(torch.nn.Module):
    """Short-time cepstrum-based F0 estimation module.

    Extracts pitch (F0) predictions by computing the short-time cepstrum of an audio signal
    and performing parabolic interpolation. Returns probability distributions over a discrete
    set of F0 classes defined logarithmically by the input parameters.
    """

    def __init__(self, fs, frame_size, hop_size, f0_min, f0_max, f0_r_cent):
        super().__init__()

        self.fs = fs
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_r_cent = f0_r_cent

        self.f0_classes_hz = get_log_frequencies(
            f_min=f0_min, f_max=f0_max, cent_step=f0_r_cent, return_as="hz"
        )

        quefrencies = self.fs / self.f0_classes_hz

        self.register_buffer("quefrencies_rounded", torch.round(quefrencies).type(torch.long))
        self.register_buffer("quefrencies_difference", quefrencies - self.quefrencies_rounded)
        self.register_buffer("hann_window", torch.hann_window(self.frame_size))

    def forward(self, x):
        """Extract F0 class probabilities from audio.

        Args:
            x: Audio waveform, shape (batch_size, num_samples)

        Returns:
            Dictionary with:
                - logits: F0 class logits, shape (batch_size, num_frames, num_f0_classes)
                - probs: F0 class probabilities, shape (batch_size, num_frames, num_f0_classes)
        """
        x_stft = torch.stft(
            x,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            window=self.hann_window,
            center=True,
            return_complex=True,
            onesided=True,
        )

        x_cep = torch.fft.irfft(
            torch.log(torch.abs(x_stft).clamp(min=torch.finfo(torch.float32).eps)),
            n=self.frame_size,
            dim=-2,
        )

        x_cep = torch.swapaxes(x_cep, -1, -2)

        # parabolic interpolation
        x_cep_padded = torch.cat([x_cep, x_cep[..., [-1]]], dim=-1)

        a = 0.5 * x_cep_padded[..., :-2] - x_cep_padded[..., 1:-1] + 0.5 * x_cep_padded[..., 2:]
        b = -0.5 * x_cep_padded[..., :-2] + 0.5 * x_cep_padded[..., 2:]
        c = x_cep_padded

        logits = (
            a[..., self.quefrencies_rounded] * self.quefrencies_difference**2
            + b[..., self.quefrencies_rounded] * self.quefrencies_difference
            + c[..., self.quefrencies_rounded]
        )

        out = {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
        }

        return out


if __name__ == "__main__":
    x = torch.rand(8, 160000)

    model = CepstrumModule(
        fs=16000, frame_size=1024, hop_size=320, f0_min=32.7, f0_max=3520.0, f0_r_cent=10
    )

    out = model(x)
    print(out["logits"].shape)
