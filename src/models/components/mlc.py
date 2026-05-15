import torch
import torch.nn.functional as F

# import nnAudio.features
from torch_fftconv.modules import FFTConv1d
from scipy.signal.windows import blackmanharris

from df0.f0_utils import get_log_frequencies

from .vqt import VQT


class MultiLayerCepstrumModule(torch.nn.Module):
    """Short-time cepstrum-based F0 estimation module.

    Extracts pitch (F0) predictions by computing the short-time cepstrum of an audio signal
    and performing parabolic interpolation. Returns probability distributions over a discrete
    set of F0 classes defined logarithmically by the input parameters.
    """

    def __init__(
        self,
        fs,
        frame_size,
        hop_size,
        f0_min,
        f0_max,
        f0_r_cent,
        gammas=(0.2, 0.9, 0.8),
    ):
        super().__init__()

        self.fs = fs
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_r_cent = f0_r_cent
        self.hipass_f = f0_min
        self.lowpass_t = 1 / f0_max * 1000

        self.hpi = int(self.hipass_f * frame_size / self.fs)
        self.lpi = int(self.lowpass_t * self.fs / 1000)

        self.f0_classes_hz = get_log_frequencies(
            f_min=f0_min, f_max=f0_max, cent_step=f0_r_cent, return_as="hz"
        )

        quefrencies = self.fs / self.f0_classes_hz
        linfrequencies = self.f0_classes_hz / self.fs * frame_size

        self.register_buffer("quefrencies_rounded", torch.round(quefrencies).long())
        self.register_buffer(
            "quefrencies_difference", quefrencies - self.quefrencies_rounded
        )
        self.register_buffer(
            "linfrequencies_rounded", torch.round(linfrequencies).long()
        )
        self.register_buffer(
            "linfrequencies_difference", linfrequencies - self.linfrequencies_rounded
        )
        self.register_buffer(
            "blackman_window",
            torch.from_numpy(blackmanharris(self.frame_size)).float(),
            #  torch.blackman_window(self.frame_size)
        )
        sigmoid_inv = lambda x: torch.log(x / (1 - x))
        self.gamma_logits = torch.nn.Parameter(
            sigmoid_inv(torch.tensor(gammas, dtype=torch.float32))
        )

        self.vqt = VQT(
            sr=fs,
            hop_length=hop_size,
            fmin=f0_min,
            # fmax=f0_max,
            n_bins=self.f0_classes_hz.numel(),
            bins_per_octave=int(1200 / f0_r_cent),
            gamma=5,
            num_zeros=128,
        )

        num_ceps = len(gammas) // 2
        num_spec = len(gammas) - num_ceps + 1

        n_input_features = num_ceps + num_spec + num_ceps * num_spec
        self.n_freq = self.f0_classes_hz.numel()

        self.input_instance_norm = torch.nn.InstanceNorm2d(
            n_input_features, affine=True
        )

        # conv only (toeplitz-like)
        self.conv = torch.nn.Conv1d(
            in_channels=n_input_features,
            out_channels=1,
            kernel_size=int(2 * self.n_freq - 1),
            padding=self.n_freq - 1,
        )

        self.fc = torch.nn.Linear(
            in_features=3 * n_input_features,
            out_features=1,
        )

    # @torch.compile(fullgraph=True, dynamic=True)
    def forward(self, x):
        """Extract F0 class probabilities from audio.

        Args:
            x: Audio waveform, shape (batch_size, num_samples)

        Returns:
            Dictionary with:
                - logits: F0 class logits, shape (batch_size, num_frames, num_f0_classes)
                - probs: F0 class probabilities, shape (batch_size, num_frames, num_f0_classes)
        """
        gammas = torch.sigmoid(self.gamma_logits)
        spec = (
            torch.stft(
                x,
                n_fft=self.frame_size,
                hop_length=self.hop_size,
                window=self.blackman_window,
                center=True,
                return_complex=True,
                onesided=False,
                normalized=True,
            )
            .abs()
            .mT
            ** gammas[0]
        )
        ceps = []
        y = spec
        spec = [spec]
        for i, gamma in enumerate(gammas[1:]):
            y = torch.fft.fft(y, dim=-1, norm="ortho").real
            if i % 2 == 0:
                y[..., : self.lpi + 1] = 0
                y[..., -self.lpi :] = 0
                y = y.relu() ** gamma
                ceps.append(y)
            else:
                y[..., : self.hpi + 1] = 0
                y[..., -self.hpi :] = 0
                y = y.relu() ** gamma
                spec.append(y)

        # x_ceps = torch.swapaxes(ceps, -1, -2)
        # x_spec = torch.swapaxes(spec, -1, -2)
        x_ceps = torch.stack(ceps, dim=1)
        x_spec = torch.stack(spec, dim=1)

        # parabolic interpolation
        # x_ceps = torch.cat([x_ceps, x_ceps[..., [-1]]], dim=-1)

        a = 0.5 * x_ceps[..., :-2] - x_ceps[..., 1:-1] + 0.5 * x_ceps[..., 2:]
        b = -0.5 * x_ceps[..., :-2] + 0.5 * x_ceps[..., 2:]
        c = x_ceps

        ceps_logits = (
            a[..., self.quefrencies_rounded] * self.quefrencies_difference**2
            + b[..., self.quefrencies_rounded] * self.quefrencies_difference
            + c[..., self.quefrencies_rounded]
        ).relu()

        a = 0.5 * x_spec[..., :-2] - x_spec[..., 1:-1] + 0.5 * x_spec[..., 2:]
        b = -0.5 * x_spec[..., :-2] + 0.5 * x_spec[..., 2:]
        c = x_spec

        spec_logits = (
            a[..., self.linfrequencies_rounded] * self.linfrequencies_difference**2
            + b[..., self.linfrequencies_rounded] * self.linfrequencies_difference
            + c[..., self.linfrequencies_rounded]
        ).relu()

        vqt_spec = self.vqt(x).mT.mul(10).log1p().unsqueeze(1)

        x_features = torch.cat(
            [
                ceps_logits,
                spec_logits,
                vqt_spec,
                (ceps_logits.unsqueeze(2) * spec_logits.unsqueeze(1)).flatten(1, 2),
                vqt_spec * ceps_logits,
            ],
            dim=1,
        )  # (batch_size, num_reps, num_frames, num_f0_classes)

        x_features_norm = self.input_instance_norm(x_features)

        # ----- F0 logit computation -----

        # Convolve over F0 frequency dimension (Toeplitz-like structure)
        # logits_f0: (batch_size, num_frames, num_f0_classes)
        bs, c, t, f = x_features_norm.shape
        # logits_f0 = self.conv(
        #     x_features_norm.transpose(2, 1).flatten(0, 1)
        # )  # (batch_size*num_frames, 1, num_f0_classes)
        # logits_f0 = logits_f0.unflatten(0, (bs, t)).squeeze(2)
        # logits_f0 = logits_f0.squeeze(dim=1)

        X = torch.fft.rfft(x_features_norm, n=int(2 * self.n_freq - 1), dim=-1)
        W = torch.fft.rfft(self.conv.weight.transpose(0, 1), dim=-1)
        # Y = X.conj() * W
        Y = torch.linalg.vecdot(X, W, dim=1)
        logits_f0 = torch.fft.irfft(Y, n=int(2 * self.n_freq - 1), dim=-1)[
            ..., : self.n_freq
        ].flip(-1)
        # print(logits_f0.shape)

        # ----- Voicing logit computation -----

        # Voicing features: max, entropy, and variance across F0 classes
        # x_features_max/ent/var: (batch_size, num_reps, num_frames)
        x_features_max = x_features_norm.max(dim=-1).values
        x_features_log_probs = F.log_softmax(x_features_norm, dim=-1)
        x_features_ent = -(x_features_log_probs.exp() * x_features_log_probs).sum(
            dim=-1
        )
        x_features_var = torch.var(x_features_norm, dim=-1)

        # x_features_unv: (batch_size, num_frames, 3*num_reps)
        x_features_unv = torch.cat(
            [x_features_max, x_features_ent, x_features_var], dim=1
        ).mT

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
            "f0_classes_hz": self.f0_classes_hz,  # (num_f0_classes,)
        }

        return out


if __name__ == "__main__":
    x = torch.rand(8, 160000)

    model = CepstrumModule(
        fs=16000,
        frame_size=1024,
        hop_size=320,
        f0_min=32.7,
        f0_max=3520.0,
        f0_r_cent=10,
    )

    out = model(x)
    print(out["logits"].shape)
