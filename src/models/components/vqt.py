import torch
import torch.nn as nn
import math
from kazane import Decimate
from torch_fftconv import fft_conv1d

from .cqt import create_cqt_kernels


class VQT(torch.nn.Module):
    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.70,
        n_bins=84,
        filter_scale=1,
        bins_per_octave=12,
        norm=True,
        basis_norm=1,
        gamma=0,
        pad_mode="reflect",
        output_format="Magnitude",
        verbose=True,
        num_zeros=32,
    ):

        super().__init__()

        self.norm = norm
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins
        self.output_format = output_format
        self.filter_scale = filter_scale
        self.bins_per_octave = bins_per_octave
        self.sr = sr
        self.gamma = gamma
        self.basis_norm = basis_norm

        # It will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        n_filters = min(bins_per_octave, n_bins)
        self.n_filters = n_filters
        self.n_octaves = int(math.ceil(n_bins / bins_per_octave))
        if verbose == True:
            print("num_octave = ", self.n_octaves)

        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / bins_per_octave)

        # Adjusting the top minimum bins
        self.fmin_t = fmax_t / 2 ** (1 - 1 / bins_per_octave)
        if fmax_t > sr / 2:
            raise ValueError("The top bin {}Hz has exceeded the Nyquist frequency, \
                            please reduce the n_bins".format(fmax_t))

        # For normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
        freqs = fmin * 2.0 ** (torch.arange(n_bins) / bins_per_octave)
        # self.frequencies = freqs
        lengths = torch.ceil(Q * sr / (freqs + gamma / alpha))

        # get max window length depending on gamma value
        # max_len = int(max(lengths.tolist()))
        # self.n_fft = int(2 ** (math.ceil(math.log2(max_len))))
        self.register_buffer("lengths", lengths)

        my_sr = self.sr
        for i in range(self.n_octaves):
            if i > 0:
                my_sr /= 2

            Q = float(self.filter_scale) / (2 ** (1 / self.bins_per_octave) - 1)

            basis, _, _, _ = create_cqt_kernels(
                Q,
                my_sr,
                self.fmin_t * 2**-i,
                self.n_filters,
                self.bins_per_octave,
                norm=self.basis_norm,
                topbin_check=False,
                gamma=self.gamma,
            )

            # cqt_kernels_real = torch.tensor(basis.real.astype(np.float32)).unsqueeze(1)
            # cqt_kernels_imag = torch.tensor(basis.imag.astype(np.float32)).unsqueeze(1)

            # self.register_buffer("cqt_kernels_real_{}".format(i), cqt_kernels_real)
            # self.register_buffer("cqt_kernels_imag_{}".format(i), cqt_kernels_imag)
            self.register_buffer(
                f"cqt_kernels_{i}",
                torch.view_as_real(basis).permute(2, 0, 1).flatten(0, 1).unsqueeze(1),
            )

        self.downsampler = Decimate(q=2, num_zeros=num_zeros)

    def forward(self, x, output_format=None, normalization_type="librosa"):
        """
        Convert a batch of waveforms to VQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        output_format = output_format or self.output_format
        hop = self.hop_length
        vqt = []

        x_down = x  # Preparing a new variable for downsampling
        my_sr = self.sr

        for i in range(self.n_octaves):
            if i > 0:
                # x_down = downsampling_by_2(x_down, self.lowpass_filter)
                x_down = self.downsampler(x_down)
                hop //= 2

            else:
                x_down = x

            cqt_kernels = getattr(self, f"cqt_kernels_{i}")
            pad_length = int(cqt_kernels.shape[-1] // 2)
            # if self.pad_mode == "constant":
            #     my_padding = nn.ConstantPad1d((pad_length, pad_length), 0)
            # elif self.pad_mode == "reflect":
            #     my_padding = nn.ReflectionPad1d((pad_length, pad_length))
            x_down_padded = nn.functional.pad(
                x_down, (pad_length, pad_length), mode=self.pad_mode
            )

            # cur_vqt = get_cqt_complex(
            #     x_down,
            #     getattr(self, "cqt_kernels_real_{}".format(i)),
            #     getattr(self, "cqt_kernels_imag_{}".format(i)),
            #     hop,
            #     my_padding,
            # )
            cur_vqt = nn.functional.conv1d(
                x_down_padded.unsqueeze(1), cqt_kernels, stride=hop
            )
            vqt.append(cur_vqt)

            # vqt.insert(0, cur_vqt)

        # print('vqt shape: ', vqt[0].shape)

        # PREVENT SIZE ALIGMENT ERROR
        # padding smaller frames, if the hop_length is not power of 2
        # max_time_axis_frame = max(vqt_i.shape[-2] for vqt_i in vqt)
        min_time_axis_frame = min(vqt_i.shape[-1] for vqt_i in vqt)
        # print('max_time_axis_frame: ', max_time_axis_frame)
        # print('min_time_axis_frame: ', min_time_axis_frame)
        # vqt = [nn.functional.pad(vqt_i, (0, 0, max_time_axis_frame - vqt_i.shape[-2], 0), mode='constant', value=0) for vqt_i in vqt]
        vqt = [
            (
                # nn.functional.interpolate(
                #     vqt_i,
                #     size=min_time_axis_frame,
                #     mode="linear",
                #     align_corners=False,
                # )
                # if vqt_i.shape[-1] > min_time_axis_frame
                # else vqt_i
                vqt_i[..., :min_time_axis_frame]
            )
            .unflatten(1, (2, -1))
            .permute(0, 2, 3, 1)
            for vqt_i in reversed(vqt)
        ]

        # for vqt_i in vqt:
        #     print(vqt_i.shape)

        # vqt = torch.view_as_complex(torch.cat(vqt, dim=1))
        vqt = torch.cat(vqt, dim=1)
        vqt = vqt[:, -self.n_bins :, ...]  # Removing unwanted bottom bins
        # vqt = vqt * self.downsample_factor

        # Normalize again to get same result as librosa
        if normalization_type == "librosa":
            vqt = vqt * torch.sqrt(self.lengths.view(-1, 1, 1))
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            vqt *= 2
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        if output_format == "Magnitude":
            return vqt.square().sum(dim=-1).sqrt()

        elif output_format == "Complex":
            return vqt
        else:
            raise ValueError(
                "The output_format %r is not part of our current options."
                % output_format
            )
