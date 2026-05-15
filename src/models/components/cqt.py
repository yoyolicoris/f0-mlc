import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import warnings
from kazane import Decimate


def create_cqt_kernels(
    Q,
    fs,
    fmin,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    norm: float = 1,
    fmax=None,
    topbin_check=True,
    gamma=0,
    # pad_fft=True,
):
    """
    Automatically create CQT kernels in time domain
    """

    fftLen = 2 ** math.ceil(math.log2(math.ceil(Q * fs / fmin)))

    match (fmax is not None, n_bins is not None):
        case (True, False):
            n_bins = math.ceil(bins_per_octave * math.log2(fmax / fmin))
        case (False, True):
            n_bins = int(n_bins)
        case _:
            warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
            n_bins = math.ceil(bins_per_octave * math.log2(fmax / fmin))

    freqs = fmin * 2.0 ** (torch.arange(n_bins) / bins_per_octave)

    if topbin_check and torch.max(freqs).item() > fs / 2:
        top_bin = torch.max(freqs).item()
        raise ValueError(
            f"The top bin {top_bin}Hz has exceeded the Nyquist frequency, please reduce the n_bins"
        )

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = torch.ceil((Q * fs) / (freqs + gamma / alpha))

    # get max window length depending on gamma value
    max_len = torch.max(lengths).item()
    fftLen = int(2 ** math.ceil(math.log2(max_len)))

    # tempKernel = torch.zeros((n_bins, fftLen), dtype=torch.complex64)

    t = torch.arange(-fftLen // 2, fftLen // 2)
    tmp = []
    for k, (freq, l) in enumerate(zip(freqs, lengths)):
        # Centering the kernels
        # if l % 2 == 1:  # pad more zeros on RHS
        #     start = math.ceil(fftLen / 2.0 - l / 2.0) - 1
        # else:
        #     start = math.ceil(fftLen / 2.0 - l / 2.0)

        # window_dispatch = get_window_dispatch(window, l, fftbins=True)
        # here we use gaussian window
        window_dispatch = torch.signal.windows.gaussian(fftLen, std=l / 6.0, sym=False)
        phase = t * (2 * torch.pi * freq / fs)
        sig = window_dispatch * torch.exp(1j * phase) / l

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            sig = sig / torch.linalg.vector_norm(sig, ord=norm)

        # tempKernel[k, start : start + l] = sig
        tmp.append(sig)

    tempKernel = torch.stack(tmp, dim=0)

    return tempKernel, fftLen, lengths, freqs


# Adapted from nnAudio
class CQT2010v2(nn.Module):
    """This function is to calculate the CQT of the input signal.
    Input signal should be in either of the following shapes.\n
    1. ``(len_audio)``\n
    2. ``(num_audio, len_audio)``\n
    3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``nn.Module``, therefore, the usage is same as ``nn.Module``.

    This alogrithm uses the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the
    input audio by a factor of 2 to convoluting it with the small CQT kernel.
    Everytime the input audio is downsampled, the CQT relative to the downsampled input is equivalent
    to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the
    code from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).

    Early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.

    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    hop_length : int
        The hop (or stride) size. Default value is 512.

    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.

    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.  If ``fmax`` is not ``None``, then the
        argument ``n_bins`` will be ignored and ``n_bins`` will be calculated automatically.
        Default is ``None``

    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.

    bins_per_octave : int
        Number of bins per octave. Default is 12.

    norm : bool
        Normalization for the CQT result.

    basis_norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.

    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'

    pad_mode : str
        The padding method. Default value is 'reflect'.

    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``

    output_format : str
        Determine the return type.
        'Magnitude' will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins, time_steps)``;
        'Complex' will return the STFT result in complex number, shape = ``(num_samples, freq_bins, time_steps, 2)``;
        'Phase' will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.

    Returns
    -------
    spectrogram : torch.tensor
    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;

    Examples
    --------
    >>> spec_layer = Spectrogram.CQT2010v2()
    >>> specs = spec_layer(x)
    """

    # To DO:
    # need to deal with the filter and other tensors

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float = 32.70,
        n_bins: int = 84,
        filter_scale: float = 1.0,
        bins_per_octave: int = 12,
        basis_norm: float = 1,
        pad_mode: str = "reflect",
        num_zeros: int = 32,
    ):

        super().__init__()
        # basis_norm is for normalizing basis
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.n_bins = n_bins

        # It will be used to calculate filter_cutoff and creating CQT kernels
        Q = filter_scale / (2 ** (1 / bins_per_octave) - 1)

        # lowpass_filter = torch.tensor(
        #     create_lowpass_filter(
        #         band_center=0.50, kernelLength=256, transitionBandwidth=0.001
        #     )
        # )

        # Broadcast the tensor to the shape that fits conv1d
        # self.register_buffer("lowpass_filter", lowpass_filter[None, None, :])

        # Caluate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(bins_per_octave, n_bins)
        self.n_octaves = int(math.ceil(float(n_bins) / bins_per_octave))

        assert (
            hop_length % 2 ** (self.n_octaves - 1) == 0
        ), "The hop_length should be divisible by 2^(n_octaves-1)"

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = fmin * 2 ** (self.n_octaves - 1)
        remainder = n_bins % bins_per_octave

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((bins_per_octave - 1) / bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (
            1 - 1 / bins_per_octave
        )  # Adjusting the top minium bins
        if fmax_t > sr / 2:
            raise ValueError(
                f"The top bin {fmax_t}Hz has exceeded the Nyquist frequency, please reduce the n_bins"
            )

        # Preparing CQT kernels
        basis, self.n_fft, lengths, _ = create_cqt_kernels(
            Q,
            sr,
            self.fmin_t,
            n_filters,
            bins_per_octave,
            norm=basis_norm,
            topbin_check=False,
        )
        # For normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        freqs = fmin * 2.0 ** (torch.arange(n_bins) / bins_per_octave)
        # self.frequencies = freqs
        self.register_buffer("frequencies", freqs, persistent=False)

        lengths = torch.ceil(Q * sr / freqs)
        # lengths = torch.tensor(lengths).float()
        self.register_buffer("lengths", lengths, persistent=False)

        self.basis = basis
        self.register_buffer("cqt_kernels", basis.unsqueeze(1), persistent=False)
        # These cqt_kernel is already in the frequency domain
        # cqt_kernels_real = torch.tensor(basis.real).unsqueeze(1)
        # cqt_kernels_imag = torch.tensor(basis.imag).unsqueeze(1)

        # if trainable:
        #     cqt_kernels_real = nn.Parameter(cqt_kernels_real, requires_grad=trainable)
        #     cqt_kernels_imag = nn.Parameter(cqt_kernels_imag, requires_grad=trainable)
        #     self.register_parameter("cqt_kernels_real", cqt_kernels_real)
        #     self.register_parameter("cqt_kernels_imag", cqt_kernels_imag)
        # else:
        #     self.register_buffer("cqt_kernels_real", cqt_kernels_real)
        #     self.register_buffer("cqt_kernels_imag", cqt_kernels_imag)

        # print("Getting cqt kernel done, n_fft = ",self.n_fft)

        # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        if self.pad_mode == "constant":
            self.padding = nn.ConstantPad1d(self.n_fft // 2, 0)
        elif self.pad_mode == "reflect":
            self.padding = nn.ReflectionPad1d(self.n_fft // 2)

        self.downsampler = Decimate(q=2, num_zeros=num_zeros)

    def forward(self, x, normalization_type="librosa"):
        """
        Convert a batch of waveforms to CQT spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in the following shapes.\n
            - ``(num_audio, len_audio)``\n
        """

        hop = self.hop_length
        # CQT = get_cqt_complex(
        #     x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding
        # )  # Getting the top octave CQT
        CQT = [
            F.conv1d(self.padding(x.unsqueeze(1)) + 0j, self.cqt_kernels, stride=hop)
        ]

        x_down = x  # Preparing a new variable for downsampling

        for _ in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = self.downsampler(x_down)
            CQT1 = F.conv1d(
                self.padding(x_down.unsqueeze(1)) + 0j, self.cqt_kernels, stride=hop
            )
            CQT.append(CQT1)

        CQT = torch.cat(
            CQT[::-1], dim=1
        )  # Concatenating the CQT results from different octaves
        CQT = CQT[:, -self.n_bins :, :]  # Removing unwanted bottom bins
        # Normalize again to get same result as librosa
        if normalization_type == "librosa":
            CQT = CQT * self.lengths.view(-1, 1).sqrt()
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            CQT *= 2
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        return CQT
