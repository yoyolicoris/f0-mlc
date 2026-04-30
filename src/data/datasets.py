"""Simplified F0 audio dataset classes.

Three main use cases:
1. MIR-1K: synced two-channel audio (vocal + accompaniment)
2. Vocadito: clean F0 audio
3. Noisex92: environmental noise source (paired with Vocadito)

Example usage:
    # MIR-1K with built-in noise
    mir1k = MIR1K(path="/data/MIR-1K", sequence_n_frames=200)

    # Vocadito with independent noise
    vocadito = Vocadito(path="/data/vocadito", sequence_n_frames=200)
    noisex = Noisex92(path="/data/NOISEX92", sequence_n_frames=200)
    vocadito_noisy = NoisyF0Dataset(vocadito, noisex)
"""

import os
from glob import glob
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from mir_eval.melody import freq_to_voicing, resample_melody_series
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Base F0 dataset
# ---------------------------------------------------------------------------


class F0AudioDataset(Dataset):
    """Base class for F0-annotated audio datasets.

    Always returns randomly selected segments of ``sequence_n_frames`` length.
    Resamples F0 annotations to match the target hop length if needed.

    Args:
        path: Root directory of the dataset.
        dir_wav: Subdirectory under ``path`` containing WAV files.
        dir_csv: Subdirectory under ``path`` containing CSV annotation files.
        sample_rate: Expected sample rate in Hz.
        hop_size: Samples between consecutive F0 annotations.
        sequence_n_frames: Number of frames returned per segment.
        subset: Name of subset CSV under ``path/subsets/``, or ``"all"``.
        seed: Random seed for segment selection.
    """

    def __init__(
        self,
        path: str,
        dir_wav: str,
        dir_csv: str,
        sample_rate: int,
        hop_size: int,
        sequence_n_frames: int,
        subset: str | None = None,
        seed: int = 17,
    ):
        self.path = path
        self.dir_wav = dir_wav
        self.dir_csv = dir_csv
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.sequence_n_frames = sequence_n_frames
        self.random = np.random.RandomState(seed)

        if subset is not None:
            files_subset = np.loadtxt(
                os.path.join("data", "splits", f"{subset}.txt"), dtype="str"
            ).tolist()

        self.data = []
        for audio_path, csv_path in self.files():
            if subset is None or os.path.basename(audio_path) in files_subset:
                self.data.append(self.load(audio_path, csv_path))

    def files(self) -> List[Tuple[str, str]]:
        """Return a sorted list of (audio_path, csv_path) pairs."""
        wavs = sorted(glob(os.path.join(self.path, self.dir_wav, "*.wav")))
        csvs = [
            os.path.join(self.path, self.dir_csv, os.path.basename(w).replace(".wav", ".csv"))
            for w in wavs
        ]
        assert all(os.path.isfile(w) for w in wavs)
        assert all(os.path.isfile(c) for c in csvs)
        return list(zip(wavs, csvs))

    def resample_f0_trajectory(self, data_csv: np.ndarray, audio: torch.Tensor) -> np.ndarray:
        """Resample annotation timestamps to match the target hop length."""
        t_old = data_csv[:, 0]
        hop_new_sec = self.hop_size / self.sample_rate

        n_frames = (len(audio) - 1) // self.hop_size + 1
        t_new = np.arange(n_frames) * hop_new_sec
        t_new = t_new[np.logical_and(t_new >= t_old[0], t_new < t_old[-1])]

        frequencies_old, voicing_old = freq_to_voicing(data_csv[:, 1])
        frequencies_new, _ = resample_melody_series(
            t_old, frequencies_old, voicing_old, t_new, kind="linear"
        )

        return np.stack((t_new, frequencies_new), axis=1)

    def load(self, audio_path: str, csv_path: str) -> dict:
        """Load audio and corresponding F0 annotation, trim to annotated region, pad if necessary.

        Returns audio in float32 and ensures it's long enough for sequence_n_frames.
        """
        audio, sr = librosa.load(
            path=audio_path, sr=self.sample_rate, mono=False, res_type="soxr_hq"
        )
        audio = torch.from_numpy(audio).float()
        if audio.ndim > 1:
            audio = audio.T  # librosa: (channels, samples) -> (samples, channels)

        data_csv = np.loadtxt(csv_path, delimiter=",", skiprows=0)
        data_csv = self.resample_f0_trajectory(data_csv, audio)

        t = torch.tensor(data_csv[:, 0], dtype=torch.float32)
        label = torch.tensor(data_csv[:, 1], dtype=torch.float32)

        # Always trim to annotated region
        start = int(t[0].item() * self.sample_rate)
        end = start + self.hop_size * (len(t) - 1) + 1

        assert start >= 0, "Annotation starts before audio"
        assert end <= audio.shape[0], "Annotation extends beyond audio"

        audio = audio[start:end]

        if self.sequence_n_frames is not None:
            # Pad if necessary to ensure we can extract sequence_n_frames
            n_samples_needed = (self.sequence_n_frames - 1) * self.hop_size + 1
            if n_samples_needed > audio.shape[0]:
                n_pad = n_samples_needed - audio.shape[0]
                pad_shape = (n_pad, *audio.shape[1:]) if audio.ndim > 1 else (n_pad,)
                silence = torch.rand(pad_shape) * torch.finfo(torch.float32).eps
                audio = torch.cat((audio, silence), dim=0)

                # Pad label to sequence_n_frames
                n_steps_pad = self.sequence_n_frames - len(label)
                label = F.pad(label, (0, n_steps_pad))

        return {"path": audio_path, "audio": audio, "label": label}

    def __getitem__(self, index: int, step_begin: Optional[int] = None) -> dict:
        """Return a segment of ``sequence_n_frames`` frames.

        Args:
            index: Index into the dataset.
            step_begin: Start frame; randomly chosen if None.

        Returns:
            dict with:
                audio:      float32 tensor, shape (n_samples,) or (n_samples, channels)
                label:      F0 in Hz, shape (sequence_n_frames,)
                path:       source audio file path
                step_begin: start frame index
        """
        data = self.data[index]
        audio = data["audio"]  # float32, shape (N,) or (N, C) for stereo
        label = data["label"]  # already padded to sequence_n_frames if needed

        if self.sequence_n_frames is None:
            audio_out = audio
            label_out = label
            step_begin = 0
        else:
            n_samples_needed = (self.sequence_n_frames - 1) * self.hop_size + 1

            if audio.shape[0] == n_samples_needed:
                # Audio was padded in load(), return as-is
                audio_out = audio
                label_out = label[: self.sequence_n_frames]
                step_begin = 0
            else:
                # Audio is longer than needed, randomly sample
                if step_begin is None:
                    max_start = label.numel() - self.sequence_n_frames
                    step_begin = self.random.randint(max_start) if max_start > 0 else 0
                step_end = step_begin + self.sequence_n_frames
                begin = int(step_begin * self.hop_size)
                end = int((step_end - 1) * self.hop_size) + 1
                audio_out = audio[begin:end]
                label_out = label[step_begin:step_end]

        return {
            "path": data["path"],
            "audio": audio_out,
            "label": label_out,
            "step_begin": step_begin,
        }

    def __len__(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# MIR-1K (stereo: vocal + accompaniment)
# ---------------------------------------------------------------------------


class MIR1K(F0AudioDataset):
    """MIR-1K dataset: stereo WAV with vocal (ch1) and accompaniment (ch0).

    Returns ``audio`` (vocal signal) and ``audio_noise`` (accompaniment) from
    the same time-aligned segment.
    """

    signal_channel = 1
    noise_channel = 0

    def __getitem__(self, index: int, step_begin: Optional[int] = None) -> dict:
        item = super().__getitem__(index, step_begin=step_begin)
        # item["audio"] is (n_samples, 2) for stereo input
        item["audio_noise"] = item["audio"][:, self.noise_channel].clone()
        item["audio"] = item["audio"][:, self.signal_channel].clone()
        return item


# ---------------------------------------------------------------------------
# Noise dataset
# ---------------------------------------------------------------------------


class NoiseDataset(Dataset):
    """Unlabeled noise audio dataset for use with ``NoisyF0Dataset``.

    Args:
        path:        Root directory containing WAV files.
        audio_dir:   Subdirectory under ``path`` to search for WAV files.
        sample_rate: Expected sample rate in Hz.
        seed:        Random seed for segment selection.
    """

    def __init__(
        self,
        path: str,
        audio_dir: str = "audio_16000",
        sample_rate: int = 16000,
        seed: int = 17,
    ):
        self.path = path
        self.sample_rate = sample_rate
        self.random = np.random.RandomState(seed)
        self.data = [self.load(f) for f in sorted(glob(os.path.join(path, audio_dir, "*.wav")))]

    def load(self, path: str) -> dict:
        audio, _ = librosa.load(path=path, sr=self.sample_rate, mono=False, res_type="soxr_hq")
        return {"path": path, "audio": torch.from_numpy(audio).float()}

    def __getitem__(self, index: int, n_samples: int) -> dict:
        """Return a noise segment of exactly ``n_samples`` samples."""
        data = self.data[index]
        audio = data["audio"]

        assert audio.shape[0] >= n_samples, (
            f"Noise file too short: {data['path']} "
            f"({audio.shape[0]} samples, need {n_samples})"
        )
        begin = self.random.randint(audio.shape[0] - n_samples + 1)
        return {"path": data["path"], "audio": audio[begin : begin + n_samples]}

    def __len__(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# Type 2: Independent noise wrapper
# ---------------------------------------------------------------------------


class NoisyF0Dataset(Dataset):
    """Wraps an F0 dataset and a noise dataset, pairing them independently.

    For each F0 item, a random noise segment of matching length is attached as
    ``audio_noise``. No mixing is performed.

    Args:
        f0_dataset:    Any F0AudioDataset instance.
        noise_dataset: Any NoiseDataset instance.
        seed:          Random seed for noise index selection.
    """

    def __init__(self, f0_dataset: F0AudioDataset, noise_dataset: NoiseDataset):
        self.f0_dataset = f0_dataset
        self.noise_dataset = noise_dataset

    def __len__(self) -> int:
        return len(self.f0_dataset) * len(self.noise_dataset)

    def __getitem__(self, index: int) -> dict:
        f0_idx = index % len(self.f0_dataset)
        noise_idx = index // len(self.f0_dataset)
        item = self.f0_dataset[f0_idx]
        n_samples = item["audio"].shape[0]
        noise_item = self.noise_dataset.__getitem__(noise_idx, n_samples)
        item["audio_noise"] = noise_item["audio"]
        return item
