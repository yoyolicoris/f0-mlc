"""PyTorch implementation of mir_eval melody metrics.

This module contains functions **copied and modified from mir_eval**
(https://github.com/craffel/mir_eval) to enable GPU-compatible batch computation.
Original implementations used NumPy arrays; this version converts them to PyTorch
tensors for efficient evaluation on GPU.

Converted metrics:
- raw_pitch_accuracy (RPA)
- raw_chroma_accuracy (RCA)
- overall_accuracy (OA)
- voicing_recall (VR)
- voicing_false_alarm (VFA)
"""

import warnings

import torch


def hz2cents(freq_hz, base_frequency=10.0):
    """Convert an array of frequency values in Hz to cents. 0 values are left in place.

    Parameters
    ----------
    freq_hz : np.ndarray
        Array of frequencies in Hz.
    base_frequency : float
        Base frequency for conversion.
        (Default value = 10.0)
    """
    freq_cent = torch.zeros_like(freq_hz)
    freq_nonz_ind = torch.nonzero(freq_hz, as_tuple=True)
    normalized_frequency = torch.abs(freq_hz[freq_nonz_ind]) / base_frequency
    freq_cent[freq_nonz_ind] = 1200.0 * torch.log2(normalized_frequency)
    return freq_cent


def freq_to_voicing(frequencies, voicing=None):
    """Convert from an array of frequency values to frequency array + voice/unvoiced array.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies.  A frequency <= 0 indicates "unvoiced".
    voicing : np.ndarray
        Array of voicing values.
        (Default value = None)
        Default None, which means the voicing is inferred from `frequencies`:
            frames with frequency <= 0.0 are considered "unvoiced"
            frames with frequency > 0.0 are considered "voiced"
        If specified, `voicing` is used as the voicing array, but
        frequencies with value 0 are forced to have 0 voicing.
            Voicing inferred by negative frequency values is ignored.

    Returns
    -------
    frequencies : np.ndarray
        Array of frequencies, all >= 0.
    voiced : np.ndarray
        Array of voicings between 0 and 1, same length as frequencies,
        which indicates voiced or unvoiced
    """
    if voicing is not None:
        voicing[frequencies == 0] = 0
    else:
        voicing = (frequencies > 0).type(torch.float32)
    return torch.abs(frequencies), voicing


def validate_voicing(ref_voicing, est_voicing):
    """Check that voicing inputs to a metric are in the correct format.

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference voicing array
    est_voicing : np.ndarray
        Estimated voicing array
    """
    if ref_voicing.numel() == 0:
        warnings.warn("Reference voicing array is empty.")
    if est_voicing.numel() == 0:
        warnings.warn("Estimated voicing array is empty.")
    if ref_voicing.sum() == 0:
        warnings.warn("Reference melody has no voiced frames.")
    if est_voicing.sum() == 0:
        warnings.warn("Estimated melody has no voiced frames.")
    # Make sure they're the same length
    if ref_voicing.shape[0] != est_voicing.shape[0]:
        raise ValueError("Reference and estimated voicing arrays should " "be the same length.")
    for voicing in [ref_voicing, est_voicing]:
        # Make sure voicing is between 0 and 1
        if torch.logical_or(voicing < 0, voicing > 1).any():
            raise ValueError("Voicing arrays must be between 0 and 1.")


def validate(ref_voicing, ref_cent, est_voicing, est_cent):
    """Check that voicing and frequency arrays are well-formed.  To be used in conjunction with
    `mir_eval.melody.validate_voicing`

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference voicing array
    ref_cent : np.ndarray
        Reference pitch sequence in cents
    est_voicing : np.ndarray
        Estimated voicing array
    est_cent : np.ndarray
        Estimate pitch sequence in cents
    """
    if ref_cent.numel() == 0:
        warnings.warn("Reference frequency array is empty.")
    if est_cent.numel() == 0:
        warnings.warn("Estimated frequency array is empty.")
    # Make sure they're the same length
    if (
        ref_voicing.shape[0] != ref_cent.shape[0]
        or est_voicing.shape[0] != est_cent.shape[0]
        or ref_cent.shape[0] != est_cent.shape[0]
    ):
        raise ValueError("All voicing and frequency arrays must have the " "same length.")


def raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50):
    """Compute the raw pitch accuracy given two pitch (frequency) sequences in cents and matching
    voicing indicator sequences. The first pitch and voicing arrays are treated as the reference
    (truth), and the second two as the estimate (prediction).  All 4 sequences must be of the same
    length.

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> raw_pitch = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c,
    ...                                                est_v, est_c)

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference voicing array. When this array is non-binary, it is treated
        as a 'reference reward', as in (Bittner & Bosch, 2019)
    ref_cent : np.ndarray
        Reference pitch sequence in cents
    est_voicing : np.ndarray
        Estimated voicing array
    est_cent : np.ndarray
        Estimate pitch sequence in cents
    cent_tolerance : float
        Maximum absolute deviation in cents for a frequency value to be
        considered correct
        (Default value = 50)

    Returns
    -------
    raw_pitch : float
        Raw pitch accuracy, the fraction of voiced frames in ref_cent for
        which est_cent provides a correct frequency values
        (within cent_tolerance cents).
    """
    validate_voicing(ref_voicing, est_voicing)
    validate(ref_voicing, ref_cent, est_voicing, est_cent)
    # When input arrays are empty, return 0 by special case
    # If there are no voiced frames in reference, metric is 0
    if (
        ref_voicing.numel() == 0
        or ref_voicing.sum() == 0
        or ref_cent.numel() == 0
        or est_cent.numel() == 0
    ):
        return 0.0

    # Raw pitch = the number of voiced frames in the reference for which the
    # estimate provides a correct frequency value (within cent_tolerance cents)
    # NB: voicing estimation is ignored in this measure

    nonzero_freqs = torch.logical_and(est_cent != 0, ref_cent != 0)

    if torch.sum(nonzero_freqs) == 0:
        return 0.0

    freq_diff_cents = torch.abs(ref_cent - est_cent)[nonzero_freqs]
    correct_frequencies = freq_diff_cents < cent_tolerance
    rpa = torch.sum(ref_voicing[nonzero_freqs] * correct_frequencies) / torch.sum(ref_voicing)
    return rpa.item()


def raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50):
    """Compute the raw chroma accuracy given two pitch (frequency) sequences in cents and matching
    voicing indicator sequences. The first pitch and voicing arrays are treated as the reference
    (truth), and the second two as the estimate (prediction).  All 4 sequences must be of the same
    length.

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> raw_chroma = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c,
    ...                                                  est_v, est_c)

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference voicing array. When this array is non-binary, it is treated
        as a 'reference reward', as in (Bittner & Bosch, 2019)
    ref_cent : np.ndarray
        Reference pitch sequence in cents
    est_voicing : np.ndarray
        Estimated voicing array
    est_cent : np.ndarray
        Estimate pitch sequence in cents
    cent_tolerance : float
        Maximum absolute deviation in cents for a frequency value to be
        considered correct
        (Default value = 50)

    Returns
    -------
    raw_chroma : float
        Raw chroma accuracy, the fraction of voiced frames in ref_cent for
        which est_cent provides a correct frequency values (within
        cent_tolerance cents), ignoring octave errors
    """
    validate_voicing(ref_voicing, est_voicing)
    validate(ref_voicing, ref_cent, est_voicing, est_cent)
    # When input arrays are empty, return 0 by special case
    # If there are no voiced frames in reference, metric is 0
    if (
        ref_voicing.numel() == 0
        or ref_voicing.sum() == 0
        or ref_cent.numel() == 0
        or est_cent.numel() == 0
    ):
        return 0.0

    # # Raw chroma = same as raw pitch except that octave errors are ignored.
    nonzero_freqs = torch.logical_and(est_cent != 0, ref_cent != 0)

    if torch.sum(nonzero_freqs) == 0:
        return 0.0

    freq_diff_cents = torch.abs(ref_cent - est_cent)[nonzero_freqs]
    octave = 1200.0 * torch.floor(freq_diff_cents / 1200 + 0.5)
    correct_chroma = torch.abs(freq_diff_cents - octave) < cent_tolerance
    rca = torch.sum(ref_voicing[nonzero_freqs] * correct_chroma) / torch.sum(ref_voicing)
    return rca.item()


def overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance=50):
    """Compute the overall accuracy given two pitch (frequency) sequences in cents and matching
    voicing indicator sequences. The first pitch and voicing arrays are treated as the reference
    (truth), and the second two as the estimate (prediction).  All 4 sequences must be of the same
    length.

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> overall_accuracy = mir_eval.melody.overall_accuracy(ref_v, ref_c,
    ...                                                     est_v, est_c)

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference voicing array. When this array is non-binary, it is treated
        as a 'reference reward', as in (Bittner & Bosch, 2019)
    ref_cent : np.ndarray
        Reference pitch sequence in cents
    est_voicing : np.ndarray
        Estimated voicing array
    est_cent : np.ndarray
        Estimate pitch sequence in cents
    cent_tolerance : float
        Maximum absolute deviation in cents for a frequency value to be
        considered correct
        (Default value = 50)

    Returns
    -------
    overall_accuracy : float
        Overall accuracy, the total fraction of correctly estimates frames,
        where provides a correct frequency values (within cent_tolerance).
    """
    validate_voicing(ref_voicing, est_voicing)
    validate(ref_voicing, ref_cent, est_voicing, est_cent)

    # When input arrays are empty, return 0 by special case
    if (
        ref_voicing.numel() == 0
        or est_voicing.numel() == 0
        or ref_cent.numel() == 0
        or est_cent.numel() == 0
    ):
        return 0.0

    nonzero_freqs = torch.logical_and(est_cent != 0, ref_cent != 0)
    freq_diff_cents = torch.abs(ref_cent - est_cent)[nonzero_freqs]
    correct_frequencies = freq_diff_cents < cent_tolerance
    ref_binary = (ref_voicing > 0).type(torch.float32)
    n_frames = ref_voicing.numel()

    if torch.sum(ref_voicing) == 0:
        ratio = 0.0
    else:
        ratio = torch.sum(ref_binary) / torch.sum(ref_voicing)

    accuracy = (
        (
            ratio
            * torch.sum(
                ref_voicing[nonzero_freqs] * est_voicing[nonzero_freqs] * correct_frequencies
            )
        )
        + torch.sum((1.0 - ref_binary) * (1.0 - est_voicing))
    ) / n_frames

    return accuracy.item()


def voicing_recall(ref_voicing, est_voicing):
    """Compute the voicing recall given two voicing indicator sequences, one as reference (truth)
    and the other as the estimate (prediction).  The sequences must be of the same length.

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> recall = mir_eval.melody.voicing_recall(ref_v, est_v)

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference boolean voicing array
    est_voicing : np.ndarray
        Estimated boolean voicing array

    Returns
    -------
    vx_recall : float
        Voicing recall rate, the fraction of voiced frames in ref
        indicated as voiced in est
    """
    validate_voicing(ref_voicing, est_voicing)
    if ref_voicing.size == 0 or est_voicing.size == 0:
        return 0.0
    ref_indicator = (ref_voicing > 0).type(torch.float32)
    if torch.sum(ref_indicator) == 0:
        return 1
    vr = torch.sum(est_voicing * ref_indicator) / torch.sum(ref_indicator)
    return vr.item()


def voicing_false_alarm(ref_voicing, est_voicing):
    """Compute the voicing false alarm rates given two voicing indicator sequences, one as
    reference (truth) and the other as the estimate (prediction).  The sequences must be of the
    same length.

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> false_alarm = mir_eval.melody.voicing_false_alarm(ref_v, est_v)

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference boolean voicing array
    est_voicing : np.ndarray
        Estimated boolean voicing array

    Returns
    -------
    vx_false_alarm : float
        Voicing false alarm rate, the fraction of unvoiced frames in ref
        indicated as voiced in est
    """
    validate_voicing(ref_voicing, est_voicing)
    if ref_voicing.size == 0 or est_voicing.size == 0:
        return 0.0
    ref_indicator = (ref_voicing == 0).type(torch.float32)
    if torch.sum(ref_indicator) == 0:
        return 0
    vfa = torch.sum(est_voicing * ref_indicator) / torch.sum(ref_indicator)
    return vfa.item()
