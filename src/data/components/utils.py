import torch


def mix_at_snr(signal, noise, snr_target_db=0):
    var_audio = signal.var(dim=1, keepdims=True)
    var_audio_acc = noise.var(dim=1, keepdims=True)

    if isinstance(snr_target_db, int) or isinstance(snr_target_db, float):
        snr_target_db = torch.ones_like(var_audio) * snr_target_db
    elif isinstance(snr_target_db, torch.Tensor) and (snr_target_db.ndim == 1):
        snr_target_db = snr_target_db.unsqueeze(dim=1)

    snr_db = 10 * (torch.log10(var_audio) - torch.log10(var_audio_acc))
    snr_target_db = snr_target_db.to(snr_db.device)

    a = torch.zeros_like(var_audio)
    a[~torch.isnan(snr_target_db)] = 10 ** (
        -(snr_target_db[~torch.isnan(snr_target_db)] - snr_db[~torch.isnan(snr_target_db)]) / 20
    )

    mix = (signal + a * noise) / (1 + a)
    return mix
