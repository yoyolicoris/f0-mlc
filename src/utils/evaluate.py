import src.utils.mir_eval_melody_torch as mir_eval_melody_torch


def compute_metrics(predictions, labels, cent_tolerance=50):
    # convert semitones to cents and infer voicing
    ref_hz, ref_voicing = mir_eval_melody_torch.freq_to_voicing(labels.flatten())
    est_hz, est_voicing = mir_eval_melody_torch.freq_to_voicing(predictions.flatten())

    ref_cent = mir_eval_melody_torch.hz2cents(ref_hz)
    est_cent = mir_eval_melody_torch.hz2cents(est_hz)

    # compute mir_eval metrics
    metrics = {}
    metrics["RPA"] = mir_eval_melody_torch.raw_pitch_accuracy(
        ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance
    )
    metrics["RCA"] = mir_eval_melody_torch.raw_chroma_accuracy(
        ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance
    )
    metrics["OA"] = mir_eval_melody_torch.overall_accuracy(
        ref_voicing, ref_cent, est_voicing, est_cent, cent_tolerance
    )
    metrics["VR"] = mir_eval_melody_torch.voicing_recall(ref_voicing, est_voicing)
    metrics["VFA"] = mir_eval_melody_torch.voicing_false_alarm(ref_voicing, est_voicing)

    return metrics
