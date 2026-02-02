import torch
import torch.nn.functional as F
from torchmetrics.functional.text import word_error_rate


def compute_wer(preds, targets, accents=None):
    assert len(preds) == len(targets), "Length of preds and targets must be the same"
    if accents is not None:
        assert len(preds) == len(accents), "Length of preds and accents must be the same"

    overall_wer = word_error_rate(preds, targets).item()
    if accents is None:
        return overall_wer, None

    preds_by_accent = {}
    tgts_by_accent = {}
    for p, t, a in zip(preds, targets, accents, strict=True):
        preds_by_accent.setdefault(a, []).append(p)
        tgts_by_accent.setdefault(a, []).append(t)

    wer_per_accent = []
    for accent_name in sorted(set(accents)):
        accent_wer = word_error_rate(
            preds_by_accent[accent_name], tgts_by_accent[accent_name]
        ).item()
        wer_per_accent.append((accent_name, accent_wer, len(preds_by_accent[accent_name])))

    return overall_wer, wer_per_accent


def repulsive_loss(x, temperature=0.1):
    assert x.ndim == 2, "Input must be a 2D tensor"
    b, d = x.shape

    # Normalize and compute cosine similarity
    x = F.normalize(x, dim=1)
    sim = x @ x.T

    # Mask self-similarity
    mask = torch.eye(b, device=x.device).bool()
    sim = sim.masked_fill(mask, torch.finfo(x.dtype).min)

    # LogSumExp over scaled similarities, normalized by sqrt(d)
    return torch.logsumexp(sim / temperature, dim=1).mean() / d**0.5


__all__ = ["compute_wer", "repulsive_loss"]
