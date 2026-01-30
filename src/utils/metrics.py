from torchmetrics.functional.text import word_error_rate


def compute_wer(preds, targets, accents=None):
    assert len(preds) == len(targets), "Length of preds and targets must be the same"
    if accents is not None:
        assert len(preds) == len(accents), (
            "Length of preds and accents must be the same"
        )

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
        wer_per_accent.append(
            (accent_name, accent_wer, len(preds_by_accent[accent_name]))
        )

    return overall_wer, wer_per_accent


__all__ = ["compute_wer"]
