import evaluate


def compute_wer(preds, targets, accents=None):
    assert len(preds) == len(targets), "Length of preds and targets must be the same"
    if accents is not None:
        assert len(preds) == len(accents), "Length of preds and accents must be the same"

    metric = evaluate.load("wer")
    overall_wer = metric.compute(predictions=preds, references=targets)
    if accents is None:
        return overall_wer, None

    preds_by_accent = {}
    tgts_by_accent = {}
    for p, t, a in zip(preds, targets, accents, strict=True):
        preds_by_accent.setdefault(a, []).append(p)
        tgts_by_accent.setdefault(a, []).append(t)

    wer_per_accent = {}
    for accent_name in sorted(set(accents)):
        accent_wer = metric.compute(
            predictions=preds_by_accent[accent_name], references=tgts_by_accent[accent_name]
        )
        wer_per_accent[accent_name] = {
            "wer": accent_wer,
            "n": len(preds_by_accent[accent_name]),
        }

    return overall_wer, wer_per_accent


__all__ = ["compute_wer"]
