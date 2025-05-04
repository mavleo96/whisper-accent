from io import StringIO
from pathlib import Path

import hydra
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@hydra.main(
    config_path="../../configs", config_name="baseline_eval.yaml", version_base=None
)
def main(cfg):
    log_dir = cfg.log_dir
    if not Path(log_dir).exists():
        raise FileNotFoundError(f"Log directory {log_dir} does not exist")

    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    events = event_acc.Tensors("predictions/text_summary")
    pred_str = (
        events[0]
        .tensor_proto.string_val[0]
        .decode("utf-8")
        .strip("<pre>")
        .strip("</pre>")
    )
    df = pd.read_csv(StringIO(pred_str))

    df.to_csv(Path(log_dir) / "predictions.csv", index=False)

    return df


if __name__ == "__main__":
    main()
