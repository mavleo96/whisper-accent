import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.data_module import EdaccDataModule
from src.models.base_model import BaseWhisperModel

torch.set_float32_matmul_precision("high")


def main(model_name="openai/whisper-base.en", batch_size=8, num_workers=12):
    model = BaseWhisperModel(model_name=model_name)
    data_module = EdaccDataModule(
        model_name=model_name,
        batch_size=batch_size,
        preprocess_batch_size=32,
        max_length=448,
        num_workers=num_workers,
        cache_dir="/data/vijay/rice-bag/data",
        subset_mode=True,
    )

    # Initialize loggers
    tensorboard_logger = TensorBoardLogger(
        save_dir="logs", name=f"baseline_eval_{model_name.split('/')[-1]}"
    )

    # Initialize trainer
    trainer = L.Trainer(
        logger=[tensorboard_logger],
        devices=1,
        precision=32,
        accelerator="gpu",
        strategy="ddp",
    )
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
