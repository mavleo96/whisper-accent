import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.text import WordErrorRate
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class BaseWhisperModel(L.LightningModule):
    """Lightning module for whisper models for baseline model pipelines"""

    def __init__(self, model_name, optimizer_config):
        super().__init__()
        self.save_hyperparameters()

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model = torch.compile(self.model)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.optimizer_config = optimizer_config

        # Reference Link: https://github.com/huggingface/transformers/pull/28687
        if self.model.generation_config.is_multilingual:
            self.model.generation_config.language = "<|en|>"
            self.model.generation_config.task = "transcribe"

        # TODO: check if this computes correctly
        self.wer = WordErrorRate()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = F.cross_entropy(
            outputs.logits.transpose(1, 2),
            batch["labels"],
            ignore_index=self.processor.tokenizer.pad_token_id,
        )
        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = F.cross_entropy(
            outputs.logits.transpose(1, 2),
            batch["labels"],
            ignore_index=self.processor.tokenizer.pad_token_id,
        )
        self.log(
            "val_loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True
        )

        predicted_ids = self.model.generate(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
        )
        predicted_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        target_text = self.processor.batch_decode(
            batch["labels"], skip_special_tokens=True
        )

        wer_score = self.wer(predicted_text, target_text)
        self.log(
            "val_wer",
            wer_score,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"val_loss": loss, "val_wer": wer_score}

    def test_step(self, batch, batch_idx):
        predicted_ids = self.model.generate(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
        )
        predicted_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        target_text = self.processor.batch_decode(
            batch["labels"], skip_special_tokens=True
        )

        wer_score = self.wer(predicted_text, target_text)
        self.log(
            "test_wer",
            wer_score,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"test_wer": wer_score}

    def configure_optimizers(self):
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="baseline_eval.yaml")

    model = BaseWhisperModel(**cfg.model).to("cuda")
    batch = {
        "input_features": torch.randn(16, 80, 3000).to("cuda"),
        "attention_mask": torch.randint(0, 2, (16, 3000)).to("cuda"),
        "labels": torch.randint(0, 51865, (16, 448)).to("cuda"),
        "decoder_attention_mask": torch.randint(0, 2, (16, 448)).to("cuda"),
    }

    outputs = model.test_step(batch, 0)
    print(outputs)
