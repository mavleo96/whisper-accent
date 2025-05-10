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
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.optimizer_config = optimizer_config

        if torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        # Reference Link: https://github.com/huggingface/transformers/pull/28687
        if self.model.generation_config.is_multilingual:
            self.model.generation_config.language = "<|en|>"
            self.model.generation_config.task = "transcribe"

        # wer defined on cpu for multi-gpu compatibility
        self.val_wer = WordErrorRate(dist_sync_on_step=True, compute_on_cpu=True)
        self.test_wer = WordErrorRate(dist_sync_on_step=True, compute_on_cpu=True)

    def forward(
        self, input_features, labels, attention_mask, decoder_attention_mask, **kwargs
    ):
        return self.model(
            input_features,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

    def generate(self, input_features, attention_mask, **kwargs):
        predicted_ids = self.model.generate(
            input_features, attention_mask=attention_mask
        )
        predicted_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return [self.processor.tokenizer.normalize(text) for text in predicted_text]

    def compute_loss(self, logits, labels):
        return F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=self.processor.tokenizer.pad_token_id,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = self.compute_loss(outputs.logits, batch["labels"])

        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = self.compute_loss(outputs.logits, batch["labels"])

        self.log(
            "val_loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        predicted_text = self.generate(**batch)
        target_text = self.processor.batch_decode(
            batch["labels"], skip_special_tokens=True
        )

        self.val_wer.update(predicted_text, target_text)
        return {
            "val_loss": loss,
            "predictions": predicted_text,
            "targets": target_text,
        }

    def on_validation_epoch_end(self):
        wer_score = self.val_wer.compute()
        self.log(
            "val_wer",
            wer_score,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.val_wer.reset()

    def test_step(self, batch, batch_idx):
        predicted_text = self.generate(**batch)
        target_text = self.processor.batch_decode(
            batch["labels"], skip_special_tokens=True
        )

        self.test_wer.update(predicted_text, target_text)
        return {
            "predictions": predicted_text,
            "targets": target_text,
        }

    def on_test_epoch_end(self):
        wer_score = self.test_wer.compute()
        self.log(
            "test_wer",
            wer_score,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.test_wer.reset()

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
