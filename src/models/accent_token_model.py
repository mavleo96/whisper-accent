import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from torchmetrics.text import WordErrorRate
from transformers import WhisperProcessor

from src.constants import NUM_ACCENTS

from .accent_token.model import WhisperWithAccentToken

ACCENT_LAMBDA = 0.1


class AccentAwareWhisperModel(L.LightningModule):
    """Lightning module for whisper models for baseline model pipelines"""

    def __init__(self, model_name, optimizer_config):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_config = optimizer_config

        # Initialize processor and add accent tokens
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.accent_tokens = [f"<|accent{i}|>" for i in range(NUM_ACCENTS)]
        self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.accent_tokens}
        )

        self.accent_token_ids = [
            self.processor.tokenizer.convert_tokens_to_ids(token)
            for token in self.accent_tokens
        ]
        self.accent_token_id_map = {
            i: token_id for i, token_id in enumerate(self.accent_token_ids)
        }

        accent_detection_decoder_input_ids = torch.tensor(
            [
                self.processor.tokenizer.encode("<|startoftranscript|>")[0],
                self.processor.tokenizer.encode("<|en|>")[0],
                self.processor.tokenizer.encode("<|transcribe|>")[0],
            ]
        )

        self.model = WhisperWithAccentToken.from_pretrained(
            model_name,
            accent_detection_decoder_input_ids=accent_detection_decoder_input_ids,
            accent_token_id_map=self.accent_token_id_map,
        )
        self.model = torch.compile(self.model)

        # Reference Link: https://github.com/huggingface/transformers/pull/28687
        if self.model.generation_config.is_multilingual:
            self.model.generation_config.language = "<|en|>"
            self.model.generation_config.task = "transcribe"

        self.val_wer = WordErrorRate(dist_sync_on_step=True, compute_on_cpu=True)
        self.test_wer = WordErrorRate(dist_sync_on_step=True, compute_on_cpu=True)
        self.val_acc = Accuracy(
            task="multiclass",
            num_classes=NUM_ACCENTS,
            dist_sync_on_step=True,
            compute_on_cpu=True,
        )
        self.test_acc = Accuracy(
            task="multiclass",
            num_classes=NUM_ACCENTS,
            dist_sync_on_step=True,
            compute_on_cpu=True,
        )

    def prepare_decoder_input(self, labels, accent_ids, decoder_attention_masks):
        accent_token_ids = torch.tensor(
            [self.accent_token_id_map[acc_id.item()] for acc_id in accent_ids],
            device=labels.device,
        ).unsqueeze(1)

        new_labels = torch.full_like(labels, 0)
        new_labels[:, :3] = labels[:, :3]
        new_labels[:, 3] = accent_token_ids.squeeze(1)
        new_labels[:, 4:] = labels[:, 3:-1]

        new_decoder_masks = torch.zeros_like(decoder_attention_masks)
        new_decoder_masks[:, :3] = decoder_attention_masks[:, :3]
        new_decoder_masks[:, 3] = 1
        new_decoder_masks[:, 4:] = decoder_attention_masks[:, 3:-1]

        return new_labels, new_decoder_masks

    def forward(
        self,
        input_features,
        labels=None,
        accent_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        return self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

    def generate(self, input_features, attention_mask, **kwargs):
        model_output = self.model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )

        pred_ids = model_output["generated_text"]
        accent_ids = model_output["accent_ids"]

        pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        pred_text = [self.processor.tokenizer.normalize(text) for text in pred_text]

        return pred_text, accent_ids

    def compute_losses(self, outputs, batch):
        transcription_loss = F.cross_entropy(
            outputs.logits.transpose(1, 2),
            batch["labels"],
            ignore_index=self.processor.tokenizer.pad_token_id,
        )

        accent_only_logits = outputs.logits[:, 3, self.accent_token_ids]
        accent_loss = F.cross_entropy(accent_only_logits, batch["accent_id"])

        combined_loss = transcription_loss + ACCENT_LAMBDA * accent_loss

        return {
            "transcription_loss": transcription_loss,
            "accent_loss": accent_loss,
            "combined_loss": combined_loss,
            "accent_logits": accent_only_logits,
        }

    def log_metrics(self, prefix, losses, accent_logits, batch):
        self.log(
            f"{prefix}_loss",
            losses["combined_loss"],
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{prefix}_transcription_loss",
            losses["transcription_loss"],
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"{prefix}_accent_loss",
            losses["accent_loss"],
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

        predicted_accent_indices = accent_logits.argmax(dim=-1)
        accent_accuracy = (
            (predicted_accent_indices == batch["accent_id"]).float().mean()
        )
        self.log(
            f"{prefix}_accent_accuracy",
            accent_accuracy,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(self, batch, batch_idx):
        decoder_input, decoder_mask = self.prepare_decoder_input(
            batch["labels"], batch["accent_id"], batch["decoder_attention_mask"]
        )

        outputs = self.forward(
            input_features=batch["input_features"],
            labels=decoder_input,
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=decoder_mask,
        )

        losses = self.compute_losses(outputs, batch)
        self.log_metrics("train", losses, losses["accent_logits"], batch)

        return {"loss": losses["combined_loss"]}

    def validation_step(self, batch, batch_idx):
        decoder_input, decoder_mask = self.prepare_decoder_input(
            batch["labels"], batch["accent_id"], batch["decoder_attention_mask"]
        )

        outputs = self.forward(
            input_features=batch["input_features"],
            labels=decoder_input,
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=decoder_mask,
        )

        losses = self.compute_losses(outputs, batch)
        self.log_metrics("val", losses, losses["accent_logits"], batch)

        return {"val_loss": losses["combined_loss"]}

    def on_validation_epoch_end(self):
        self.val_wer.reset()
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        predicted_text, predicted_accent = self.generate(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"],
        )
        target_text = self.processor.batch_decode(
            batch["labels"], skip_special_tokens=True
        )

        self.test_wer.update(predicted_text, target_text)
        self.test_acc.update(predicted_accent.cpu(), batch["accent_id"].cpu())

        return {
            "predictions": predicted_text,
            "targets": target_text,
            "predicted_accent": predicted_accent,
            "target_accent": batch["accent_id"],
        }

    def on_test_epoch_end(self):
        wer_score = self.test_wer.compute()
        acc_score = self.test_acc.compute()

        self.log(
            "test_wer",
            wer_score,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_acc",
            acc_score,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.test_wer.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="accent_token_model.yaml")

    model = AccentAwareWhisperModel(**cfg.model).to("cuda")
    batch = {
        "input_features": torch.randn(16, 80, 3000).to("cuda"),
        "attention_mask": torch.randint(0, 2, (16, 3000)).to("cuda"),
        "labels": torch.randint(0, 51865, (16, 448)).to("cuda"),
        "decoder_attention_mask": torch.randint(0, 2, (16, 448)).to("cuda"),
        "accent_id": torch.randint(0, 11, (16,)).to("cuda"),
    }

    outputs = model.test_step(batch, 0)
