import lightning as L
import torch
from torchmetrics.text import WordErrorRate
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class BaseWhisperModel(L.LightningModule):
    """Lightning module for whisper models for baseline model pipelines"""

    def __init__(self, model_name="openai/whisper-base"):
        super().__init__()

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        # TODO: check if this is the correct way to set the language and task
        # https://github.com/huggingface/transformers/pull/28687
        self.model.generation_config.language = "<|en|>"
        self.model.generation_config.task = "transcribe"
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="en",  # Force English language
            task="transcribe",  # Keep it transcription (not translation)
        )
        self.wer = WordErrorRate()

    def forward(self, input_features, attention_mask, labels):
        return self.model(
            input_features=input_features,
            labels=labels,
            attention_mask=attention_mask,
        )

    def training_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Replace padding token id with -100 for label loss masking
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        outputs = self(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        outputs = self(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)

        # Generate predictions
        predicted_ids = self.model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=self.forced_decoder_ids,
        )
        predicted_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        target_text = self.processor.batch_decode(labels, skip_special_tokens=True)

        # Compute WER
        wer_score = self.wer(predicted_text, target_text)
        self.log("val_wer", wer_score, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_features = batch["input_features"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Generate predictions
        predicted_ids = self.model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=self.forced_decoder_ids,
        )
        predicted_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        target_text = self.processor.batch_decode(labels, skip_special_tokens=True)

        # Compute WER
        wer_score = self.wer(predicted_text, target_text)
        self.log("test_wer", wer_score, prog_bar=True)

        return wer_score

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = BaseWhisperModel()
    batch = {
        "input_features": torch.randn(16, 80, 3000),
        "attention_mask": torch.randn(16, 3000),
        "labels": torch.randint(0, 100, (16, 448)),
    }

    outputs = model.test_step(batch, 0)
    print(outputs)
