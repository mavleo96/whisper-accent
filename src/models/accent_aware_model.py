import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.text import WordErrorRate
from torchmetrics.classification import Accuracy
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class AccentAwareWhisperModel(L.LightningModule):
    """Lightning module for whisper models for baseline model pipelines"""

    def __init__(self, model_name, optimizer_config, num_accents=11):
        super().__init__()
        self.save_hyperparameters()

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.optimizer_config = optimizer_config
        self.num_accents = num_accents

        self.accent_tokens = [f"<|accent_{i}|>" for i in range(num_accents)]
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": self.accent_tokens})
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.model = torch.compile(self.model)

        # Reference Link: https://github.com/huggingface/transformers/pull/28687
        if self.model.generation_config.is_multilingual:
            self.model.generation_config.language = "<|en|>"
            self.model.generation_config.task = "transcribe"

        # Create separate WER metrics for each accent
        self.val_wer = WordErrorRate(dist_sync_on_step=True, compute_on_cpu=True)
        self.test_wer = WordErrorRate(dist_sync_on_step=True, compute_on_cpu=True)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_accents)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_accents)

        self.accent_token_ids = torch.tensor([
            self.processor.tokenizer.convert_tokens_to_ids(tok) for tok in self.accent_tokens
        ])
        
        

    def prepare_decoder_input(self, labels, accent_ids, decoder_attention_masks):
        batch_size = labels.size(0)
        pad_id = self.processor.tokenizer.pad_token_id
        sot_token_id = self.processor.tokenizer.encode("<|startoftranscript|>")[0]
        lang_token_id = self.processor.tokenizer.encode("<|en|>")[0]
        task_token_id = self.processor.tokenizer.encode("<|transcribe|>")[0]

        new_labels = torch.full_like(labels, pad_id)
        new_decoder_masks = torch.zeros_like(decoder_attention_masks)

        for i in range(batch_size):
            accent_token_id = self.processor.tokenizer.convert_tokens_to_ids(f"<|accent_{accent_ids[i].item()}|>")
            prefix = [sot_token_id, lang_token_id, task_token_id, accent_token_id]

            original = labels[i]
            prefix_len = len(prefix)

            new_labels[i, :prefix_len] = torch.tensor(prefix, device=labels.device)

            original_trimmed = original[3:]
            remaining = original_trimmed[original_trimmed != pad_id]
            max_copy_len = new_labels.size(1) - prefix_len

            copy_len = min(len(remaining), max_copy_len)
            new_labels[i, prefix_len:prefix_len + copy_len] = remaining[:copy_len]
            new_decoder_masks[i, :prefix_len + copy_len] = 1

        return new_labels, new_decoder_masks

    
    def forward(self, input_features, labels=None, accent_ids=None,
                attention_mask=None, decoder_attention_mask=None, **kwargs):
        return self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
    

    def predict_accent(self, input_features, attention_mask):
        accent_only_logits = self.get_accent_logits(input_features, attention_mask)
        predicted_accent_idx = accent_only_logits.argmax(dim=-1)

        return predicted_accent_idx
    
    def get_accent_logits(self, input_features, attention_mask):
        batch_size = input_features.size(0)
        pad_id = self.processor.tokenizer.pad_token_id

        prefix = [
            self.processor.tokenizer.encode("<|startoftranscript|>")[0],
            self.processor.tokenizer.encode("<|en|>")[0],
            self.processor.tokenizer.encode("<|transcribe|>")[0],
        ]

        labels = torch.full((batch_size, 4), pad_id, device=input_features.device)
        for i in range(batch_size):
            labels[i, :3] = torch.tensor(prefix, device=input_features.device)

        decoder_attention_mask = (labels != pad_id).long()

        outputs = self.forward(
            input_features=input_features,
            labels=labels,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

        # Logits of 4th position (where accent token would be predicted)
        accent_logits = outputs.logits[:, 3, :]
        accent_only_logits = accent_logits[:, self.accent_token_ids.to(accent_logits.device)]
        return accent_only_logits

    def generate(self, input_features, attention_mask, **kwargs):    
        predicted_accent_ids = self.predict_accent(input_features, attention_mask)
        start_token = self.processor.tokenizer.encode("<|startoftranscript|>")[0]
        en_token = self.processor.tokenizer.encode("<|en|>")[0]
        transcribe_token = self.processor.tokenizer.encode("<|transcribe|>")[0]
        pad_token = self.processor.tokenizer.pad_token_id
        
        # Create forced_decoder_ids for each sample
        batch_decoder_ids = []
        for acc_id in predicted_accent_ids.tolist():
            accent_token = self.processor.tokenizer.encode(f"<|accent_{acc_id}|>")[0]
            prefix = [start_token, en_token, transcribe_token, accent_token]
            batch_decoder_ids.append(torch.tensor(prefix, device=input_features.device))


        max_len = max(len(ids) for ids in batch_decoder_ids)
        decoder_input_ids = torch.full((len(batch_decoder_ids), max_len), pad_token, device=input_features.device)
        for i, ids in enumerate(batch_decoder_ids):
            decoder_input_ids[i, :len(ids)] = ids

        pred_ids = self.model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
                
        pred_text = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        pred_text = [self.processor.tokenizer.normalize(text) for text in pred_text]
        return pred_text, predicted_accent_ids
        

    def training_step(self, batch, batch_idx):
        decoder_input, decoder_mask = self.prepare_decoder_input(batch["labels"], batch["accent_id"], batch["decoder_attention_mask"])
        outputs = self.forward(
            input_features=batch["input_features"],
            labels=decoder_input,
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=decoder_mask
        )
        transcription_loss = F.cross_entropy(
            outputs.logits.transpose(1, 2),
            decoder_input,
            ignore_index=self.processor.tokenizer.pad_token_id,
        )
        accent_only_logits = self.get_accent_logits(batch["input_features"], batch["attention_mask"])
        accent_loss = F.cross_entropy(accent_only_logits, batch["accent_id"])

        alpha = 0.9  
        beta = 0.1
        combined_loss = alpha * transcription_loss + beta * accent_loss


        self.log(
            "train_loss",
            combined_loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_transcription_loss", transcription_loss, sync_dist=True, on_epoch=True)
        self.log("train_accent_loss", accent_loss, sync_dist=True, on_epoch=True)

        predicted_accent_idx = accent_only_logits.argmax(dim=-1)
        accent_accuracy = (predicted_accent_idx == batch["accent_id"]).float().mean()
        self.log("train_accent_accuracy", accent_accuracy, sync_dist=True, on_epoch=True)

        return {"loss": combined_loss}

    def validation_step(self, batch, batch_idx):
        decoder_input, decoder_mask = self.prepare_decoder_input(batch["labels"], batch["accent_id"], batch["decoder_attention_mask"])
        outputs = self.forward(
            input_features=batch["input_features"],
            labels=decoder_input,
            attention_mask=batch["attention_mask"],
            decoder_attention_mask=decoder_mask
        )
        transcription_loss = F.cross_entropy(
            outputs.logits.transpose(1, 2),
            decoder_input,
            ignore_index=self.processor.tokenizer.pad_token_id,
        )

        accent_only_logits = self.get_accent_logits(batch["input_features"], batch["attention_mask"])
        accent_loss = F.cross_entropy(accent_only_logits, batch["accent_id"])

        alpha = 0.9  
        beta = 0.1  
        combined_loss = alpha * transcription_loss + beta * accent_loss


        self.log(
            "val_loss", combined_loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("val_transcription_loss", transcription_loss, sync_dist=True, on_epoch=True)
        self.log("val_accent_loss", accent_loss, sync_dist=True, on_epoch=True)

        predicted_text, predicted_accent = self.generate(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"]
        )
        target_text = self.processor.batch_decode(
            batch["labels"], skip_special_tokens=True
        )

        # Update overall WER
        self.val_wer.update(predicted_text, target_text)
        self.val_acc.update(predicted_accent.cpu(), batch["accent_id"].cpu())
        return {"val_loss": combined_loss, "targets": target_text, "predictions": predicted_text, "target_accent": batch["accent_id"], "predicted_accent": predicted_accent}

    def on_validation_epoch_end(self):
        # Log overall WER
        wer_score = self.val_wer.compute()
        self.log(
            "val_wer",
            wer_score,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        acc_score = self.val_acc.compute()
        self.log("val_acc", acc_score, sync_dist=True, prog_bar=True)
        self.val_wer.reset()
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        predicted_text, predicted_accent = self.generate(
            input_features=batch["input_features"],
            attention_mask=batch["attention_mask"]
        )
        target_text = self.processor.batch_decode(
            batch["labels"], skip_special_tokens=True
        )

        # Update overall WER
        self.test_wer.update(predicted_text, target_text)
        self.test_acc.update(predicted_accent.cpu(), batch["accent_id"].cpu())

        
        return {
            "predictions": predicted_text,
            "targets": target_text,
            "predicted_accent": predicted_accent,
            "target_accent": batch["accent_id"]
        }

    def on_test_epoch_end(self):
        # Log overall WER
        wer_score = self.test_wer.compute()
        self.log(
            "test_wer",
            wer_score,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        acc_score = self.test_acc.compute()
        self.log("test_acc", acc_score, sync_dist=True, prog_bar=True)

        self.test_wer.reset()
        self.test_acc.reset()
        

    def configure_optimizers(self):
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="baseline_eval.yaml")

    model = AccentAwareWhisperModel(**cfg.model).to("cuda")
    batch = {
        "input_features": torch.randn(16, 80, 3000).to("cuda"),
        "attention_mask": torch.randint(0, 2, (16, 3000)).to("cuda"),
        "labels": torch.randint(0, 51865, (16, 448)).to("cuda"),
        "decoder_attention_mask": torch.randint(0, 2, (16, 448)).to("cuda"),
        "accent_id": torch.randint(0, 11, (16,)).to("cuda"),
    }

    outputs = model.test_step(batch, 0)
    print(outputs)