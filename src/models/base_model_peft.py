import lightning as L
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.text import WordErrorRate
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class BaseWhisperModelWithPEFT(L.LightningModule):
    """Lightning module for whisper models with PEFT for efficient fine-tuning"""

    def __init__(self, model_name, optimizer_config):
        super().__init__()
        self.save_hyperparameters()

        # Initialize base model
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.optimizer_config = optimizer_config

        peft_config = LoraConfig(
            inference_mode=False,
            r=8,  # rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        )
        self.model = get_peft_model(self.model, peft_config)

        # Print trainable parameters info
        self.model.print_trainable_parameters()

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
            input_features=input_features,
            attention_mask=attention_mask,
        )
        predicted_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return [self.processor.tokenizer.normalize(text) for text in predicted_text]

    def compute_loss(self, logits, labels):
        return F.cross_entropy(
            logits.transpose(1, 2),
            labels,
        )

    def on_train_epoch_start(self):
        self.model.train()

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

    def on_validation_epoch_start(self):
        self.model.eval()

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
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        self.val_wer.reset()

    def on_test_epoch_start(self):
        self.model.eval()

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
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay
        warmup_steps = 100

        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay
        )

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)

        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "linear_warmup",
        }

        return [optimizer], [scheduler_dict]


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

    outputs = model.forward(**batch)
    print(outputs)

    outputs = model.test_step(batch, 0)
    print(outputs)
