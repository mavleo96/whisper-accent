import lightning as L
import torch
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from src.data.utils import preprocess_labels


class EdaccDataModule(L.LightningDataModule):
    def __init__(self, model_name, batch_size, num_workers, cache_dir):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def prepare_data(self):
        load_dataset(
            "edinburghcstr/edacc", cache_dir=self.cache_dir
        )  # Triggers download

    def setup(self, stage):
        # Note: edacc has no train split
        val_dataset = load_dataset(
            "edinburghcstr/edacc", split="validation", cache_dir=self.cache_dir
        ).select(range(100))
        test_dataset = load_dataset(
            "edinburghcstr/edacc", split="test", cache_dir=self.cache_dir
        ).select(range(100))

        val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
        test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Process datasets in batches
        # TODO: investigate weird behavior, tensors automatically convert to lists
        val_dataset = val_dataset.map(
            self._preprocess,
            batched=True,
            batch_size=32,
            num_proc=self.num_workers,
            remove_columns=val_dataset.column_names,
        )
        test_dataset = test_dataset.map(
            self._preprocess,
            batched=True,
            batch_size=32,
            num_proc=self.num_workers,
            remove_columns=test_dataset.column_names,
        )

        if stage == "fit":
            self.train_dataset = val_dataset
            self.val_dataset = test_dataset
        elif stage == "validate":
            self.val_dataset = test_dataset
        elif stage == "test":
            self.test_dataset = test_dataset

    def _preprocess(self, batch):
        # TODO: need to extract and preprocess accent labels
        audio_arrays = [item["array"] for item in batch["audio"]]
        texts = batch["text"]

        # Process audio features in batch
        input_values = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Process text labels in batch
        label_values = self.processor.tokenizer(
            [preprocess_labels(text) for text in texts],
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            max_length=448,
        )

        return {
            "input_features": input_values.input_features,
            "attention_mask": input_values.attention_mask,
            "labels": label_values.input_ids,
            "labels_attention_mask": label_values.attention_mask,
        }

    def collate_fn(self, batch):
        # Note: torch.tensor used instead of torch.stack to handle weird behavior
        return {
            "input_features": torch.tensor([item["input_features"] for item in batch]),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
            "labels": torch.tensor([item["labels"] for item in batch]),
            # "labels_attention_mask": torch.tensor([item["labels_attention_mask"] for item in batch]),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    data_module = EdaccDataModule(
        model_name="openai/whisper-small",
        batch_size=16,
        num_workers=4,
        cache_dir="data",
    )
    data_module.setup("fit")
    print(len(data_module.train_dataset))
    print(len(data_module.val_dataset))

    print({i: torch.tensor(j).shape for i, j in data_module.train_dataset[0].items()})

    for batch in data_module.train_dataloader():
        print("train", {i: j.shape for i, j in batch.items()})
        break

    for batch in data_module.val_dataloader():
        print("val", {i: j.shape for i, j in batch.items()})
        break
