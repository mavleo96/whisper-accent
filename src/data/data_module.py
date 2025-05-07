from pathlib import Path

import lightning as L
import torch
from datasets import Audio, Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from src.constants import SAMPLING_RATE, ACCENT_TO_ID_MAP


class EdaccDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        preprocess_batch_size,
        max_length,
        num_workers,
        cache_dir,
        subset_mode=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.preprocess_batch_size = preprocess_batch_size
        self.num_workers = num_workers
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.max_length = max_length
        self.subset_mode = subset_mode
        self.accent_to_id_map = ACCENT_TO_ID_MAP

        # Path to cache and prepared datasets
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = self.cache_dir / "processed"
        if not self.processed_dir.exists():
            self.processed_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self):
        load_dataset(
            "edinburghcstr/edacc", cache_dir=self.cache_dir
        )  # Triggers download

        # Note: edacc has no train split
        val_dataset = load_dataset(
            "edinburghcstr/edacc", split="validation", cache_dir=self.cache_dir
        )
        test_dataset = load_dataset(
            "edinburghcstr/edacc", split="test", cache_dir=self.cache_dir
        )
        if self.subset_mode:
            # Note: for testing purposes
            val_dataset = val_dataset.select(range(100))
            test_dataset = test_dataset.select(range(100))

        # Resample audio to SAMPLING_RATE
        val_dataset = val_dataset.cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )
        test_dataset = test_dataset.cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )

        val_dataset = val_dataset.map(
            self._preprocess,
            batched=True,
            batch_size=self.preprocess_batch_size,
            num_proc=self.num_workers,
            remove_columns=val_dataset.column_names,
            desc="Processing validation dataset",
        )
        test_dataset = test_dataset.map(
            self._preprocess,
            batched=True,
            batch_size=self.preprocess_batch_size,
            num_proc=self.num_workers,
            remove_columns=test_dataset.column_names,
            desc="Processing test dataset",
        )

        val_dataset.save_to_disk(self.processed_dir / "val_dataset")
        test_dataset.save_to_disk(self.processed_dir / "test_dataset")

    def setup(self, stage):
        val_dataset = Dataset.load_from_disk(self.processed_dir / "val_dataset")
        test_dataset = Dataset.load_from_disk(self.processed_dir / "test_dataset")

        val_dataset.set_format(type="torch", columns=val_dataset.column_names)
        test_dataset.set_format(type="torch", columns=test_dataset.column_names)

        # Set datasets
        if stage == "fit":
            self.train_dataset = val_dataset
            self.val_dataset = test_dataset
        elif stage == "validate":
            self.val_dataset = test_dataset
        elif stage == "test":
            self.test_dataset = test_dataset

    def _preprocess(self, batch):
        # TODO: need to extract and preprocess accent labels
        audio_arrays = [i["array"] for i in batch["audio"]]
        texts = [self.processor.tokenizer.normalize(i) for i in batch["text"]]
        accents_ids = torch.tensor([self.accent_to_id_map.get(accent, self.accent_to_id_map['Unknown']) for accent in batch["accent"]], dtype=torch.long)

        input_values = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        label_values = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        return {
            "input_features": input_values.input_features,
            "attention_mask": input_values.attention_mask,
            "labels": label_values.input_ids,
            "decoder_attention_mask": label_values.attention_mask,
            "accent_id": accents_ids
        }

    def collate_fn(self, batch):
        return {
            "input_features": torch.stack([i["input_features"] for i in batch]),
            "attention_mask": torch.stack([i["attention_mask"] for i in batch]),
            "labels": torch.stack([i["labels"] for i in batch]),
            "decoder_attention_mask": torch.stack(
                [i["decoder_attention_mask"] for i in batch]
            ),
            "accent_id": torch.stack([x["accent_id"] for x in batch]),
        }

    def train_dataloader(self):
        # TODO: check if persistent workers are needed
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
            # persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
            # persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
            # persistent_workers=True,
        )


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="baseline_eval.yaml")

    data_module = EdaccDataModule(subset_mode=True, **cfg.data)
    data_module.prepare_data()
    data_module.setup("fit")

    print("train length", len(data_module.train_dataset))
    print("val length", len(data_module.val_dataset))

    print({i: torch.tensor(j).shape for i, j in data_module.train_dataset[0].items()})

    for batch in data_module.train_dataloader():
        print("train", {i: j.shape for i, j in batch.items()})
        print("train", {i: j.dtype for i, j in batch.items()})
        break

    for batch in data_module.val_dataloader():
        print("val", {i: j.shape for i, j in batch.items()})
        print("val", {i: j.dtype for i, j in batch.items()})
        break
