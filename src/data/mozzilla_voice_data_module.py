from pathlib import Path

import lightning as L
import torch
from datasets import Audio, Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from src.constants import ACCENT_TO_ID_MAP, SAMPLING_RATE


class MozillaVoiceDataModule(L.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        preprocess_batch_size,
        max_length,
        num_workers,
        cache_dir,
        subset_mode=False,
        force_prepare=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.preprocess_batch_size = preprocess_batch_size
        self.num_workers = num_workers
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.max_length = max_length
        self.subset_mode = subset_mode
        self.force_prepare = force_prepare
        self.accent_to_id_map = ACCENT_TO_ID_MAP

        # Path to cache and prepared datasets
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = self.cache_dir / "processed"
        if not self.processed_dir.exists():
            self.processed_dir.mkdir(parents=True, exist_ok=True)

        model_dir_name = model_name.replace("/", "_").replace(".", "_")
        model_dir = self.processed_dir / model_dir_name
        self.subset_dir = model_dir / "subset"
        self.full_dir = model_dir / "full"
        if not self.subset_dir.exists():
            self.subset_dir.mkdir(parents=True, exist_ok=True)
        if not self.full_dir.exists():
            self.full_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self):
        target_dir = self.subset_dir if self.subset_mode else self.full_dir
        train_path = target_dir / "train_dataset" / "state.json"
        val_path = target_dir / "val_dataset" / "state.json"
        test_path = target_dir / "test_dataset" / "state.json"
        
        if not self.force_prepare and train_path.exists() and val_path.exists() and test_path.exists():
            print(f"Processed datasets already exist in {target_dir}, skipping preparation...")
            return

        if self.force_prepare:
            print(f"Force prepare enabled, reprocessing datasets in {target_dir}...")

        # Load Common Voice dataset with default config
        dataset = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            'en',  # Using the English language configuration
            cache_dir=self.cache_dir,
            split=["train", "validation", "test"]
        )

        # Select a small subset of the dataset (e.g., first 1000 samples for train, 100 for validation/test)
        train_dataset = dataset[0].select(range(100))  # Load first 1000 samples for training
        val_dataset = dataset[1].select(range(100))    # Load first 100 samples for validation
        test_dataset = dataset[2].select(range(100))   # Load first 100 samples for testing

        # Filter for English language (optional if you want to ensure only English)
        train_dataset = train_dataset.filter(lambda x: x["locale"] == "en")
        val_dataset = val_dataset.filter(lambda x: x["locale"] == "en")
        test_dataset = test_dataset.filter(lambda x: x["locale"] == "en")

        # Resample audio to SAMPLING_RATE
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
        test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

        # Process datasets
        train_dataset = train_dataset.map(
            self._preprocess,
            batched=True,
            batch_size=self.preprocess_batch_size,
            num_proc=self.num_workers,
            remove_columns=train_dataset.column_names,
            desc="Processing training dataset",
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

        # Save processed datasets
        train_dataset.save_to_disk(target_dir / "train_dataset")
        val_dataset.save_to_disk(target_dir / "val_dataset")
        test_dataset.save_to_disk(target_dir / "test_dataset")

    def setup(self, stage):
        target_dir = self.subset_dir if self.subset_mode else self.full_dir

        if stage == "fit":
            self.train_dataset = Dataset.load_from_disk(target_dir / "train_dataset")
            self.val_dataset = Dataset.load_from_disk(target_dir / "val_dataset")
            self.train_dataset.set_format(type="torch", columns=self.train_dataset.column_names)
            self.val_dataset.set_format(type="torch", columns=self.val_dataset.column_names)
        elif stage == "validate":
            self.val_dataset = Dataset.load_from_disk(target_dir / "val_dataset")
            self.val_dataset.set_format(type="torch", columns=self.val_dataset.column_names)
        elif stage == "test":
            self.test_dataset = Dataset.load_from_disk(target_dir / "test_dataset")
            self.test_dataset.set_format(type="torch", columns=self.test_dataset.column_names)

    def _preprocess(self, batch):
        # Extract audio arrays and normalize text
        audio_arrays = [i["array"] for i in batch["audio"]]
        texts = [self.processor.tokenizer.normalize(i) for i in batch["sentence"]]
        
        # Map accents to IDs (Common Voice uses 'accent' field)
        # Convert accents to lowercase and handle missing accents
        accents_ids = torch.tensor(
            [
                self.accent_to_id_map.get(
                    accent.lower() if accent else "unknown",
                    self.accent_to_id_map["Unknown"]
                )
                for accent in batch["accent"]
            ],
            dtype=torch.long,
        )

        # Process audio features with Whisper processor
        input_values = self.processor(
            audio_arrays,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Process text labels with Whisper tokenizer
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
            "accent_id": accents_ids,
            "text": texts,  # Keep original text for WER calculation
        }

    def collate_fn(self, batch):
        return {
            "input_features": torch.stack([i["input_features"] for i in batch]),
            "attention_mask": torch.stack([i["attention_mask"] for i in batch]),
            "labels": torch.stack([i["labels"] for i in batch]),
            "decoder_attention_mask": torch.stack([i["decoder_attention_mask"] for i in batch]),
            "accent_id": torch.stack([x["accent_id"] for x in batch]),
            "text": [x["text"] for x in batch],  # Keep as list for WER calculation
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
    from hydra import compose, initialize

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="baseline_eval.yaml")

    data_module = MozillaVoiceDataModule(subset_mode=True, **cfg.data)
    data_module.prepare_data()
    data_module.setup("fit")

    print("train length", len(data_module.train_dataset))
    print("val length", len(data_module.val_dataset))

    print({i: torch.tensor(j).shape if isinstance(j, torch.Tensor) else j for i, j in data_module.train_dataset[0].items()})

    for batch in data_module.train_dataloader():
        print("train", {i: j.shape if isinstance(j, torch.Tensor) else len(j) for i, j in batch.items()})
        print("train", {i: j.dtype if isinstance(j, torch.Tensor) else type(j) for i, j in batch.items()})
        break

    for batch in data_module.val_dataloader():
        print("val", {i: j.shape if isinstance(j, torch.Tensor) else len(j) for i, j in batch.items()})
        print("val", {i: j.dtype if isinstance(j, torch.Tensor) else type(j) for i, j in batch.items()})
        break
