from pathlib import Path

import lightning as L
import pandas as pd
import torch
from datasets import Audio, Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from src.constants import ACCENT_TO_ID_MAP, SAMPLING_RATE


class CommonVoiceDataModule(L.LightningDataModule):
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

        # Create common voice specific directory
        self.common_voice_dir = self.processed_dir / "common_voice"
        if not self.common_voice_dir.exists():
            self.common_voice_dir.mkdir(parents=True, exist_ok=True)

        model_dir_name = model_name.replace("/", "_").replace(".", "_")
        model_dir = self.common_voice_dir / model_dir_name
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

        # if not self.force_prepare and train_path.exists() and val_path.exists() and test_path.exists():
        #     print(f"Processed datasets already exist in {target_dir}, skipping preparation...")
        #     return

        # if self.force_prepare:
        #     print(f"Force prepare enabled, reprocessing datasets in {target_dir}...")

        # Load Common Voice dataset with default config
        dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "en",  # Using the English language configuration
            cache_dir=self.cache_dir,
            split=["train", "validation", "test"],
        )

        # Select a small subset of the dataset
        train_dataset = dataset[0].select(
            range(100)
        )  # Load first 100 samples for training
        val_dataset = dataset[1].select(
            range(100)
        )  # Load first 100 samples for validation
        test_dataset = dataset[2].select(
            range(100)
        )  # Load first 100 samples for testing

        # Filter for English language
        train_dataset = train_dataset.filter(lambda x: x["locale"] == "en")
        val_dataset = val_dataset.filter(lambda x: x["locale"] == "en")
        test_dataset = test_dataset.filter(lambda x: x["locale"] == "en")

        # Resample audio to SAMPLING_RATE
        train_dataset = train_dataset.cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )
        val_dataset = val_dataset.cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )
        test_dataset = test_dataset.cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )

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
            self.train_dataset.set_format(
                type="torch", columns=self.train_dataset.column_names
            )
            self.val_dataset.set_format(
                type="torch", columns=self.val_dataset.column_names
            )
        elif stage == "validate":
            self.val_dataset = Dataset.load_from_disk(target_dir / "val_dataset")
            self.val_dataset.set_format(
                type="torch", columns=self.val_dataset.column_names
            )
        elif stage == "test":
            self.test_dataset = Dataset.load_from_disk(target_dir / "test_dataset")
            self.test_dataset.set_format(
                type="torch", columns=self.test_dataset.column_names
            )

    def _preprocess(self, batch):
        # Extract audio arrays and normalize text
        audio_arrays = [i["array"] for i in batch["audio"]]
        texts = [self.processor.tokenizer.normalize(i) for i in batch["sentence"]]

        # Filter out samples with invalid accents
        valid_indices = []
        valid_accents = []
        for idx, accent in enumerate(batch["accent"]):
            if not isinstance(accent, str) or not accent.strip():
                continue

            # Split by comma and check each part
            accent_parts = [a.strip() for a in accent.split(",")]
            for part in accent_parts:
                # Try exact match first
                if part in self.accent_to_id_map:
                    valid_indices.append(idx)
                    valid_accents.append(part)
                    break
                # Try case-insensitive match
                elif part.lower() in {
                    k.lower(): k for k in self.accent_to_id_map.keys()
                }:
                    valid_indices.append(idx)
                    valid_accents.append(part)
                    break

        # If no valid accents found, return empty dict instead of None
        if not valid_indices:
            return {
                "input_features": [],
                "attention_mask": [],
                "labels": [],
                "decoder_attention_mask": [],
                "accent_id": [],
                "text": [],
            }

        # Keep only valid samples
        audio_arrays = [audio_arrays[i] for i in valid_indices]
        texts = [texts[i] for i in valid_indices]

        # Map valid accents to IDs
        accents_ids = torch.tensor(
            [self.accent_to_id_map[accent] for accent in valid_accents],
            dtype=torch.long,
        )

        # Process audio features with Whisper processor
        input_values = self.processor.feature_extractor(
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

        # Set padding tokens to -100 in labels
        labels = label_values.input_ids.clone()
        labels[~label_values.attention_mask.bool()] = -100

        # Convert tensors to lists for dataset storage
        return {
            "input_features": input_values.input_features.tolist(),
            "attention_mask": input_values.attention_mask.tolist(),
            "labels": labels.tolist(),
            "decoder_attention_mask": label_values.attention_mask.tolist(),
            "accent_id": accents_ids.tolist(),
            "text": texts,
        }

    def collate_fn(self, batch):
        return {
            "input_features": torch.stack([i["input_features"] for i in batch]),
            "attention_mask": torch.stack([i["attention_mask"] for i in batch]),
            "labels": torch.stack([i["labels"] for i in batch]),
            "decoder_attention_mask": torch.stack(
                [i["decoder_attention_mask"] for i in batch]
            ),
            "accent_id": [i["accent_id"] for i in batch],  # Include accent in the batch
        }

    def train_dataloader(self):
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
    from collections import Counter

    from hydra import compose, initialize

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="baseline_eval.yaml")

    data_module = CommonVoiceDataModule(subset_mode=True, **cfg.data)
    print("DEBUG: Created data module")

    data_module.prepare_data()
    print("DEBUG: Prepared data")

    # Setup all stages
    data_module.setup("fit")
    data_module.setup("test")  # Add this line to setup test dataset
    print("DEBUG: Setup complete")

    print("train length", len(data_module.train_dataset))
    print("val length", len(data_module.val_dataset))
    print("test length", len(data_module.test_dataset))  # Add this line

    # Print accent statistics
    train_accents = data_module.train_dataset["accent_id"]
    val_accents = data_module.val_dataset["accent_id"]
    test_accents = data_module.test_dataset["accent_id"]

    print("\nAccent Statistics:")
    print("\nTraining Set:")
    for accent, count in sorted(Counter(train_accents).items()):
        print(f"{accent}: {count}")

    print("\nValidation Set:")
    for accent, count in sorted(Counter(val_accents).items()):
        print(f"{accent}: {count}")

    print("\nTest Set:")
    for accent, count in sorted(Counter(test_accents).items()):
        print(f"{accent}: {count}")

    print("Train dataset features:", data_module.train_dataset.features)
    print("\nSample shapes and dtypes:")
    print(
        {
            i: torch.tensor(j).shape if isinstance(j, torch.Tensor) else j
            for i, j in data_module.train_dataset[0].items()
        }
    )

    for batch in data_module.train_dataloader():
        print(
            "train",
            {
                i: j.shape if isinstance(j, torch.Tensor) else len(j)
                for i, j in batch.items()
            },
        )
        print(
            "train",
            {
                i: j.dtype if isinstance(j, torch.Tensor) else type(j)
                for i, j in batch.items()
            },
        )
        break

    for batch in data_module.val_dataloader():
        print(
            "val",
            {
                i: j.shape if isinstance(j, torch.Tensor) else len(j)
                for i, j in batch.items()
            },
        )
        print(
            "val",
            {
                i: j.dtype if isinstance(j, torch.Tensor) else type(j)
                for i, j in batch.items()
            },
        )
        break

    for batch in data_module.test_dataloader():
        print(
            "test",
            {
                i: j.shape if isinstance(j, torch.Tensor) else len(j)
                for i, j in batch.items()
            },
        )
        print(
            "test",
            {
                i: j.dtype if isinstance(j, torch.Tensor) else type(j)
                for i, j in batch.items()
            },
        )
        break
