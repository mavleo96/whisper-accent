#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys

import torch
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, os.getcwd())

from src.constants import SAMPLING_RATE, WESTBROOK_DATASET_ACCENT_MAP
from src.utils import compute_wer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_fn(batch):
    return {
        "id": [b["id"] for b in batch],
        "input_features": torch.tensor([b["input_features"] for b in batch]),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
        "accent_name": [b["accent_name"] for b in batch],
        "target": [b["target"] for b in batch],
        "raw_target": [b["raw_target"] for b in batch],
    }


def make_preprocess_fn(processor):
    def preprocess(item):
        input_values = processor.feature_extractor(
            item["audio"]["array"],
            sampling_rate=SAMPLING_RATE,
            return_attention_mask=True,
        )
        raw_text = item["raw_text"]
        return {
            "id": item["audio_id"],
            "input_features": input_values.input_features[0],
            "attention_mask": input_values.attention_mask[0],
            "raw_target": raw_text,
            "target": processor.tokenizer.normalize(raw_text),
            "accent_name": WESTBROOK_DATASET_ACCENT_MAP[item["accent"]],
        }

    return preprocess


def load_model_and_processor(model_name, device, dtype):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device).to(dtype)
    model.eval()
    if model.generation_config.is_multilingual:
        model.generation_config.language = "<|en|>"
        model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return model, processor


def run_evaluation(model, processor, dataloader, device, dtype):
    all_ids, all_raw_preds, all_preds = [], [], []
    all_raw_targets, all_targets, all_accents = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        feats = batch["input_features"].to(device).to(dtype)
        mask = batch["attention_mask"].to(device).to(dtype)

        with torch.no_grad():
            pred_ids = model.generate(feats, attention_mask=mask)

        raw_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        preds = [processor.tokenizer.normalize(t) for t in raw_preds]

        all_ids.extend(batch["id"])
        all_raw_preds.extend(raw_preds)
        all_preds.extend(preds)
        all_raw_targets.extend(batch["raw_target"])
        all_targets.extend(batch["target"])
        all_accents.extend(batch["accent_name"])

    return {
        "ids": all_ids,
        "raw_preds": all_raw_preds,
        "preds": all_preds,
        "raw_targets": all_raw_targets,
        "targets": all_targets,
        "accents": all_accents,
    }


def save_results(
    output_path, model_name, dataset_name, split, overall_wer, wer_per_accent, eval_data
):
    predictions = {
        uid: {
            "raw_prediction": raw_pred,
            "prediction": pred,
            "raw_target": raw_tgt,
            "target": tgt,
            "accent": accent,
        }
        for uid, raw_pred, pred, raw_tgt, tgt, accent in zip(
            eval_data["ids"],
            eval_data["raw_preds"],
            eval_data["preds"],
            eval_data["raw_targets"],
            eval_data["targets"],
            eval_data["accents"],
            strict=True,
        )
    }
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "overall_wer": overall_wer,
        "wer_per_accent": wer_per_accent,
        "predictions": predictions,
    }
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on ASR task")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny.en")
    parser.add_argument(
        "--dataset_name", type=str, default="westbrook/English_Accent_DataSet"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "validation"]
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    logger.info("device=%s dtype=%s", device, dtype)
    torch.set_float32_matmul_precision("high")

    logger.info("Loading model: %s", args.model_name)
    model, processor = load_model_and_processor(args.model_name, device, dtype)

    logger.info("Loading dataset: %s split=%s", args.dataset_name, args.split)
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    dataset = dataset.map(
        make_preprocess_fn(processor),
        remove_columns=dataset.column_names,
        desc="Preprocessing",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    logger.info("Running evaluation")
    eval_data = run_evaluation(model, processor, dataloader, device, dtype)
    overall_wer, wer_per_accent = compute_wer(
        eval_data["preds"], eval_data["targets"], eval_data["accents"]
    )
    logger.info("overall_wer=%.4f", overall_wer)
    for name, wer, n in wer_per_accent:
        logger.info("  %s wer=%.4f n=%d", name, wer, n)

    if args.output:
        save_results(
            args.output,
            args.model_name,
            args.dataset_name,
            args.split,
            overall_wer,
            wer_per_accent,
            eval_data,
        )


if __name__ == "__main__":
    main()
