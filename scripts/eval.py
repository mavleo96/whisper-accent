#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
import sys
from functools import partial

import evaluate
import numpy as np
import torch
from datasets import Audio, Value, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer

sys.path.insert(0, os.getcwd())

from src.constants import SAMPLING_RATE, WESTBROOK_DATASET_ACCENT_MAP
from src.model import register_whisper_accent
from src.model.configuration import ACCENTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

register_whisper_accent()

# id -> label (ACCENTS is label -> id); used for per-accent dict keys and logging
ACCENT_MAP = {v: k for k, v in ACCENTS.items()}


def collate_fn(batch):
    return {
        "id": [b["id"] for b in batch],
        "input_features": torch.tensor([b["input_features"] for b in batch]),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
        "accent": [b["accent"] for b in batch],
        "target": [b["target"] for b in batch],
        "raw_target": [b["raw_target"] for b in batch],
    }


def preprocess(item, processor):
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
        "accent": ACCENTS[item["accent"]],  # dataset accent is label string
    }


def compute_wer(preds, targets, accents):
    metric = evaluate.load("wer")
    overall_wer = metric.compute(predictions=preds, references=targets)
    preds = np.asarray(preds, dtype=object)
    targets = np.asarray(targets, dtype=object)
    accents_arr = np.asarray(accents)
    wer_per_accent = {}
    for accent_id in np.unique(accents_arr):
        mask = accents_arr == accent_id
        wer = metric.compute(predictions=preds[mask].tolist(), references=targets[mask].tolist())
        wer_per_accent[ACCENT_MAP[accent_id]] = {"wer": wer, "n": int(np.sum(mask))}
    return overall_wer, wer_per_accent


def compute_accuracy(accent_preds, accents):
    metric = evaluate.load("accuracy")
    overall_accuracy = metric.compute(predictions=accent_preds, references=accents)["accuracy"]
    accents_arr = np.asarray(accents)
    accent_preds_arr = np.asarray(accent_preds)
    accuracy_per_accent = {}
    for accent_id in np.unique(accents_arr):
        mask = accents_arr == accent_id
        accuracy = metric.compute(
            predictions=accent_preds_arr[mask].tolist(),
            references=accents_arr[mask].tolist(),
        )["accuracy"]
        accuracy_per_accent[ACCENT_MAP[accent_id]] = {
            "accuracy": accuracy,
            "n": int(np.sum(mask)),
        }
    return overall_accuracy, accuracy_per_accent


def run_evaluation(model, processor, dataloader, device, dtype):
    all_ids, all_raw_preds, all_preds = [], [], []
    all_raw_targets, all_targets, all_accents = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating Generation"):
        feats = batch["input_features"].to(device).to(dtype)
        mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            pred_ids = model.generate(feats, attention_mask=mask)

        raw_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        preds = [processor.tokenizer.normalize(t) for t in raw_preds]

        all_ids.extend(batch["id"])
        all_raw_preds.extend(raw_preds)
        all_preds.extend(preds)
        all_raw_targets.extend(batch["raw_target"])
        all_targets.extend(batch["target"])
        all_accents.extend(batch["accent"])

    eval_data = {
        "ids": all_ids,
        "raw_preds": all_raw_preds,
        "preds": all_preds,
        "raw_targets": all_raw_targets,
        "targets": all_targets,
        "accents": all_accents,
    }
    if model.config.model_type == "whisper":
        return eval_data

    # whisper_accent only: second pass for accent head
    all_accent_preds = []
    for batch in tqdm(dataloader, desc="Predicting Accent"):
        feats = batch["input_features"].to(device).to(dtype)
        mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            accent_preds = model.predict_accent(feats, mask)
            all_accent_preds.extend(accent_preds.tolist())

    eval_data["accent_preds"] = all_accent_preds

    return eval_data


def save_results(
    output_path,
    model_name,
    dataset_name,
    split,
    overall_wer,
    wer_per_accent,
    overall_accuracy,
    accuracy_per_accent,
    eval_data,
):
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "overall_wer": overall_wer,
        "wer_per_accent": wer_per_accent,
    }

    if overall_accuracy is not None:
        results["overall_accuracy"] = overall_accuracy
        results["accuracy_per_accent"] = accuracy_per_accent

    predictions = {}
    for i in range(len(eval_data["ids"])):
        pred_info = {
            "raw_prediction": eval_data["raw_preds"][i],
            "prediction": eval_data["preds"][i],
            "raw_target": eval_data["raw_targets"][i],
            "target": eval_data["targets"][i],
            "accent": eval_data["accents"][i],
        }
        if "accent_preds" in eval_data:
            pred_info["accent_pred"] = eval_data["accent_preds"][i]
        predictions[eval_data["ids"][i]] = pred_info
    results["predictions"] = predictions

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on ASR task")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny.en")
    parser.add_argument("--dataset_name", type=str, default="westbrook/English_Accent_DataSet")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    logger.info("device=%s dtype=%s", device, dtype)
    torch.set_float32_matmul_precision("high")

    logger.info("Loading model: %s", args.model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)
    # Note: hack since tokenizer registration is not working
    if model.config.model_type == "whisper_accent":
        processor.tokenizer = WhisperTokenizer.from_pretrained(args.model_name)
    model.to(device).to(dtype)
    model.eval()

    # Update generation config; https://github.com/openai/whisper/discussions/2094
    if model.generation_config.is_multilingual:
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None

    logger.info("Loading dataset: %s split=%s", args.dataset_name, args.split)
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    dataset = dataset.cast_column("accent", Value("string"))
    dataset = dataset.map(lambda x: {"accent": WESTBROOK_DATASET_ACCENT_MAP[int(x["accent"])]})
    dataset = dataset.map(partial(preprocess, processor=processor), desc="Preprocessing")

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
    for name, wer_info in wer_per_accent.items():
        msg = f"  {name} wer={wer_info['wer']:.4f} n={wer_info['n']}"
        if "acc" in wer_info:
            msg += f" acc={wer_info['acc']:.4f}"
        logger.info(msg)
    if model.config.model_type == "whisper_accent":
        overall_accuracy, accuracy_per_accent = compute_accuracy(
            eval_data["accent_preds"],
            eval_data["accents"],
        )
        logger.info("overall_accuracy=%.4f", overall_accuracy)
        for name, accuracy_info in accuracy_per_accent.items():
            msg = f"  {name} accuracy={accuracy_info['accuracy']:.4f} n={accuracy_info['n']}"
            logger.info(msg)
    else:
        overall_accuracy = None
        accuracy_per_accent = None

    if args.output:
        save_results(
            output_path=args.output,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            split=args.split,
            overall_wer=overall_wer,
            wer_per_accent=wer_per_accent,
            overall_accuracy=overall_accuracy,
            accuracy_per_accent=accuracy_per_accent,
            eval_data=eval_data,
        )


if __name__ == "__main__":
    main()
