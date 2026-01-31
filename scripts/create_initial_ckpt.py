#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.getcwd())

import argparse

from src.model import WhisperAccentForConditionalGeneration, WhisperAccentProcessor
from src.model.tokenization import ACCENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny.en")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/whisper-accent-tiny.en",
    )
    args = parser.parse_args()

    # Load processor and model
    processor = WhisperAccentProcessor.from_pretrained(args.model_name)
    model = WhisperAccentForConditionalGeneration.from_pretrained(args.model_name)

    # Update tokenizer and model
    processor.tokenizer.add_special_tokens(
        {
            "additional_special_tokens": list(ACCENTS.values()),
            "bos_token": "<|startoftranscript|>",
        }
    )
    processor.tokenizer.model_max_length = model.generation_config.max_length
    model.resize_token_embeddings(len(processor.tokenizer))
    model.generation_config.accent_to_id = {
        k: v for k, v in processor.tokenizer.vocab.items() if k in ACCENTS.values()
    }

    # Save processor and model
    processor.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
