from dataclasses import dataclass

import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.model import WhisperAccentForConditionalGeneration, WhisperAccentProcessor
from src.model.tokenization import ACCENTS


@dataclass
class LoraArguments:
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    use_rslora: bool = True
    task_type: str = "SEQ_2_SEQ_LM"


@dataclass
class ModelArguments:
    model_name_or_path: str = "openai/whisper-tiny.en"
    is_multilingual: bool = False


def processor_init(model_args: ModelArguments) -> WhisperAccentProcessor:
    # Load processor and add accent tokens to tokenizer
    # Note: BOS token updated from <|endoftext|> to <|startoftranscript|>
    processor = WhisperAccentProcessor.from_pretrained(model_args.model_name_or_path)
    processor.tokenizer.add_special_tokens(
        {
            "additional_special_tokens": list(ACCENTS.values()),
            "bos_token": "<|startoftranscript|>",
        }
    )
    return processor


def model_init(
    model_args: ModelArguments,
    lora_args: LoraArguments,
    processor: WhisperAccentProcessor,
) -> WhisperAccentForConditionalGeneration:
    # Load whisper weights into whisper_accent model
    # and resize token embeddings + update generation config
    model = WhisperAccentForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.generation_config.accent_to_id = {
        k: v for k, v in processor.tokenizer.vocab.items() if k in ACCENTS.values()
    }
    # Note: proj_out is tied to decoder.embed_tokens; resize token embeddings weight
    # tying is not working
    model.proj_out = nn.Linear(
        model.proj_out.in_features,
        len(processor.tokenizer),
        bias=model.proj_out.bias is not None,
    )
    model.tie_weights()

    # Update generation config; https://github.com/openai/whisper/discussions/2094
    if model.generation_config.is_multilingual:
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Add LoRA layers
    if lora_args.lora_enable:
        # Target linear layers in decoder
        target_modules = []
        for name, _ in model.named_modules():
            m_list = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            if "model.decoder" in name and any(suffix in name for suffix in m_list):
                target_modules.append(name)

        # Trainable token indices for new accent tokens
        accent_token_indices = list(model.generation_config.accent_to_id.values())

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            use_rslora=lora_args.use_rslora,
            target_modules=target_modules,
            trainable_token_indices=accent_token_indices,
            task_type=lora_args.task_type,
            ensure_weight_tying=True,
        )
        model = get_peft_model(model, lora_config)

    return model
