from dataclasses import dataclass, field

import torch.nn as nn
from transformers import Seq2SeqTrainingArguments, WhisperForConditionalGeneration

from src.model import (
    WhisperAccentConfig,
    WhisperAccentForConditionalGeneration,
)


@dataclass
class ModelArguments:
    model_type: str = field(metadata={"choices": ["whisper_accent", "whisper"]})
    base_model_name_or_path: str
    is_multilingual: bool


@dataclass
class DatasetArguments:
    train_data_path: str
    eval_data_path: str
    num_proc: int = 16


@dataclass
class WhisperAccentTrainingArguments(Seq2SeqTrainingArguments):
    embedding_learning_rate: float = 1e-4
    report_to: None | str | list[str] = field(
        default="none",
        metadata={"help": "The list of integrations to report logs to.", "nargs": "+"},
    )

    def __post_init__(self):
        super().__post_init__()
        self.include_for_metrics = ["inputs"]
        self.batch_eval_metrics = True


@dataclass
class LoraArguments:
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    use_rslora: bool = True
    task_type: str = "SEQ_2_SEQ_LM"


def model_init(base_model_name_or_path):
    # Load pretrained whisper model
    pretrained_model = WhisperForConditionalGeneration.from_pretrained(base_model_name_or_path)
    state_dict = pretrained_model.state_dict()

    # Load whisper weights into whisper_accent model
    config = WhisperAccentConfig.from_pretrained(base_model_name_or_path)
    model = WhisperAccentForConditionalGeneration(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    for layer_idx, layer in enumerate(model.get_decoder().layers):
        # Update self attn layer norm weights and biases
        weight = state_dict[f"model.decoder.layers.{layer_idx}.self_attn_layer_norm.weight"]
        bias = state_dict[f"model.decoder.layers.{layer_idx}.self_attn_layer_norm.bias"]
        layer.self_attn_layer_norm.norm.weight.data.copy_(weight)
        layer.self_attn_layer_norm.norm.bias.data.copy_(bias)
        nn.init.zeros_(layer.self_attn_layer_norm.modulation[-1].weight)

        # Update encoder attn layer norm weights and biases
        weight = state_dict[f"model.decoder.layers.{layer_idx}.encoder_attn_layer_norm.weight"]
        bias = state_dict[f"model.decoder.layers.{layer_idx}.encoder_attn_layer_norm.bias"]
        layer.encoder_attn_layer_norm.norm.weight.data.copy_(weight)
        layer.encoder_attn_layer_norm.norm.bias.data.copy_(bias)
        nn.init.zeros_(layer.encoder_attn_layer_norm.modulation[-1].weight)

        # Update final layer norm weights and biases
        weight = state_dict[f"model.decoder.layers.{layer_idx}.final_layer_norm.weight"]
        bias = state_dict[f"model.decoder.layers.{layer_idx}.final_layer_norm.bias"]
        layer.final_layer_norm.norm.weight.data.copy_(weight)
        layer.final_layer_norm.norm.bias.data.copy_(bias)
        nn.init.zeros_(layer.final_layer_norm.modulation[-1].weight)

    nn.init.normal_(model.get_accent_embeddings().weight, mean=0.0, std=0.02)

    # Update model config and generation config
    model.config.architectures = [model.__class__.__name__]
    model.config.model_type = "whisper_accent"
    model.generation_config = pretrained_model.generation_config

    # # Update bos token id; https://github.com/huggingface/transformers/issues/24342
    # model.config.bos_token_id = processor.tokenizer.bos_token_id
    # model.generation_config.bos_token_id = processor.tokenizer.bos_token_id

    # Update generation config; https://github.com/openai/whisper/discussions/2094
    if model.generation_config.is_multilingual:
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.return_timestamps = False

    return model
