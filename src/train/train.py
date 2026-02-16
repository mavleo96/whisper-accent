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
    learning_rate: float = 5e-5
    embedding_learning_rate: float = 1e-4
    accent_classifier_learning_rate: float = 5e-5
    lambda_accent: float = 1.0
    report_to: None | str | list[str] = field(
        default="none",
        metadata={"help": "The list of integrations to report logs to.", "nargs": "+"},
    )

    def __post_init__(self):
        super().__post_init__()
        self.include_for_metrics = ["inputs"]
        self.batch_eval_metrics = True


def init_adaln_weights(module, state_dict, name):
    # Initialize modulation weights to 0
    nn.init.zeros_(module.modulation[-1].weight)

    # Copy old gamma and beta as bias in the modulation layer
    weight = state_dict[f"{name}.weight"]
    module.modulation[-1].bias[: module.hidden_dim].data.copy_(weight)
    if f"{name}.bias" in state_dict:
        bias = state_dict[f"{name}.bias"]
        module.modulation[-1].bias[module.hidden_dim :].data.copy_(bias)


def model_init(base_model_name_or_path):
    # Load pretrained whisper model
    pretrained_model = WhisperForConditionalGeneration.from_pretrained(base_model_name_or_path)
    state_dict = pretrained_model.state_dict()

    # Load whisper weights into whisper_accent model
    config = WhisperAccentConfig.from_pretrained(base_model_name_or_path)
    model = WhisperAccentForConditionalGeneration(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    for layer_idx, layer in enumerate(model.get_decoder().layers):
        init_adaln_weights(
            layer.self_attn_layer_norm,
            state_dict,
            f"model.decoder.layers.{layer_idx}.self_attn_layer_norm",
        )
        init_adaln_weights(
            layer.encoder_attn_layer_norm,
            state_dict,
            f"model.decoder.layers.{layer_idx}.encoder_attn_layer_norm",
        )
        init_adaln_weights(
            layer.final_layer_norm, state_dict, f"model.decoder.layers.{layer_idx}.final_layer_norm"
        )

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
