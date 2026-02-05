from dataclasses import dataclass, field

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainingArguments

from src.model import WhisperAccentForConditionalGeneration, WhisperAccentProcessor
from src.model.tokenization import ACCENTS


@dataclass
class ModelArguments:
    model_type: str = field(choices=["whisper_accent", "whisper"])
    base_model_name_or_path: str
    is_multilingual: bool


@dataclass
class DatasetArguments:
    train_data_path: str
    eval_data_path: str
    num_proc: int = 16


@dataclass
class WhisperAccentTrainingArguments(Seq2SeqTrainingArguments):
    lambda_accent_loss: float = 0.0
    lambda_diversity_loss: float = 0.0
    embedding_learning_rate: float = 1e-4
    report_to: None | str | list[str] = field(
        default=None,
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


def processor_init(model_args):
    # Load processor and add accent tokens to tokenizer
    # Note: BOS token updated from <|endoftext|> to <|startoftranscript|>
    processor = WhisperAccentProcessor.from_pretrained(model_args.base_model_name_or_path)
    processor.tokenizer.add_special_tokens(
        {
            "additional_special_tokens": list(ACCENTS.values()),
            "bos_token": "<|startoftranscript|>",
        }
    )
    return processor


def model_init(model_args, lora_args, processor):
    # Load whisper weights into whisper_accent model
    # and resize token embeddings + update generation config
    model = WhisperAccentForConditionalGeneration.from_pretrained(
        model_args.base_model_name_or_path
    )
    model.config.architectures = [model.__class__.__name__]
    model.resize_token_embeddings(len(processor.tokenizer))
    model.generation_config.accent_to_id = {
        k: v for k, v in processor.tokenizer.vocab.items() if k in ACCENTS.values()
    }
    # Note: proj_out is tied to decoder.embed_tokens; resize token embeddings weight tying is
    # not working
    model.proj_out = nn.Linear(
        model.proj_out.in_features,
        len(processor.tokenizer),
        bias=model.proj_out.bias is not None,
    )
    model.tie_weights()

    # Update bos token id; https://github.com/huggingface/transformers/issues/24342
    model.config.bos_token_id = processor.tokenizer.bos_token_id
    model.generation_config.bos_token_id = processor.tokenizer.bos_token_id

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
        accent_token_indices = sorted(list(model.generation_config.accent_to_id.values()))

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
    else:
        raise NotImplementedError("Non-LoRA training is not implemented yet")

    return model
