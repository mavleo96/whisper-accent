from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import Seq2SeqTrainingArguments

from src.model import WhisperAccentForConditionalGeneration, WhisperAccentProcessor
from src.model.tokenization import ACCENTS


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
    lambda_accent_loss: float = 0.0
    lambda_diversity_loss: float = 0.0
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


def processor_init(base_model_name_or_path):
    # Load processor and add accent tokens to tokenizer
    # Note: BOS token updated from <|endoftext|> to <|startoftranscript|>
    processor = WhisperAccentProcessor.from_pretrained(base_model_name_or_path)
    processor.tokenizer.add_special_tokens(
        {
            "additional_special_tokens": list(ACCENTS.values()),
            "bos_token": "<|startoftranscript|>",
        }
    )
    return processor


def orthogonal_init(model, n_tokens):
    # Note: this is making the forward pass very slow in the beginning
    embeddings = model.get_input_embeddings()
    device = embeddings.weight.device
    dtype = embeddings.weight.dtype
    d_model = model.config.d_model
    with torch.no_grad():
        A = torch.randn(d_model, n_tokens, device="cpu", dtype=dtype)
        Q, _ = torch.linalg.qr(A)  # Q: (d_model, n_tokens)
        new = Q.T.to(device=device).contiguous()  # (n_tokens, d_model)
        embeddings.weight.data[-n_tokens:].copy_(new)
    if hasattr(model, "tie_weights") and callable(model.tie_weights):
        model.tie_weights()


def model_init(base_model_name_or_path, processor):
    # Load whisper weights into whisper_accent model
    # and resize token embeddings + update generation config
    model = WhisperAccentForConditionalGeneration.from_pretrained(base_model_name_or_path)

    # Update model config
    model.config.architectures = [model.__class__.__name__]
    model.config.model_type = "whisper_accent"

    model.resize_token_embeddings(len(processor.tokenizer))
    # orthogonal_init(model, len(ACCENTS))
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
    model.generation_config.return_timestamps = False

    return model
