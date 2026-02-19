from dataclasses import dataclass

import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput


@dataclass
class WhisperAccentEncoderOutput(BaseModelOutput):
    accent_loss: torch.Tensor | None = None
    accent_logits: torch.Tensor | None = None


@dataclass
class WhisperAccentSeq2SeqModelOutput(Seq2SeqModelOutput):
    accent_loss: torch.Tensor | None = None
    accent_logits: torch.Tensor | None = None


@dataclass
class WhisperAccentSeq2SeqLMOutput(Seq2SeqLMOutput):
    accent_loss: torch.Tensor | None = None
    accent_logits: torch.Tensor | None = None
