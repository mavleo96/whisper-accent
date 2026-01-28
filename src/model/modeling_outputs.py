from dataclasses import dataclass

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class Seq2SeqLMOutputForWhisperAccent(Seq2SeqLMOutput):
    ce_loss: torch.FloatTensor | None = None
    accent_diversity_loss: torch.FloatTensor | None = None
