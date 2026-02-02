import numpy as np
import torch
import torch.nn as nn
from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperModel,
)
from transformers.modeling_outputs import BaseModelOutput

from .configuration import WhisperAccentConfig


class WhisperAccentModel(WhisperModel):
    config_class = WhisperAccentConfig

    def __init__(self, config: WhisperConfig):
        super().__init__(config)


class WhisperAccentForConditionalGeneration(WhisperForConditionalGeneration):
    config_class = WhisperAccentConfig

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperAccentModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions

        # Initialize weights and apply final processing
        self.post_init()

    def detect_accent(
        self,
        decoder_input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        encoder_outputs: torch.FloatTensor | BaseModelOutput | None = None,
        generation_config: GenerationConfig | None = None,
        num_segment_frames: int = 3000,
    ) -> torch.LongTensor:
        if input_features is None and encoder_outputs is None:
            raise ValueError("You have to specify either `input_features` or `encoder_outputs`")
        elif input_features is not None and encoder_outputs is not None:
            raise ValueError(
                "Make sure to specify only one of `input_features` or `encoder_outputs` - not both!"
            )
        elif input_features is not None:
            inputs = {"input_features": input_features[:, :, :num_segment_frames]}
        elif encoder_outputs is not None:
            inputs = {"encoder_outputs": encoder_outputs}

        generation_config = generation_config or self.generation_config

        with torch.no_grad():
            logits = self(**inputs, decoder_input_ids=decoder_input_ids, use_cache=False)
            logits = logits.logits[:, -1]

        # Mask out non-accent tokens
        non_accent_mask = torch.ones_like(logits[0], dtype=torch.bool)
        non_accent_mask[list(generation_config.accent_to_id.values())] = False

        # Set non-accent tokens to -inf
        logits[:, non_accent_mask] = -np.inf

        # Get accent ids
        accent_ids = logits.argmax(-1)

        return accent_ids

    def _retrieve_init_tokens(
        self,
        input_features,
        batch_size,
        generation_config,
        config,
        num_segment_frames,
        kwargs,
    ):
        init_tokens = super()._retrieve_init_tokens(
            input_features,
            batch_size,
            generation_config,
            config,
            num_segment_frames,
            kwargs,
        )

        # Check if init tokens have no_timestamps_token_id
        last_token = init_tokens[:, -1]
        has_no_timestamps_token = torch.all(
            last_token == self.generation_config.no_timestamps_token_id
        ).item()

        # Consistency check: return_timestamps flag vs. no_timestamps_token_id presence
        if not self.generation_config.return_timestamps and not has_no_timestamps_token:
            raise ValueError(
                "Generation config return_timestamps is set to False, but the init tokens do not"
                " end with no_timestamps_token_id."
            )
        if self.generation_config.return_timestamps and has_no_timestamps_token:
            raise ValueError(
                "Generation config return_timestamps is set to True, but the init tokens end with"
                " no_timestamps_token_id."
            )

        # If has no_timestamps_token_id, use init_tokens without last token
        if has_no_timestamps_token:
            decoder_input_ids = init_tokens[:, :-1]
        # If no no_timestamps_token_id, use init_tokens
        else:
            decoder_input_ids = init_tokens
        # Detect accent
        accent_ids = self.detect_accent(
            decoder_input_ids=decoder_input_ids,
            input_features=input_features,
            encoder_outputs=kwargs.get("encoder_outputs", None),
            generation_config=generation_config,
            num_segment_frames=num_segment_frames,
        )

        # Insert accent ids before no_timestamps_token_id; if return_timestamps is set to True,
        # insert at last index
        if has_no_timestamps_token:
            init_tokens = torch.cat(
                [init_tokens[:, :-1], accent_ids.unsqueeze(1), init_tokens[:, -1:]],
                dim=1,
            )
        else:
            init_tokens = torch.cat([init_tokens, accent_ids.unsqueeze(1)], dim=1)

        return init_tokens


__all__ = [
    "WhisperAccentModel",
    "WhisperAccentForConditionalGeneration",
]
