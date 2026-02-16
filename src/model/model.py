import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperModel,
)
from transformers.cache_utils import Cache
from transformers.models.whisper.modeling_whisper import (
    shift_tokens_right,
)

from .configuration import WhisperAccentConfig
from .decoder import WhisperAccentDecoder
from .encoder import WhisperAccentEncoder
from .modelling_outputs import (
    WhisperAccentEncoderOutput,
    WhisperAccentSeq2SeqLMOutput,
    WhisperAccentSeq2SeqModelOutput,
)


class WhisperAccentModel(WhisperModel):
    config_class = WhisperAccentConfig

    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperAccentEncoder(config)
        self.decoder = WhisperAccentDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_accent_embeddings(self):
        return self.decoder.embed_accents

    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = None,
        past_key_values: Cache | None = None,
        decoder_inputs_embeds: tuple[torch.FloatTensor] | None = None,
        decoder_position_ids: tuple[torch.LongTensor] | None = None,
        accent_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | WhisperAccentSeq2SeqModelOutput:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(
                input_features, attention_mask=attention_mask
            )

            encoder_outputs = self.encoder(
                input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                accent_labels=accent_labels,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a WhisperAccentEncoderOutput
        # when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, WhisperAccentEncoderOutput):
            encoder_outputs = WhisperAccentEncoderOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                accent_loss=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                accent_logits=encoder_outputs[4] if len(encoder_outputs) > 4 else None,
            )

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            accent_logits=encoder_outputs[-1],
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return WhisperAccentSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            accent_loss=encoder_outputs.accent_loss,
            accent_logits=encoder_outputs.accent_logits,
        )


class WhisperAccentForConditionalGeneration(WhisperForConditionalGeneration):
    config_class = WhisperAccentConfig

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperAccentModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_accent_embeddings(self):
        return self.model.get_accent_embeddings()

    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: tuple[tuple[torch.FloatTensor]] | None = None,
        past_key_values: Cache | None = None,
        decoder_inputs_embeds: tuple[torch.FloatTensor] | None = None,
        decoder_position_ids: tuple[torch.LongTensor] | None = None,
        labels: torch.LongTensor | None = None,
        accent_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | WhisperAccentSeq2SeqLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed"
                    f" length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            accent_labels=accent_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return WhisperAccentSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            accent_loss=outputs.accent_loss,
            accent_logits=outputs.accent_logits,
        )


__all__ = [
    "WhisperAccentModel",
    "WhisperAccentForConditionalGeneration",
]
