"""
A modified version of the WhisperForConditionalGeneration model that includes an accent embedding.
This model is used to generate text from audio with an accent embedding.
However, this approach is not used in the final model.

We mention this approach in the paper and include it for completeness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperModel
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.whisper.modeling_whisper import (
    BaseModelOutput,
    shift_tokens_right,
)

from src.constants import NUM_ACCENTS


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.scale = float(d_model) ** -0.5  # Convert to float and use standard scaling

    def forward(self, query, key, value, mask=None):
        # Use PyTorch's built-in scaled dot product attention
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )
        return (
            output,
            None,
        )  # Return None for attention weights to maintain API compatibility


class AccentWhisperModel(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.accent_codebook = nn.Parameter(torch.randn(NUM_ACCENTS, config.d_model))

        self.cross_attention = ScaledDotProductAttention(
            d_model=config.d_model, dropout=0.1
        )

        nn.init.xavier_uniform_(self.accent_codebook)

    def forward(
        self,
        input_features=None,
        accent_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        decoder_position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            input_features = self._mask_input_features(
                input_features, attention_mask=attention_mask
            )

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if accent_ids is None:
            accent_ids = torch.zeros(
                input_features.size(0), dtype=torch.long, device=input_features.device
            )
        accent_embeddings = self.accent_codebook[accent_ids]  # [batch_size, d_model]

        # Expand accent embeddings to match sequence length
        accent_embeddings = accent_embeddings.unsqueeze(1).expand(
            -1, encoder_outputs.last_hidden_state.size(1), -1
        )

        # Apply scaled dot-product attention
        enhanced_states, attention_weights = self.cross_attention(
            query=encoder_outputs.last_hidden_state,
            key=accent_embeddings,
            value=accent_embeddings,
            mask=None,
        )
        encoder_outputs.last_hidden_state = (
            encoder_outputs.last_hidden_state + enhanced_states
        )

        if encoder_outputs is None:
            input_features = self._mask_input_features(
                input_features, attention_mask=attention_mask
            )

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class AccentWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = AccentWhisperModel(config)

    def forward(
        self,
        input_features=None,
        accent_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        decoder_position_ids=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if labels.shape[1] > self.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features=input_features,
            accent_ids=accent_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
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
            loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate(self, input_features, attention_mask=None, accent_ids=None, **kwargs):
        # Set performance-focused defaults
        if "use_cache" not in kwargs:
            kwargs["use_cache"] = True

        if "num_beams" not in kwargs:
            kwargs["num_beams"] = 1  # Use greedy decoding by default for speed

        if "max_length" not in kwargs:
            kwargs["max_length"] = 448  # Set reasonable default max length

        return super().generate(
            input_features=input_features,
            attention_mask=attention_mask,
            accent_ids=accent_ids,
            **kwargs,
        )


if __name__ == "__main__":
    # Sample dimensions for debugging
    batch_size = 4
    seq_length = 3000
    num_mel_bins = 80
    max_decoder_length = 448

    # Initialize model from pretrained Whisper
    model = AccentWhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-base.en",
    )

    # Create sample input tensors
    input_features = torch.randn(batch_size, num_mel_bins, seq_length)
    attention_mask = torch.randint(0, 2, (batch_size, seq_length))
    accent_ids = torch.randint(0, NUM_ACCENTS, (batch_size,))
    labels = torch.randint(0, model.config.vocab_size, (batch_size, max_decoder_length))
    decoder_attention_mask = torch.randint(0, 2, (batch_size, max_decoder_length))

    # Test forward pass
    outputs = model(
        input_features=input_features,
        attention_mask=attention_mask,
        accent_ids=accent_ids,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
    )

    # Print detailed output information
    print("\nModel Output Statistics:")
    print("-" * 50)
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits mean: {outputs.logits.mean().item():.4f}")
    print(f"Logits std: {outputs.logits.std().item():.4f}")
    print(f"Logits min: {outputs.logits.min().item():.4f}")
    print(f"Logits max: {outputs.logits.max().item():.4f}")
    print(f"Number of NaNs in logits: {torch.isnan(outputs.logits).sum().item()}")
    print(f"Number of Infs in logits: {torch.isinf(outputs.logits).sum().item()}")

    if outputs.loss is not None:
        print(f"\nLoss value: {outputs.loss.item():.4f}")
        print(f"Loss is NaN: {torch.isnan(outputs.loss).item()}")

    # Test generation
    generated_ids = model.generate(
        input_features=input_features,
        attention_mask=attention_mask,
        accent_ids=accent_ids,
    )

    print("\nGeneration Statistics:")
    print("-" * 50)
    print(f"Generated ids shape: {generated_ids.shape}")
    print(f"Unique tokens in generation: {torch.unique(generated_ids).shape[0]}")
    print(f"Number of NaNs in generated ids: {torch.isnan(generated_ids).sum().item()}")

    # Print model configuration
    print("\nModel Configuration:")
    print("-" * 50)
    print(f"Number of accents: {NUM_ACCENTS}")
    print(f"Model dimension: {model.config.d_model}")
    print(f"Vocabulary size: {model.config.vocab_size}")
    print(f"Number of attention heads: {model.config.encoder_attention_heads}")
