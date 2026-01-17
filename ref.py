import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoderLayer, WhisperForConditionalGeneration)


class EncoderCodebookAdapter(nn.Module):
    def __init__(self, hidden_dim, codebook_dim, num_codes, num_heads=4):
        super().__init__()
        self.codebook = nn.Parameter(
            torch.randn(num_codes, codebook_dim)
        )  # [N, D_code]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            kdim=codebook_dim,
            vdim=codebook_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_out):
        B = encoder_out.size(0)
        codebook_exp = self.codebook.unsqueeze(0).repeat(B, 1, 1)  # [B, N, D_code]
        adapted, attn_weights = self.cross_attn(encoder_out, codebook_exp, codebook_exp)
        return self.layer_norm(encoder_out + adapted), attn_weights


class WhisperEncoderLayerWithCodebook(WhisperEncoderLayer):
    def __init__(self, config, adapter=None, insert_after_self_attn=True):
        super().__init__(config)
        self.adapter = adapter
        self.insert_after_self_attn = insert_after_self_attn

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, attention_mask=attention_mask, **kwargs
        )
        hidden_states = residual + hidden_states

        if self.insert_after_self_attn and self.adapter:
            hidden_states, self.attn_weights = self.adapter(hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if not self.insert_after_self_attn and self.adapter:
            hidden_states, self.attn_weights = self.adapter(hidden_states)

        return hidden_states


class CodebookDiversityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights):
        code_usage = attn_weights.mean(dim=1)
        prob_mean = code_usage.mean(dim=0)
        entropy = -torch.sum(prob_mean * torch.log(prob_mean + 1e-9))
        return -entropy


class CodebookAccentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, accent_labels):
        B, T, N = attn_weights.shape
        usage = attn_weights.mean(dim=1)  # [B, N]
        accent_set = torch.unique(accent_labels)
        accent_usages = []

        for acc in accent_set:
            mask = accent_labels == acc
            acc_usage = usage[mask]
            accent_usages.append(acc_usage.mean(0))

        penalty = 0
        for i in range(len(accent_usages)):
            for j in range(i + 1, len(accent_usages)):
                penalty += F.cosine_similarity(
                    accent_usages[i], accent_usages[j], dim=0
                )

        return penalty / (len(accent_usages) * (len(accent_usages) - 1) / 2)


class WhisperWithCodebook(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        self.adapter = EncoderCodebookAdapter(
            hidden_dim=config.d_model,
            codebook_dim=config.d_model,
            num_codes=16,
            num_heads=4,
        )

        self.insert_layers = [3, 6, 9]  # Example positions
        for i in self.insert_layers:
            self.model.encoder.layers[i] = WhisperEncoderLayerWithCodebook(
                config=config, adapter=self.adapter, insert_after_self_attn=True
            )

    def forward(
        self,
        input_features=None,
        decoder_input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        accent_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_features,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        encoder_hidden_states = encoder_outputs[0]

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            # Additional loss terms from codebook
            if accent_labels is not None:
                diversity_loss = CodebookDiversityLoss()(
                    self.adapter.cross_attn.attn_weights
                )
                accent_loss = CodebookAccentLoss()(
                    self.adapter.cross_attn.attn_weights, accent_labels
                )
                loss += 0.01 * diversity_loss + 0.01 * accent_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        from transformers.modeling_outputs import Seq2SeqLMOutput

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
