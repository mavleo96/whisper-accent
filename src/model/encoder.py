import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from .configuration import WhisperAccentConfig
from .modelling_outputs import WhisperAccentEncoderOutput


class AccentClassifier(nn.Module):
    def __init__(self, config: WhisperAccentConfig) -> None:
        super().__init__()
        self.config = config

        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        self.projector = nn.Linear(config.d_model, config.accent_proj_size)
        self.classifier = nn.Linear(config.accent_proj_size, config.num_accents)

    def forward(
        self, hidden_states: tuple[torch.Tensor, ...], labels: torch.LongTensor | None = None
    ) -> torch.Tensor:
        assert len(hidden_states) == self.config.num_hidden_layers + 1, (
            "Number of hidden states must match number of layers"
        )
        # Average the hidden states along the layer dimension
        # using learnable weights
        hidden_states = torch.stack(hidden_states, dim=1)
        norm_weights = F.softmax(self.layer_weights, dim=-1)
        hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)

        # Project hidden states and mean pool them
        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)

        # Classify the accent
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.num_accents), labels.view(-1))

        return loss, logits


class WhisperAccentEncoder(WhisperEncoder):
    def __init__(self, config: WhisperAccentConfig):
        super().__init__(config)

        self.accent_classifier = AccentClassifier(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        accent_labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | WhisperAccentEncoderOutput:
        output = super().forward(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        accent_loss, accent_logits = self.accent_classifier(output.hidden_states, accent_labels)

        encoder_states = output.hidden_states if output_hidden_states else None
        all_attentions = output.attentions if output_attentions else None
        if not return_dict:
            # TODO: think if these positions are safe to return
            return tuple(
                v
                for v in [
                    output.last_hidden_state,
                    encoder_states,
                    all_attentions,
                    accent_loss,
                    accent_logits,
                ]
                if v is not None
            )

        return WhisperAccentEncoderOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=encoder_states,
            attentions=all_attentions,
            accent_loss=accent_loss,
            accent_logits=accent_logits,
        )
