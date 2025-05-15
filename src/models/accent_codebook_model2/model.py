import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer


# --- Accent Components ---
class AccentCodebook(nn.Module):
    def __init__(self, num_accents=10, code_dim=384):
        super().__init__()
        # Initialize with Xavier/Glorot initialization
        self.codebook = nn.Parameter(torch.empty(num_accents, code_dim))
        nn.init.xavier_uniform_(self.codebook)
        self.norm = nn.LayerNorm(code_dim)

    def forward(self):
        # Add safeguards against NaN
        codebook = self.codebook
        if torch.isnan(codebook).any():
            print("NaN detected in raw codebook, reinitializing...")
            nn.init.xavier_uniform_(self.codebook)
            codebook = self.codebook

        # Normalize and clamp values
        codebook = self.norm(codebook)
        codebook = torch.clamp(codebook, min=-10.0, max=10.0)
        return codebook


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, accent_codebook):
        batch_size = hidden_states.size(0)
        codebook_expanded = accent_codebook.unsqueeze(0).expand(batch_size, -1, -1)

        # Add safeguards
        if torch.isnan(hidden_states).any():
            print("NaN detected in hidden_states before cross attention")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)

        if torch.isnan(codebook_expanded).any():
            print("NaN detected in expanded codebook")
            codebook_expanded = torch.nan_to_num(codebook_expanded, nan=0.0)

        # Normalize inputs
        hidden_states = self.norm(hidden_states)

        # Compute attention with safeguards
        try:
            attn_output, _ = self.cross_attn(
                hidden_states, codebook_expanded, codebook_expanded
            )
        except RuntimeError as e:
            print(f"Error in cross attention: {e}")
            # Fallback to identity if attention fails
            attn_output = hidden_states

        attn_output = self.dropout(attn_output)

        # Add residual with safeguards
        output = hidden_states + attn_output
        if torch.isnan(output).any():
            print("NaN detected after residual connection")
            output = torch.nan_to_num(output, nan=0.0)

        return output


class WhisperEncoderLayerWithAccent(WhisperEncoderLayer):
    def __init__(self, config, accent_codebook):
        super().__init__(config)
        self.cross_attn = CrossAttentionBlock(
            config.d_model, config.encoder_attention_heads
        )
        self.accent_codebook = accent_codebook
        self.cross_attn_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # Check for NaN in input
        if torch.isnan(hidden_states).any():
            print("NaN detected in hidden_states before self attention")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        # Check for NaN after self attention
        if torch.isnan(hidden_states).any():
            print("NaN detected in hidden_states after self attention")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        # Check for NaN after FFN
        if torch.isnan(hidden_states).any():
            print("NaN detected in hidden_states after FFN")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        # Cross attention over accent codebook
        codebook = self.accent_codebook()
        if torch.isnan(codebook).any():
            print("NaN detected in accent codebook")
            codebook = torch.nan_to_num(codebook, nan=0.0)

        # Add residual connection for cross attention
        residual = hidden_states
        hidden_states = self.cross_attn(hidden_states, codebook)
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        # Final NaN check
        if torch.isnan(hidden_states).any():
            print("NaN detected in hidden_states after cross attention")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class WhisperForConditionalGenerationWithAccent(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # Initialize accent codebook first
        self.accent_codebook = AccentCodebook(num_accents=12, code_dim=config.d_model)

        # Only replace the last few layers with accent-aware layers
        num_accent_layers = 1
        num_original_layers = config.encoder_layers - num_accent_layers
        new_layers = nn.ModuleList()

        # Keep original layers for the first part
        for i in range(num_original_layers):
            new_layers.append(self.model.encoder.layers[i])

        # Replace last few layers with accent-aware layers
        for i in range(num_original_layers, config.encoder_layers):
            new_layer = WhisperEncoderLayerWithAccent(config, self.accent_codebook)
            # Copy weights from original layer
            original_layer = self.model.encoder.layers[i]
            new_layer.load_state_dict(original_layer.state_dict(), strict=False)
            new_layers.append(new_layer)

        self.model.encoder.layers = new_layers

    def forward(
        self,
        input_features,
        decoder_input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        output_attentions=False,
        **kwargs,
    ):
        if encoder_outputs is None:
            # Check input features for NaN
            if torch.isnan(input_features).any():
                print("NaN detected in input features")
                input_features = torch.nan_to_num(input_features, nan=0.0)

            encoder_outputs = self.model.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            # Check encoder outputs for NaN
            if torch.isnan(encoder_outputs.last_hidden_state).any():
                print("NaN detected in encoder outputs")
                encoder_outputs.last_hidden_state = torch.nan_to_num(
                    encoder_outputs.last_hidden_state, nan=0.0
                )

        outputs = super().forward(
            input_features=None,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

        # Check logits for NaN
        if torch.isnan(outputs.logits).any():
            print("NaN detected in logits")
            outputs.logits = torch.nan_to_num(outputs.logits, nan=0.0)

        return outputs


if __name__ == "__main__":
    import numpy as np
    from transformers import WhisperConfig, WhisperProcessor

    torch.autograd.set_detect_anomaly(True)

    # Load processor and config
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    config = WhisperConfig.from_pretrained("openai/whisper-base")

    # Instantiate modified model
    model = WhisperForConditionalGenerationWithAccent.from_pretrained(
        "openai/whisper-base"
    )

    # Prepare batched inputs
    batch_size = 8
    # Generate random audio of different lengths (1-3 seconds)
    raw_speech = [np.random.randn(16000 * length) for length in [1, 2, 3, 2]]

    # Process inputs with padding
    inputs = processor.feature_extractor(
        raw_speech=raw_speech,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    decoder_input_ids = torch.randint(0, 50000, (batch_size, 100))
    decoder_attention_mask = torch.randint(0, 2, (batch_size, 100))
    print(inputs["input_features"].shape)
    print(inputs["attention_mask"].shape)
    print(decoder_input_ids.shape)

    # Forward
    model.train()
    output = model(
        input_features=inputs["input_features"],
        attention_mask=inputs["attention_mask"],
        decoder_attention_mask=decoder_attention_mask,
        labels=decoder_input_ids,
    )

    print(output.logits.shape)
    print(output.logits)
    print(output.loss)

    # Backward pass
    output.loss.backward()

    # Check gradients
    has_nan = False
    has_inf = False
    no_grad = False

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"No gradient for parameter: {name} (shape: {param.shape})")
            no_grad = True
            continue

        if torch.isnan(param.grad).any():
            print(f"NaN in gradients for parameter: {name} (shape: {param.shape})")
            has_nan = True

        if torch.isinf(param.grad).any():
            print(f"Inf in gradients for parameter: {name} (shape: {param.shape})")
            has_inf = True

        # Check gradient statistics
        grad_norm = param.grad.norm().item()
        if grad_norm > 10:
            print(f"Large gradient norm ({grad_norm:.2f}) for parameter: {name}")

    # Summary
    print("\nGradient Check Summary:")
    print(f"Parameters with no gradients: {no_grad}")
    print(f"Parameters with NaN gradients: {has_nan}")
    print(f"Parameters with Inf gradients: {has_inf}")

    # Check model parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in parameter: {name}")
        if torch.isinf(param).any():
            print(f"Inf in parameter: {name}")
